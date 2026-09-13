"""Caller-owned, reusable image buffers shared between local processes."""

from __future__ import annotations

import contextlib
import math
import mmap
import operator
import os
from multiprocessing import shared_memory

import numpy as np


def _format(shape, dtype):
    shape = tuple(operator.index(value) for value in shape)
    dtype = np.dtype(dtype)
    if len(shape) not in (2, 3) or any(value <= 0 for value in shape):
        raise ValueError("Image shape must have two or three positive dimensions")
    if dtype not in (np.dtype("uint8"), np.dtype("float32"), np.dtype("uint32"), np.dtype("int32")):
        raise ValueError("Shared images require native uint8, float32, uint32, or int32 data")
    return shape, dtype, math.prod(shape) * dtype.itemsize


def _map(name, size):
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError("Invalid shared image name")
    if os.name == "nt":
        with contextlib.closing(shared_memory.SharedMemory(name=name)) as segment:
            if segment.size < size:
                raise ValueError("Shared image allocation is smaller than its shape")
            return mmap.mmap(-1, size, tagname=name)
    # Python 3.11/3.12 cannot disable SharedMemory's resource tracker on attach.
    # Map the POSIX descriptor directly so an independent reader never unlinks
    # the caller's allocation. mmap stays alive through NumPy's buffer references.
    fd = shared_memory._posixshmem.shm_open("/" + name, os.O_RDWR, mode=0o600)
    try:
        if os.fstat(fd).st_size < size:
            raise ValueError("Shared image allocation is smaller than its shape")
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)


class SharedImage:
    """A fixed-shape shared image with explicit allocation and lifetime ownership.

    Create once and reuse with ``RpcClient.capture_into`` or
    ``PassiveViewer.capture_into``. ``array`` is a writable zero-copy NumPy view;
    the next capture into this buffer overwrites it. Copy to retain a frame.
    Closing the owner unlinks the POSIX name. Existing local array views remain
    valid until released. On Windows the last view releases the allocation.
    Only one capture may write a given buffer at a time.
    """

    def __init__(self, shape, dtype=np.uint8):
        shape, dtype, size = _format(shape, dtype)
        self._owner = shared_memory.SharedMemory(create=True, size=size)
        self._array = None
        try:
            self._name = self._owner.name
            mapping = _map(self._name, size)
            self._array = np.frombuffer(mapping, dtype=dtype).reshape(shape)
            self._owner.close()
        except BaseException:
            self.close()
            raise

    @classmethod
    def attach(cls, descriptor: dict) -> SharedImage:
        """Map a descriptor without taking ownership of the shared allocation."""
        shape, dtype, size = _format(descriptor["shape"], descriptor["dtype"])
        mapping = _map(descriptor["name"], size)
        result = cls.__new__(cls)
        result._owner = None
        result._name = descriptor["name"]
        result._array = np.frombuffer(mapping, dtype=dtype).reshape(shape)
        return result

    @property
    def array(self) -> np.ndarray:
        """Return a zero-copy view; raises after this handle has closed."""
        if self._array is None:
            raise RuntimeError("The shared image is closed")
        return self._array

    @property
    def descriptor(self) -> dict:
        """Return the small JSON-compatible descriptor sent to another process."""
        image = self.array
        return {"name": self._name, "shape": list(image.shape), "dtype": image.dtype.str}

    def close(self) -> None:
        """Unlink an owned allocation and release this handle's array reference."""
        owner, self._owner = getattr(self, "_owner", None), None
        if owner is not None:
            with contextlib.suppress(FileNotFoundError):
                owner.unlink()
            owner.close()
        self._array = None

    def __reduce__(self):
        # Spawn/pickle transfers only the descriptor, never pixels or ownership.
        return type(self).attach, (self.descriptor,)

    def __enter__(self) -> SharedImage:
        _ = self.array
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
