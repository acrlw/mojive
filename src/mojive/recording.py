"""Stream rendered video and snapshot packets incrementally."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SNAPSHOT_PREFIX = b"MOJIVE-SNAPSHOT\x00"
SNAPSHOT_FORMAT = "mojive.snapshot-recording"
SNAPSHOT_FORMAT_VERSION = 2
SNAPSHOT_MAGIC = SNAPSHOT_PREFIX + bytes((SNAPSHOT_FORMAT_VERSION,))
LEGACY_SNAPSHOT_PREFIXES = frozenset({b"FORGE-SNAPSHOT\x00"})
LEGACY_SNAPSHOT_FORMATS = frozenset({"forge.snapshot-recording"})


@dataclass(frozen=True)
class SnapshotHeader:
    """Version metadata stored at the start of a snapshot recording."""

    format: str = SNAPSHOT_FORMAT
    version: int = SNAPSHOT_FORMAT_VERSION


class _SnapshotUnpickler(pickle.Unpickler):
    """Load recordings made before the package was renamed to Mojive."""

    def find_class(self, module: str, name: str):
        if module == "forge_viewer" or module.startswith("forge_viewer."):
            module = f"mojive{module[len('forge_viewer') :]}"
        return super().find_class(module, name)


def _load_packet(stream):
    return _SnapshotUnpickler(stream).load()


class VideoRecorder:
    """Stream RGB frames to an encoded video without retaining them in memory."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float = 30.0) -> None:
        from imageio_ffmpeg import write_frames

        self.path = Path(path)
        self.size = (int(size[0]), int(size[1]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = write_frames(
            str(self.path),
            self.size,
            fps=float(fps),
            pix_fmt_out="yuv444p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        self._writer.send(None)
        self.frames = 0

    def append(self, frame: np.ndarray) -> None:
        """Encode one uint8 RGB image matching the configured frame size."""
        image = np.asarray(frame)
        expected = (self.size[1], self.size[0])
        if image.shape[:2] != expected or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                f"video frames must be {expected[1]}×{expected[0]} RGB, got {image.shape}"
            )

        rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        self._writer.send(rgb)
        self.frames += 1

    def close(self) -> None:
        """Finalize the video stream."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SnapshotWriter:
    """Append-only stream of remote structure and frame packets."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        self._file.write(SNAPSHOT_MAGIC)
        pickle.dump(SnapshotHeader(), self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.packets = 0

    def write(self, packet: object) -> None:
        """Append one remote structure or frame packet."""
        pickle.dump(packet, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.packets += 1

    def close(self) -> None:
        """Flush and close the recording file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> SnapshotWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_snapshots(path: Path):
    """Yield a snapshot stream and reject unrelated files before unpickling packets."""
    with Path(path).open("rb") as stream:
        prefix = bytearray()
        while len(prefix) < 64:
            byte = stream.read(1)
            if not byte:
                break
            prefix += byte
            if byte == b"\x00":
                break
        if bytes(prefix) != SNAPSHOT_PREFIX and bytes(prefix) not in LEGACY_SNAPSHOT_PREFIXES:
            raise ValueError("not a Mojive snapshot recording")
        encoded_version = stream.read(1)
        if len(encoded_version) != 1:
            raise ValueError("truncated Mojive snapshot header")
        version = encoded_version[0]
        if version == SNAPSHOT_FORMAT_VERSION:
            try:
                header = _load_packet(stream)
            except (EOFError, pickle.UnpicklingError) as exc:
                raise ValueError("invalid Mojive snapshot header") from exc
            supported_formats = {SNAPSHOT_FORMAT, *LEGACY_SNAPSHOT_FORMATS}
            if (
                not isinstance(header, SnapshotHeader)
                or header.version != SNAPSHOT_FORMAT_VERSION
                or header.format not in supported_formats
            ):
                raise ValueError("invalid Mojive snapshot header")
        else:
            raise ValueError(f"unsupported Mojive snapshot version: {version}")
        while True:
            offset = stream.tell()
            if not stream.read(1):
                return
            stream.seek(offset)
            try:
                yield _load_packet(stream)
            except (EOFError, pickle.UnpicklingError) as exc:
                raise ValueError("truncated Mojive snapshot packet") from exc
