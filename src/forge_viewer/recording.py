"""Stream rendered video and snapshot packets incrementally."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SNAPSHOT_MAGIC = b"FORGE-SNAPSHOT\x00\x01"


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
        self.packets = 0

    def write(self, packet: object) -> None:
        pickle.dump(packet, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.packets += 1

    def close(self) -> None:
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
        if stream.read(len(SNAPSHOT_MAGIC)) != SNAPSHOT_MAGIC:
            raise ValueError("not a forge snapshot recording")
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return
