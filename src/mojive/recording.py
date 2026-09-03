"""Stream rendered video and snapshot packets incrementally."""

from __future__ import annotations

import operator
import pickle
import subprocess
import tempfile
import warnings
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
    """Stream RGB frames to video and check that FFmpeg finishes successfully.

    Use as a context manager or call :meth:`close` to finalize the file. ``size``
    is the input (width, height). For ``yuv420p``, odd dimensions are edge-padded
    on the right/bottom to :attr:`encoded_size`, never scaled or cropped.
    """

    def __init__(
        self,
        path: Path,
        size: tuple[int, int],
        fps: float = 30.0,
        *,
        pixel_format: str = "yuv420p",
        codec: str | None = None,
    ) -> None:
        """Configure input dimensions, playback FPS, pixel format, and optional encoder.

        ``pixel_format`` accepts ``yuv420p`` (player-compatible default) or
        ``yuv444p`` (full chroma resolution). MP4 defaults to ``libx264``;
        WMV retains the ``msmpeg4`` encoder. Physics stepping is caller-owned.
        """
        self.path = Path(path)
        if len(size) != 2:
            raise ValueError("video size must be a (width, height) pair")
        self.size = tuple(operator.index(value) for value in size)
        if min(self.size) <= 0:
            raise ValueError("video width and height must be positive integers")
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("video frame rate must be finite and positive")
        if pixel_format not in {"yuv420p", "yuv444p"}:
            raise ValueError("video pixel_format must be yuv420p or yuv444p")
        self.pixel_format = pixel_format
        self.codec = codec or ("msmpeg4" if self.path.suffix.lower() == ".wmv" else "libx264")
        self.encoded_size = tuple(
            value + value % 2 if pixel_format == "yuv420p" else value for value in self.size
        )
        self._padded = None
        if self.encoded_size != self.size:
            warnings.warn(
                f"Video frames {self.size} will be edge-padded to {self.encoded_size} for "
                "yuv420p, without resizing. Use even dimensions or pixel_format='yuv444p' "
                "to preserve the exact output dimensions.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._padded = np.empty((self.encoded_size[1], self.encoded_size[0], 3), np.uint8)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio-ffmpeg. Run `uv sync` in the Mojive "
                "checkout or `python -m pip install imageio-ffmpeg`."
            ) from exc
        try:
            executable = get_ffmpeg_exe()
        except RuntimeError as exc:
            raise RuntimeError(
                "FFmpeg is unavailable. Reinstall imageio-ffmpeg or set IMAGEIO_FFMPEG_EXE "
                f"to a working FFmpeg executable. Original error: {exc}"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.encoded_size[0]}x{self.encoded_size[1]}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-vcodec",
            self.codec,
            "-pix_fmt",
            self.pixel_format,
        ]
        if self.codec == "libx264":
            command += ["-crf", "25"]
        elif self.codec == "msmpeg4":
            command += ["-q:v", "16"]
        command.append(str(self.path.resolve()))
        # A file avoids blocking on a full stderr pipe while frames are written.
        # Own the process here: write_frames() does not check the final exit code.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - owned until close()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException as exc:
            self._stderr.close()
            if not isinstance(exc, OSError):
                raise
            raise RuntimeError(
                f"Cannot start FFmpeg ({executable}). Check IMAGEIO_FFMPEG_EXE and executable "
                f"permissions. Original error: {exc}"
            ) from exc
        self.frames = 0

    def append(self, frame: np.ndarray) -> None:
        """Encode one uint8 RGB image matching the configured frame size."""
        if self._process is None:
            raise RuntimeError("Cannot append video frames after close()")
        image = np.asarray(frame)
        expected = (self.size[1], self.size[0])
        if image.shape[:2] != expected or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                f"video frames must be {expected[1]}×{expected[0]} RGB, got {image.shape}"
            )

        rgb = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if self._padded is not None:
            width, height = self.size
            self._padded[:height, :width] = rgb
            self._padded[:height, width:] = rgb[:, -1:]
            self._padded[height:] = self._padded[height - 1 : height]
            rgb = self._padded
        try:
            self._process.stdin.write(memoryview(rgb).cast("B"))
        except OSError as exc:
            try:
                self.close()
            except RuntimeError as failure:
                raise failure from exc
            raise RuntimeError(f"Failed to write video {self.path}: {exc}") from exc
        self.frames += 1

    def close(self) -> None:
        """Finalize once; report empty recordings and encoder failures."""
        process = self._process
        if process is None:
            return
        self._process = None
        reason = ""
        try:
            try:
                process.stdin.close()
            except OSError as exc:
                reason = str(exc)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                reason = "FFmpeg did not finish within 30 seconds"
                process.kill()
                process.wait()
            self._stderr.seek(0, 2)
            self._stderr.seek(max(0, self._stderr.tell() - 8192))
            details = self._stderr.read().decode("utf-8", errors="replace").strip()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            self._stderr.close()
        if process.returncode or reason or self.frames == 0:
            raise RuntimeError(
                f"Video recording failed for {self.path} (FFmpeg exit {process.returncode}, "
                f"{self.frames} frames, codec={self.codec}, pixel_format={self.pixel_format}). "
                f"{reason or ('No frames were written.' if self.frames == 0 else '')} "
                f"{details}"
            )

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except Exception as exc:
            if exc_value is None:
                raise
            exc_value.add_note(str(exc))


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
