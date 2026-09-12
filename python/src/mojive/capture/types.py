"""Capture and interactive recording contracts shared by viewer frontends."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


def encode_image(image: np.ndarray, encoding: str = "raw") -> dict:
    """Encode a top-left image for JSON transport without creating a file."""
    image = np.ascontiguousarray(image)
    if encoding == "raw":
        payload = memoryview(image).cast("B")
    else:
        buffer = io.BytesIO()
        if encoding == "png":
            from PIL import Image

            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("PNG encoding requires an RGB uint8 image")
            Image.fromarray(image).save(buffer, format="PNG")
        elif encoding == "npy":
            np.save(buffer, image, allow_pickle=False)
        else:
            raise ValueError(f"Unknown image encoding: {encoding}")
        payload = buffer.getbuffer()
    return {
        "transport": "base64",
        "encoding": encoding,
        "data": base64.b64encode(payload).decode("ascii"),
        "shape": list(image.shape),
        "dtype": image.dtype.str,
        "orientation": "top_left",
    }


def decode_image(payload: dict) -> np.ndarray:
    """Decode an in-memory capture response into an owned NumPy array."""
    data = base64.b64decode(payload["data"], validate=True)
    encoding = payload["encoding"]
    if encoding == "raw":
        image = np.frombuffer(data, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])
    elif encoding == "npy":
        image = np.load(io.BytesIO(data), allow_pickle=False)
    elif encoding == "png":
        from PIL import Image

        with Image.open(io.BytesIO(data)) as decoded:
            image = np.asarray(decoded)
    else:
        raise ValueError(f"Unknown image encoding: {encoding}")
    if list(image.shape) != payload["shape"] or image.dtype != np.dtype(payload["dtype"]):
        raise ValueError("Capture array does not match its shape or dtype metadata")
    return image.copy()


class CaptureSurface(StrEnum):
    """Choose which composed image a capture or recording contains."""

    SCENE = "scene"
    VIEWPORT = "viewport"
    WINDOW = "window"


class RecordingPhase(StrEnum):
    """Lifecycle state of an interactive viewer recording."""

    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    PAUSED = "paused"


@dataclass(frozen=True)
class RecordingInfo:
    """Read-only snapshot of the interactive recorder state."""

    phase: RecordingPhase = RecordingPhase.IDLE
    surface: CaptureSurface = CaptureSurface.SCENE
    path: Path | None = None
    fps: float = 60.0
    frames: int = 0
    duration: float = 0.0
    countdown_remaining: float = 0.0

    @property
    def active(self) -> bool:
        return self.phase is not RecordingPhase.IDLE


__all__ = ["CaptureSurface", "RecordingInfo", "RecordingPhase", "decode_image", "encode_image"]
