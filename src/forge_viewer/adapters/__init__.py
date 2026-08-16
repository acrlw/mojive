"""Scene adapter interfaces and built-in adapters."""

from typing import TYPE_CHECKING

from .base import AdapterCaps, CameraInfo, SceneAdapter, SceneAdapterBase

if TYPE_CHECKING:
    from .mujoco_adapter import MuJoCoAdapter


def __getattr__(name: str):
    if name != "MuJoCoAdapter":
        raise AttributeError(name)
    from .mujoco_adapter import MuJoCoAdapter

    globals()[name] = MuJoCoAdapter
    return MuJoCoAdapter


__all__ = ["AdapterCaps", "CameraInfo", "MuJoCoAdapter", "SceneAdapter", "SceneAdapterBase"]
