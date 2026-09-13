"""Render passes used by the wgpu backend."""

from __future__ import annotations

from .debug import DebugPass
from .gizmo import GizmoPass
from .outline import OutlinePass
from .present import PresentPass
from .reflect import ReflectPass
from .shadow import ShadowPass
from .skybox import SkyboxPass
from .tendon import TendonPass

__all__ = [
    "DebugPass",
    "GizmoPass",
    "OutlinePass",
    "PresentPass",
    "ReflectPass",
    "ShadowPass",
    "SkyboxPass",
    "TendonPass",
]
