"""Render passes for the webgpu backend.

Static imports only — unlike ``render.forge.passes`` there is no dynamic
registration; ``WgpuBackend`` wires the passes it needs directly.
"""

from __future__ import annotations

from .outline import OutlinePass
from .present import PresentPass
from .reflect import ReflectPass
from .shadow import ShadowPass
from .skybox import SkyboxPass

__all__ = ["OutlinePass", "PresentPass", "ReflectPass", "ShadowPass", "SkyboxPass"]
