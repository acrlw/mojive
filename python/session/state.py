"""Session: state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    PhysicsState,
    SceneSource,
)
from mojive.types import (
    CameraView,
    Environment,
    Light,
    Material,
)

FRAME_HISTORY_LIMIT = 300
FRAME_HISTORY_BYTE_LIMIT = 64 * 1024 * 1024
FRAME_HISTORY_TRIM_RATIO = 0.9
STATE_TAKE_FRAME_LIMIT = 100_000
STATE_TAKE_BYTE_LIMIT = 64 * 1024 * 1024


@dataclass
class PerturbState:
    """Transient state for an active translation or rotation perturbation."""

    active: bool = False
    node_id: int = -1
    object_id: int = 0
    mode: str = "translate"  # translate / rotate
    grab_point: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    start_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    target_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    plane_depth: float = 0.0

    has_local_bounds: bool = False
    local_bounds_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    local_bounds_half: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))


@dataclass(frozen=True)
class _DocumentState:
    adapter_state: object
    selected: int
    authored: AuthoredSceneOverlay


@dataclass(frozen=True)
class _StateTakeFrame:
    """One exact, transient simulation sample used by editor take replay."""

    step: int
    state: PhysicsState


@dataclass(frozen=True)
class SceneSnapshotInfo:
    """Lightweight metadata; captured arrays stay private to the owning session."""

    snapshot_id: int
    name: str
    time: float


@dataclass(frozen=True)
class _SceneSnapshot:
    info: SceneSnapshotInfo
    frame: _StateTakeFrame
    structure_revision: int


@dataclass(frozen=True)
class _LightOverride:
    """One render-slot edit retained against a stable scene object when available."""

    object_id: int
    light_index: int
    light: Light


@dataclass
class AuthoredSceneOverlay:
    """Mojive-owned property edits layered over an adapter scene."""

    lights: dict[int, _LightOverride] = field(default_factory=dict)
    environment: Environment | None = None
    materials: dict[int, Material] = field(default_factory=dict)
    geometry_colors: dict[int, np.ndarray] = field(default_factory=dict)
    geometry_color_targets: dict[int, tuple] = field(default_factory=dict)
    cameras: dict[int, CameraView] = field(default_factory=dict)

    def clear(self) -> None:
        """Discard all authored overrides and reveal adapter-owned values."""

        self.lights.clear()
        self.environment = None
        self.materials.clear()
        self.geometry_colors.clear()
        self.geometry_color_targets.clear()
        self.cameras.clear()


def _apply_geometry_color_overrides(source: SceneSource, overrides: dict[int, np.ndarray]) -> None:
    """Apply retained colors without multiplying override and instance counts."""
    if len(overrides) <= 8:
        for node_id, rgba in overrides.items():
            source.geom_rgba[source.geom_node == node_id] = rgba
        return
    for instance, node_id in enumerate(source.geom_node):
        rgba = overrides.get(int(node_id))
        if rgba is not None:
            source.geom_rgba[instance] = rgba


_SCENE_EDIT_COMMANDS = (
    cmd.AddSceneModel,
    cmd.RemoveSceneModel,
    cmd.SetSceneModelTransform,
    cmd.AddModelElement,
    cmd.DuplicateModelElement,
    cmd.RemoveModelElement,
    cmd.RenameModelElement,
    cmd.ModelEditBatch,
    cmd.SetModelSource,
    cmd.AddModelKeyframe,
    cmd.SetModelKeyframe,
    cmd.RemoveModelKeyframe,
    cmd.AddModelComponent,
    cmd.UpdateModelComponent,
    cmd.RemoveModelComponent,
    cmd.AddResourceRoot,
    cmd.RemoveResourceRoot,
    cmd.SetPose,
    cmd.SetScale,
    cmd.SetJointProperties,
    cmd.SetJointAdvancedProperties,
    cmd.SetSiteProperties,
    cmd.SetGeometryProperties,
    cmd.SetGeometryAdvancedProperties,
    cmd.SetGeometryShape,
    cmd.ImportModelGeometryResource,
    cmd.ImportModelAsset,
    cmd.SetHeightFieldSize,
    cmd.RenameModelAsset,
    cmd.DuplicateModelAsset,
    cmd.ReplaceModelAssetFile,
    cmd.RemoveModelAsset,
    cmd.SetBodyProperties,
    cmd.CreateModelMaterial,
    cmd.AddModelMaterial,
    cmd.ImportModelTexture,
    cmd.SetGeometryMaterial,
    cmd.SetLight,
    cmd.SetEnvironment,
    cmd.SetSkybox,
    cmd.SetMaterial,
    cmd.SetGeometryColor,
    cmd.SetGeometrySize,
    cmd.SetSceneCamera,
    cmd.AddSceneObject,
    cmd.RemoveSceneObject,
    cmd.AddSceneLight,
    cmd.RemoveSceneLight,
    cmd.AddSceneCamera,
    cmd.RemoveSceneCamera,
    cmd.DuplicateSceneEntity,
    cmd.RemoveSceneEntity,
    cmd.RenameSceneEntity,
)
