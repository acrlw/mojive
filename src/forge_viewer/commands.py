"""Application commands and command result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import (
    DEFAULT_MATERIAL,
    CameraView,
    Environment,
    Light,
    Material,
    MeshKey,
    MeshShape,
)


@dataclass(frozen=True)
class CommandResult:
    """Result returned after a session command is routed to its owner."""

    ok: bool
    message: str = ""
    entity_id: int = -1

    def __bool__(self) -> bool:
        return self.ok

    @staticmethod
    def good(message: str = "", entity_id: int = -1) -> CommandResult:
        """Create a successful result with an optional affected entity ID."""

        return CommandResult(True, message, int(entity_id))

    @staticmethod
    def bad(message: str) -> CommandResult:
        """Create a failed result carrying a user-facing explanation."""

        assert message
        return CommandResult(False, message)


class Command:
    """Base type for state-changing operations submitted to a session."""


@dataclass(frozen=True)
class Pause(Command):
    """Pause simulation stepping."""


@dataclass(frozen=True)
class Play(Command):
    """Resume simulation stepping."""


@dataclass(frozen=True)
class Step(Command):
    """Advance the simulation by ``count`` fixed steps."""

    count: int = 1


@dataclass(frozen=True)
class Reset(Command):
    """Restore the adapter's initial state."""


@dataclass(frozen=True)
class Reload(Command):
    """Reload the current file-backed scene."""


@dataclass(frozen=True)
class LoadAsset(Command):
    """Replace the current scene with a model or workspace file."""

    path: Path


@dataclass(frozen=True)
class AddSceneModel(Command):
    """Add a file-backed model at a world-space transform."""

    path: Path
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))


@dataclass(frozen=True)
class RemoveSceneModel(Command):
    """Remove one top-level model from a composed scene."""

    model_id: int


@dataclass(frozen=True)
class SetSceneModelTransform(Command):
    """Set a model root's world-space position and 3x3 rotation matrix."""

    model_id: int
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class AddModelElement(Command):
    """Add a topology element beneath an editable model node."""

    parent_node_id: int
    kind: str
    name: str


@dataclass(frozen=True)
class RemoveModelElement(Command):
    """Remove an editable model topology element."""

    node_id: int


@dataclass(frozen=True)
class RenameModelElement(Command):
    """Rename an editable model topology element."""

    node_id: int
    name: str


@dataclass(frozen=True)
class SetModelSource(Command):
    """Replace a model with MJCF source text."""

    model_id: int
    mjcf: str


@dataclass(frozen=True)
class AddModelComponent(Command):
    """Add a model-level actuator, tendon, sensor, or equality component."""

    model_id: int
    category: str
    subtype: str
    name: str


@dataclass(frozen=True)
class UpdateModelComponent(Command):
    """Replace the editable fields and path of a model-level component."""

    model_id: int
    category: str
    component_id: int
    name: str
    fields: tuple[tuple[str, str], ...]
    path: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()


@dataclass(frozen=True)
class RemoveModelComponent(Command):
    """Remove a model-level component by stable component ID."""

    model_id: int
    category: str
    component_id: int


@dataclass(frozen=True)
class NewScene(Command):
    """Create an empty authored scene document."""


@dataclass(frozen=True)
class OpenScene(Command):
    """Open a Forge scene, workspace, MJCF, or URDF file."""

    path: Path


@dataclass(frozen=True)
class SaveScene(Command):
    """Save the active scene and optionally insert the current pose as a keyframe."""

    path: Path
    current_pose_keyframe: str | None = None


@dataclass(frozen=True)
class AddResourceRoot(Command):
    """Add a directory used to resolve external model resources."""

    path: Path


@dataclass(frozen=True)
class RemoveResourceRoot(Command):
    """Remove a resource search directory from the workspace."""

    path: Path


@dataclass(frozen=True)
class BeginEditTransaction(Command):
    """Begin grouping subsequent edits into one undo record."""

    label: str = "Edit"


@dataclass(frozen=True)
class EndEditTransaction(Command):
    """Commit the active edit transaction to undo history."""


@dataclass(frozen=True)
class Undo(Command):
    """Restore the document state before the latest edit transaction."""


@dataclass(frozen=True)
class Redo(Command):
    """Reapply the latest undone edit transaction."""


@dataclass(frozen=True)
class LoadKeyframe(Command):
    """Restore a model-defined keyframe by stable keyframe ID."""

    keyframe_id: int


@dataclass(frozen=True)
class Select(Command):
    """Select the hierarchy node associated with an object ID."""

    object_id: int


@dataclass(frozen=True)
class SelectNode(Command):
    """Select a hierarchy node directly by node ID."""

    node_id: int


@dataclass(frozen=True)
class SetPose(Command):
    """Set a node's world-space position and 3x3 rotation matrix."""

    node_id: int
    position: np.ndarray
    rotation: np.ndarray  # 3×3


@dataclass(frozen=True)
class SetLight(Command):
    """Replace an existing scene light definition."""

    light_id: int
    light: Light


@dataclass(frozen=True)
class SetEnvironment(Command):
    """Replace scene-wide environment and atmosphere settings."""

    environment: Environment


@dataclass(frozen=True)
class SetMaterial(Command):
    """Replace a scene material by material ID."""

    material_id: int
    material: Material


@dataclass(frozen=True)
class SetGeometryColor(Command):
    """Set an editable geometry node's linear RGBA color."""

    node_id: int
    rgba: np.ndarray


@dataclass(frozen=True)
class SetGeometrySize(Command):
    """Set an editable geometry node's shape-specific size vector."""

    node_id: int
    size: np.ndarray


@dataclass(frozen=True)
class SetSceneCamera(Command):
    """Replace a scene camera by camera ID."""

    camera_id: int
    camera: CameraView


@dataclass(frozen=True)
class AddSceneObject(Command):
    """Add a primitive or mesh object to an authored scene."""

    shape: MeshShape | MeshKey
    name: str = "object"
    size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    color: tuple[float, float, float, float] = (0.65, 0.68, 0.72, 1.0)
    material: Material = DEFAULT_MATERIAL


@dataclass(frozen=True)
class RemoveSceneObject(Command):
    """Remove an authored scene object by object ID."""

    object_id: int


@dataclass(frozen=True)
class AddSceneLight(Command):
    """Add a named light to an authored scene."""

    name: str
    light: Light


@dataclass(frozen=True)
class RemoveSceneLight(Command):
    """Remove an authored light by light ID."""

    light_id: int


@dataclass(frozen=True)
class AddSceneCamera(Command):
    """Add a named camera to an authored scene."""

    name: str
    camera: CameraView


@dataclass(frozen=True)
class RemoveSceneCamera(Command):
    """Remove an authored camera by camera ID."""

    camera_id: int


@dataclass(frozen=True)
class DuplicateSceneEntity(Command):
    """Duplicate an authored object, light, or camera."""

    object_id: int


@dataclass(frozen=True)
class RemoveSceneEntity(Command):
    """Remove an authored object, light, or camera by object ID."""

    object_id: int


@dataclass(frozen=True)
class RenameSceneEntity(Command):
    """Rename an authored object, light, or camera."""

    object_id: int
    name: str


@dataclass(frozen=True)
class SetQpos(Command):
    """Set one generalized position by flat qpos index."""

    index: int
    value: float


@dataclass(frozen=True)
class SetEqualityEnabled(Command):
    """Enable or disable a model equality constraint."""

    constraint_id: int
    enabled: bool


@dataclass(frozen=True)
class SetCtrl(Command):
    """Set one actuator control value by flat control index."""

    index: int
    value: float


@dataclass(frozen=True)
class Perturb(Command):
    """Apply a world-space translation or rotation perturbation target."""

    node_id: int
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    target_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    mode: str = "translate"


@dataclass(frozen=True)
class ClearPerturb(Command):
    """Release the active physical perturbation."""


@dataclass(frozen=True)
class SetCamera(Command):
    """Set the editor viewport camera."""

    camera: CameraView


@dataclass(frozen=True)
class SetVisible(Command):
    """Set hierarchy-node visibility."""

    node_id: int
    visible: bool


@dataclass(frozen=True)
class SetVisualGroup(Command):
    """Set visibility for one numbered model visual group."""

    category: str
    group: int
    visible: bool


@dataclass(frozen=True)
class SetSpeed(Command):
    """Set simulation playback speed relative to real time."""

    factor: float


class Query:
    """Base type for read-only session operations."""


@dataclass(frozen=True)
class Pick(Query):
    """Raycast from a world-space origin along a normalized direction."""

    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class NodeAt(Query):
    """Resolve an object ID to its hierarchy node."""

    object_id: int


@dataclass(frozen=True)
class Bounds(Query):
    """Return the current scene's world-space axis-aligned bounds."""
