"""Application state, selection, overrides, and command routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import commands as cmd
from .adapters.base import (
    ENVIRONMENT_OBJECT_ID,
    ActuatorInfo,
    BodyProperties,
    CameraInfo,
    EqualityConstraintInfo,
    FrameNeeds,
    GeometryAdvancedProperties,
    GeometryProperties,
    JointInfo,
    KeyframeInfo,
    NodeType,
    PhysicsState,
    SceneAdapter,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSaveOptions,
    SceneSource,
    SensorInfo,
)
from .commands import Command, CommandResult, Query
from .types import CameraView, Environment, Light, Material, TextureType


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

    body_radius: float = 0.1


@dataclass(frozen=True)
class _DocumentState:
    adapter_state: object
    selected: int


@dataclass(frozen=True)
class _EditRecord:
    label: str
    before: _DocumentState
    after: _DocumentState
    before_revision: int
    after_revision: int


@dataclass(frozen=True)
class _LightOverride:
    """One render-slot edit retained against a stable scene object when available."""

    object_id: int
    light_index: int
    light: Light


@dataclass
class AuthoredSceneOverlay:
    """Forge-owned property edits layered over an adapter scene."""

    lights: dict[int, _LightOverride] = field(default_factory=dict)
    environment: Environment | None = None
    materials: dict[int, Material] = field(default_factory=dict)
    geometry_colors: dict[int, np.ndarray] = field(default_factory=dict)
    cameras: dict[int, CameraView] = field(default_factory=dict)

    def clear(self) -> None:
        """Discard all authored overrides and reveal adapter-owned values."""

        self.lights.clear()
        self.environment = None
        self.materials.clear()
        self.geometry_colors.clear()
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
    cmd.RemoveModelElement,
    cmd.RenameModelElement,
    cmd.ModelEditBatch,
    cmd.SetModelSource,
    cmd.AddModelComponent,
    cmd.UpdateModelComponent,
    cmd.RemoveModelComponent,
    cmd.AddResourceRoot,
    cmd.RemoveResourceRoot,
    cmd.SetPose,
    cmd.SetJointProperties,
    cmd.SetGeometryProperties,
    cmd.SetGeometryAdvancedProperties,
    cmd.SetBodyProperties,
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


class Session:
    """Own viewer state and route typed commands to one scene adapter.

    The session separates UI and renderer code from physics-specific methods. It
    tracks selection, pause and step state, authored overrides, edit history, and
    stable-structure generations while the adapter owns simulation data.
    """

    def __init__(self, adapter: SceneAdapter, asset_path: Path | None = None) -> None:
        self._adapter = adapter
        self._asset_path = asset_path
        self._paused = not adapter.caps.simulation
        self._speed = 1.0
        self._sim_time_credit = 0.0
        self._selected = 0
        self._selected_node_id = -1
        self._unlocked_entity_gizmos: set[int] = set()
        self._step_counter = 0
        self._pending_steps = 0
        self._frame = SceneFrame()
        self._source: SceneSource | None = None
        self._authored = AuthoredSceneOverlay()
        self._nodes: list[SceneNode] = []
        self._by_node_id: dict[int, SceneNode] = {}
        self._by_object_id: dict[int, SceneNode] = {}
        self._joints: list[JointInfo] = []
        self._joints_by_body: dict[int, tuple[JointInfo, ...]] = {}
        self._actuators: list[ActuatorInfo] = []
        self._actuators_by_joint: dict[int, tuple[ActuatorInfo, ...]] = {}
        self._cameras: list[CameraInfo] = []
        self._camera_slot_by_id: dict[int, int] = {}
        self._keyframes: list[KeyframeInfo] = []
        self._sensor_infos: list[SensorInfo] = []
        self._equality_constraints: list[EqualityConstraintInfo] = []
        self._active_keyframe = -1
        self._perturb = PerturbState()
        self._camera = CameraView()
        self._last_message = ""
        self._undo_stack: list[_EditRecord] = []
        self._redo_stack: list[_EditRecord] = []
        self._edit_before: _DocumentState | None = None
        self._edit_before_revision = 0
        self._edit_label = ""
        self._edit_changed = False
        self._document_revision = 0
        self._saved_revision = 0
        self._next_document_revision = 1
        self._structure_generation = 0
        self._adapter_revision = -1
        self._refresh_structure()

    @property
    def adapter(self) -> SceneAdapter:
        """Return the scene adapter owned by this session."""
        return self._adapter

    @property
    def paused(self) -> bool:
        """Return the effective simulation pause state."""
        return self._paused

    @property
    def speed(self) -> float:
        """Return the real-time simulation speed multiplier."""
        return self._speed

    @property
    def selected(self) -> int:
        """Return the selected render object ID, or zero when selection is empty."""
        return self._selected

    @property
    def selected_node(self) -> SceneNode | None:
        """Return the selected hierarchy node."""
        return self.node(self._selected_node_id)

    def entity_gizmo_lock_enabled(self, node: SceneNode) -> bool:
        """Return the runtime gizmo-lock preference for a camera or light."""
        return node.object_id not in self._unlocked_entity_gizmos

    def set_entity_gizmo_lock(self, node: SceneNode, enabled: bool) -> None:
        """Set the runtime gizmo-lock preference for a camera or light."""
        if enabled:
            self._unlocked_entity_gizmos.discard(node.object_id)
        else:
            self._unlocked_entity_gizmos.add(node.object_id)

    def entity_gizmo_locked(self, node: SceneNode) -> bool:
        """Return whether a running simulation currently locks this entity gizmo."""
        return (
            not self._paused
            and node.type in (NodeType.CAMERA, NodeType.LIGHT)
            and self.entity_gizmo_lock_enabled(node)
        )

    @property
    def frame(self) -> SceneFrame:
        """Return the most recent dynamic frame produced by :meth:`tick`."""
        return self._frame

    @property
    def source(self) -> SceneSource | None:
        """Return the current stable scene source."""
        return self._source

    @property
    def nodes(self) -> list[SceneNode]:
        """Return hierarchy nodes for the current structure generation."""
        return self._nodes

    @property
    def joints(self) -> list[JointInfo]:
        """Return editable joint metadata from the adapter."""
        return self._joints

    def joints_for_body(self, body_index: int) -> tuple[JointInfo, ...]:
        """Return joints attached directly to one physics body."""
        return self._joints_by_body.get(int(body_index), ())

    @property
    def actuators(self) -> list[ActuatorInfo]:
        """Return actuator control metadata from the adapter."""
        return self._actuators

    def actuators_for_joint(self, joint_id: int) -> tuple[ActuatorInfo, ...]:
        """Return actuators attached directly to one joint."""
        return self._actuators_by_joint.get(int(joint_id), ())

    @property
    def cameras(self) -> list[CameraInfo]:
        """Return selectable model and authored camera metadata."""
        return self._cameras

    @property
    def keyframes(self) -> list[KeyframeInfo]:
        """Return available physics keyframes."""
        return self._keyframes

    @property
    def active_keyframe(self) -> int:
        """Return the loaded keyframe ID, or ``-1`` for the current state."""
        return self._active_keyframe

    @property
    def sensor_infos(self) -> list[SensorInfo]:
        """Return sensor metadata for slices of the current sensor array."""
        return self._sensor_infos

    @property
    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        """Return editable equality-constraint metadata."""
        return self._equality_constraints

    @property
    def perturb(self) -> PerturbState:
        """Return the active physics perturbation state."""
        return self._perturb

    @property
    def camera(self) -> CameraView:
        """Return the current viewport camera submitted through commands."""
        return self._camera

    @property
    def authored_overlay(self) -> AuthoredSceneOverlay:
        """Return Forge-authored overrides layered over adapter structure."""
        return self._authored

    @property
    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        """Return file-backed models participating in the composed scene."""
        return self._adapter.scene_models()

    def model_components(self, model_id: int, category: str):
        """Return editable components in one model-level MJCF category."""
        return self._adapter.model_components(model_id, category)

    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]:
        """Return supported component subtypes for a model and category."""
        return self._adapter.model_component_presets(model_id, category)

    def model_material_indices(self, model_id: int) -> tuple[int, ...]:
        """Return render material indices owned by one editable model."""
        return self._adapter.model_material_indices(model_id)

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        """Return editable contact parameters for one model geometry."""
        return self._adapter.geometry_properties(node_id)

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        """Return geometry properties backed by rebuilt MuJoCo constants."""
        return self._adapter.geometry_advanced_properties(node_id)

    def body_properties(self, node_id: int) -> BodyProperties | None:
        """Return editable inertial and dynamic properties for one model body."""
        return self._adapter.body_properties(node_id)

    def model_texture_names(self, model_id: int) -> tuple[str, ...]:
        """Return compiled texture names owned by one editable model."""
        return self._adapter.model_texture_names(model_id)

    @property
    def asset_path(self) -> Path | None:
        """Return the current document or model path."""
        return self._asset_path

    @property
    def last_message(self) -> str:
        """Return the latest user-facing command result message."""
        return self._last_message

    @property
    def dirty(self) -> bool:
        """Return whether the current document contains unsaved edits."""
        return self._edit_changed or self._document_revision != self._saved_revision

    @property
    def current_pose_modified(self) -> bool:
        """Return whether physics state differs from its saved initial pose."""
        return self._adapter.current_pose_modified()

    @property
    def can_undo(self) -> bool:
        """Return whether one document edit can be undone."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Return whether one document edit can be redone."""
        return bool(self._redo_stack)

    @property
    def editing(self) -> bool:
        """Return whether a continuous edit transaction is active."""
        return self._edit_before is not None

    @property
    def structure_generation(self) -> int:
        """Return the generation incremented after stable structure changes."""
        return self._structure_generation

    def node(self, node_id: int) -> SceneNode | None:
        """Look up a hierarchy node by node ID."""
        return self._by_node_id.get(int(node_id))

    def node_by_object_id(self, object_id: int) -> SceneNode | None:
        """Look up a hierarchy node by selectable object ID."""
        return self._by_object_id.get(int(object_id))

    def restore_physics_state(
        self, state: PhysicsState, *, active_keyframe: int = -1
    ) -> CommandResult:
        """Restore a complete physics state while the session is paused."""
        if not self._paused:
            return CommandResult.bad("physics is running; pause to restore a scene snapshot")
        if not self._adapter.restore_state(state):
            return CommandResult.bad("scene snapshot state is incompatible with this model")
        keyframe_id = int(active_keyframe)
        self._active_keyframe = (
            keyframe_id
            if keyframe_id == -1
            or any(keyframe.keyframe_id == keyframe_id for keyframe in self._keyframes)
            else -1
        )
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        return CommandResult.good("Scene state restored")

    def tick(self, needs: FrameNeeds, wall_dt: float | None = None) -> SceneFrame:
        """Advance simulation time and obtain one composed dynamic frame.

        Args:
            needs: Optional dynamic arrays required by current consumers.
            wall_dt: Elapsed wall time used for real-time simulation scheduling.
        """
        if not self._paused and not self._adapter.caps.external_clock:
            timestep = self._adapter.timestep()
            if wall_dt is not None and timestep > 0.0:
                self._sim_time_credit += float(wall_dt) * self._speed
                n = int(self._sim_time_credit / timestep + 1e-9)
                self._sim_time_credit -= n * timestep
            else:
                n = max(1, round(self._speed))
            if n:
                self._adapter.step(n)
                self._step_counter += n
        elif self._pending_steps > 0:
            self._adapter.step(self._pending_steps)
            self._step_counter += self._pending_steps
            self._pending_steps = 0

        prepare_frame = getattr(self._adapter, "prepare_frame", None)
        if prepare_frame is not None:
            prepare_frame(needs)
        if self._adapter.structure_revision != self._adapter_revision:
            self._refresh_structure()

        self._frame = self._adapter.frame(needs)
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()
        if self._adapter.caps.external_clock:
            self._paused = bool(self._frame.paused)
        else:
            self._frame.paused = self._paused
            self._frame.step = self._step_counter
        return self._frame

    def submit(self, command: Command) -> CommandResult:
        """Apply one typed command and update edit history and status text."""
        if isinstance(command, cmd.BeginEditTransaction):
            result = self._begin_edit(command.label)
            self._last_message = result.message
            return result
        if isinstance(command, cmd.EndEditTransaction):
            result = self._end_edit()
            self._last_message = result.message
            return result
        if isinstance(command, cmd.Undo):
            result = self._undo()
            self._last_message = result.message
            return result
        if isinstance(command, cmd.Redo):
            result = self._redo()
            self._last_message = result.message
            return result

        scene_edit = isinstance(command, _SCENE_EDIT_COMMANDS)
        before = (
            self._capture_document_state()
            if scene_edit and self._adapter.caps.edit_history and not self.editing
            else None
        )
        result = self._dispatch(command)
        if result.ok and scene_edit and self._adapter.caps.scene_files:
            if self.editing:
                self._edit_changed = True
            elif before is not None:
                self._commit_edit(type(command).__name__, before)
            else:
                self._advance_document_revision()
        self._last_message = result.message
        return result

    def _capture_document_state(self) -> _DocumentState:
        state = self._adapter.capture_edit_state()
        if state is None:
            raise RuntimeError(f"{self._adapter.caps.name} did not provide an edit state")
        return _DocumentState(state, self._selected)

    def _restore_document_state(self, state: _DocumentState) -> bool:
        if not self._adapter.restore_edit_state(state.adapter_state):
            return False
        self._authored.clear()
        self._selected = int(state.selected)
        self._selected_node_id = -1
        self._refresh_structure()
        if self._selected not in self._by_object_id:
            self._selected = 0
            self._selected_node_id = -1
        return True

    def _begin_edit(self, label: str) -> CommandResult:
        if not self._adapter.caps.edit_history:
            return CommandResult.bad(f"{self._adapter.caps.name} does not support edit history")
        if self.editing:
            return CommandResult.bad("An edit transaction is already active")
        self._edit_before = self._capture_document_state()
        self._edit_before_revision = self._document_revision
        self._edit_label = str(label) or "Edit"
        self._edit_changed = False
        return CommandResult.good()

    def _end_edit(self) -> CommandResult:
        if not self.editing:
            return CommandResult.bad("No edit transaction is active")
        before = self._edit_before
        label = self._edit_label
        changed = self._edit_changed
        before_revision = self._edit_before_revision
        self._edit_before = None
        self._edit_label = ""
        self._edit_changed = False
        if changed and before is not None:
            self._commit_edit(label, before, before_revision)
            return CommandResult.good(label)
        return CommandResult.good()

    def _commit_edit(
        self,
        label: str,
        before: _DocumentState,
        before_revision: int | None = None,
    ) -> None:
        revision = self._document_revision if before_revision is None else before_revision
        after_revision = self._next_document_revision
        self._next_document_revision += 1
        self._undo_stack.append(
            _EditRecord(
                label,
                before,
                self._capture_document_state(),
                revision,
                after_revision,
            )
        )
        del self._undo_stack[:-100]
        self._redo_stack.clear()
        self._document_revision = after_revision

    def _advance_document_revision(self) -> None:
        self._document_revision = self._next_document_revision
        self._next_document_revision += 1
        self._redo_stack.clear()

    def _undo(self) -> CommandResult:
        if self.editing:
            return CommandResult.bad("Finish the active edit before undo")
        if not self._undo_stack:
            return CommandResult.bad("Nothing to undo")
        record = self._undo_stack.pop()
        if not self._restore_document_state(record.before):
            self._undo_stack.append(record)
            return CommandResult.bad("Undo state is incompatible with this scene")
        self._redo_stack.append(record)
        self._document_revision = record.before_revision
        return CommandResult.good(f"Undo {record.label}")

    def _redo(self) -> CommandResult:
        if self.editing:
            return CommandResult.bad("Finish the active edit before redo")
        if not self._redo_stack:
            return CommandResult.bad("Nothing to redo")
        record = self._redo_stack.pop()
        if not self._restore_document_state(record.after):
            self._redo_stack.append(record)
            return CommandResult.bad("Redo state is incompatible with this scene")
        self._undo_stack.append(record)
        self._document_revision = record.after_revision
        return CommandResult.good(f"Redo {record.label}")

    def _reset_edit_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._edit_before = None
        self._edit_label = ""
        self._edit_changed = False
        self._document_revision = 0
        self._saved_revision = 0
        self._next_document_revision = 1

    def _pause_loaded_scene(self) -> None:
        """Request an editable pause and reset state associated with the old scene."""

        self._paused = not self._adapter.caps.simulation or self._adapter.set_paused(True)
        self._step_counter = 0
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        self._active_keyframe = -1

    def _dispatch(self, c: Command) -> CommandResult:
        caps = self._adapter.caps

        if isinstance(c, cmd.Pause):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to pause")
            if self._paused:
                return CommandResult.good("Simulation is already paused")
            if not self._adapter.set_paused(True):
                return CommandResult.bad("physics backend rejected pause")
            self._paused = True
            self._sim_time_credit = 0.0
            return CommandResult.good("Simulation paused")

        if isinstance(c, cmd.Play):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to resume")
            if not self._paused:
                return CommandResult.good("Simulation is already running")
            if not self._adapter.set_paused(False):
                return CommandResult.bad("physics backend rejected play")
            self._paused = False
            self._sim_time_credit = 0.0
            self._perturb = PerturbState()
            return CommandResult.good("Simulation resumed")

        if isinstance(c, cmd.Step):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation to step")
            if not self._paused:
                return CommandResult.bad("Pause the simulation before stepping")
            count = int(c.count)
            if count <= 0:
                return CommandResult.bad("step count must be positive")
            self._pending_steps += count
            return CommandResult.good(f"Stepped {count} frame(s)")

        if isinstance(c, cmd.Reset):
            self._adapter.reset()
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._equality_constraints = (
                self._adapter.equality_constraints() if caps.equality_constraints else []
            )

            return CommandResult.good("Scene reset")

        if isinstance(c, cmd.Reload):
            if not caps.reload:
                return CommandResult.bad(f"{caps.name} does not support reload")
            try:
                self._adapter.reload()
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._pause_loaded_scene()
            self._step_counter = 0
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._authored.clear()
            self._refresh_structure()
            self._reset_edit_history()
            return CommandResult.good("Scene reloaded")

        if isinstance(c, cmd.NewScene):
            if not caps.scene_files:
                return CommandResult.bad(f"{caps.name} does not support scene files")
            self._adapter.new_scene()
            self._pause_loaded_scene()
            self._asset_path = None
            self._selected = 0
            self._selected_node_id = -1
            self._authored.clear()
            self._refresh_structure()
            self._reset_edit_history()
            return CommandResult.good("New scene")

        if isinstance(c, cmd.OpenScene):
            if not caps.scene_files:
                return CommandResult.bad(f"{caps.name} does not support scene files")
            path = Path(c.path).expanduser().resolve()
            try:
                self._adapter.open_scene(path)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._pause_loaded_scene()
            self._asset_path = path
            self._selected = 0
            self._selected_node_id = -1
            self._authored.clear()
            self._refresh_structure()
            self._reset_edit_history()
            return CommandResult.good(f"Opened {path.name}")

        if isinstance(c, cmd.SaveScene):
            if not caps.scene_files:
                return CommandResult.bad(f"{caps.name} does not support scene files")
            path = Path(c.path).expanduser().resolve()
            try:
                self._adapter.save_scene(
                    path,
                    SceneSaveOptions(current_pose_keyframe=c.current_pose_keyframe),
                )
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._asset_path = path
            self._saved_revision = self._document_revision
            return CommandResult.good(f"Saved {path.name}")

        if isinstance(c, cmd.LoadAsset):
            if not caps.asset_loading:
                return CommandResult.bad(f"{caps.name} does not support model loading")
            path = Path(c.path).expanduser().resolve()
            try:
                self._adapter.load(path)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            self._pause_loaded_scene()
            self._asset_path = path
            self._step_counter = 0
            self._selected = 0
            self._selected_node_id = -1
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._authored.clear()
            self._refresh_structure()
            self._reset_edit_history()
            return CommandResult.good(f"Loaded {c.path.name}")

        if isinstance(c, cmd.AddSceneModel):
            if not caps.model_composition:
                return CommandResult.bad(f"{caps.name} does not support model composition")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            path = Path(c.path).expanduser().resolve()
            try:
                model_id = self._adapter.add_scene_model(path, c.position, c.rotation)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if model_id < 0:
                return CommandResult.bad(f"Failed to add {path.name}")
            self._selected = 0
            self._selected_node_id = -1
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._refresh_structure()
            return CommandResult.good(f"Added {path.name}", model_id)

        if isinstance(c, cmd.RemoveSceneModel):
            if not caps.model_composition:
                return CommandResult.bad(f"{caps.name} does not support model composition")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            info = next(
                (item for item in self.scene_models if item.model_id == int(c.model_id)), None
            )
            if info is None or not info.removable:
                return CommandResult.bad(f"Model {c.model_id} cannot be removed")
            try:
                removed = self._adapter.remove_scene_model(c.model_id)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not removed:
                return CommandResult.bad(f"Failed to remove {info.name}")
            self._selected = 0
            self._selected_node_id = -1
            self._perturb = PerturbState()
            self._active_keyframe = -1
            self._refresh_structure()
            return CommandResult.good(f"Removed {info.name}")

        if isinstance(c, cmd.SetSceneModelTransform):
            if not caps.model_composition:
                return CommandResult.bad(f"{caps.name} does not support model composition")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before moving a model root")
            try:
                changed = self._adapter.set_scene_model_transform(
                    c.model_id, c.position, c.rotation
                )
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"Model {c.model_id} cannot be transformed")
            self._refresh_structure()
            return CommandResult.good("Updated model transform")

        if isinstance(c, cmd.AddModelElement):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            try:
                node_id = self._adapter.add_model_element(c.parent_node_id, c.element_type, c.name)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if node_id < 0:
                return CommandResult.bad(f"Failed to add {c.element_type}")
            self._refresh_structure()
            return CommandResult.good(f"Added {c.name}", node_id)

        if isinstance(c, cmd.RemoveModelElement):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            try:
                changed = self._adapter.remove_model_element(c.node_id)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"{node.name} cannot be removed from the model")
            self._selected = 0
            self._selected_node_id = -1
            self._refresh_structure()
            return CommandResult.good(f"Removed {node.name}")

        if isinstance(c, cmd.RenameModelElement):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            try:
                changed = self._adapter.rename_model_element(c.node_id, c.name)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"{node.name} cannot be renamed")
            self._refresh_structure()
            return CommandResult.good(f"Renamed {node.name}")

        if isinstance(c, cmd.ModelEditBatch):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            if not c.edits:
                return CommandResult.bad("A model edit batch cannot be empty")
            if not all(isinstance(edit, cmd.ModelEdit) for edit in c.edits):
                return CommandResult.bad("A model edit batch contains an unsupported operation")
            selected = self.selected_node
            selected_identity = (
                (selected.model_id, selected.type, selected.name) if selected is not None else None
            )
            if selected is not None:
                for edit in c.edits:
                    if (
                        not isinstance(
                            edit, (cmd.RemoveModelElementEdit, cmd.RenameModelElementEdit)
                        )
                        or int(edit.target.node_id) != selected.node_id
                    ):
                        continue
                    if isinstance(edit, cmd.RemoveModelElementEdit):
                        selected_identity = None
                    elif isinstance(edit, cmd.RenameModelElementEdit):
                        selected_identity = (selected.model_id, selected.type, edit.name)
            try:
                node_ids = self._adapter.apply_model_edit_batch(c.edits)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if len(node_ids) != len(c.edits):
                return CommandResult.bad("The model edit batch was not applied")
            self._selected = 0
            self._selected_node_id = -1
            self._refresh_structure()
            if selected_identity is not None:
                model_id, node_type, name = selected_identity
                body_types = {NodeType.LINK, NodeType.ROBOT}
                restored = next(
                    (
                        node
                        for node in self._nodes
                        if node.model_id == model_id
                        and node.name == name
                        and (
                            node.type is node_type
                            or (node.type in body_types and node_type in body_types)
                        )
                    ),
                    None,
                )
                if restored is not None:
                    self._selected = restored.object_id
                    self._selected_node_id = restored.node_id
            entity_id = next((node_id for node_id in reversed(node_ids) if node_id >= 0), -1)
            return CommandResult.good(f"Applied {len(c.edits)} model edits", entity_id)

        if isinstance(c, cmd.SetModelSource):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            try:
                changed = self._adapter.set_scene_model_xml(c.model_id, c.mjcf)
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"Model {c.model_id} source cannot be updated")
            self._selected = 0
            self._selected_node_id = -1
            self._refresh_structure()
            return CommandResult.good("Updated MJCF source")

        if isinstance(c, cmd.AddModelComponent):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            try:
                component_id = self._adapter.add_model_component(
                    c.model_id, c.category, c.subtype, c.name
                )
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if component_id < 0:
                return CommandResult.bad(f"Failed to add {c.category} {c.name}")
            self._refresh_structure()
            return CommandResult.good(f"Added {c.category} {c.name}", component_id)

        if isinstance(c, cmd.UpdateModelComponent):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            try:
                changed = self._adapter.update_model_component(
                    c.model_id,
                    c.category,
                    c.component_id,
                    c.name,
                    c.fields,
                    c.path,
                )
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"{c.category} {c.component_id} cannot be updated")
            self._refresh_structure()
            return CommandResult.good(f"Updated {c.category} {c.name}")

        if isinstance(c, cmd.RemoveModelComponent):
            if not caps.topology_editing:
                return CommandResult.bad(f"{caps.name} does not support topology editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before changing model topology")
            try:
                changed = self._adapter.remove_model_component(
                    c.model_id, c.category, c.component_id
                )
            except Exception as exc:
                return CommandResult.bad(str(exc))
            if not changed:
                return CommandResult.bad(f"{c.category} {c.component_id} cannot be removed")
            self._refresh_structure()
            return CommandResult.good(f"Removed {c.category}")

        if isinstance(c, cmd.AddResourceRoot):
            if not caps.scene_files or not c.path.is_dir():
                return CommandResult.bad(f"Resource directory is unavailable: {c.path}")
            return (
                CommandResult.good(f"Added resource directory {c.path}")
                if self._adapter.add_resource_root(c.path)
                else CommandResult.bad(f"Failed to add resource directory {c.path}")
            )

        if isinstance(c, cmd.RemoveResourceRoot):
            return (
                CommandResult.good(f"Removed resource directory {c.path}")
                if self._adapter.remove_resource_root(c.path)
                else CommandResult.bad(f"Resource directory is unavailable: {c.path}")
            )

        if isinstance(c, cmd.LoadKeyframe):
            if not caps.keyframes:
                return CommandResult.bad(f"{caps.name} does not expose keyframes")
            if not self._paused:
                return CommandResult.bad("physics is running; pause to load a keyframe")
            i = int(c.keyframe_id)
            slot = self._keyframe_slot(i)
            if slot < 0:
                return CommandResult.bad(f"keyframe {i} is unavailable")
            if not self._adapter.load_keyframe(i):
                return CommandResult.bad(f"failed to load keyframe {i}")
            self._step_counter = 0
            self._sim_time_credit = 0.0
            self._pending_steps = 0
            self._perturb = PerturbState()
            self._active_keyframe = i
            return CommandResult.good(f"loaded {self._keyframes[slot].name}")

        if isinstance(c, cmd.Select):
            node = self._by_object_id.get(int(c.object_id))
            if c.object_id and node is None:
                return CommandResult.bad(f"Unknown object_id={c.object_id}")
            self._selected = int(c.object_id)
            self._selected_node_id = node.node_id if node is not None else -1
            return CommandResult.good(node.name if node else "Selection cleared")

        if isinstance(c, cmd.SelectNode):
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            self._selected_node_id = node.node_id
            self._selected = int(node.object_id)
            return CommandResult.good(node.name)

        if isinstance(c, cmd.SetVisible):
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            node.visible = c.visible
            if self._source is not None:
                source_node = next(
                    (item for item in self._source.nodes if item.node_id == c.node_id), None
                )
                if source_node is not None:
                    source_node.visible = c.visible
            self._structure_generation += 1
            return CommandResult.good("")

        if isinstance(c, cmd.SetVisualGroup):
            if not caps.visual_groups:
                return CommandResult.bad(f"{caps.name} does not expose visual groups")
            ok = self._adapter.set_visual_group(c.category, c.group, c.visible)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"visual group {c.category}:{c.group} is unavailable")
            )

        if isinstance(c, cmd.SetPose):
            if not caps.write_pose:
                return CommandResult.bad(f"{caps.name} does not support pose editing")
            if not self._paused:
                return CommandResult.bad("physics is running; pause to move things")
            node = self.node(c.node_id)
            if node is None:
                return CommandResult.bad(f"Unknown node_id={c.node_id}")
            if not node.posable:
                return CommandResult.bad(
                    "This link is joint-driven; use its viewport gizmo or the Joints panel"
                )
            ok = self._adapter.set_pose(c.node_id, c.position, c.rotation)
            return CommandResult.good("") if ok else CommandResult.bad("Pose update failed")

        if isinstance(c, cmd.SetQpos):
            if not caps.write_qpos:
                return CommandResult.bad(f"{caps.name} does not support joint editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing joints")
            ok = self._adapter.set_qpos(c.index, c.value)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"Joint {c.index} update failed")
            )

        if isinstance(c, cmd.SetQposBatch):
            if not caps.write_qpos:
                return CommandResult.bad(f"{caps.name} does not support joint editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing joints")
            raw_indices = np.asarray(c.indices).reshape(-1)
            if not np.issubdtype(raw_indices.dtype, np.integer):
                return CommandResult.bad("Joint batch indices must be integers")
            indices = raw_indices.astype(np.intp, copy=False)
            values = np.asarray(c.values, np.float64).reshape(-1)
            if not len(indices) or len(indices) != len(values):
                return CommandResult.bad("Joint batch indices and values must have equal lengths")
            if len(np.unique(indices)) != len(indices):
                return CommandResult.bad("Joint batch indices must be unique")
            if not np.all(np.isfinite(values)):
                return CommandResult.bad("Joint batch values must be finite")
            ok = self._adapter.set_qpos_batch(indices, values)
            return CommandResult.good("") if ok else CommandResult.bad("Joint batch update failed")

        if isinstance(c, cmd.SetJointProperties):
            if not caps.model_properties:
                return CommandResult.bad(f"{caps.name} does not support model property editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing joint properties")
            joint = next((item for item in self._joints if item.joint_id == c.joint_id), None)
            if joint is None:
                return CommandResult.bad(f"Joint {c.joint_id} is unavailable")
            if joint.type == "free":
                return CommandResult.bad("Free-joint properties are not editable here")
            axis = np.asarray(c.axis, np.float64).reshape(3)
            value_range = np.asarray(c.range, np.float64).reshape(2)
            damping = float(c.damping)
            stiffness = float(c.stiffness)
            if (
                not np.all(np.isfinite(axis))
                or not np.all(np.isfinite(value_range))
                or not np.isfinite((damping, stiffness)).all()
            ):
                return CommandResult.bad("Joint properties must contain finite values")
            if joint.type in ("hinge", "slide") and np.linalg.norm(axis) <= 1e-12:
                return CommandResult.bad("Joint axis must be non-zero")
            if joint.type == "ball":
                if bool(c.limited) and value_range[1] <= 0.0:
                    return CommandResult.bad("Ball-joint limit must be positive")
                value_range[0] = 0.0
            elif bool(c.limited) and value_range[1] <= value_range[0]:
                return CommandResult.bad("Joint range upper bound must exceed its lower bound")
            if damping < 0.0 or stiffness < 0.0:
                return CommandResult.bad("Joint damping and stiffness cannot be negative")
            changed = self._adapter.set_joint_properties(
                c.joint_id,
                axis,
                bool(c.limited),
                (float(value_range[0]), float(value_range[1])),
                damping,
                stiffness,
            )
            if not changed:
                return CommandResult.bad(f"Joint {joint.name} properties cannot be edited")
            self._refresh_joint_metadata()
            self._adapter_revision = self._adapter.structure_revision
            self._structure_generation += 1
            return CommandResult.good("")

        if isinstance(c, cmd.SetGeometryProperties):
            if not caps.model_properties:
                return CommandResult.bad(f"{caps.name} does not support model property editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing geometry properties")
            current = self._adapter.geometry_properties(c.node_id)
            if current is None:
                return CommandResult.bad(f"Geometry node {c.node_id} is unavailable")
            try:
                friction = np.asarray(c.friction, np.float64).reshape(3)
                collision_type_mask = int(c.collision_type_mask)
                collision_affinity_mask = int(c.collision_affinity_mask)
                contact_dimension = int(c.contact_dimension)
                contact_priority = int(c.contact_priority)
                margin = float(c.margin)
                gap = float(c.gap)
                solver_mix = float(c.solver_mix)
                solver_reference = np.asarray(
                    current.solver_reference if c.solver_reference is None else c.solver_reference,
                    np.float64,
                ).reshape(2)
                solver_impedance = np.asarray(
                    current.solver_impedance if c.solver_impedance is None else c.solver_impedance,
                    np.float64,
                ).reshape(5)
                adhesion = float(current.adhesion if c.adhesion is None else c.adhesion)
                surface_velocity = np.asarray(
                    current.surface_velocity if c.surface_velocity is None else c.surface_velocity,
                    np.float64,
                ).reshape(6)
            except (TypeError, ValueError, OverflowError):
                return CommandResult.bad("Geometry contact properties have invalid value types")
            finite_values = np.concatenate(
                (
                    friction,
                    np.array((margin, gap, solver_mix, adhesion)),
                    solver_reference,
                    solver_impedance,
                    surface_velocity,
                )
            )
            if not np.all(np.isfinite(finite_values)):
                return CommandResult.bad("Geometry properties must contain finite values")
            if np.any(friction < 0.0):
                return CommandResult.bad("Geometry friction cannot be negative")
            if collision_type_mask < 0 or collision_affinity_mask < 0:
                return CommandResult.bad("Collision masks cannot be negative")
            if max(collision_type_mask, collision_affinity_mask) > np.iinfo(np.int32).max:
                return CommandResult.bad("Collision masks exceed MuJoCo's 31-bit positive range")
            if contact_dimension not in (1, 3, 4, 6):
                return CommandResult.bad("Contact dimension must be 1, 3, 4, or 6")
            if not 0 <= contact_priority <= np.iinfo(np.int32).max:
                return CommandResult.bad("Contact priority exceeds MuJoCo's positive integer range")
            if margin < 0.0 or gap < 0.0:
                return CommandResult.bad("Contact margin and gap cannot be negative")
            if not 0.0 <= solver_mix <= 1.0:
                return CommandResult.bad("Solver mix must be between 0 and 1")
            standard_reference = np.all(solver_reference > 0.0)
            direct_reference = np.all(solver_reference <= 0.0)
            if not standard_reference and not direct_reference:
                return CommandResult.bad(
                    "Solver reference values must both use standard or direct format"
                )
            if (
                not 0.0 <= solver_impedance[0] <= solver_impedance[1] <= 1.0
                or solver_impedance[2] <= 0.0
                or not 0.0 <= solver_impedance[3] <= 1.0
                or solver_impedance[4] < 1.0
            ):
                return CommandResult.bad("Solver impedance values are outside MuJoCo limits")
            if adhesion < 0.0:
                return CommandResult.bad("Geometry adhesion cannot be negative")
            properties = GeometryProperties(
                node_id=int(c.node_id),
                friction=tuple(float(value) for value in friction),
                collision_type_mask=collision_type_mask,
                collision_affinity_mask=collision_affinity_mask,
                contact_dimension=contact_dimension,
                contact_priority=contact_priority,
                margin=margin,
                gap=gap,
                solver_mix=solver_mix,
                solver_reference=tuple(float(value) for value in solver_reference),
                solver_impedance=tuple(float(value) for value in solver_impedance),
                adhesion=adhesion,
                surface_velocity=tuple(float(value) for value in surface_velocity),
            )
            if not self._adapter.set_geometry_properties(properties):
                return CommandResult.bad("Geometry contact properties could not be edited")
            self._adapter_revision = self._adapter.structure_revision
            self._structure_generation += 1
            return CommandResult.good("")

        if isinstance(c, cmd.SetGeometryAdvancedProperties):
            if not caps.model_properties:
                return CommandResult.bad(f"{caps.name} does not support model property editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing geometry properties")
            if self._adapter.geometry_advanced_properties(c.node_id) is None:
                return CommandResult.bad(f"Geometry node {c.node_id} is unavailable")
            mass_mode = str(c.mass_mode).strip().lower()
            inertia_mode = str(c.inertia_mode).strip().lower()
            if mass_mode not in ("density", "mass"):
                return CommandResult.bad("Geometry mass mode must be density or mass")
            if inertia_mode not in ("volume", "shell"):
                return CommandResult.bad("Geometry inertia mode must be volume or shell")
            try:
                visual_group = int(c.visual_group)
                mass = float(c.mass)
                density = float(c.density)
                fluid_coefficients = np.asarray(c.fluid_coefficients, np.float64).reshape(5)
            except (TypeError, ValueError, OverflowError):
                return CommandResult.bad("Advanced geometry properties have invalid value types")
            if not np.isfinite((mass, density, *fluid_coefficients)).all():
                return CommandResult.bad("Advanced geometry properties must be finite")
            if not 0 <= visual_group < 6:
                return CommandResult.bad("Geometry visual group must be between 0 and 5")
            if mass_mode == "mass" and mass <= 0.0:
                return CommandResult.bad("Geometry mass must be positive")
            if mass_mode == "density" and density <= 0.0:
                return CommandResult.bad("Geometry density must be positive")
            if np.any(fluid_coefficients < 0.0):
                return CommandResult.bad("Geometry fluid coefficients cannot be negative")
            properties = GeometryAdvancedProperties(
                node_id=int(c.node_id),
                visual_group=visual_group,
                mass_mode=mass_mode,
                mass=mass,
                density=density,
                inertia_mode=inertia_mode,
                fluid_ellipsoid=bool(c.fluid_ellipsoid),
                fluid_coefficients=tuple(float(value) for value in fluid_coefficients),
            )
            try:
                changed = self._adapter.set_geometry_advanced_properties(properties)
            except Exception as exc:
                return CommandResult.bad(f"Geometry properties could not be applied: {exc}")
            if not changed:
                return CommandResult.bad("Advanced geometry properties could not be edited")
            self._refresh_structure()
            return CommandResult.good("")

        if isinstance(c, cmd.SetBodyProperties):
            if not caps.model_properties:
                return CommandResult.bad(f"{caps.name} does not support model property editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before editing body properties")
            if self._adapter.body_properties(c.node_id) is None:
                return CommandResult.bad(f"Body node {c.node_id} is unavailable")
            inertia_mode = str(c.inertia_mode).strip().lower()
            sleep_policy = str(c.sleep_policy).strip().lower()
            if inertia_mode not in ("auto", "diagonal", "full"):
                return CommandResult.bad("Body inertia mode must be auto, diagonal, or full")
            if sleep_policy not in ("auto", "never", "allowed", "init"):
                return CommandResult.bad(
                    "Body sleep policy must be auto, never, allowed, or initially asleep"
                )
            try:
                mass = float(c.mass)
                inertial_position = np.asarray(c.inertial_position, np.float64).reshape(3)
                inertial_quaternion = np.asarray(c.inertial_quaternion, np.float64).reshape(4)
                diagonal_inertia = np.asarray(c.diagonal_inertia, np.float64).reshape(3)
                full_inertia = np.asarray(c.full_inertia, np.float64).reshape(6)
                gravity_compensation = float(c.gravity_compensation)
            except (TypeError, ValueError, OverflowError):
                return CommandResult.bad("Body properties have invalid value types")
            values = np.concatenate(
                (
                    np.array((mass, gravity_compensation), np.float64),
                    inertial_position,
                    inertial_quaternion,
                    diagonal_inertia,
                    full_inertia,
                )
            )
            if not np.all(np.isfinite(values)):
                return CommandResult.bad("Body properties must contain finite values")
            quaternion_norm = float(np.linalg.norm(inertial_quaternion))
            if quaternion_norm <= 1e-12:
                return CommandResult.bad("Body inertial rotation must be non-zero")
            inertial_quaternion /= quaternion_norm
            if inertia_mode != "auto" and mass <= 0.0:
                return CommandResult.bad("Explicit body mass must be positive")
            if inertia_mode == "diagonal":
                if np.any(diagonal_inertia <= 0.0):
                    return CommandResult.bad("Diagonal inertia values must be positive")
                if 2.0 * float(np.max(diagonal_inertia)) > float(np.sum(diagonal_inertia)):
                    return CommandResult.bad("Diagonal inertia violates the triangle inequality")
            if inertia_mode == "full":
                ixx, iyy, izz, ixy, ixz, iyz = full_inertia
                tensor = np.array(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)), np.float64)
                principal = np.linalg.eigvalsh(tensor)
                if np.any(principal <= 0.0):
                    return CommandResult.bad("Full inertia tensor must be positive definite")
                if 2.0 * float(np.max(principal)) > float(np.sum(principal)):
                    return CommandResult.bad("Full inertia tensor violates the triangle inequality")
            properties = BodyProperties(
                node_id=int(c.node_id),
                inertia_mode=inertia_mode,
                mass=mass,
                inertial_position=tuple(float(value) for value in inertial_position),
                inertial_quaternion=tuple(float(value) for value in inertial_quaternion),
                diagonal_inertia=tuple(float(value) for value in diagonal_inertia),
                full_inertia=tuple(float(value) for value in full_inertia),
                gravity_compensation=gravity_compensation,
                mocap=bool(c.mocap),
                sleep_policy=sleep_policy,
            )
            try:
                changed = self._adapter.set_body_properties(properties)
            except Exception as exc:
                return CommandResult.bad(f"Body properties could not be applied: {exc}")
            if not changed:
                return CommandResult.bad("Body properties could not be edited")
            self._refresh_structure()
            return CommandResult.good("")

        if isinstance(c, cmd.AddModelMaterial):
            if not caps.model_assets:
                return CommandResult.bad(f"{caps.name} does not support model asset editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before creating materials")
            value = str(c.name).strip()
            if not value:
                return CommandResult.bad("A material name cannot be empty")
            material_index = self._adapter.add_model_material(c.node_id, value, int(c.copy_from))
            if material_index < 0:
                return CommandResult.bad(f"Material {value!r} could not be created")
            self._refresh_structure()
            return CommandResult.good(f"Created material {value}", material_index)

        if isinstance(c, cmd.ImportModelTexture):
            if not caps.model_assets:
                return CommandResult.bad(f"{caps.name} does not support model asset editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before importing textures")
            path = Path(c.path).expanduser().resolve()
            value = str(c.name).strip()
            if not path.is_file():
                return CommandResult.bad(f"Texture file does not exist: {path}")
            if path.suffix.lower() != ".png":
                return CommandResult.bad("MuJoCo 2D image textures must use PNG files")
            if not value:
                return CommandResult.bad("A texture name cannot be empty")
            try:
                changed = self._adapter.import_model_texture(
                    c.model_id, path, value, int(c.material_index)
                )
            except (RuntimeError, ValueError) as exc:
                return CommandResult.bad(f"Texture {value!r} could not be imported: {exc}")
            if not changed:
                return CommandResult.bad(f"Texture {value!r} could not be imported")
            self._refresh_structure()
            return CommandResult.good(f"Imported texture {value}")

        if isinstance(c, cmd.SetGeometryMaterial):
            if not caps.model_assets:
                return CommandResult.bad(f"{caps.name} does not support model asset editing")
            if caps.simulation and not self._paused:
                return CommandResult.bad("Pause the simulation before binding materials")
            if not self._adapter.set_geometry_material(c.node_id, c.material_index):
                return CommandResult.bad("The material is unavailable for this model geometry")
            self._refresh_structure()
            return CommandResult.good("")

        if isinstance(c, cmd.SetEqualityEnabled):
            if not caps.equality_constraints:
                return CommandResult.bad(f"{caps.name} does not expose equality constraints")
            i = int(c.constraint_id)
            slot = self._equality_slot(i)
            if slot < 0:
                return CommandResult.bad(f"equality constraint {i} is unavailable")
            if not self._adapter.set_equality_enabled(i, c.enabled):
                return CommandResult.bad(f"equality constraint {i} update failed")
            self._equality_constraints[slot] = replace(
                self._equality_constraints[slot], enabled=bool(c.enabled)
            )
            return CommandResult.good("")

        if isinstance(c, cmd.SetCtrl):
            ok = self._adapter.set_ctrl(c.index, c.value)
            return (
                CommandResult.good("")
                if ok
                else CommandResult.bad(f"Actuator {c.index} update failed")
            )

        if isinstance(c, cmd.Perturb):
            if not caps.perturb:
                return CommandResult.bad(f"{caps.name} does not support perturbation")
            ok = self._adapter.apply_perturb(
                c.node_id, c.target_position, c.target_rotation, c.mode
            )
            return CommandResult.good("") if ok else CommandResult.bad("Perturbation failed")

        if isinstance(c, cmd.ClearPerturb):
            self._adapter.clear_perturb()
            self._perturb = PerturbState()
            return CommandResult.good("")

        if isinstance(c, cmd.SetLight):
            index = int(c.light_index)
            if self._source is None or not 0 <= index < len(self._source.lights.lights):
                return CommandResult.bad(f"light index {index} is unavailable")
            writeback = self._adapter.set_light(index, c.light)
            node = next((node for node in self._nodes if node.light_index == index), None)
            object_id = int(node.object_id) if node is not None else 0
            override_key = object_id if object_id > 0 else -(index + 1)
            if self._preserve_authored_override(writeback):
                self._authored.lights[override_key] = _LightOverride(object_id, index, c.light)
            else:
                self._authored.lights.pop(override_key, None)
            lights = list(self._source.lights.lights)
            lights[index] = c.light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
            for node in self._nodes:
                if node.light_index == index:
                    node.visible = c.light.active
                    break
            self._compose_lights()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetEnvironment):
            if self._source is None:
                return CommandResult.bad("environment is unavailable")
            writeback = self._adapter.set_environment(c.environment)
            self._authored.environment = (
                c.environment if self._preserve_authored_override(writeback) else None
            )
            self._source.lights = self._source.lights.with_environment(c.environment)
            self._compose_lights()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetSkybox):
            if self._source is None:
                return CommandResult.bad("skybox is unavailable")
            texture = self._source.textures.get(c.texture or "")
            if c.texture is not None and (
                texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX)
            ):
                return CommandResult.bad(f"cube texture {c.texture!r} is unavailable")
            if not self._adapter.set_skybox(c.texture):
                return CommandResult.bad("skybox update failed")
            self._source.skybox = c.texture
            self._structure_generation += 1
            return CommandResult.good("")

        if isinstance(c, cmd.SetMaterial):
            index = int(c.material_index)
            if self._source is None or not 0 <= index < len(self._source.materials):
                return CommandResult.bad(f"material index {index} is unavailable")
            writeback = self._adapter.set_material(index, c.material)
            if self._preserve_authored_override(writeback):
                self._authored.materials[index] = c.material
            else:
                self._authored.materials.pop(index, None)
            self._source.materials[index] = c.material
            self._structure_generation += 1
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetGeometryColor):
            if self._source is None:
                return CommandResult.bad("geometry is unavailable")
            instances = np.flatnonzero(self._source.geom_node == int(c.node_id))
            if not len(instances):
                return CommandResult.bad(f"geometry node {c.node_id} is unavailable")
            rgba = np.asarray(c.rgba, np.float32).reshape(4).copy()
            writeback = self._adapter.set_geometry_color(c.node_id, rgba)
            if self._preserve_authored_override(writeback):
                self._authored.geometry_colors[c.node_id] = rgba
            else:
                self._authored.geometry_colors.pop(c.node_id, None)
            self._source.geom_rgba[instances] = rgba
            self._structure_generation += 1
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.SetGeometrySize):
            if self._source is None:
                return CommandResult.bad("geometry is unavailable")
            instances = np.flatnonzero(self._source.geom_node == int(c.node_id))
            if not len(instances):
                return CommandResult.bad(f"geometry node {c.node_id} is unavailable")
            size = np.asarray(c.size, np.float32).reshape(3)
            if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
                return CommandResult.bad("geometry size must contain three positive values")
            if not self._adapter.set_geometry_size(c.node_id, size):
                return CommandResult.bad("geometry size cannot be edited")
            self._refresh_structure()
            return CommandResult.good("")

        if isinstance(c, cmd.SetSceneCamera):
            camera_id = int(c.camera_id)
            slot = self._camera_slot(camera_id)
            if self._source is None or slot < 0 or slot >= len(self._source.cameras):
                return CommandResult.bad(f"camera {camera_id} is unavailable")
            writeback = self._adapter.set_camera_view(camera_id, c.camera)
            cameras = list(self._source.cameras)
            cameras[slot] = c.camera
            self._source.cameras = tuple(cameras)
            if self._preserve_authored_override(writeback):
                self._authored.cameras[camera_id] = c.camera
            else:
                self._authored.cameras.pop(camera_id, None)
            self._compose_cameras()
            message = "" if writeback else "edited in Forge; backend write-back is unavailable"
            return CommandResult.good(message)

        if isinstance(c, cmd.AddSceneObject):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            object_id = self._adapter.add_scene_object(
                c.shape, c.name, c.size, c.position, c.rotation, c.color, c.material
            )
            if object_id < 0:
                return CommandResult.bad("Object creation failed")
            self._refresh_structure()
            return CommandResult.good(f"Added {c.name}", object_id)

        if isinstance(c, cmd.RemoveSceneObject):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            if not self._adapter.remove_scene_object(c.object_id):
                return CommandResult.bad(f"object {c.object_id} is unavailable")
            if self._selected == c.object_id:
                self._selected = 0
                self._selected_node_id = -1
            self._refresh_structure()
            return CommandResult.good("Object removed", c.object_id)

        if isinstance(c, cmd.AddSceneLight):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            light_id = self._adapter.add_scene_light(c.name, c.light)
            if light_id < 0:
                return CommandResult.bad("Light creation failed")
            self._refresh_structure()
            return CommandResult.good(f"Added {c.name}", light_id)

        if isinstance(c, cmd.RemoveSceneLight):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            if not self._adapter.remove_scene_light(c.light_id):
                return CommandResult.bad(f"light {c.light_id} is unavailable")
            self._refresh_structure()
            return CommandResult.good("Light removed", c.light_id)

        if isinstance(c, cmd.AddSceneCamera):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            camera_id = self._adapter.add_scene_camera(c.name, c.camera)
            if camera_id < 0:
                return CommandResult.bad("Camera creation failed")
            self._refresh_structure()
            return CommandResult.good(f"Added {c.name}", camera_id)

        if isinstance(c, cmd.RemoveSceneCamera):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            if not self._adapter.remove_scene_camera(c.camera_id):
                return CommandResult.bad(f"camera {c.camera_id} is unavailable")
            self._authored.cameras.pop(c.camera_id, None)
            self._refresh_structure()
            return CommandResult.good("Camera removed", c.camera_id)

        if isinstance(c, cmd.DuplicateSceneEntity):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            object_id = self._adapter.duplicate_scene_entity(c.object_id)
            if not object_id:
                return CommandResult.bad(f"entity {c.object_id} cannot be duplicated")
            self._selected = object_id
            self._selected_node_id = -1
            self._refresh_structure()
            return CommandResult.good("Entity duplicated", object_id)

        if isinstance(c, cmd.RemoveSceneEntity):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            if not self._adapter.remove_scene_entity(c.object_id):
                return CommandResult.bad(f"entity {c.object_id} cannot be removed")
            if self._selected == c.object_id:
                self._selected = 0
                self._selected_node_id = -1
            self._refresh_structure()
            return CommandResult.good("Entity removed", c.object_id)

        if isinstance(c, cmd.RenameSceneEntity):
            if not caps.scene_authoring:
                return CommandResult.bad(f"{caps.name} does not support scene authoring")
            if not self._adapter.rename_scene_entity(c.object_id, c.name):
                return CommandResult.bad(f"entity {c.object_id} cannot be renamed")
            self._refresh_structure()
            return CommandResult.good("Entity renamed", c.object_id)

        if isinstance(c, cmd.SetSpeed):
            if not caps.simulation:
                return CommandResult.bad(f"{caps.name} has no simulation speed")
            factor = float(c.factor)
            if not np.isfinite(factor) or factor <= 0.0:
                return CommandResult.bad("simulation speed must be finite and positive")
            self._speed = max(0.05, factor)
            return CommandResult.good(f"Speed ×{self._speed:g}")

        if isinstance(c, cmd.SetCamera):
            self._camera = c.camera
            return CommandResult.good("")

        return CommandResult.bad(f"Unknown command: {type(c).__name__}")

    def query(self, q: Query):
        """Evaluate a read-only pick, node lookup, or bounds query."""
        if isinstance(q, cmd.Pick):
            if not self._adapter.caps.raycast:
                return (0, float("inf"))
            return self._adapter.raycast(q.origin, q.direction)
        if isinstance(q, cmd.NodeAt):
            return self._by_object_id.get(int(q.object_id))
        if isinstance(q, cmd.Bounds):
            return self.bounds()
        raise TypeError(f"Unknown query: {type(q).__name__}")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return an axis-aligned world-space bound for finite scene geometry."""
        src = self._source
        frame = self._frame
        if src is None:
            return np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)

        lo: np.ndarray | None = None
        hi: np.ndarray | None = None
        if frame.geom_xpos is not None and len(frame.geom_xpos):
            pos = np.asarray(frame.geom_xpos, np.float32)
            size = src.geom_size[: len(pos)] if len(src.geom_size) >= len(pos) else None
            finite = np.isfinite(pos).all(axis=1)

            if len(src.geom_infinite_plane) == len(pos):
                finite &= ~src.geom_infinite_plane
            if finite.any():
                p = pos[finite]
                r = np.max(size[finite], axis=1, keepdims=True) if size is not None else 0.0
                lo, hi = (p - r).min(axis=0), (p + r).max(axis=0)

        for key in src.dynamic_meshes:
            points = np.asarray(src.meshes[key].positions, np.float32)
            points = points[np.isfinite(points).all(axis=1)]
            if not len(points):
                continue
            mesh_lo, mesh_hi = points.min(axis=0), points.max(axis=0)
            lo = mesh_lo if lo is None else np.minimum(lo, mesh_lo)
            hi = mesh_hi if hi is None else np.maximum(hi, mesh_hi)

        if lo is None or hi is None:
            return np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)
        return lo.astype(np.float32), hi.astype(np.float32)

    def camera_hint(self) -> CameraView | None:
        """Return the adapter camera suggested for initial framing."""
        return self._adapter.camera_hint()

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Return an authored override or adapter camera by stable ID."""
        i = int(camera_id)
        if i in self._authored.cameras:
            return self._authored.cameras[i]
        return self._adapter.camera_view(i) if self._adapter.caps.model_cameras else None

    def _preserve_authored_override(self, writeback: bool) -> bool:
        caps = self._adapter.caps
        return not writeback or caps.external_clock or caps.model_composition

    def visual_groups(self):
        """Return numbered visual group states exposed by the adapter."""
        return self._adapter.visual_groups() if self._adapter.caps.visual_groups else ()

    def _refresh_structure(self) -> None:
        self._source = self._adapter.scene_source()
        if self._authored.environment is not None:
            self._source.lights = self._source.lights.with_environment(self._authored.environment)
        for material_index, material in self._authored.materials.items():
            if material_index < len(self._source.materials):
                self._source.materials[material_index] = material
        _apply_geometry_color_overrides(self._source, self._authored.geometry_colors)
        self._nodes = [
            replace(node, children=list(node.children)) for node in self._adapter.nodes()
        ]
        if self._authored.lights:
            light_nodes = {
                node.object_id: node
                for node in self._nodes
                if node.object_id > 0 and node.light_index >= 0
            }
            lights = list(self._source.lights.lights)
            for override in self._authored.lights.values():
                node = light_nodes.get(override.object_id) if override.object_id > 0 else None
                index = node.light_index if node is not None else override.light_index
                if 0 <= index < len(lights):
                    lights[index] = override.light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
        if not any(node.type is NodeType.ENVIRONMENT for node in self._nodes):
            parent = next(
                (node for node in self._nodes if node.type is NodeType.WORLD and node.parent < 0),
                None,
            )
            node_id = max((node.node_id for node in self._nodes), default=-1) + 1
            environment = SceneNode(
                node_id,
                "environment",
                NodeType.ENVIRONMENT,
                parent=parent.node_id if parent is not None else -1,
                object_id=ENVIRONMENT_OBJECT_ID,
            )
            self._nodes.append(environment)
            if parent is not None:
                parent.children.append(node_id)
        for node in self._nodes:
            if 0 <= node.light_index < len(self._source.lights.lights):
                node.visible = self._source.lights.lights[node.light_index].active
        self._refresh_joint_metadata()

        self._actuators = self._adapter.actuators()
        actuators_by_joint: dict[int, list[ActuatorInfo]] = {}
        for actuator in self._actuators:
            actuators_by_joint.setdefault(int(actuator.joint), []).append(actuator)
        self._actuators_by_joint = {
            joint: tuple(actuators) for joint, actuators in actuators_by_joint.items()
        }
        self._cameras = self._adapter.cameras() if self._adapter.caps.model_cameras else []
        self._camera_slot_by_id = {
            camera.camera_id: slot for slot, camera in enumerate(self._cameras)
        }
        if self._authored.cameras:
            cameras = list(self._source.cameras)
            for camera_id, camera in self._authored.cameras.items():
                slot = self._camera_slot(camera_id)
                if 0 <= slot < len(cameras):
                    cameras[slot] = camera
            self._source.cameras = tuple(cameras)
        self._keyframes = self._adapter.keyframes() if self._adapter.caps.keyframes else []
        self._sensor_infos = self._adapter.sensors() if self._adapter.caps.sensors else []
        self._equality_constraints = (
            self._adapter.equality_constraints() if self._adapter.caps.equality_constraints else []
        )
        if self._active_keyframe != -1 and self._keyframe_slot(self._active_keyframe) < 0:
            self._active_keyframe = -1
        self._by_node_id = {n.node_id: n for n in self._nodes}
        self._by_object_id = {n.object_id: n for n in self._nodes if n.object_id}
        self._unlocked_entity_gizmos.intersection_update(self._by_object_id)
        if self._selected:
            selected = self._by_object_id.get(self._selected)
            if selected is None:
                self._selected = 0
                self._selected_node_id = -1
            else:
                self._selected_node_id = selected.node_id
        elif (selected := self.node(self._selected_node_id)) is None or selected.object_id:
            self._selected_node_id = -1
        self._adapter_revision = self._adapter.structure_revision
        self._structure_generation += 1
        self._frame = self._adapter.frame(FrameNeeds())
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()

    def _refresh_joint_metadata(self) -> None:
        """Refresh joint lookup tables without rebuilding stable scene geometry."""
        self._joints = self._adapter.joints()
        joints_by_body: dict[int, list[JointInfo]] = {}
        for joint in self._joints:
            joints_by_body.setdefault(int(joint.body), []).append(joint)
        self._joints_by_body = {body: tuple(joints) for body, joints in joints_by_body.items()}

    def _compose_lights(self) -> None:
        """Combine Forge-authored light settings with backend-driven transforms.

        A physics backend may move a body-attached light and publish its world
        position/direction in ``SceneFrame``.  Color, intensity, range, fog and
        every other render setting still come from the Forge scene.
        """
        if self._source is None:
            return
        authored = self._source.lights
        driven = self._frame.lights
        if driven is None or len(driven.lights) != len(authored.lights):
            self._frame.lights = authored
            return
        lights = tuple(
            replace(light, position=dynamic.position, direction=dynamic.direction)
            for light, dynamic in zip(authored.lights, driven.lights, strict=True)
        )
        self._frame.lights = replace(authored, lights=lights)

    def _sync_equality_state(self) -> None:
        values = self._frame.equality_enabled
        if values is None or len(values) != len(self._equality_constraints):
            return
        for i, enabled in enumerate(values):
            if self._equality_constraints[i].enabled != bool(enabled):
                self._equality_constraints[i] = replace(
                    self._equality_constraints[i], enabled=bool(enabled)
                )

    def _compose_cameras(self) -> None:
        if self._source is None:
            return
        driven = self._frame.cameras
        if not self._authored.cameras:
            if driven is None or len(driven) != len(self._source.cameras):
                self._frame.cameras = self._source.cameras
            return
        cameras = list(
            driven
            if driven is not None and len(driven) == len(self._source.cameras)
            else self._source.cameras
        )
        for camera_id, camera in self._authored.cameras.items():
            slot = self._camera_slot(camera_id)
            if 0 <= slot < len(cameras):
                cameras[slot] = camera
        self._frame.cameras = tuple(cameras)

    def _camera_slot(self, camera_id: int) -> int:
        return self._camera_slot_by_id.get(int(camera_id), -1)

    def _keyframe_slot(self, keyframe_id: int) -> int:
        return next(
            (
                slot
                for slot, keyframe in enumerate(self._keyframes)
                if keyframe.keyframe_id == keyframe_id
            ),
            -1,
        )

    def _equality_slot(self, constraint_id: int) -> int:
        return next(
            (
                slot
                for slot, constraint in enumerate(self._equality_constraints)
                if constraint.constraint_id == constraint_id
            ),
            -1,
        )

    def release(self) -> None:
        """Release resources owned by the scene adapter."""
        self._adapter.release()
