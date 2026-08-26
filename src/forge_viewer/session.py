"""Application state, selection, overrides, and command routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import commands as cmd
from .adapters.base import (
    ENVIRONMENT_OBJECT_ID,
    ActuatorInfo,
    CameraInfo,
    EqualityConstraintInfo,
    FrameNeeds,
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
from .types import CameraView, Environment, Light, Material


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


@dataclass
class AuthoredSceneOverlay:
    """Forge-owned property edits layered over an adapter scene."""

    lights: dict[int, Light] = field(default_factory=dict)
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
    cmd.SetModelSource,
    cmd.AddModelComponent,
    cmd.UpdateModelComponent,
    cmd.RemoveModelComponent,
    cmd.AddResourceRoot,
    cmd.RemoveResourceRoot,
    cmd.SetPose,
    cmd.SetLight,
    cmd.SetEnvironment,
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
                return CommandResult.bad("this link is driven by joints; use the Joints panel")
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
            if self._source is None or not 0 <= c.light_id < len(self._source.lights.lights):
                return CommandResult.bad(f"light {c.light_id} is unavailable")
            writeback = self._adapter.set_light(c.light_id, c.light)
            if self._preserve_authored_override(writeback):
                self._authored.lights[c.light_id] = c.light
            else:
                self._authored.lights.pop(c.light_id, None)
            lights = list(self._source.lights.lights)
            lights[c.light_id] = c.light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
            for node in self._nodes:
                if node.light_index == c.light_id:
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

        if isinstance(c, cmd.SetMaterial):
            if self._source is None or not 0 <= c.material_id < len(self._source.materials):
                return CommandResult.bad(f"material {c.material_id} is unavailable")
            writeback = self._adapter.set_material(c.material_id, c.material)
            if self._preserve_authored_override(writeback):
                self._authored.materials[c.material_id] = c.material
            else:
                self._authored.materials.pop(c.material_id, None)
            self._source.materials[c.material_id] = c.material
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
        if self._authored.lights:
            lights = list(self._source.lights.lights)
            for i, light in self._authored.lights.items():
                if i < len(lights):
                    lights[i] = light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
        for material_id, material in self._authored.materials.items():
            if material_id < len(self._source.materials):
                self._source.materials[material_id] = material
        _apply_geometry_color_overrides(self._source, self._authored.geometry_colors)
        self._nodes = [
            replace(node, children=list(node.children)) for node in self._adapter.nodes()
        ]
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
        self._joints = self._adapter.joints()
        joints_by_body: dict[int, list[JointInfo]] = {}
        for joint in self._joints:
            joints_by_body.setdefault(int(joint.body), []).append(joint)
        self._joints_by_body = {body: tuple(joints) for body, joints in joints_by_body.items()}
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
