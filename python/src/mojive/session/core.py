"""Session: core."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mojive.adapters.base import (
    ActuatorInfo,
    BodyProperties,
    CameraInfo,
    EqualityConstraintInfo,
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    KeyframeInfo,
    KeyframeProperties,
    ModelAssetInfo,
    NodeType,
    SceneAdapter,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSource,
    SensorInfo,
    SiteProperties,
)
from mojive.commands import Command, CommandResult
from mojive.scene.bounds import (
    SceneBounds,
    _MeshBoundsCache,
    _node_local_bounds,
    _node_world_bounds,
)
from mojive.session.history import EditHistory
from mojive.session.rates import StepRate
from mojive.session.simulation import SimulationDriver
from mojive.types import (
    CameraView,
    CenteredBounds,
)

from .dispatch import dispatch
from .editing import _Editing
from .playback import _Playback
from .source import _Source
from .state import (
    AuthoredSceneOverlay,
    PerturbState,
    SceneSnapshotInfo,
    _DocumentState,
    _SceneSnapshot,
    _StateTakeFrame,
)


class Session(_Editing, _Playback, _Source):
    """Own viewer state and route typed commands to one scene adapter.

    The session separates UI and renderer code from physics-specific methods. It
    tracks selection, pause and step state, authored overrides, edit history, and
    stable-structure generations while the adapter owns simulation data.
    """

    def __init__(
        self,
        adapter: SceneAdapter,
        asset_path: Path | None = None,
        *,
        history_record_limit: int = 100,
        history_byte_limit: int = 256 * 1024 * 1024,
    ) -> None:
        self._adapter = adapter
        self._simulation_driver: SimulationDriver | None = None
        self._asset_path = asset_path
        self._paused = not adapter.caps.simulation
        self._speed = 1.0
        self._sim_time_credit = 0.0
        self._selected = 0
        self._selected_node_id = -1
        self._selection_revision = 0
        self._unlocked_entity_gizmos: set[int] = set()
        self._step_counter = 0
        self._pending_steps = 0
        self._frame = SceneFrame()
        self._physics_rate = StepRate()
        self._source: SceneSource | None = None
        self._model_edit_preview = None
        self._preview_generation = 0
        self._mesh_bounds_cache: _MeshBoundsCache = {}
        self._scene_bounds: SceneBounds | None = None
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
        self._state_take: list[_StateTakeFrame] = []
        self._scene_snapshots: dict[int, _SceneSnapshot] = {}
        self._scene_snapshot_info: tuple[SceneSnapshotInfo, ...] = ()
        self._scene_snapshot_serial = 0
        self._scene_snapshot_bytes = 0
        self._state_take_times: list[float] = []
        self._state_take_offsets: list[float] = []
        self._state_take_loop: tuple[int, int] | None = None
        self._state_take_cursor = -1
        self._state_take_recording = False
        self._state_take_playing = False
        self._state_take_use_loop = True
        self._state_take_elapsed = 0.0
        self._state_take_signature: tuple[tuple[int, ...], ...] | None = None
        self._state_take_bytes = 0
        self._state_take_append_error = ""
        self._state_take_limit_reached = False
        self._frame_history: list[_StateTakeFrame] = []
        self._frame_history_cursor = -1
        self._frame_history_bytes = 0
        self._frame_history_signature: tuple[tuple[int, ...], ...] | None = None
        self._frame_history_dirty = False
        self._perturb = PerturbState()
        self._camera = CameraView()
        self._last_message = ""
        self._message_revision = 0
        self._last_message_level = "info"
        self._last_message_duration: float | None = 5.0
        self._last_message_copy_text: str | None = None
        self._history: EditHistory[_DocumentState] = EditHistory(
            record_limit=history_record_limit, byte_limit=history_byte_limit
        )
        self._undo_stack = self._history.undo
        self._redo_stack = self._history.redo
        self._edit_before: _DocumentState | None = None
        self._edit_before_revision = 0
        self._edit_label = ""
        self._edit_changed = False
        self._edit_error = ""
        self._document_revision = 0
        self._document_id = uuid4().hex
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

    @property
    def selection_revision(self) -> int:
        """Advance for every accepted selection command, including reselecting a node."""
        return self._selection_revision

    @property
    def selection_highlight_object_id(self) -> int:
        """Return the render object representing the logical hierarchy selection."""

        if self._selected > 0:
            return self._selected
        node = self.selected_node
        if node is None or node.type is not NodeType.JOINT:
            return 0
        parent_id = int(node.parent)
        while parent_id >= 0:
            parent = self.node(parent_id)
            if parent is None:
                break
            if parent.object_id > 0 and parent.body_index == node.body_index:
                return int(parent.object_id)
            parent_id = int(parent.parent)
        return 0

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
        preview = self._model_edit_preview
        if preview is not None and preview.visible and preview.geometry is not None:
            return preview.geometry.frame(self._frame)
        return self._frame

    @property
    def source(self) -> SceneSource | None:
        """Return the current stable scene source."""
        preview = getattr(self, "_model_edit_preview", None)
        return preview.source if preview is not None and preview.visible else self._source

    @property
    def nodes(self) -> list[SceneNode]:
        """Return hierarchy nodes for the current structure generation."""
        preview = getattr(self, "_model_edit_preview", None)
        return preview.nodes if preview is not None and preview.visible else self._nodes

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
    def state_take_recording(self) -> bool:
        """Return whether the editor is recording transient simulation samples."""
        return self._state_take_recording

    @property
    def state_take_playing(self) -> bool:
        """Return whether the editor is replaying its transient simulation take."""
        return self._state_take_playing

    @property
    def state_take_cursor(self) -> int:
        """Return the current transient take frame, or ``-1`` when no frame is active."""
        return self._state_take_cursor

    @property
    def state_take_times(self) -> list[float]:
        """Return recorded simulation times in take order."""
        return self._state_take_times

    @property
    def state_take_loop(self) -> tuple[int, int] | None:
        """Return the inclusive replay range, or None when replay stops at the take end."""
        return self._state_take_loop

    @property
    def can_step_back(self) -> bool:
        """Return whether frame history can restore an earlier displayed state."""

        return bool(
            self._adapter.caps.simulation
            and self._adapter.caps.state_snapshots
            and self._paused
            and not self._state_take_recording
            and not self._state_take_playing
            and (
                self._frame_history_cursor > 0
                or (self._frame_history_dirty and self._frame_history_cursor >= 0)
            )
        )

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
        """Return Mojive-authored overrides layered over adapter structure."""
        return self._authored

    @property
    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        """Return file-backed models participating in the composed scene."""
        return self._adapter.scene_models()

    def model_component_count(self, model_id: int, category: str) -> int:
        """Count model declarations without building their editor fields."""
        return self._adapter.model_component_count(model_id, category)

    def model_components(self, model_id: int, category: str):
        """Return editable components in one model-level MJCF category."""
        return self._adapter.model_components(model_id, category)

    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]:
        """Return supported component subtypes for a model and category."""
        return self._adapter.model_component_presets(model_id, category)

    def model_assets(self, model_id: int) -> tuple[ModelAssetInfo, ...]:
        """Return model-local assets and their current reference summaries."""
        return self._adapter.model_assets(model_id)

    def model_material_indices(self, model_id: int) -> tuple[int, ...]:
        """Return render material indices owned by one editable model."""
        return self._adapter.model_material_indices(model_id)

    def keyframe_properties(self, keyframe_id: int) -> KeyframeProperties | None:
        """Return complete editable state for one model-local keyframe."""
        value = self._adapter.keyframe_properties(keyframe_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("keyframe_properties", keyframe_id, value)
            if preview is not None
            else value
        )

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        """Return editable contact parameters for one model geometry."""
        return self._adapter.geometry_properties(node_id)

    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None:
        """Return joint properties backed by rebuilt MuJoCo constants."""
        value = self._adapter.joint_advanced_properties(joint_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("joint_advanced_properties", joint_id, value)
            if preview is not None
            else value
        )

    def site_properties(self, node_id: int) -> SiteProperties | None:
        """Return editable shape and endpoint properties for one model site."""
        value = self._adapter.site_properties(node_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("site_properties", node_id, value) if preview is not None else value
        )

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        """Return geometry properties backed by rebuilt MuJoCo constants."""
        value = self._adapter.geometry_advanced_properties(node_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("geometry_advanced_properties", node_id, value)
            if preview is not None
            else value
        )

    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None:
        """Return geometry type and model-local resource choices."""
        value = self._adapter.geometry_shape_properties(node_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("geometry_shape_properties", node_id, value)
            if preview is not None
            else value
        )

    def body_properties(self, node_id: int) -> BodyProperties | None:
        """Return editable inertial and dynamic properties for one model body."""
        value = self._adapter.body_properties(node_id)
        preview = getattr(self, "_model_edit_preview", None)
        return (
            preview.properties("body_properties", node_id, value) if preview is not None else value
        )

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
    def message_revision(self) -> int:
        """Return the monotonic revision of non-empty user-facing messages."""
        return self._message_revision

    @property
    def last_message_level(self) -> str:
        return self._last_message_level

    @property
    def last_message_duration(self) -> float | None:
        return self._last_message_duration

    @property
    def last_message_copy_text(self) -> str | None:
        return self._last_message_copy_text

    def report_message(
        self,
        message: str,
        *,
        level: str = "warning",
        duration: float | None = 5.0,
        copy_text: str | None = None,
    ) -> None:
        """Publish a user-facing UI or runtime diagnostic without creating a command."""
        self._publish_message(str(message), level=level, duration=duration, copy_text=copy_text)

    def _publish_message(
        self,
        message: str,
        *,
        level: str,
        duration: float | None,
        copy_text: str | None = None,
    ) -> None:
        self._last_message = str(message)
        if not self._last_message:
            return
        self._message_revision += 1
        self._last_message_level = str(level)
        self._last_message_duration = duration
        self._last_message_copy_text = copy_text

    def _record_result(self, result: CommandResult) -> CommandResult:
        self._last_message = result.message
        if result.message:
            self._publish_message(
                result.message,
                level="info" if result.ok else "error",
                duration=5.0 if result.ok else 10.0,
            )
        return result

    @property
    def dirty(self) -> bool:
        """Return whether the current document contains unsaved edits."""
        preview = getattr(self, "_model_edit_preview", None)
        return (
            bool(preview is not None and preview.active)
            or self._edit_changed
            or self._document_revision != self._saved_revision
        )

    @property
    def document_id(self) -> str:
        """Return the current document identity, renewed after new/open/load/reload."""
        return self._document_id

    @property
    def document_revision(self) -> int:
        """Return the authored revision represented by the current undo history state."""
        return self._document_revision

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
    def history_bytes(self) -> int:
        """Return estimated Python/NumPy storage retained by Undo and Redo."""
        return self._history.bytes

    @property
    def structure_generation(self) -> int:
        """Return the generation incremented after stable structure changes."""
        return self._structure_generation + getattr(self, "_preview_generation", 0)

    def node(self, node_id: int) -> SceneNode | None:
        """Look up a hierarchy node by node ID."""
        preview = getattr(self, "_model_edit_preview", None)
        if preview is not None and preview.visible:
            return preview.by_node_id.get(int(node_id))
        return self._by_node_id.get(int(node_id))

    def node_local_bounds(self, node_id: int) -> CenteredBounds | None:
        """Return center and half extent in the owning body's local frame."""
        node = self.node(node_id)
        return (
            None
            if node is None
            else _node_local_bounds(self.source, self.frame, node, self._mesh_bounds_cache)
        )

    def scale_target(self, node_id: int) -> SceneNode | None:
        """Resolve one editable geometry shared by an object and its Inspector row."""
        node = self.node(node_id)
        if node is None or not node.scalable or not self._adapter.caps.write_scale:
            return None
        if node.type is NodeType.GEOM:
            return node
        children = [self.node(child) for child in node.children]
        geometry = [child for child in children if child is not None and child.scalable]
        return geometry[0] if len(geometry) == 1 and geometry[0].type is NodeType.GEOM else None

    def scale_factors(self, node_id: int) -> tuple[float, float, float]:
        """Return pending local scale; baked geometry always has identity scale."""
        target = self.scale_target(node_id)
        preview = getattr(self, "_model_edit_preview", None)
        if target is not None and preview is not None and preview.visible:
            command = preview.pending_scale(target.node_id)
            if command is not None:
                return tuple(command.scale)
        return (1.0, 1.0, 1.0)

    def node_world_bounds(self, node_id: int) -> CenteredBounds | None:
        """Return selected geometry center and half extent in world space."""
        node = self.node(node_id)
        return (
            None
            if node is None
            else _node_world_bounds(
                self.source, self.frame, node, self.nodes, self._mesh_bounds_cache
            )
        )

    def node_by_object_id(self, object_id: int) -> SceneNode | None:
        """Look up a hierarchy node by selectable object ID."""
        preview = self._model_edit_preview
        if preview is not None and preview.visible:
            return preview.by_object_id.get(int(object_id))
        return self._by_object_id.get(int(object_id))

    def _dispatch(self, c: Command) -> CommandResult:
        return dispatch(self, c)

    def release(self) -> None:
        """Release resources owned by the scene adapter."""
        self.set_threaded_physics(False)
        self._adapter.release()
