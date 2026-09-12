"""App: core."""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds, NodeType
from mojive.capture import CaptureSurface, RecordingPhase
from mojive.config import (
    CameraTrackingConfig,
    InteractionConfig,
    RecordingConfig,
    SelectionStyle,
    ViewerConfig,
    ViewportLayers,
    ViewportOverlayConfig,
)
from mojive.log import add_output_sink, remove_output_sink
from mojive.render.backend import FrameMode, LabelMode, RenderFlag, ShadowQuality
from mojive.scene.queries import node_world_pose
from mojive.scene.workspace import MissingResource
from mojive.session.model_edits import ModelEditDraft, model_edit_scope
from mojive.types import ViewportImage
from mojive.ui import gestures as gs
from mojive.ui.camera import (
    CameraOut,
    OrbitCamera,
    ProjectionTransition,
    ndc_from_viewport,
    unproject,
)
from mojive.ui.camera_preview import CameraPreview
from mojive.ui.camera_tracking import CameraTracker
from mojive.ui.gizmo import ObjectGizmo, PreciseGizmoInput
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from mojive.ui.layers import visible_debug_layers
from mojive.ui.localization import Localizer
from mojive.ui.messages import OutputBuffer
from mojive.ui.panels import (
    PanelContext,
    PanelManager,
)
from mojive.ui.perturb import (
    PerturbController,
)
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.scene_capture import SceneCapture
from mojive.ui.scene_entities import SceneEntityHelpers
from mojive.ui.take_video import TakeVideo
from mojive.ui.theme import THEME, Theme
from mojive.ui.viewcube import DEFAULT_SELECTION_PADDING, ViewCube
from mojive.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    MAX_VIEWPORT_CAPSULE_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_CAPSULE_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
    ToolHint,
    ViewportChromeRegistry,
    localized_viewport_labels,
)
from mojive.ui.window import Window, WindowConfig

if TYPE_CHECKING:
    from mojive.session import Session


from .capture import _Capture
from .gizmo_input import _GizmoInput
from .input import _Input
from .loading import _Loading
from .menus import _Menus
from .model_edits import _ModelEdits
from .navigation import _Navigation
from .resource_dialogs import _ResourceDialogs
from .status import _Status
from .support import (
    _NO_INPUT_CLAIM,
    _fit_image_rect,
    _FrameRateDisplay,
    _GizmoHintHoverState,
    _JointLimitHoverState,
    _ModelLoadCompletion,
    _ModelLoadJob,
    log,
)
from .viewport import _Viewport


class ViewerApp(
    _Capture,
    _Navigation,
    _Loading,
    _ModelEdits,
    _ResourceDialogs,
    _Menus,
    _Input,
    _GizmoInput,
    _Viewport,
    _Status,
):
    viewport_layers = ViewportLayers()

    def __init__(
        self,
        session: Session,
        backend: Any,
        window: Window | None = None,
        *,
        title: str = "Mojive",
        theme: Theme | None = None,
        debug_bridge: Any | None = None,
        config: ViewerConfig | None = None,
    ) -> None:
        self.session = session
        self.backend = backend
        self.window = window
        self.title = title
        self.theme = theme or THEME
        if self.window is not None:
            self.window.apply_theme(self.theme)
        self.debug_bridge = debug_bridge
        self.localizer = Localizer.load()
        explicit_config = config is not None
        viewer_config = config or ViewerConfig(
            interactions=InteractionConfig.from_mapping(
                self.localizer.preference("interactions", {})
            ),
            selection=SelectionStyle.from_mapping(self.localizer.preference("selection_style", {})),
        )
        self.model_edits = ModelEditDraft(session)
        self.live_model_updates = bool(
            self.localizer.preference("live_model_updates", False)
            if viewer_config.live_model_updates is None
            else viewer_config.live_model_updates
        )
        self._apply_model_edits_requested = False
        self.interactions = viewer_config.interactions
        self._threaded_physics = viewer_config.threaded_physics
        self.selection_style = viewer_config.selection
        overlay_config = ViewportOverlayConfig.from_mapping(
            asdict(viewer_config.viewport_overlays)
            if explicit_config
            else self.localizer.preference("viewport_overlays", {})
        )
        self.viewport_overlays = overlay_config
        self.viewport_layers = ViewportLayers.from_mapping(
            asdict(viewer_config.layers)
            if explicit_config
            else self.localizer.preference("viewport_layers", {})
        )
        self.recording_config = RecordingConfig.from_mapping(
            asdict(viewer_config.recording)
            if explicit_config
            else self.localizer.preference("recording", {})
        )
        self.camera_tracker = CameraTracker(
            CameraTrackingConfig.from_mapping(
                asdict(viewer_config.tracking)
                if explicit_config
                else self.localizer.preference("camera_tracking", {})
            )
        )
        self._tracking_adapter = None
        self._input_handler = None
        self._input_claim = _NO_INPUT_CLAIM
        self._popup_owned_frame = False
        self._rpc_service = None
        self._viewport_focused = False
        self._selection_press_started_focused = False
        self._selection_transform = np.eye(4, dtype=np.float32)
        self._selection_bounds_corners = np.empty((8, 3), np.float32)
        self._selection_bounds_starts = np.empty((12, 3), np.float32)
        self._selection_bounds_ends = np.empty((12, 3), np.float32)
        requested_shadow_quality = (
            viewer_config.shadow_quality
            if viewer_config.shadow_quality is not None
            else self.localizer.preference("shadow_quality", ShadowQuality.BALANCED.value)
        )
        try:
            shadow_quality = ShadowQuality(requested_shadow_quality)
        except (TypeError, ValueError):
            shadow_quality = ShadowQuality.BALANCED
        set_shadow_quality = getattr(self.backend, "set_shadow_quality", None)
        if set_shadow_quality is not None:
            set_shadow_quality(shadow_quality)
        self._viewport_labels = localized_viewport_labels(self.localizer.text)
        metric_mode = self.localizer.preference("status_metric", "time")
        self._status_metric_mode = "steps" if metric_mode == "steps" else "time"
        self._panel_status_hints: tuple[ToolHint, ...] = ()
        self._status_panel = "Viewport"
        self.camera = OrbitCamera()

        self.camera_out = CameraOut(backend=backend, session=session)
        self.camera.attach(self.camera_out)
        self.camera_preview = CameraPreview()
        self._scene_capture = SceneCapture()
        self.gizmo = ObjectGizmo(enabled=False)
        remember_precise = self.localizer.preference("remember_precise_input_choices", True)
        if isinstance(remember_precise, bool):
            self.gizmo.remember_precise_input_choices = remember_precise
        selection_padding = self.localizer.preference(
            "view_selection_padding", DEFAULT_SELECTION_PADDING
        )
        try:
            selection_padding = float(selection_padding)
        except (TypeError, ValueError):
            selection_padding = DEFAULT_SELECTION_PADDING
        self.view_cube = ViewCube(selection_padding)
        self.perturb = PerturbController()
        self.scene_entities = SceneEntityHelpers()
        self.router = gs.GestureRouter()
        self.input_bindings = InputBindings.from_preferences(
            self.localizer.preference("input_bindings", {})
        )
        self.viewport_chrome = ViewportChromeRegistry()
        # Kept as a direct public alias for callers that only customize hints.
        self.tool_hints = self.viewport_chrome.tool_hints
        self.output = OutputBuffer()
        self.panels = PanelManager(config=dict(viewer_config.panels))
        if os.environ.get("MOJIVE_OPEN_SETTINGS") == "1":
            self.panels.open_panel("Settings")
        self._started = False
        self._released = False
        self._frame_index = 0
        self._last_time = time.perf_counter()
        self._viewport_rect = (0.0, 0.0, 640.0, 480.0)
        self._viewport_panel_position = (0.0, 0.0)
        self._viewport_panel_size = (640.0, 480.0)
        self._viewport_image: ViewportImage | None = None
        self._dt = 0.0
        self._frame_rate = _FrameRateDisplay()
        self._structure_generation = -1
        self._state = gs.InputState()
        self._model_camera_id = -1
        self._model_camera_view = None
        self._camera_transition = None
        self._model_camera_projection = ProjectionTransition()
        self._model_camera_projection_target: bool | None = None
        self._fixed_render_size: tuple[int, int] | None = None
        self._model_dialog: Any | None = None
        self._model_dialog_action = ""
        self._scene_dialog: Any | None = None
        self._scene_dialog_action = ""
        self._resource_dialog: Any | None = None
        self._texture_dialog: Any | None = None
        self._texture_import_target = (-1, -1, "2d")
        self._geometry_resource_dialog: Any | None = None
        self._geometry_resource_import_target = (-1, "")
        self._model_asset_dialog: Any | None = None
        self._model_asset_dialog_target: tuple[str, int, str, str, tuple[tuple[str, str], ...]] = (
            "",
            -1,
            "",
            "",
            (),
        )
        self._resource_repair_dialog: Any | None = None
        self._resource_repair_dialog_action = ""
        self._resource_repair_model_index = -1
        self._resource_repair_path: Path | None = None
        self._missing_resources: tuple[MissingResource, ...] = ()
        self._resource_repair_status = ""
        self._open_resource_repair_popup = False
        self._pending_document_action: tuple[str, Path | None] | None = None
        self._after_save_action: tuple[str, Path | None] | None = None
        self._pending_pose_save: tuple[Path, tuple[str, Path | None] | None] | None = None
        self._rename_object_id = 0
        self._rename_model_node_id = -1
        self._rename_value = ""
        self._open_rename_popup = False
        self._precise_gizmo_edit: PreciseGizmoInput | None = None
        self._precise_gizmo_value = 0.0
        self._precise_gizmo_absolute = False
        preferred_absolute = self.localizer.preference("precise_gizmo_absolute", False)
        self._precise_gizmo_preferred_absolute = (
            preferred_absolute if isinstance(preferred_absolute, bool) else False
        )
        angle_unit = self.localizer.preference("precise_gizmo_angle_unit", "degrees")
        self._precise_gizmo_angle_unit = (
            str(angle_unit) if angle_unit in ("degrees", "radians") else "degrees"
        )
        self._precise_gizmo_error = ""
        self._open_precise_gizmo_popup = False
        # When an outside click dismisses precise input, keep ownership until
        # that physical press has fully ended.  The same click must never fall
        # through to viewport picking and clear the selected joint.
        self._consume_scene_pointer_until_release = False
        self._last_viewport_click: tuple[float, tuple[float, float], int] | None = None
        self._joint_picker_node_id = -1
        self._pending_node_focus_id: int | None = None
        self._pending_joint_focus_id: int | None = None
        self._joint_limit_hover = _JointLimitHoverState()
        self._gizmo_hint_hover = _GizmoHintHoverState()
        self._window_title = ""
        self._closing_without_save = False
        self._model_load_error = ""
        self._show_model_load_error = False
        self._model_load_executor: ThreadPoolExecutor | None = None
        self._model_load_future: Future[Any] | None = None
        self._model_load_job: _ModelLoadJob | None = None
        self._model_load_queue: list[_ModelLoadJob] = []
        self._model_load_started = 0.0
        self._model_source_prepared = 0.0
        self._model_prepared_resources = None
        self._model_load_completion: _ModelLoadCompletion | None = None
        self._close_after_model_load = False
        self._model_drop_notice = ""
        self._model_drop_notice_until = 0.0
        self._display_scale_generation = -1
        self._output_sink_id: int | None = None
        self._seen_message_revision = int(getattr(session, "message_revision", 0))
        self._snap_latched = False
        self._capture_request: tuple[Path, CaptureSurface] | None = None
        self._capture_tasks: list[
            tuple[Path | None, CaptureSurface, Future, np.ndarray | None]
        ] = []
        self._viewport_recording_mode = "video"
        self._viewport_recorder: Any | None = None
        self._viewport_recording_path: Path | None = None
        self._viewport_record_elapsed = 0.0
        self._viewport_recording_phase = RecordingPhase.IDLE
        self._viewport_recording_surface = self.recording_config.surface
        self._viewport_recording_fps = self.recording_config.fps
        self._recording_deadline = 0.0
        self._viewport_recording_frames = 0
        self._viewport_recording_duration = 0.0
        self._take_video: TakeVideo | None = None
        self._playback_widget_rect: tuple[float, float, float, float] | None = None
        self._tool_widget_rect: tuple[float, float, float, float] | None = None
        self._overlay_drag_chord = None
        self._overlay_drag_kind = ""
        self._overlay_drag_offset = (0.0, 0.0)
        overlay_scale = self.localizer.preference(
            "viewport_overlay_scale", DEFAULT_VIEWPORT_OVERLAY_SCALE
        )
        try:
            overlay_scale = float(overlay_scale)
        except (TypeError, ValueError):
            overlay_scale = DEFAULT_VIEWPORT_OVERLAY_SCALE
        self._viewport_overlay_scale = min(
            MAX_VIEWPORT_OVERLAY_SCALE,
            max(MIN_VIEWPORT_OVERLAY_SCALE, overlay_scale),
        )

    def set_language(self, language: str) -> None:
        self.localizer.set_language(language)
        self._viewport_labels = localized_viewport_labels(self.localizer.text)

    def set_shadow_quality(self, quality: ShadowQuality | str, *, persist: bool = True) -> bool:
        try:
            quality = ShadowQuality(quality)
        except (TypeError, ValueError):
            return False
        setter = getattr(self.backend, "set_shadow_quality", None)
        if setter is None or not setter(quality):
            return False
        if persist:
            self.localizer.set_preferences({"shadow_quality": quality.value})
        return True

    def set_camera_tracking(self, value: CameraTrackingConfig, *, persist: bool = True) -> None:
        """Change tracking axes or smoothing without moving the camera immediately."""
        if not isinstance(value, CameraTrackingConfig):
            raise TypeError("tracking must be a CameraTrackingConfig")
        self.camera_tracker.config = value
        if persist:
            self.localizer.set_preferences({"camera_tracking": asdict(value)})

    def set_viewport_layers(self, value: ViewportLayers, *, persist: bool = True) -> None:
        """Set live viewport visibility while preserving individual tool settings."""
        self.viewport_layers = ViewportLayers.from_mapping(asdict(value))
        if not self.viewport_layers.gizmos:
            self.gizmo.cancel()
            self._precise_gizmo_edit = None
        if not self.viewport_layers.viewport_ui:
            self._playback_widget_rect = None
            self._tool_widget_rect = None
            self._overlay_drag_kind = ""
        if persist:
            self.localizer.set_preferences({"viewport_layers": asdict(self.viewport_layers)})

    def _toggle_status_metric(self) -> None:
        self._status_metric_mode = "steps" if self._status_metric_mode == "time" else "time"
        self.localizer.set_preferences({"status_metric": self._status_metric_mode})

    def set_precise_input_choice_memory(self, enabled: bool) -> None:
        self.gizmo.remember_precise_input_choices = bool(enabled)
        values: dict[str, object] = {"remember_precise_input_choices": bool(enabled)}
        if enabled:
            values.update(
                precise_gizmo_absolute=self._precise_gizmo_preferred_absolute,
                precise_gizmo_angle_unit=self._precise_gizmo_angle_unit,
            )
        self.localizer.set_preferences(values)

    def set_view_selection_padding(self, value: float) -> None:
        self.view_cube.selection_padding = value
        self.localizer.set_preferences({"view_selection_padding": self.view_cube.selection_padding})

    def set_viewport_overlay_scale(self, value: float, *, persist: bool = True) -> None:
        self._viewport_overlay_scale = min(
            MAX_VIEWPORT_OVERLAY_SCALE,
            max(MIN_VIEWPORT_OVERLAY_SCALE, float(value)),
        )
        if persist:
            self.localizer.set_preferences({"viewport_overlay_scale": self._viewport_overlay_scale})

    def set_viewport_overlays(self, value: ViewportOverlayConfig, *, persist: bool = True) -> None:
        """Replace playback/tool capsule presentation and movement policy."""

        if not isinstance(value, ViewportOverlayConfig):
            raise TypeError("viewport overlays must be a ViewportOverlayConfig")
        self.viewport_overlays = ViewportOverlayConfig.from_mapping(asdict(value))
        if not self.viewport_overlays.movable:
            self._overlay_drag_kind = ""
        if persist:
            self.localizer.set_preferences({"viewport_overlays": asdict(self.viewport_overlays)})

    def reset_layout(self, *, persist: bool = True) -> None:
        """Restore dock panels and movable viewport chrome to product defaults."""

        self.window.reset_layout()
        self._overlay_drag_kind = ""
        self.set_viewport_overlays(
            replace(
                self.viewport_overlays,
                playback_position=None,
                tool_position=None,
            ),
            persist=persist,
        )

    def set_viewport_capsule_scale(self, name: str, value: float, *, persist: bool = True) -> None:
        """Set the shared thickness of both viewport capsules."""

        value = float(value)
        if not np.isfinite(value):
            raise ValueError("viewport capsule scale must be finite")
        value = min(MAX_VIEWPORT_CAPSULE_SCALE, max(MIN_VIEWPORT_CAPSULE_SCALE, value))
        if name in ("playback", "tools"):
            config = replace(self.viewport_overlays, playback_scale=value, tool_scale=value)
        else:
            raise ValueError(f"unknown viewport capsule: {name!r}")
        self.set_viewport_overlays(config, persist=persist)

    def set_input_binding(
        self,
        action: InputAction,
        key_id: str | None,
        *,
        persist: bool = True,
    ) -> None:
        """Atomically remap one viewport action and persist the whole map."""

        self.input_bindings = self.input_bindings.remap(action, key_id)
        if persist:
            self.localizer.set_preferences({"input_bindings": self.input_bindings.preferences()})

    def set_pointer_binding(
        self, action: PointerAction, identifiers: tuple[str, ...], *, persist: bool = True
    ) -> None:
        """Replace one mouse action's alternatives after conflict validation."""
        self.input_bindings = self.input_bindings.remap_pointer(action, identifiers)
        if persist:
            self.localizer.set_preferences({"input_bindings": self.input_bindings.preferences()})

    def set_navigation_preset(self, name: str, *, persist: bool = True) -> None:
        """Apply camera navigation bindings without changing editing gestures."""
        self.input_bindings = self.input_bindings.navigation_preset(name)
        if persist:
            self.localizer.set_preferences({"input_bindings": self.input_bindings.preferences()})

    def set_interactions(self, value: InteractionConfig, *, persist: bool = True) -> None:
        """Replace the built-in input policy without changing application bindings."""

        if not isinstance(value, InteractionConfig):
            raise TypeError("interactions must be an InteractionConfig")
        self.interactions = value
        if persist:
            self.localizer.set_preferences({"interactions": asdict(value)})
        if not value.gizmo:
            self.gizmo.cancel()
        if not value.perturb and self.session.perturb.active:
            self.perturb.end(self.session)

    def set_selection_style(self, value: SelectionStyle, *, persist: bool = True) -> None:
        """Replace selection presentation independently from logical selection."""

        if not isinstance(value, SelectionStyle):
            raise TypeError("selection style must be a SelectionStyle")
        self.selection_style = value
        if persist:
            self.localizer.set_preferences({"selection_style": asdict(value)})
        if not value.gizmo:
            self.gizmo.cancel()

    def set_input_handler(self, handler) -> None:
        """Set a callback that observes input and returns an optional InputClaim."""

        if handler is not None and not callable(handler):
            raise TypeError("input handler must be callable or None")
        self._input_handler = handler

    def reset_input_bindings(self) -> None:
        self.input_bindings = DEFAULT_INPUT_BINDINGS
        self.localizer.set_preferences({"input_bindings": self.input_bindings.preferences()})

    def set_fixed_render_size(self, width: int, height: int) -> None:
        self._fixed_render_size = (max(1, int(width)), max(1, int(height)))
        self.backend.resize(*self._fixed_render_size)
        self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])

    @property
    def fixed_render_size(self) -> tuple[int, int] | None:
        """Return the active fixed output size, if viewport sizing is overridden."""
        return self._fixed_render_size

    def clear_fixed_render_size(self) -> None:
        """Return rendering to the interactive viewport size."""
        self._fixed_render_size = None
        if self._started:
            self._sync_viewport_size()

    def _startup(self) -> None:
        if self._started:
            return
        if self._output_sink_id is None:
            self._output_sink_id = add_output_sink(self.output.loguru_sink)
        if self.window is None:
            self.window = Window(WindowConfig(title=self.title))
            self.window.apply_theme(self.theme)
        self._sync_structure()
        self._reset_source_camera()
        self.session.set_threaded_physics(self._threaded_physics)
        if self.window.config.show_on_start:
            self.window.show()
        self._started = True
        self._last_time = time.perf_counter()

    def run(self, max_frames: int | None = None) -> None:
        self._startup()
        while not self._should_close():
            if max_frames is not None and self._frame_index >= max_frames:
                break
            self.frame()

    def sync(self) -> None:
        self._startup()
        self.frame()

    def _should_close(self) -> bool:
        closing = bool(self.window.should_close())
        if closing and self._model_load_future is not None:
            self.window.cancel_close()
            self._close_after_model_load = True
            return False
        if closing and self._closing_without_save:
            return True
        if closing and self.session.dirty and self._pending_document_action is None:
            self.window.cancel_close()
            self._pending_document_action = ("quit", None)
            return False
        return closing

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for _path, _surface, future, _out in getattr(self, "_capture_tasks", []):
            if not future.done():
                future.set_exception(RuntimeError("The viewer closed before capture completed"))
        self._capture_tasks = []
        executor = getattr(self, "_model_load_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._model_load_executor = None
            self._model_load_future = None
            self._model_load_job = None
            self._model_load_queue.clear()
            self._model_prepared_resources = None
        for attribute in (
            "_model_dialog",
            "_scene_dialog",
            "_resource_dialog",
            "_resource_repair_dialog",
        ):
            dialog = getattr(self, attribute)
            setattr(self, attribute, None)
            if dialog is not None:
                self._release_resource(dialog, "kill", attribute)
        if self.debug_bridge is not None:
            self._release_resource(self.debug_bridge, "close", "debug bridge")
            self.debug_bridge = None
        self._stop_viewport_recording(report=False)
        self._release_resource(self.camera_preview, "release", "camera preview")
        self._release_resource(self._scene_capture, "release", "scene capture")
        self._release_resource(self.backend, "release", "render backend")
        self._release_resource(self.session, "release", "session")
        output_sink_id = getattr(self, "_output_sink_id", None)
        if output_sink_id is not None:
            remove_output_sink(output_sink_id)
            self._output_sink_id = None

    @staticmethod
    def _release_resource(resource: Any, operation: str, name: str) -> None:
        try:
            getattr(resource, operation)()
        except Exception as exc:
            log.warning("Failed to release {}: {}", name, exc)

    def _sync_window_title(self) -> None:
        path = self.session.asset_path
        document = path.name if path is not None else "Untitled"
        title = f"{document}{' *' if self.session.dirty else ''} — {self.title}"
        if title != self._window_title:
            self.window.set_title(title)
            self._window_title = title

    def frame(self) -> None:
        with model_edit_scope(self.session, self._intercept_model_edit):
            self._frame()

    def _frame(self) -> None:
        window = self.window
        now = time.perf_counter()
        elapsed = now - self._last_time
        dt = self._dt = min(0.1, elapsed)
        self._last_time = now
        self._frame_rate.update(elapsed)

        window.begin_frame()
        self._popup_owned_frame = window.popup_owned_frame
        self._advance_recording_countdown()
        self._sync_display_scale()
        if self._model_load_future is not None and self._poll_model_load():
            self._draw_model_loading_frame()
            self._present_frame(dt)
            self._frame_index += 1
            return
        if self._rpc_service is not None:
            with model_edit_scope(self.session, None):
                self._rpc_service.pump()
        self._poll_model_dialog()
        self._poll_scene_dialog()
        self._poll_resource_dialog()
        self._poll_texture_dialog()
        self._poll_geometry_resource_dialog()
        self._poll_model_asset_dialog()
        self._poll_resource_repair_dialog()
        self._poll_model_drop()
        self._start_pending_model_edits()
        if self._start_model_load():
            self._draw_model_loading_frame()
            self._present_frame(dt)
            self._frame_index += 1
            return
        self._draw_main_menu()
        self._draw_application_status_bar()
        self._draw_collapsed_output()
        window.begin_dockspace()
        self._begin_viewport_panel()
        self._sync_viewport_size()
        self._poll_input_handler()
        self._poll_application_shortcuts()
        keys = self._poll_keys()
        self.apply_keys(keys)

        state = self._state = self._input_state()

        self._claim_gesture(state)

        self._poll_gizmo(state, keys)
        self._poll_camera(state, keys, dt)
        self._poll_perturb(state)
        self._poll_pick(state)
        self._advance_camera(dt)
        self._finish_consumed_scene_pointer()

        playback_dt = self._prepare_take_video(dt)
        frame = self.session.tick(self.frame_needs(), wall_dt=playback_dt)
        if self._take_video is not None:
            self._take_video.cursor = self.session.state_take_cursor
            self._take_video.playing = self.session.state_take_playing
            if (
                self._take_video.started
                and self.recording.phase is RecordingPhase.RECORDING
                and not self._take_video.playing
                and self._take_video.cursor != self._take_video.frame_count - 1
            ):
                self.stop_recording(report=False)
        self._sync_structure()
        self._sync_camera_tracking(elapsed)
        self._apply_pending_joint_focus()
        self._apply_pending_node_focus()
        self._sync_model_camera()
        self.backend.update(frame)

        selected_node = self.session.selected_node
        self.backend.highlight(
            self.session.selection_highlight_object_id,
            xray=bool(selected_node is not None and selected_node.type is NodeType.JOINT),
            fill=self.selection_style.highlight and self.viewport_layers.selection,
            outline=self.selection_style.outline and self.viewport_layers.selection,
        )

        if self.debug_bridge is not None:
            self.debug_bridge.pump()
            if frame.debug_commands:
                self.debug_bridge.apply_batch(frame.debug_commands)

        preview_name, preview_camera = self.camera_preview.selected_camera(self.session)
        preview_width = min(
            1024, max(320, int(self.window.points_to_pixels(340.0 * self.window.style_scale)))
        )
        preview_size = (preview_width, max(1, preview_width * 9 // 16))
        self._publish_perturb_marks()
        self.scene_entities.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect[3],
            self.window.ui_scale,
            view_through_camera=self._model_camera_id >= 0,
            # The selected camera helper represents the fixed 16:9 inspector
            # preview surface even while that surface is hidden. A checkbox
            # must not change authored camera helper geometry.
            selected_camera_aspect=preview_size[0] / preview_size[1],
        )
        self._publish_selection_style()
        self._publish_gizmo()

        with visible_debug_layers(self.backend.debug, self.viewport_layers):
            self._viewport_image = self.backend.render()
        self.camera_preview.update(
            self.backend,
            self.session.source,
            self.session.structure_generation,
            frame,
            preview_camera if self.viewport_layers.viewport_ui else None,
            preview_size,
        )

        self._sync_session_status()
        ctx = self._panel_context()
        self._draw_viewport_contents(preview_name)
        self._draw_playback_widget()
        self._draw_tool_column_widget()
        self._draw_context_hint_widget()
        self.panels.draw(ctx)
        self._update_status_context(ctx)
        self._draw_precise_gizmo_popup()
        self._draw_rename_popup()
        self._draw_unsaved_changes()
        self._draw_pose_save_prompt()
        self._draw_resource_repair()
        self._draw_model_load_error()
        self._draw_recording_countdown()
        self._sync_window_title()
        self._present_frame(dt)
        self._frame_index += 1

    def _draw_model_loading_frame(self) -> None:
        """Keep the native window responsive without reading a mutating Session."""

        self._draw_loading_main_menu()
        self._draw_application_status_bar(loading=True)
        self.window.begin_dockspace()
        self.panels.draw_shells(self.localizer.text, self.window.style_scale)
        self._begin_viewport_panel()
        self._sync_viewport_size()
        with visible_debug_layers(self.backend.debug, self.viewport_layers):
            self._viewport_image = self.backend.render()
        self._draw_viewport_contents(session_busy=True)
        self._draw_model_loading_window()

    def _draw_loading_main_menu(self) -> None:
        """Preserve the menu-bar geometry without touching the loading Session."""

        if not imgui.begin_main_menu_bar():
            return
        for label in ("File", "Edit", "Entity"):
            imgui.begin_menu(self.localizer.text(label), False)
        job = self._model_load_job
        if job is not None:
            imgui.text_disabled(job.path.name)
        imgui.end_main_menu_bar()

    def _draw_model_loading_window(self) -> None:
        job = self._model_load_job
        if job is None:
            return
        x, y, viewport_width, viewport_height = self._viewport_rect
        width = min(460.0, max(1.0, viewport_width - 32.0))
        imgui.set_next_window_pos(
            imgui.ImVec2(x + viewport_width * 0.5, y + viewport_height * 0.5),
            imgui.Cond_.always.value,
            imgui.ImVec2(0.5, 0.5),
        )
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(width, 0.0),
            imgui.ImVec2(width, float(np.finfo(np.float32).max)),
        )
        flags = (
            imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        t = self.localizer.text
        visible, _ = imgui.begin(f"{t('Loading')}###model_loading", None, flags)
        if visible:
            elapsed = max(0.0, time.monotonic() - self._model_load_started)
            dots = "." * (int(elapsed * 2.0) % 3 + 1)
            imgui.text(f"{t(self._model_load_verb(job.action))}{dots}")
            imgui.separator()
            imgui.text_disabled(t("File"))
            imgui.text_wrapped(str(job.path))
            imgui.spacing()
            imgui.text(f"{t('Elapsed')}: {elapsed:.1f} s")
            if self._model_load_queue:
                imgui.text(f"{t('Queued')}: {len(self._model_load_queue)}")
        imgui.end()

    def _publish_gizmo(self) -> None:
        self.gizmo.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
            yielding=not self.selection_style.gizmo
            or not self.viewport_layers.gizmos
            or gs.gizmo_yields(self._state)
            or self._viewing_selected_camera(),
            interactive=self.interactions.gizmo
            and self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO),
        )

    def _has_scene_content(self) -> bool:
        source = self.session.source
        if source is None:
            return False
        lights = getattr(getattr(source, "lights", None), "lights", ())
        cameras = getattr(source, "cameras", ())
        debug = getattr(self.backend, "debug", None)
        debug_primitives = int(getattr(debug, "primitives", 0))
        return bool(getattr(source, "instance_count", 0) or lights or cameras or debug_primitives)

    def frame_needs(self) -> FrameNeeds:
        needs = FrameNeeds(poses=True).merge(self.panels.frame_needs())
        interactions = getattr(self, "interactions", None)
        selection_style = getattr(self, "selection_style", None)
        if (interactions is None or interactions.gizmo) and (
            selection_style is None or selection_style.gizmo
        ):
            needs = needs.merge(self.gizmo.frame_needs(self.session))
        if getattr(self, "_pending_joint_focus_id", None) is not None:
            needs = needs.merge(FrameNeeds(poses=False, qpos=True, joint_frames=True))
        label_mode = self.backend.get_label_mode()
        frame_mode = self.backend.get_frame_mode()
        needs.contacts = needs.contacts or (
            self.backend.get_flag(RenderFlag.CONTACTPOINT)
            or self.backend.get_flag(RenderFlag.CONTACTFORCE)
            or label_mode in (LabelMode.CONTACT_POINT, LabelMode.CONTACT_FORCE)
            or frame_mode is FrameMode.CONTACT
        )
        needs.tendons = needs.tendons or (
            self.backend.get_flag(RenderFlag.TENDON)
            or self.backend.get_flag(RenderFlag.ACTUATOR)
            or label_mode is LabelMode.TENDON
        )
        needs.actuator = needs.actuator or (
            self.backend.get_flag(RenderFlag.ACTUATOR) or label_mode is LabelMode.ACTUATOR
        )
        needs.deformables = needs.deformables or bool(
            (self.session.source and self.session.source.dynamic_meshes)
            or self.backend.get_flag(RenderFlag.FLEXVERT)
            or self.backend.get_flag(RenderFlag.FLEXEDGE)
            or label_mode is LabelMode.FLEX
        )
        needs.islands = needs.islands or self.backend.get_flag(RenderFlag.ISLAND)
        needs.bvh = (
            needs.bvh
            or self.backend.get_flag(RenderFlag.BODYBVH)
            or self.backend.get_flag(RenderFlag.MESHBVH)
        )
        needs.diagnostics = needs.diagnostics or (
            needs.bvh
            or any(
                self.backend.get_flag(flag)
                for flag in (
                    RenderFlag.ACTUATOR,
                    RenderFlag.JOINT,
                    RenderFlag.COM,
                    RenderFlag.INERTIA,
                    RenderFlag.CAMERA,
                    RenderFlag.LIGHT,
                    RenderFlag.RANGEFINDER,
                    RenderFlag.CONSTRAINT,
                    RenderFlag.AUTOCONNECT,
                )
            )
            or label_mode
            in (
                LabelMode.JOINT,
                LabelMode.ACTUATOR,
                LabelMode.CONSTRAINT,
                LabelMode.CAMERA,
                LabelMode.LIGHT,
            )
            or frame_mode in (FrameMode.CAMERA, FrameMode.LIGHT)
        )
        return needs

    def _sync_structure(self) -> None:
        gen = self.session.structure_generation
        if gen != self._structure_generation:
            self._structure_generation = gen
            self.backend.set_scene(self.session.source)

    def _current_viewport_render_size(self) -> tuple[int, int]:
        target = getattr(self.backend, "target", None)
        if target is not None and hasattr(target, "width") and hasattr(target, "height"):
            return max(1, int(target.width)), max(1, int(target.height))
        image = self._viewport_image
        if image is not None:
            return max(1, int(image.width)), max(1, int(image.height))
        width, height = self.window.points_to_pixels(self._viewport_panel_size)
        return max(1, int(width)), max(1, int(height))

    def _sync_viewport_size(self) -> None:
        if self._fixed_render_size is not None:
            self.backend.resize(*self._fixed_render_size)
            self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])
        else:
            settled = self.window.poll_render_size(self._viewport_panel_size)
            if settled is not None:
                sw, sh = settled
                self.backend.resize(sw, sh)
                self.camera.set_aspect(max(sw, 1) / max(sh, 1))
        self._viewport_rect = _fit_image_rect(
            self._viewport_panel_position,
            self._viewport_panel_size,
            self._current_viewport_render_size(),
        )

    def _sync_display_scale(self) -> None:
        generation = self.window.scale_generation
        if generation == self._display_scale_generation:
            return
        configure_text = getattr(self.backend, "configure_text", None)
        if configure_text is not None:
            font = self.window.font_report
            configure_text(
                font.mono_path,
                font.mono_index,
                font.cjk_path,
                font.cjk_index,
                self.window.config.font_size_pt * self.window.ui_scale,
            )
        self._display_scale_generation = generation

    def _panel_context(self) -> PanelContext:
        return PanelContext(
            session=self.session,
            backend=self.backend,
            camera=self.camera,
            model_camera_id=self._model_camera_id,
            model_camera_view=self._model_camera_view,
            select_model_camera=self._select_model_camera_animated,
            tracking=self.camera_tracker.config,
            tracking_node_id=self.tracking_node_id,
            track_node=self.track_node,
            set_camera_tracking=self.set_camera_tracking,
            focus_node=self.request_node_focus,
            focus_joint=self.request_joint_focus,
            request_rename=self.request_rename,
            request_model_rename=self.request_model_rename,
            request_texture_import=self._open_texture_dialog,
            request_geometry_resource_import=self._open_geometry_resource_dialog,
            request_model_asset_import=self._open_model_asset_import_dialog,
            request_model_asset_replace=self._open_model_asset_replace_dialog,
            queue_model_edit=self._queue_model_edit,
            model_keyframe_names=self.model_edits.model_keyframe_names,
            live_model_updates=self.live_model_updates,
            set_live_model_updates=self.set_live_model_updates,
            gizmo=self.gizmo,
            view_cube=self.view_cube,
            perturb=self.perturb,
            scene_entities=self.scene_entities,
            camera_preview=self.camera_preview,
            panels=self.panels,
            theme=self.theme,
            style_scale=self.window.style_scale,
            viewport_rect=self._viewport_rect,
            dt=self._dt,
            status=self.session.last_message,
            popup_owned_frame=self._popup_owned_frame,
            language=self.localizer.language.value,
            translate=self.localizer.text,
            set_language=self.set_language,
            set_shadow_quality=self.set_shadow_quality,
            interactions=self.interactions,
            set_interactions=self.set_interactions,
            selection_style=self.selection_style,
            set_selection_style=self.set_selection_style,
            set_precise_input_memory=self.set_precise_input_choice_memory,
            set_view_selection_padding=self.set_view_selection_padding,
            viewport_overlay_scale=self._viewport_overlay_scale,
            set_viewport_overlay_scale=self.set_viewport_overlay_scale,
            viewport_overlays=self.viewport_overlays,
            viewport_layers=self.viewport_layers,
            set_viewport_layers=self.set_viewport_layers,
            recording_config=self.recording_config,
            set_recording_config=self.set_recording_config,
            recording=self.recording,
            take_video_active=self._take_video is not None,
            start_take_video=self.start_take_video,
            stop_recording=self.stop_recording,
            set_viewport_overlays=self.set_viewport_overlays,
            set_viewport_capsule_scale=self.set_viewport_capsule_scale,
            input_bindings=self.input_bindings,
            set_input_binding=self.set_input_binding,
            set_pointer_binding=self.set_pointer_binding,
            set_navigation_preset=self.set_navigation_preset,
            input_claim=self._input_claim,
            reset_input_bindings=self.reset_input_bindings,
            font_report=self.window.font_report,
            output=self.output,
        )

    def _cursor_ray(self, cursor: tuple[float, float]):
        ndc = ndc_from_viewport(cursor[0], cursor[1], self._viewport_rect)
        return unproject(self._camera_view(), *ndc)

    def _node_pose(self, node) -> tuple[np.ndarray, np.ndarray]:
        return node_world_pose(self.session, node)
