"""Main viewer UI loop and panel coordination."""

from __future__ import annotations

import os
import re
import sys
import time
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs

from .. import commands as cmd
from ..adapters.base import FrameNeeds, NodeType
from ..gizmo import axis_active_color, axis_hover_color
from ..log import add_output_sink, get_logger, remove_output_sink
from ..render.backend import FrameMode, LabelMode, RenderFlag
from ..types import Light, LightType, MeshShape, ViewportImage
from ..workspace_io import (
    MissingResource,
    missing_resource_entries,
    relocate_workspace_resource,
    repair_workspace_resources,
)
from . import gestures as gs
from .camera import (
    ISO_PITCH,
    CameraOut,
    OrbitCamera,
    ProjectionTransition,
    camera_basis,
    closest_perpendicular_view_direction,
    elevated_focus_view_direction,
    ndc_from_viewport,
    oblique_axis_view_directions,
    unproject,
)
from .camera_preview import CameraPreview
from .draw2d import ImguiDraw2D
from .gizmo import JointLimitHit, ObjectGizmo, PreciseGizmoInput, node_world_pose
from .input_bindings import DEFAULT_INPUT_BINDINGS, InputAction, InputBindings
from .localization import Localizer
from .messages import OutputBuffer
from .panels import (
    PanelContext,
    PanelSet,
    button_width,
    segmented_control,
)
from .perturb import (
    PerturbController,
    cursor_grab_point,
    draw_fallback,
)
from .scene_entities import SceneEntityHelpers
from .theme import THEME, Theme
from .viewcube import DEFAULT_SELECTION_PADDING, ViewCube
from .viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    HINT_CHROME_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
    OVERLAY_CLIP_PADDING,
    OVERLAY_GEOMETRY,
    PLAYBACK_CHROME_SCALE,
    TOOL_CHROME_SCALE,
    ToolHint,
    ViewportChromeRegistry,
    default_tool_hints,
    draw_playback,
    draw_scene_tool_hints,
    draw_status,
    draw_tool_column,
    fitting_tool_hints,
    localized_viewport_labels,
    playback_size,
    tool_column_size,
    tool_hints_size,
    viewport_chrome_scale,
)
from .window import Window, WindowConfig

if TYPE_CHECKING:
    from ..adapters.base import SceneNode
    from ..commands import CommandResult
    from ..session import Session

CLICK_SLOP_PT = 4.0
PRECISE_GIZMO_WIDTH_PT = 204.0
JOINT_LIMIT_LABEL_DELAY_SECONDS = 0.5
JOINT_LIMIT_HOVER_GRACE_SECONDS = 0.12
PRECISE_GIZMO_HINT_DELAY_SECONDS = 0.5
JOINT_FOCUS_MARGIN = 1.5
JOINT_FOCUS_OBLIQUE_DEGREES = 35.0
VIEWPORT_DOUBLE_CLICK_SECONDS = 0.3
VIEWPORT_DOUBLE_CLICK_RADIUS_PT = 6.0
STEP_BACK_REPEAT_DELAY_SECONDS = 0.35
STEP_BACK_REPEAT_RATE_SECONDS = 0.1
log = get_logger("ui")


PICK_SCREEN_RADIUS_PT = 40.0

MODEL_EXTENSIONS = frozenset((".xml", ".mjcf", ".urdf"))
MODEL_FILTERS = [
    "All supported models (*.xml, *.mjcf, *.urdf)",
    "*.xml *.mjcf *.urdf",
    "MuJoCo XML / MJCF (*.xml, *.mjcf)",
    "*.xml *.mjcf",
    "URDF (*.urdf)",
    "*.urdf",
    "All files",
    "*",
]
SCENE_SUFFIX = ".mojive.json"
LEGACY_SCENE_SUFFIX = ".forge.json"
SCENE_SUFFIXES = (SCENE_SUFFIX, LEGACY_SCENE_SUFFIX)
SCENE_FILTERS = [
    "Mojive scenes (*.mojive.json, *.forge.json)",
    "*.mojive.json *.forge.json",
    "MuJoCo XML / MJCF (*.xml, *.mjcf)",
    "*.xml *.mjcf",
    "All files",
    "*",
]
IMAGE_FILTERS = [
    "PNG images (*.png)",
    "*.png",
    "All files",
    "*",
]
MESH_FILTERS = [
    "MuJoCo mesh files (*.stl, *.obj, *.msh, *.ply)",
    "*.stl *.obj *.msh *.ply",
    "All files",
    "*",
]


def precise_input_status_hints(edit: PreciseGizmoInput, translate) -> tuple[ToolHint, ...]:
    """Return only keyboard actions that the precise-input popup implements."""

    hints = [
        ToolHint("key", "Enter", translate("Apply"), hint_id="precise.apply"),
        ToolHint("key", "Esc", translate("Cancel"), hint_id="precise.cancel"),
    ]
    if edit.unit == "°":
        hints.append(
            ToolHint(
                "key",
                "U",
                translate("Switch angle unit"),
                hint_id="precise.angle-unit",
            )
        )
    return tuple(hints)


def _translated_file_filters(filters: list[str], translate) -> list[str]:
    """Translate file-dialog descriptions while preserving their glob entries."""

    return [translate(value) if index % 2 == 0 else value for index, value in enumerate(filters)]


def _fit_image_rect(
    position: tuple[float, float],
    available: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Aspect-fit a render target inside its current viewport panel."""

    x, y = float(position[0]), float(position[1])
    width, height = max(float(available[0]), 1.0), max(float(available[1]), 1.0)
    image_width, image_height = max(int(image_size[0]), 1), max(int(image_size[1]), 1)
    scale = min(width / image_width, height / image_height)
    fitted_width = image_width * scale
    fitted_height = image_height * scale
    return (
        x + (width - fitted_width) * 0.5,
        y + (height - fitted_height) * 0.5,
        fitted_width,
        fitted_height,
    )


def _scene_save_target(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.name.endswith(SCENE_SUFFIXES) or target.suffix.lower() in {".xml", ".mjcf"}:
        return target
    return target.with_name(target.name + SCENE_SUFFIX)


def _prepare_modal(width_pt: float, style_scale: float = 1.0) -> None:
    """Keep blocking prompts readable and centered as the host window resizes."""

    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        viewport.get_center(),
        imgui.Cond_.always.value,
        imgui.ImVec2(0.5, 0.5),
    )
    # ``width_pt`` is an authored logical size while ImGui layout coordinates
    # follow the platform style scale. This differs from framebuffer scaling:
    # an Ubuntu 2x override enlarges the font and must enlarge the dialog too.
    scale = max(float(style_scale), 1e-6)
    margin = 32.0 * scale
    width = min(float(width_pt) * scale, max(1.0, float(viewport.work_size.x) - margin))
    max_height = max(1.0, float(viewport.work_size.y) - margin)
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(width, 0.0),
        imgui.ImVec2(width, max_height),
    )


def _clipped_overlay_host_rect(
    viewport_rect: tuple[float, float, float, float],
    content_rect: tuple[float, float, float, float],
    padding: float,
) -> tuple[float, float, float, float] | None:
    """Intersect one overlay host with the viewport while retaining its authored origin."""

    viewport_x, viewport_y, viewport_width, viewport_height = viewport_rect
    content_x0, content_y0, content_x1, content_y1 = content_rect
    pad = max(0.0, float(padding))
    x0 = max(float(viewport_x), float(content_x0) - pad)
    y0 = max(float(viewport_y), float(content_y0) - pad)
    x1 = min(float(viewport_x + viewport_width), float(content_x1) + pad)
    y1 = min(float(viewport_y + viewport_height), float(content_y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


@contextmanager
def _clipped_overlay_draw(
    viewport_rect: tuple[float, float, float, float],
):
    """Yield the current window draw list with a hard viewport clip rect."""

    x, y, width, height = viewport_rect
    imgui.push_clip_rect(
        imgui.ImVec2(x, y),
        imgui.ImVec2(x + width, y + height),
        True,
    )
    try:
        yield ImguiDraw2D(imgui.get_window_draw_list())
    finally:
        imgui.pop_clip_rect()


@contextmanager
def _clipped_foreground_overlay_draw(
    viewport_rect: tuple[float, float, float, float],
):
    """Yield the top-most draw list while retaining the hard viewport clip."""

    x, y, width, height = viewport_rect
    draw_list = imgui.get_foreground_draw_list()
    draw_list.push_clip_rect(
        imgui.ImVec2(x, y),
        imgui.ImVec2(x + width, y + height),
        True,
    )
    try:
        yield ImguiDraw2D(draw_list)
    finally:
        draw_list.pop_clip_rect()


def _equal_modal_buttons(
    labels: tuple[str, ...],
    theme: Theme,
    *,
    primary: int = -1,
) -> tuple[bool, ...]:
    """Draw a full-width modal action row with equal, predictable targets."""

    spacing = float(imgui.get_style().item_spacing.x)
    available = float(imgui.get_content_region_avail().x)
    width = max(1.0, (available - spacing * max(0, len(labels) - 1)) / max(1, len(labels)))
    clicked: list[bool] = []
    for index, label in enumerate(labels):
        if index:
            imgui.same_line()
        clicked.append(
            _primary_button(label, width, theme)
            if index == primary
            else bool(imgui.button(label, imgui.ImVec2(width, 0.0)))
        )
    return tuple(clicked)


def _primary_button(label: str, width: float, theme: Theme) -> bool:
    """Draw the committing action with the shared selected-control colors."""

    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*theme.bg_frame_active))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*theme.primary_bright))
    clicked = imgui.button(label, imgui.ImVec2(width, 0.0))
    imgui.pop_style_color(2)
    return clicked


def _middle_elide_text(value: str, max_width: float, measure) -> str:
    """Keep both ends of a dynamic name inside one fixed-width line."""

    value = str(value)
    if measure(value) <= max_width:
        return value
    ellipsis = "…"
    for count in range(max(0, len(value) - 1), -1, -1):
        left = (count + 1) // 2
        right = count // 2
        candidate = f"{value[:left]}{ellipsis}{value[len(value) - right :]}"
        if measure(candidate) <= max_width:
            return candidate
    return ellipsis


def _toggle_angle_input(value: float, unit: str) -> tuple[float, str]:
    """Change the displayed unit without changing the represented angle."""

    if unit == "degrees":
        return float(np.radians(value)), "radians"
    return float(np.degrees(value)), "degrees"


def _compact_status_for_selection(message: str, selected: str) -> str:
    """Remove selection wording already represented by the status bar's first field."""

    message = " ".join(str(message).split())
    selected = " ".join(str(selected).split())
    if not message:
        return ""
    if selected == "no selection" and message.casefold() == "selection cleared":
        return ""
    if message.casefold() in {
        selected.casefold(),
        f"selected {selected}".casefold(),
        f"{selected} selected".casefold(),
    }:
        return ""
    prefix = re.match(rf"^{re.escape(selected)}\s*(?:[|:·—-]\s*)?(.*)$", message, re.I)
    if prefix is not None:
        remainder = prefix.group(1).strip()
        return "" if remainder.casefold() == "selected" else remainder
    return message


_STATUS_ACTION_PREFIXES = (
    "saved ",
    "opened ",
    "loaded ",
    "added ",
    "removed ",
    "renamed ",
    "duplicated ",
    "imported ",
    "applied ",
    "undo ",
    "redo ",
    "recorded ",
    "recording ",
    "replaying ",
    "cleared ",
    "scene reset",
    "scene reloaded",
    "scene state restored",
    "new scene",
    "cancelled ",
    "model placement unlocked",
    "viewport ",
)


def _status_message_for_bar(message: str, selected: str, level: str) -> str:
    """Keep only actionable results and diagnostics in the transient status slot."""

    compact = _compact_status_for_selection(message, selected)
    if not compact:
        return ""
    severity = str(level).casefold()
    if severity in {"error", "warning", "success"}:
        return compact
    folded = compact.casefold()
    return compact if folded.startswith(_STATUS_ACTION_PREFIXES) else ""


def _rectangles_overlap(a, b) -> bool:
    return bool(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def _simulation_timestep(adapter, *, loading: bool = False) -> float:
    """Return the fixed scene step, never the variable UI frame duration."""

    return 0.0 if loading else float(adapter.timestep())


@dataclass
class Keys:
    fly: tuple[float, float, float] = (0.0, 0.0, 0.0)
    toggle_pause: bool = False
    step_back_count: int = 0
    clear_selection: bool = False
    frame_scene: bool = False
    gizmo_translate: bool = False
    gizmo_rotate: bool = False
    gizmo_space: bool = False
    gizmo_axis: int = -1


@dataclass(frozen=True)
class _ModelLoadJob:
    action: str
    path: Path
    command: Any


@dataclass
class _FrameRateDisplay:
    """Low-pass and rate-limit the status-bar FPS readout."""

    smoothed_dt: float = 1.0 / 60.0
    value: float = 60.0
    elapsed: float = 0.0

    def update(self, dt: float) -> float:
        dt = min(0.1, max(float(dt), 1e-6))
        # A half-second time constant follows sustained performance changes
        # without turning normal frame-time jitter into flashing text.
        alpha = 1.0 - float(np.exp(-dt / 0.5))
        self.smoothed_dt += alpha * (dt - self.smoothed_dt)
        self.elapsed += dt
        if self.elapsed >= 0.25:
            self.value = 1.0 / max(self.smoothed_dt, 1e-6)
            self.elapsed %= 0.25
        return self.value


@dataclass
class _JointLimitHoverState:
    """Apply a reveal delay and short dropout grace to endpoint hover."""

    key: tuple[int, int, str] | None = None
    entered_at: float = 0.0
    last_seen_at: float = 0.0

    def update(
        self,
        hovered_key: tuple[int, int, str] | None,
        available_keys: tuple[tuple[int, int, str], ...],
        now: float,
    ) -> tuple[int, int, str] | None:
        now = float(now)
        if self.key not in available_keys:
            self.reset()
        if hovered_key is not None:
            if hovered_key != self.key:
                self.entered_at = now
            self.key = hovered_key
            self.last_seen_at = now
        elif self.key is not None and now - self.last_seen_at > JOINT_LIMIT_HOVER_GRACE_SECONDS:
            self.reset()
        if self.key is None or now - self.entered_at < JOINT_LIMIT_LABEL_DELAY_SECONDS:
            return None
        return self.key

    def reset(self) -> None:
        self.key = None
        self.entered_at = 0.0
        self.last_seen_at = 0.0


@dataclass
class _GizmoHintHoverState:
    """Reveal typed-input help only after an uninterrupted actionable hover."""

    entered_at: float | None = None
    visible: bool = False

    def update(self, hovered: bool, now: float) -> bool:
        now = float(now)
        if not hovered:
            self.reset()
            return False
        if self.entered_at is None:
            self.entered_at = now
        self.visible = now - self.entered_at >= PRECISE_GIZMO_HINT_DELAY_SECONDS
        return self.visible

    def reset(self) -> None:
        self.entered_at = None
        self.visible = False


class ViewerApp:
    def __init__(
        self,
        session: Session,
        backend: Any,
        window: Window | None = None,
        *,
        title: str = "Mojive",
        theme: Theme | None = None,
        debug_bridge: Any | None = None,
    ) -> None:
        self.session = session
        self.backend = backend
        self.window = window
        self.title = title
        self.theme = theme or THEME
        self.debug_bridge = debug_bridge
        self.localizer = Localizer.load()
        self._viewport_labels = localized_viewport_labels(self.localizer.text)
        metric_mode = self.localizer.preference("status_metric", "time")
        self._status_metric_mode = "steps" if metric_mode == "steps" else "time"
        self._panel_status_hints: tuple[ToolHint, ...] = ()
        self._status_panel = "Viewport"
        self.camera = OrbitCamera()

        self.camera_out = CameraOut(backend=backend, session=session)
        self.camera.attach(self.camera_out)
        self.camera_preview = CameraPreview()
        self.gizmo = ObjectGizmo()
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
        self.panels = PanelSet()
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
        self._close_after_model_load = False
        self._model_drop_notice = ""
        self._model_drop_notice_until = 0.0
        self._display_scale_generation = -1
        self._output_sink_id: int | None = None
        self._seen_message_revision = int(getattr(session, "message_revision", 0))
        self._snap_latched = False
        self._capture_viewport_requested = False
        self._viewport_recorder: Any | None = None
        self._viewport_recording_path: Path | None = None
        self._viewport_record_elapsed = 0.0
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

    def set_input_binding(self, action: InputAction, key_id: str) -> None:
        """Atomically remap one viewport action and persist the whole map."""

        self.input_bindings = self.input_bindings.remap(action, key_id)
        self.localizer.set_preferences({"input_bindings": self.input_bindings.preferences()})

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
        self._sync_structure()
        self._reset_source_camera()
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
        executor = getattr(self, "_model_load_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._model_load_executor = None
            self._model_load_future = None
            self._model_load_job = None
            self._model_load_queue.clear()
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

    def load_model(self, path: str | Path) -> CommandResult:
        result = self.session.submit(cmd.LoadAsset(Path(path)))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(
                f"{self.localizer.text('Loaded')} {self.session.asset_path.name}"
            )
        else:
            self._report_model_error(result.message)
        return result

    def add_model(
        self, path: str | Path, position: tuple[float, float, float] | None = None
    ) -> CommandResult:
        location = position or tuple(float(value) for value in self.camera.pivot)
        result = self.session.submit(cmd.AddSceneModel(Path(path), location))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def remove_model(self, model_id: int) -> CommandResult:
        result = self.session.submit(cmd.RemoveSceneModel(model_id))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def open_scene(self, path: str | Path) -> CommandResult:
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() in MODEL_EXTENSIONS:
            return self.load_model(target)
        try:
            missing = missing_resource_entries(target)
        except Exception:
            missing = ()
        if missing:
            self._begin_resource_repair(target, missing)
            return cmd.CommandResult.bad(
                f"{len(missing)} workspace resource(s) must be located before opening"
            )
        result = self.session.submit(cmd.OpenScene(target))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(
                f"{self.localizer.text('Opened')} {self.session.asset_path.name}"
            )
        else:
            self._report_model_error(result.message)
        return result

    def _queue_scene_open(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        if target.suffix.lower() in MODEL_EXTENSIONS:
            self._queue_model_load("load", target)
            return
        try:
            missing = missing_resource_entries(target)
        except Exception:
            missing = ()
        if missing:
            self._begin_resource_repair(target, missing)
            return
        self._queue_model_load("open", target)

    def _queue_model_load(
        self,
        action: str,
        path: str | Path,
        position: tuple[float, float, float] | None = None,
    ) -> None:
        target = Path(path).expanduser().resolve()
        if action == "add":
            command = cmd.AddSceneModel(target, position or tuple(self.camera.pivot))
        elif action == "open":
            command = cmd.OpenScene(target)
        elif action == "reload":
            command = cmd.Reload()
        else:
            command = cmd.LoadAsset(target)
        self._model_load_queue.append(_ModelLoadJob(action, target, command))

    def _start_model_load(self) -> bool:
        if self._model_load_future is not None or not self._model_load_queue:
            return self._model_load_future is not None
        if self.session.adapter.caps.simulation and not self.session.paused:
            paused = self.session.submit(cmd.Pause())
            if not paused.ok:
                job = self._model_load_queue.pop(0)
                self._model_load_queue.clear()
                self._report_model_error(
                    f"Cannot {self._model_load_verb(job.action).lower()} while physics is running: "
                    f"{paused.message}"
                )
                return False
        if self._model_load_executor is None:
            self._model_load_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mojive-model-loader",
            )
        job = self._model_load_queue.pop(0)
        self._model_load_job = job
        self._model_load_started = time.monotonic()
        log.info("{} {}", self._model_load_verb(job.action), job.path)
        self._model_load_future = self._model_load_executor.submit(self.session.submit, job.command)
        return True

    def _poll_model_load(self) -> bool:
        future = self._model_load_future
        job = self._model_load_job
        if future is None or job is None:
            return False
        if not future.done():
            return True
        elapsed = time.monotonic() - self._model_load_started
        try:
            result = future.result()
        except Exception as exc:
            result = cmd.CommandResult.bad(str(exc))
        self._model_load_future = None
        self._model_load_job = None
        log.info("{} {} in {:.3f}s", result.message or "Finished", job.path, elapsed)
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(result.message)
        else:
            self._model_load_queue.clear()
            self._report_model_error(result.message)
        if self._model_load_queue:
            self._start_model_load()
            return True
        if self._close_after_model_load:
            self._close_after_model_load = False
            self._request_document_action("quit")
        return False

    @staticmethod
    def _model_load_verb(action: str) -> str:
        return {
            "add": "Adding model",
            "open": "Opening scene",
            "reload": "Reloading model",
        }.get(action, "Loading model")

    def save_scene(
        self, path: str | Path, *, current_pose_keyframe: str | None = None
    ) -> CommandResult:
        target = _scene_save_target(path)
        result = self.session.submit(cmd.SaveScene(target, current_pose_keyframe))
        if result.ok:
            self._set_model_drop_notice(result.message)
        else:
            self._report_model_error(result.message)
        return result

    def _request_scene_save(
        self,
        path: str | Path,
        pending: tuple[str, Path | None] | None = None,
    ) -> None:
        target = _scene_save_target(path)
        if target.suffix.lower() in {".xml", ".mjcf"} and self.session.current_pose_modified:
            self._pending_pose_save = (target, pending)
            return
        if self.save_scene(target).ok and pending is not None:
            self._execute_document_action(*pending)

    def _after_model_change(self) -> None:
        self.router.abort()
        self.gizmo.cancel()
        self.gizmo.cancel_model_placement(self.session)
        self._model_camera_id = -1
        self._model_camera_view = None
        self._model_camera_projection_target = None
        self._pending_node_focus_id = None
        self._pending_joint_focus_id = None
        self._last_viewport_click = None
        self._gizmo_hint_hover.reset()
        self._structure_generation = -1
        self._sync_structure()
        self._reset_source_camera()

    def _open_model_dialog(self, action: str = "open") -> None:
        if self._model_dialog is not None:
            return
        t = self.localizer.text
        current = self.session.asset_path
        default_path = str(current.parent if current is not None else Path.cwd())
        self._model_dialog = portable_file_dialogs.open_file(
            t("Add MJCF or URDF models") if action == "add" else t("Open an MJCF or URDF model"),
            default_path,
            _translated_file_filters(MODEL_FILTERS, t),
            portable_file_dialogs.opt.multiselect
            if action == "add"
            else portable_file_dialogs.opt.none,
        )
        self._model_dialog_action = action
        self._set_model_drop_notice(
            t("Choose a model to add") if action == "add" else t("Choose an MJCF or URDF model")
        )

    def _open_scene_dialog(self, action: str) -> None:
        if self._scene_dialog is not None:
            return
        t = self.localizer.text
        current = self.session.asset_path
        if action == "save":
            default = current or (Path.cwd() / f"scene{SCENE_SUFFIX}")
            self._scene_dialog = portable_file_dialogs.save_file(
                t("Save scene"), str(default), _translated_file_filters(SCENE_FILTERS, t)
            )
        else:
            default = current.parent if current is not None else Path.cwd()
            self._scene_dialog = portable_file_dialogs.open_file(
                t("Open Mojive scene"),
                str(default),
                _translated_file_filters(SCENE_FILTERS, t),
            )
        self._scene_dialog_action = action

    def _open_resource_dialog(self) -> None:
        if self._resource_dialog is not None:
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        self._resource_dialog = portable_file_dialogs.select_folder(
            self.localizer.text("Add Mojive resource directory"), str(default)
        )

    def _open_texture_dialog(
        self, model_id: int, material_index: int = -1, texture_type: str = "2d"
    ) -> None:
        if self._texture_dialog is not None:
            return
        kind = str(texture_type).strip().lower()
        if kind not in ("2d", "cube", "skybox"):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        label = "2D" if kind == "2d" else kind
        self._texture_dialog = portable_file_dialogs.open_file(
            f"{self.localizer.text('Import')} "
            f"{self.localizer.text(label)} {self.localizer.text('texture')}",
            str(default),
            _translated_file_filters(IMAGE_FILTERS, self.localizer.text),
        )
        self._texture_import_target = (int(model_id), int(material_index), kind)

    def _poll_texture_dialog(self) -> None:
        dialog = self._texture_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._texture_dialog = None
        model_id, material_index, texture_type = self._texture_import_target
        self._texture_import_target = (-1, -1, "2d")
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or "texture"
        prefix = f"opengl_{model_id}_"
        existing = {
            name.removeprefix(prefix) for name in self.session.model_texture_names(model_id)
        }
        name = base
        suffix = 2
        while name in existing:
            name = f"{base}{suffix}"
            suffix += 1
        result = self.session.submit(
            cmd.ImportModelTexture(model_id, path, name, material_index, texture_type)
        )
        if not result.ok:
            self._report_model_error(result.message)

    def _open_geometry_resource_dialog(self, node_id: int, resource_type: str) -> None:
        if self._geometry_resource_dialog is not None:
            return
        kind = str(resource_type).strip().lower()
        if kind not in ("mesh", "hfield"):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        filters = MESH_FILTERS if kind == "mesh" else IMAGE_FILTERS
        title = self.localizer.text("Import mesh" if kind == "mesh" else "Import PNG height field")
        self._geometry_resource_dialog = portable_file_dialogs.open_file(
            title,
            str(default),
            _translated_file_filters(filters, self.localizer.text),
        )
        self._geometry_resource_import_target = (int(node_id), kind)

    def _poll_geometry_resource_dialog(self) -> None:
        dialog = self._geometry_resource_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._geometry_resource_dialog = None
        node_id, resource_type = self._geometry_resource_import_target
        self._geometry_resource_import_target = (-1, "")
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or resource_type
        properties = self.session.geometry_shape_properties(node_id)
        if properties is None:
            self._report_model_error(
                self.localizer.text("The target geometry is no longer available")
            )
            return
        existing = set(
            properties.mesh_names if resource_type == "mesh" else properties.height_field_names
        )
        name = base
        suffix = 2
        while name in existing:
            name = f"{base}{suffix}"
            suffix += 1
        result = self.session.submit(
            cmd.ImportModelGeometryResource(node_id, resource_type, path, name)
        )
        if not result.ok:
            self._report_model_error(result.message)

    def _open_model_asset_import_dialog(
        self,
        model_id: int,
        asset_type: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self._open_model_asset_dialog("import", model_id, asset_type, "", fields)

    def _open_model_asset_replace_dialog(self, model_id: int, asset_type: str, name: str) -> None:
        self._open_model_asset_dialog("replace", model_id, asset_type, name, ())

    def _open_model_asset_dialog(
        self,
        action: str,
        model_id: int,
        asset_type: str,
        name: str,
        fields: tuple[tuple[str, str], ...],
    ) -> None:
        if self._model_asset_dialog is not None:
            return
        kind = str(asset_type).strip().lower()
        if kind not in ("mesh", "hfield", "texture") or action not in (
            "import",
            "replace",
        ):
            return
        current = self.session.asset_path
        default = current.parent if current is not None else Path.cwd()
        filters = MESH_FILTERS if kind == "mesh" else IMAGE_FILTERS
        verb = self.localizer.text("Import" if action == "import" else "Replace")
        label = (
            "mesh" if kind == "mesh" else "PNG height field" if kind == "hfield" else "PNG texture"
        )
        self._model_asset_dialog = portable_file_dialogs.open_file(
            f"{verb} {self.localizer.text(label)}",
            str(default),
            _translated_file_filters(filters, self.localizer.text),
        )
        self._model_asset_dialog_target = (
            action,
            int(model_id),
            kind,
            str(name),
            tuple(fields),
        )

    def _poll_model_asset_dialog(self) -> None:
        dialog = self._model_asset_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._model_asset_dialog = None
        action, model_id, asset_type, name, fields = self._model_asset_dialog_target
        self._model_asset_dialog_target = ("", -1, "", "", ())
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if action == "replace":
            result = self.session.submit(
                cmd.ReplaceModelAssetFile(model_id, asset_type, name, path)
            )
        else:
            base = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or asset_type
            existing = {
                item.name for item in self.session.model_assets(model_id) if item.type == asset_type
            }
            name = base
            suffix = 2
            while name in existing:
                name = f"{base}{suffix}"
                suffix += 1
            result = self.session.submit(
                cmd.ImportModelAsset(model_id, asset_type, path, name, fields)
            )
        if not result.ok:
            self._report_model_error(result.message)

    def _begin_resource_repair(self, path: Path, missing: tuple[MissingResource, ...]) -> None:
        self._resource_repair_path = path
        self._missing_resources = missing
        self._resource_repair_status = ""
        self._open_resource_repair_popup = True

    def _open_resource_repair_dialog(self, action: str, model_index: int = -1) -> None:
        if self._resource_repair_dialog is not None or self._resource_repair_path is None:
            return
        default = self._resource_repair_path.parent
        if action == "locate":
            missing = next(
                (item for item in self._missing_resources if item.model_index == model_index), None
            )
            if missing is None:
                return
            self._resource_repair_dialog = portable_file_dialogs.open_file(
                f"{self.localizer.text('Locate')} {missing.model_name}",
                str(default),
                _translated_file_filters(MODEL_FILTERS, self.localizer.text),
            )
        else:
            self._resource_repair_dialog = portable_file_dialogs.select_folder(
                self.localizer.text("Search a directory for missing resources"), str(default)
            )
        self._resource_repair_dialog_action = action
        self._resource_repair_model_index = model_index

    def _poll_resource_repair_dialog(self) -> None:
        dialog = self._resource_repair_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._resource_repair_dialog_action
        model_index = self._resource_repair_model_index
        self._resource_repair_dialog = None
        self._resource_repair_dialog_action = ""
        self._resource_repair_model_index = -1
        try:
            selected = dialog.result()
        except Exception as exc:
            self._resource_repair_status = str(exc)
            self._open_resource_repair_popup = True
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            self._open_resource_repair_popup = True
            return
        path = self._resource_repair_path
        if path is None:
            return
        try:
            if action == "locate":
                repair = relocate_workspace_resource(path, model_index, selected)
            else:
                repair = repair_workspace_resources(path, selected)
        except Exception as exc:
            self._resource_repair_status = str(exc)
            self._open_resource_repair_popup = True
            return
        self._missing_resources = repair.missing
        if repair.missing:
            self._resource_repair_status = (
                f"Repaired {repair.repaired}; {len(repair.missing)} resource(s) still missing."
            )
            self._open_resource_repair_popup = True
            return
        self._resource_repair_path = None
        self._resource_repair_status = ""
        self._set_model_drop_notice(f"Repaired {repair.repaired} resource path(s)")
        self._queue_scene_open(path)

    def _poll_resource_dialog(self) -> None:
        dialog = self._resource_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._resource_dialog = None
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if selected:
            result = self.session.submit(cmd.AddResourceRoot(Path(selected)))
            if not result.ok:
                self._report_model_error(result.message)

    def _draw_resource_repair(self) -> None:
        t = self.localizer.text
        popup_title = f"{t('Missing Resources')}###Missing Resources"
        if self._open_resource_repair_popup:
            imgui.open_popup(popup_title)
            self._open_resource_repair_popup = False
        if imgui.is_popup_open(popup_title):
            _prepare_modal(480.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(
            t(
                "This Mojive scene references model files that are no longer available. Locate "
                "files individually or search one directory to repair every unambiguous path."
            )
        )
        imgui.spacing()
        locate = -1
        for missing in self._missing_resources:
            imgui.text(f"{missing.model_name}: {missing.reference}")
            imgui.same_line()
            if imgui.small_button(f"{t('Locate...')}##missing-resource-{missing.model_index}"):
                locate = missing.model_index
        if self._resource_repair_status:
            imgui.spacing()
            imgui.text_wrapped(self._resource_repair_status)
            if imgui.small_button(f"{t('Copy details')}##resource-repair"):
                imgui.set_clipboard_text(self._resource_repair_status)
        imgui.spacing()
        search = imgui.button(t("Search Directory..."), imgui.ImVec2(160.0, 0.0))
        imgui.same_line()
        cancel = imgui.button(t("Cancel"), imgui.ImVec2(100.0, 0.0))
        if locate >= 0:
            self._open_resource_repair_dialog("locate", locate)
            imgui.close_current_popup()
        elif search:
            self._open_resource_repair_dialog("search")
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._resource_repair_path = None
            self._missing_resources = ()
            self._resource_repair_status = ""
            imgui.close_current_popup()
        imgui.end_popup()

    def _poll_scene_dialog(self) -> None:
        dialog = self._scene_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._scene_dialog_action
        self._scene_dialog = None
        self._scene_dialog_action = ""
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            self._after_save_action = None
            return
        if isinstance(selected, (list, tuple)):
            selected = selected[0] if selected else ""
        if not selected:
            self._after_save_action = None
            return
        if action == "save":
            pending = self._after_save_action
            self._after_save_action = None
            self._request_scene_save(selected, pending)
        else:
            self._request_document_action("open_scene", Path(selected))

    def _poll_model_dialog(self) -> None:
        dialog = self._model_dialog
        if dialog is None or not dialog.ready(0):
            return
        action = self._model_dialog_action
        self._model_dialog = None
        self._model_dialog_action = ""
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if not selected:
            return
        if action == "add":
            position = tuple(float(value) for value in self.camera.pivot)
            for path in selected:
                self._queue_model_load("add", path, position)
        else:
            self._queue_model_load("load", selected[0])

    def _poll_model_drop(self) -> None:
        paths = self.window.consume_file_drops()
        if not paths:
            return
        if len(paths) == 1 and paths[0].name.endswith(SCENE_SUFFIXES):
            path = paths[0]
            if not self.session.adapter.caps.scene_files:
                self._report_model_error(
                    self.localizer.text("The current workspace cannot open Mojive scene files")
                )
                return
            self._request_document_action("open_scene", path)
            return
        unsupported = next(
            (path for path in paths if path.suffix.lower() not in MODEL_EXTENSIONS), None
        )
        if unsupported is not None:
            self._report_model_error(
                f"{self.localizer.text('Unsupported file')}: {unsupported.name}"
            )
            return
        can_add = self.session.adapter.caps.model_composition
        source = self.session.source
        has_scene_content = source is not None and source.instance_count > 0
        position = tuple(float(value) for value in self.camera.pivot)
        for path in paths:
            if has_scene_content and can_add:
                self._queue_model_load("add", path, position)
            else:
                self._queue_model_load("load", path)
            has_scene_content = True

    def _set_model_drop_notice(self, message: str) -> None:
        self._model_drop_notice = message
        self._model_drop_notice_until = time.monotonic() + 1.8

    def _draw_main_menu(self) -> None:
        t = self.localizer.text
        caps = self.session.adapter.caps
        can_load = bool(caps.asset_loading)
        can_edit = bool(caps.scene_authoring)
        can_scene_files = bool(caps.scene_files)
        shortcut = "Cmd" if sys.platform == "darwin" else "Ctrl"
        new_scene = False
        open_scene = False
        save_scene = False
        save_scene_as = False
        open_model = False
        add_model = False
        remove_model_id = -1
        add_resource_root = False
        remove_resource_root: Path | None = None
        reload_model = False
        undo = False
        redo = False
        open_settings = False
        frame_scene = False
        capture_viewport = False
        toggle_viewport_recording = False
        reset_layout = False
        open_help = False
        open_documentation = False
        open_about = False
        quit_viewer = False
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu(t("File")):
                if can_scene_files:
                    new_scene, _ = imgui.menu_item(t("New Scene"), f"{shortcut}+N", False)
                    open_scene, _ = imgui.menu_item(
                        t("Open Scene..."),
                        f"{shortcut}+O",
                        False,
                        self._scene_dialog is None,
                    )
                    save_scene, _ = imgui.menu_item(
                        t("Save"), f"{shortcut}+S", False, self.session.dirty
                    )
                    save_scene_as, _ = imgui.menu_item(
                        t("Save As..."), f"{shortcut}+Shift+S", False
                    )
                if can_load:
                    if can_scene_files:
                        imgui.separator()
                    open_model, _ = imgui.menu_item(
                        t("Open Model (MJCF / URDF)..."),
                        f"{shortcut}+O" if not can_scene_files else "",
                        False,
                        self._model_dialog is None,
                    )
                    if caps.model_composition:
                        add_model, _ = imgui.menu_item(
                            t("Add Models (MJCF / URDF)..."),
                            "",
                            False,
                            self._model_dialog is None,
                        )
                        removable = [item for item in self.session.scene_models if item.removable]
                        if imgui.begin_menu(t("Remove Model"), bool(removable)):
                            for item in removable:
                                clicked, _ = imgui.menu_item(item.name, "", False)
                                if clicked:
                                    remove_model_id = item.model_id
                            imgui.end_menu()
                    reload_model, _ = imgui.menu_item(
                        t("Reload Model"),
                        f"{shortcut}+Shift+O",
                        False,
                        self.session.asset_path is not None,
                    )
                if can_scene_files and imgui.begin_menu(t("Resource Directories")):
                    add_resource_root, _ = imgui.menu_item(
                        t("Add Directory..."), "", False, self._resource_dialog is None
                    )
                    for root in self.session.adapter.resource_roots:
                        clicked, _ = imgui.menu_item(f"{t('Remove')} {root}", "", False)
                        if clicked:
                            remove_resource_root = root
                    imgui.end_menu()
                imgui.separator()
                quit_viewer, _ = imgui.menu_item(t("Quit"), f"{shortcut}+Q", False, True)
                imgui.end_menu()
            if imgui.begin_menu(t("Edit")):
                undo, _ = imgui.menu_item(
                    t("Undo"),
                    f"{shortcut}+Z",
                    False,
                    caps.edit_history and self.session.can_undo,
                )
                redo, _ = imgui.menu_item(
                    t("Redo"),
                    f"{shortcut}+Shift+Z",
                    False,
                    caps.edit_history and self.session.can_redo,
                )
                imgui.separator()
                open_settings, _ = imgui.menu_item(t("Settings..."), f"{shortcut}+,", False)
                imgui.end_menu()
            self._draw_entity_menu(shortcut, can_edit)
            if imgui.begin_menu(t("View")):
                frame_scene, _ = imgui.menu_item(t("Frame All"), "F", False)
                capture_viewport, _ = imgui.menu_item(
                    t("Capture Viewport"), f"{shortcut}+Shift+P", False
                )
                toggle_viewport_recording, _ = imgui.menu_item(
                    t(
                        "Stop Viewport Recording"
                        if self._viewport_recorder is not None
                        else "Record Viewport"
                    ),
                    f"{shortcut}+Shift+R",
                    self._viewport_recorder is not None,
                )
                imgui.separator()
                helpers, _ = imgui.menu_item(
                    t("Scene Helpers"), "", bool(self.scene_entities.visible)
                )
                if helpers:
                    self.scene_entities.visible = not self.scene_entities.visible
                influence, _ = imgui.menu_item(
                    t("Influence Volumes"),
                    "",
                    bool(self.scene_entities.show_influence),
                    bool(self.scene_entities.visible),
                )
                if influence:
                    self.scene_entities.show_influence = not self.scene_entities.show_influence
                imgui.end_menu()
            if imgui.begin_menu(t("Window")):
                for panel in self.panels:
                    if panel.modal:
                        continue
                    label = t(panel.name)
                    clicked, _ = imgui.menu_item(
                        label,
                        panel.shortcut,
                        panel.open,
                    )
                    if clicked:
                        panel.toggle()
                imgui.separator()
                reset_layout, _ = imgui.menu_item(t("Reset Layout"), "", False)
                imgui.end_menu()
            if imgui.begin_menu(t("Help")):
                open_help, _ = imgui.menu_item(t("Interaction Reference"), "F1", False)
                open_documentation, _ = imgui.menu_item(t("Documentation"), "", False)
                imgui.separator()
                open_about, _ = imgui.menu_item(t("About"), "", False)
                imgui.end_menu()
            path = self.session.asset_path
            document = ""
            if path is not None:
                document = path.name + (" ●" if self.session.dirty else "")
            elif can_scene_files:
                document = t("Untitled") + (" ●" if self.session.dirty else "")
            if document:
                target_x = imgui.get_window_width() - imgui.calc_text_size(document).x - 10.0
                imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), target_x))
                imgui.text_disabled(document)
            imgui.end_main_menu_bar()

        io = imgui.get_io()
        modifier = bool(io.key_ctrl or io.key_super)
        keyboard_free = not io.want_text_input and not imgui.is_any_item_active()
        if modifier and keyboard_free:
            if caps.edit_history:
                undo |= imgui.is_key_pressed(imgui.Key.z, False) and not io.key_shift
                redo |= imgui.is_key_pressed(imgui.Key.z, False) and bool(io.key_shift)
            if can_scene_files:
                new_scene |= imgui.is_key_pressed(imgui.Key.n, False)
                open_scene |= imgui.is_key_pressed(imgui.Key.o, False) and not io.key_shift
                save_scene |= imgui.is_key_pressed(imgui.Key.s, False) and not io.key_shift
                save_scene_as |= imgui.is_key_pressed(imgui.Key.s, False) and bool(io.key_shift)
            elif can_load:
                open_model |= imgui.is_key_pressed(imgui.Key.o, False) and not io.key_shift
            if can_load:
                reload_model |= imgui.is_key_pressed(imgui.Key.o, False) and bool(io.key_shift)
            open_settings |= imgui.is_key_pressed(imgui.Key.comma, False)
            capture_viewport |= bool(io.key_shift) and imgui.is_key_pressed(imgui.Key.p, False)
            toggle_viewport_recording |= bool(io.key_shift) and imgui.is_key_pressed(
                imgui.Key.r, False
            )
            editable_selected = bool(
                self._selected_entity() or self._selected_model_element() is not None
            )
            if can_edit and editable_selected and imgui.is_key_pressed(imgui.Key.d, False):
                self._duplicate_selected()
        quit_viewer |= modifier and imgui.is_key_pressed(imgui.Key.q, False)
        editable_selected = bool(
            self._selected_entity() or self._selected_model_element() is not None
        )
        if can_edit and keyboard_free and editable_selected:
            if imgui.is_key_pressed(imgui.Key.delete, False):
                self._remove_selected()
            if imgui.is_key_pressed(imgui.Key.f2, False):
                self._request_selected_rename()

        if new_scene:
            self._request_document_action("new_scene")
        if undo:
            self.session.submit(cmd.Undo())
        if redo:
            self.session.submit(cmd.Redo())
        if open_settings:
            self.panels.open_panel("Settings")
        if frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)
        if capture_viewport:
            self._capture_viewport_requested = True
        if toggle_viewport_recording:
            self._toggle_viewport_recording()
        if reset_layout:
            self.window.reset_layout()
        if open_help:
            self.panels.open_panel("Help")
        if open_documentation:
            webbrowser.open("https://github.com/acrlw/mojive#readme")
        if open_about:
            self.panels.open_panel("Info")
        if open_scene:
            self._open_scene_dialog("open")
        if save_scene:
            if self.session.asset_path is None:
                self._open_scene_dialog("save")
            else:
                self._request_scene_save(self.session.asset_path)
        if save_scene_as:
            self._open_scene_dialog("save")
        if open_model:
            self._open_model_dialog()
        if add_model:
            self._open_model_dialog("add")
        if remove_model_id >= 0:
            self.remove_model(remove_model_id)
        if add_resource_root:
            self._open_resource_dialog()
        if remove_resource_root is not None:
            self.session.submit(cmd.RemoveResourceRoot(remove_resource_root))
        if reload_model:
            self._queue_model_load("reload", self.session.asset_path)
        if quit_viewer:
            self._request_document_action("quit")

    def _draw_entity_menu(self, shortcut: str, enabled: bool) -> None:
        t = self.localizer.text
        if not imgui.begin_menu(t("Entity"), enabled):
            return
        if imgui.begin_menu(t("Create")):
            for label, shape in (
                ("Box", MeshShape.BOX),
                ("Sphere", MeshShape.SPHERE),
                ("Cylinder", MeshShape.CYLINDER),
                ("Cone", MeshShape.CONE),
                ("Plane", MeshShape.PLANE),
            ):
                clicked, _ = imgui.menu_item(t(label), "", False)
                if clicked:
                    self._add_scene_object(shape, label.lower())
            imgui.separator()
            point_light, _ = imgui.menu_item(t("Point Light"), "", False)
            camera, _ = imgui.menu_item(t("Camera"), "", False)
            site, _ = imgui.menu_item(
                t("Site"), "", False, self.session.adapter.caps.topology_editing
            )
            if point_light:
                self._add_scene_light()
            if camera:
                self._add_scene_camera()
            if site:
                self._add_model_site()
            imgui.end_menu()
        scene_selected = bool(self._selected_entity())
        model_selected = self._selected_model_element() is not None
        duplicate, _ = imgui.menu_item(
            t("Duplicate"), f"{shortcut}+D", False, scene_selected or model_selected
        )
        rename, _ = imgui.menu_item(t("Rename"), "F2", False, scene_selected or model_selected)
        remove, _ = imgui.menu_item(t("Delete"), "Delete", False, scene_selected or model_selected)
        if duplicate:
            self._duplicate_selected()
        if rename:
            self._request_selected_rename()
        if remove:
            self._remove_selected()
        imgui.end_menu()

    def _entity_name(self, base: str) -> str:
        names = {node.name for node in self.session.nodes}
        if base not in names:
            return base
        index = 2
        while f"{base} {index}" in names:
            index += 1
        return f"{base} {index}"

    def _model_child_parent(self) -> SceneNode | None:
        """Resolve the owning MuJoCo body for a top-level create action."""
        node = self.session.selected_node
        while node is not None:
            if node.type in (NodeType.MODEL, NodeType.WORLD, NodeType.LINK, NodeType.ROBOT) and (
                node.type in (NodeType.WORLD, NodeType.MODEL) or node.source_editable
            ):
                return node
            node = self.session.node(node.parent)
        return next(
            (
                node
                for node in self.session.nodes
                if node.type is NodeType.WORLD and node.parent < 0
            ),
            None,
        )

    def _add_model_site(self) -> None:
        parent = self._model_child_parent()
        if parent is None:
            self.session.report_message(
                self.localizer.text("Select a model or body before creating a site")
            )
            return
        result = self.session.submit(
            cmd.AddModelElement(parent.node_id, "site", self._entity_name("site"))
        )
        if result.ok:
            self.session.submit(cmd.SelectNode(result.entity_id))

    def _add_scene_object(self, shape: MeshShape, base_name: str) -> None:
        if shape is MeshShape.PLANE and self.session.adapter.caps.topology_editing:
            world = next(
                (
                    node
                    for node in self.session.nodes
                    if node.type is NodeType.WORLD and node.parent < 0
                ),
                None,
            )
            if world is not None:
                result = self.session.submit(
                    cmd.AddModelElement(
                        world.node_id,
                        "geom:plane",
                        self._entity_name(base_name),
                    )
                )
                if result.ok:
                    node = self.session.node(result.entity_id)
                    if node is not None:
                        self.session.submit(cmd.Select(node.object_id))
                    return
        position = tuple(float(value) for value in self._camera_view().target)
        size = (4.0, 4.0, 0.02) if shape is MeshShape.PLANE else (0.5, 0.5, 0.5)
        result = self.session.submit(
            cmd.AddSceneObject(shape, self._entity_name(base_name), size=size, position=position)
        )
        if result.ok:
            self.session.submit(cmd.Select(result.entity_id))

    def _add_scene_light(self) -> None:
        view = self._camera_view()
        name = self._entity_name("point light")
        result = self.session.submit(
            cmd.AddSceneLight(
                name,
                Light(type=LightType.POINT, position=np.asarray(view.eye, np.float32).copy()),
            )
        )
        if result.ok:
            node = next(
                (
                    node
                    for node in reversed(self.session.nodes)
                    if node.type is NodeType.LIGHT and node.name == name
                ),
                None,
            )
            if node is not None:
                self.session.submit(cmd.Select(node.object_id))

    def _add_scene_camera(self) -> None:
        name = self._entity_name("camera")
        result = self.session.submit(cmd.AddSceneCamera(name, self._camera_view()))
        if result.ok:
            node = next(
                (
                    node
                    for node in reversed(self.session.nodes)
                    if node.type is NodeType.CAMERA and node.name == name
                ),
                None,
            )
            if node is not None:
                self.session.submit(cmd.Select(node.object_id))

    def _duplicate_selected(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            result = self.session.submit(cmd.DuplicateModelElement(node.node_id))
            if result.ok:
                self.session.submit(cmd.SelectNode(result.entity_id))
            return
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.DuplicateSceneEntity(object_id))

    def _remove_selected(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            self.session.submit(cmd.RemoveModelElement(node.node_id))
            return
        object_id = self._selected_entity()
        if object_id:
            self.session.submit(cmd.RemoveSceneEntity(object_id))

    def _selected_model_element(self) -> SceneNode | None:
        node = self.session.selected_node
        if (
            node is None
            or node.model_id < 0
            or not node.source_editable
            or node.type
            not in {
                NodeType.ROBOT,
                NodeType.LINK,
                NodeType.GEOM,
                NodeType.JOINT,
                NodeType.SITE,
                NodeType.CAMERA,
                NodeType.LIGHT,
            }
        ):
            return None
        return node

    def _selected_entity(self) -> int:
        node = self.session.selected_node
        if (
            node is None
            or node.model_id >= 0
            or node.type not in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
        ):
            return 0
        return int(node.object_id)

    def request_rename(self, object_id: int) -> None:
        node = self.session.node_by_object_id(object_id)
        if (
            node is None
            or node.model_id >= 0
            or node.type not in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
        ):
            return
        self._rename_object_id = int(object_id)
        self._rename_model_node_id = -1
        self._rename_value = node.name
        self._open_rename_popup = True

    def request_model_rename(self, node_id: int) -> None:
        node = self.session.node(node_id)
        if (
            node is None
            or node.model_id < 0
            or not node.source_editable
            or node.type
            not in {
                NodeType.ROBOT,
                NodeType.LINK,
                NodeType.GEOM,
                NodeType.JOINT,
                NodeType.SITE,
                NodeType.CAMERA,
                NodeType.LIGHT,
            }
        ):
            return
        self._rename_object_id = 0
        self._rename_model_node_id = int(node_id)
        self._rename_value = node.name
        self._open_rename_popup = True

    def _request_selected_rename(self) -> None:
        node = self._selected_model_element()
        if node is not None:
            self.request_model_rename(node.node_id)
        else:
            self.request_rename(self.session.selected)

    def _report_model_error(self, message: str) -> None:
        self._model_load_error = message
        self._show_model_load_error = True
        self.session.report_message(message, level="error", duration=10.0)

    def _draw_model_load_error(self) -> None:
        t = self.localizer.text
        popup_title = f"{t('File operation failed')}###File operation failed"
        if self._show_model_load_error:
            imgui.open_popup(popup_title)
            self._show_model_load_error = False
        if imgui.is_popup_open(popup_title):
            _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(self._model_load_error)
        imgui.spacing()
        copy_error = t("Copy error")
        if imgui.button(copy_error, imgui.ImVec2(button_width(copy_error, 110.0), 0.0)):
            imgui.set_clipboard_text(self._model_load_error)
        imgui.same_line()
        ok = t("OK")
        if imgui.button(ok, imgui.ImVec2(button_width(ok, 88.0), 0.0)) or imgui.is_key_pressed(
            imgui.Key.escape, False
        ):
            imgui.close_current_popup()
        imgui.end_popup()

    def _request_document_action(self, action: str, path: Path | None = None) -> None:
        if self.session.dirty:
            self._pending_document_action = (action, path)
            return
        self._execute_document_action(action, path)

    def _execute_document_action(self, action: str, path: Path | None = None) -> None:
        if action == "new_scene":
            result = self.session.submit(cmd.NewScene())
            if result.ok:
                self._after_model_change()
                self._set_model_drop_notice(self.localizer.text("New Mojive scene"))
            else:
                self._report_model_error(result.message)
        elif action == "open_scene" and path is not None:
            self._queue_scene_open(path)
        elif action == "quit":
            self._closing_without_save = True
            self.window.request_close()

    def _draw_unsaved_changes(self) -> None:
        pending = self._pending_document_action
        if pending is None:
            return
        t = self.localizer.text
        popup_title = f"{t('Unsaved changes')}###Unsaved changes"
        imgui.open_popup(popup_title)
        _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        name = (
            self.session.asset_path.name if self.session.asset_path is not None else t("Untitled")
        )
        imgui.text(t("Save changes to this file?"))
        imgui.text_wrapped(name)
        imgui.spacing()
        cancel, discard, save = _equal_modal_buttons(
            (t("Cancel"), t("Discard"), t("Save")), self.theme, primary=2
        )
        if save:
            if self.session.asset_path is None:
                self._after_save_action = pending
                self._pending_document_action = None
                self._open_scene_dialog("save")
            else:
                self._pending_document_action = None
                self._request_scene_save(self.session.asset_path, pending)
            imgui.close_current_popup()
        elif discard:
            self._pending_document_action = None
            self._execute_document_action(*pending)
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._pending_document_action = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_pose_save_prompt(self) -> None:
        pending = self._pending_pose_save
        if pending is None:
            return
        t = self.localizer.text
        labels = (t("Cancel"), t("Save without keyframe"), t("Save as key0"))
        popup_title = f"{t('Save current pose')}###Save current pose"
        imgui.open_popup(popup_title)
        _prepare_modal(360.0, self.window.style_scale)
        visible, _ = imgui.begin_popup_modal(
            popup_title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(
            t(
                "The current pose differs from the model default. Add the current qpos as "
                "keyframe key0 in the exported MJCF?"
            )
        )
        imgui.spacing()
        target, after = pending
        cancel, without_key, save_key = _equal_modal_buttons(labels, self.theme, primary=2)
        if save_key:
            self._pending_pose_save = None
            if self.save_scene(target, current_pose_keyframe="key0").ok and after is not None:
                self._execute_document_action(*after)
            imgui.close_current_popup()
        elif without_key:
            self._pending_pose_save = None
            if self.save_scene(target).ok and after is not None:
                self._execute_document_action(*after)
            imgui.close_current_popup()
        elif cancel or imgui.is_key_pressed(imgui.Key.escape, False):
            self._pending_pose_save = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_rename_popup(self) -> None:
        popup_title = f"{self.localizer.text('Rename Entity')}###Rename Entity"
        if self._open_rename_popup:
            imgui.open_popup(popup_title)
            self._open_rename_popup = False
        width = min(220.0, max(1.0, float(imgui.get_main_viewport().work_size.x) - 32.0))
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(width, 0.0),
            imgui.ImVec2(width, float(np.finfo(np.float32).max)),
        )
        visible = imgui.begin_popup(popup_title, imgui.WindowFlags_.always_auto_resize.value)
        if not visible:
            return
        imgui.text_disabled(self.localizer.text("Rename"))
        imgui.separator()
        imgui.set_next_item_width(-1.0)
        submitted, self._rename_value = imgui.input_text(
            "##entity_name",
            self._rename_value,
            imgui.InputTextFlags_.enter_returns_true.value
            | imgui.InputTextFlags_.auto_select_all.value,
        )
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here(-1)
        if submitted and self._rename_value.strip():
            value = self._rename_value.strip()
            command = (
                cmd.RenameModelElement(self._rename_model_node_id, value)
                if self._rename_model_node_id >= 0
                else cmd.RenameSceneEntity(self._rename_object_id, value)
            )
            self.session.submit(command)
            imgui.close_current_popup()
        elif imgui.is_key_pressed(imgui.Key.escape, False):
            imgui.close_current_popup()
        imgui.end_popup()

    def _sync_window_title(self) -> None:
        path = self.session.asset_path
        document = path.name if path is not None else "Untitled"
        title = f"{document}{' *' if self.session.dirty else ''} — {self.title}"
        if title != self._window_title:
            self.window.set_title(title)
            self._window_title = title

    def frame(self) -> None:
        window = self.window
        now = time.perf_counter()
        dt = self._dt = min(0.1, now - self._last_time)
        self._last_time = now
        self._frame_rate.update(dt)

        window.begin_frame()
        self._sync_display_scale()
        if self._model_load_future is not None and self._poll_model_load():
            self._draw_model_loading_frame()
            window.end_frame()
            self._frame_index += 1
            return
        self._poll_model_dialog()
        self._poll_scene_dialog()
        self._poll_resource_dialog()
        self._poll_texture_dialog()
        self._poll_geometry_resource_dialog()
        self._poll_model_asset_dialog()
        self._poll_resource_repair_dialog()
        self._poll_model_drop()
        if self._start_model_load():
            self._draw_model_loading_frame()
            window.end_frame()
            self._frame_index += 1
            return
        self._draw_main_menu()
        self._draw_application_status_bar()
        window.begin_dockspace()
        self._begin_viewport_panel()
        self._sync_viewport_size()
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

        frame = self.session.tick(self.frame_needs(), wall_dt=dt)
        self._sync_structure()
        self._apply_pending_joint_focus()
        self._apply_pending_node_focus()
        self._sync_model_camera()
        self.backend.update(frame)

        selected_node = self.session.selected_node
        self.backend.highlight(
            self.session.selection_highlight_object_id,
            xray=bool(selected_node is not None and selected_node.type is NodeType.JOINT),
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
        self._publish_gizmo()

        self._viewport_image = self.backend.render()
        self._record_viewport_frame(dt)
        if self._capture_viewport_requested:
            self._capture_viewport()
        self.camera_preview.update(
            self.backend,
            self.session.source,
            self.session.structure_generation,
            frame,
            preview_camera,
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
        self._sync_window_title()
        window.end_frame()
        self._frame_index += 1

    def _draw_model_loading_frame(self) -> None:
        """Keep the native window responsive without reading a mutating Session."""

        self._draw_loading_main_menu()
        self._draw_application_status_bar(loading=True)
        self.window.begin_dockspace()
        self.panels.draw_shells(self.localizer.text, self.window.style_scale)
        self._begin_viewport_panel()
        self._sync_viewport_size()
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
            imgui.spacing()
            imgui.text_disabled(t("The loader does not expose reliable stage progress."))
        imgui.end()

    def _scene_input_blocked(self) -> bool:
        """Return whether UI owns the whole interaction frame.

        This check runs before gesture classification so a modal cannot leave
        an old camera, gizmo, or perturb claim alive underneath it.
        """

        pending_prompt = bool(
            self._pending_document_action is not None
            or self._pending_pose_save is not None
            or self._show_model_load_error
            or self._open_resource_repair_popup
            or self._open_rename_popup
            or self._precise_gizmo_edit is not None
        )
        native_dialog = any(
            dialog is not None
            for dialog in (
                self._model_dialog,
                self._scene_dialog,
                self._resource_dialog,
                self._texture_dialog,
                self._geometry_resource_dialog,
                self._model_asset_dialog,
                self._resource_repair_dialog,
            )
        )
        popup_flags = imgui.PopupFlags_.any_popup_id.value | imgui.PopupFlags_.any_popup_level.value
        any_popup = bool(imgui.is_popup_open("", popup_flags))
        io = imgui.get_io()
        return bool(
            pending_prompt
            or native_dialog
            or any_popup
            or io.want_text_input
            or self._consume_scene_pointer_until_release
        )

    def _poll_keys(self) -> Keys:
        io = imgui.get_io()
        if self._scene_input_blocked():
            return Keys()
        self.panels.poll_shortcuts()

        clear_selection = self._selection_clear_enabled() and imgui.is_key_pressed(
            imgui.Key.escape, False
        )
        if io.key_ctrl or io.key_super:
            return Keys(clear_selection=clear_selection)

        bindings = self.input_bindings

        def down(action: InputAction) -> float:
            return 1.0 if bindings.down(action) else 0.0

        axis = next(
            (
                index
                for index, action in enumerate(
                    (InputAction.AXIS_X, InputAction.AXIS_Y, InputAction.AXIS_Z)
                )
                if bindings.down(action)
            ),
            -1,
        )
        return Keys(
            fly=(
                down(InputAction.FLY_FORWARD) - down(InputAction.FLY_BACK),
                down(InputAction.FLY_RIGHT) - down(InputAction.FLY_LEFT),
                down(InputAction.FLY_UP) - down(InputAction.FLY_DOWN),
            ),
            toggle_pause=bindings.pressed(InputAction.TOGGLE_PAUSE),
            step_back_count=(
                bindings.press_count(
                    InputAction.STEP_BACK,
                    delay=STEP_BACK_REPEAT_DELAY_SECONDS,
                    rate=STEP_BACK_REPEAT_RATE_SECONDS,
                )
                if self.session.can_step_back
                else 0
            ),
            clear_selection=clear_selection,
            frame_scene=bindings.pressed(InputAction.FRAME_SCENE),
            gizmo_translate=bindings.pressed(InputAction.GIZMO_TRANSLATE),
            gizmo_rotate=bindings.pressed(InputAction.GIZMO_ROTATE),
            gizmo_space=bindings.pressed(InputAction.GIZMO_SPACE),
            gizmo_axis=axis,
        )

    def _input_state(self) -> gs.InputState:
        io = imgui.get_io()
        blocked = self._scene_input_blocked()
        cursor = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        rect = self._viewport_rect
        inside = (
            rect[0] <= cursor[0] <= rect[0] + rect[2] and rect[1] <= cursor[1] <= rect[1] + rect[3]
        )
        hovered_window = imgui.get_current_context().hovered_window
        hovered_name = None if hovered_window is None else str(hovered_window.name)
        viewport_window_busy = self._viewport_window_is_being_manipulated()
        over_viewport = (
            gs.viewport_input_allowed(inside, hovered_name)
            and not viewport_window_busy
            and not blocked
        )
        view = self._camera_view()
        hovered_ball = self.view_cube.update(
            view,
            rect,
            cursor,
            self.window.style_scale,
            enabled=over_viewport,
        )
        self.gizmo.update_hover(
            self.session,
            view,
            rect,
            cursor,
            enabled=over_viewport and not self._viewing_selected_camera(),
            style_scale=self.window.style_scale,
        )
        self._gizmo_hint_hover.update(
            over_viewport
            and self.gizmo.precise_input_hovered
            and not any(imgui.is_mouse_down(button) for button in range(3)),
            time.monotonic(),
        )
        node = self.session.selected_node
        return gs.InputState(
            blocked=blocked,
            left=imgui.is_mouse_down(0),
            right=imgui.is_mouse_down(1),
            middle=imgui.is_mouse_down(2),
            ctrl=self.input_bindings.down(InputAction.PERTURB),
            shift=self.input_bindings.down(InputAction.SNAP),
            alt=io.key_alt,
            wheel=float(io.mouse_wheel),
            cursor=cursor,
            delta=(float(io.mouse_delta.x), float(io.mouse_delta.y)),
            over_viewport=over_viewport,
            over_view_cube=over_viewport and hovered_ball is not None,
            gizmo_available=(self.gizmo.style == "2d" or self.backend.caps.gizmo)
            and self.gizmo.last_verdict.ok,
            gizmo_hovered=over_viewport and self.gizmo.hovered,
            has_selection=node is not None,
            perturbing=self.session.perturb.active,
            ui_wants_mouse=blocked
            or viewport_window_busy
            or (io.want_capture_mouse and not over_viewport),
        )

    @staticmethod
    def _viewport_window_is_being_manipulated() -> bool:
        """Keep scene gestures out of floating-window move and resize gestures."""

        context = imgui.get_current_context()
        window = context.current_window
        if window is None or str(window.name).rsplit("###", 1)[-1] != "Viewport":
            return False
        moving = context.moving_window
        if moving is not None and int(moving.id_) == int(window.id_):
            return True
        return bool(
            int(window.resize_border_held) >= 0
            or (int(window.resize_border_hovered) >= 0 and imgui.is_mouse_down(0))
        )

    def _claim_gesture(self, state: gs.InputState) -> gs.Claim:
        return self.router.update(state)

    def _poll_gizmo(self, state: gs.InputState, keys: Keys) -> None:
        if state.blocked:
            self.gizmo.cancel()
            return
        if self._precise_gizmo_edit is not None:
            self.gizmo.cancel()
            return
        if self._viewing_selected_camera():
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                -1,
                style_scale=self.window.style_scale,
            )
            self.gizmo.interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                claimed=False,
                left_down=state.left,
                released=self.router.released,
                style_scale=self.window.style_scale,
            )
            return
        keyboard_was_active = self.gizmo.keyboard_using
        axis = keys.gizmo_axis
        if not keyboard_was_active and (not state.over_viewport or state.any_button):
            axis = -1
        if keyboard_was_active or axis >= 0:
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                axis,
                snap=state.shift or self._snap_latched,
                style_scale=self.window.style_scale,
            )
            return
        if state.gizmo_hovered and imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
            edit = self.gizmo.precise_input(self.session)
            if edit is not None:
                self.gizmo.cancel()
                self.router.abort()
                self._begin_precise_gizmo_input(edit)
                return
        self.gizmo.interact(
            self.session,
            self._camera_view(),
            self._viewport_rect,
            state.cursor,
            claimed=self.router.wants_gizmo(),
            left_down=state.left,
            released=self.router.released,
            snap=state.shift or self._snap_latched,
            style_scale=self.window.style_scale,
        )

    def _begin_precise_gizmo_input(self, edit: PreciseGizmoInput) -> None:
        self._precise_gizmo_edit = edit
        if not self.gizmo.remember_precise_input_choices:
            self._precise_gizmo_absolute = False
            self._precise_gizmo_angle_unit = "degrees"
        else:
            self._precise_gizmo_absolute = bool(
                self._precise_gizmo_preferred_absolute and edit.absolute_value is not None
            )
        self._precise_gizmo_value = (
            self._precise_gizmo_reference(edit) if self._precise_gizmo_absolute else 0.0
        )
        self._precise_gizmo_error = ""
        self._open_precise_gizmo_popup = True

    def _draw_precise_gizmo_popup(self) -> None:
        edit = self._precise_gizmo_edit
        if edit is None:
            return
        scale = self.window.style_scale
        just_opened = self._open_precise_gizmo_popup
        popup_name = f"{self.localizer.text('Type value')}###precise_gizmo_input"
        if just_opened:
            x, y, width, height = self._viewport_rect
            mouse = imgui.get_io().mouse_pos
            window_width = PRECISE_GIZMO_WIDTH_PT * scale
            estimated_height = 118.0 * scale
            min_x = x + 8.0 * scale
            min_y = y + 8.0 * scale
            max_x = max(min_x, x + width - window_width - 8.0 * scale)
            max_y = max(min_y, y + height - estimated_height - 8.0 * scale)
            imgui.set_next_window_pos(
                imgui.ImVec2(
                    min(max(min_x, mouse.x + 12.0 * scale), max_x),
                    min(max(min_y, mouse.y + 12.0 * scale), max_y),
                ),
                imgui.Cond_.always,
            )
            imgui.set_next_window_focus()
            imgui.open_popup(popup_name)
            self._open_precise_gizmo_popup = False
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(PRECISE_GIZMO_WIDTH_PT * scale, 0.0),
            imgui.ImVec2(
                PRECISE_GIZMO_WIDTH_PT * scale,
                float(np.finfo(np.float32).max),
            ),
        )
        visible = imgui.begin_popup(
            popup_name,
            imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_scroll_with_mouse.value
            | imgui.WindowFlags_.no_saved_settings.value,
        )
        if not visible:
            if not just_opened:
                self._consume_scene_pointer_until_release = bool(
                    self._consume_scene_pointer_until_release
                    or imgui.is_mouse_down(imgui.MouseButton_.left)
                    or imgui.is_mouse_down(imgui.MouseButton_.right)
                    or imgui.is_mouse_down(imgui.MouseButton_.middle)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.left)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.right)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.middle)
                )
                self.router.abort()
                self.gizmo.cancel()
                self._precise_gizmo_edit = None
                self._precise_gizmo_error = ""
            return
        imgui.set_scroll_x(0.0)
        appearing = imgui.is_window_appearing()

        angular = edit.unit == "°"
        unit_shortcut = angular and imgui.is_key_pressed(imgui.Key.u, False)
        if unit_shortcut:
            self._toggle_precise_gizmo_angle_unit()
            # Rebuild InputScalar's private edit buffer from the converted
            # value. Its active text state otherwise keeps the pre-conversion
            # string even though the numeric model has changed.
            imgui.internal.clear_active_id()
        unit = "rad" if angular and self._precise_gizmo_angle_unit == "radians" else edit.unit
        title = f"{self.localizer.text(edit.action)} {edit.label}"
        # Some popup placements preserve a negative horizontal cursor offset
        # from the activating item. Clamp the first line to the popup's own
        # content padding so the beginning of "Rotate" cannot be clipped.
        imgui.set_cursor_pos_x(
            max(
                float(imgui.get_cursor_pos_x()),
                float(imgui.get_style().window_padding.x),
            )
        )
        title_width = max(1.0, float(imgui.get_content_region_avail().x))
        shown_title = _middle_elide_text(
            title,
            title_width,
            lambda value: float(imgui.calc_text_size(value).x),
        )
        imgui.text(shown_title)
        if shown_title != title:
            imgui.set_item_tooltip(title)
        imgui.separator()
        modes = (
            (
                self.localizer.text("Relative"),
                self.localizer.text("Absolute"),
            )
            if edit.absolute_value is not None
            else (self.localizer.text("Relative"),)
        )
        mode_index = 1 if self._precise_gizmo_absolute and len(modes) > 1 else 0
        if len(modes) > 1:
            imgui.text_disabled(self.localizer.text("Mode"))
            next_mode = segmented_control(
                "precise-gizmo-mode",
                modes,
                mode_index,
                theme=self.theme,
            )
        else:
            next_mode = 0
        if next_mode != mode_index:
            mode_index = next_mode
            self._set_precise_gizmo_absolute(edit, bool(mode_index))
        imgui.spacing()
        if edit.absolute_value is None:
            imgui.text("Δ")
            imgui.same_line()
        unit_width = 82.0 * scale if angular else float(imgui.calc_text_size(unit).x)
        if not angular:
            unit_width += 2.0 * float(imgui.get_style().frame_padding.x)
        input_width = max(
            72.0 * scale,
            float(imgui.get_content_region_avail().x)
            - float(imgui.get_style().item_spacing.x)
            - unit_width,
        )
        if appearing or unit_shortcut:
            imgui.set_keyboard_focus_here()
        imgui.set_next_item_width(input_width)
        submitted, self._precise_gizmo_value = imgui.input_double(
            "##precise_gizmo_value",
            self._precise_gizmo_value,
            0.0,
            0.0,
            "%.6f" if edit.unit == "m" or unit == "rad" else "%.3f",
            imgui.InputTextFlags_.enter_returns_true.value
            | imgui.InputTextFlags_.auto_select_all.value
            | imgui.InputTextFlags_.chars_scientific.value,
        )
        imgui.same_line()
        if angular:
            next_unit = segmented_control(
                "precise-gizmo-angle-unit",
                ("°", "rad"),
                1 if self._precise_gizmo_angle_unit == "radians" else 0,
                width=unit_width,
                theme=self.theme,
            )
            if next_unit != (1 if self._precise_gizmo_angle_unit == "radians" else 0):
                self._toggle_precise_gizmo_angle_unit()
        else:
            imgui.text(unit)
        if self._precise_gizmo_error:
            imgui.spacing()
            imgui.text_wrapped(self._precise_gizmo_error)
            if imgui.small_button(f"{self.localizer.text('Copy error')}##precise-gizmo"):
                imgui.set_clipboard_text(self._precise_gizmo_error)
        submit_requested = submitted or imgui.is_key_pressed(imgui.Key.enter, False)
        cancel = imgui.is_key_pressed(imgui.Key.escape, False)
        if submit_requested:
            value = self._precise_gizmo_value
            if angular and self._precise_gizmo_angle_unit == "radians":
                value = float(np.degrees(value))
            result = self.gizmo.apply_precise_value(
                self.session,
                self._camera_view(),
                edit,
                value,
                absolute=self._precise_gizmo_absolute,
            )
            if result.ok:
                imgui.close_current_popup()
                self._precise_gizmo_edit = None
                self._precise_gizmo_error = ""
            else:
                self._precise_gizmo_error = result.message
        elif cancel:
            imgui.close_current_popup()
            self._precise_gizmo_edit = None
            self._precise_gizmo_error = ""
        imgui.end_popup()

    def _finish_consumed_scene_pointer(self) -> None:
        """Release a consumed pointer gesture after one clean release frame."""

        if not self._consume_scene_pointer_until_release:
            return
        if any(imgui.is_mouse_down(button) for button in range(3)):
            return
        self._consume_scene_pointer_until_release = False

    def _precise_gizmo_reference(self, edit: PreciseGizmoInput) -> float:
        value = float(edit.absolute_value or 0.0)
        if edit.unit == "°" and self._precise_gizmo_angle_unit == "radians":
            return float(np.radians(value))
        return value

    def _set_precise_gizmo_absolute(self, edit: PreciseGizmoInput, absolute: bool) -> None:
        absolute = bool(absolute and edit.absolute_value is not None)
        if absolute == self._precise_gizmo_absolute:
            return
        reference = self._precise_gizmo_reference(edit)
        self._precise_gizmo_value += reference if absolute else -reference
        self._precise_gizmo_absolute = absolute
        self._precise_gizmo_preferred_absolute = absolute
        self._persist_precise_gizmo_choices()

    def _toggle_precise_gizmo_angle_unit(self) -> None:
        self._precise_gizmo_value, self._precise_gizmo_angle_unit = _toggle_angle_input(
            self._precise_gizmo_value,
            self._precise_gizmo_angle_unit,
        )
        self._persist_precise_gizmo_choices()

    def _persist_precise_gizmo_choices(self) -> None:
        if not self.gizmo.remember_precise_input_choices:
            return
        self.localizer.set_preferences(
            {
                "precise_gizmo_absolute": self._precise_gizmo_preferred_absolute,
                "precise_gizmo_angle_unit": self._precise_gizmo_angle_unit,
            }
        )

    def _publish_gizmo(self) -> None:
        self.gizmo.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
            yielding=gs.gizmo_yields(self._state) or self._viewing_selected_camera(),
            interactive=self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO),
        )

    def _poll_camera(self, state: gs.InputState, keys: Keys, dt: float) -> None:
        fwd, right, up = keys.fly
        if fwd or right or up:
            self._leave_model_camera()
            self.camera.fly(dt, forward=fwd, right=right, up=up)
        if keys.frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)

        if self.router.wants_view_cube():
            ball = self.view_cube.hovered

            if self.router.travel >= CLICK_SLOP_PT and state.delta != (0.0, 0.0):
                self._leave_model_camera()
                self.view_cube.drag(self.camera, *state.delta)
            elif ball is not None and self.router.released and self.router.travel < CLICK_SLOP_PT:
                self._leave_model_camera()
                self.view_cube.click(
                    self.camera,
                    ball,
                    self.camera_out,
                    focus=self._selected_view_focus(),
                )
            return

        if not self.router.wants_camera():
            return
        gesture = gs.camera_gesture(state)

        settled = self.router.travel >= CLICK_SLOP_PT
        if gesture is gs.CameraGesture.ORBIT and settled:
            self._leave_model_camera()
            self.camera.orbit(*state.delta)
        elif gesture is gs.CameraGesture.PAN and settled:
            self._leave_model_camera()
            self.camera.pan(state.delta[0], state.delta[1], self._viewport_rect[3])
        elif gesture is gs.CameraGesture.DOLLY:
            self._leave_model_camera()
            self.camera.dolly(state.wheel)

    def _advance_camera(self, dt: float) -> None:
        if self._model_camera_id >= 0:
            return
        self.camera.advance(dt, self.camera_out)

    def _camera_view(self):
        return self._model_camera_view or self.camera.view()

    def select_model_camera(self, camera_id: int) -> None:
        i = int(camera_id)
        if i >= 0 and not any(c.camera_id == i for c in self.session.cameras):
            return
        if i < 0:
            self._leave_model_camera(publish=True)
            return
        if i != self._model_camera_id:
            self._model_camera_projection_target = None
        self._model_camera_id = i

    def _viewing_selected_camera(self) -> bool:
        node = self.session.selected_node
        if node is None or node.type is not NodeType.CAMERA:
            return False
        index = int(node.camera_index)
        if not 0 <= index < len(self.session.cameras):
            return False
        return int(self.session.cameras[index].camera_id) == self._model_camera_id

    def _sync_model_camera(self) -> None:
        if self._model_camera_id < 0:
            return
        view = self.session.camera_view(self._model_camera_id)
        if view is None:
            self._leave_model_camera(publish=True)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        view = view.with_aspect(aspect)
        target = bool(view.orthographic)
        if self._model_camera_projection_target is None:
            self._model_camera_projection.snap(target)
        elif target != self._model_camera_projection_target:
            self._model_camera_projection.set(target, animate=True)
        self._model_camera_projection_target = target
        self._model_camera_projection.advance(self._dt)
        if self._model_camera_projection.active:
            view = replace(view, orthographic_blend=self._model_camera_projection.value)
        self._model_camera_view = view
        self.backend.set_camera(view)
        self.session.submit(cmd.SetCamera(view))

    def _leave_model_camera(self, *, publish: bool = False) -> None:
        if self._model_camera_id < 0:
            return
        # Model cameras remain scene entities; the editor orbit camera keeps its own view.
        self._model_camera_id = -1
        self._model_camera_view = None
        self._model_camera_projection_target = None
        if publish:
            self.camera.publish(self.camera_out)

    def _frame_scene(self, *, animate: bool = True) -> None:
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.frame_scene(
            self.session.bounds(),
            self.camera_out,
            animate=animate,
            clip=self.session.camera_hint(),
            minimum_pitch=ISO_PITCH,
        )

    def _reset_source_camera(self) -> None:
        """Restore the scene source's authored/default free camera."""
        hint = self.session.camera_hint()
        if hint is None:
            self._frame_scene(animate=False)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        self.camera.adopt(hint.with_aspect(aspect))
        self.camera.publish(self.camera_out)

    def _poll_perturb(self, state: gs.InputState) -> None:
        st = self.session.perturb
        if not self.router.wants_perturb():
            if st.active:
                self.perturb.end(self.session)
            return

        node = self.session.selected_node
        if node is None:
            return
        cam = self._camera_view()
        ray = self._cursor_ray(state.cursor) if self.router.mode == "translate" else None
        if not st.active:
            pos, _ = self._node_pose(node)
            grab_point = pos
            if ray is not None:
                grab_point = cursor_grab_point(cam, pos, ray[0], ray[1])
            self.perturb.begin(
                self.session,
                cam,
                node,
                grab_point,
                self.router.mode,
                local_bounds=self.session.node_local_bounds(node.node_id),
            )
        if st.mode == "translate":
            origin, direction = ray if ray is not None else self._cursor_ray(state.cursor)
            self.perturb.drag_translate(self.session, cam, origin, direction)
        else:
            self.perturb.drag_rotate(self.session, cam, state.delta[0], state.delta[1])
        self.perturb.apply(self.session)

    def _publish_perturb_marks(self) -> None:
        self.perturb.publish_marks(
            self.backend,
            self.session,
            self._camera_view(),
            rect=self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
        )

    def _selected_view_focus(self) -> tuple[np.ndarray, float] | None:
        node = self.session.selected_node
        if node is None:
            return None
        bounds = self.session.node_world_bounds(node.node_id)
        if bounds is None:
            return None
        center, world_half = bounds
        center = np.asarray(center, np.float64)
        radius = float(np.linalg.norm(world_half))
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius < 1e-6:
            return None
        return center, radius

    def request_joint_focus(self, joint_id: int) -> bool:
        """Request one diagnostics-backed camera focus on a stable joint ID."""

        requested = int(joint_id)
        if not any(joint.joint_id == requested for joint in self.session.joints):
            return False
        self._pending_node_focus_id = None
        self._pending_joint_focus_id = requested
        return True

    def request_node_focus(self, node_id: int) -> bool:
        """Request a camera focus on one stable hierarchy node."""

        node = self.session.node(int(node_id))
        if node is None:
            return False
        if node.type is NodeType.JOINT:
            return self.request_joint_focus(node.joint_index)
        self._pending_joint_focus_id = None
        self._pending_node_focus_id = node.node_id
        return True

    def _request_node_joint_focus(self, node: SceneNode) -> bool:
        if node.type not in (NodeType.JOINT, NodeType.LINK, NodeType.ROBOT, NodeType.GEOM):
            return False
        joint_id = int(node.joint_index) if node.type is NodeType.JOINT else -1
        if joint_id < 0 and node.body_index >= 0:
            candidates = tuple(
                joint
                for joint in self.session.joints_for_body(node.body_index)
                if joint.type in ("hinge", "slide", "ball")
            )
            selected = self.gizmo.selected_joint_id(node.body_index)
            if any(joint.joint_id == selected for joint in candidates):
                joint_id = selected
            elif len(candidates) == 1:
                joint_id = candidates[0].joint_id
        if joint_id < 0 or not self.request_joint_focus(joint_id):
            return False
        joint_node = next(
            (
                candidate
                for candidate in self.session.nodes
                if candidate.type is NodeType.JOINT and candidate.joint_index == joint_id
            ),
            None,
        )
        if joint_node is not None:
            self.session.submit(cmd.SelectNode(joint_node.node_id))
        return True

    def _apply_pending_joint_focus(self) -> None:
        joint_id = self._pending_joint_focus_id
        if joint_id is None:
            return
        self._pending_joint_focus_id = None
        joint = next(
            (candidate for candidate in self.session.joints if candidate.joint_id == joint_id),
            None,
        )
        node = next(
            (
                candidate
                for candidate in self.session.nodes
                if candidate.type is NodeType.JOINT and candidate.joint_index == joint_id
            ),
            None,
        )
        if joint is None or node is None:
            return

        body_position, body_rotation = self._node_pose(node)
        center = np.asarray(body_position, np.float64).reshape(3)
        axis = np.asarray(body_rotation, np.float64).reshape(3, 3) @ np.asarray(
            joint.axis, np.float64
        ).reshape(3)
        diagnostics = self.session.frame.diagnostics
        if diagnostics is not None:
            if 0 <= joint_id < len(diagnostics.joint_xaxis):
                candidate_axis = np.asarray(diagnostics.joint_xaxis[joint_id], np.float64).reshape(
                    3
                )
                if np.isfinite(candidate_axis).all() and np.linalg.norm(candidate_axis) > 1e-9:
                    axis = candidate_axis
            if joint.type != "slide" and 0 <= joint_id < len(diagnostics.joint_xpos):
                candidate_center = np.asarray(diagnostics.joint_xpos[joint_id], np.float64).reshape(
                    3
                )
                if np.isfinite(candidate_center).all():
                    center = candidate_center

        axis_length = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or not np.isfinite(axis_length) or axis_length <= 1e-9:
            axis = np.array((0.0, 0.0, 1.0), np.float64)
        else:
            axis = axis / axis_length

        slide_half_span = 0.0
        frame = self.session.frame
        if joint.type == "slide" and joint.limited:
            lower, upper = map(float, joint.range)
            if (
                np.isfinite((lower, upper)).all()
                and upper > lower
                and frame.qpos is not None
                and 0 <= joint.qpos_adr < len(frame.qpos)
            ):
                current = float(frame.qpos[joint.qpos_adr])
                if np.isfinite(current):
                    midpoint = (lower + upper) * 0.5
                    center = center + axis * (midpoint - current)
                    slide_half_span = (upper - lower) * 0.5

        current_view = self._camera_view()
        eye_offset = np.asarray(current_view.eye, np.float64).reshape(3) - center
        camera_right = camera_basis(current_view)[0]
        elevated_direction = elevated_focus_view_direction(eye_offset, camera_right)
        if joint.type == "hinge":
            candidates = oblique_axis_view_directions(
                axis,
                eye_offset,
                camera_right,
                JOINT_FOCUS_OBLIQUE_DEGREES,
            )
        elif joint.type == "slide":
            nearest = closest_perpendicular_view_direction(
                axis,
                eye_offset,
                camera_right,
            )
            tangent = np.cross(axis, nearest)
            candidates = (nearest, -nearest, tangent, -tangent, elevated_direction)
        else:
            candidates = (elevated_direction,)

        radius = self._joint_focus_radius(node)
        if slide_half_span > 0.0:
            radius += slide_half_span
        eye_direction = self._least_occluded_joint_direction(
            center,
            radius,
            candidates,
            node,
            eye_offset,
            preferred_up=(0.0, 0.0, 1.0),
            minimum_elevation_degrees=ISO_PITCH,
        )
        if self._model_camera_id >= 0:
            self.camera.adopt(current_view)
        self._leave_model_camera()
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.focus_target(
            center,
            radius,
            eye_direction,
            self.camera_out,
            margin=JOINT_FOCUS_MARGIN,
            animate=True,
        )

    def _least_occluded_joint_direction(
        self,
        center: np.ndarray,
        radius: float,
        candidates: tuple[np.ndarray, ...],
        node: SceneNode,
        eye_offset: np.ndarray,
        *,
        preferred_up: tuple[float, float, float] | None = None,
        minimum_elevation_degrees: float = 0.0,
    ) -> np.ndarray:
        """Prefer a requested elevation, then a clear focus ray and a short turn."""

        directions: list[np.ndarray] = []
        for value in candidates:
            direction = np.asarray(value, np.float64).reshape(3)
            length = float(np.linalg.norm(direction))
            if not np.isfinite(direction).all() or not np.isfinite(length) or length <= 1e-9:
                continue
            direction = direction / length
            if not any(float(np.dot(direction, known)) > 1.0 - 1e-7 for known in directions):
                directions.append(direction)
        if not directions:
            return np.asarray(self.camera.direction(), np.float64)

        current_direction = np.asarray(eye_offset, np.float64)
        current_length = float(np.linalg.norm(current_direction))
        if current_length > 1e-9:
            current_direction /= current_length
        else:
            current_direction = directions[0]

        up = None
        if preferred_up is not None:
            candidate_up = np.asarray(preferred_up, np.float64).reshape(3)
            up_length = float(np.linalg.norm(candidate_up))
            if np.isfinite(candidate_up).all() and np.isfinite(up_length) and up_length > 1e-9:
                up = candidate_up / up_length

        if up is not None:
            minimum_elevation = np.deg2rad(float(np.clip(minimum_elevation_degrees, 0.0, 90.0)))
            preferred = [
                direction
                for direction in directions
                if float(np.dot(direction, up)) >= float(np.sin(minimum_elevation)) - 1e-6
            ]
            if preferred:
                directions = preferred
            else:
                above = [
                    direction for direction in directions if float(np.dot(direction, up)) >= -1e-6
                ]
                if above:
                    directions = above

        def turn_from_current(direction: np.ndarray) -> float:
            return 1.0 - float(np.clip(np.dot(direction, current_direction), -1.0, 1.0))

        caps = getattr(getattr(self.session, "adapter", None), "caps", None)
        if not bool(getattr(caps, "raycast", False)):
            return min(directions, key=turn_from_current)

        lo, hi = self.session.bounds()
        scene_span = float(np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)))
        current_distance = float(np.linalg.norm(np.asarray(eye_offset, np.float64)))
        probe_distance = max(current_distance, scene_span * 0.75, float(radius) * 8.0, 1.0)

        best = directions[0]
        best_score = (2.0, 0.0, 0.0)
        for direction in directions:
            origin = np.asarray(center, np.float64) + direction * probe_distance
            try:
                hit_id, hit_distance = self.session.query(
                    cmd.Pick(origin=origin, direction=-direction)
                )
            except (AttributeError, TypeError, ValueError):
                return min(directions, key=turn_from_current)
            hit_id = int(hit_id)
            hit_distance = float(hit_distance)
            hit_node = self.session.node_by_object_id(hit_id) if hit_id > 0 else None
            hits_target = bool(
                hit_id == node.body_index
                or (hit_node is not None and hit_node.body_index == node.body_index)
            )
            blocked = bool(
                hit_id > 0
                and not hits_target
                and np.isfinite(hit_distance)
                and hit_distance < probe_distance
            )
            clearance = min(max(hit_distance, 0.0), probe_distance) if blocked else 0.0
            score = (1.0 if blocked else 0.0, -clearance, turn_from_current(direction))
            if score < best_score:
                best = direction
                best_score = score
        return best

    def _apply_pending_node_focus(self) -> None:
        node_id = self._pending_node_focus_id
        if node_id is None:
            return
        self._pending_node_focus_id = None
        node = self.session.node(node_id)
        if node is None:
            return

        bounds = self.session.node_world_bounds(node.node_id)
        if bounds is None and node.type in (NodeType.WORLD, NodeType.ENVIRONMENT):
            bounds = self.session.bounds()
        if bounds is not None:
            lo_or_center = np.asarray(bounds[0], np.float64).reshape(3)
            hi_or_half = np.asarray(bounds[1], np.float64).reshape(3)
            if node.type in (NodeType.WORLD, NodeType.ENVIRONMENT):
                center = (lo_or_center + hi_or_half) * 0.5
                radius = float(np.linalg.norm(hi_or_half - lo_or_center) * 0.5)
            else:
                center = lo_or_center
                radius = float(np.linalg.norm(hi_or_half))
        else:
            center, _rotation = self._node_pose(node)
            lo, hi = self.session.bounds()
            radius = float(
                np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)) * 0.025
            )
        center = np.asarray(center, np.float64).reshape(3)
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 1e-6:
            return

        current_view = self._camera_view()
        eye_offset = np.asarray(current_view.eye, np.float64).reshape(3) - center
        eye_direction = elevated_focus_view_direction(
            eye_offset,
            camera_basis(current_view)[0],
        )
        if self._model_camera_id >= 0:
            self.camera.adopt(current_view)
        self._leave_model_camera()
        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.focus_target(
            center,
            radius,
            eye_direction,
            self.camera_out,
            animate=True,
        )

    def _joint_focus_radius(self, node: SceneNode) -> float:
        for bounds in (
            self.session.node_local_bounds(node.node_id),
            self.session.node_world_bounds(node.node_id),
        ):
            if bounds is None:
                continue
            half = np.sort(np.abs(np.asarray(bounds[1], np.float64).reshape(3)))
            if np.isfinite(half).all() and half[-1] > 1e-6:
                return max(float(half[-1]), 1e-4)
        lo, hi = self.session.bounds()
        diagonal = float(np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64)))
        if np.isfinite(diagonal) and diagonal > 1e-6:
            return max(diagonal * 0.025, 1e-4)
        return max(self.camera.distance * 0.04, 1e-4)

    def _poll_pick(self, state: gs.InputState) -> None:
        if not self.router.wants_camera():
            return
        if not self.router.released:
            return
        if self.router.travel > CLICK_SLOP_PT:
            self._last_viewport_click = None
            return
        if not self.router.started_with_left:
            return
        if not state.over_viewport:
            return
        object_id = self._pick_at(state.cursor)
        now = time.monotonic()
        previous = self._last_viewport_click
        radius = VIEWPORT_DOUBLE_CLICK_RADIUS_PT * self.window.style_scale
        double_clicked = bool(
            previous is not None
            and object_id > 0
            and previous[2] == object_id
            and now - previous[0] <= VIEWPORT_DOUBLE_CLICK_SECONDS
            and (state.cursor[0] - previous[1][0]) ** 2 + (state.cursor[1] - previous[1][1]) ** 2
            <= radius * radius
        )
        self._last_viewport_click = None if double_clicked else (now, state.cursor, object_id)
        if double_clicked:
            node = self.session.node_by_object_id(object_id)
            if node is not None:
                if self._request_node_joint_focus(node):
                    return
                self.request_node_focus(node.node_id)
        self.session.submit(cmd.Select(object_id))

    def _pick_at(self, cursor: tuple[float, float]) -> int:
        rect = self._viewport_rect

        helper = self.scene_entities.pick(
            self.session,
            self._camera_view(),
            rect,
            cursor,
            self.window.style_scale,
            self._model_camera_id >= 0,
        )
        if self._selectable(helper):
            return helper

        img = self._viewport_image
        if self.backend.caps.gpu_pick and img is not None:
            hit = img.pixel_from_viewport_point(cursor, rect)
            if hit is not None:
                object_id = int(self.backend.pick(*hit))
                if self._selectable(object_id):
                    return object_id

        if self.session.adapter.caps.raycast:
            origin, direction = self._cursor_ray(cursor)
            object_id, _dist = self.session.query(cmd.Pick(origin=origin, direction=direction))
            if self._selectable(int(object_id)):
                return int(object_id)

        return self._nearest_link(cursor)

    def _selectable(self, object_id: int) -> bool:
        if object_id <= 0:
            return False
        node = self.session.node_by_object_id(object_id)
        if node is None:
            return False
        return node.type is not NodeType.WORLD and node.parent >= 0

    def _nearest_link(self, cursor: tuple[float, float]) -> int:
        frame = self.session.frame
        if frame.body_xpos is None or len(frame.body_xpos) == 0:
            return 0
        cam = self._camera_view()
        mvp = cam.proj_matrix() @ cam.view_matrix()
        pts = np.asarray(frame.body_xpos, np.float64)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ mvp.T
        w = np.where(np.abs(h[:, 3]) < 1e-9, 1e-9, h[:, 3])
        rect = self._viewport_rect
        sx = rect[0] + (h[:, 0] / w * 0.5 + 0.5) * rect[2]
        sy = rect[1] + (0.5 - h[:, 1] / w * 0.5) * rect[3]
        d2 = (sx - cursor[0]) ** 2 + (sy - cursor[1]) ** 2
        d2[w <= 0.0] = np.inf
        best_body = int(np.argmin(d2))
        limit = (PICK_SCREEN_RADIUS_PT * self.window.style_scale) ** 2
        if not np.isfinite(d2[best_body]) or d2[best_body] > limit:
            return 0
        for node in self.session.nodes:
            if node.body_index == best_body and self._selectable(node.object_id):
                return int(node.object_id)
        return 0

    def _begin_viewport_panel(self) -> None:
        """Resolve the current dock layout before sizing or rendering the scene."""

        title = self.localizer.text("Viewport")
        if title != "Viewport":
            title += "###Viewport"
        imgui.begin(title, None, imgui.WindowFlags_.no_scrollbar.value)
        pos = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        panel_position = (float(pos.x), float(pos.y))
        self._viewport_panel_position = panel_position
        self._viewport_panel_size = (max(float(size.x), 1.0), max(float(size.y), 1.0))
        self._viewport_rect = _fit_image_rect(
            panel_position,
            self._viewport_panel_size,
            self._current_viewport_render_size(),
        )

    def _draw_viewport_contents(
        self,
        preview_name: str = "",
        *,
        session_busy: bool = False,
    ) -> None:
        image = self._viewport_image
        if image is None:
            imgui.text_disabled(self.localizer.text("No viewport image is available"))
        else:
            self._viewport_rect = _fit_image_rect(
                self._viewport_panel_position,
                self._viewport_panel_size,
                (image.width, image.height),
            )
            uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
            uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
            x, y, width, height = self._viewport_rect
            imgui.set_cursor_screen_pos(imgui.ImVec2(x, y))
            imgui.image(
                self.window.viewport_texture_ref(image),
                imgui.ImVec2(width, height),
                uv0,
                uv1,
            )
        x, y, w, h = self._viewport_rect
        imgui.push_clip_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), True)
        try:
            overlay = ImguiDraw2D()
            if not session_busy:
                st = self.session.perturb
                if st.active and not self.backend.caps.debug_draw:
                    node = self.session.node(st.node_id)
                    center = self._node_pose(node)[0] if node is not None else st.target_pos
                    draw_fallback(
                        self._camera_view(),
                        st,
                        self._viewport_rect,
                        (imgui.get_io().mouse_pos.x, imgui.get_io().mouse_pos.y),
                        center,
                        overlay,
                        self.window.style_scale,
                    )
                self.gizmo.draw_overlay(
                    self._camera_view(),
                    self._viewport_rect,
                    overlay,
                    style_scale=self.window.style_scale,
                )
                self.view_cube.draw(overlay, self.window.style_scale)
                self._draw_model_drop_overlay(overlay)
        finally:
            imgui.pop_clip_rect()
        if not session_busy:
            self.camera_preview.draw(
                self.window,
                self._viewport_rect,
                preview_name,
                self.localizer.text,
            )
        imgui.end()
        if not session_busy:
            self._draw_joint_limit_controls()
            self._draw_joint_gizmo_picker()

    def _draw_joint_limit_controls(self) -> None:
        """Make MIN/MAX ticks direct controls and reveal read-only values on dwell."""

        hits = self.gizmo.joint_limit_hits
        now = time.monotonic()
        if not self.session.paused or not hits:
            self._joint_limit_hover.reset()
            return
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
        )
        hovered_hit: JointLimitHit | None = None
        for hit in hits:
            x0, y0, x1, y1 = hit.rect
            # Keep the host clip outside the antialiased tick feedback.
            clip_pad = 2.0 * self.window.style_scale
            host_rect = _clipped_overlay_host_rect(
                self._viewport_rect,
                (x0, y0, x1, y1),
                clip_pad,
            )
            if host_rect is None:
                continue
            imgui.set_next_window_pos(
                imgui.ImVec2(host_rect[0], host_rect[1]),
                imgui.Cond_.always,
            )
            imgui.set_next_window_size(
                imgui.ImVec2(host_rect[2], host_rect[3]),
                imgui.Cond_.always,
            )
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
            imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))
            visible, _ = imgui.begin(
                f"{self.localizer.text('Joint')} {hit.label}"
                f"###joint_limit_{hit.joint_id}_{hit.label[:3]}",
                None,
                flags,
            )
            if visible:
                viewport_x, viewport_y, viewport_width, viewport_height = self._viewport_rect
                imgui.push_clip_rect(
                    imgui.ImVec2(viewport_x, viewport_y),
                    imgui.ImVec2(viewport_x + viewport_width, viewport_y + viewport_height),
                    True,
                )
                try:
                    imgui.set_cursor_screen_pos(imgui.ImVec2(x0, y0))
                    clicked = imgui.invisible_button(
                        f"##joint_limit_hit_{hit.joint_id}_{hit.label[:3]}",
                        imgui.ImVec2(max(x1 - x0, 1.0), max(y1 - y0, 1.0)),
                    )
                    hovered = imgui.is_item_hovered()
                    active = imgui.is_item_active()
                    if hovered or active:
                        hovered_hit = hit
                        self._draw_joint_limit_feedback(hit, hovered=hovered, active=active)
                    if clicked:
                        result = self.gizmo.apply_joint_limit(self.session, hit)
                        if not result.ok:
                            self.session.report_message(result.message, level="warning")
                finally:
                    imgui.pop_clip_rect()
            imgui.end()
            imgui.pop_style_var(2)

        keys = tuple(self._joint_limit_key(hit) for hit in hits)
        visible_key = self._joint_limit_hover.update(
            self._joint_limit_key(hovered_hit) if hovered_hit is not None else None,
            keys,
            now,
        )
        visible_hit = next(
            (hit for hit in hits if self._joint_limit_key(hit) == visible_key),
            None,
        )
        if visible_hit is not None:
            with _clipped_foreground_overlay_draw(self._viewport_rect) as draw:
                self.gizmo.draw_joint_limit_label(draw, visible_hit, self.window.style_scale)

    @staticmethod
    def _joint_limit_key(hit: JointLimitHit) -> tuple[int, int, str]:
        return hit.joint_id, hit.qpos_adr, hit.label[:3]

    def _draw_joint_limit_feedback(
        self,
        hit: JointLimitHit,
        *,
        hovered: bool,
        active: bool,
    ) -> None:
        """Repaint one endpoint tick with its pointer interaction state."""

        color = (
            axis_active_color(hit.semantic_color)
            if active
            else axis_hover_color(hit.semantic_color)
            if hovered
            else hit.semantic_color
        )
        with _clipped_foreground_overlay_draw(self._viewport_rect) as draw:
            draw.line(
                hit.tick_start,
                hit.tick_end,
                color,
                hit.tick_width,
                cap=hit.tick_cap,
            )

    def _draw_joint_gizmo_picker(self) -> None:
        """Draw a movable chooser near the click that selected a multi-joint link."""

        if not self.session.paused:
            return
        joints = self.gizmo.joint_choices(self.session)
        node = self.session.selected_node
        if not joints or node is None:
            self._joint_picker_node_id = -1
            return
        x, y, width, height = self._viewport_rect
        scale = self.window.style_scale
        if node.node_id != self._joint_picker_node_id:
            mouse = imgui.get_io().mouse_pos
            inside = x <= mouse.x <= x + width and y <= mouse.y <= y + height
            desired_x = mouse.x + 14.0 * scale if inside else x + 18.0 * scale
            desired_y = mouse.y + 14.0 * scale if inside else y + 18.0 * scale
            estimate_width = 280.0 * scale
            estimate_height = (74.0 + 28.0 * len(joints)) * scale
            min_x = x + 8.0 * scale
            min_y = y + 8.0 * scale
            max_x = max(min_x, x + width - estimate_width)
            max_y = max(min_y, y + height - estimate_height)
            imgui.set_next_window_pos(
                imgui.ImVec2(
                    min(max(min_x, desired_x), max_x),
                    min(max(min_y, desired_y), max_y),
                ),
                imgui.Cond_.always,
            )
            self._joint_picker_node_id = node.node_id
        imgui.set_next_window_bg_alpha(0.92)
        flags = (
            imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        visible, _ = imgui.begin(
            f"{self.localizer.text('Joint gizmo')}###viewport_joint_gizmo",
            None,
            flags,
        )
        if visible:
            imgui.text_disabled(node.name)
            imgui.separator()
            selected = self.gizmo.selected_joint_id(node.body_index)
            for joint in joints:
                supported = joint.type in ("hinge", "slide", "ball")
                imgui.begin_disabled(not supported)
                label = joint.name or f"joint {joint.joint_id}"
                clicked, _selected = imgui.selectable(
                    f"{label}  ({joint.type})##viewport-joint-{joint.joint_id}",
                    selected == joint.joint_id,
                )
                if clicked:
                    self.gizmo.select_joint(node.body_index, joint.joint_id)
                if (
                    supported
                    and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
                    and imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
                ):
                    self.gizmo.select_joint(node.body_index, joint.joint_id)
                    self._request_node_joint_focus(
                        next(
                            (
                                candidate
                                for candidate in self.session.nodes
                                if candidate.type is NodeType.JOINT
                                and candidate.joint_index == joint.joint_id
                            ),
                            node,
                        )
                    )
                imgui.end_disabled()
        imgui.end()

    def _draw_playback_widget(self) -> None:
        """Draw the final playback capsule at the viewport's top center."""

        caps = self.session.adapter.caps
        if not caps.simulation or not self._has_scene_content():
            return
        x, y, width, _height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale,
            PLAYBACK_CHROME_SCALE,
        )
        widget_width, widget_height = playback_size(scale, self.viewport_chrome.playback_controls)
        if widget_width <= 0.0 or widget_height <= 0.0:
            return
        widget_rect = (
            x + (width - widget_width) * 0.5,
            y + 12.0 * style_scale,
            x + (width + widget_width) * 0.5,
            y + 12.0 * style_scale + widget_height,
        )
        preview_rect = self.camera_preview.bounds
        if preview_rect is not None and _rectangles_overlap(widget_rect, preview_rect):
            # A tiny HiDPI viewport cannot expose two large overlays at once.
            # Keep the camera preview header reachable so it can be moved or
            # disabled instead of placing playback above its drag target.
            return
        clip_pad = OVERLAY_CLIP_PADDING * scale
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always.value,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
        )
        imgui.push_style_var(
            imgui.StyleVar_.window_padding,
            imgui.ImVec2(0.0, 0.0),
        )
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(0.0, 0.0),
        )
        visible, _ = imgui.begin(
            f"{self.localizer.text('Playback')}###viewport_playback", None, flags
        )
        if visible:
            take_playing = self.session.state_take_playing
            paused = self.session.paused and not take_playing
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                action = draw_playback(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    playing=not paused,
                    step_enabled=paused and not take_playing,
                    previous_enabled=self.session.can_step_back,
                    enabled=not self._scene_input_blocked(),
                    bindings=self.input_bindings,
                    labels=self._viewport_labels,
                    control_specs=self.viewport_chrome.playback_controls,
                )
            if action and self.viewport_chrome.dispatch("playback", action):
                pass
            elif action == "toggle":
                self._toggle_playback()
            elif action == "step":
                self.session.submit(cmd.Step(1))
            elif action == "previous":
                self.session.submit(cmd.StepBack())
            elif action in ("reset", "stop"):
                self._reset_playback()
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_tool_column_widget(self) -> None:
        """Draw the viewport tool capsule without construction geometry."""

        if not self._has_scene_content():
            return
        enabled = bool(self._state.has_selection and self._state.gizmo_available)
        # A column made entirely of disabled tools is visual noise and can
        # obscure joint selection. It returns as soon as one tool is usable.
        if not enabled:
            return
        x, y, _width, height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale,
            TOOL_CHROME_SCALE,
        )
        widget_width, widget_height = tool_column_size(scale, self.viewport_chrome.tool_groups)
        if widget_width <= 0.0 or widget_height <= 0.0:
            return
        clip_pad = OVERLAY_CLIP_PADDING * scale
        if height < widget_height + 120.0 * style_scale:
            return
        widget_rect = (
            x + 12.0 * style_scale,
            y + (height - widget_height) * 0.5,
            x + 12.0 * style_scale + widget_width,
            y + (height + widget_height) * 0.5,
        )
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        visible, _ = imgui.begin(f"{self.localizer.text('Tools')}###viewport_tools", None, flags)
        if visible:
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                action = draw_tool_column(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    mode=self.gizmo.mode,
                    space=self.gizmo.space,
                    snap=self._snap_latched or self.gizmo.snapping,
                    enabled=not self._scene_input_blocked(),
                    bindings=self.input_bindings,
                    labels=self._viewport_labels,
                    groups=self.viewport_chrome.tool_groups,
                )
            if action and self.viewport_chrome.dispatch("tool", action):
                pass
            elif action == "move":
                self.gizmo.set_mode("translate")
            elif action == "rotate":
                self.gizmo.set_mode("rotate")
            elif action == "frame":
                self.gizmo.toggle_space()
            elif action == "snap":
                self._snap_latched = not self._snap_latched
        imgui.end()
        imgui.pop_style_var()

    def _draw_context_hint_widget(self) -> None:
        """Draw caller-defined scene hints; defaults live in the status bar."""

        if self._scene_input_blocked() or not self._has_scene_content():
            return
        hints = self.tool_hints.resolve(surface="scene")
        if not hints:
            return
        x, y, width, height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale,
            HINT_CHROME_SCALE,
        )
        hint_font_scale = scale / max(style_scale, 1e-6)
        imgui.push_font(None, imgui.get_font_size() * hint_font_scale)
        measure = ImguiDraw2D(imgui.get_foreground_draw_list())
        widget_width, widget_height = tool_hints_size(
            measure,
            scale,
            hints,
            labels=self._viewport_labels,
            padding=True,
        )
        clip_pad = OVERLAY_CLIP_PADDING * scale
        if widget_width > width - 24.0 * style_scale:
            content_width = max(
                0.0,
                width - 24.0 * style_scale - 2.0 * OVERLAY_GEOMETRY.hint_padding_x * scale,
            )
            hints = fitting_tool_hints(
                measure,
                scale,
                hints,
                content_width,
                labels=self._viewport_labels,
            )
            if not hints:
                imgui.pop_font()
                return
            widget_width, widget_height = tool_hints_size(
                measure,
                scale,
                hints,
                labels=self._viewport_labels,
                padding=True,
            )
        widget_rect = (
            x + (width - widget_width) * 0.5,
            y + height - widget_height - 16.0 * style_scale,
            x + (width + widget_width) * 0.5,
            y + height - 16.0 * style_scale,
        )
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            imgui.pop_font()
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_inputs.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        visible, _ = imgui.begin(f"{self.localizer.text('Hints')}###viewport_hints", None, flags)
        if visible:
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                draw_scene_tool_hints(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    hints,
                    labels=self._viewport_labels,
                    size=(widget_width, widget_height),
                    pixel_size=self.window.pixels_to_points(1.0),
                )
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_font()

    def _context_tool_hint_variant(self) -> str:
        """Return the active input grammar independently of its surface."""

        state = self._state
        if state.ctrl or self.session.perturb.active:
            return "perturb"
        if self.gizmo.using:
            return "dragging"
        if not state.has_selection or not state.gizmo_available:
            return "camera"
        return "ready"

    def _update_status_context(self, ctx: PanelContext) -> None:
        """Latch status ownership on clicks, never on pointer travel or scrolling."""

        if not self._consume_scene_pointer_until_release and any(
            imgui.is_mouse_clicked(button) for button in (0, 1, 2)
        ):
            context = imgui.get_current_context()
            hovered = context.hovered_window
            if hovered is not None:
                owner = hovered.root_window
                # Dock tabs belong to the dock host, not the panel window.
                # Use ImGui's hit-tested tab ID (including clipping/scrolling),
                # rather than its navigation focus, which updates a frame later.
                if context.hovered_id:
                    owner = next(
                        (
                            window
                            for window in context.windows
                            if window.tab_id == context.hovered_id
                            and window.dock_node is not None
                            and window.dock_node.host_window is not None
                            and window.dock_node.host_window.id_ == hovered.id_
                        ),
                        owner,
                    )
                name = str(owner.name).rsplit("###", 1)[-1]
                if name == "Viewport" or name.startswith("joint_limit_"):
                    self._status_panel = "Viewport"
                elif name in ctx.status_hints_by_panel:
                    self._status_panel = name
        if self._status_panel != "Viewport" and self._status_panel not in ctx.status_hints_by_panel:
            self._status_panel = "Viewport"
        self._panel_status_hints = ctx.status_hints_by_panel.get(self._status_panel, ())

    def _status_tool_hints(self, *, loading: bool) -> tuple[ToolHint, ...]:
        if loading:
            return ()
        edit = self._precise_gizmo_edit
        if edit is not None:
            defaults = precise_input_status_hints(edit, self.localizer.text)
            return self.tool_hints.resolve(defaults, surface="status")
        if self._scene_input_blocked():
            return self.tool_hints.resolve((), surface="status")
        if self._gizmo_hint_hover.visible:
            defaults = default_tool_hints(
                "ready_minimal",
                self.input_bindings,
                self._viewport_labels,
            )
        elif self._status_panel != "Viewport":
            defaults = self._panel_status_hints
        elif self._has_scene_content():
            defaults = tuple(
                hint
                for hint in default_tool_hints(
                    self._context_tool_hint_variant(),
                    self.input_bindings,
                    self._viewport_labels,
                )
                if hint.hint_id != "gizmo.type_value"
            )
        else:
            defaults = ()
        # Selection is application state, not part of a panel's hover/gesture
        # grammar. Compose its action before the clicked panel's own hints.
        if self._selection_clear_enabled():
            defaults = (
                ToolHint(
                    "key",
                    "Esc",
                    self._viewport_labels.clear_selection,
                    hint_id="selection.clear",
                ),
                *defaults,
            )
        if self._status_panel == "Viewport" and self.session.can_step_back:
            defaults = (
                ToolHint(
                    "key",
                    self.input_bindings.label(InputAction.STEP_BACK),
                    self._viewport_labels.rewind,
                    hint_id="playback.previous",
                ),
                *defaults,
            )
        return self.tool_hints.resolve(defaults, surface="status")

    def _selection_clear_enabled(self) -> bool:
        """Allow navigation alongside selection; protect edits that own the target."""

        if imgui.is_any_item_active():
            context = imgui.get_current_context()
            window = context.active_id_window
            # ImGui assigns a window's MoveId to an empty-space left press,
            # even when the docked window is not moving. That focus capture
            # is not a widget edit and must not suppress orbit/Shift-pan Esc.
            if (
                window is None
                or context.active_id != window.move_id
                or context.moving_window is not None
            ):
                return False
        return bool(
            self.session.paused
            and not self.session.state_take_playing
            and self.session.selected_node is not None
            and not (
                self.router.held and self.router.claim in (gs.Claim.OBJECT_GIZMO, gs.Claim.PERTURB)
            )
            and not self.session.perturb.active
            and not self.gizmo.using
            and not self.gizmo.keyboard_using
        )

    def _has_scene_content(self) -> bool:
        source = self.session.source
        if source is None:
            return False
        lights = getattr(getattr(source, "lights", None), "lights", ())
        cameras = getattr(source, "cameras", ())
        return bool(getattr(source, "instance_count", 0) or lights or cameras)

    def _draw_application_status_bar(self, *, loading: bool = False) -> None:
        """Draw persistent selection, simulation, backend, and frame-rate status."""

        viewport = imgui.get_main_viewport()
        scale = self.window.style_scale
        height = 28.0 * scale
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_nav_focus.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        visible = imgui.internal.begin_viewport_side_bar(
            "Status###application_status",
            viewport,
            imgui.Dir.down,
            height,
            flags,
        )
        if visible:
            origin = imgui.get_window_pos()
            size = imgui.get_window_size()
            if loading:
                selected_name = "no selection"
                has_selection = False
                state = "static"
                sim_time = 0.0
                sim_step = 0
            else:
                selected = self.session.selected_node
                selected_name = selected.name if selected is not None else "no selection"
                has_selection = selected is not None
                caps = self.session.adapter.caps
                state = (
                    "static"
                    if not caps.simulation
                    else "paused"
                    if self.session.paused and not self.session.state_take_playing
                    else "running"
                )
                sim_time = float(self.session.frame.time)
                sim_step = int(self.session.frame.step)
            active_status = None if loading else self.output.active_status()
            status_text = (
                ""
                if active_status is None
                else _status_message_for_bar(
                    active_status.text,
                    selected_name,
                    active_status.level,
                )
            )
            status_layout = draw_status(
                ImguiDraw2D(),
                (origin.x, origin.y),
                size.x,
                size.y,
                self.theme,
                scale,
                selected=(
                    ""
                    if self._precise_gizmo_edit is not None
                    else selected_name
                    if has_selection
                    else self._viewport_labels.no_selection
                ),
                # The precise-input title already identifies its target. Give
                # Enter / Esc / U the reclaimed status width at extreme scale.
                state=state,
                sim_time=sim_time,
                step=sim_step,
                metric_mode=self._status_metric_mode,
                backend=(
                    "OpenGL" if self.backend.caps.name == "opengl" else str(self.backend.caps.name)
                ),
                dt=_simulation_timestep(self.session.adapter, loading=loading),
                fps=self._frame_rate.value,
                status=status_text,
                status_level="info" if active_status is None else active_status.level,
                tool_hints=self._status_tool_hints(loading=loading),
                labels=self._viewport_labels,
                pixel_size=self.window.pixels_to_points(1.0),
            )
            if status_layout.metric_rect is not None and not loading:
                x0, y0, x1, y1 = status_layout.metric_rect
                imgui.set_cursor_screen_pos(imgui.ImVec2(x0, y0))
                if imgui.invisible_button(
                    "##status_simulation_metric",
                    imgui.ImVec2(x1 - x0, y1 - y0),
                ):
                    self._toggle_status_metric()
                hovered = imgui.is_item_hovered()
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    imgui.set_clipboard_text(status_layout.metric_exact)
                if hovered:
                    switch = (
                        self._viewport_labels.show_steps
                        if self._status_metric_mode == "time"
                        else self._viewport_labels.show_time
                    )
                    imgui.set_tooltip(f"{switch} · {self._viewport_labels.copy_exact}")
        imgui.end()
        imgui.pop_style_var(2)

    def _capture_viewport(self) -> None:
        self._capture_viewport_requested = False
        output = Path("output") / f"viewport-{time.strftime('%Y%m%d-%H%M%S')}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            captured = bool(self.backend.capture(output))
        except Exception as exc:
            self.session.report_message(
                f"{self.localizer.text('Viewport capture failed')}: {exc}", level="error"
            )
            return
        if captured:
            self.session.report_message(
                f"{self.localizer.text('Saved viewport capture to')} {output}", level="success"
            )
        else:
            self.session.report_message(
                self.localizer.text("Viewport capture failed"), level="error"
            )

    def _toggle_viewport_recording(self) -> None:
        if self._viewport_recorder is not None:
            self._stop_viewport_recording()
            return
        from ..recording import VideoRecorder

        target = getattr(self.backend, "target", None)
        if target is None or not hasattr(target, "read_color"):
            self.session.report_message(
                self.localizer.text("Viewport recording is unavailable"), level="error"
            )
            return
        output = Path("output") / f"viewport-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        try:
            recorder = VideoRecorder(
                output,
                (max(1, int(target.width)), max(1, int(target.height))),
                fps=30.0,
            )
        except Exception as exc:
            self.session.report_message(
                f"{self.localizer.text('Viewport recording failed')}: {exc}", level="error"
            )
            return
        self._viewport_recorder = recorder
        self._viewport_recording_path = output
        self._viewport_record_elapsed = 1.0 / 30.0
        self.session.report_message(
            f"{self.localizer.text('Recording viewport to')} {output}", level="success"
        )

    def _record_viewport_frame(self, dt: float) -> None:
        recorder = self._viewport_recorder
        if recorder is None:
            return
        period = 1.0 / 30.0
        self._viewport_record_elapsed += max(0.0, min(float(dt), 0.1))
        frames = min(3, int(self._viewport_record_elapsed / period))
        if frames <= 0:
            return
        try:
            image = self.backend.target.read_color(flip=True)[..., :3]
            for _ in range(frames):
                recorder.append(image)
        except Exception as exc:
            self._stop_viewport_recording(report=False)
            self.session.report_message(
                f"{self.localizer.text('Viewport recording stopped')}: {exc}", level="error"
            )
            return
        self._viewport_record_elapsed -= frames * period

    def _stop_viewport_recording(self, *, report: bool = True) -> None:
        recorder = getattr(self, "_viewport_recorder", None)
        if recorder is None:
            return
        path = self._viewport_recording_path
        frames = int(getattr(recorder, "frames", 0))
        self._viewport_recorder = None
        self._viewport_recording_path = None
        self._viewport_record_elapsed = 0.0
        try:
            recorder.close()
        except Exception as exc:
            if report:
                self.session.report_message(
                    f"{self.localizer.text('Viewport recording failed')}: {exc}", level="error"
                )
            return
        if report and path is not None:
            self.session.report_message(
                f"{self.localizer.text('Saved')} {frames} "
                f"{self.localizer.text('viewport frame(s) to')} {path}",
                level="success",
            )

    def _toggle_playback(self) -> None:
        if self.session.state_take_recording:
            self.session.submit(cmd.StopStateTakeRecording())
        elif self.session.state_take_playing:
            self.session.submit(cmd.PauseStateTake())
        else:
            self.session.submit(cmd.Play() if self.session.paused else cmd.Pause())

    def _reset_playback(self) -> None:
        if self.session.state_take_recording:
            self.session.submit(cmd.StopStateTakeRecording())
        elif self.session.state_take_cursor >= 0:
            self.session.submit(cmd.PauseStateTake())
            self.session.submit(cmd.SeekStateTake(0))
        else:
            if not self.session.paused:
                self.session.submit(cmd.Pause())
            self.session.submit(cmd.Reset())

    def _draw_model_drop_overlay(self, overlay: ImguiDraw2D) -> None:
        source = self.session.source
        empty = source is not None and source.instance_count == 0
        notice = self._model_drop_notice if time.monotonic() < self._model_drop_notice_until else ""
        caps = self.session.adapter.caps
        dragging = self.window.file_drag_active and (caps.asset_loading or caps.scene_files)
        if not empty and not notice and not dragging:
            return
        empty_hint = (
            "Drop a .mojive.json scene here\nFile > Open Scene...  ·  Entity > Create"
            if caps.scene_files
            else "Drop an MJCF or URDF model here\nFile > Open Model...  ·  Add Model..."
        )
        if dragging:
            message = (
                "Release to add model(s)"
                if caps.model_composition and self.session.scene_models
                else "Release to open model"
            )
        else:
            message = notice or empty_hint
        self._draw_center_notice(overlay, message, border=dragging)

    def _sync_session_status(self) -> None:
        revision = int(getattr(self.session, "message_revision", 0))
        if revision == self._seen_message_revision:
            return
        self._seen_message_revision = revision
        self.output.publish(
            self.session.last_message,
            level=getattr(self.session, "last_message_level", "info"),
            duration=getattr(self.session, "last_message_duration", 5.0),
        )

    def _draw_center_notice(
        self,
        overlay: ImguiDraw2D,
        message: str,
        *,
        border: bool = False,
    ) -> None:
        lines = message.splitlines()
        sizes = [overlay.text_size(line) for line in lines]
        scale = self.window.style_scale
        pad_x, pad_y = 18.0 * scale, 12.0 * scale
        width = max(size[0] for size in sizes) + 2.0 * pad_x
        height = sum(size[1] for size in sizes) + 2.0 * pad_y + (len(lines) - 1) * 3.0 * scale
        x, y, w, h = self._viewport_rect
        left = x + (w - width) * 0.5
        top = y + (h - height) * 0.5
        if border:
            overlay.rect(
                (x + 3.0 * scale, y + 3.0 * scale),
                (x + w - 3.0 * scale, y + h - 3.0 * scale),
                (0.95, 0.68, 0.24, 0.95),
                2.0 * scale,
                rounding=8.0 * scale,
            )
        overlay.rect_filled(
            (left, top),
            (left + width, top + height),
            (0.08, 0.09, 0.11, 0.88),
            rounding=7.0 * scale,
        )
        cursor_y = top + pad_y
        for line, size in zip(lines, sizes, strict=True):
            overlay.text(
                (left + (width - size[0]) * 0.5, cursor_y),
                (0.93, 0.94, 0.95, 1.0),
                line,
            )
            cursor_y += size[1] + 3.0 * scale

    def frame_needs(self) -> FrameNeeds:
        needs = (
            FrameNeeds(poses=True)
            .merge(self.panels.frame_needs())
            .merge(self.gizmo.frame_needs(self.session))
        )
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
            select_model_camera=self.select_model_camera,
            focus_node=self.request_node_focus,
            focus_joint=self.request_joint_focus,
            request_rename=self.request_rename,
            request_model_rename=self.request_model_rename,
            request_texture_import=self._open_texture_dialog,
            request_geometry_resource_import=self._open_geometry_resource_dialog,
            request_model_asset_import=self._open_model_asset_import_dialog,
            request_model_asset_replace=self._open_model_asset_replace_dialog,
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
            language=self.localizer.language.value,
            translate=self.localizer.text,
            set_language=self.set_language,
            set_precise_input_memory=self.set_precise_input_choice_memory,
            set_view_selection_padding=self.set_view_selection_padding,
            viewport_overlay_scale=self._viewport_overlay_scale,
            set_viewport_overlay_scale=self.set_viewport_overlay_scale,
            input_bindings=self.input_bindings,
            set_input_binding=self.set_input_binding,
            reset_input_bindings=self.reset_input_bindings,
            font_report=self.window.font_report,
            output=self.output,
        )

    def _cursor_ray(self, cursor: tuple[float, float]):
        ndc = ndc_from_viewport(cursor[0], cursor[1], self._viewport_rect)
        return unproject(self._camera_view(), *ndc)

    def _node_pose(self, node) -> tuple[np.ndarray, np.ndarray]:
        return node_world_pose(self.session, node)

    def apply_keys(self, keys: Keys) -> None:
        if keys.toggle_pause:
            self._toggle_playback()
        for _ in range(keys.step_back_count):
            if not self.session.submit(cmd.StepBack()):
                break
        if keys.clear_selection:
            self.session.submit(cmd.Select(0))
            hierarchy = self.panels.get("Hierarchy")
            if hierarchy is not None:
                clear_selection = getattr(hierarchy, "clear_selection", None)
                if clear_selection is not None:
                    clear_selection()
            self.gizmo.cancel()
            self.router.abort()
            # Esc may be pressed during orbit/pan, including before it crosses
            # click slop. Consume the rest of that press so its release cannot
            # restart a gesture or pick the just-cleared object again.
            self._consume_scene_pointer_until_release = any(
                imgui.is_mouse_down(button) for button in range(3)
            )
        if keys.gizmo_translate:
            self.gizmo.set_mode("translate")
        if keys.gizmo_rotate:
            self.gizmo.set_mode("rotate")
        if keys.gizmo_space:
            self.gizmo.toggle_space()
