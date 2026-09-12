"""App: support."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import imgui

from mojive.config import (
    InteractionConfig,
)
from mojive.interaction.input import InputClaim
from mojive.log import get_logger
from mojive.ui.draw2d import ImguiDraw2D
from mojive.ui.gizmo import PreciseGizmoInput
from mojive.ui.theme import Theme
from mojive.ui.viewport_widgets import (
    ToolHint,
)

CLICK_SLOP_PT = 4.0
PRECISE_GIZMO_WIDTH_PT = 204.0
JOINT_LIMIT_LABEL_DELAY_SECONDS = 0.5
JOINT_LIMIT_HOVER_GRACE_SECONDS = 0.12
PRECISE_GIZMO_HINT_DELAY_SECONDS = 0.5
APPLICATION_STATUS_HEIGHT_PT = 28.0
JOINT_FOCUS_MARGIN = 1.5
JOINT_FOCUS_OBLIQUE_DEGREES = 35.0
JOINT_FOCUS_OCCLUSION_NEIGHBORHOOD_DEGREES = 25.0
VIEWPORT_DOUBLE_CLICK_SECONDS = 0.3
VIEWPORT_DOUBLE_CLICK_RADIUS_PT = 6.0
STEP_BACK_REPEAT_DELAY_SECONDS = 0.35
STEP_BACK_REPEAT_RATE_SECONDS = 0.1
_DEFAULT_INTERACTIONS = InteractionConfig()
_NO_INPUT_CLAIM = InputClaim()
_SELECTION_BOX_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    np.float32,
)
_SELECTION_BOX_EDGES = np.asarray(
    (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
    np.intp,
)
log = get_logger("ui")
PICK_SCREEN_RADIUS_PT = 40.0


def _model_filters(caps) -> list[str]:
    patterns = " ".join(f"*{extension}" for extension in caps.model_formats)
    return ["Supported models", patterns] if caps.asset_loading and patterns else []


SCENE_SUFFIX = ".mojive.json"
LEGACY_SCENE_SUFFIX = ".forge.json"
SCENE_SUFFIXES = (SCENE_SUFFIX, LEGACY_SCENE_SUFFIX)


def _scene_filters(caps, *, saving: bool = False) -> list[str]:
    filters = ["Mojive scenes (*.mojive.json, *.forge.json)", "*.mojive.json *.forge.json"]
    if saving:
        if caps.supports("mujoco.mjcf"):
            filters += ["MuJoCo XML / MJCF (*.xml, *.mjcf)", "*.xml *.mjcf"]
    else:
        filters += _model_filters(caps)
    return filters


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
    gizmo_dimensions: bool = False
    gizmo_space: bool = False
    gizmo_axis: int = -1


@dataclass(frozen=True)
class _ApplyModelEdits:
    commands: tuple


@dataclass(frozen=True)
class _ModelLoadJob:
    action: str
    path: Path
    command: Any
    completed: Any = None


@dataclass(frozen=True)
class _ModelLoadCompletion:
    job: _ModelLoadJob
    message: str
    started: float
    prepared: float
    resources_ready: float


@dataclass
class _FrameRateDisplay:
    """Low-pass and rate-limit the status-bar FPS readout."""

    smoothed_dt: float = 1.0 / 60.0
    value: float = 60.0
    elapsed: float = 0.0

    def update(self, dt: float) -> float:
        dt = max(float(dt), 1e-6)
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
