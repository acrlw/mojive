"""Editable UI review state, per-icon overrides and production tuning defaults."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from mojive.geometry2d.curves import CORNER_SMOOTHING
from mojive.ui import gizmo as gizmo_ui
from mojive.ui import theme as theme_mod
from mojive.ui.icons import (
    ICON_ALIGNMENT_CHOICES,
    ICON_ALIGNMENT_EDITABLE_ICONS,
    ICON_GLYPH_ALIGNMENT_DEFAULTS,
    ICON_GLYPH_PADDING_DEFAULTS,
    ICON_GLYPH_STROKE_DEFAULTS,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_GROUP_STROKE_DEFAULTS,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_TUNING_DEFAULTS,
    STATUS_MOUSE_DEFAULT_WIDTH,
    IconTuning,
    icon_component_group,
)
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.messages import OutputBuffer
from mojive.ui.paint_protocol import Draw2D
from mojive.ui.panels.keyframes import KeyframesPanel
from mojive.ui.panels.output import OutputPanel
from mojive.ui.perturb import OUTLINE_CORNER_RADIUS_PT
from mojive.ui.viewcube import DEFAULT_SELECTION_PADDING
from mojive.ui.viewport_widgets import DEFAULT_VIEWPORT_OVERLAY_SCALE, OVERLAY_GEOMETRY

from ..ui_capsule_geometry import CAPSULE_SMOOTHING
from ..ui_redesign import RedesignState


@dataclass
class ProbeState:
    painter: Callable[[], Draw2D] = ImguiDraw2D
    renderer: str = "opengl"
    page: str = "Workspace"
    redesign: RedesignState = field(default_factory=RedesignState)
    imgui_rounding: float = theme_mod.DEFAULT_CORNER_RADIUS
    imgui_example_value: float = 0.0
    imgui_example_enabled: bool = True
    capsule_smoothing: float = CAPSULE_SMOOTHING
    capsule_outline: str = "Soft white"
    playback_smoothing: float = CAPSULE_SMOOTHING
    tool_smoothing: float = CAPSULE_SMOOTHING
    mouse_smoothing: float = CORNER_SMOOTHING
    transform_smoothing: float = CORNER_SMOOTHING
    joint_smoothing: float = CORNER_SMOOTHING
    view_smoothing: float = CORNER_SMOOTHING
    perturb_smoothing: float = CORNER_SMOOTHING

    geometry_tab: str = "Playback"
    geometry_tab_initialized: bool = False
    icon_library_tab: str = "Overview"
    preview_icon_library: bool = False
    show_playback: bool = True
    show_tool_column: bool = True
    show_joint_gizmos: bool = True
    show_context_hints: bool = False
    show_icon_bounds: bool = False
    show_state_circles: bool = False
    show_construction_notes: bool = False
    highlight_g3: bool = False
    optical_capsule_spacing: bool = True
    playing: bool = False
    active_tool: str = "move"
    gizmo_space: str = "world"
    gizmo_style: int = 1
    frame: int = 1
    remember_input: bool = True
    viewport_overlay_scale: float = DEFAULT_VIEWPORT_OVERLAY_SCALE
    position_snap: float = gizmo_ui.DEFAULT_TRANSLATION_SNAP_M
    rotation_snap: float = gizmo_ui.DEFAULT_ROTATION_SNAP_DEG
    tick_scale: float = gizmo_ui.DEFAULT_ROTATION_TICK_SCALE
    selection_padding: float = DEFAULT_SELECTION_PADDING
    corner_radius: float = OUTLINE_CORNER_RADIUS_PT
    scene_icons: bool = True
    influence_volumes: bool = True
    value_open: bool = False
    value_mode: int = 0
    value: float = 45.0
    unit: int = 0
    joint_value_open: bool = False
    joint_value_title: str = "slide_joint"
    joint_value: float = 0.0
    joint_value_unit: str = "m"
    output_filter: str = ""
    output_panel: OutputPanel = field(default_factory=OutputPanel)
    timeline_panel: KeyframesPanel = field(default_factory=KeyframesPanel)
    timeline_session: object = None
    angular_degrees: bool = False
    output_buffer: OutputBuffer | None = None
    diagnostic_levels: set[str] = field(default_factory=lambda: {"info", "warning", "error"})
    hinge_ctrl: float = 0.25
    slide_ctrl: float = 0.0
    weld_enabled: bool = True
    connect_enabled: bool = False
    hinge_position: float = 0.35
    slide_position: float = 0.0
    hierarchy_filter: str = ""
    hierarchy_kind: int = 0
    hierarchy_selection: int = 2
    control_filter: str = ""
    control_sort_by_name: bool = False
    joint_filter: str = ""
    joint_sort_by_name: bool = False
    workspace_right_tab: str = "Control"
    hierarchy_visibility: list[bool] = field(
        default_factory=lambda: [True, True, False, True, True, True]
    )
    asset_selection: int = 1
    asset_filter: str = ""
    asset_type: int = 0
    sensor_index: int = 0
    helper_selection: int = 1
    camera_projection: int = 0
    settings_page: int = 1
    settings_filter: str = ""
    language: int = 1
    outline_enabled: bool = True
    tonemap_enabled: bool = True
    msaa_enabled: bool = True
    debug_view: int = 0
    debug_labels: int = 0
    debug_frames: int = 0
    sim_running: bool = False
    aux_tab: str = "Output"
    mujoco_groups: list[bool] = field(default_factory=lambda: [True] * 42)
    render_flags: list[bool] = field(default_factory=lambda: [True] * 8)
    visual_flags: list[bool] = field(default_factory=lambda: [True] * 27)
    overlay_icon_radius: int = int(OVERLAY_GEOMETRY.icon_radius)
    overlay_radial_step: int = int(OVERLAY_GEOMETRY.radial_step)
    icon_adjustment_group: str = "Viewport tools"
    icon_padding_by_group: dict[str, float] = field(
        default_factory=lambda: dict(ICON_GROUP_LAYOUT_DEFAULTS)
    )
    icon_padding_by_glyph: dict[str, float] = field(default_factory=dict)
    icon_padding_draft_by_group: dict[str, float] = field(default_factory=dict)
    icon_alignment_by_glyph: dict[str, str] = field(default_factory=dict)
    icon_stroke_by_group: dict[str, float] = field(
        default_factory=lambda: dict(ICON_GROUP_STROKE_DEFAULTS)
    )
    icon_stroke_by_glyph: dict[str, float] = field(default_factory=dict)
    icon_stroke_draft_by_group: dict[str, float] = field(default_factory=dict)
    overlay_center_step: int = int(OVERLAY_GEOMETRY.center_step)
    tool_group_gap: int = int(OVERLAY_GEOMETRY.tool_group_gap)
    divider_width: int = int(OVERLAY_GEOMETRY.divider_width)
    construction_playback_scale: float = 3.0
    construction_tool_scale: float = 1.5
    tool_stroke_width: float = OVERLAY_GEOMETRY.tool_stroke
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP
    move_head_scale: float = ICON_TUNING_DEFAULTS.move_head_scale
    scale_handle_scale: float = ICON_TUNING_DEFAULTS.scale_handle_scale
    snap_endpoint_scale: float = ICON_TUNING_DEFAULTS.snap_endpoint_scale
    key_fit_arm_length: float = ICON_TUNING_DEFAULTS.key_fit_arm_length
    reset_head_scale: float = ICON_TUNING_DEFAULTS.reset_head_scale
    hint_control_height: int = int(OVERLAY_GEOMETRY.hint_control_height)
    hint_padding_x: int = int(OVERLAY_GEOMETRY.hint_padding_x)
    hint_padding_y: int = int(OVERLAY_GEOMETRY.hint_padding_y)
    hint_input_gap: int = int(OVERLAY_GEOMETRY.hint_input_gap)
    hint_group_gap: int = int(OVERLAY_GEOMETRY.hint_group_gap)
    hint_chord_gap: int = int(OVERLAY_GEOMETRY.hint_chord_gap)
    hint_key_padding_x: int = int(OVERLAY_GEOMETRY.hint_key_padding_x)
    hint_mouse_width: int = int(OVERLAY_GEOMETRY.hint_mouse_width)
    hint_mouse_stroke: float = OVERLAY_GEOMETRY.hint_mouse_stroke
    hint_mouse_button_width_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_width_ratio
    hint_mouse_button_shell_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
    hint_mouse_button_height_ratio: float = OVERLAY_GEOMETRY.hint_mouse_button_height_ratio
    hint_mouse_wheel_width_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_width_ratio
    hint_mouse_wheel_height_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_height_ratio
    hint_mouse_wheel_gap_ratio: float = OVERLAY_GEOMETRY.hint_mouse_wheel_gap_ratio

    def icon_padding_for(self, group: str) -> float:
        return self.icon_padding_by_group.get(group, ICON_GROUP_LAYOUT_DEFAULTS[group])

    def set_icon_padding_for(self, group: str, padding: float) -> None:
        self.icon_padding_by_group[group] = float(padding)

    def icon_padding_for_glyph(self, name: str) -> float:
        return self.icon_padding_by_glyph.get(
            name,
            ICON_GLYPH_PADDING_DEFAULTS.get(
                name, self.icon_padding_for(icon_component_group(name))
            ),
        )

    def set_icon_padding_for_glyph(self, name: str, padding: float) -> None:
        self.icon_padding_by_glyph[name] = float(padding)

    def icon_alignment_for_glyph(self, name: str) -> str | None:
        return self.icon_alignment_by_glyph.get(name, ICON_GLYPH_ALIGNMENT_DEFAULTS.get(name))

    def set_icon_alignment_for_glyph(self, name: str, alignment: str) -> None:
        if name not in ICON_ALIGNMENT_EDITABLE_ICONS:
            raise ValueError(f"icon alignment is not editable for {name!r}")
        if alignment not in ICON_ALIGNMENT_CHOICES:
            raise ValueError(f"icon alignment must be one of {ICON_ALIGNMENT_CHOICES!r}")
        self.icon_alignment_by_glyph[name] = alignment

    def icon_stroke_for(self, group: str) -> float:
        return self.icon_stroke_by_group.get(group, ICON_GROUP_STROKE_DEFAULTS[group])

    def set_icon_stroke_for(self, group: str, stroke_width: float) -> None:
        self.icon_stroke_by_group[group] = float(stroke_width)

    def icon_stroke_for_glyph(self, name: str) -> float:
        return self.icon_stroke_by_glyph.get(
            name,
            ICON_GLYPH_STROKE_DEFAULTS.get(name, self.icon_stroke_for(icon_component_group(name))),
        )

    def set_icon_stroke_for_glyph(self, name: str, stroke_width: float) -> None:
        self.icon_stroke_by_glyph[name] = float(stroke_width)

    def icon_tuning(self) -> IconTuning:
        return IconTuning(
            move_head_scale=self.move_head_scale,
            scale_handle_scale=self.scale_handle_scale,
            snap_endpoint_scale=self.snap_endpoint_scale,
            key_fit_arm_length=self.key_fit_arm_length,
            reset_head_scale=self.reset_head_scale,
        )

    def concept_mouse_width(self) -> float:
        """Return the master width that preserves the chosen status aspect ratio."""

        return (
            STATUS_MOUSE_DEFAULT_WIDTH
            * self.hint_mouse_width
            / OVERLAY_GEOMETRY.hint_mouse_width
            * OVERLAY_GEOMETRY.hint_control_height
            / self.hint_control_height
        )
