"""Panel specimens and a live production Keyframes review session."""

from __future__ import annotations

import math
from functools import partial

from imgui_bundle import imgui

from mojive.ui.compound_fields import draw_joined_field_frame
from mojive.ui.icons import draw_concept_icon, draw_icon_label
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.messages import OutputBuffer
from mojive.ui.panels import PanelContext, button_row_layout, button_width, search_input
from mojive.ui.panels import keyframes as keyframes_panel_module
from mojive.ui.panels import output as output_panel_module
from mojive.ui.panels.hierarchy import disclosure_triangle
from mojive.ui.text_layout import text_line_y
from mojive.ui.viewport_widgets import ToolHint, default_tool_hints, draw_status

from .fixtures import (
    AXIS_BADGE_ACTIVE,
    AXIS_BADGE_COLORS,
    AXIS_BADGE_HOVERED,
    AXIS_BADGE_TEXT,
    CONCEPT_THEME,
    PANEL_CANVAS_SIZE,
    PROBE_FRAME_SAMPLES,
    PROBE_PLOT_SAMPLES,
)
from .layout import (
    _draw_segmented,
    _flags,
    _probe_checkbox,
    _property_label,
    _table_text,
    _virtual_canvas_size,
    _wrapped_tabs,
)
from .settings import _draw_settings, _settings_properties
from .state import ProbeState
from .widgets import (
    _draw_concept_control_icon,
    _draw_concept_projection_icon,
    _draw_icon_library_command_icon,
    _draw_icon_library_label,
    _preview_search_input,
    _preview_searchable_header,
)


def _draw_keyframes(size, scale: float, state: ProbeState) -> None:
    """Exercise the production timeline against a real tiny simulation."""
    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.scene.assets import resolve
    from mojive.session import Session

    if not imgui.begin_child("Keyframes###ProbeKeyframes", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    if state.timeline_session is None:
        state.timeline_session = Session(MuJoCoAdapter(resolve("joint_types")))
        session = state.timeline_session
        session.submit(cmd.Pause())
        for count in (0, 100, 160):
            if count:
                session.submit(cmd.Step(count))
                session.tick(FrameNeeds.none(), wall_dt=0)
            session.submit(cmd.CaptureSceneSnapshot())
    session = state.timeline_session
    session.tick(FrameNeeds.none(), wall_dt=min(0.05, imgui.get_io().delta_time))
    ctx = PanelContext(session, None, theme=CONCEPT_THEME, style_scale=scale, painter=state.painter)
    state.timeline_panel.follow_mode_icon_drawer = (
        partial(_draw_icon_library_label, state=state)
        if state.preview_icon_library
        else draw_icon_label
    )
    original_icon = keyframes_panel_module._draw_command_icon
    if state.preview_icon_library:
        keyframes_panel_module._draw_command_icon = lambda *args, **kwargs: (
            _draw_icon_library_command_icon(
                *args,
                context_scale=scale,
                state=state,
                **kwargs,
            )
        )
    try:
        state.timeline_panel.draw(ctx)
    finally:
        keyframes_panel_module._draw_command_icon = original_icon
    imgui.end_child()


def _draw_output(size, state: ProbeState, scale: float) -> None:
    if not imgui.begin_child("Output###ProbeOutput", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Output")
    imgui.separator()
    if state.output_buffer is None:
        state.output_buffer = OutputBuffer()
        for level, message in (
            ("info", "Loaded scene.xml"),
            ("warning", "Joint limit reached"),
            ("error", "Model could not be compiled"),
        ):
            state.output_buffer.write(message, level=level, timestamp="09:44:25")
    ctx = PanelContext(
        None,
        None,
        theme=CONCEPT_THEME,
        style_scale=scale,
        painter=state.painter,
        output=state.output_buffer,
    )
    original_search = output_panel_module.search_input
    if state.preview_icon_library:
        output_panel_module.search_input = lambda *args, **kwargs: search_input(
            *args,
            **kwargs,
            icon_drawer=lambda *values: _draw_concept_control_icon(
                *values,
                state=state,
            ),
        )
    try:
        state.output_panel.draw(ctx)
    finally:
        output_panel_module.search_input = original_search
    imgui.end_child()


def _begin_gallery_panel(title: str, item_id: str, size) -> bool:
    opened = imgui.begin_child(
        f"{title}###{item_id}",
        size,
        imgui.ChildFlags_.borders.value,
    )
    if opened:
        imgui.text(title)
        imgui.separator()
    return opened


def _begin_gallery_properties(item_id: str) -> bool:
    flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if not imgui.begin_table(item_id, 2, flags):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.64)
    return True


def _draw_search_header(
    state: ProbeState,
    item_id: str,
    hint: str,
    value: str,
    sort_by_name: bool,
    *,
    state_order: str,
) -> tuple[str, bool]:
    """Use the production searchable-list header with specimen state."""

    _changed, value, _sort_changed, sort_by_name = _preview_searchable_header(
        state,
        f"##{item_id}",
        value,
        sort_by_name,
        hint=hint,
        search_tooltip=hint,
        clear_tooltip="Clear search",
        state_order=state_order,
        translate=lambda text: text,
    )
    return value, sort_by_name


def _draw_copy_buttons(item_id: str, labels: tuple[str, str]) -> None:
    flags = _flags(
        imgui.TableFlags_.sizing_stretch_same,
        imgui.TableFlags_.no_pad_outer_x,
    )
    if not imgui.begin_table(f"##{item_id}-copy", 2, flags):
        return
    for label in labels:
        imgui.table_next_column()
        imgui.button(f"{label}##{item_id}", imgui.ImVec2(-1.0, 0.0))
    imgui.end_table()


def _draw_control_content(state: ProbeState) -> None:
    if imgui.collapsing_header("actuators", imgui.TreeNodeFlags_.default_open.value):
        state.control_filter, state.control_sort_by_name = _draw_search_header(
            state,
            "probe-actuator",
            "Search actuators",
            state.control_filter,
            state.control_sort_by_name,
            state_order="ctrl / act",
        )
        _draw_copy_buttons("probe-actuator-state", ("Copy ctrl", "Copy act"))
        if _begin_gallery_properties("##probe-control-actuators"):
            _property_label("hinge_pos")
            _, state.hinge_ctrl = imgui.slider_float(
                "##probe-hinge-ctrl", state.hinge_ctrl, -1.0, 1.0, "%+.3f"
            )
            _property_label("slide_pos")
            _, state.slide_ctrl = imgui.slider_float(
                "##probe-slide-ctrl", state.slide_ctrl, -1.0, 1.0, "%+.3f"
            )
            imgui.end_table()

    if imgui.collapsing_header(
        "equality", imgui.TreeNodeFlags_.default_open.value
    ) and _begin_gallery_properties("##probe-control-equality"):
        for item_id, name, enabled in (
            ("weld", "eq_weld_0", state.weld_enabled),
            ("connect", "eq_connect_0", state.connect_enabled),
        ):
            _property_label(name)
            _changed, enabled = _probe_checkbox(f"##probe-equality-{item_id}", enabled)
            if item_id == "weld":
                state.weld_enabled = enabled
            else:
                state.connect_enabled = enabled
        imgui.end_table()


def _draw_control_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Control", "ProbeControl", size)
    if opened:
        _draw_control_content(state)
    imgui.end_child()


def _draw_joints_content(state: ProbeState) -> None:
    state.joint_filter, state.joint_sort_by_name = _draw_search_header(
        state,
        "probe-joint",
        "Search joints",
        state.joint_filter,
        state.joint_sort_by_name,
        state_order="qpos / qvel",
    )
    _draw_copy_buttons("probe-joint-state", ("Copy qpos", "Copy qvel"))
    if _begin_gallery_properties("##probe-all-joints"):
        _property_label("floating_base")
        _table_text("free · 6 dof", disabled=True)
        _property_label("shoulder_ball")
        _table_text("ball · 3 dof", disabled=True)
        _property_label("slide")
        _, state.slide_position = imgui.slider_float(
            "##probe-slide-position", state.slide_position, -0.34, 0.34, "%+.4f"
        )
        _property_label("hinge_limited")
        _, state.hinge_position = imgui.slider_float(
            "##probe-hinge-position", state.hinge_position, -1.2, 1.2, "%+.4f"
        )
        imgui.end_table()


def _draw_joints_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Joints", "ProbeJoints", size)
    if opened:
        _draw_joints_content(state)
    imgui.end_child()


def _draw_camera_content(state: ProbeState) -> None:
    imgui.set_next_item_width(-1.0)
    imgui.combo("##probe-camera", 0, ("source: free", "overview", "tracking"))
    imgui.spacing()
    imgui.text_disabled("presets")
    preset_flags = _flags(
        imgui.TableFlags_.sizing_stretch_same,
        imgui.TableFlags_.no_saved_settings,
        imgui.TableFlags_.no_pad_outer_x,
    )
    if imgui.begin_table("##probe-camera-presets", 4, preset_flags):
        for index, label in enumerate(
            ("front", "back", "left", "right", "top", "bottom", "iso", "frame all")
        ):
            imgui.table_next_column()
            imgui.button(f"{label}##probe-camera-preset-{index}", imgui.ImVec2(-1.0, 0.0))
        imgui.end_table()
    imgui.separator()
    if _begin_gallery_properties("##probe-camera-params"):
        for label, value, lo, hi, fmt in (
            ("yaw", -90.0, -180.0, 180.0, "%.1f deg"),
            ("pitch", -20.0, -89.9, 89.9, "%.1f deg"),
            ("distance", 3.0, 0.05, 200.0, "%.3f m"),
            ("fov_y_deg", 45.0, 10.0, 120.0, "%.1f deg"),
            ("far", 200.0, 1.0, 100000.0, "%.1f m"),
        ):
            _property_label(label)
            imgui.slider_float(f"##probe-camera-{label}", value, lo, hi, fmt)
        _property_label("projection")
        segment_width = max(44.0, (imgui.get_content_region_avail().x - 1.0) * 0.5)
        state.camera_projection = _draw_segmented(
            "camera-projection",
            ("persp", "ortho"),
            state.camera_projection,
            width=segment_width,
            icons=("persp", "ortho"),
            icon_drawer=(
                (
                    lambda *values: _draw_concept_projection_icon(
                        *values,
                        state=state,
                    )
                )
                if state.preview_icon_library
                else None
            ),
        )
        imgui.end_table()
    if imgui.collapsing_header("camera bookmarks"):
        imgui.text_disabled("camera bookmark")
        imgui.input_text("##probe-camera-bookmark", "view-1")
        imgui.button("save")
        imgui.same_line()
        imgui.button("copy")
        imgui.same_line()
        imgui.button("delete")


def _draw_camera_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Camera", "ProbeCamera", size)
    if opened:
        _draw_camera_content(state)
    imgui.end_child()


def _probe_axis_field(
    label: str,
    item_id: str,
    value: float,
    axis: int,
    fmt: str,
    scale: float,
    *,
    editable: bool = True,
) -> None:
    """Mirror the Inspector's visually continuous axis + value control."""

    color = AXIS_BADGE_COLORS[axis]
    hovered_color = AXIS_BADGE_HOVERED[axis]
    active_color = AXIS_BADGE_ACTIVE[axis]
    axis_width = 18.0 * scale
    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    splitter.set_current_channel(draw_list, 1)
    transparent = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
    imgui.push_style_color(imgui.Col_.button, transparent)
    imgui.push_style_color(imgui.Col_.button_hovered, transparent)
    imgui.push_style_color(imgui.Col_.button_active, transparent)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*AXIS_BADGE_TEXT))
    if not editable:
        imgui.push_style_var(imgui.StyleVar_.disabled_alpha, 1.0)
    imgui.begin_disabled(not editable)
    imgui.button(f"{label}##{item_id}-axis", imgui.ImVec2(axis_width, 0.0))
    imgui.end_disabled()
    if not editable:
        imgui.pop_style_var()
    button_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    button_active = imgui.is_item_active()
    button_lo, button_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    imgui.pop_style_color(4)
    imgui.same_line(0.0, 0.0)
    group_gap = 3.0 * scale if axis < 2 else 0.0
    imgui.set_next_item_width(max(1.0, imgui.get_content_region_avail().x - group_gap))
    imgui.push_style_color(imgui.Col_.frame_bg, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_hovered, transparent)
    imgui.push_style_color(imgui.Col_.frame_bg_active, transparent)
    imgui.begin_disabled(not editable)
    imgui.drag_float(f"##{item_id}-value", value, 0.01 if editable else 0.0, 0.0, 0.0, fmt)
    imgui.end_disabled()
    imgui.pop_style_color(3)
    field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    field_active = imgui.is_item_active()
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    splitter.set_current_channel(draw_list, 0)
    draw_joined_field_frame(
        draw_list,
        button_lo,
        button_hi,
        field_lo,
        field_hi,
        badge_color=(
            active_color
            if button_active and editable
            else hovered_color
            if button_hovered and editable
            else color
        ),
        field_color=(
            CONCEPT_THEME.bg_frame_active
            if field_active
            else CONCEPT_THEME.bg_frame_hovered
            if field_hovered and editable
            else CONCEPT_THEME.bg_frame
        ),
        rounding=float(imgui.get_style().frame_rounding),
        badge_opacity=float(imgui.get_style().alpha),
        field_opacity=float(imgui.get_style().alpha)
        * (1.0 if editable else float(imgui.get_style().disabled_alpha)),
    )
    splitter.merge(draw_list)


def _probe_property_vector_row(
    name: str,
    values: tuple[float, float, float],
    fmt: str,
    scale: float,
) -> None:
    """Mirror the Inspector's responsive label/control XYZ property row."""

    _property_label(name)
    compact = imgui.get_content_region_avail().x < 210.0 * scale
    flags = _flags(
        imgui.TableFlags_.sizing_stretch_same,
        imgui.TableFlags_.no_saved_settings,
        imgui.TableFlags_.no_pad_inner_x,
        imgui.TableFlags_.no_pad_outer_x,
    )
    columns = 1 if compact else 3
    if not imgui.begin_table(f"##probe-{name}-property-axes", columns, flags):
        return
    for axis in "xyz"[:columns]:
        imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch.value, 1.0)
    for axis, label in enumerate("XYZ"):
        if compact:
            imgui.table_next_row()
        imgui.table_next_column()
        _probe_axis_field(label, f"probe-{name}-{axis}", values[axis], axis, fmt, scale)
    imgui.end_table()


def _draw_inspector_gallery(size, scale: float) -> None:
    opened = _begin_gallery_panel("Inspector", "ProbeInspector", size)
    if opened:
        style = imgui.get_style()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(style.item_spacing.x, 0.0),
        )
        transform_open = imgui.collapsing_header(
            "transform", imgui.TreeNodeFlags_.default_open.value
        )
        imgui.pop_style_var()
        if transform_open:
            imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*CONCEPT_THEME.bg_popup))
            imgui.push_style_var(
                imgui.StyleVar_.window_padding,
                imgui.ImVec2(6.0 * scale, 4.0 * scale),
            )
            child_flags = imgui.ChildFlags_.always_use_window_padding.value
            window_flags = _flags(
                imgui.WindowFlags_.no_scrollbar,
                imgui.WindowFlags_.no_scroll_with_mouse,
            )
            child_visible = imgui.begin_child(
                "##probe-transform-body",
                imgui.ImVec2(0.0, 156.0 * scale),
                child_flags,
                window_flags,
            )
            if child_visible:
                imgui.push_font(None, 12.0 * scale)
                if _begin_gallery_properties("##probe-transform"):
                    _probe_property_vector_row("position", (0.193, 0.047, 0.445), "%.3f", scale)
                    _probe_property_vector_row("rotation", (0.0, -0.0, 0.0), "%.1f", scale)
                    imgui.end_table()
                imgui.pop_font()
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
        imgui.spacing()
        if imgui.collapsing_header(
            "material", imgui.TreeNodeFlags_.default_open.value
        ) and _begin_gallery_properties("##probe-appearance"):
            _property_label("assigned material")
            _table_text("robot_metal")
            _property_label("base color")
            imgui.color_button("##probe-base-color", imgui.ImVec4(0.82, 0.40, 0.33, 1.0))
            _property_label("roughness")
            imgui.slider_float("##probe-roughness", 0.42, 0.0, 1.0, "%.2f")
            imgui.end_table()
    imgui.end_child()


def _draw_hierarchy_gallery(size, state: ProbeState, scale: float) -> None:
    opened = _begin_gallery_panel("Hierarchy", "ProbeHierarchy", size)
    if opened:
        imgui.set_next_item_width(-1.0)
        _changed, state.hierarchy_filter = _preview_search_input(
            state,
            "##probe-hierarchy-filter",
            state.hierarchy_filter,
            hint="Search hierarchy",
            search_tooltip="Search hierarchy",
            clear_tooltip="Clear search",
        )
        chip_flags = _flags(
            imgui.WindowFlags_.horizontal_scrollbar, imgui.WindowFlags_.no_scroll_with_mouse
        )
        chip_height = imgui.get_frame_height() + imgui.get_style().scrollbar_size + 8.0 * scale
        if imgui.begin_child(
            "##probe-hierarchy-chips",
            imgui.ImVec2(-1.0, chip_height),
            imgui.ChildFlags_.none.value,
            chip_flags,
        ):
            for index, label in enumerate(
                ("All", "link", "geom", "joint", "site", "camera", "light", "robot", "flex")
            ):
                if index:
                    imgui.same_line()
                selected = state.hierarchy_kind == index
                if selected:
                    imgui.push_style_color(
                        imgui.Col_.button, imgui.ImVec4(*CONCEPT_THEME.bg_frame_active)
                    )
                    imgui.push_style_color(
                        imgui.Col_.text, imgui.ImVec4(*CONCEPT_THEME.primary_bright)
                    )
                if imgui.button(
                    f"{label}##probe-hierarchy-chip-{index}",
                    imgui.ImVec2(max(42.0, imgui.calc_text_size(label).x + 18.0), 0.0),
                ):
                    state.hierarchy_kind = index
                if selected:
                    imgui.pop_style_color(2)
        imgui.end_child()
        imgui.separator()
        rows = (
            (0, "▾", "world", "world", 0),
            (1, "▾", "a_sphere", "link", 1),
            (2, "", "sphere_geom", "geom", 2),
            (1, "▸", "hinge_body", "link", 3),
            (2, "", "hinge_joint", "joint", 4),
            (1, "▸", "overview", "camera", 5),
        )
        row_draw = state.painter()
        row_flags = _flags(
            imgui.TableFlags_.sizing_stretch_prop,
            imgui.TableFlags_.no_saved_settings,
            imgui.TableFlags_.pad_outer_x,
        )
        if imgui.begin_table("##probe-hierarchy-rows", 3, row_flags):
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            imgui.table_setup_column(
                "Type", imgui.TableColumnFlags_.width_fixed.value, 74.0 * scale
            )
            imgui.table_setup_column(
                "Show", imgui.TableColumnFlags_.width_fixed.value, 42.0 * scale
            )
            for depth, disclosure, name, kind, index in rows:
                row_height = 31.0 * scale
                item_height = 25.0 * scale
                imgui.table_next_row(imgui.TableRowFlags_.none.value, row_height)
                imgui.table_next_column()
                clicked, _ = imgui.selectable(
                    f"##probe-hierarchy-{index}",
                    state.hierarchy_selection == index,
                    _flags(
                        imgui.SelectableFlags_.span_all_columns,
                        imgui.SelectableFlags_.allow_overlap,
                    ),
                    imgui.ImVec2(0.0, item_height),
                )
                hovered = imgui.is_item_hovered()
                lo = imgui.get_item_rect_min()
                hi = imgui.get_item_rect_max()
                center_y = (lo.y + hi.y) * 0.5
                text_y = text_line_y(row_draw, center_y)
                node_x = lo.x + (6.0 + depth * 22.0) * scale
                if disclosure:
                    if state.preview_icon_library:
                        concept_name = "panel-down" if disclosure == "▾" else "panel-right"
                        draw_concept_icon(
                            row_draw,
                            (node_x + 5 * scale, center_y),
                            10.0 * scale,
                            concept_name,
                            CONCEPT_THEME.text,
                            padding=state.icon_padding_for_glyph(concept_name),
                            stroke_width=state.icon_stroke_for_glyph(concept_name),
                            tuning=state.icon_tuning(),
                            alignment=state.icon_alignment_for_glyph(concept_name),
                        )
                    else:
                        row_draw.fringed_concave_fill(
                            disclosure_triangle(
                                (node_x + 5 * scale, center_y),
                                4 * scale,
                                opened=disclosure == "▾",
                                smoothing=state.tool_smoothing,
                            ),
                            CONCEPT_THEME.text,
                        )
                row_draw.text(
                    (node_x + 18.0 * scale, text_y),
                    CONCEPT_THEME.text,
                    name,
                    pixel_snap=False,
                )
                if clicked:
                    state.hierarchy_selection = index

                imgui.table_next_column()
                type_x = imgui.get_cursor_screen_pos().x
                imgui.set_cursor_screen_pos(imgui.ImVec2(type_x, text_y))
                imgui.text_disabled(kind)

                imgui.table_next_column()
                cell = imgui.get_cursor_screen_pos()
                button_size = 20.0 * scale
                cell_width = imgui.get_content_region_avail().x
                imgui.set_cursor_screen_pos(
                    imgui.ImVec2(
                        cell.x + max(0.0, (cell_width - button_size) * 0.5),
                        lo.y + max(0.0, (hi.y - lo.y - button_size) * 0.5),
                    )
                )
                if imgui.invisible_button(
                    f"##probe-hierarchy-visible-{index}",
                    imgui.ImVec2(button_size, button_size),
                ):
                    state.hierarchy_visibility[index] = not state.hierarchy_visibility[index]
                visible_lo = imgui.get_item_rect_min()
                visible_hi = imgui.get_item_rect_max()
                center = (
                    (visible_lo.x + visible_hi.x) * 0.5,
                    (visible_lo.y + visible_hi.y) * 0.5,
                )
                radius_x = 6.5 * scale
                radius_y = 3.6 * scale
                color = (
                    CONCEPT_THEME.primary_bright
                    if hovered or state.hierarchy_selection == index
                    else CONCEPT_THEME.primary
                    if state.hierarchy_visibility[index]
                    else CONCEPT_THEME.text_disabled
                )
                if state.preview_icon_library:
                    concept_name = (
                        "panel-visible" if state.hierarchy_visibility[index] else "panel-hidden"
                    )
                    draw_concept_icon(
                        row_draw,
                        center,
                        16.0 * scale,
                        concept_name,
                        color,
                        padding=state.icon_padding_for_glyph(concept_name),
                        stroke_width=state.icon_stroke_for_glyph(concept_name),
                        tuning=state.icon_tuning(),
                        alignment=state.icon_alignment_for_glyph(concept_name),
                    )
                elif state.hierarchy_visibility[index]:
                    top = tuple(
                        (
                            center[0] - radius_x + radius_x * 2.0 * point / 8.0,
                            center[1] - math.sin(math.pi * point / 8.0) * radius_y,
                        )
                        for point in range(9)
                    )
                    bottom = tuple(
                        (
                            center[0] + radius_x - radius_x * 2.0 * point / 8.0,
                            center[1] + math.sin(math.pi * point / 8.0) * radius_y,
                        )
                        for point in range(9)
                    )
                    row_draw.polyline((*top, *bottom[1:-1]), color, 1.35 * scale, closed=True)
                    row_draw.circle(center, 1.8 * scale, color, 1.2 * scale, segments=16)
                else:
                    lid = tuple(
                        (
                            center[0] - radius_x + radius_x * 2.0 * point / 8.0,
                            center[1] + math.sin(math.pi * point / 8.0) * radius_y * 0.72,
                        )
                        for point in range(9)
                    )
                    row_draw.polyline(lid, color, 1.45 * scale)
                    for offset in (-0.52, 0.0, 0.52):
                        lash_x = center[0] + radius_x * offset
                        lash_y = center[1] + radius_y * 0.72 * math.sqrt(max(0.0, 1.0 - offset**2))
                        row_draw.line(
                            (lash_x, lash_y),
                            (lash_x + offset * 1.6 * scale, lash_y + 2.2 * scale),
                            color,
                            1.15 * scale,
                        )
            imgui.end_table()
    imgui.end_child()


def _draw_assets_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Assets", "ProbeAssets", size)
    if opened:
        if _begin_gallery_properties("##probe-asset-model"):
            _property_label("model")
            imgui.set_next_item_width(-1.0)
            imgui.combo("##probe-asset-model-name", 0, ("gizmo",))
            imgui.end_table()

        import_labels = ("Import Mesh...", "Import Height Field...", "Import Texture...")
        inline = button_row_layout(
            tuple(button_width(label) for label in import_labels),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        for index, label in enumerate(import_labels):
            if inline[index]:
                imgui.same_line()
            imgui.button(f"{label}##probe-asset-import-{index}")

        if _begin_gallery_properties("##probe-texture-type"):
            _property_label("texture type")
            imgui.set_next_item_width(-1.0)
            imgui.combo("##probe-texture-type-value", 0, ("2D", "Cube", "Skybox"))
            imgui.end_table()
        imgui.collapsing_header("Height-field import size")
        imgui.collapsing_header("New material")
        imgui.separator()
        imgui.set_next_item_width(-1.0)
        _changed, state.asset_filter = _preview_search_input(
            state,
            "##probe-asset-filter",
            state.asset_filter,
            hint="Filter assets...",
            search_tooltip="Search assets",
            clear_tooltip="Clear search",
        )
        imgui.set_next_item_width(-1.0)
        _changed, state.asset_type = imgui.combo(
            "##probe-asset-type", state.asset_type, ("All", "mesh", "texture", "material", "hfield")
        )
        list_height = min(150.0, max(86.0, imgui.get_content_region_avail().y * 0.48))
        if imgui.begin_child(
            "##probe-asset-list", imgui.ImVec2(0.0, list_height), imgui.ChildFlags_.borders.value
        ):
            flags = _flags(
                imgui.TableFlags_.sizing_stretch_prop,
                imgui.TableFlags_.row_bg,
                imgui.TableFlags_.no_saved_settings,
            )
            if imgui.begin_table("##probe-asset-table", 3, flags):
                imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch.value, 1.0)
                imgui.table_setup_column("Type", imgui.TableColumnFlags_.width_fixed.value)
                imgui.table_setup_column("Used", imgui.TableColumnFlags_.width_fixed.value)
                imgui.table_headers_row()
                for index, (name, kind, used) in enumerate(
                    (
                        ("robot_body", "mesh", "2"),
                        ("floor_albedo", "texture", "1"),
                        ("robot_metal", "material", "12"),
                        ("terrain", "hfield", "1"),
                    )
                ):
                    imgui.table_next_row()
                    imgui.table_next_column()
                    clicked, _ = imgui.selectable(
                        f"{name}##probe-asset-{index}",
                        state.asset_selection == index,
                        imgui.SelectableFlags_.span_all_columns.value,
                    )
                    if clicked:
                        state.asset_selection = index
                    imgui.table_next_column()
                    imgui.text_disabled(kind)
                    imgui.table_next_column()
                    imgui.text_disabled(used)
                imgui.end_table()
        imgui.end_child()
    imgui.end_child()


def _draw_stats_gallery(size) -> None:
    opened = _begin_gallery_panel("Stats", "ProbeStats", size)
    if opened:
        imgui.plot_lines(
            "##probe-frame-plot",
            PROBE_FRAME_SAMPLES,
            overlay_text="8.35 ms   119.8 fps",
            scale_min=0.0,
            scale_max=16.7,
            graph_size=imgui.ImVec2(-1.0, 60.0),
        )
        imgui.text_disabled("scale 0 .. 16.7 ms   (60 fps = 16.7 ms)")
        flags = _flags(imgui.TableFlags_.sizing_stretch_same, imgui.TableFlags_.row_bg)
        if imgui.begin_table("##probe-stats-counts", 2, flags):
            imgui.table_setup_column("metric", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 1.0)
            for label, value in (
                ("draw calls", "148"),
                ("instances", "216"),
                ("triangles", "284,612"),
                ("buckets", "9"),
                ("frame cpu", "3.610 ms"),
            ):
                _property_label(label)
                _table_text(value)
            imgui.end_table()
        imgui.separator()
        if imgui.begin_table(
            "##probe-stats-passes",
            3,
            _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.row_bg),
        ):
            for heading in ("pass", "cpu ms", "gpu ms"):
                imgui.table_setup_column(heading)
            imgui.table_headers_row()
            for name, cpu, gpu in (("opaque", "1.842", "1.109"), ("overlay", "0.381", "0.214")):
                imgui.table_next_row()
                for cell in (name, cpu, gpu):
                    imgui.table_next_column()
                    imgui.text(cell)
            imgui.end_table()
    imgui.end_child()


def _draw_sensors_gallery(size, state: ProbeState) -> None:
    opened = _begin_gallery_panel("Sensors", "ProbeSensors", size)
    if opened:
        imgui.set_next_item_width(-1.0)
        _changed, state.sensor_index = imgui.combo(
            "##probe-sensor", state.sensor_index, ("camera_grid", "imu_accel", "hinge_pos")
        )
        if _begin_gallery_properties("##probe-sensor-properties"):
            if state.sensor_index == 0:
                _property_label("type")
                _table_text("rangefinder")
                _property_label("dimension")
                _table_text("42")
            elif state.sensor_index == 1:
                _property_label("type")
                _table_text("accelerometer")
                _property_label("dimension")
                _table_text("3")
                _property_label("value")
                _table_text("[0.02, 0.01, 9.81]")
            else:
                _property_label("type")
                _table_text("jointpos")
                _property_label("dimension")
                _table_text("1")
                _property_label("value")
                _table_text("[0.350]")
            imgui.end_table()
        if state.sensor_index == 0:
            imgui.separator()
            imgui.text_disabled("value")
            imgui.button("Copy")
            imgui.same_line()
            imgui.button("Open in Plot")
            if imgui.begin_child(
                "##probe-sensor-values", imgui.ImVec2(0.0, 112.0), imgui.ChildFlags_.borders.value
            ):
                imgui.text_wrapped(
                    "[ 2.410969, -0.774737, 0.387369, 0.640125, 1.129701,  ... ,\n"
                    "  1.884203, 2.004288, 2.101055 ]"
                )
            imgui.end_child()
    imgui.end_child()


def _draw_panel_gallery(available, state: ProbeState, scale: float) -> None:
    spacing = imgui.get_style().item_spacing
    row_height = (available.y - spacing.y) * 0.5
    top_width = (available.x - spacing.x * 3.0) / 4.0
    top_size = imgui.ImVec2(top_width, row_height)
    for index, draw_panel in enumerate(
        (
            lambda: _draw_control_gallery(top_size, state),
            lambda: _draw_joints_gallery(top_size, state),
            lambda: _draw_camera_gallery(top_size, state),
            lambda: _draw_inspector_gallery(top_size, scale),
        )
    ):
        draw_panel()
        if index != 3:
            imgui.same_line()

    # M14 deliberately receives the widest lower slot. Its fixed type/visibility
    # metadata must not steal width from entity names as it did in the old 320 px panel.
    lower_widths = (
        available.x * 0.28,
        available.x * 0.24,
        available.x * 0.18,
    )
    final_width = available.x - sum(lower_widths) - spacing.x * 3.0
    lower_sizes = tuple(imgui.ImVec2(width, row_height) for width in (*lower_widths, final_width))
    for index, draw_panel in enumerate(
        (
            lambda: _draw_hierarchy_gallery(lower_sizes[0], state, scale),
            lambda: _draw_assets_gallery(lower_sizes[1], state),
            lambda: _draw_stats_gallery(lower_sizes[2]),
            lambda: _draw_sensors_gallery(lower_sizes[3], state),
        )
    ):
        draw_panel()
        if index != 3:
            imgui.same_line()


def _draw_workspace_right_dock(size, state: ProbeState) -> None:
    if not imgui.begin_child(
        "Right dock###ProbeWorkspaceRightDock",
        size,
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return
    active = _wrapped_tabs(
        (
            ("Control", "right-dock", "control"),
            ("Joints", "right-dock", "joints"),
            ("Camera", "right-dock", "camera"),
        ),
        state.workspace_right_tab,
        imgui.get_content_region_avail().x,
    )
    state.workspace_right_tab = active
    if active == "Control":
        _draw_control_content(state)
    elif active == "Joints":
        _draw_joints_content(state)
    else:
        _draw_camera_content(state)
    imgui.end_child()


def _draw_panel_page(available, state: ProbeState, scale: float) -> None:
    flags = _flags(
        imgui.WindowFlags_.horizontal_scrollbar,
        imgui.WindowFlags_.always_vertical_scrollbar,
    )
    if not imgui.begin_child(
        "Panel gallery canvas###ProbePanelGalleryCanvas",
        available,
        imgui.ChildFlags_.none.value,
        flags,
    ):
        imgui.end_child()
        return
    cursor = imgui.get_cursor_pos()
    width, height = _virtual_canvas_size(available, scale, PANEL_CANVAS_SIZE)
    _draw_panel_gallery(imgui.ImVec2(width, height), state, scale)
    imgui.set_cursor_pos(imgui.ImVec2(cursor.x + width - 1.0, cursor.y + height - 1.0))
    imgui.dummy(imgui.ImVec2(1.0, 1.0))
    imgui.end_child()


def _probe_status_hints(
    variant: str,
    *,
    selected: bool,
    selection_clear: bool = True,
) -> tuple[ToolHint, ...]:
    """Compose the production status grammar for deterministic specimens."""

    hints = default_tool_hints(variant, DEFAULT_INPUT_BINDINGS)
    if variant != "ready_minimal":
        hints = tuple(hint for hint in hints if hint.hint_id != "gizmo.type_value")
    if selected and selection_clear:
        hints = (
            ToolHint("key", "Esc", "Clear selection", hint_id="selection.clear"),
            *hints,
        )
    return (
        ToolHint("key", "Backspace", "Rewind", hint_id="playback.previous"),
        *hints,
    )


def _draw_status_strip(
    draw: ImguiDraw2D,
    origin,
    width: float,
    height: float,
    scale: float,
    *,
    selected: str,
    running: bool,
    fps: str,
    backend: str,
) -> None:
    """Draw the same persistent status surface used by the application."""

    rate = float(fps)
    selected_item = selected.strip() or "No selection"
    has_selection = selected_item != "No selection"
    variant = "ready" if has_selection else "camera"
    hints = _probe_status_hints(variant, selected=has_selection)
    draw_status(
        draw,
        origin,
        width,
        height,
        CONCEPT_THEME,
        scale,
        selected=selected_item,
        state="running" if running else "paused",
        sim_time=1.204,
        step=1204,
        metric_mode="time",
        backend=backend,
        dt=0.002,
        fps=rate,
        tool_hints=hints,
    )


def _draw_shell_settings_tab(available, scale: float, state: ProbeState) -> None:
    spacing = imgui.get_style().item_spacing
    shell_width = max(360.0 * scale, available.x * 0.36)
    shell_size = imgui.ImVec2(shell_width, available.y)
    if imgui.begin_child("Shell###ProbeShell", shell_size, imgui.ChildFlags_.borders.value):
        imgui.text("Application shell · M1 / M4 / M7")
        imgui.separator()
        imgui.text("File   Edit   Entity   View   Window   Help")
        imgui.same_line()
        name = "showcase.xml  ●"
        name_width = imgui.calc_text_size(name).x
        imgui.set_cursor_pos_x(
            max(imgui.get_cursor_pos_x(), imgui.get_window_width() - name_width - 16.0)
        )
        imgui.text_disabled(name)

        imgui.spacing()
        imgui.text("Simulation state")
        imgui.separator()
        selected = 1 if state.sim_running else 0
        selected = _draw_segmented("simulation-state", ("Paused", "Running"), selected, width=92.0)
        state.sim_running = selected == 1

        imgui.spacing()
        imgui.text("Edit gate")
        imgui.separator()
        if _settings_properties("##probe-edit-gate"):
            _property_label("Joint position")
            imgui.begin_disabled(state.sim_running)
            imgui.slider_float("##probe-gated-joint", state.hinge_position, -1.2, 1.2, "%+.3f")
            imgui.end_disabled()
            _property_label("Actuator ctrl")
            imgui.slider_float("##probe-live-actuator", state.hinge_ctrl, -1.0, 1.0, "%+.3f")
            imgui.end_table()
        if state.sim_running:
            imgui.text_colored(imgui.ImVec4(*CONCEPT_THEME.warning), "Pause to edit (Space)")

        imgui.spacing()
        imgui.text("Dock layout")
        imgui.separator()
        if _settings_properties("##probe-dock-layout"):
            for label, value in (
                ("Panels", "Dockable / floating"),
                ("Persist", "imgui.ini"),
                ("Default", "Hierarchy 22% · Viewport 48% · Right 30%"),
            ):
                _property_label(label)
                _table_text(value, disabled=True)
            imgui.end_table()
        imgui.button("Reset Layout")

        imgui.spacing()
        imgui.text("Status bar")
        imgui.separator()
        start = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        height = 34.0 * scale
        draw = state.painter()
        _draw_status_strip(
            draw,
            (start.x, start.y),
            width,
            height,
            scale,
            selected="hinge_body",
            running=state.sim_running,
            fps="119.8",
            backend=state.renderer,
        )
        imgui.dummy(imgui.ImVec2(width, height))
    imgui.end_child()
    imgui.same_line()
    _draw_settings(
        imgui.ImVec2(available.x - shell_width - spacing.x, available.y),
        state,
        scale,
    )


def _draw_status_tab(available, scale: float, state: ProbeState) -> None:
    """Show the persistent status surface and its mutually exclusive states."""

    if not imgui.begin_child(
        "Status###ProbeStatus",
        available,
        imgui.ChildFlags_.borders.value,
    ):
        imgui.end_child()
        return
    imgui.text("Status · persistent application surface")
    imgui.separator()
    imgui.text_disabled(
        "Selected object names live here; viewport labels are reserved for values and limits."
    )
    imgui.spacing()
    draw = state.painter()
    samples = (
        ("Paused · selected", "a_sphere", False, "119.8"),
        ("Running · selected", "a_sphere", True, "60.0"),
        ("Paused · no selection", "No selection", False, "119.8"),
    )
    width = min(1180.0 * scale, imgui.get_content_region_avail().x)
    height = 34.0 * scale
    for label, selected, running, fps in samples:
        imgui.text_disabled(label)
        lo = imgui.get_cursor_screen_pos()
        _draw_status_strip(
            draw,
            (lo.x, lo.y),
            width,
            height,
            scale,
            selected=selected,
            running=running,
            fps=fps,
            backend=state.renderer,
        )
        imgui.dummy(imgui.ImVec2(width, height + 12.0 * scale))
    imgui.spacing()
    imgui.text_disabled(
        "Status remains visible across workspaces; Stats keeps detailed timing history."
    )
    imgui.end_child()


def _draw_plot_aux(size) -> None:
    if not imgui.begin_child("Plot###ProbePlot", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Plot")
    imgui.separator()
    imgui.text_disabled("hinge_pos · rad")
    imgui.plot_lines("##probe-plot", PROBE_PLOT_SAMPLES, graph_size=imgui.ImVec2(-1.0, -1.0))
    imgui.end_child()


def _draw_help_aux(size) -> None:
    if not imgui.begin_child("Help###ProbeHelp", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Help")
    imgui.separator()
    if imgui.begin_table("##probe-help", 2, imgui.TableFlags_.row_bg.value):
        imgui.table_setup_column("Input", imgui.TableColumnFlags_.width_fixed.value, 160.0)
        imgui.table_setup_column("Action", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_headers_row()
        for input_name, action in (
            ("Space", "Play / Pause"),
            ("Backspace", "Previous frame · hold to rewind"),
            ("Esc", "Clear selection"),
            ("Shift", "Snap"),
            ("T", "World / Body"),
            ("Double-click item", "Focus item"),
            ("Double-click gizmo", "Type value after hover hint"),
            ("Ctrl + drag", "Push / Twist"),
        ):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(input_name)
            imgui.table_next_column()
            imgui.text(action)
        imgui.end_table()
    imgui.end_child()


def _draw_info_aux(size) -> None:
    if not imgui.begin_child("Info###ProbeInfo", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Info")
    imgui.separator()
    if _settings_properties("##probe-info"):
        for label, value in (
            ("Viewer", "mojive"),
            ("Backend", "OpenGL / OpenGL"),
            ("Scene source", "MuJoCo"),
            ("Document", "joint_types.xml"),
        ):
            _property_label(label)
            _table_text(value)
        imgui.end_table()
    imgui.end_child()


def _draw_workspaces_tab(available, scale: float, state: ProbeState) -> None:
    keyframe_height = min(320.0 * scale, max(240.0 * scale, available.y * 0.34))
    _draw_keyframes(imgui.ImVec2(available.x, keyframe_height), scale, state)
    imgui.text_disabled("M15 · full-width bottom dock; transport uses compact icon groups")
    active = _wrapped_tabs(
        (
            ("Output", "aux", "output"),
            ("Plot", "aux", "plot"),
            ("Help", "aux", "help"),
            ("Info", "aux", "info"),
        ),
        state.aux_tab,
        imgui.get_content_region_avail().x,
    )
    state.aux_tab = active
    remaining = imgui.get_content_region_avail()
    if active == "Output":
        _draw_output(remaining, state, scale)
    elif active == "Plot":
        _draw_plot_aux(remaining)
    elif active == "Help":
        _draw_help_aux(remaining)
    else:
        _draw_info_aux(remaining)
