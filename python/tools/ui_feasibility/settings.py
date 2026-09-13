"""Interactive settings specimens for layout and control review."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.panels.settings import settings_uses_stacked_layout

from .layout import (
    _draw_segmented,
    _flags,
    _probe_checkbox,
    _property_label,
    _table_next_control_row,
    _table_text,
)
from .state import ProbeState
from .widgets import _preview_search_input


def _settings_properties(table_id: str) -> bool:
    flags = _flags(imgui.TableFlags_.sizing_stretch_prop, imgui.TableFlags_.pad_outer_x)
    if not imgui.begin_table(table_id, 2, flags):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch.value, 0.64)
    return True


def _draw_settings_general(state: ProbeState) -> None:
    if not _settings_properties("##probe-settings-general"):
        return
    _property_label("Language")
    _, state.language = imgui.combo("##probe-language", state.language, ("简体中文", "English"))
    _property_label("UI font")
    _table_text("JetBrains Mono", disabled=True)
    _property_label("CJK font")
    _table_text("PingFang SC", disabled=True)
    imgui.end_table()


def _draw_settings_interaction(state: ProbeState) -> None:
    imgui.text_disabled("Gizmo")
    imgui.separator()
    if _settings_properties("##probe-settings-gizmo"):
        _property_label("Style")
        state.gizmo_style = _draw_segmented("gizmo", ("2D", "3D"), state.gizmo_style)
        _property_label("Orientation")
        state.frame = _draw_segmented("frame", ("Body", "World"), state.frame)
        _property_label("Overlay size")
        _, state.viewport_overlay_scale = imgui.drag_float(
            "##probe-overlay-size", state.viewport_overlay_scale, 0.02, 0.85, 1.6, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Input")
    imgui.separator()
    if _settings_properties("##probe-settings-input"):
        _property_label("Keep mode/unit")
        _, state.remember_input = _probe_checkbox("##probe-remember", state.remember_input)
        imgui.set_item_tooltip("Reuse relative/absolute mode and angle unit")
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Snap · Shift")
    imgui.separator()
    if _settings_properties("##probe-settings-snap"):
        _property_label("Position")
        _, state.position_snap = imgui.drag_float(
            "##probe-position-snap", state.position_snap, 0.01, 0.0, 10.0, "%.3f m"
        )
        _property_label("Rotation")
        _, state.rotation_snap = imgui.drag_float(
            "##probe-rotation-snap", state.rotation_snap, 0.5, 0.0, 180.0, "%.1f deg"
        )
        _property_label("Tick scale")
        _, state.tick_scale = imgui.drag_float(
            "##probe-tick-scale", state.tick_scale, 0.05, 0.25, 4.0, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("View")
    imgui.separator()
    if _settings_properties("##probe-settings-view"):
        _property_label("Padding")
        _, state.selection_padding = imgui.drag_float(
            "##probe-selection-padding", state.selection_padding, 0.05, 1.0, 3.0, "%.2fx"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Perturb")
    imgui.separator()
    if _settings_properties("##probe-settings-perturb"):
        _property_label("Corner radius")
        _, state.corner_radius = imgui.drag_float(
            "##probe-corner-radius", state.corner_radius, 0.5, 0.0, 16.0, "%.1f px"
        )
        imgui.end_table()

    imgui.spacing()
    imgui.text_disabled("Helpers")
    imgui.separator()
    if _settings_properties("##probe-settings-helpers"):
        _property_label("Entities")
        _, state.scene_icons = _probe_checkbox("##probe-scene-icons", state.scene_icons)
        _property_label("Volumes")
        imgui.begin_disabled(not state.scene_icons)
        _, state.influence_volumes = _probe_checkbox(
            "##probe-influence-volumes", state.influence_volumes
        )
        imgui.end_disabled()
        imgui.end_table()


def _draw_settings_rendering(state: ProbeState) -> None:
    imgui.text("OpenGL")
    imgui.separator()
    if _settings_properties("##probe-settings-rendering"):
        for label, attribute in (
            ("outline", "outline_enabled"),
            ("tonemap", "tonemap_enabled"),
            ("msaa", "msaa_enabled"),
        ):
            _property_label(label)
            _, value = _probe_checkbox(f"##probe-{label}", bool(getattr(state, attribute)))
            setattr(state, attribute, value)
        imgui.end_table()
    imgui.spacing()
    if imgui.collapsing_header("Debug") and _settings_properties("##probe-settings-debug"):
        _property_label("Debug view")
        _, state.debug_view = imgui.combo(
            "##probe-debug-view",
            state.debug_view,
            ("shaded", "albedo", "normal", "depth", "segment", "idcolor", "overdraw", "wireframe"),
        )
        _property_label("Labels")
        _, state.debug_labels = imgui.combo(
            "##probe-debug-labels",
            state.debug_labels,
            (
                "none",
                "body",
                "joint",
                "geom",
                "site",
                "camera",
                "light",
                "tendon",
                "actuator",
                "constraint",
                "flex",
                "contact point",
                "contact force",
                "selection",
            ),
        )
        _property_label("Frames")
        _, state.debug_frames = imgui.combo(
            "##probe-debug-frames",
            state.debug_frames,
            ("none", "body", "geom", "site", "camera", "light", "contact", "world"),
        )
        imgui.end_table()
    imgui.separator()
    imgui.text_disabled(f"Backend  {state.renderer}")
    imgui.text_disabled("Device  system renderer")


def _draw_checkbox_grid(
    table_id: str,
    labels: tuple[str, ...],
    values: list[bool],
    *,
    columns: int = 3,
) -> None:
    flags = _flags(imgui.TableFlags_.sizing_stretch_same)
    if not imgui.begin_table(table_id, columns, flags):
        return
    for index, label in enumerate(labels):
        imgui.table_next_column()
        changed, value = _probe_checkbox(f"{label}##{table_id}-{index}", values[index])
        if changed:
            values[index] = value
    imgui.end_table()


def _draw_settings_mujoco(state: ProbeState) -> None:
    categories = ("geom", "site", "joint", "tendon", "actuator", "flex", "skin")
    if imgui.collapsing_header("Visual groups", imgui.TreeNodeFlags_.default_open.value):
        flags = _flags(imgui.TableFlags_.sizing_stretch_same)
        if imgui.begin_table("##probe-visual-groups", 7, flags):
            imgui.table_setup_column("category", imgui.TableColumnFlags_.width_fixed.value, 92.0)
            for group in range(6):
                imgui.table_setup_column(str(group))
            imgui.table_headers_row()
            for row, category in enumerate(categories):
                _table_next_control_row()
                imgui.table_next_column()
                _table_text(category)
                for group in range(6):
                    imgui.table_next_column()
                    index = row * 6 + group
                    changed, value = _probe_checkbox(
                        f"##probe-group-{category}-{group}", state.mujoco_groups[index]
                    )
                    if changed:
                        state.mujoco_groups[index] = value
            imgui.end_table()
    if imgui.collapsing_header("mjtRndFlag", imgui.TreeNodeFlags_.default_open.value):
        _draw_checkbox_grid(
            "##probe-rnd-flags",
            ("shadow", "wireframe", "reflection", "additive", "skybox", "fog", "haze", "cull face"),
            state.render_flags,
            columns=4,
        )
    if imgui.collapsing_header("mjtVisFlag", imgui.TreeNodeFlags_.default_open.value):
        _draw_checkbox_grid(
            "##probe-vis-flags",
            (
                "convexhull",
                "texture",
                "joint",
                "actuator",
                "activation",
                "camera",
                "light",
                "rangefinder",
                "constraint",
                "static",
                "skin",
                "flexface",
                "flexskin",
                "flexvert",
                "flexedge",
                "contactpoint",
                "contactforce",
                "contactsplit",
                "island",
                "autoconnect",
                "tendon",
                "transparent",
                "com",
                "inertia",
                "sclinertia",
                "bodybvh",
                "meshbvh",
            ),
            state.visual_flags,
        )


def _draw_settings(size, state: ProbeState, scale: float) -> None:
    if not imgui.begin_child("Settings###ProbeSettings", size, imgui.ChildFlags_.borders.value):
        imgui.end_child()
        return
    imgui.text("Settings")
    imgui.separator()
    available = imgui.get_content_region_avail()
    categories = ("General", "Interaction", "Rendering", "MuJoCo Visuals")
    stacked = settings_uses_stacked_layout(available.x, scale)
    table_open = False
    if stacked:
        imgui.set_next_item_width(-1.0)
        if imgui.begin_combo("##probe-settings-category", categories[state.settings_page]):
            for index, label in enumerate(categories):
                clicked, _ = imgui.selectable(
                    f"{label}##probe-settings-category-{index}",
                    state.settings_page == index,
                )
                if clicked:
                    state.settings_page = index
            imgui.end_combo()
        imgui.spacing()
    else:
        nav_width = 132.0 * scale
        table_open = imgui.begin_table(
            "##probe-settings-layout",
            2,
            imgui.TableFlags_.borders_inner_v.value,
        )
        if not table_open:
            imgui.end_child()
            return
        imgui.table_setup_column(
            "categories",
            imgui.TableColumnFlags_.width_fixed.value,
            nav_width,
        )
        imgui.table_setup_column("page", imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.push_style_var(
            imgui.StyleVar_.selectable_text_align,
            imgui.ImVec2(0.5, 0.5),
        )
        for index, label in enumerate(categories):
            clicked, _ = imgui.selectable(
                f"{label}##probe-settings-category-{index}", state.settings_page == index
            )
            if clicked:
                state.settings_page = index
        imgui.pop_style_var()
        imgui.table_next_column()

    page_open = imgui.begin_child(
        "##probe-settings-page",
        imgui.ImVec2(0.0, 0.0),
        imgui.ChildFlags_.always_use_window_padding.value,
    )
    if not page_open:
        imgui.end_child()
        if table_open:
            imgui.end_table()
        imgui.end_child()
        return
    imgui.set_next_item_width(-1.0)
    _, state.settings_filter = _preview_search_input(
        state,
        "##probe-settings-filter",
        state.settings_filter,
        search_tooltip="Search settings",
        clear_tooltip="Clear search",
    )
    imgui.separator()
    if state.settings_page == 0:
        _draw_settings_general(state)
    elif state.settings_page == 1:
        _draw_settings_interaction(state)
    elif state.settings_page == 2:
        _draw_settings_rendering(state)
    else:
        _draw_settings_mujoco(state)
    imgui.end_child()
    if table_open:
        imgui.end_table()
    imgui.end_child()
