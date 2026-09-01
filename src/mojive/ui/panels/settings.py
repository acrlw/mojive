"""Renderer, visual group, and interaction settings."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from ...render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from ..gizmo import (
    DEFAULT_ROTATION_SNAP_DEG,
    DEFAULT_ROTATION_TICK_SCALE,
    DEFAULT_TRANSLATION_SNAP_M,
)
from ..input_bindings import InputAction, input_action_name, key_choices
from ..localization import LANGUAGE_LABELS, Language, parse_language
from ..perturb import OUTLINE_CORNER_RADIUS_PT
from ..viewcube import (
    DEFAULT_SELECTION_PADDING,
    MAX_SELECTION_PADDING,
    MIN_SELECTION_PADDING,
)
from ..viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
)
from . import Panel, PanelContext, search_input, segmented_control, themed_checkbox

_RND_FLAGS: tuple[RenderFlag, ...] = (
    RenderFlag.SHADOW,
    RenderFlag.WIREFRAME,
    RenderFlag.REFLECTION,
    RenderFlag.ADDITIVE,
    RenderFlag.SKYBOX,
    RenderFlag.FOG,
    RenderFlag.HAZE,
    RenderFlag.CULL_FACE,
)
_VIS_FLAGS: tuple[RenderFlag, ...] = (
    RenderFlag.CONVEXHULL,
    RenderFlag.TEXTURE,
    RenderFlag.JOINT,
    RenderFlag.ACTUATOR,
    RenderFlag.ACTIVATION,
    RenderFlag.CAMERA,
    RenderFlag.LIGHT,
    RenderFlag.RANGEFINDER,
    RenderFlag.CONSTRAINT,
    RenderFlag.STATIC,
    RenderFlag.SKIN,
    RenderFlag.FLEXFACE,
    RenderFlag.FLEXSKIN,
    RenderFlag.FLEXVERT,
    RenderFlag.FLEXEDGE,
    RenderFlag.CONTACTPOINT,
    RenderFlag.CONTACTFORCE,
    RenderFlag.CONTACTSPLIT,
    RenderFlag.ISLAND,
    RenderFlag.AUTOCONNECT,
    RenderFlag.TENDON,
    RenderFlag.TRANSPARENT,
    RenderFlag.COM,
    RenderFlag.INERTIA,
    RenderFlag.SCLINERTIA,
    RenderFlag.BODYBVH,
    RenderFlag.MESHBVH,
)

_CATEGORIES = ("General", "Interaction", "Rendering", "MuJoCo Visuals")
_CATEGORY_WIDTH_PT = 132.0
_PAGE_MIN_WIDTH_PT = 224.0
_COLUMN_GAP_PT = 8.0
_CATEGORY_SEARCH_TERMS = {
    "General": ("language", "ui font", "cjk font"),
    "Interaction": (
        "gizmo",
        "style",
        "orientation",
        "overlay size",
        "shortcuts key bindings remap reset",
        "reuse mode unit",
        "snap position rotation tick scale",
        "view selection padding",
        "perturb corner radius",
        "helpers entities influence volumes",
    ),
    "Rendering": (
        "backend graphics device scene lights shadow casters",
        "debug view labels frames",
        "opengl render flags outline tonemap msaa",
    ),
    "MuJoCo Visuals": (
        "visual groups bvh depth",
        "mjt rnd flag shadow wireframe reflection additive skybox fog haze cull face",
        "mjt vis flag joint actuator camera light contact force split inertia bvh",
    ),
}


def settings_category_matches(category: str, query: str) -> bool:
    """Return whether a settings search can be satisfied by one category."""

    tokens = query.casefold().split()
    if not tokens:
        return True
    searchable = " ".join((category, *_CATEGORY_SEARCH_TERMS.get(category, ()))).casefold()
    return all(token in searchable for token in tokens)


def settings_uses_stacked_layout(available_width: float, style_scale: float) -> bool:
    """Keep the settings page usable when scaled columns no longer fit side by side."""

    minimum_width = _CATEGORY_WIDTH_PT + _COLUMN_GAP_PT + _PAGE_MIN_WIDTH_PT
    return float(available_width) < minimum_width * float(style_scale)


def responsive_flag_groups(requested: int, available_width: float, style_scale: float) -> int:
    """Reduce render-flag columns before translated labels start clipping."""

    fit = int(float(available_width) / max(96.0 * float(style_scale), 1.0))
    return max(1, min(int(requested), fit))


def render_flag_label(flag: RenderFlag, translate, *, localized: bool) -> str:
    """Keep official MuJoCo flag tokens intact while localizing OpenGL flags."""

    return translate(flag.value) if localized else flag.value


def flag_groups() -> tuple[tuple[str, tuple[RenderFlag, ...]], ...]:
    rest = tuple(f for f in RenderFlag if f not in _RND_FLAGS and f not in _VIS_FLAGS)
    return (
        ("mjtRndFlag", _RND_FLAGS),
        ("mjtVisFlag", _VIS_FLAGS),
        ("opengl", rest),
    )


class SettingsPanel(Panel):
    name = "Settings"
    default_open = False
    shortcut = "F9"
    modal = False
    dock_with = "Camera"

    def __init__(self) -> None:
        super().__init__()
        self._view = DebugView.SHADED
        self._message = ""
        self._category = "General"
        self._search = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        scale = ctx.style_scale
        stacked = settings_uses_stacked_layout(
            imgui.get_content_region_avail().x,
            scale,
        )
        if stacked:
            imgui.set_next_item_width(-1.0)
            if imgui.begin_combo("##settings_category", ctx.tr(self._category)):
                for category in _CATEGORIES:
                    matched = settings_category_matches(category, self._search)
                    imgui.begin_disabled(not matched)
                    selected, _ = imgui.selectable(
                        f"{ctx.tr(category)}##settings_{category}",
                        self._category == category,
                    )
                    if selected:
                        self._category = category
                    imgui.end_disabled()
                imgui.end_combo()
            imgui.spacing()
        else:
            self._draw_category_rail(ctx, scale)
            imgui.same_line()

        imgui.begin_child(
            "settings_page",
            imgui.ImVec2(0.0, 0.0),
            imgui.ChildFlags_.always_use_window_padding,
        )
        imgui.set_next_item_width(-1.0)
        changed, self._search = search_input(
            "##settings_search",
            self._search,
            hint=ctx.tr("Search settings"),
            search_tooltip=ctx.tr("Search settings"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed and self._search and not settings_category_matches(self._category, self._search):
            self._category = next(
                (
                    category
                    for category in _CATEGORIES
                    if settings_category_matches(category, self._search)
                ),
                self._category,
            )
        imgui.spacing()
        imgui.text(ctx.tr(self._category))
        imgui.separator()
        if self._search and not settings_category_matches(self._category, self._search):
            imgui.text_disabled(ctx.tr("No matching settings"))
            imgui.end_child()
            return
        if self._category == "General":
            self._general(ctx)
        elif self._category == "Interaction":
            self._interaction(ctx)
        elif self._category == "Rendering":
            self._rendering(ctx)
        else:
            self._mujoco_visuals(ctx)
        imgui.end_child()

    def _draw_category_rail(self, ctx: PanelContext, scale: float) -> None:
        imgui.begin_child(
            "settings_categories",
            imgui.ImVec2(_CATEGORY_WIDTH_PT * scale, 0.0),
            0,
        )
        imgui.push_style_var(
            imgui.StyleVar_.selectable_text_align,
            imgui.ImVec2(0.5, 0.5),
        )
        for category in _CATEGORIES:
            matched = settings_category_matches(category, self._search)
            imgui.begin_disabled(not matched)
            selected, _ = imgui.selectable(
                f"{ctx.tr(category)}##settings_{category}", self._category == category
            )
            if selected:
                self._category = category
            imgui.end_disabled()
        imgui.pop_style_var()
        imgui.end_child()

    def _general(self, ctx: PanelContext) -> None:
        t = ctx.tr
        languages = tuple(Language)
        current = parse_language(ctx.language)
        labels = [LANGUAGE_LABELS[language] for language in languages]
        if not self._begin_properties("settings_general"):
            return
        self._property(t("Language"))
        changed, index = imgui.combo("##ui_language", languages.index(current), labels)
        if changed and ctx.set_language is not None:
            ctx.set_language(languages[index].value)
        if ctx.font_report is not None:
            self._property(t("UI font"))
            imgui.text_disabled(ctx.font_report.mono)
            self._property(t("CJK font"))
            imgui.text_disabled(ctx.font_report.cjk or ctx.tr("none"))
        imgui.end_table()

    def _rendering(self, ctx: PanelContext) -> None:
        t = ctx.tr
        caps = ctx.backend.caps
        opengl_flags = flag_groups()[-1][1]
        if opengl_flags:
            self._group_heading(t("OpenGL render flags"))
            self._flag_table(ctx, "opengl_render_flags", opengl_flags, groups=3)

        flags = imgui.TreeNodeFlags_.default_open if self._search else 0
        if imgui.collapsing_header(
            f"{t('Debug')}###render_debug", flags
        ) and self._begin_properties("settings_render_debug"):
            self._property(t("Debug view"))
            self._debug_view(ctx)
            self._property(t("Labels"))
            self._label_mode(ctx)
            self._property(t("Frames"))
            self._frame_mode(ctx)
            imgui.end_table()

        self._group_heading(t("Backend info"))
        if self._begin_properties("settings_render_backend"):
            self._property(t("Backend"))
            imgui.text_disabled(caps.name)
            if caps.gl_version:
                self._property(t("Graphics device"))
                imgui.text_wrapped(f"{caps.gl_version}  {caps.renderer}")
            light_notes = ctx.backend.stats.notes
            for name in ("scene lights", "shadow casters"):
                if name in light_notes:
                    self._property(t(name))
                    imgui.text_disabled(str(light_notes[name]))
            imgui.end_table()

    def _interaction(self, ctx: PanelContext) -> None:
        t = ctx.tr
        if ctx.gizmo is not None:
            self._group_heading(t("Gizmo"))
            if self._begin_properties("settings_interaction_gizmo"):
                self._property(t("Style"))
                style_index = segmented_control(
                    "gizmo-style",
                    ("2D", "3D"),
                    1 if ctx.gizmo.style == "3d" else 0,
                    theme=ctx.theme,
                )
                ctx.gizmo.set_style("3d" if style_index == 1 else "2d")
                self._property(t("Orientation"))
                frame_index = segmented_control(
                    "gizmo-frame",
                    (t("Body"), t("World")),
                    1 if ctx.gizmo.space == "world" else 0,
                    theme=ctx.theme,
                )
                ctx.gizmo.set_space("world" if frame_index == 1 else "body")
                self._property(t("Overlay size"))
                changed, overlay_scale = imgui.drag_float(
                    "##viewport_overlay_scale",
                    float(ctx.viewport_overlay_scale),
                    0.02,
                    MIN_VIEWPORT_OVERLAY_SCALE,
                    MAX_VIEWPORT_OVERLAY_SCALE,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                committed = imgui.is_item_deactivated_after_edit()
                reset = False
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed = True
                    reset = True
                    overlay_scale = DEFAULT_VIEWPORT_OVERLAY_SCALE
                if hovered:
                    imgui.set_tooltip(
                        t("Scale playback, tools, and context hints; right-click to reset")
                    )
                if (changed or committed or reset) and ctx.set_viewport_overlay_scale is not None:
                    ctx.set_viewport_overlay_scale(
                        overlay_scale,
                        persist=committed or reset,
                    )
                imgui.end_table()

            self._group_heading(t("Input"))
            if self._begin_properties("settings_interaction_input"):
                self._property(t("Keep mode/unit"))
                changed, remember = themed_checkbox(
                    "##remember_precise_input_choices",
                    bool(ctx.gizmo.remember_precise_input_choices),
                    ctx.theme,
                )
                imgui.set_item_tooltip(
                    t("Reuse the last relative/absolute mode and angle unit across editor sessions")
                )
                if changed:
                    if ctx.set_precise_input_memory is not None:
                        ctx.set_precise_input_memory(remember)
                    else:
                        ctx.gizmo.remember_precise_input_choices = remember
                imgui.end_table()

            if ctx.input_bindings is not None:
                self._shortcut_settings(ctx)

            self._group_heading(t("Snap · Shift"))
            if self._begin_properties("settings_interaction_snap"):
                self._property(t("Position"))
                changed, step = imgui.drag_float(
                    "##position_snap",
                    float(ctx.gizmo.translation_snap_m),
                    0.01,
                    0.01,
                    100.0,
                    "%.3f m",
                )
                hovered = imgui.is_item_hovered()
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed, step = True, DEFAULT_TRANSLATION_SNAP_M
                if hovered:
                    imgui.set_tooltip(t("Right-click to reset"))
                if changed:
                    ctx.gizmo.translation_snap_m = step

                self._property(t("Rotation"))
                changed, step = imgui.drag_float(
                    "##rotation_snap",
                    float(ctx.gizmo.rotation_snap_deg),
                    0.1,
                    0.5,
                    180.0,
                    "%.1f deg",
                )
                hovered = imgui.is_item_hovered()
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed, step = True, DEFAULT_ROTATION_SNAP_DEG
                if hovered:
                    imgui.set_tooltip(t("Right-click to reset"))
                if changed:
                    ctx.gizmo.rotation_snap_deg = step

                self._property(t("Tick scale"))
                changed, tick_scale = imgui.drag_float(
                    "##rotation_tick_scale",
                    float(ctx.gizmo.rotation_tick_scale),
                    0.05,
                    0.5,
                    3.0,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed, tick_scale = True, DEFAULT_ROTATION_TICK_SCALE
                if hovered:
                    imgui.set_tooltip(t("Right-click to reset"))
                if changed:
                    ctx.gizmo.rotation_tick_scale = tick_scale
                imgui.end_table()

        if ctx.view_cube is not None:
            self._group_heading(t("View"))
            if self._begin_properties("settings_interaction_view"):
                self._property(t("Padding"))
                changed, padding = imgui.drag_float(
                    "##view_selection_padding",
                    float(ctx.view_cube.selection_padding),
                    0.02,
                    MIN_SELECTION_PADDING,
                    MAX_SELECTION_PADDING,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                committed = imgui.is_item_deactivated_after_edit()
                reset = False
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed, padding = True, DEFAULT_SELECTION_PADDING
                    reset = True
                if hovered:
                    imgui.set_tooltip(
                        t("1x is a tight fit; larger values move the view farther away")
                    )
                if changed:
                    ctx.view_cube.selection_padding = padding
                if ctx.set_view_selection_padding is not None and (committed or reset):
                    ctx.set_view_selection_padding(ctx.view_cube.selection_padding)
                imgui.end_table()

        if ctx.perturb is not None:
            self._group_heading(t("Perturb"))
            if self._begin_properties("settings_interaction_perturb"):
                self._property(t("Corner radius"))
                changed, radius = imgui.drag_float(
                    "##perturb_corner_radius",
                    float(ctx.perturb.outline_corner_radius_pt),
                    0.1,
                    0.0,
                    24.0,
                    "%.1f px",
                )
                hovered = imgui.is_item_hovered()
                if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    changed = True
                    radius = OUTLINE_CORNER_RADIUS_PT
                if hovered:
                    imgui.set_tooltip(t("Right-click to reset"))
                if changed:
                    ctx.perturb.outline_corner_radius_pt = radius
                imgui.end_table()

        if ctx.scene_entities is not None:
            self._group_heading(t("Helpers"))
            if self._begin_properties("settings_interaction_helpers"):
                self._property(t("Entities"))
                changed, visible = themed_checkbox(
                    "##scene_entity_helpers",
                    ctx.scene_entities.visible,
                    ctx.theme,
                )
                if changed:
                    ctx.scene_entities.visible = visible
                self._property(t("Volumes"))
                imgui.begin_disabled(not ctx.scene_entities.visible)
                changed, influence = themed_checkbox(
                    "##selected_influence_volumes",
                    ctx.scene_entities.show_influence,
                    ctx.theme,
                )
                imgui.end_disabled()
                if changed:
                    ctx.scene_entities.show_influence = influence
                imgui.end_table()

    def _shortcut_settings(self, ctx: PanelContext) -> None:
        """Draw conflict-free viewport key remapping from the shared map."""

        t = ctx.tr
        choices = key_choices()
        labels = tuple(choice.label for choice in choices)
        identifiers = tuple(choice.identifier for choice in choices)
        self._group_heading(t("Shortcuts"))
        if not self._begin_properties("settings_interaction_shortcuts"):
            return
        for action in InputAction:
            self._property(t(input_action_name(action)))
            current = ctx.input_bindings.key_id(action)
            index = identifiers.index(current)
            changed, index = imgui.combo(
                f"##shortcut_{action.value}",
                index,
                labels,
            )
            imgui.set_item_tooltip(t("A key already in use swaps the two actions"))
            if changed and ctx.set_input_binding is not None:
                ctx.set_input_binding(action, identifiers[index])
        self._property("")
        if imgui.button(t("Reset shortcuts")) and ctx.reset_input_bindings is not None:
            ctx.reset_input_bindings()
        imgui.end_table()

    def _mujoco_visuals(self, ctx: PanelContext) -> None:
        self._visual_groups(ctx)
        self._bvh_depth(ctx)

        for title, flags in flag_groups()[:2]:
            if not flags:
                continue
            if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
                continue
            self._flag_table(ctx, f"settings_{title}", flags, translate_labels=False)

        if self._message:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._message)

    @staticmethod
    def _begin_properties(str_id: str) -> bool:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.pad_outer_x
        if not imgui.begin_table(str_id, 2, flags):
            return False
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 0.64)
        return True

    @staticmethod
    def _group_heading(title: str) -> None:
        imgui.spacing()
        imgui.text_disabled(title)
        imgui.separator()

    @staticmethod
    def _property(label: str) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        width = imgui.get_content_region_avail().x
        text_width = imgui.calc_text_size(label).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, width - text_width))
        imgui.text(label)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)

    def _visual_groups(self, ctx: PanelContext) -> None:
        groups = ctx.session.visual_groups()
        if not groups:
            return
        if imgui.collapsing_header(
            f"{ctx.tr('visual groups')}###visual_groups", imgui.TreeNodeFlags_.default_open
        ):
            flags = imgui.TableFlags_.sizing_stretch_same
            if imgui.begin_table("mujoco_visual_groups", 7, flags):
                imgui.table_setup_column(
                    "category",
                    imgui.TableColumnFlags_.width_fixed,
                    84.0 * ctx.style_scale,
                )
                for index in range(6):
                    imgui.table_setup_column(str(index), imgui.TableColumnFlags_.width_stretch, 1.0)
                row_height = imgui.get_frame_height()

                def centered_text(value: str, *, disabled: bool = False) -> None:
                    imgui.align_text_to_frame_padding()
                    available = imgui.get_content_region_avail().x
                    text_width = imgui.calc_text_size(value).x
                    imgui.set_cursor_pos_x(
                        imgui.get_cursor_pos_x() + max(0.0, (available - text_width) * 0.5)
                    )
                    if disabled:
                        imgui.text_disabled(value)
                    else:
                        imgui.text(value)

                imgui.table_next_row(0, row_height)
                for label in ("category", "0", "1", "2", "3", "4", "5"):
                    imgui.table_next_column()
                    centered_text(label, disabled=True)
                for family in groups:
                    imgui.table_next_row(0, row_height)
                    imgui.table_next_column()
                    centered_text(family.category)
                    for i, visible in enumerate(family.visible):
                        imgui.table_next_column()
                        available = imgui.get_content_region_avail().x
                        checkbox_size = imgui.get_frame_height()
                        imgui.set_cursor_pos_x(
                            imgui.get_cursor_pos_x() + max(0.0, (available - checkbox_size) * 0.5)
                        )
                        changed, value = themed_checkbox(
                            f"##visual_group_{family.category}_{i}", visible, ctx.theme
                        )
                        if changed:
                            ctx.submit(cmd.SetVisualGroup(family.category, i, value))
                imgui.end_table()
        imgui.separator()

    def _bvh_depth(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        if not (
            backend.caps.supports(RenderFlag.BODYBVH) or backend.caps.supports(RenderFlag.MESHBVH)
        ):
            return
        if self._begin_properties("settings_bvh_depth"):
            self._property(ctx.tr("BVH depth"))
            changed, depth = imgui.drag_int(
                "##bvh_depth", backend.get_bvh_depth(), 1.0, 0, 64, "%d"
            )
            hovered = imgui.is_item_hovered()
            if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                changed, depth = True, 0
            if hovered:
                imgui.set_tooltip(ctx.tr("Right-click to reset"))
            if changed:
                backend.set_bvh_depth(depth)
            imgui.end_table()
        imgui.separator()

    def _flag_row(self, ctx: PanelContext, flag: RenderFlag, display_label: str) -> None:
        caps = ctx.backend.caps
        supported = caps.supports(flag)
        imgui.begin_disabled(not supported)
        changed, value = themed_checkbox(
            f"##rf_{flag.value}", ctx.backend.get_flag(flag), ctx.theme
        )
        imgui.end_disabled()
        if not supported:
            imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{display_label}”")
            return
        if changed and not ctx.backend.set_flag(flag, value):
            self._message = (
                f"{display_label} · {ctx.tr('backend refused the change')} ({caps.name})"
            )

    def _flag_table(
        self,
        ctx: PanelContext,
        table_id: str,
        flags: tuple[RenderFlag, ...],
        groups: int = 2,
        *,
        translate_labels: bool = True,
    ) -> None:
        groups = responsive_flag_groups(
            groups,
            imgui.get_content_region_avail().x,
            ctx.style_scale,
        )
        table_flags = imgui.TableFlags_.sizing_stretch_prop
        if not imgui.begin_table(table_id, groups * 2, table_flags):
            return
        for group in range(groups):
            imgui.table_setup_column(f"label {group}", imgui.TableColumnFlags_.width_stretch, 1.0)
            imgui.table_setup_column(
                f"value {group}",
                imgui.TableColumnFlags_.width_fixed,
                28.0 * ctx.style_scale,
            )
        row_count = (len(flags) + groups - 1) // groups
        for row in range(row_count):
            imgui.table_next_row()
            for group in range(groups):
                index = group * row_count + row
                imgui.table_next_column()
                if index >= len(flags):
                    imgui.table_next_column()
                    continue
                flag = flags[index]
                imgui.align_text_to_frame_padding()
                width = imgui.get_content_region_avail().x
                display_label = render_flag_label(
                    flag,
                    ctx.tr,
                    localized=translate_labels,
                )
                label_width = imgui.calc_text_size(display_label).x
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, width - label_width))
                imgui.text(display_label)
                imgui.table_next_column()
                self._flag_row(ctx, flag, display_label)
        imgui.end_table()

    def current_view(self, backend) -> DebugView:
        getter = getattr(backend, "get_debug_view", None)
        return getter() if callable(getter) else self._view

    def _debug_view(self, ctx: PanelContext) -> None:
        caps = ctx.backend.caps
        current = self.current_view(ctx.backend)
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##debugview", current.value):
            return

        for view in DebugView:
            ok = (view in caps.debug_views) if caps.debug_views else False
            imgui.begin_disabled(not ok)
            selected, _ = imgui.selectable(view.value, view is current)
            imgui.end_disabled()
            if not ok:
                imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{view.value}”")
            elif selected:
                if ctx.backend.set_debug_view(view):
                    self._view = view
                    self._message = ""
                else:
                    self._message = (
                        f"{view.value} · {ctx.tr('backend refused the change')} ({caps.name})"
                    )
        imgui.end_combo()

    def _label_mode(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        label = backend.get_label_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##label_mode", label.value):
            for mode in LabelMode:
                supported = mode in backend.caps.label_modes
                imgui.begin_disabled(not supported)
                selected, _ = imgui.selectable(mode.value, mode is label)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_label_mode(mode)
            imgui.end_combo()

    def _frame_mode(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        frame = backend.get_frame_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##frame_mode", frame.value):
            for mode in FrameMode:
                supported = mode in backend.caps.frame_modes
                imgui.begin_disabled(not supported)
                selected, _ = imgui.selectable(mode.value, mode is frame)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_frame_mode(mode)
            imgui.end_combo()
