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
    RotationDialProjection,
)
from ..localization import LANGUAGE_LABELS, Language, parse_language
from ..perturb import OUTLINE_CORNER_RADIUS_PT
from . import Panel, PanelContext

_RND_FLAGS: tuple[RenderFlag, ...] = (
    RenderFlag.SHADOW,
    RenderFlag.WIREFRAME,
    RenderFlag.REFLECTION,
    RenderFlag.ADDITIVE,
    RenderFlag.SKYBOX,
    RenderFlag.FOG,
    RenderFlag.HAZE,
    RenderFlag.SEGMENT,
    RenderFlag.IDCOLOR,
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


def flag_groups() -> tuple[tuple[str, tuple[RenderFlag, ...]], ...]:
    rest = tuple(f for f in RenderFlag if f not in _RND_FLAGS and f not in _VIS_FLAGS)
    return (
        ("mjtRndFlag", _RND_FLAGS),
        ("mjtVisFlag", _VIS_FLAGS),
        ("forge", rest),
    )


class SettingsPanel(Panel):
    name = "Settings"
    default_open = False
    shortcut = "F9"
    modal = True
    initial_size = (820.0, 620.0)

    def __init__(self) -> None:
        super().__init__()
        self._view = DebugView.SHADED
        self._message = ""
        self._category = "General"

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        scale = ctx.style_scale
        footer_height = 42.0 * scale
        imgui.begin_child(
            "settings_categories",
            imgui.ImVec2(176.0 * scale, -footer_height),
            imgui.ChildFlags_.borders.value,
        )
        for category in ("General", "Interaction", "Rendering", "MuJoCo Visuals"):
            selected, _ = imgui.selectable(
                f"{ctx.tr(category)}##settings_{category}", self._category == category
            )
            if selected:
                self._category = category
        imgui.end_child()

        imgui.same_line()
        imgui.begin_child(
            "settings_page",
            imgui.ImVec2(0.0, -footer_height),
            imgui.ChildFlags_.borders.value,
        )
        imgui.text(ctx.tr(self._category))
        imgui.separator()
        if self._category == "General":
            self._general(ctx)
        elif self._category == "Interaction":
            self._interaction(ctx)
        elif self._category == "Rendering":
            self._rendering(ctx)
        else:
            self._mujoco_visuals(ctx)
        imgui.end_child()

        imgui.separator()
        close_width = 92.0 * scale
        imgui.set_cursor_pos_x(
            max(
                imgui.get_cursor_pos_x(),
                imgui.get_window_width() - close_width - 16.0 * scale,
            )
        )
        if imgui.button(ctx.tr("Close"), imgui.ImVec2(close_width, 0.0)) or imgui.is_key_pressed(
            imgui.Key.escape, False
        ):
            self.open = False
            imgui.close_current_popup()

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
            imgui.text(ctx.font_report.mono)
            self._property(t("CJK font"))
            imgui.text(ctx.font_report.cjk or "none")
        imgui.end_table()

    def _rendering(self, ctx: PanelContext) -> None:
        t = ctx.tr
        caps = ctx.backend.caps
        if self._begin_properties("settings_rendering"):
            self._property(t("Backend"))
            imgui.text(caps.name)
            if caps.gl_version:
                self._property(t("Graphics device"))
                imgui.text_wrapped(f"{caps.gl_version}  {caps.renderer}")
            light_notes = ctx.backend.stats.notes
            for name in ("scene lights", "shadow casters"):
                if name in light_notes:
                    self._property(t(name))
                    imgui.text(str(light_notes[name]))
            imgui.end_table()
        imgui.spacing()
        self._debug_view(ctx)
        self._overlay_modes(ctx)
        forge_flags = flag_groups()[-1][1]
        if forge_flags and imgui.collapsing_header(t("Forge render flags")):
            for flag in forge_flags:
                self._flag_row(ctx, flag)

    def _interaction(self, ctx: PanelContext) -> None:
        t = ctx.tr
        if not self._begin_properties("settings_interaction"):
            return
        if ctx.gizmo is not None:
            solid = ctx.gizmo.style == "3d"
            self._property(t("Gizmo style"))
            changed, solid = imgui.checkbox(f"{t('3D gizmo')}##3d_gizmo", solid)
            imgui.set_item_tooltip(t("Use the flat 2D overlay"))
            if changed:
                ctx.gizmo.set_style("3d" if solid else "2d")
            world = ctx.gizmo.space == "world"
            self._property(t("Gizmo orientation"))
            changed, world = imgui.checkbox(f"{t('World frame (T)')}##world_frame", world)
            if changed:
                ctx.gizmo.set_space("world" if world else "body")

            self._property(t("position snap (Shift)"))
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
                imgui.set_tooltip("drag: adjust · double-click: enter value · right-click: reset")
            if changed:
                ctx.gizmo.translation_snap_m = step

            self._property(t("rotation snap (Shift)"))
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
                imgui.set_tooltip("drag: adjust · double-click: enter value · right-click: reset")
            if changed:
                ctx.gizmo.rotation_snap_deg = step

            self._property(t("rotation tick scale"))
            changed, scale = imgui.drag_float(
                "##rotation_tick_scale",
                float(ctx.gizmo.rotation_tick_scale),
                0.05,
                0.5,
                3.0,
                "%.2fx",
            )
            hovered = imgui.is_item_hovered()
            if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                changed, scale = True, DEFAULT_ROTATION_TICK_SCALE
            if hovered:
                imgui.set_tooltip("drag: adjust · double-click: enter value · right-click: reset")
            if changed:
                ctx.gizmo.rotation_tick_scale = scale

            projection = ctx.gizmo.rotation_dial_projection
            self._property(t("Rotation dial projection"))
            if imgui.begin_combo("##rotation_dial_projection", projection.value):
                for option in RotationDialProjection:
                    selected, _ = imgui.selectable(option.value, option is projection)
                    if selected:
                        ctx.gizmo.set_rotation_dial_projection(option.value)
                imgui.end_combo()
            imgui.set_item_tooltip(
                "Orthographic keeps the dial screen-affine; classic follows the viewport camera"
            )

        if ctx.perturb is not None:
            self._property(t("perturb corner radius"))
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
                imgui.set_tooltip("drag: adjust · double-click: enter value · right-click: reset")
            if changed:
                ctx.perturb.outline_corner_radius_pt = radius

        if ctx.scene_entities is not None:
            self._property(t("Scene helpers"))
            changed, visible = imgui.checkbox(
                f"{t('scene entity helpers')}##scene_entity_helpers",
                ctx.scene_entities.visible,
            )
            if changed:
                ctx.scene_entities.visible = visible
            imgui.begin_disabled(not ctx.scene_entities.visible)
            changed, influence = imgui.checkbox(
                f"{t('selected influence volumes')}##selected_influence_volumes",
                ctx.scene_entities.show_influence,
            )
            imgui.end_disabled()
            if changed:
                ctx.scene_entities.show_influence = influence
        imgui.end_table()

    def _mujoco_visuals(self, ctx: PanelContext) -> None:
        self._visual_groups(ctx)
        self._bvh_depth(ctx)

        for title, flags in flag_groups()[:2]:
            if not flags:
                continue
            if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
                continue
            for flag in flags:
                self._flag_row(ctx, flag)

        if self._message:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._message)

    @staticmethod
    def _begin_properties(str_id: str) -> bool:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.pad_outer_x
        if not imgui.begin_table(str_id, 2, flags):
            return False
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.42)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 0.58)
        return True

    @staticmethod
    def _property(label: str) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
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
            for family in groups:
                imgui.align_text_to_frame_padding()
                imgui.text(family.category)
                for i, visible in enumerate(family.visible):
                    imgui.same_line()
                    changed, value = imgui.checkbox(f"{i}##visual_group_{family.category}", visible)
                    if changed:
                        ctx.submit(cmd.SetVisualGroup(family.category, i, value))
        imgui.separator()

    def _bvh_depth(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        if not (
            backend.caps.supports(RenderFlag.BODYBVH) or backend.caps.supports(RenderFlag.MESHBVH)
        ):
            return
        imgui.align_text_to_frame_padding()
        imgui.text(ctx.tr("BVH depth"))
        imgui.same_line()
        imgui.set_next_item_width(-1.0)
        changed, depth = imgui.drag_int("##bvh_depth", backend.get_bvh_depth(), 1.0, 0, 64, "%d")
        hovered = imgui.is_item_hovered()
        if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
            changed, depth = True, 0
        if hovered:
            imgui.set_tooltip("drag: adjust · double-click: enter value · right-click: reset")
        if changed:
            backend.set_bvh_depth(depth)
        imgui.separator()

    def _flag_row(self, ctx: PanelContext, flag: RenderFlag) -> None:
        caps = ctx.backend.caps
        supported = caps.supports(flag)
        imgui.begin_disabled(not supported)
        changed, value = imgui.checkbox(f"{flag.value}##rf", ctx.backend.get_flag(flag))
        imgui.end_disabled()
        if not supported:
            imgui.set_item_tooltip(f"{caps.name} does not implement '{flag.value}'")
            return
        if changed and not ctx.backend.set_flag(flag, value):
            self._message = f"'{flag.value}' refused by {caps.name}"

    def current_view(self, backend) -> DebugView:
        getter = getattr(backend, "get_debug_view", None)
        return getter() if callable(getter) else self._view

    def _debug_view(self, ctx: PanelContext) -> None:
        caps = ctx.backend.caps
        current = self.current_view(ctx.backend)
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##debugview", f"debug view: {current.value}"):
            return

        for view in DebugView:
            ok = (view in caps.debug_views) if caps.debug_views else False
            imgui.begin_disabled(not ok)
            selected, _ = imgui.selectable(view.value, view is current)
            imgui.end_disabled()
            if not ok:
                imgui.set_item_tooltip(f"{caps.name} does not implement '{view.value}'")
            elif selected:
                if ctx.backend.set_debug_view(view):
                    self._view = view
                    self._message = ""
                else:
                    self._message = f"debug view '{view.value}' refused by {caps.name}"
        imgui.end_combo()

    def _overlay_modes(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        label = backend.get_label_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##label_mode", f"labels: {label.value}"):
            for mode in LabelMode:
                supported = mode in backend.caps.label_modes
                imgui.begin_disabled(not supported)
                selected, _ = imgui.selectable(mode.value, mode is label)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_label_mode(mode)
            imgui.end_combo()

        frame = backend.get_frame_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##frame_mode", f"frames: {frame.value}"):
            for mode in FrameMode:
                supported = mode in backend.caps.frame_modes
                imgui.begin_disabled(not supported)
                selected, _ = imgui.selectable(mode.value, mode is frame)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_frame_mode(mode)
            imgui.end_combo()
