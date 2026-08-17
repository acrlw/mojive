from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from ...render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from ..gizmo import DEFAULT_ROTATION_SNAP_DEG, DEFAULT_TRANSLATION_SNAP_M
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

    def __init__(self) -> None:
        super().__init__()
        self._view = DebugView.SHADED

        self._message = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        caps = ctx.backend.caps
        imgui.text_disabled(f"backend: {caps.name}")
        if caps.gl_version:
            imgui.text_disabled(f"{caps.gl_version}  {caps.renderer}")

        if ctx.gizmo is not None:
            solid = ctx.gizmo.style == "3d"
            changed, solid = imgui.checkbox("3D gizmo", solid)
            imgui.set_item_tooltip("Use the flat 2D overlay")
            if changed:
                ctx.gizmo.set_style("3d" if solid else "2d")
            world = ctx.gizmo.space == "world"
            changed, world = imgui.checkbox("World frame (T)", world)
            if changed:
                ctx.gizmo.set_space("world" if world else "body")

            imgui.align_text_to_frame_padding()
            imgui.text("position snap (Shift)")
            imgui.same_line()
            imgui.set_next_item_width(-1.0)
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

            imgui.align_text_to_frame_padding()
            imgui.text("rotation snap (Shift)")
            imgui.same_line()
            imgui.set_next_item_width(-1.0)
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

        if ctx.perturb is not None:
            imgui.align_text_to_frame_padding()
            imgui.text("perturb corner radius")
            imgui.same_line()
            imgui.set_next_item_width(-1.0)
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

        self._debug_view(ctx)
        self._overlay_modes(ctx)
        imgui.separator()

        self._visual_groups(ctx)

        for title, flags in flag_groups():
            if not flags:
                continue
            if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
                continue
            for flag in flags:
                self._flag_row(ctx, flag)

        if self._message:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._message)

    def _visual_groups(self, ctx: PanelContext) -> None:
        groups = ctx.session.visual_groups()
        if not groups:
            return
        if imgui.collapsing_header("visual groups", imgui.TreeNodeFlags_.default_open):
            for family in groups:
                imgui.align_text_to_frame_padding()
                imgui.text(family.category)
                for i, visible in enumerate(family.visible):
                    imgui.same_line()
                    changed, value = imgui.checkbox(f"{i}##visual_group_{family.category}", visible)
                    if changed:
                        ctx.submit(cmd.SetVisualGroup(family.category, i, value))
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
