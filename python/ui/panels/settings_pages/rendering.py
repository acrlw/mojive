"""Rendering settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.render.backend import DebugView, FrameMode, LabelMode, RenderFlag, ShadowQuality
from mojive.ui.localization import render_note_text
from mojive.ui.panels import (
    PanelContext,
    padded_selectable,
    segmented_control,
    themed_checkbox,
)

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


def responsive_flag_groups(requested: int, available_width: float, style_scale: float) -> int:
    """Reduce render-flag columns before translated labels start clipping."""

    fit = int(float(available_width) / max(96.0 * float(style_scale), 1.0))
    return max(1, min(int(requested), fit))


def render_flag_label(flag: RenderFlag, translate, *, localized: bool) -> str:
    """Keep official MuJoCo tokens intact and name renderer display controls."""

    if flag == RenderFlag.MESH_LOD:
        return translate("Mesh LOD")
    return translate(flag.value) if localized else flag.value


def flag_groups() -> tuple[tuple[str, tuple[RenderFlag, ...]], ...]:
    rest = tuple(
        f
        for f in RenderFlag
        if f not in _RND_FLAGS
        and f not in _VIS_FLAGS
        and f not in (RenderFlag.VISUAL_GEOMETRY, RenderFlag.COLLISION_GEOMETRY)
    )
    return (
        ("mjtRndFlag", _RND_FLAGS),
        ("mjtVisFlag", _VIS_FLAGS),
        ("renderer", rest),
    )


def draw(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    caps = ctx.backend.caps
    if caps.shadows:
        panel._group_heading(t("Shadows"))
        if panel._begin_properties("settings_shadow_quality"):
            panel._property(t("Shadow quality"))
            getter = getattr(ctx.backend, "get_shadow_quality", None)
            try:
                current = ShadowQuality(getter()) if getter is not None else ShadowQuality.BALANCED
            except (TypeError, ValueError):
                current = ShadowQuality.BALANCED
            qualities = tuple(ShadowQuality)
            imgui.begin_group()
            index = segmented_control(
                "shadow-quality",
                tuple(t(label) for label in ("Performance", "Balanced", "High")),
                qualities.index(current),
                theme=ctx.theme,
            )
            imgui.end_group()
            imgui.set_item_tooltip(
                t("Higher quality smooths close-up shadow edges but costs more GPU time")
            )
            selected = qualities[index]
            if selected is not current and ctx.set_shadow_quality is not None:
                ctx.apply_setting(ctx.set_shadow_quality, selected)
            imgui.end_table()

    renderer_flags = flag_groups()[-1][1]
    if renderer_flags:
        panel._group_heading(t("Renderer flags"))
        panel._flag_table(ctx, "renderer_flags", renderer_flags, groups=3)

    flags = imgui.TreeNodeFlags_.default_open if panel._search else 0
    if panel._collapsing_header(f"{t('Debug')}###render_debug", flags) and panel._begin_properties(
        "settings_render_debug"
    ):
        panel._property(t("Debug view"))
        debug_view(panel, ctx)
        panel._property(t("Labels"))
        label_mode(panel, ctx)
        panel._property(t("Frames"))
        frame_mode(panel, ctx)
        imgui.end_table()

    panel._group_heading(t("Backend info"))
    if panel._begin_properties("settings_render_backend"):
        panel._property(t("Backend"))
        imgui.text_disabled(caps.name)
        if caps.gl_version:
            panel._property(t("Graphics device"))
            imgui.text_wrapped(f"{caps.gl_version}  {caps.renderer}")
        light_notes = ctx.backend.stats.notes
        for name in ("scene lights", "shadow casters"):
            if name in light_notes:
                panel._property(t(name))
                imgui.text_disabled(render_note_text(light_notes[name], t))
        imgui.end_table()


def flag_row(panel, ctx: PanelContext, flag: RenderFlag, display_label: str) -> None:
    caps = ctx.backend.caps
    supported = caps.supports(flag)
    imgui.begin_disabled(not supported)
    changed, value = themed_checkbox(f"##rf_{flag.value}", ctx.backend.get_flag(flag), ctx.theme)
    imgui.end_disabled()
    if not supported:
        imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{display_label}”")
        return
    if flag == RenderFlag.MESH_LOD:
        imgui.set_item_tooltip(
            ctx.tr("Adaptive display detail; depth and picking use the displayed mesh")
        )
    if changed and not ctx.backend.set_flag(flag, value):
        panel._message = f"{display_label} · {ctx.tr('backend refused the change')} ({caps.name})"


def flag_table(
    panel,
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
            panel._flag_row(ctx, flag, display_label)
    imgui.end_table()


def debug_view(panel, ctx: PanelContext) -> None:
    caps = ctx.backend.caps
    current = panel.current_view(ctx.backend)
    imgui.set_next_item_width(-1)
    if not imgui.begin_combo("##debugview", current.value):
        return

    for view in DebugView:
        ok = (view in caps.debug_views) if caps.debug_views else False
        imgui.begin_disabled(not ok)
        selected, _ = padded_selectable(view.value, view is current)
        imgui.end_disabled()
        if not ok:
            imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{view.value}”")
        elif selected:
            if ctx.backend.set_debug_view(view):
                panel._view = view
                panel._message = ""
            else:
                panel._message = (
                    f"{view.value} · {ctx.tr('backend refused the change')} ({caps.name})"
                )
    imgui.end_combo()


def label_mode(panel, ctx: PanelContext) -> None:
    backend = ctx.backend
    label = backend.get_label_mode()
    imgui.set_next_item_width(-1)
    if imgui.begin_combo("##label_mode", label.value):
        for mode in LabelMode:
            supported = mode in backend.caps.label_modes
            imgui.begin_disabled(not supported)
            selected, _ = padded_selectable(mode.value, mode is label)
            imgui.end_disabled()
            if selected and supported:
                backend.set_label_mode(mode)
        imgui.end_combo()


def frame_mode(panel, ctx: PanelContext) -> None:
    backend = ctx.backend
    frame = backend.get_frame_mode()
    imgui.set_next_item_width(-1)
    if imgui.begin_combo("##frame_mode", frame.value):
        for mode in FrameMode:
            supported = mode in backend.caps.frame_modes
            imgui.begin_disabled(not supported)
            selected, _ = padded_selectable(mode.value, mode is frame)
            imgui.end_disabled()
            if selected and supported:
                backend.set_frame_mode(mode)
        imgui.end_combo()
