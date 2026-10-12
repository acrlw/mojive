"""Visuals settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.render.backend import RenderFlag
from mojive.types import ContactStyle
from mojive.ui.geometry_view import draw_geometry_view
from mojive.ui.panels import (
    PanelContext,
    pointer_hint,
    pointer_pressed,
    themed_checkbox,
)
from mojive.ui.pointer_bindings import PointerAction

from .rendering import flag_groups


def draw(panel, ctx: PanelContext) -> None:
    imgui.text(ctx.tr("Geometry view"))
    draw_geometry_view(ctx.backend, ctx.tr, compact=True, theme=ctx.theme)
    geometry_style(panel, ctx)
    contact_style(panel, ctx)
    panel._visual_groups(ctx)
    bvh_depth(panel, ctx)

    for title, flags in flag_groups()[:2]:
        if not flags:
            continue
        if not panel._collapsing_header(ctx.tr(title), imgui.TreeNodeFlags_.default_open):
            continue
        panel._flag_table(ctx, f"settings_{title}", flags, translate_labels=False)

    if panel._message:
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), panel._message)


def contact_style(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    if not panel._collapsing_header(
        f"{t('Contact points')}###contact_style", imgui.TreeNodeFlags_.default_open
    ):
        return
    style = ctx.backend.get_contact_style()
    committed = False
    if panel._begin_properties("contact_style_properties"):
        panel._property(t("Show contact points"))
        changed, enabled = themed_checkbox(
            "##contact_enabled", ctx.backend.get_flag(RenderFlag.CONTACTPOINT), ctx.theme
        )
        if changed:
            ctx.backend.set_flag(RenderFlag.CONTACTPOINT, enabled)
        panel._property(t("Shape"))
        if imgui.begin_combo("##contact_shape", t(style.shape.title())):
            for shape in ("cylinder", "sphere", "point"):
                if imgui.selectable(t(shape.title()), style.shape == shape)[0]:
                    style = replace(style, shape=shape)
                    committed = True
            imgui.end_combo()
        panel._property(t("Use model color"))
        changed, use_model_color = themed_checkbox(
            "##contact_model_color", style.use_model_color, ctx.theme
        )
        if changed:
            style = replace(style, use_model_color=use_model_color)
            committed = True
        panel._property(t("Color"))
        imgui.begin_disabled(style.use_model_color)
        changed, color = panel._color_edit4("##contact_color", style.color)
        committed |= panel._color_committed()
        imgui.end_disabled()
        if changed:
            style = replace(style, color=tuple(color))
        imgui.set_item_tooltip(t("Island visualization overrides contact colors."))
        panel._property(t("Size"))
        changed, scale = panel._slider_float(
            "##contact_scale", style.scale, 0.1, 10, "%.2fx", imgui.SliderFlags_.always_clamp
        )
        committed |= panel._value_committed()
        if changed:
            try:
                style = replace(style, scale=scale)
            except ValueError as error:
                ctx.report(str(error))
        imgui.end_table()
    if imgui.button(t("Reset contact style")):
        style = ContactStyle()
        committed = True
    if committed or style != ctx.backend.get_contact_style():
        if ctx.set_contact_style is not None:
            ctx.apply_setting(ctx.set_contact_style, style, persist=committed)
        else:
            ctx.backend.set_contact_style(style)


def geometry_style(panel, ctx: PanelContext) -> None:
    if not panel._collapsing_header(
        f"{ctx.tr('Both appearance')}###geometry_style", imgui.TreeNodeFlags_.default_open
    ):
        return
    style = ctx.backend.get_geometry_style()
    committed = False
    if panel._begin_properties("geometry_style_properties"):
        panel._property(ctx.tr("Collision color"))
        changed, color = panel._color_edit3("##collision_color", style.collision_color)
        committed |= panel._color_committed()
        if changed:
            style = replace(style, collision_color=tuple(color))
        for name, label in (
            ("visual_opacity", "Visual opacity"),
            ("collision_opacity", "Collision opacity"),
        ):
            panel._property(ctx.tr(label))
            changed, opacity = panel._slider_float(
                f"##{name}", getattr(style, name), 0.0, 1.0, "%.2f"
            )
            committed |= panel._value_committed()
            if changed:
                style = replace(style, **{name: opacity})
        imgui.end_table()
    if committed or style != ctx.backend.get_geometry_style():
        if ctx.set_geometry_style is not None:
            ctx.apply_setting(ctx.set_geometry_style, style, persist=committed)
        else:
            ctx.backend.set_geometry_style(style)


def visual_groups(panel, ctx: PanelContext) -> None:
    groups = ctx.session.visual_groups()
    if not groups:
        return
    if panel._collapsing_header(
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


def bvh_depth(panel, ctx: PanelContext) -> None:
    backend = ctx.backend
    if not (backend.caps.supports(RenderFlag.BODYBVH) or backend.caps.supports(RenderFlag.MESHBVH)):
        return
    if panel._begin_properties("settings_bvh_depth"):
        panel._property(ctx.tr("BVH depth"))
        changed, depth = panel._drag_int("##bvh_depth", backend.get_bvh_depth(), 1.0, 0, 64, "%d")
        hovered = imgui.is_item_hovered()
        if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
            changed, depth = True, 0
        if hovered:
            imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
        if changed:
            backend.set_bvh_depth(depth)
        imgui.end_table()
    imgui.separator()
