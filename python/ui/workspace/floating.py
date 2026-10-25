"""Shared presentation for application-owned floating forms and dialogs."""

from contextlib import contextmanager
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect, independent_corner_scope, part_scope
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.text_layout import fit_text
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.form_popup import form_corners
from mojive.ui.workspace.menu_geometry import native_edges, native_unit
from mojive.ui.workspace.panel_geometry import framed_control_radius_limit
from mojive.ui.workspace.property_style import advance
from mojive.ui.workspace.widget_style import TEXT
from mojive.ui.workspace.window_chrome import surface_ink


@contextmanager
def floating_style(ui, *, part="form_popup"):
    """Give each floating owner independent surface, typography and control geometry."""
    if ui is None:
        yield
        return
    with independent_corner_scope(), part_scope(part, inherit=False):
        scale, style = ui.s, ui.component_style(part)
        field = ui.component_style("input_text")
        field_padding = native_edges(field, "padding", scale, (8, 0, 8, 0))
        font = style.get("fontSize", BODY_SIZE)
        height = max(field.get("height", DESIGN.control_height), font * 1.4)
        corners = form_corners(
            ui, part, control_part="input_text", inner_limit=framed_control_radius_limit(ui)
        )
        padding, radius, field_radius = corners.padding, corners.radius, corners.inner_radius
        ink = surface_ink(ui, part, "popup")
        colors = (
            (imgui.Col_.window_bg, ink),
            (imgui.Col_.popup_bg, ink),
            (imgui.Col_.title_bg, ink),
            (imgui.Col_.title_bg_active, ink),
        )
        values = (
            (imgui.StyleVar_.window_padding, padding[:2]),
            (imgui.StyleVar_.window_min_size, (0, 0)),
            (imgui.StyleVar_.window_rounding, radius),
            (imgui.StyleVar_.popup_rounding, radius),
            (imgui.StyleVar_.window_border_size, 0),
            (imgui.StyleVar_.popup_border_size, 0),
            (imgui.StyleVar_.frame_rounding, field_radius),
            (
                imgui.StyleVar_.frame_padding,
                (field_padding[0], (ceil(height * scale) - font * scale) / 2),
            ),
            (
                imgui.StyleVar_.item_spacing,
                (
                    native_unit(DESIGN.item_spacing_x * scale),
                    native_unit(ui.metric("form_popup", "rowGap", 6) * scale),
                ),
            ),
        )
        for key, value in colors:
            imgui.push_style_color(key, tuple(channel / 255 for channel in value))
        for key, value in values:
            imgui.push_style_var(key, value)
        imgui.push_font(ui.resolve_font(style), font * scale)
        try:
            with (
                ui.component_scope(part),
                ui.corner_scope(
                    field_radius,
                    (
                        "button",
                        "input_text",
                        "input_number",
                        "axis_input",
                        "textarea",
                        "segment",
                        "select",
                        "select_option",
                    ),
                ),
            ):
                yield
        finally:
            imgui.pop_font()
            imgui.pop_style_var(len(values))
            imgui.pop_style_color(len(colors))


@contextmanager
def floating_number(ui):
    """Keep native scalar text entry aligned with the shared numeric typography."""
    if ui is None:
        yield
        return
    style, scale = ui.component_style("input_number"), ui.s
    font = style.get("fontSize", BODY_SIZE)
    height = max(style.get("height", DESIGN.control_height), style.line_height(font * 1.4))
    padding = style.edges("padding", scale, (8, 0, 8, 0))
    imgui.push_font(ui.resolve_font(style, mono=True), font * scale)
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding, (padding[0], (ceil(height * scale) - font * scale) / 2)
    )
    try:
        yield
    finally:
        imgui.pop_style_var()
        imgui.pop_font()


def floating_segment(ui, key, labels, selected, *, theme, width=None):
    from mojive.ui.controls import segmented_control

    if ui is None:
        return segmented_control(
            key, labels, selected, theme=theme, width=0.0 if width is None else width
        )
    pos = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x if width is None else width
    return ui.segmented(key, pos.x, pos.y, width / ui.s, labels, selected)


def floating_button_width(ui, label):
    if ui is None:
        return (
            imgui.calc_text_size(label, hide_text_after_double_hash=True).x
            + 2 * imgui.get_style().frame_padding.x
        )
    style = ui.component_style("button")
    padding = style.edges("padding", ui.s, (8, 0, 8, 0))
    caption = label.split("##", 1)[0]
    natural = ui.measure(caption, style.get("fontSize", BODY_SIZE), translate=False, style=style)
    return natural + padding[0] + padding[2]


def floating_button(ui, label, *, width=0, primary=False):
    """Use the shared button painter inside application-owned dialogs."""
    if ui is None:
        return imgui.button(label, (width, 0))
    pos = imgui.get_cursor_screen_pos()
    caption = label.split("##", 1)[0]
    width = min(width or floating_button_width(ui, label), imgui.get_content_region_avail().x)
    return ui.button(
        label,
        pos.x,
        pos.y,
        width / ui.s,
        label=caption,
        filled=True,
        active=primary,
        solid=primary,
        translate=False,
        width_limit=width,
    )


def floating_actions(ui, labels, *, primary=0):
    """Align dialog actions with the content edge and leave space above the footer."""
    scale = ui.s if ui is not None else imgui.get_font_size() / BODY_SIZE
    gap = 8 * scale
    width = min(
        max(88 * scale, *(floating_button_width(ui, label) for label in labels)),
        max(1, (imgui.get_content_region_avail().x - gap * (len(labels) - 1)) / len(labels)),
    )
    height = (
        ui.metric("button", "height", DESIGN.control_height) * scale
        if ui is not None
        else imgui.get_frame_height()
    )
    pos = imgui.get_cursor_screen_pos()
    available = imgui.get_content_region_avail()
    y = max(pos.y + 16 * scale, pos.y + available.y - height)
    x = pos.x + available.x - width * len(labels) - gap * (len(labels) - 1)
    actions = []
    for index, label in enumerate(labels):
        imgui.set_cursor_screen_pos((x + index * (width + gap), y))
        actions.append(floating_button(ui, label, width=width, primary=index == primary))
    return tuple(actions)


def floating_action_gap(ui):
    """Separate confirmation actions from the preceding body by sixteen points."""
    if ui is None:
        imgui.spacing()
        return
    pos = imgui.get_cursor_screen_pos()
    advance(ui, pos.x, pos.y + max(0, 16 * ui.s - imgui.get_style().item_spacing.y))


def floating_heading_height(ui, *, modal=False, closable=False):
    part = "dialog_header" if modal else "form_label"
    style, scale = ui.component_style(part), ui.s
    font = style.get("titleSize", 20) if modal else style.get("fontSize", BODY_SIZE)
    gap = native_unit(
        (style.get("gap", 8) if modal else ui.metric("form_popup", "rowGap", 6)) * scale
    )
    height = ceil(style.line_height(font * 1.4) * scale)
    if closable:
        height = max(height, control_size(ui, "floating-close", 24, 24, icon="close")[1])
    return height + 2 * gap + scale


def floating_heading(ui, title, *, modal=False, movable=False, closable=False):
    """Paint a measured heading; a movable form retains a native window drag target."""
    if ui is None:
        imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), imgui.get_style().window_padding.x))
        fitted = fit_text(ImguiDraw2D(), title, imgui.get_content_region_avail().x, middle=True)
        imgui.text(fitted)
        if fitted != title and imgui.is_item_hovered():
            imgui.set_tooltip(title)
        imgui.separator()
        return False
    part = "dialog_header" if modal else "form_label"
    style, scale = ui.component_style(part), ui.s
    font = style.get("titleSize", 20) if modal else style.get("fontSize", BODY_SIZE)
    imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), imgui.get_style().window_padding.x))
    pos = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    height = ceil(style.line_height(font * 1.4) * scale)
    close_size = control_size(ui, "floating-close", 24, 24, icon="close") if closable else (0, 0)
    height = max(height, close_size[1])
    title_width = max(1, width - close_size[0] - (6 * scale if closable else 0))
    rect = Rect(pos.x, pos.y, title_width, height)
    key = "floating-heading/" + imgui.get_current_context().current_window.name
    imgui.invisible_button(key, (title_width, height))
    ui.hits[key] = rect.tuple()
    if movable and imgui.is_item_active() and imgui.is_mouse_dragging(0):
        imgui.internal.start_mouse_moving_window(imgui.get_current_context().current_window)
    ui.component_text(part, rect, title, TEXT, size=font, bold=modal, translate=False)
    if (
        imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal)
        and ui.measure(title, font, translate=False, style=style) > title_width
    ):
        ui.tooltip(title, translate=False)
    closed = False
    if closable:
        frame = control_frame_size(ui, "floating-close", *close_size, icon="close")
        closed = ui.button(
            f"floating-close/{imgui.get_current_context().current_window.id_}",
            pos.x + width - close_size[0],
            pos.y + (height - close_size[1]) / 2,
            *frame,
            icon="close",
            icon_role="subtle",
            tooltip="Close",
        )
    gap = native_unit(
        (style.get("gap", 8) if modal else ui.metric("form_popup", "rowGap", 6)) * scale
    )
    ui.line(pos.x, pos.y + height + gap, width)
    advance(ui, pos.x, pos.y + floating_heading_height(ui, modal=modal, closable=closable))
    return closed


@contextmanager
def floating_window(ui, title, *, popup=False, modal=False, flags=0, heading=None, on_close=None):
    """Pair native Begin/End while applying the production floating-form presentation.

    ViewerApp also serves legacy surfaces without a workspace UI. Those retain
    native window chrome; the application continues to own every edit and action.
    """
    part = "dialog" if modal else "form_popup"
    with floating_style(ui, part=part):
        closed = False
        if ui is not None:
            flags |= imgui.WindowFlags_.no_title_bar
            imgui.set_next_window_bg_alpha(1)
        if modal:
            opened, _ = imgui.begin_popup_modal(title, None, flags)
        elif popup:
            opened = imgui.begin_popup(title, flags)
        else:
            opened, retained = imgui.begin(title, True if on_close else None, flags)
            if on_close and not retained:
                closed = True
                on_close()
        try:
            if opened:
                text = title.split("##", 1)[0] if heading is None else heading
                if text and (ui is not None or heading is not None):
                    closed = floating_heading(
                        ui,
                        text,
                        modal=modal,
                        movable=not bool(flags & imgui.WindowFlags_.no_move),
                        closable=on_close is not None,
                    )
                    if closed:
                        on_close()
                if ui is not None:
                    window = imgui.get_current_context().current_window
                    outer = Rect(*window.pos, *window.size)
                    padding = (*window.window_padding, *window.window_padding)
                    ui.hits["floating/" + title] = outer.tuple()
                    ui.record_component_box(
                        part,
                        Box(outer, outer, outer.inset(*padding), padding, (0,) * 4),
                        key="floating/" + title,
                        nativeWindowRadius=window.window_rounding,
                        nativeWindowPadding=list(window.window_padding),
                    )
            yield opened and not closed
        finally:
            if modal or popup:
                if opened:
                    imgui.end_popup()
            else:
                imgui.end()
