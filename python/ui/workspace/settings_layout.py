"""Measure settings navigation and pages independently of preference actions."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.workspace.component_layout import (
    component_flow,
    control_frame_size,
    control_size,
)
from mojive.ui.workspace.design import DESIGN, current_design, use_design
from mojive.ui.workspace.property_style import property_controls
from mojive.ui.workspace.widgets import ACCENT, PANEL, TEXT


def _header(panel, rect):
    u, s = panel.workspace.ui, panel.workspace.s
    style = u.component_style("panel_header")
    font, icon = style.get("fontSize", DESIGN.title_size), style.get("iconSize", 20)
    pads, margin = style.edges("padding", s), style.edges("margin", s)
    close = control_size(u, "settings-close", 28, 28, icon="close")
    content_height = max(style.line_height(font * 1.4) * s, icon * s, close[1])
    minimum = content_height + pads[1] + pads[3]
    height = max(style.get("height", 32) * s, minimum) + margin[1] + margin[3]
    box = u.component_box(
        "panel_header",
        rect.x,
        rect.y,
        rect.width,
        height,
        key="settings-header",
        minimum_height=minimum,
    )
    gap = style.get("columnGap", style.get("gap", 8)) * s
    icon_rect = style.align_rect(box.content, icon * s, icon * s, horizontal="start")
    u.icon("settings", icon_rect.x, icon_rect.y, icon, ACCENT)
    text = Rect(
        icon_rect.right + gap,
        box.content.y,
        max(0, box.content.right - icon_rect.right - gap - close[0]),
        box.content.height,
    )
    u.component_text("panel_header", text, "Settings", TEXT, size=font, bold=True)
    u.hits["settings-title"] = text.tuple()
    frame_width, frame_height = control_frame_size(u, "settings-close", *close)
    close_y = box.content.y + max(0, box.content.height - close[1]) / 2
    pressed = u.button(
        "settings-close",
        box.content.right - close[0],
        close_y,
        frame_width,
        frame_height,
        icon="close",
        tooltip="Close",
        width_limit=frame_width * s,
    )
    u.line(box.frame.x, box.frame.bottom, box.frame.width)
    if pressed:
        panel.workspace.open["Settings"] = panel.open = False
    return pressed, box.outer.bottom + DESIGN.row_gap * s


def _navigation(panel, ctx, rect, narrow):
    from mojive.ui.workspace.settings_panel import CATEGORIES, CATEGORY_ICONS

    u, s = panel.workspace.ui, panel.workspace.s
    style = u.component_style("settings_nav")
    pad, margin = style.edges("padding", s), style.edges("margin", s)
    minimum = max(u.measure(ctx.tr(c)) for c in CATEGORIES) + 39 * s
    width = (
        rect.width
        if narrow
        else min(
            rect.width,
            rect.width / 2,
            u.metric("settings_layout", "navWidth", max(128, minimum / s)) * s,
        )
    )
    content_width = max(s, width - pad[0] - pad[2] - margin[0] - margin[2])
    columns = max(1, min(3, int(content_width / max(s, minimum)))) if narrow else 1
    gap = style.get("columnGap", style.get("gap", DESIGN.item_spacing_x)) * s
    cell = max(s, (content_width - (columns - 1) * gap) / columns)
    with u.component_scope("settings_nav"):
        sizes = [
            control_size(u, "settings-" + c, cell / s, DESIGN.control_height, label=c, icon=icon)
            for c, icon in zip(CATEGORIES, CATEGORY_ICONS, strict=True)
        ]
    box, flow = component_flow(
        u,
        "settings_nav",
        rect.x,
        rect.y,
        width,
        sizes,
        key="settings-navigation",
        column_gap=gap / s,
        row_gap=DESIGN.row_gap,
    )
    height = box.outer.height if narrow else rect.height
    # The navigation background occupies the same width as its rows.
    u.rect(
        box.frame.x,
        box.frame.y,
        box.frame.width,
        max(box.frame.height, height - margin[1] - margin[3]),
        PANEL,
        style.get("radius", DESIGN.container_radius),
    )
    imgui.set_cursor_screen_pos((box.content.x, box.content.y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.begin_child(
        "Reference settings navigation",
        (box.content.width, max(s, height - pad[1] - pad[3] - margin[1] - margin[3])),
    )
    imgui.pop_style_var()
    scroll_y = imgui.get_scroll_y()
    try:
        for (category, icon), item in zip(
            zip(CATEGORIES, CATEGORY_ICONS, strict=True), flow.rectangles, strict=True
        ):
            with u.component_scope("settings_nav"):
                frame_width, frame_height = control_frame_size(
                    u, "settings-" + category, item.width, item.height, label=category, icon=icon
                )
                if u.button(
                    "settings-" + category,
                    item.x,
                    item.y - scroll_y,
                    frame_width,
                    frame_height,
                    label=category,
                    icon=icon,
                    align="left",
                    active=panel._category == category,
                    width_limit=frame_width * s,
                ):
                    panel._category = category
    finally:
        imgui.end_child()
    return box.outer.width, height


def draw_settings(panel, ctx):
    from mojive.ui.workspace.settings_panel import _CHECKBOX_PROPERTIES

    u, s = panel.workspace.ui, panel.workspace.s
    p, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
    closed, top = _header(panel, Rect(p.x, p.y, available.x, available.y))
    if closed:
        return
    layout = u.component_style("settings_layout")
    height = max(available.y - (top - p.y), layout.get("minHeight", 0) * s)
    box = u.component_box(
        "settings_layout", p.x, top, available.x, height, key="settings-layout", observe=False
    )
    rect = box.content
    narrow = rect.width < 560 * s
    nav_width, nav_height = _navigation(panel, ctx, rect, narrow)
    gap = layout.get("rowGap" if narrow else "columnGap", layout.get("gap", DESIGN.label_gap)) * s
    page = Rect(
        rect.x if narrow else rect.x + nav_width + gap,
        rect.y + nav_height + gap if narrow else rect.y,
        rect.width if narrow else max(s, rect.width - nav_width - gap),
        max(s, rect.height - nav_height - gap) if narrow else rect.height,
    )
    u.record_component_box(
        "settings_layout",
        box,
        key="settings-layout",
        fields=[[rect.x, rect.y, nav_width, nav_height], list(page.tuple())],
    )
    page_box = u.component_box("settings_page", *page.tuple(), key="settings-page")
    imgui.set_cursor_screen_pos((page_box.content.x, page_box.content.y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.begin_child(
        "Reference settings content", (page_box.content.width, max(s, page_box.content.height))
    )
    imgui.pop_style_var()
    style = u.component_style("settings_page")
    font, title = style.get("fontSize", DESIGN.body_size), style.get("titleSize", DESIGN.title_size)
    line = max(title, style.line_height(title * 1.4)) * s
    position = imgui.get_cursor_screen_pos()
    title_rect = Rect(position.x, position.y, imgui.get_content_region_avail().x, line)
    u.component_text(
        "settings_page",
        title_rect,
        panel._category,
        TEXT,
        size=title,
        font_field="titleSize",
        bold=True,
    )
    u.line(title_rect.x, title_rect.bottom + 8 * s, title_rect.width)
    imgui.dummy((0, line + style.get("titleGap", 20) * s))
    panel._checkbox_labels = frozenset(ctx.tr(label) for label in _CHECKBOX_PROPERTIES)
    dimensions = replace(
        current_design(), body_size=font, row_gap=style.get("rowGap", DESIGN.row_gap)
    )
    try:
        with (
            use_design(lambda _: dimensions),
            u.component_scope("settings_page"),
            property_controls(u),
        ):
            panel._control_padding()
            {
                "General": panel._general,
                "Camera": panel._camera_navigation,
                "Interaction": panel._interaction,
                "Rendering": panel._rendering,
                "Recording": panel._recording,
                "MuJoCo Visuals": panel._mujoco_visuals,
            }[panel._category](ctx)
    finally:
        imgui.end_child()
