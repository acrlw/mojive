"""Measure settings navigation and pages independently of preference actions."""

from dataclasses import replace
from math import ceil

from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.geometry import (
    Rect,
    classify_button,
    independent_corner_scope,
    part_scope,
)
from mojive.ui.panels.settings_pages.catalog import settings_category_label
from mojive.ui.workspace.component_layout import (
    control_frame_size,
    control_size,
    flow_layout,
)
from mojive.ui.workspace.corner_geometry import nested_corners
from mojive.ui.workspace.design import DESIGN, current_design, use_design
from mojive.ui.workspace.menu_geometry import native_box, native_edges, native_unit, row_radius
from mojive.ui.workspace.panel_geometry import corner_inset
from mojive.ui.workspace.property_style import property_controls
from mojive.ui.workspace.section_columns import section_column
from mojive.ui.workspace.widget_icons import icon_role_canvas
from mojive.ui.workspace.widgets import ACCENT, PANEL, TEXT
from mojive.ui.workspace.window_chrome import fit_native_content_width


def prepare_window(panel, *, opening=False):
    """Apply the settings window width without changing its manually chosen height."""
    ui, scale = panel.workspace.ui, panel.workspace.s
    viewport = imgui.get_main_viewport()
    window = imgui.internal.find_window_by_name("Settings")
    requested = ui.metric("settings_layout", "windowWidth", 0) * scale
    limit = max(scale, viewport.work_size.x - 32 * scale)
    automatic = min(820 * scale, limit)
    width = min(requested, limit) if requested else automatic
    height = min(620 * scale, max(scale, viewport.work_size.y - 64 * scale))
    if opening:
        imgui.set_next_window_pos(
            (
                viewport.work_pos.x + (viewport.work_size.x - width) / 2,
                viewport.work_pos.y + (viewport.work_size.y - height) / 2,
            ),
            imgui.Cond_.appearing,
        )
        imgui.set_next_window_size((width, height), imgui.Cond_.appearing)
    if requested:
        if panel._automatic_width is None:
            panel._automatic_width = window.size.x if window is not None else automatic
        imgui.set_next_window_size((width, window.size.y if window is not None else height))
    elif panel._automatic_width is not None:
        imgui.set_next_window_size(
            (
                min(panel._automatic_width, limit),
                window.size.y if window is not None else height,
            )
        )
        panel._automatic_width = None
    if requested and window is not None and not opening:
        x = max(
            viewport.work_pos.x,
            min(window.pos.x, viewport.work_pos.x + viewport.work_size.x - width),
        )
        if x != window.pos.x:
            imgui.set_next_window_pos((x, window.pos.y))


@part_scope("panel_header", inherit=False)
def _header(panel, rect):
    u, s = panel.workspace.ui, panel.workspace.s
    style = u.component_style("panel_header")
    font, icon = style.get("fontSize", DESIGN.title_size), style.get("iconSize", 20)
    pads, margin = style.edges("padding", s), style.edges("margin", s)
    close = control_size(u, "settings-close", 28, 28, icon="close")
    frame_width, frame_height = control_frame_size(u, "settings-close", *close)
    close_parts = (
        u.design.button_parts("settings-close", "")
        if u.design is not None
        else classify_button("settings-close", "")
    )
    close_margin = u.component_style(close_parts).edges("margin", s)
    host = imgui.internal.get_current_window()
    # Both the production panel and the Guide add transparent content children.
    # Their nearest painted ancestor owns the corner the close plate must follow.
    while host.parent_window is not None and host.flags & imgui.WindowFlags_.no_background:
        host = host.parent_window
    host_right = host.pos.x + host.size.x
    extra = native_unit(4 * s)
    requested = extra + max(
        rect.x - host.pos.x + margin[0] + pads[0],
        rect.y - host.pos.y + margin[1] + pads[1] + close_margin[1],
        host_right - rect.right + margin[2] + pads[2] + close_margin[2],
    )
    inset = corner_inset(
        host.window_rounding,
        requested,
        max(0, min(frame_width, frame_height) * s / 2 - 1),
    )
    radius = max(0, host.window_rounding - inset)
    row_y = host.pos.y + inset
    content_height = max(style.line_height(font * 1.4) * s, icon * s, frame_height * s)
    minimum = (
        row_y
        - rect.y
        - margin[1]
        + max(content_height, frame_height * s + close_margin[3])
        + extra
        + pads[3]
    )
    height = max(style.get("height", 32) * s, minimum) + margin[1] + margin[3]
    box = u.component_box(
        "panel_header",
        rect.x,
        rect.y,
        rect.width,
        height,
        minimum_height=minimum,
        observe=False,
    )
    content = Rect(
        host.pos.x + inset,
        row_y,
        max(0, host_right - host.pos.x - 2 * inset),
        content_height,
    )
    box = replace(
        box,
        content=content,
        padding=(
            content.x - box.frame.x,
            content.y - box.frame.y,
            box.frame.right - content.right,
            box.frame.bottom - content.bottom,
        ),
    )
    gap = style.get("columnGap", style.get("gap", 8)) * s
    icon_rect = style.align_rect(box.content, icon * s, icon * s, horizontal="start")
    u.icon("settings", icon_rect.x, icon_rect.y, icon, ACCENT)
    text = Rect(
        icon_rect.right + gap,
        box.content.y,
        max(0, box.content.right - icon_rect.right - gap - frame_width * s - close_margin[0]),
        box.content.height,
    )
    u.component_text("panel_header", text, "Settings", TEXT, size=font, bold=True)
    u.hits["settings-title"] = text.tuple()
    # Extra authored header height belongs below the action row; centering it
    # would make the top offset differ from the right offset again.
    with u.corner_scope(radius, close_parts):
        pressed = u.button(
            "settings-close",
            box.content.right - frame_width * s - close_margin[0],
            row_y - close_margin[1],
            frame_width,
            frame_height,
            icon="close",
            tooltip="Close",
            width_limit=frame_width * s,
        )
    u.record_component_box(
        "panel_header",
        box,
        key="settings-header",
        extraInset=extra,
        cornerInset=inset,
        parentRadius=host.window_rounding,
        parentRect=[*host.pos, *host.size],
        closeRadius=radius,
        closeRect=list(u.hits["settings-close"]),
        iconRect=list(icon_rect.tuple()),
    )
    u.line(box.frame.x, box.frame.bottom, box.frame.width)
    if pressed:
        panel.workspace.open["Settings"] = panel.open = False
    return pressed, box.outer.bottom + DESIGN.row_gap * s


@part_scope("settings_nav", inherit=False)
def draw_navigation(panel, ctx, rect, narrow, *, fit_content=False):
    from mojive.ui.workspace.settings_panel import CATEGORIES, CATEGORY_ICONS

    u, s = panel.workspace.ui, panel.workspace.s
    style = u.component_style("settings_nav")
    pad = (max(native_edges(style, "padding", s, (4, 4, 4, 4))),) * 4
    margin = native_edges(style, "margin", s)
    widths, row_heights = [], []
    for category in CATEGORIES:
        parts = (
            u.design.button_parts("settings-" + category, category)
            if u.design is not None
            else classify_button("settings-" + category, category)
        )
        item = u.component_style(parts)
        label = u.measure(
            ctx.tr(settings_category_label(category)),
            item.get("fontSize", DESIGN.body_size),
            translate=False,
            font_resolved=True,
            style=item,
        )
        icon = item.get("iconSize", icon_role_canvas("inline")) * s
        inner_gap = item.get("columnGap", item.get("gap", 6)) * s
        padding, margins = item.edges("padding", s, (8, 0, 8, 0)), item.edges("margin", s)
        natural = label + icon + inner_gap + padding[0] + padding[2]
        widths.append(item.dimension("width", natural / s) * s + margins[0] + margins[2])
        row_heights.append(
            ceil(
                control_size(
                    u,
                    "settings-" + category,
                    natural / s,
                    DESIGN.control_height,
                    label=settings_category_label(category),
                    icon=icon,
                )[1]
            )
        )
    # Native child frames round physical widths; keep the complete caption inside.
    minimum = ceil(max(widths)) + 1
    natural_width = ceil(minimum + pad[0] + pad[2] + margin[0] + margin[2])
    row_gap = native_unit(style.get("rowGap", style.get("gap", DESIGN.row_gap)) * s)
    rows_height = sum(row_heights) + max(0, len(row_heights) - 1) * row_gap
    if (
        not narrow
        and not fit_content
        and rows_height > rect.height - pad[1] - pad[3] - margin[1] - margin[3]
    ):
        natural_width += ceil(imgui.get_style().scrollbar_size)
    requested = style.get("navWidth", u.metric("settings_layout", "navWidth", 0)) * s
    natural = rect.width if narrow else max(128 * s, natural_width)
    gap = u.component_style("settings_layout").get("columnGap", DESIGN.label_gap) * s
    limit = rect.width if narrow else max(s, rect.width - gap - min(160 * s, rect.width / 2))
    width = min(limit, max(natural_width, requested or natural))
    box = native_box(
        u.component_box(
            "settings_nav",
            rect.x,
            rect.y,
            width,
            rect.height,
            key="settings-navigation",
            padding=pad,
            observe=False,
        )
    )
    box = native_box(replace(box, content=box.frame.inset(*pad), padding=pad))
    gap = native_unit(style.get("columnGap", style.get("gap", DESIGN.item_spacing_x)) * s)
    row_gap = native_unit(style.get("rowGap", style.get("gap", DESIGN.row_gap)) * s)
    columns = (
        max(1, min(3, int((box.content.width + gap) / max(s, minimum + gap)))) if narrow else 1
    )

    def content_flow(available):
        cell = max(0, (available - (columns - 1) * gap) / columns)
        with u.component_scope("settings_nav"):
            sizes = [
                control_size(
                    u,
                    "settings-" + c,
                    cell / s,
                    DESIGN.control_height,
                    label=settings_category_label(c),
                    icon=icon,
                )
                for c, icon in zip(CATEGORIES, CATEGORY_ICONS, strict=True)
            ]
        sizes = [(min(cell, width), ceil(height)) for width, height in sizes]
        return flow_layout(
            Rect(box.content.x, box.content.y, available, 0), sizes, column_gap=gap, row_gap=row_gap
        )

    flow = content_flow(box.content.width)
    if narrow or fit_content:
        height = flow.height + sum(box.padding[1::2]) + sum(box.margin[1::2])
        box = native_box(
            u.component_box(
                "settings_nav",
                rect.x,
                rect.y,
                width,
                height,
                key="settings-navigation",
                padding=pad,
                observe=False,
            )
        )
        box = native_box(replace(box, content=box.frame.inset(*pad), padding=pad))
    height = box.outer.height
    first = flow.rectangles[0]
    with independent_corner_scope():
        corners = nested_corners(
            style,
            s,
            box.padding,
            row_radius(u.component_style("settings_nav_button"), s, first.width, first.height),
        )
    radius = corners.radius
    u.rect(*box.frame.tuple(), PANEL, radius / s)
    imgui.set_cursor_screen_pos((box.content.x, box.content.y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.begin_child(
        "Reference settings navigation",
        (max(s, box.content.width), max(s, box.content.height)),
        window_flags=imgui.WindowFlags_.no_background,
    )
    imgui.pop_style_var()
    own_right = fit_native_content_width(imgui.internal.get_current_window())
    # Absolute button positions need the native child budget as well as native queries.
    flow = content_flow(max(0, min(box.content.right, own_right) - box.content.x))
    u.record_component_box(
        "settings_nav",
        box,
        key="settings-navigation",
        fields=[list(item.tuple()) for item in flow.rectangles],
        fieldRegions={"navWidth": list(box.outer.tuple())},
        radius=radius,
        innerRadius=corners.inner_radius,
        parentCornerConstrained=corners.parent_constrained,
        cornerOffsetDegenerate=radius < box.padding[0],
        nativePaintableRight=own_right,
    )
    scroll_y = imgui.get_scroll_y()
    try:
        for (category, icon), item in zip(
            zip(CATEGORIES, CATEGORY_ICONS, strict=True), flow.rectangles, strict=True
        ):
            with (
                u.component_scope("settings_nav"),
                u.corner_scope(
                    corners.inner_radius, ("button", "icon_button", "settings_nav_button")
                ),
            ):
                frame_width, frame_height = control_frame_size(
                    u,
                    "settings-" + category,
                    item.width,
                    item.height,
                    label=settings_category_label(category),
                    icon=icon,
                )
                if u.button(
                    "settings-" + category,
                    item.x,
                    item.y - scroll_y,
                    frame_width,
                    frame_height,
                    label=settings_category_label(category),
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
    nav_width, nav_height = draw_navigation(panel, ctx, rect, narrow)
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
        windowRect=[*imgui.get_window_pos(), *imgui.get_window_size()],
        fieldRegions={
            "navWidth": [rect.x, rect.y, nav_width, nav_height],
            "windowWidth": [*imgui.get_window_pos(), *imgui.get_window_size()],
        },
    )
    with group_scope("Settings/" + panel._category):
        _draw_page(panel, ctx, page, _CHECKBOX_PROPERTIES)


def _draw_page(panel, ctx, page, checkbox_properties):
    u, s = panel.workspace.ui, panel.workspace.s
    style = u.component_style("settings_page")
    # Form fields stay close to their labels when the window is expanded.
    # An authored fixed page width continues to own the layout.
    if not style.explicit("width"):
        page = Rect(page.x, page.y, min(page.width, 640 * s), page.height)
    page_box = u.component_box("settings_page", *page.tuple(), key="settings-page")
    imgui.set_cursor_screen_pos((page_box.content.x, page_box.content.y))
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    imgui.begin_child(
        "Reference settings content",
        (page_box.content.width, max(s, page_box.content.height)),
        window_flags=imgui.WindowFlags_.no_background,
    )
    imgui.pop_style_var()
    fit_native_content_width(imgui.internal.get_current_window())
    try:
        with section_column(u):
            _draw_page_contents(panel, ctx, style, checkbox_properties)
    finally:
        imgui.end_child()


def _draw_page_contents(panel, ctx, style, checkbox_properties):
    u, s = panel.workspace.ui, panel.workspace.s
    font, title = style.get("fontSize", DESIGN.body_size), style.get("titleSize", DESIGN.title_size)
    line = max(title, style.line_height(title * 1.4)) * s
    position = imgui.get_cursor_screen_pos()
    title_rect = Rect(position.x, position.y, imgui.get_content_region_avail().x, line)
    u.component_text(
        "settings_page",
        title_rect,
        settings_category_label(panel._category),
        TEXT,
        size=title,
        font_field="titleSize",
        bold=True,
    )
    imgui.dummy((0, line + style.get("titleGap", 20) * s))
    panel._page_content_top = imgui.get_cursor_pos_y()
    panel._checkbox_labels = frozenset(ctx.tr(label) for label in checkbox_properties)
    dimensions = replace(
        current_design(), body_size=font, row_gap=style.get("rowGap", DESIGN.row_gap)
    )
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
