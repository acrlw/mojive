"""Measured inspector header and tabs; scene selection remains owned by Session."""

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import DESIGN, TITLE_SIZE
from mojive.ui.workspace.widget_style import ACCENT, MUTED, TEXT


def draw_heading(workspace, entity, x, y, width):
    bottom = _draw_heading(workspace, entity, x, y, width)
    return bottom if entity is None else draw_tabs(workspace, x, bottom, width)


@part_scope("inspector_header", inherit=False)
def _draw_heading(workspace, entity, x, y, width):
    ui, s = workspace.ui, workspace.s
    style = ui.component_style("inspector_header")
    font = style.get("fontSize", TITLE_SIZE)
    padding = style.edges("padding", s)
    margin = style.edges("margin", s)
    glyph = style.get("iconSize", 20) * s
    line_height = max(font, style.line_height(font * 1.4)) * s
    action_sizes = (
        control_size(ui, "pin", 26, 28, icon="pin", icon_role="subtle"),
        control_size(ui, "inspector-close", 26, 28, icon="close", icon_role="subtle"),
    )
    height = max(
        style.get("minHeight", 30) * s,
        max(line_height, glyph, *(size[1] for size in action_sizes)) + padding[1] + padding[3],
    )
    box = ui.component_box(
        "inspector_header",
        x,
        y,
        width,
        height + margin[1] + margin[3],
        key="inspector-header-frame",
        observe=False,
    )
    content = box.content
    gap = style.get("columnGap", style.get("gap", 8)) * s
    action_width = sum(size[0] for size in action_sizes)
    width_factor = min(1, max(0, content.width - gap) / action_width) if action_width else 1
    pin_width, close_width = (size[0] * width_factor for size in action_sizes)
    title_width = max(0, content.width - pin_width - close_width - gap)
    imgui.set_cursor_screen_pos((content.x, content.y))
    imgui.invisible_button("inspector-header", (max(s, title_width), max(s, content.height)))
    ui.hits["inspector-header"] = (
        content.x,
        content.y,
        max(s, title_width),
        max(s, content.height),
    )
    hovered = imgui.is_item_hovered()
    if imgui.is_item_active() and imgui.is_mouse_dragging(0):
        window = imgui.get_current_context().current_window
        if window.dock_node is not None:
            imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), window)
        else:
            imgui.internal.start_mouse_moving_window(window)
    icon_rect = style.align_rect(content, glyph, glyph, horizontal="start")
    ui.icon(
        entity["icon"] if entity else "inspect",
        icon_rect.x,
        icon_rect.y,
        glyph / s,
        entity["icon_color"] if entity else MUTED,
    )
    title = entity["name"] if entity else "Inspector"
    text_rect = Rect(
        content.x + glyph + gap, content.y, max(0, title_width - glyph - gap), content.height
    )
    ui.component_text(
        "inspector_header",
        text_rect,
        title,
        TEXT,
        size=TITLE_SIZE,
        bold=True,
        translate=entity is None,
    )
    if (
        hovered
        and entity
        and ui.measure(title, font, translate=False, font_resolved=True) > text_rect.width
    ):
        ui.tooltip(title, translate=False)
    vertical = style.get("alignItems", "center")
    factor = 0 if vertical in ("start", "top") else 1 if vertical in ("end", "bottom") else 0.5
    pin_y = content.y + max(0, content.height - action_sizes[0][1]) * factor
    close_y = content.y + max(0, content.height - action_sizes[1][1]) * factor
    pin_frame = control_frame_size(ui, "pin", pin_width, action_sizes[0][1], icon="pin")
    close_frame = control_frame_size(
        ui, "inspector-close", close_width, action_sizes[1][1], icon="close"
    )
    if ui.button(
        "pin",
        content.right - close_width - gap - pin_width,
        pin_y,
        *pin_frame,
        icon="pin",
        icon_role="subtle",
        width_limit=pin_frame[0] * s,
        active=bool(workspace.pinned),
        enabled=entity is not None,
        tooltip="Pin Inspector",
    ):
        workspace.pinned = None if workspace.pinned else entity["identity"]
    if ui.button(
        "inspector-close",
        content.right - close_width,
        close_y,
        *close_frame,
        icon="close",
        icon_role="subtle",
        width_limit=close_frame[0] * s,
        tooltip="Close",
    ):
        workspace.open["Inspector"] = False
    ui.record_component_box(
        "inspector_header",
        box,
        key="inspector-header-frame",
        iconRect=list(icon_rect.tuple()),
        textRect=list(text_rect.tuple()),
        fields=[
            list(icon_rect.tuple()),
            list(text_rect.tuple()),
            ui.hits["pin"],
            ui.hits["inspector-close"],
        ],
    )
    if entity is None:
        return box.outer.bottom + 14 * s
    return box.outer.bottom + 8 * s


@part_scope("tabs", inherit=False)
def draw_tabs(workspace, x, y, width):
    ui, s = workspace.ui, workspace.s
    style = ui.component_style("tabs")
    tab_style = ui.component_style(("button", "tab"))
    labels = ("Properties", "Physics", "Material", "Joint")
    icons = ("inspect", "body", "shading", "joint")
    font = tab_style.get("fontSize", DESIGN.body_size)
    gap = style.get("columnGap", style.get("gap", 0)) * s
    pads = style.edges("padding", s)
    margins = style.edges("margin", s)
    height = tab_style.box(
        0, 0, 1, 32 * s, scale=s, intrinsic=(0, tab_style.line_height(font * 1.4) * s)
    ).outer.height
    box = style.box(
        x, y, max(s, width - margins[0] - margins[2]), height + pads[1] + pads[3], scale=s
    )
    content = box.content
    minima = [ui.measure(label, font, font_resolved=True) + 12 * s for label in labels]
    compact = content.width < sum(minima) + gap * 3
    if compact:
        widths = [max(s, (content.width - 3 * gap) / 4)] * 4
    elif style.explicit("align") or tab_style.explicit("width"):
        widths = [tab_style.box(0, 0, w, height, scale=s).outer.width for w in minima]
    else:
        extra = max(0, content.width - sum(minima) - gap * 3) / 4
        widths = [w + extra for w in minima]
    from mojive.ui.workspace.component_layout import component_flow, control_frame_size

    tab_margin = tab_style.edges("margin", s)
    if tab_style.explicit("width") and tab_style.get("widthMode", "auto") == "fixed":
        widths = [
            min(content.width, tab_style.get("width", 70) * s + tab_margin[0] + tab_margin[2])
        ] * 4
    available = max(s, content.width)
    widths = [min(available, w) for w in widths]
    box, flow = component_flow(
        ui,
        "tabs",
        x,
        y,
        width,
        [(w, height) for w in widths],
        key="inspector-tabs",
        column_gap=gap / s,
        row_gap=gap / s,
    )
    positions = flow.rectangles
    for label, icon, rect in zip(labels, icons, positions, strict=True):
        with ui.component_scope("tabs"):
            if ui.button(
                "tab-" + label,
                rect.x,
                rect.y,
                *control_frame_size(
                    ui,
                    "tab-" + label,
                    rect.width,
                    rect.height,
                    label="" if compact else label,
                    icon=icon if compact else None,
                ),
                label="" if compact else label,
                icon=icon if compact else None,
                tooltip=label if compact else "",
                align="center",
                text_color=ACCENT if workspace.inspector_tab == label else MUTED,
                icon_color=ACCENT if workspace.inspector_tab == label else MUTED,
            ):
                workspace.inspector_tab = label
        if workspace.inspector_tab == label:
            ui.draw.add_line(
                (rect.x + 8 * s, rect.bottom),
                (rect.right - 8 * s, rect.bottom),
                imgui.get_color_u32(tuple(c / 255 for c in ui.ink(ACCENT))),
                2 * s,
            )
    return box.outer.bottom + 12 * s
