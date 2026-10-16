"""Measure native material shortcuts without owning model resource actions."""

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.property_style import advance, property_section
from mojive.ui.workspace.widgets import ACCENT, INPUT, TEXT, color

PALETTE = (
    ("Mist blue ceramic", "#8daecc"),
    ("Amber satin", "#e3bc67"),
    ("Slate matte", "#829cb7"),
    ("Coral polymer", "#d58979"),
)


def draw_material_library(workspace, x, y, width):
    u, s = workspace.ui, workspace.s
    imgui.set_cursor_screen_pos((x, y))
    if not property_section(u, "MATERIAL LIBRARY", imgui.TreeNodeFlags_.default_open):
        return imgui.get_cursor_screen_pos().y
    y = imgui.get_cursor_screen_pos().y
    style, listing = u.component_style("material"), u.component_style("material_list")
    pad, margin = style.edges("padding", s), style.edges("margin", s)
    font, swatch, icon = (
        style.get("fontSize", DESIGN.body_size),
        style.get("swatchSize", 32),
        style.get("iconSize", 16),
    )
    gap = style.get("columnGap", style.get("gap", 12)) * s
    height = max(
        style.get("height", 48) * s,
        max(swatch, icon, style.line_height(font * 1.4)) * s + pad[1] + pad[3],
    )
    spacing = listing.get("rowGap", listing.get("gap", DESIGN.row_gap)) * s
    top = y
    for index, (name, hexcolor) in enumerate(PALETTE):
        box = u.component_box(
            "material",
            x,
            y,
            width,
            height + margin[1] + margin[3],
            key="material-card-" + str(index),
            minimum_height=height,
            observe=False,
        )
        imgui.set_cursor_screen_pos((box.frame.x, box.frame.y))
        if imgui.invisible_button("material-" + name, (box.frame.width, box.frame.height)):
            workspace.material = index
        u.hits["material-" + name] = box.frame.tuple()
        u.rect(*box.frame.tuple(), INPUT, style.get("radius", DESIGN.control_radius))
        content = box.content
        swatch_rect = style.align_rect(
            content,
            min(swatch * s, content.width),
            min(swatch * s, content.height),
            horizontal="start",
        )
        channels = tuple(int(hexcolor[i : i + 2], 16) for i in (1, 3, 5))
        u.draw.add_circle_filled(
            (swatch_rect.x + swatch_rect.width / 2, swatch_rect.y + swatch_rect.height / 2),
            min(swatch_rect.width, swatch_rect.height) / 2,
            color((*channels, 255)),
        )
        caption = Rect(
            swatch_rect.right + gap,
            content.y,
            max(0, content.right - swatch_rect.right - gap - icon * s - gap),
            content.height,
        )
        u.record_component_box(
            "material",
            box,
            key="material-card-" + str(index),
            selected=workspace.material == index,
            fields=[list(swatch_rect.tuple()), list(caption.tuple())],
        )
        u.component_text("material", caption, name, TEXT, size=font)
        if workspace.material == index:
            mark = style.align_rect(content, icon * s, icon * s, horizontal="end")
            u.icon("check", mark.x, mark.y, icon, ACCENT)
        y += box.outer.height + spacing
    if u.design is not None:
        u.design.record(
            "material-library",
            ("material_list",),
            (x, top, width, y - top - spacing),
            "material_list",
        )
    with u.component_scope("material"):
        if u.button(
            "apply-material",
            x,
            y,
            width / s,
            DESIGN.control_height,
            icon="check",
            label="Apply material",
            align="center",
            active=True,
            solid=True,
            enabled=workspace.app.entity is not None and workspace.app.entity["colorable"],
        ):
            workspace.app.entity["color"] = PALETTE[workspace.material][1]
            workspace.app.apply_document_edits()
    y = u.hits["apply-material"][1] + u.hits["apply-material"][3] + DESIGN.row_gap * s
    advance(u, x, y)
    return y
