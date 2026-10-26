"""Complete icon inventory rendered through the production entry points."""

from dataclasses import dataclass
from functools import cache

from imgui_bundle import imgui as g

from mojive.ui.icons.model import ICON_FAMILIES
from mojive.ui.interaction_paint import ItemPaint, reference_paint
from mojive.ui.workspace.widget_icons import (
    GLYPHS,
    ICON_STROKE,
    PRODUCTION_GLYPHS,
)

LABELS = {
    "sun": "Light theme",
    "moon": "Dark theme",
    "geometry-default": "Default",
    "general": "General",
    "mujoco-visuals": "Content Visualization",
    "chart": "Statistics",
    "performance": "Statistics",
    "scene": "Scene hierarchy",
    "tree": "Scene hierarchy",
    "body": "Body",
    "body-frame": "Body frame",
    "geom": "Geometry",
    "joint-hinge": "Hinge joint",
    "joint-slide": "Slide joint",
    "joint-ball": "Ball joint",
    "joint-free": "Free joint",
    "stepb": "Previous frame",
    "stepf": "Next frame",
    "x": "Close",
    "eye": "Visible",
    "eyeoff": "Hidden",
    "gear": "Settings",
    "rec": "Record",
    "err": "Error",
    "warn": "Warning",
    "info": "Information",
    "env": "Environment",
    "shading": "Shading",
    "both": "Overlay",
    "visual": "Visual",
    "collision": "Collision",
    "cube": "Cube",
    "box": "Box",
    "magnet": "Snap",
    "diamond": "Keyframe",
    "key": "Keyframe",
    "fit": "Fit",
    "inspect": "Inspector",
    "sidebar-left": "Left panel",
    "chev": "Expand",
    "perturb": "Perturb",
    "target": "Target",
    "robot": "Robot",
    "sphere": "Sphere",
    "circle": "Circle",
    "right": "Expand",
    "down": "Expand",
    "sensor": "Sensors",
    "waveform": "Curves",
    "first": "First frame",
    "last": "Last frame",
    "follow-off": "Follow off",
    "follow-page": "Follow page",
    "follow-locked": "Follow locked",
    "view-front": "Front",
    "view-back": "Back",
    "view-left": "Left",
    "view-right": "Right",
    "view-top": "Top",
    "view-bottom": "Bottom",
    "view-iso": "Isometric",
    "shape-box": "Box",
    "shape-sphere": "Sphere",
    "shape-capsule": "Capsule",
    "shape-cylinder": "Cylinder",
    "shape-cone": "Cone",
    "shape-ellipsoid": "Ellipsoid",
    "shape-plane": "Plane",
    "capture-image": "Capture image",
}


@dataclass(frozen=True)
class IconEntry:
    name: str
    label: str
    family: str
    native: bool = False


@cache
def icon_catalogue():
    family_labels = {name: label for _family, entries in ICON_FAMILIES for label, name in entries}
    family_labels["tool-body"] = "Body frame"
    workspace = tuple(
        IconEntry(
            name,
            LABELS.get(
                name,
                family_labels.get(
                    name,
                    name.removeprefix("playback-")
                    .removeprefix("transport-")
                    .replace("-", " ")
                    .title(),
                ),
            ),
            "Workspace icons",
        )
        for name in sorted(GLYPHS)
    )
    native = tuple(
        IconEntry(
            name, "Body frame" if name == "tool-body" else label, family, name in PRODUCTION_GLYPHS
        )
        for family, entries in ICON_FAMILIES
        for label, name in entries
    )
    return workspace + native


def caption_x(center, measured_width, available_width):
    """Center the visible label box on the same axis as its glyph canvas."""
    return center - min(measured_width, available_width) / 2


def paint(v, entry, cx, cy, size, role="text"):
    r = v.rect(cx - size / 2, cy - size / 2, size, size)
    tint = v.ink(role)
    v.ui.icon(entry.name, r.x, r.y, size, tint, color_resolved=True)


def draw_icons(v):
    v.title(
        "Icon catalogue", "24-unit grid · Rounded strokes · Independent glyph and control sizes"
    )
    entries = icon_catalogue()
    families = ("All icons", "Workspace icons", *(family for family, _ in ICON_FAMILIES))
    _, family = v.viewer_select(
        "icon-family", 0, 96, 240, v.values.get("icon-family", "Workspace icons"), families
    )
    v.values["icon-family"] = family
    g.set_cursor_screen_pos((v.rect(256, 96, 0, 0).x, v.rect(0, 96, 0, 0).y))
    g.set_next_item_width(min(240, v.width - 256) * v.s)
    _, query = g.input_text_with_hint(
        "##icon-search", v.tr("Search icons"), v.values.get("icon-search", "")
    )
    v.values["icon-search"] = query
    chosen = v.values.get("icon-selected", "camera")
    candidates = tuple(
        entry for entry in entries if family == "All icons" or entry.family == family
    )
    selected = next((entry for entry in candidates if entry.name == chosen), candidates[0])
    chosen = v.values["icon-selected"] = selected.name
    v.box(0, 152, v.width, 144, "canvas", 12)
    v.text(16, 168, selected.label, size=16, bold=True)
    v.ui.text(
        *v.rect(16, 200, 0, 0).tuple()[:2],
        selected.name,
        v.ink("muted"),
        12,
        translate=False,
        font_resolved=True,
        color_resolved=True,
    )
    v.text(
        16, 232, v.tr("Canonical stroke: {stroke} / 24 units", stroke=ICON_STROKE), "tertiary", 12
    )
    for index, size in enumerate((14, 16, 20, 24, 32)):
        start = max(304, v.width - 352)
        step = min(64, (v.width - start - 32) / 4)
        cx = start + index * step
        paint(v, selected, cx, 194, size)
        v.text(cx - 20, 240, f"{size} pt", "tertiary", 12, width=40)
    v.text(0, 328, "Interaction states", size=16, bold=True)
    for index, (label, state) in enumerate(
        (
            ("Default", ItemPaint()),
            ("Hovered", ItemPaint(hovered=True)),
            ("Active", ItemPaint(active=True)),
            ("Disabled", ItemPaint(disabled=True)),
        )
    ):
        cx, cy = 256 + index * 96, 336
        with reference_paint(state):
            v.ui.design.primitives.button(
                "icon-state/" + label,
                v.rect(cx - 20, cy - 20, 40, 40),
                enabled=not state.disabled,
                parts="icon_button",
                filled=False,
            )
            paint(v, selected, cx, cy, 20, "tertiary" if state.disabled else "text")
        caption = v.tr(label)
        measured = v.ui.measure(caption, 12, translate=False, font_resolved=True) / v.s
        v.text(caption_x(cx, measured, 88), 368, caption, "muted", 12, width=88)
    filtered = tuple(
        entry
        for entry in entries
        if (family == "All icons" or entry.family == family)
        and (not query or query.casefold() in (entry.name + " " + v.tr(entry.label)).casefold())
    )
    columns = max(1, int(v.width // 128))
    cell_width = v.width / columns
    v.icon_cells = []
    for index, entry in enumerate(filtered):
        x, y = (index % columns) * cell_width, 408 + (index // columns) * 112
        rect = v.rect(x + 4, y, cell_width - 8, 100)
        g.set_cursor_screen_pos((rect.x, rect.y))
        g.invisible_button("icon/" + entry.family + "/" + entry.name, (rect.width, rect.height))
        hovered = g.is_item_hovered()
        if hovered or chosen == entry.name:
            v.box(
                x + 4, y, cell_width - 8, 100, "selected" if chosen == entry.name else "raised", 8
            )
        if g.is_item_clicked():
            v.values["icon-selected"] = entry.name
        if g.is_item_hovered(g.HoveredFlags_.delay_normal):
            v.ui.tooltip(v.tr(entry.label) + " · " + entry.name, translate=False)
        center = x + cell_width / 2
        paint(v, entry, center, y + 30, 24)
        label = v.tr(entry.label)
        measured = v.ui.measure(label, 12, translate=False, font_resolved=True) / v.s
        left = caption_x(center, measured, cell_width - 24)
        v.text(left, y + 64, label, size=12, width=cell_width - 24)
        v.hits["icon/" + entry.family + "/" + entry.name] = rect.tuple()
        v.icon_cells.append((entry.name, center, left, min(measured, cell_width - 24)))
    return 432 + ((len(filtered) + columns - 1) // columns) * 112
