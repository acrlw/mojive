"""Actual-size typography, palette and spacing pages for the native guide."""

from imgui_bundle import imgui as g

from .guide_system import (
    GRAY,
    GROUPS,
    REFERENCES,
    SAGE,
    SEMANTIC,
    SPACING,
    TYPE_ROLES,
    VIEWER_LAYOUT,
    contrast,
    hex_ink,
    hsl_caption,
)


def draw_foundation(guide):
    return {
        "Overview": overview,
        "Typography": typography,
        "Palette": palette,
        "Spacing": spacing,
        "Icons": icons,
        "Controls": controls,
        "Corners": corners,
        "Viewer layout": viewer_layout,
        "Group laboratory": laboratory,
    }[guide.page](guide)


def overview(v):
    v.title(
        "Interface specification",
        "Typography, color, geometry and interactive component specimens",
    )
    v.box(0, 88, v.width, 144)
    v.text(24, 112, "Information hierarchy", size=20, bold=True)
    v.text(
        24,
        152,
        "Proximity, type weight and surface contrast define relationships between controls.",
        "muted",
        width=v.width - 48,
    )
    v.text(
        24,
        184,
        "Accent colors identify selection and actions. The hierarchy remains readable in grayscale.",
        "muted",
        width=v.width - 48,
    )
    for index, (title, detail, value) in enumerate(
        (
            ("Typography", "Regular, Semibold and numeric faces", "12 / 14 / 16 / 20 / 24"),
            ("Spacing", "Control padding and group spacing", "4 / 8 / 12 / 16 / 24"),
            ("Ownership", "Shared defaults with group overrides", "Panel > Group > Control"),
        )
    ):
        y = 264 + index * 104
        v.text(0, y, title, size=16, bold=True)
        v.text(0, y + 32, detail, "muted")
        v.text(min(420, v.width * 0.46), y + 32, value, "action", width=v.width * 0.54)
        v.line(0, y + 76, v.width)
    v.text(0, 608, "Interactive specimens", size=16, bold=True)
    v.text(
        0,
        644,
        "Panel pages contain local interactions. Group laboratory edits the selected component owner.",
        "muted",
        width=v.width,
    )
    v.text(
        0,
        672,
        "Export draft writes a valid Mojive specification; it does not replace your current design.",
        "muted",
        width=v.width,
    )
    reference_lines(v, 0, 736)
    return 760 + len(REFERENCES) * 48


def typography(v):
    v.title(
        "Typography specification",
        "Logical points · Same production atlas and Chinese fallback as the application",
    )
    v.text(0, 96, "ROLE / SIZE / LINE BOX", "tertiary", 12)
    v.text(300, 96, "ACTUAL-SIZE SPECIMEN", "tertiary", 12)
    for index, (role, size, weight, line, sample) in enumerate(TYPE_ROLES):
        y = 136 + index * 80
        v.text(0, y, role, bold=True)
        v.text(
            0,
            y + 28,
            v.tr("{size} pt / {line} pt · {weight}", size=size, line=line, weight=v.tr(weight)),
            "muted",
            12,
        )
        v.text(
            300,
            y + 8,
            sample,
            size=size,
            bold=weight == "Emphasis",
            mono=weight == "Numeric",
            width=v.width - 300,
        )
        v.line(0, y + 64, v.width)
    v.box(0, 644, v.width, 144)
    v.text(16, 660, "Installed font faces", bold=True)
    for index, (source, _) in enumerate(v.ui.font_sources):
        v.text(
            16,
            696 + index * 24,
            v.tr(
                ("Regular: {face}", "Emphasis: {face}", "Numeric: {face}")[index], face=source.label
            ),
            "muted",
            12,
        )
    v.text(
        0,
        816,
        "MiSans is used under Xiaomi's free commercial-use license; original font files and attribution are included.",
        "muted",
        12,
        width=v.width,
    )
    return 864


def palette(v):
    v.title(
        "Color specification",
        "Active specification colors and exported hexadecimal values",
    )
    y = 100
    for heading, ramp in (
        ("Neutral surfaces and text", GRAY),
        ("Sage / action and selection", SAGE),
        ("Semantic colors / icons and contextual labels", SEMANTIC),
    ):
        v.text(0, y, heading, size=16, bold=True)
        y += 40
        cell = min(148, v.width / len(ramp))
        for index, (name, _) in enumerate(ramp):
            x, ink = index * cell, v.ink(name)
            v.ui.rect(*v.rect(x, y, cell - 8, 64).tuple(), v.ink(name), 4)
            v.text(x, y + 76, name, size=12, bold=True)
            v.text(x, y + 98, hex_ink(ink), "muted", 12)
        y += 144
    v.text(0, y, "Measured foreground / surface pairs", size=16, bold=True)
    y += 40
    for foreground, background in (
        ("text", "panel"),
        ("muted", "panel"),
        ("tertiary", "field"),
        ("canvas", "action"),
        ("border", "field"),
    ):
        first, second = v.ink(foreground), v.ink(background)
        v.box(0, y - 4, 220, 32, background, 4)
        v.text(12, y + 2, foreground + " / " + background, foreground, 12)
        v.text(248, y + 2, f"{contrast(first, second):.2f}:1", "muted", 12, mono=True)
        y += 44
    v.text(
        0,
        y + 8,
        v.tr(
            "{hsl} · Decorative dividers and functional control outlines are separate roles.",
            hsl=hsl_caption(v.ink("action")),
        ),
        "muted",
        12,
        width=v.width,
    )
    return y + 64


def spacing(v):
    v.title(
        "Spacing specification",
        "Control padding, row spacing and separation between component groups",
    )
    for index, value in enumerate(SPACING):
        y = 100 + index * 40
        v.text(0, y, f"{value:2d} pt", "muted", 12, mono=True)
        v.box(80, y, value * 3, 16, "action", 4)
    x = min(400, v.width * 0.48)
    v.text(x, 100, "Shared defaults", size=16, bold=True)
    from mojive.ui.workspace.design import DESIGN

    rows = (
        ("Control height", f"{DESIGN.control_height:g} pt"),
        ("Body line box", f"{DESIGN.body_size * 1.4:g} pt"),
        ("Inline icon", "16 pt"),
        ("Activity icon / slot", "24 / 48 pt"),
        ("Label to field", f"{DESIGN.label_gap:g} pt"),
        ("Rows within a group", f"{DESIGN.row_gap:g} pt"),
        ("Between groups", f"{DESIGN.data_row_gap:g} pt"),
        ("Panel content inset", f"{DESIGN.panel_padding_x:g} / {DESIGN.panel_padding_y:g} pt"),
    )
    for index, (label, value) in enumerate(rows):
        v.text(x, 144 + index * 32, label, "muted", 12)
        v.text(x + 210, 144 + index * 32, value, size=12, mono=True)
    v.box(0, 468, v.width, 216)
    v.text(16, 488, "Rendering / options", bold=True)
    v.text(16, 520, "Each checkbox stays next to its own label.", "muted")
    for index, label in enumerate(("Selection outline", "Tone mapping", "Multisampling")):
        r = v.rect(16, 560 + index * 32, v.width - 32, 24)
        changed, value = v.ui.checkbox(
            "spacing-" + label, r.x, r.y, v.values.get(label, True), label=label, width=r.width
        )
        if changed:
            v.values[label] = value
    v.text(
        0,
        716,
        "A group owns gaps between controls. A control owns padding inside its frame.",
        "muted",
        width=v.width,
    )
    return 768


def laboratory(v):
    from .guide_panels import draw_group_demo

    v.title(
        "Group appearance",
        "Width, height, padding, family, ink and icon size resolve against a concrete production owner",
    )
    _, v.group = v.viewer_select("group-selector", 0, 92, min(320, v.width), v.group, tuple(GROUPS))
    editor_width = 320
    preview_width = v.width - editor_width - 24
    region_height = max(
        240, g.get_window_height() / v.s - 148 - 2 * v.viewer_layout["Content panes"]["paddingY"]
    )
    with v.region("Group comparison", 0, 148, preview_width, region_height) as width:
        v.text(0, 0, "EDITED GROUP", "tertiary", 12)
        bottom = draw_group_demo(v, v.group, 0, 32, width)
        y = bottom + 32
        v.line(0, y, width)
        v.text(0, y + 24, "UNCHANGED NEIGHBOR", "tertiary", 12)
        neighbor = "Scene tree" if v.group != "Scene tree" else "Settings navigation"
        bottom = draw_group_demo(v, neighbor, 0, y + 56, width)
        g.set_cursor_screen_pos((v.origin[0], v.origin[1] + (bottom + 16) * v.s))
        g.dummy((width * v.s, 1))
    with v.region(
        "Group appearance", preview_width + 24, 148, editor_width, region_height
    ) as width:
        end = v.group_editor(0, 0, width)
        g.set_cursor_screen_pos((v.origin[0], v.origin[1] + end * v.s))
        g.dummy((width * v.s, 1))
    return 148 + region_height


def viewer_layout(v):
    from copy import deepcopy

    v.title(
        "Guide layout settings",
        "Content padding and spacing for each guide component group",
    )
    _, v.viewer_group = v.viewer_select(
        "viewer-group", 0, 92, 320, v.viewer_group, tuple(VIEWER_LAYOUT)
    )
    group = v.viewer_group
    values = v.viewer_layout[group]
    labels = {
        "paddingLeft": "Left content inset",
        "paddingRight": "Right content inset",
        "paddingTop": "Top content inset",
        "paddingBottom": "Bottom content inset",
        "paddingX": "Horizontal content inset",
        "paddingY": "Vertical content inset",
        "rowGap": "Gap between buttons",
    }
    v.text(
        0,
        148,
        "These settings change the guide itself. Mojive sample groups stay independent.",
        "muted",
        width=v.width,
    )
    y = 200
    row_height = g.get_frame_height() / v.s + 16
    for field, value in values.items():
        v.text(0, y + 9, labels[field], "muted", 14)
        g.set_cursor_screen_pos((v.origin[0] + 208 * v.s, v.origin[1] + y * v.s))
        g.set_next_item_width(min(240, v.width - 224) * v.s)
        limit = (
            16
            if field in ("paddingTop", "paddingBottom", "rowGap")
            or (group == "Parameter fields" and field == "paddingY")
            else 32
        )
        changed, out = g.slider_int("##viewer-" + field, value, 0, limit)
        v.record("viewer/" + field)
        if changed:
            values[field] = out
        y += row_height
    v.text(0, y + 24, "LIVE CONTENT INSETS", "tertiary", 12)
    y += 64
    if group in ("Navigation buttons", "Header actions", "Parameter buttons"):
        v.button(
            "viewer-sample",
            0,
            y,
            256,
            "Group laboratory",
            active=True,
            align="start",
            viewer_group=group,
        )
        frame = v.hits["viewer-sample"]
        content = v.hits["viewer-sample/content"]
        v.text(
            0,
            y + 80,
            v.tr(
                "Content starts {inset} pt inside the button.",
                inset=round((content[0] - frame[0]) / v.s),
            ),
            "muted",
            width=v.width,
        )
        y += 128
    elif group == "Parameter fields":
        g.set_cursor_screen_pos((v.origin[0], v.origin[1] + y * v.s))
        g.set_next_item_width(320 * v.s)
        _, v.values["viewer-entry"] = g.input_text(
            "##viewer-entry", v.values.get("viewer-entry", v.tr("A padded text field"))
        )
        y += g.get_frame_height() / v.s + 40
    else:
        g.push_style_color(g.Col_.child_bg, tuple(c / 255 for c in v.ink("field")))
        with v.region("Content inset sample", 0, y, min(480, v.width), 120):
            v.text(0, 0, "Content padding", bold=True)
            v.text(0, 32, "The pane owns its horizontal and vertical inset.", "muted", 12)
        g.pop_style_color()
        y += 160
    if v.button(
        "reset-viewer", 0, y, 192, "Reset this viewer group", viewer_group="Parameter buttons"
    )[0]:
        v.viewer_layout[group] = deepcopy(VIEWER_LAYOUT[group])
    v.text(
        0,
        y + 72,
        "Export draft also saves viewer-layout.json beside the Mojive specification.",
        "muted",
        12,
        width=v.width,
    )
    return y + 112


def icons(v):
    from .guide_icons import draw_icons

    return draw_icons(v)


def reference_lines(v, x, y):
    for index, (title, url) in enumerate(REFERENCES):
        v.text(x, y + index * 48, title, "muted", 12)
        v.text(x, y + index * 48 + 20, url, "tertiary", 12, width=v.width - x)


def controls(v):
    from mojive.ui.interaction_paint import ItemPaint, reference_paint

    from .guide_panels import draw_group_demo, switches

    v.title(
        "Control specification",
        "Shared geometry and interaction states for native controls",
    )
    v.text(0, 104, "Switch / immediate setting", size=16, bold=True)
    v.values.setdefault("guide-switch-1", False)
    switches(v, ("Visible", "Transform handles"), 0, 144, min(v.width, 400), prefix="guide-switch-")
    if v.width >= 720:
        for index, (label, state) in enumerate(
            (
                ("Default", ItemPaint()),
                ("Hovered", ItemPaint(hovered=True)),
                ("Active", ItemPaint(active=True)),
                ("Disabled", ItemPaint(disabled=True)),
            )
        ):
            x = 416 + index * 80
            v.text(x, 116, label, "tertiary", 12, width=76)
            for row, value in enumerate((True, False)):
                with reference_paint(state):
                    v.ui.design.primitives.switch(
                        "switch-state/" + label + str(row),
                        v.rect(x, 144 + row * 48, 64, 36),
                        value,
                        enabled=not state.disabled,
                    )
    v.text(0, 256, "Checkbox / independent option", size=16, bold=True)
    for index, label in enumerate(("Selection outline", "Tone mapping")):
        key = "guide-check-" + str(index)
        rect = v.rect(0, 296 + index * 48, min(v.width, 400), 28)
        result = v.ui.design.primitives.checkbox(
            key, rect.x, rect.y, v.values.get(key, True), label=label, width=rect.width
        )
        v.values[key] = result.value
    v.text(0, 408, "Select / one choice", size=16, bold=True)
    _, v.values["control-choice"] = v.viewer_select(
        "control-select",
        0,
        448,
        min(v.width, 320),
        v.values.get("control-choice", "Perspective"),
        ("Perspective", "Orthographic"),
    )
    v.text(0, 528, "Filter / capsule tag", size=16, bold=True)
    end = draw_group_demo(v, "Scene filters", 0, 568, v.width)
    v.text(
        0,
        end + 32,
        "Switches change a setting. Filters show which content is included.",
        "muted",
        width=v.width,
    )
    return end + 96


def corners(v):
    v.title(
        "Corner geometry",
        "G3 curves, inset relationships and capsule end spacing reuse the feasibility geometry",
    )
    radius = v.values.get("corner-radius", 24)
    inset = v.values.get("corner-inset", 8)
    for index, (key, label, value, low, high) in enumerate(
        (
            ("corner-radius", "Outer radius", radius, 12, 40),
            ("corner-inset", "Content inset", inset, 4, 20),
        )
    ):
        y = 104 + index * 48
        v.text(0, y + 8, label, "muted")
        g.set_cursor_screen_pos((v.origin[0] + 176 * v.s, v.origin[1] + y * v.s))
        g.set_next_item_width(min(240, v.width - 176) * v.s)
        _, v.values[key] = g.slider_int("##" + key, value, low, high)
    radius, inset = v.values.get("corner-radius", radius), v.values.get("corner-inset", inset)
    inner = max(0, radius - inset)
    frame_width = min(v.width, 560)
    v.box(0, 248, frame_width, 208, "raised", radius)
    v.box(inset, 248 + inset, frame_width - inset * 2, 208 - inset * 2, "panel", inner)
    v.text(
        inset + 24,
        288,
        "Nested corner geometry",
        size=16,
        bold=True,
        width=frame_width - inset * 2 - 48,
    )
    v.text(
        inset + 24,
        332,
        v.tr(
            "Outer {outer} pt · inset {inset} pt · inner {inner} pt",
            outer=radius,
            inset=inset,
            inner=inner,
        ),
        "muted",
        width=frame_width - inset * 2 - 48,
    )
    v.text(
        0,
        496,
        "Inner radius = max(0, outer radius - inset)",
        size=16,
        bold=True,
        width=v.width,
    )
    v.text(
        0,
        536,
        "G3 profiles are reviewed at their rendered size, including the nested edges.",
        "muted",
        width=v.width,
    )
    v.box(0, 600, min(320, v.width), 48, "field", 24)
    v.text(24, 616, "Capsule / G3 smoothing 0.382", size=14)
    v.text(
        0,
        688,
        "Rounded rectangle smoothing: 0.618 · Capsule smoothing: 0.382",
        "muted",
        12,
        width=v.width,
    )
    v.text(
        0,
        724,
        "Source: make ui-feasibility · Geometry / Corners & controls",
        "tertiary",
        12,
        width=v.width,
    )
    return 784
