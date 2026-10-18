"""Interactive local specimens composed with the application's shared controls."""

from imgui_bundle import imgui as g


def status(v):
    from mojive.design.specification import component_catalog

    from .guide_panels import telemetry

    if v.values.get("status-readout") in ("time", "steps"):
        v.values["metric-mode"] = v.values["status-readout"]
    telemetry(v, 0, 104, v.width)
    v.text(0, 192, "Status spacing", size=16, bold=True)
    options = ("backend", "time", "steps", "delta", "physics", "render")
    changed, chosen = v.viewer_select(
        "status-readout", 0, 232, 240, v.values.get("status-readout", "physics"), options
    )
    v.values["status-readout"] = chosen
    if changed and chosen in ("time", "steps"):
        v.values["metric-mode"] = chosen
    _, context = v.viewer_select(
        "status-hint-context",
        264,
        232,
        min(240, v.width - 264),
        v.values.get("status-hint-context", "camera"),
        ("camera", "ready", "perturb"),
    )
    v.values["status-hint-context"] = context
    part = "status_" + chosen
    values = v.ui.component_style(part)
    supported = component_catalog()[part]["fields"]
    row = 0
    if chosen == "time":
        formats = {"seconds": "Seconds (3 decimals)", "clock": "Hours:minutes:seconds"}
        v.text(0, 305, "Time format", "muted")
        changed, selected = v.viewer_select(
            "status-time-format",
            240,
            296,
            min(240, v.width - 240),
            formats[values.get("displayFormat", "seconds")],
            tuple(formats.values()),
        )
        if changed:
            v.spec["parts"].setdefault(part, {})["displayFormat"] = next(
                key for key, label in formats.items() if label == selected
            )
            v._dirty = True
        row += 1
    for field, label in (
        ("paddingLeft", "Text left inset"),
        ("paddingRight", "Text right inset"),
        ("leftGap", "After divider"),
        ("rightGap", "Before divider"),
        ("valueWidth", "Numeric width"),
        ("labelGap", "Label / value"),
        ("unitGap", "Value / unit"),
    ):
        if field not in supported:
            continue
        y = 296 + row * 48
        row += 1
        v.text(
            0,
            y + 9,
            "Numeric width (0 = auto compact)"
            if chosen == "time" and field == "valueWidth"
            else label,
            "muted",
        )
        r = v.rect(240, y, min(240, v.width - 240), 32)
        g.set_cursor_screen_pos((r.x, r.y))
        g.set_next_item_width(r.width)
        changed, value = g.slider_int(
            "##status-" + field, round(values.get(field)), 0, 80 if field == "valueWidth" else 24
        )
        v.record("status-edit/" + field)
        if changed:
            v.spec["parts"].setdefault(part, {})[field] = value
            v._dirty = True
    return 296 + row * 48 + 44


def menus(v):
    from mojive.ui.workspace.design import DESIGN
    from mojive.ui.workspace.menu_catalog import TOP_MENUS

    menu = v.sample.app.workspace.menus
    with v.ui.design.family_scope("chrome"), v.ui.component_scope("titlebar"):
        style = v.ui.component_style("titlebar")
        padding = style.edges("padding")
        trigger_height = max(menu.trigger_size(name)[1] for name in (*TOP_MENUS, "More"))
        height = max(
            style.get("height", DESIGN.application_height),
            trigger_height + padding[1] + padding[3],
        )
        rect = v.rect(0, 104, v.width, height)
        g.set_cursor_screen_pos((rect.x, rect.y))
        # BeginMenuBar clips to the height reserved when its window begins.
        g.push_style_var(
            g.StyleVar_.frame_padding,
            (g.get_style().frame_padding.x, max(0, (rect.height - g.get_font_size()) / 2)),
        )
        g.push_style_var(g.StyleVar_.window_padding, (0, 0))
        g.begin_child(
            "Native menu bar", (rect.width, rect.height), window_flags=g.WindowFlags_.menu_bar
        )
        g.pop_style_var(2)
        box = v.ui.component_box(
            "titlebar", *rect.tuple(), key="guide-menu-titlebar", padding=(8 * v.s, 0, 8 * v.s, 0)
        )
        menu.bar(region=box.content, right_edge=box.content.right)
        for name in TOP_MENUS:
            if "menu-" + name in v.ui.hits:
                v.hits["sample-menu-" + name] = v.ui.hits["menu-" + name]
        g.end_child()
    detail_y = 104 + height + 48
    v.text(
        0,
        detail_y,
        "Menu actions use the native inventory and availability rules.",
        "muted",
        width=v.width,
    )
    v.text(
        0,
        detail_y + 32,
        "File operations are recorded as local preview actions.",
        "muted",
        width=v.width,
    )
    return detail_y + 96
