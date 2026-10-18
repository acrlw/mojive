"""Measured output actions; history, filtering and selection remain with OutputPanel."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.panels.filters import severity_color
from mojive.ui.workspace.component_layout import (
    component_flow,
    control_frame_size,
    control_size,
)


def _measure(ui, key, icon, label, baseline=28):
    parts = ui.design.button_parts(key, label) if ui.design is not None else "button"
    style, scale = ui.component_style(parts), ui.s
    font = style.get("fontSize", 14)
    pads = style.edges("padding", scale, (8, 0, 8, 0))
    width = baseline
    if label:
        glyph = style.get("iconSize", 14) * scale
        gap = style.get("columnGap", style.get("gap", 6)) * scale
        width = max(
            40 if key.startswith("output-level-") else 70,
            (ui.measure(label, font, font_resolved=True) + glyph + gap + pads[0] + pads[2]) / scale,
        )
    return control_size(ui, key, width, 28, icon=icon, label=label)


def _popup(output, ctx, levels, labels, visible, entries, newest):
    menus = output.workspace.menus
    with menus.local_popup("##output-levels") as opened:
        if opened:
            output._draw_level_menu(levels, labels)
    with menus.local_popup("##output-actions") as opened:
        if opened:
            selected = any(entry.sequence in output._selected_sequences for entry in visible)
            if menus.local_item(
                "Copy selection" if selected else "Copy shown",
                enabled=bool(visible),
                key="output-menu-copy",
                icon="copy",
            )[0]:
                imgui.set_clipboard_text(output._copy_text(ctx.output))
            if menus.local_item(
                "Auto-scroll",
                checked=output.auto_scroll,
                checkable=True,
                key="output-menu-auto-scroll",
                icon="follow",
            )[0]:
                output._toggle_auto_scroll()
            if menus.local_item(
                "Clear output", enabled=bool(entries), key="output-menu-clear", icon="delete"
            )[0]:
                output._clear_output(ctx.output)
                entries, newest = (), 0
            menus.local_separator()
            output._draw_level_menu(levels, labels)
    return entries, newest


def draw_output_toolbar(output, ctx, entries, newest):
    ui, scale = output.workspace.ui, output.workspace.s
    origin = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    style = ui.component_style("output_toolbar")
    pad, margin = style.edges("padding", scale), style.edges("margin", scale)
    available = max(0, width - pad[0] - pad[2] - margin[0] - margin[2])
    gap = style.get("columnGap", style.get("gap", 4)) * scale
    row_gap = style.get("rowGap", style.get("gap", 8)) * scale
    font = style.get("fontSize", 14)
    minimum_row = max(28, font, style.line_height(font * 1.4)) * scale
    levels = tuple(
        (
            level,
            ctx.tr(label),
            icon,
            tuple(round(c * 255) for c in severity_color(ctx.theme, level)),
        )
        for level, label, icon in (
            ("info", "Info", "info"),
            ("warning", "Warning", "warn"),
            ("error", "Error", "err"),
        )
    )
    labels = tuple(f"{label} {output._level_counts[level]}" for level, label, _, _ in levels)
    visible = output._filter_entries(entries, newest)
    selected = any(entry.sequence in output._selected_sequences for entry in visible)
    actions = [
        (
            "output-search",
            "search",
            "",
            {"active": output.show_search, "tooltip": ctx.tr("Search output")},
        ),
        (
            "output-copy",
            "copy",
            "",
            {
                "enabled": bool(visible),
                "tooltip": ctx.tr("Copy selection" if selected else "Copy shown"),
            },
        ),
        (
            "output-follow",
            "follow",
            "",
            {"active": output.auto_scroll, "tooltip": ctx.tr("Auto-scroll")},
        ),
        (
            "output-clear",
            "delete",
            "",
            {"enabled": bool(entries), "tooltip": ctx.tr("Clear output")},
        ),
        ("collapse-output", "down", "", {"tooltip": ctx.tr("Collapse Output")}),
    ]
    trigger_size = _measure(ui, "output-levels", "filter", "")
    action_sizes = [_measure(ui, key, icon, label) for key, icon, label, _ in actions]
    compact = sum(v[0] for v in action_sizes) + len(actions) * gap + trigger_size[0] > available
    if compact:
        actions = [
            actions[0],
            ("output-more", "more", "", {"tooltip": ctx.tr("Output actions")}),
            actions[-1],
        ]
        action_sizes = [_measure(ui, key, icon, label) for key, icon, label, _ in actions]
    severity = [
        (
            "output-level-" + level,
            icon,
            str(output._level_counts[level]),
            {"active": level in output._levels, "icon_color": color, "tooltip": caption},
        )
        for level, caption, icon, color in levels
    ]
    severity_sizes = [_measure(ui, key, icon, label) for key, icon, label, _ in severity]
    expanded = (
        sum(v[0] for v in (*severity_sizes, *action_sizes))
        + (len(severity) + len(actions) - 1) * gap
        <= available
    )
    if not expanded:
        show_trigger = (
            trigger_size[0] + sum(v[0] for v in action_sizes) + len(actions) * gap <= available
        )
        severity = (
            [
                (
                    "output-levels",
                    "filter",
                    "",
                    {"active": len(output._levels) < 3, "tooltip": ctx.tr("Severity")},
                )
            ]
            if show_trigger
            else []
        )
        severity_sizes = [trigger_size] if show_trigger else []
    controls = [*severity, *actions]
    actual_sizes = [*severity_sizes, *action_sizes]
    box, flow = component_flow(
        ui,
        "output_toolbar",
        origin.x,
        origin.y,
        width,
        [(w, max(h, minimum_row)) for w, h in actual_sizes],
        key="output-toolbar",
        column_gap=4,
        row_gap=8,
        observe=False,
    )
    placements = list(flow.rectangles)
    if placements and len({r.y for r in placements}) == 1:
        remaining = max(0, box.content.right - placements[-1].right)
        for index in range(len(severity), len(placements)):
            placements[index] = replace(placements[index], x=placements[index].x + remaining)
    fields = []
    for (key, icon, label, options), allocated, (_w, h) in zip(
        controls, placements, actual_sizes, strict=True
    ):
        placed = style.align_rect(allocated, allocated.width, h, horizontal="start")
        frame = control_frame_size(ui, key, placed.width, h, icon=icon, label=label)
        clicked = ui.button(
            key,
            placed.x,
            placed.y,
            *frame,
            icon=icon,
            label=label,
            width_limit=frame[0] * scale,
            **options,
        )
        fields.append(list(ui.hits[key]))
        if not clicked:
            continue
        if key.startswith("output-level-"):
            output._levels.symmetric_difference_update({key.removeprefix("output-level-")})
            output._last_sequence = 0
        elif key == "output-levels":
            imgui.open_popup("##output-levels")
        elif key == "output-search":
            output._toggle_search()
        elif key == "output-more":
            imgui.open_popup("##output-actions")
        elif key == "output-copy":
            imgui.set_clipboard_text(output._copy_text(ctx.output))
        elif key == "output-follow":
            output._toggle_auto_scroll()
        elif key == "output-clear":
            output._clear_output(ctx.output)
            entries, newest = (), 0
        elif key == "collapse-output":
            output.workspace.collapse_bottom()
    entries, newest = _popup(output, ctx, levels, labels, visible, entries, newest)
    if output.show_search:
        search_parts = ("input_text", "search")
        search_style = ui.component_style(search_parts)
        search_margin = search_style.edges("margin", scale)
        search_height = (
            (ui.design.primitives.field_height(search_parts) if ui.design is not None else 28)
            * scale
            + search_margin[1]
            + search_margin[3]
        )
        clear_size = (
            _measure(ui, "output-filter-clear", "close", "") if output._filter_text else (0, 0)
        )
        search_y = box.content.y + flow.height + row_gap
        search_row_height = max(search_height, clear_size[1])
        search_rect = Rect(
            box.content.x,
            search_y,
            max(0, box.content.width - clear_size[0] - (gap if clear_size[0] else 0)),
            search_row_height,
        )
        search_at = style.align_rect(
            search_rect, search_rect.width, search_height, horizontal="start"
        )
        if output._focus_search:
            imgui.set_keyboard_focus_here()
            output._focus_search = False
        changed, output._filter_text = ui.search(
            "##output-filter",
            search_at.x,
            search_at.y,
            search_at.width,
            output._filter_text,
            hint=ctx.tr("Search output"),
        )
        fields.append(list(ui.hits["##output-filter"]))
        if changed:
            output._last_sequence = 0
        if clear_size[0]:
            clear = style.align_rect(
                Rect(search_rect.right + gap, search_y, clear_size[0], search_row_height),
                *clear_size,
                horizontal="start",
            )
            frame = control_frame_size(ui, "output-filter-clear", *clear_size, icon="close")
            if ui.button(
                "output-filter-clear",
                clear.x,
                clear.y,
                *frame,
                icon="close",
                tooltip=ctx.tr("Clear search"),
                width_limit=frame[0] * scale,
            ):
                output._filter_text = ""
                output._last_sequence = 0
            fields.append(list(ui.hits["output-filter-clear"]))
        box = ui.component_box(
            "output_toolbar",
            origin.x,
            origin.y,
            width,
            box.outer.height + row_gap + search_row_height,
            observe=False,
        )
    ui.record_component_box(
        "output_toolbar",
        box,
        key="output-toolbar",
        fields=fields,
        expandedLevels=expanded,
        searchVisible=output.show_search,
    )
    imgui.set_cursor_screen_pos((origin.x, box.outer.bottom))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
    imgui.dummy((0, 0))
    imgui.pop_style_var()
    return entries, newest
