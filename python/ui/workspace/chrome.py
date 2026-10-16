"""Native workspace chrome; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.workspace.component_layout import control_frame_size, control_size, record_group
from mojive.ui.workspace.constants import ICONS, LEFT_PANELS, PANELS
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.widgets import (
    DIM,
    MUTED,
)


def sidebar(workspace, name, direction, size):
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.no_docking
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
    )
    if name == "Application":
        flags |= imgui.WindowFlags_.menu_bar
    imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
    visible = imgui.internal.begin_viewport_side_bar(
        name, imgui.get_main_viewport(), direction, size * workspace.s, flags
    )
    return visible


def end_sidebar(workspace):
    imgui.end()
    imgui.pop_style_var()


def chrome(workspace):
    u, scale = workspace.ui, workspace.s
    style = u.component_style("titlebar")
    action_sizes = tuple(
        control_size(u, key, 26, 28, icon=icon)
        for key, icon in (("toggle-inspector", "inspect"), ("preferences", "settings"))
    )
    with u.component_scope("titlebar"):
        menu_sizes = tuple(
            workspace.menus.trigger_size(name) for name in (*workspace.menus.top_menus, "More")
        )
    pads = style.edges("padding")
    height = max(
        style.get("height", DESIGN.application_height),
        max(v[1] for v in action_sizes) / scale + pads[1] + pads[3],
        max(v[1] for v in menu_sizes) + pads[1] + pads[3],
        style.line_height(style.get("fontSize", BODY_SIZE) * 1.4) + pads[1] + pads[3],
    )
    margins = style.edges("margin")
    reserved_height = height + margins[1] + margins[3] + u.metric("application", "rowGap", 0)
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding, (8 * scale, max(0, (height - BODY_SIZE) * scale / 2))
    )
    with workspace.app.scrollbar_visibility.panel_scope("Titlebar", window_name="Application"):
        if workspace.sidebar("Application", imgui.Dir.up, reserved_height):
            workspace.app.scrollbar_visibility.bind_current_window()
            p = imgui.get_window_pos()
            width = imgui.get_window_width()
            box = u.component_box(
                "titlebar",
                p.x,
                p.y,
                width,
                (height + margins[1] + margins[3]) * scale,
                key="application-titlebar",
                padding=(8 * scale, 0, 8 * scale, 0),
                minimum_height=(
                    max(max(v[1] for v in menu_sizes), max(v[1] for v in action_sizes) / scale)
                    + pads[1]
                    + pads[3]
                )
                * scale,
                observe=False,
            )
            content = box.content
            imgui.push_clip_rect(
                (box.frame.x, box.frame.y), (box.frame.right, box.frame.bottom), False
            )
            if u.design is not None:
                window = imgui.get_current_context().current_window
                u.design.record(
                    "application-chrome",
                    ("application",),
                    box.frame.tuple(),
                    "application",
                    content=list(box.content.tuple()),
                    fieldRegions={
                        "rowGap": [
                            box.outer.x,
                            box.outer.bottom,
                            box.outer.width,
                            reserved_height * scale - box.outer.height,
                        ]
                    },
                    layoutRect=[window.pos.x, window.pos.y, window.size.x, window.size.y],
                    consumedFields={
                        "application": [
                            "fontSize",
                            "lineHeightMode",
                            "lineHeight",
                            "titleHeight",
                            "rowGap",
                        ]
                    },
                )
            actions = u.component_style("title_actions")
            action_gap = actions.get("columnGap", actions.get("gap", 2)) * scale
            action_width = sum(v[0] for v in action_sizes) + action_gap
            action_y = style.align_rect(
                content, action_width, max(v[1] for v in action_sizes), horizontal="end"
            ).y
            first_size = control_frame_size(u, "toggle-inspector", *action_sizes[0], icon="inspect")
            second_size = control_frame_size(u, "preferences", *action_sizes[1], icon="settings")
            with u.component_scope("titlebar"):
                workspace.menus.bar(
                    region=content,
                    right_edge=content.right
                    - action_width
                    - style.get("columnGap", style.get("gap", 10)) * scale,
                )
            center_y = workspace.menus.menu_center_y
            from mojive.ui.workspace.widgets import ACCENT

            brand_width = u.measure("M", 18)
            u.text(
                content.x + (28 * scale - brand_width) / 2,
                center_y - 9 * scale,
                "M",
                ACCENT,
                18,
                bold=True,
            )
            workspace.rects["application-logo"] = [
                content.x + 2 * scale,
                center_y - 12 * scale,
                24 * scale,
                24 * scale,
            ]
            menu_right = max(
                (r[0] + r[2] for r in workspace.menus.menu_rects.values()), default=content.x
            )
            tx = max(
                content.x + content.width * 0.49,
                menu_right + style.get("columnGap", style.get("gap", 10)) * scale,
            )
            if width > 950 * scale and content.right - action_width - tx > 40 * scale:
                from mojive.design.native.geometry import Rect

                title_style = u.component_style("document_title")
                gap = title_style.get("columnGap", title_style.get("gap", 8)) * scale
                icon_size = 14 * scale
                icon_rect = style.align_rect(
                    Rect(tx, content.y, icon_size, content.height), icon_size, icon_size
                )
                u.icon("file", icon_rect.x, icon_rect.y, icon_size / scale, DIM)
                path = workspace.app.session.asset_path

                u.component_text(
                    "titlebar",
                    Rect(
                        tx + icon_size + gap,
                        content.y,
                        max(0, content.right - action_width - tx - icon_size - gap),
                        content.height,
                    ),
                    path.name if path else "Untitled",
                    MUTED,
                )
                record_group(
                    u,
                    "document_title",
                    "document-title",
                    (
                        (tx, content.y, icon_size, content.height),
                        (
                            tx + icon_size + gap,
                            content.y,
                            max(0, content.right - action_width - tx - icon_size - gap),
                            content.height,
                        ),
                    ),
                )
            if u.button(
                "toggle-inspector",
                content.right - action_width,
                action_y,
                *first_size,
                icon="inspect",
                active=workspace.open["Inspector"],
                tooltip="Toggle Inspector",
                width_limit=first_size[0] * scale,
            ):
                workspace.open["Inspector"] = not workspace.open["Inspector"]
            if u.button(
                "preferences",
                content.right - action_sizes[1][0],
                action_y,
                *second_size,
                icon="settings",
                tooltip="Settings · F9",
                active=workspace.open["Settings"],
                width_limit=second_size[0] * scale,
            ):
                workspace.activate("Settings")
            record_group(
                u,
                "title_actions",
                "title-actions",
                (u.hits[key] for key in ("toggle-inspector", "preferences")),
            )
            gap = style.get("columnGap", style.get("gap", 10)) * scale
            after_logo = content.x + 26 * scale
            u.record_component_box(
                "titlebar",
                box,
                key="application-titlebar",
                fields=[
                    workspace.rects["application-logo"],
                    *workspace.menus.menu_rects.values(),
                    *(u.hits[key] for key in ("toggle-inspector", "preferences")),
                ],
                fieldRegions={
                    name: [
                        [after_logo, content.y, gap, content.height],
                        [menu_right, content.y, gap, content.height],
                        [content.right - action_width - gap, content.y, gap, content.height],
                    ]
                    for name in ("columnGap", "gap")
                },
            )
            workspace.menus.dialogs()
            imgui.pop_clip_rect()
        workspace.end_sidebar()
    imgui.pop_style_var()
    rail = u.component_style("rail")
    rail_margin = rail.edges("margin")
    rail_width = rail.get("width", DESIGN.sidebar_width)
    with workspace.app.scrollbar_visibility.panel_scope("Navigation", window_name="Navigation"):
        if workspace.sidebar(
            "Navigation", imgui.Dir.left, rail_width + rail_margin[0] + rail_margin[2]
        ):
            workspace.app.scrollbar_visibility.bind_current_window()
            p, h = imgui.get_window_pos(), imgui.get_window_height()
            box = u.component_box(
                "rail",
                p.x,
                p.y,
                (rail_width + rail_margin[0] + rail_margin[2]) * scale,
                h,
                key="activity-rail",
                padding=(6 * scale, 8 * scale, 6 * scale, 8 * scale),
                observe=False,
            )
            slot = rail.get("slotSize", 36)
            gap = rail.get("rowGap", rail.get("gap", 6))
            child_sizes = tuple(
                control_size(
                    u,
                    "rail-" + name,
                    slot,
                    slot,
                    icon=icon,
                    icon_size=rail.get("iconSize", 24),
                    icon_role="activity",
                )
                for name, icon in zip(PANELS, ICONS, strict=True)
            )
            stride = max(v[1] for v in child_sizes) + gap * scale
            compact = box.content.height < len(PANELS) * stride
            rail_children = []
            for n, (name, icon) in enumerate(zip(PANELS, ICONS, strict=True)):
                yy = (
                    box.content.y + n * stride - imgui.get_scroll_y()
                    if compact or n < 7
                    else box.content.bottom - (len(PANELS) - n) * stride + gap * scale
                )
                frame = control_frame_size(
                    u,
                    "rail-" + name,
                    min(child_sizes[n][0], box.content.width),
                    child_sizes[n][1],
                    icon=icon,
                )
                if u.button(
                    "rail-" + name,
                    box.content.x,
                    yy,
                    *frame,
                    icon=icon,
                    icon_size=rail.get("iconSize", 24),
                    icon_role="activity",
                    width_limit=frame[0] * scale,
                    active=(
                        workspace.left_active == name
                        if name in LEFT_PANELS
                        else workspace.bottom_active == name
                    )
                    and workspace.open[name],
                    tooltip=name,
                ):
                    workspace.toggle_activity(name)
                rail_children.append(
                    [
                        box.content.x,
                        yy,
                        min(child_sizes[n][0], box.content.width),
                        child_sizes[n][1],
                    ]
                )
            rail_gaps = [
                [rect[0], rect[1] + rect[3], rect[2], gap * scale]
                for n, rect in enumerate(rail_children[:-1])
                if compact or n != 6
            ]
            u.record_component_box(
                "rail",
                box,
                key="activity-rail",
                fields=rail_children,
                fieldRegions=dict.fromkeys(("gap", "rowGap"), rail_gaps),
            )
            if compact:
                imgui.set_cursor_screen_pos(
                    (box.content.x, box.content.y + len(PANELS) * stride - imgui.get_scroll_y())
                )
                imgui.dummy((1, 1))
        workspace.end_sidebar()
