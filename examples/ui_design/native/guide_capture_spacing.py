"""Capture production spacing, native scrollbars and fixed menu columns."""

import json
import math
from copy import deepcopy

from imgui_bundle import imgui as g

from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.settings_panel import CATEGORIES


def _rect(bounds):
    return [bounds.min.x, bounds.min.y, bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y]


def _window(page, suffix):
    return next(
        window
        for window in g.get_current_context().windows
        if window.active and "Native specimen/" + page in window.name and suffix in window.name
    )


def capture_panel_surfaces(v, page, click, save, event):
    """Record native panel, section-hover and timeline-form boundaries."""
    previous = v.page, deepcopy(v.values), deepcopy(v.selected)
    observed = v.runtime.observe_components
    report = {"scale": v.s}

    def record(key):
        return deepcopy(next(row for row in reversed(v.runtime.observed) if row["key"] == key))

    def inside(frame, child, epsilon):
        x, y, width, height = frame
        cx, cy, cw, ch = child
        return (
            x - epsilon <= cx
            and y - epsilon <= cy
            and cx + cw <= x + width + epsilon
            and cy + ch <= y + height + epsilon
        )

    def show(name):
        page(name)
        for window in g.get_current_context().windows:
            if window.active and "Guide page/" + name in window.name:
                g.internal.set_scroll_y(window, 0)
        for _ in range(4):
            v.frame()

    v.runtime.observe_components = True
    try:
        v.values["camera-size"] = "Wide"
        show("Cameras")
        save("panel-alignment-cameras")
        header = record("panel-header-Cameras")
        body = record("panel-body-Cameras")
        scroll = record("panel-scroll-Cameras")
        sections = [
            deepcopy(row) for row in v.runtime.observed if row.get("kind") == "section_header"
        ]
        epsilon = native_epsilon(*header["hit"], *scroll["windowRect"])
        for rect in (header["titleRect"], *header["actionRects"]):
            assert inside(header["hit"], rect, epsilon), "Panel header content exceeds its frame"
        hx, _, hw, _ = header["hit"]
        dx, _, dw, _ = header["dividerRect"]
        assert hx - epsilon <= dx and dx + dw <= hx + hw + epsilon
        for section in sections:
            sx, _, sw, _ = section["hit"]
            assert scroll["hit"][0] - epsilon <= sx
            assert sx + sw <= scroll["contentRight"] + epsilon
        if scroll["scrollbarRect"] is not None:
            assert scroll["contentRight"] <= scroll["scrollbarRect"][0] + epsilon
            assert inside(scroll["windowRect"], scroll["scrollbarRect"], epsilon)
        report["cameras"] = {
            "image": "panel-alignment-cameras.png",
            "header": header,
            "body": body,
            "scroll": scroll,
            "sections": sections,
            "native_epsilon": epsilon,
        }

        show("Layers")
        key = "section-" + v.tr("Editing aids")
        x, y, width, height = v.ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        save("section-hover-layers")
        section = record(key)
        frame, content = section["hit"], section["content"]
        epsilon = native_epsilon(*frame)
        assert section["hovered"] and tuple(section["backgroundRect"]) == tuple(frame)
        assert inside(frame, section["labelRect"], epsilon)
        assert inside(frame, section["iconRect"], epsilon)
        assert abs(section["labelRect"][0] - content[0]) <= epsilon
        icon = section["iconRect"]
        assert abs(icon[0] + icon[2] - content[0] - content[2]) <= epsilon
        report["section_hover"] = {
            "image": "section-hover-layers.png",
            "section": section,
            "text_inset_logical": (section["labelRect"][0] - frame[0]) / v.s,
            "chevron_inset_logical": (frame[0] + frame[2] - icon[0] - icon[2]) / v.s,
            "native_epsilon": epsilon,
        }

        show("Timeline")
        click("timeline-settings")
        save("timeline-settings")
        popup = record("timeline-settings-popup")
        epsilon = native_epsilon(*popup["hit"])
        assert all(inside(popup["hit"], row, epsilon) for row in popup["rows"])
        assert all(inside(popup["hit"], field, epsilon) for field in popup["fields"])
        assert min(popup["padding"]) > 0
        assert abs(popup["radius"] - popup["innerRadius"] - popup["padding"][0]) <= epsilon
        final = popup["fields"][-1]
        assert (
            abs(popup["hit"][1] + popup["hit"][3] - final[1] - final[3] - popup["padding"][3])
            <= epsilon
        )
        report["timeline_settings"] = {
            "image": "timeline-settings.png",
            "popup": popup,
            "native_epsilon": epsilon,
        }
        (v.output / "panel-alignment-geometry.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        v.page, v.values, v.selected = previous
        v.runtime.observe_components = observed
        event("add_mouse_pos_event", 0, 0)


def _scrollbar_geometry(v, window):
    """Read the native track and its rendered grab mesh after a real frame."""
    track = g.internal.get_window_scrollbar_rect(window, g.internal.Axis.y)
    colors = {
        g.color_convert_float4_to_u32(g.get_style().color_(slot)) & 0xFFFFFF
        for slot in (
            g.Col_.scrollbar_grab,
            g.Col_.scrollbar_grab_hovered,
            g.Col_.scrollbar_grab_active,
        )
    }
    # ImGui may merge a child draw list into its parent during Render().
    points = [
        (vertex.pos.x, vertex.pos.y)
        for host in g.get_current_context().windows
        if host.active
        for vertex in host.draw_list.vtx_buffer
        if vertex.col >> 24
        and vertex.col & 0xFFFFFF in colors
        and track.min.x - 1 <= vertex.pos.x <= track.max.x + 1
        and track.min.y - 1 <= vertex.pos.y <= track.max.y + 1
    ]
    assert points, "The native scrollbar thumb is not visible: " + window.name
    mesh = [
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    ]
    padding = math.trunc(
        min(
            g.get_style().scrollbar_padding,
            (track.max.x - track.min.x) / 2,
            (track.max.y - track.min.y) / 2,
        )
    )
    content = [
        window.work_rect.min.x,
        window.inner_rect.min.y,
        min(window.work_rect.max.x, window.inner_rect.max.x),
        window.inner_rect.max.y,
    ]
    submitted = [
        row["hit"]
        for row in v.runtime.observed
        if row.get("windowId") == window.id_ and "hit" in row
    ]
    return {
        "window": window.name,
        "window_id": window.id_,
        "native_track": _rect(track),
        "native_content_bounds": content,
        "inner_clip": _rect(window.inner_clip_rect),
        "submitted_content_bounds": (
            [
                min(rect[0] for rect in submitted),
                min(rect[1] for rect in submitted),
                max(rect[0] + rect[2] for rect in submitted),
                max(rect[1] + rect[3] for rect in submitted),
            ]
            if submitted
            else None
        ),
        "scroll_y": window.scroll.y,
        "scroll_max_y": window.scroll_max.y,
        "padding_logical": g.get_style().scrollbar_padding / v.s,
        "native_padding_pixels": padding,
        "thumb_width_logical": (track.max.x - track.min.x - 2 * padding) / v.s,
        "thumb_mesh_bounds": mesh,
        "mesh_clearance_to_content": (mesh[0] - content[2]) / v.s,
        "mesh_clearance_to_outer_edge": (track.max.x - mesh[2]) / v.s,
    }


def capture_spacing(v, page, click, save, event):
    """Exercise the actual guide without changing its authored specification."""
    app, workspace = v.sample.app, v.sample.app.workspace
    previous = v.page, deepcopy(v.values), deepcopy(v.selected), app.tool
    observed = v.runtime.observe_components
    scroll = {window.id_: (window, window.scroll.y) for window in g.get_current_context().windows}
    menu = workspace.menus
    menu_state = menu.requested_path, menu.reset_pending
    scene_state = pose_state = None
    query, filter_ = workspace.query, workspace.filter
    report = {"scale": v.s}

    def show(name):
        page(name)
        window = next(
            window
            for window in g.get_current_context().windows
            if window.active
            and "Guide page/" + name in window.name
            and "Native specimen/" not in window.name
        )
        g.internal.set_scroll_y(window, 0)
        for _ in range(4):
            v.frame()

    def scrollbar(name, suffix):
        window = _window(name, suffix)
        assert window.scrollbar_y and window.scroll_max.y > 0, name + " does not overflow"
        clip = window.inner_clip_rect
        event("add_mouse_pos_event", (clip.min.x + clip.max.x) / 2, (clip.min.y + clip.max.y) / 2)
        event("add_mouse_wheel_event", 0, -2)
        for _ in range(4):
            v.frame()
        track = g.internal.get_window_scrollbar_rect(window, g.internal.Axis.y)
        y = (max(track.min.y, clip.min.y) + min(track.max.y, clip.max.y)) / 2
        event("add_mouse_pos_event", (track.min.x + track.max.x) / 2, y)
        save(name.lower() + "-scrollbar")
        result = _scrollbar_geometry(v, window)
        assert result["scroll_y"] > 0, name + " did not receive the wheel gesture"
        report[name.lower() + "_scrollbar"] = result

    def transform_menu():
        click("menu-View")
        key = "menu-item-View/Transform tool"
        x, y, width, height = v.ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        for _ in range(60):
            if "menu-item-View/Transform tool/Rotate" in v.ui.hits:
                return
            v.frame()
        raise AssertionError("The production Transform tool submenu did not open")

    v.runtime.observe_components = True
    try:
        v.values["pose-size"] = "Narrow"
        show("Pose")
        pose_state = dict(workspace.navigation.pose_editor._expanded)
        scrollbar("Pose", "/Panel content")

        show("Scene")
        hierarchy = workspace.navigation.hierarchy
        scene_state = dict(hierarchy._open_state)
        workspace.query, workspace.filter = "", "All"
        hierarchy._open_state.update((node.node_id, True) for node in app.session.nodes)
        for _ in range(4):
            v.frame()
        scrollbar("Scene", "/Scene entities")

        show("Timeline")
        save("timeline-gutter")
        layout = workspace.timeline_panel.last_layout
        gutter = v.ui.hits["timeline-gutter"]
        report["timeline_gutter"] = {
            "hit": gutter,
            "left_background_right": layout.channel_right,
            "right_background_left": layout.track_left,
            "width_logical": layout.column_gap / v.s,
            "time_bounds": layout.time_bounds,
        }
        assert math.isclose(gutter[0], layout.channel_right)
        assert math.isclose(gutter[0] + gutter[2], layout.track_left)

        show("Menus")
        transform_menu()
        click("menu-item-View/Transform tool/Rotate")
        assert app.tool == "rotate"
        transform_menu()
        x, y, width, height = v.ui.hits["menu-item-View/Transform tool/Rotate"]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        save("menu-columns")
        rows = menu.observed["View/Transform tool"]["rows"]
        rotate = next(row for row in rows if row["label"] == "Rotate")
        assert rotate["checked"] and rotate["visible"]
        columns = [
            row
            for row in v.runtime.observed
            if row["key"].startswith("menu-item-View/Transform tool/") and "checkRect" in row
        ]
        item = next(row for row in columns if row["label"] == "Rotate")
        check, icon, label, shortcut = (
            item[field] for field in ("checkRect", "iconSlotRect", "labelRect", "shortcutRect")
        )
        assert check[0] + check[2] < icon[0] < label[0] < shortcut[0]
        assert (
            next(row.shortcut for row in menu.items("Transform tool") if row.label == "Rotate")
            == "E"
        )
        report["menu_columns"] = {"rows": rows, "geometry": columns, "checked_shortcut": "E"}

        show("Settings")
        v.selected["Settings navigation"] = CATEGORIES.index("MuJoCo Visuals")
        save("settings-navigation-inset")
        navigation = next(row for row in v.runtime.observed if row["key"] == "settings-navigation")
        rows = [
            row
            for row in v.ui.design_primitives.observed
            if row["key"] in {"settings-" + category for category in CATEGORIES}
        ]
        x, y, width, height = navigation["hit"]
        cx, cy, cw, ch = navigation["content"]
        report["settings_navigation"] = {
            "surface": navigation,
            "rows": rows,
            "insets_logical": [
                (cx - x) / v.s,
                (cy - y) / v.s,
                (x + width - cx - cw) / v.s,
                (y + height - cy - ch) / v.s,
            ],
        }
        from .guide_capture_more_menus import capture_more_menus

        report["panel_surfaces"] = capture_panel_surfaces(v, show, click, save, event)
        report["more_menus"] = capture_more_menus(v, show, click, save, event)
        (v.output / "spacing-geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        app.tool = previous[3]
        menu.requested_path, menu.reset_pending = menu_state
        workspace.query, workspace.filter = query, filter_
        if scene_state is not None:
            workspace.navigation.hierarchy._open_state = scene_state
        if pose_state is not None:
            workspace.navigation.pose_editor._expanded = pose_state
        v.page, v.values, v.selected = previous[:3]
        v.runtime.observe_components = observed
        for window, value in scroll.values():
            g.internal.set_scroll_y(window, value)
        event("add_mouse_pos_event", 0, 0)
