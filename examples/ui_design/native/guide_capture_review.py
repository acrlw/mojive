"""Capture a compact production-consumer review without the full Guide gallery."""

import json
from copy import deepcopy

from imgui_bundle import imgui as g
from PIL import Image

from mojive import commands as cmd
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.settings_panel import CATEGORIES


def capture_review(v):
    """Visit real readouts, filter chips, hints, geometry friction and color popups."""
    previous = v.page, deepcopy(v.values), deepcopy(v.selected), v.animate, v.clock
    observed = v.runtime.observe_components
    process, queued, pointer = v.window._input.process_inputs, [], [0.0, 0.0]
    original_text, emissions = ImguiDraw2D.text, []
    report = {"scale": v.s, "captures": {}}
    target = v.output / "focused-review-geometry.json"
    selection, scrolls = None, {}

    def inputs():
        process()
        io = g.get_io()
        io.add_mouse_pos_event(*pointer)
        for method, args in queued:
            getattr(io, method)(*args)
        queued.clear()
        io.delta_time = 1 / 60

    def text(draw, position, ink, value, **options):
        emissions.append(
            {"text": str(value), "position": list(position), "fontSize": g.get_font_size() / v.s}
        )
        return original_text(draw, position, ink, value, **options)

    def frame(count=1, *, readback=False):
        for _ in range(count):
            emissions.clear()
            pixels = v.frame(readback=readback)
        return pixels

    def event(method, *args):
        queued.append((method, args))
        frame()

    def records():
        return [*v.runtime.observed, *v.ui.design.primitives.observed]

    def show(page):
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        pointer[:] = (0.0, 0.0)
        v.page = page
        frame(4)
        for window in g.get_current_context().windows:
            if window.active and "Guide page/" + page in window.name:
                scrolls.setdefault(window.id_, (window.scroll.x, window.scroll.y))
                g.internal.set_scroll_y(window, 0)
        frame(4)

    def reveal(key):
        for _ in range(20):
            rect = v.ui.hits.get(key, v.hits.get(key))
            assert rect is not None, "Missing production control: " + key
            record = next(
                (
                    row
                    for row in reversed(records())
                    if row["key"] in (key, key.removesuffix("-frame"))
                ),
                None,
            )
            window = next(
                (
                    item
                    for item in g.get_current_context().windows
                    if record is not None and item.id_ == record.get("windowId")
                ),
                None,
            )
            if window is None:
                return rect
            while window is not None:
                clip = window.inner_clip_rect
                delta = min(0, rect[1] - clip.min.y - 8 * v.s) + max(
                    0, rect[1] + rect[3] - clip.max.y + 8 * v.s
                )
                target_y = min(window.scroll_max.y, max(0, window.scroll.y + delta))
                if abs(target_y - window.scroll.y) > native_epsilon(target_y, window.scroll.y):
                    scrolls.setdefault(window.id_, (window.scroll.x, window.scroll.y))
                    g.internal.set_scroll_y(window, target_y)
                    frame(4)
                    break
                window = window.parent_window
            else:
                return rect
        raise AssertionError("Production control cannot be revealed: " + key)

    def hover(key, *, frames=40):
        x, y, width, height = reveal(key)
        pointer[:] = (x + width / 2, y + height / 2)
        frame(frames)

    def click(key):
        hover(key, frames=2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(4)

    def save(name):
        pixels = frame(4, readback=True)
        Image.fromarray(pixels[::-1]).save(v.output / (name + ".png"))
        snapshot = {
            "image": name + ".png",
            "page": v.page,
            "pixelSize": [pixels.shape[1], pixels.shape[0]],
            "records": deepcopy(records()),
            "drawText": deepcopy(emissions),
        }
        report["captures"][name] = snapshot
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return snapshot

    def hint(name, key, *, shortcut=False):
        hover(key)
        snapshot = save(name)
        hints = [row for row in snapshot["records"] if row["kind"] == "tooltip"]
        assert hints and all(row["fontSize"] == 12 for row in hints)
        assert any(bool(row["shortcut"]) == shortcut for row in hints)
        return snapshot

    def color(name, key, axes):
        click(key + "-format-2")
        channels = [v.ui.hits[key + axis + "-frame"] for axis in axes]
        epsilon = native_epsilon(*(value for rect in channels for value in rect))
        assert max(rect[1] for rect in channels) - min(rect[1] for rect in channels) <= epsilon
        snapshot = save(name)
        editor = next(row for row in snapshot["records"] if row["kind"] == "color_editor")
        sv = editor["svRect"]
        assert abs(sv[2] - sv[3]) <= native_epsilon(*sv)
        planned = v.ui.component_style("color_editor").get("pickerHeight", 160)
        assert sv[3] <= planned * v.s + native_epsilon(*sv)
        snapshot["channels"] = channels
        snapshot["pickerHeightLogical"] = sv[3] / v.s
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    v.window._input.process_inputs, ImguiDraw2D.text = inputs, text
    v.runtime.observe_components, v.animate = True, False
    try:
        frame()
        session = v.sample.app.session
        selection = session.selected_node.node_id if session.selected_node is not None else None
        v.values["status-readout"], v.values["metric-mode"] = "time", "time"
        show("Status")
        snapshot = save("review-status-time")
        metrics = [
            row for row in snapshot["records"] if row["key"].startswith("application-status_")
        ]
        assert any(row["key"] == "application-status_time" for row in metrics)
        assert snapshot["drawText"], "Production status text was not observed"
        show("Scene")
        snapshot = hint("review-collision-hint", "scene-geometry-collision")
        chips = [
            row
            for row in snapshot["records"]
            if row["key"].startswith("scene-filter-") and "count" in row
        ]
        assert len(chips) >= 4
        assert all(
            row["fontSize"] == v.ui.component_style("filter").get("fontSize", 14) for row in chips
        )
        show("Viewport")
        hint("review-shortcut-hint", "tool-rotate", shortcut=True)
        geometry = next(node for node in session.nodes if node.name == "free_box")
        assert geometry.type.value == "geom"
        assert session.submit(cmd.SelectNode(geometry.node_id)).ok
        v.selected["Inspector tabs"] = 1
        for size in ("Wide", "Narrow"):
            v.values["inspector-size"] = size
            show("Inspector")
            key = "##contact_friction-0-frame"
            if key not in v.ui.hits:
                click("section-" + v.tr("contact properties").upper())
            reveal(key)
            snapshot = save("review-geometry-friction-" + size.lower())
            fields = [v.ui.hits[f"##contact_friction-{index}-frame"] for index in range(3)]
            group = next(
                row for row in snapshot["records"] if row["key"] == "##contact_friction/controls"
            )
            assert (
                len({rect[1] for rect in fields}) == (3 + group["columns"] - 1) // group["columns"]
            )
            assert all(field[2] > 0 for field in fields)
            if size == "Wide":
                assert group["columns"] == 3
            snapshot["geometryNodeId"], snapshot["frictionFields"] = geometry.node_id, fields
        v.selected["Settings navigation"] = CATEGORIES.index("MuJoCo Visuals")
        show("Settings")
        click("##collision_color-swatch")
        color("review-color-rgb-compact", "##collision_color", "RGB")
        v.values["inspector-size"], v.selected["Inspector tabs"] = "Wide", 2
        show("Inspector")
        click("material-color")
        color("review-color-rgba-compact", "color-editor", "RGBA")
        return report
    finally:
        try:
            event("add_key_event", g.Key.escape, True)
            event("add_key_event", g.Key.escape, False)
            if v.sample is not None:
                command = cmd.SelectNode(selection) if selection is not None else cmd.Select(0)
                v.sample.app.session.submit(command)
            for window in g.get_current_context().windows:
                if window.id_ in scrolls:
                    x, y = scrolls[window.id_]
                    g.internal.set_scroll_x(window, x)
                    g.internal.set_scroll_y(window, y)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        finally:
            v.window._input.process_inputs, ImguiDraw2D.text = process, original_text
            v.page, values, selected, v.animate, v.clock = previous
            v.values.clear()
            v.values.update(values)
            v.selected.clear()
            v.selected.update(selected)
            v.runtime.observe_components = observed
