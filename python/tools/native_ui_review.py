"""Capture interaction regressions through the actual production workspace."""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from itertools import pairwise
from math import ceil, floor
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.capture import CaptureSurface
from mojive.tools.native_workspace import capture
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.menu_geometry import native_epsilon


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported review observation: {type(value).__name__}")


def _records(viewer):
    ui = viewer.app.workspace.ui
    primitives = getattr(ui, "design_primitives", None)
    return deepcopy([*(primitives.observed if primitives else ()), *viewer.app.design.observed])


def _tooltips():
    return [
        {"name": window.name, "rect": [*window.pos, *window.size]}
        for window in imgui.get_current_context().windows
        if window.active and window.flags & imgui.WindowFlags_.tooltip and not window.hidden
    ]


class _NativeInput:
    """Queue events after backend polling and retain hover throughout readback settling."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.queued = []
        self.pointer = (0, 0)
        self.emitted = []
        self.nominal_type = None

    def __enter__(self):
        self.process = self.viewer.window._input.process_inputs
        self.paint_text = ImguiDraw2D.text
        ui = self.viewer.app.workspace.ui
        self.text_draw = ui.text_draw

        def text_draw(draw, part):
            result = self.text_draw(draw, part)
            original = result.text

            def nominal_text(*args, **kwargs):
                previous = self.nominal_type
                self.nominal_type = {
                    "fontSize": result.size / ui.s,
                    "fontFamily": result.style.get("fontFamily", "inherit"),
                    "part": part,
                }
                try:
                    return original(*args, **kwargs)
                finally:
                    self.nominal_type = previous

            result.text = nominal_text
            return result

        def inputs():
            self.process()
            io = imgui.get_io()
            io.add_mouse_pos_event(*self.pointer)
            for method, args in self.queued:
                getattr(io, method)(*args)
            self.queued.clear()
            io.delta_time = 1 / 60
            self.emitted.clear()

        def text(painter, pos, color, value, **kwargs):
            window = imgui.get_current_context().current_window
            self.emitted.append(
                {
                    "text": value,
                    "position": list(pos),
                    "fontPixels": imgui.get_font_size(),
                    "fontId": imgui.get_font().font_id,
                    "fontFace": imgui.get_font().get_debug_name(),
                    "nominalTypography": self.nominal_type,
                    "color": list(color),
                    "windowId": window.id_,
                }
            )
            return self.paint_text(painter, pos, color, value, **kwargs)

        self.viewer.window._input.process_inputs = inputs
        ui.text_draw = text_draw
        ImguiDraw2D.text = text
        return self

    def __exit__(self, *_):
        self.viewer.window._input.process_inputs = self.process
        self.viewer.app.workspace.ui.text_draw = self.text_draw
        ImguiDraw2D.text = self.paint_text

    def sync(self, count=4):
        for _ in range(count):
            self.viewer.sync()

    def event(self, method, *args):
        self.queued.append((method, args))
        self.sync(1)

    def move(self, point, *, frames=4):
        self.pointer = tuple(point)
        self.sync(frames)

    def click(self, key):
        rect = self.viewer.app.workspace.ui.hits[key]
        self.click_at((rect[0] + rect[2] / 2, rect[1] + rect[3] / 2))

    def click_at(self, point):
        self.move(point)
        self.event("add_mouse_button_event", 0, True)
        self.event("add_mouse_button_event", 0, False)
        self.sync()

    def escape(self):
        self.event("add_key_event", imgui.Key.escape, True)
        self.event("add_key_event", imgui.Key.escape, False)
        self.move((0, 0))

    def hover(self, key):
        x, y, width, height = self.viewer.app.workspace.ui.hits[key]
        self.move((x + width / 2, y + height / 2), frames=45)


def _content_window(fragment):
    matches = [
        window
        for window in imgui.get_current_context().windows
        if window.active and fragment in window.name
    ]
    matches = [
        window
        for window in matches
        if not any(
            window.name.startswith(parent.name + "/") for parent in matches if parent is not window
        )
    ]
    assert len(matches) == 1, (fragment, [window.name for window in matches])
    return matches[0]


def _reveal(inputs, key, child):
    """Scroll a real child so an observed control can receive native pointer input."""
    ui = inputs.viewer.app.workspace.ui
    window = _content_window(child)
    rect = ui.hits[key]
    clip = window.inner_clip_rect
    top, bottom = clip.min.y + 12 * ui.s, clip.max.y - 12 * ui.s
    if rect[1] < top or rect[1] + rect[3] > bottom:
        destination = window.scroll.y + rect[1] - top
        imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, destination)))
        inputs.sync()
    return ui.hits[key]


@contextmanager
def _floating_panel(inputs, panel):
    """Resize a production review panel while retaining its original dock assignment."""
    workspace = inputs.viewer.app.workspace
    inputs.sync()
    window = imgui.internal.find_window_by_name(panel)
    assert window is not None
    position, size = tuple(window.pos), tuple(window.size)
    dock_id, begin = window.dock_id, workspace.panel
    layout = imgui.save_ini_settings_to_memory()
    width = [560]
    if dock_id:
        imgui.internal.dock_context_process_undock_window(imgui.get_current_context(), window, True)

    def place(name, *args, **kwargs):
        if (name if isinstance(name, str) else name.name) == panel:
            scale = workspace.s
            viewport = imgui.get_main_viewport()
            imgui.set_next_window_pos(
                (viewport.work_pos.x + 350 * scale, viewport.work_pos.y + 30 * scale)
            )
            imgui.set_next_window_size(
                (width[0] * scale, min(720 * scale, viewport.work_size.y - 60 * scale))
            )
        return begin(name, *args, **kwargs)

    workspace.panel = place
    try:
        inputs.sync()
        yield width
    finally:
        workspace.panel = begin
        if dock_id:
            # Removing the only window collapses its leaf and renames sibling
            # nodes. Restore the native layout, rather than reuse a deleted ID.
            imgui.load_ini_settings_from_memory(layout)
        else:
            imgui.set_window_pos(panel, position)
            imgui.set_window_size(panel, size)
        inputs.sync()
        if dock_id:
            assert imgui.internal.find_window_by_name(panel).dock_id == dock_id


def _floating_inspector(inputs):
    return _floating_panel(inputs, "Inspector")


def capture_regressions(viewer, output):
    """Save production evidence and fail when any exercised regression returns.

    The caller owns the isolated viewer and GPU serialization. Every widget is
    activated through queued native events; no design-guide specimen replaces a
    production panel. Diagnostic JSON is flushed after each completed capture.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    workspace, app, session = viewer.app.workspace, viewer.app, viewer.session
    runtime, ui = app.design, workspace.ui
    observing = runtime.observe_components
    report = {"scale": viewer.window.style_scale, "captures": {}, "checks": {}}

    def flush():
        (output / "review-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=_json) + "\n",
            encoding="utf-8",
        )

    def save(inputs, name, *, panels=(), extra=None):
        image_path = output / (name + ".png")
        snapshot = capture(viewer, image_path)
        records = _records(viewer)
        state = {
            "image": image_path.name,
            "snapshot": snapshot,
            "records": records,
            "emittedText": deepcopy(inputs.emitted),
            "nativeTooltips": _tooltips(),
            "crops": {},
            **(extra or {}),
        }
        with Image.open(image_path) as image:
            display = imgui.get_io().display_size
            ratio = image.width / display.x, image.height / display.y
            rects = {}
            status = next((r for r in records if r["key"] == "application-statusbar"), None)
            if status is not None:
                rects["status"] = status["hit"]
            if name in ("scene-counts", "navigation-settings-hint"):
                rail = next(r for r in records if r["key"] == "activity-rail")
                x, y, width, height = rail["hit"]
                rects["navigation-bottom"] = (
                    x,
                    max(y, y + height - 156 * ui.s),
                    width + 100 * ui.s,
                    min(height, 156 * ui.s),
                )
            for panel in panels:
                window = imgui.internal.find_window_by_name(panel)
                if window is not None and window.active:
                    rects[panel.lower()] = (*window.pos, *window.size)
            for index, tip in enumerate(state["nativeTooltips"]):
                rects[f"tooltip-{index}"] = tip["rect"]
            for index, record in enumerate(r for r in records if r["kind"] == "color_popup"):
                rects[f"color-popup-{index}"] = record["hit"]
            for label, (x, y, width, height) in rects.items():
                padding = 4 * ui.s
                crop = (
                    max(0, floor((x - padding) * ratio[0])),
                    max(0, floor((y - padding) * ratio[1])),
                    min(image.width, ceil((x + width + padding) * ratio[0])),
                    min(image.height, ceil((y + height + padding) * ratio[1])),
                )
                if crop[2] > crop[0] and crop[3] > crop[1]:
                    path = output / f"{name}-{label}.png"
                    image.crop(crop).save(path)
                    state["crops"][label] = path.name
        report["captures"][name] = state
        flush()
        return state

    def quiet_hint(state, *, shortcut=False):
        tips = [r for r in state["records"] if r["kind"] == "tooltip"]
        assert len(tips) == 1, tips
        assert len(state["nativeTooltips"]) == 1, state["nativeTooltips"]
        assert tips[0]["fontSize"] == 12, tips
        assert bool(tips[0]["shortcut"]) is shortcut, tips
        assert " · " not in tips[0]["text"], tips
        return tips[0]

    def colors(state, key, count, mode):
        editor = next(r for r in state["records"] if r["key"] == key + "-editor")
        popup = next(r for r in state["records"] if r["kind"] == "color_popup")
        epsilon = native_epsilon(*popup["hit"], *editor["svRect"]) / ui.s
        assert 120 <= editor["pickerHeight"] <= 160 + epsilon, editor
        assert popup["hit"][2] / ui.s <= 282 + epsilon, popup
        if mode:
            fields = [ui.hits[key + channel + "-frame"] for channel in "RGBA"[:count]]
            epsilon = native_epsilon(*(value for rect in fields for value in rect))
            assert max(r[1] for r in fields) - min(r[1] for r in fields) <= epsilon, fields
            assert all(r[2] > 0 for r in fields), fields
            for first, second in pairwise(fields):
                assert first[0] + first[2] <= second[0] + epsilon, fields
        else:
            assert key + "-hex" in ui.hits
        return {"mode": mode, "editor": editor, "popup": popup}

    def status_readouts(state):
        bar = next(r for r in state["records"] if r["key"] == "application-statusbar")
        epsilon = native_epsilon(*bar["hit"], *bar["content"])
        assert 28 * ui.s - epsilon <= bar["hit"][3] <= 28 * ui.s + 1, bar
        assert min(bar["padding"][1::2]) >= 2 * ui.s - epsilon, bar
        assert bar["content"][3] >= 24 * ui.s - epsilon, bar
        fields = sorted(
            (r for r in state["records"] if "valueTextRect" in r), key=lambda r: r["hit"][0]
        )
        assert fields, "Production status readouts must be observed"
        separators = [r for r in state["records"] if "status_separator" in r["parts"]]
        typography, phrases, gaps = set(), [], []
        for field in fields:
            x, y, width, height = field["hit"]
            clip = field["clip"]
            epsilon = native_epsilon(x, y, x + width, y + height, *clip)
            assert x >= clip[0] - epsilon and x + width <= clip[2] + epsilon, field
            assert y >= clip[1] - epsilon and y + height <= clip[3] + epsilon, field
            assert field["trailingReserve"] <= epsilon, field
            content = field["content"]
            assert abs(field["unitRight"] - content[0] - content[2]) <= epsilon, field
            painted = sorted(
                (
                    item
                    for item in state["emittedText"]
                    if item["windowId"] == field["windowId"]
                    and x - epsilon <= item["position"][0] < x + width
                    and y <= item["position"][1] < y + height
                ),
                key=lambda item: item["position"][0],
            )
            assert painted and all(item["text"] and "…" not in item["text"] for item in painted)
            for item in painted:
                nominal = item["nominalTypography"]
                assert nominal is not None, item
                typography.add(
                    (item["fontId"], item["fontFace"], item["fontPixels"], nominal["fontSize"])
                )
            phrases.append({"key": field["key"], "text": "".join(item["text"] for item in painted)})
        for before, after in pairwise(fields):
            divider = next(
                r for r in separators if before["unitRight"] <= r["hit"][0] < after["content"][0]
            )
            left = divider["hit"][0] - before["unitRight"]
            right = after["content"][0] - divider["hit"][0] - divider["hit"][2]
            assert abs(left - right) <= native_epsilon(left, right, *divider["hit"]), (
                before,
                after,
                left,
                right,
            )
            gaps.append(
                {"divider": divider["key"], "phraseToDivider": left, "dividerToPhrase": right}
            )
        assert len(typography) == 1, typography
        return {
            "phrases": phrases,
            "typography": [list(item) for item in typography],
            "gaps": gaps,
            "bar": bar,
        }

    def inline_boolean_geometry(state, row_key, control_key, caption):
        rows = {record["key"]: record for record in state["records"]}
        row, switch = rows[row_key], rows[control_key]
        label, control = row["labelRect"], switch["hit"]
        track = switch.get("switchRect", control)
        frame = row["frameRect"]
        epsilon = native_epsilon(*label, *control, *track, *frame)
        assert row["windowId"] == switch["windowId"], (row, switch)
        assert row["layout"] == "left", row
        assert abs(label[0] - frame[0]) <= epsilon, row
        assert abs(label[1] + label[3] / 2 - control[1] - control[3] / 2) <= epsilon, (
            row,
            switch,
        )
        assert track[0] >= label[0] + label[2] - epsilon, (row, switch)
        assert track[0] + track[2] <= frame[0] + frame[2] + epsilon, (row, switch)
        shown = workspace._panel_context.tr(caption)
        shown = shown[:1].upper() + shown[1:]
        assert any(
            item["text"] == shown
            and item["windowId"] == row["windowId"]
            and row["hit"][1] <= item["position"][1] < row["hit"][1] + row["hit"][3]
            for item in state["emittedText"]
        ), (shown, row)
        return {"row": row, "control": switch}

    def body_switch_geometry(state, *, points):
        prefix = "body_inertial_properties/row/"
        geometry = inline_boolean_geometry(
            state, prefix + "mocap body", "##body-mocap", "mocap body"
        )
        gravity = next(r for r in state["records"] if r["key"] == prefix + "gravity compensation")
        label = geometry["row"]["labelRect"]
        epsilon = native_epsilon(*label, *gravity["labelRect"])
        assert abs(label[0] - gravity["labelRect"][0]) <= epsilon, (geometry, gravity)
        # Boolean fit must not change the neighboring numeric grid's reflow.
        assert gravity["layout"] == ("top" if points == 310 else "left"), gravity
        return {"panelWidthPoints": points, **geometry, "gravity": gravity}

    try:
        runtime.observe_components = True
        with _NativeInput(viewer) as inputs:
            workspace.menus.preview(())
            inputs.escape()
            inputs.escape()
            workspace.open["Settings"] = False
            workspace.collapse_bottom()
            workspace.activate("Scene")
            workspace.open["Inspector"] = True
            inputs.sync()
            state = save(inputs, "scene-counts", panels=("Scene",))
            report["checks"]["statusReadouts"] = status_readouts(state)
            chips = [r for r in state["records"] if r["key"].startswith("scene-filter-")]
            expected = dict(workspace.navigation.scene_filters(workspace._panel_context))
            assert {r["key"].removeprefix("scene-filter-"): r["count"] for r in chips} == expected
            for chip in chips:
                caption = chip["fields"][1]
                matches = [
                    item
                    for item in state["emittedText"]
                    if item["windowId"] == chip["windowId"]
                    and abs(item["position"][0] - caption[0]) < 1
                    and abs(item["position"][1] - caption[1]) < caption[3]
                ]
                assert any(item["text"] == str(chip["count"]) for item in matches), (chip, matches)
                assert chip["fontSize"] >= 14, chip
                assert chip["fontFamily"] != "mono", chip
            report["checks"]["sceneCounts"] = expected

            visible_entities = {
                f"tree-row-{node.node_id}"
                for node in session.nodes
                if node.type not in (NodeType.MODEL, NodeType.ENVIRONMENT)
            }
            row = next(
                r
                for r in state["records"]
                if r["key"] in visible_entities
                and r["clip"][1] <= r["hit"][1]
                and r["hit"][1] + r["hit"][3] <= r["clip"][3]
            )
            eye = row["visibilityRect"]
            inputs.move((eye[0] + eye[2] / 2, eye[1] + eye[3] / 2), frames=45)
            state = save(inputs, "scene-visibility-hover", panels=("Scene",))
            row = next(r for r in state["records"] if r["key"] == row["key"])
            eye, frame = row["visibilityRect"], row["hit"]
            epsilon = native_epsilon(*eye, *frame)
            right_gap = frame[0] + frame[2] - eye[0] - eye[2]
            assert right_gap >= 4 * ui.s - epsilon, row
            assert row["visibilityInset"] >= 3 * ui.s - epsilon, row
            assert row["hovered"], row
            assert "visible-" + row["key"].removeprefix("tree-row-") in ui.hits
            report["checks"]["sceneVisibilityInsets"] = row
            inputs.move((0, 0))

            inputs.hover("rail-Settings")
            state = save(inputs, "navigation-settings-hint")
            report["checks"]["navigationHint"] = quiet_hint(state)
            rail = next(r for r in state["records"] if r["key"] == "activity-rail")
            button = ui.hits["rail-Settings"]
            frame = rail["hit"]
            bottom_gap = frame[1] + frame[3] - button[1] - button[3]
            epsilon = native_epsilon(*frame, *button)
            left_gap = button[0] - frame[0]
            viewport = imgui.get_main_viewport()
            window_bottom_gap = viewport.pos.y + viewport.size.y - button[1] - button[3]
            assert abs(bottom_gap - left_gap) <= epsilon, (rail, button)
            assert abs(window_bottom_gap - left_gap) <= epsilon, (rail, button)
            assert abs(rail["outerRadius"] - rail["cornerInset"] - rail["itemRadius"]) <= epsilon
            report["checks"]["navigationInsets"] = {
                "rail": rail,
                "settingsButton": button,
                "bottomGap": bottom_gap,
                "windowBottomGap": window_bottom_gap,
                "leftGap": left_gap,
            }
            inputs.move((0, 0))

            inputs.hover("scene-geometry-collision")
            state = save(inputs, "scene-collision-hint", panels=("Scene",))
            report["checks"]["sceneHint"] = quiet_hint(state)
            inputs.move((0, 0))
            inputs.hover("tool-move")
            state = save(inputs, "viewport-move-hint", panels=("Viewport",))
            report["checks"]["moveHint"] = quiet_hint(state, shortcut=True)
            inputs.move((0, 0))

            geometry = next((node for node in session.nodes if node.name == "geom9"), None)
            assert geometry is not None, "The production review requires the joint_gizmo asset"
            result = session.submit(cmd.Pause())
            assert result.ok, result.message
            result = session.submit(cmd.SelectNode(geometry.node_id))
            assert result.ok, result.message
            workspace.inspector_tab = "Physics"
            with _floating_inspector(inputs) as width:
                body = next(node for node in session.nodes if node.name == "04_free")
                result = session.submit(cmd.SelectNode(body.node_id))
                assert result.ok, result.message
                inputs.sync()
                inspector = app.panels.get("inspector")
                for inertia_mode in ("auto", "diagonal"):
                    inspector._body_property_edit = replace(
                        session.body_properties(body.node_id), inertia_mode=inertia_mode
                    )
                    for points, name in ((310, "narrow"), (704, "wide")):
                        width[0] = points
                        inputs.sync()
                        _reveal(inputs, "##body-mocap", "Inspector content")
                        key = f"inspector-body-switch-{inertia_mode}-{name}"
                        state = save(inputs, key, panels=("Inspector",))
                        switch_geometry = body_switch_geometry(state, points=points)
                        state["bodySwitchGeometry"] = switch_geometry
                        report["checks"][key] = switch_geometry
                        flush()
                inspector._body_property_edit = session.body_properties(body.node_id)
                workspace.inspector_tab = "Properties"
                # Floats retain the workspace's 230-point minimum. Smaller
                # content-column cases are covered by the CPU property tests.
                width[0] = 230
                inputs.sync()
                _reveal(inputs, "gizmos", "Inspector content")
                state = save(inputs, "inspector-display-switch-fit", panels=("Inspector",))
                rows = {r["key"]: r for r in state["records"]}
                display_rows = [rows[key] for key in ("entity-visible", "gizmos")]
                for row, caption in zip(
                    display_rows, ("Visible", "Transform handles"), strict=True
                ):
                    label, track = row["labelRect"], row["switchRect"]
                    epsilon = native_epsilon(*label, *track)
                    assert row["textPresent"] and row["label"] == ui.translate(caption), row
                    assert label[0] + label[2] <= track[0] - row["gap"] + epsilon, row
                    assert abs(label[1] + label[3] / 2 - track[1] - track[3] / 2) <= epsilon, row
                    assert any(
                        item["text"] == ui.translate(caption)
                        and item["windowId"] == row["windowId"]
                        and row["hit"][1] <= item["position"][1] < row["hit"][1] + row["hit"][3]
                        for item in state["emittedText"]
                    ), row
                epsilon = native_epsilon(*(v for row in display_rows for v in row["hit"]))
                assert (
                    abs(display_rows[0]["labelRect"][0] - display_rows[1]["labelRect"][0])
                    <= epsilon
                )
                assert (
                    abs(
                        sum(display_rows[0]["switchRect"][::2])
                        - sum(display_rows[1]["switchRect"][::2])
                    )
                    <= epsilon
                )
                report["checks"]["displaySwitchFit"] = {
                    "panelWidthPoints": width[0],
                    "visible": rows["entity-visible"],
                    "handles": rows["gizmos"],
                }
                for kind, key, row_key, caption in (
                    (
                        NodeType.LIGHT,
                        "##light_cast_shadow",
                        "insp_light_properties/row/cast shadow",
                        "cast shadow",
                    ),
                    (
                        NodeType.CAMERA,
                        "##camera-gizmo-lock",
                        "insp_camera_behavior/row/gizmo lock",
                        "gizmo lock",
                    ),
                ):
                    node = next(node for node in session.nodes if node.type is kind)
                    result = session.submit(cmd.SelectNode(node.node_id))
                    assert result.ok, result.message
                    width[0] = 210
                    inputs.sync()
                    _reveal(inputs, key, "Inspector content")
                    state = save(
                        inputs, "inspector-" + kind.value + "-switch-fit", panels=("Inspector",)
                    )
                    report["checks"][kind.value + "SwitchFit"] = inline_boolean_geometry(
                        state, row_key, key, caption
                    )
                result = session.submit(cmd.SelectNode(geometry.node_id))
                assert result.ok, result.message
                workspace.inspector_tab = "Physics"
                inputs.sync()
                for points, name in ((560, "wide"), (310, "narrow")):
                    width[0] = points
                    inputs.sync()
                    _reveal(inputs, "##contact_friction-0-frame", "Inspector content")
                    state = save(inputs, "inspector-friction-" + name, panels=("Inspector",))
                    fields = [ui.hits[f"##contact_friction-{index}-frame"] for index in range(3)]
                    following = ui.hits["property-label-contact dimension"]
                    gap = following[1] - max(rect[1] + rect[3] for rect in fields)
                    epsilon = native_epsilon(*following, *(v for rect in fields for v in rect))
                    assert -epsilon <= gap <= 2 * DESIGN.row_gap * ui.s + epsilon, (gap, fields)
                    state["frictionGeometry"] = {
                        "panelWidthPoints": points,
                        "fields": fields,
                        "followingLabel": following,
                        "gap": gap,
                    }
                    report["checks"]["friction-" + name] = state["frictionGeometry"]
                width[0] = 560
                workspace.inspector_tab = "Material"
                inputs.sync()
                _reveal(inputs, "material-color", "Inspector content")
                before = np.array(app.entity["rgba"], copy=True)
                inputs.click("material-color")
                for mode, label in ((0, "hex"), (1, "rgba255"), (2, "rgba1")):
                    inputs.click("color-editor-format-" + str(mode))
                    inputs.move((0, 0))
                    state = save(inputs, "material-color-" + label, panels=("Inspector",))
                    report["checks"]["material-color-" + label] = colors(
                        state, "color-editor", 4, mode
                    )
                sv = ui.hits["color-editor-sv"]
                inputs.click_at((sv[0] + sv[2] * 0.25, sv[1] + sv[3] * 0.3))
                assert not np.allclose(before, app.entity["rgba"]), (
                    "Picker must preview a real edit"
                )
                save(inputs, "material-color-preview", panels=("Inspector",))
                inputs.escape()
                assert workspace.menus.color_edit is None
                assert np.allclose(before, app.entity["rgba"]), (
                    "Escape must restore the original color"
                )
                save(inputs, "material-color-canceled", panels=("Inspector",))
                report["checks"]["materialCancel"] = True

            workspace.activate("Settings")
            workspace.settings.show_category("Recording")
            inputs.sync()
            with _floating_panel(inputs, "Settings") as width:
                width[0] = 244
                inputs.sync()
                _reveal(inputs, "##capture_clipboard", "Reference settings content")
                state = save(inputs, "settings-checkbox-fit", panels=("Settings",))
                caption = workspace._panel_context.tr("Copy to clipboard")
                report["checks"]["settingsCheckboxFit"] = inline_boolean_geometry(
                    state, "settings_recording/row/" + caption, "##capture_clipboard", caption
                )
            _reveal(inputs, "##recording_surface", "Reference settings content")
            inputs.click("##recording_surface")
            inputs.move((0, 0))
            state = save(inputs, "recording-choice-idle", panels=("Settings",))
            choices = [
                r for r in state["records"] if r["key"].startswith("##recording_surface/choice/")
            ]
            assert len(choices) == 3, choices
            selected_index = tuple(CaptureSurface).index(app.recording_config.surface)
            selected_key = f"##recording_surface/choice/{selected_index}"
            other_key = f"##recording_surface/choice/{(selected_index + 1) % 3}"
            assert {row["key"] for row in choices if row["background"]} == {selected_key}, choices
            report["checks"]["recording-choice-idle"] = {
                "selectedIndex": selected_index,
                "rows": choices,
            }
            for key, label in ((other_key, "neutral-hover"), (selected_key, "selected-hover")):
                inputs.hover(key)
                state = save(inputs, "recording-choice-" + label, panels=("Settings",))
                hovered = next(r for r in state["records"] if r["key"] == key)
                assert hovered["hovered"] and hovered["background"], hovered
                assert not hovered["clippedLabel"], hovered
                with Image.open(output / state["image"]) as image:
                    display = imgui.get_io().display_size
                    frame = hovered["backgroundRect"]
                    point = (
                        round((frame[0] + frame[2] - 6 * ui.s) * image.width / display.x),
                        round((frame[1] + frame[3] / 2) * image.height / display.y),
                    )
                    observed_ink = image.convert("RGB").getpixel(point)
                expected_ink = ui.ink(rgba8("field_hovered"), ui.component_style("select_option"))
                assert (
                    max(abs(a - b) for a, b in zip(observed_ink, expected_ink[:3], strict=True))
                    <= 3
                ), (
                    observed_ink,
                    expected_ink,
                )
                report["checks"]["recording-choice-" + label] = {
                    "row": hovered,
                    "selectedIndex": selected_index,
                    "sampledBackground": observed_ink,
                    "expectedBackground": expected_ink,
                }
            inputs.escape()
            workspace.settings.show_category("MuJoCo Visuals")
            inputs.sync()
            settings_content = _content_window("Reference settings content")
            imgui.internal.set_scroll_y(settings_content, 0)
            inputs.sync()
            inputs.hover("settings-geometry-collision")
            state = save(inputs, "settings-geometry-hint", panels=("Settings",))
            report["checks"]["settingsHint"] = quiet_hint(state)
            inputs.move((0, 0))
            contact_before = app.backend.get_contact_style()
            if contact_before.use_model_color:
                _reveal(inputs, "##contact_model_color", "Reference settings content")
                inputs.click("##contact_model_color")
                assert not app.backend.get_contact_style().use_model_color
            _reveal(inputs, "##contact_color-swatch", "Reference settings content")
            before = tuple(app.backend.get_contact_style().color)
            inputs.click("##contact_color-swatch")
            for mode, label in ((0, "hex"), (1, "rgba255"), (2, "rgba1")):
                inputs.click("##contact_color-format-" + str(mode))
                inputs.move((0, 0))
                state = save(inputs, "settings-color-" + label, panels=("Settings",))
                report["checks"]["settings-color-" + label] = colors(
                    state, "##contact_color", 4, mode
                )
            sv = ui.hits["##contact_color-sv"]
            inputs.click_at((sv[0] + sv[2] * 0.65, sv[1] + sv[3] * 0.3))
            assert not np.allclose(before, app.backend.get_contact_style().color)
            save(inputs, "settings-color-preview", panels=("Settings",))
            inputs.click("##contact_color-cancel")
            assert np.allclose(before, app.backend.get_contact_style().color)
            save(inputs, "settings-color-canceled", panels=("Settings",))
            app.set_contact_style(contact_before, persist=False)
            report["checks"]["settingsCancel"] = True
            workspace.open["Settings"] = False
            inputs.escape()
            flush()

        from mojive.tools.native_timeline_review import capture_timeline_regressions

        report["timeline"] = capture_timeline_regressions(viewer, output)
        report["passed"] = True
        flush()
        return report
    except Exception as error:
        report["passed"] = False
        report["failure"] = f"{type(error).__name__}: {error}"
        flush()
        raise
    finally:
        runtime.observe_components = observing
