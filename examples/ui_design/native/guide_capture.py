"""Capture actual popup, hover and slice interactions for the guide's Make target."""

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from imgui_bundle import imgui as g
from PIL import Image

from mojive.ui.icons import ICON_FAMILIES


def _camera_content_window():
    return next(
        window
        for window in g.get_current_context().windows
        if window.active
        and "Native specimen/Cameras" in window.name
        and "/Panel content" in window.name
    )


def _reveal_camera_control(v, key):
    """Reveal through both the specimen scroll and its clipped Guide page."""
    trace = []
    for _ in range(16):
        rect = v.ui.hits[key]
        window = _camera_content_window()
        entry = {"key": key, "rect": rect, "windows": []}
        ancestor = window
        while ancestor is not None:
            entry["windows"].append(
                {
                    "name": ancestor.name,
                    "inner": [ancestor.inner_rect.min.y, ancestor.inner_rect.max.y],
                    "clip": [ancestor.inner_clip_rect.min.y, ancestor.inner_clip_rect.max.y],
                    "scroll": ancestor.scroll.y,
                    "max": ancestor.scroll_max.y,
                    "target": ancestor.scroll_target.y,
                }
            )
            ancestor = ancestor.parent_window
        trace.append(entry)
        top, bottom = window.inner_clip_rect.min.y, window.inner_clip_rect.max.y
        if top <= rect[1] and rect[1] + rect[3] <= bottom:
            return rect
        while window is not None:
            top, bottom = window.inner_clip_rect.min.y, window.inner_clip_rect.max.y
            if top <= rect[1] and rect[1] + rect[3] <= bottom:
                window = window.parent_window
                continue
            delta = min(0, rect[1] - top - 12 * v.s) + max(0, rect[1] + rect[3] - bottom + 12 * v.s)
            target = min(window.scroll_max.y, max(0, window.scroll.y + delta))
            if abs(target - window.scroll.y) > 0.01:
                entry["requested"] = {"window": window.name, "delta": delta, "target": target}
                previous_scroll, previous_rect = window.scroll.y, rect
                g.internal.set_scroll_y(window, target)
                for _ in range(4):
                    v.frame()
                rect = v.ui.hits[key]
                # Native scrolling rounds/clamps its target. A fractional
                # unreachable remainder must not keep this ancestor selected.
                if window.scroll.y != previous_scroll or rect != previous_rect:
                    break
            window = window.parent_window
        else:
            (v.output / "camera-reveal-failure.json").write_text(json.dumps(trace, indent=2) + "\n")
            raise AssertionError("Camera control cannot be revealed: " + key)
    (v.output / "camera-reveal-failure.json").write_text(json.dumps(trace, indent=2) + "\n")
    raise AssertionError("Camera scroll did not settle: " + key)


def capture_interactions(v, *, context_only=False):
    queued = []
    process = v.window._input.process_inputs
    previous_animation = v.animate
    previous_page = v.page
    previous_values, previous_selected = deepcopy(v.values), deepcopy(v.selected)
    v.animate = False

    def inputs():
        process()
        for method, args in queued:
            getattr(g.get_io(), method)(*args)
        queued.clear()
        g.get_io().delta_time = 1 / 60

    def event(method, *args):
        queued.append((method, args))
        v.frame()

    def hover(rect):
        x, y, width, height = rect
        event("add_mouse_pos_event", x + width / 2, y + height / 2)

    def click(rect):
        hover(rect)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        for _ in range(3):
            v.frame()

    def save(name, frames=4):
        for _ in range(frames):
            pixels = v.frame(readback=True)
        Image.fromarray(pixels[::-1]).save(v.output / (name + ".png"))

    def page(name):
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        v.page = name
        event("add_mouse_pos_event", 0, 0)
        for _ in range(4):
            v.frame()

    v.window._input.process_inputs = inputs
    try:
        if context_only:
            _capture_context_updates(v, page, click, save, event)
            return
        page("Viewer layout")
        v.viewer_group = "Header actions"
        v.frame()
        click(v.hits["viewer-group"])
        hover(v.ui.hits["viewer-group/choice/4"])
        save("viewer-popup-hover")
        page("Controls")
        hover(v.ui.hits["guide-switch-0"])
        save("switch-hover")
        click(v.hits["control-select"])
        hover(v.ui.hits["control-select/choice/1"])
        save("select-hover")
        page("Menus")
        click(v.hits["sample-menu-View"])
        save("menu-open")
        page("Viewport")
        click(v.hits["playback-toggle"])
        save("viewport-playing")
        click(v.hits["playback-toggle"])
        page("Output")
        click(v.ui.hits["output-search"])
        save("output-search")
        page("Inspector")
        v.selected["Inspector tabs"] = 0
        for size in ("Wide", "Medium", "Narrow"):
            v.values["inspector-size"] = size
            save("inspector-" + size.lower())
        v.values["inspector-size"] = "Wide"
        for index, name in enumerate(("properties", "physics", "material", "joint")):
            v.selected["Inspector tabs"] = index
            save("inspector-tab-" + name)
        page("Settings")
        from mojive.ui.workspace.settings_panel import CATEGORIES

        for index, name in enumerate(CATEGORIES):
            v.selected["Settings navigation"] = index
            save("settings-" + name.lower().replace(" ", "-"))
        page("Icons")
        for family, _entries in ICON_FAMILIES:
            v.values["icon-family"] = family
            save("icons-" + family.lower().replace(" & ", "-").replace(" ", "-"))
        page("Status")
        for context in ("camera", "ready", "perturb"):
            v.values["status-hint-context"] = context
            save("status-hints-" + context)
        metric = v.status_layout.metrics[-1]
        left, top, right, bottom = metric.rect
        hover((left, top, right - left, bottom - top))
        save("status-tooltip", frames=45)
        _capture_context_updates(v, page, click, save, event)
    finally:
        v.window._input.process_inputs = process
        v.animate = previous_animation
        v.page = previous_page
        v.values.clear()
        v.values.update(previous_values)
        v.selected.clear()
        v.selected.update(previous_selected)


def _capture_context_updates(v, page, click, save, event):
    from mojive import commands as cmd
    from mojive.render.geometry import geometry_view, set_geometry_view
    from mojive.scene.state import apply_camera_bookmark, camera_bookmark, list_named_snapshots
    from mojive.ui.camera import PRESETS
    from mojive.ui.workspace.geometry_view import GEOMETRY_VIEWS
    from mojive.ui.workspace.settings_panel import CATEGORIES

    app, session = v.sample.app, v.sample.app.session
    panel = app.panels.get("camera")
    sensors = app.panels.get("Sensors")
    original_sensor = sensors.sensor_index
    original_selection = (
        session.selected_node.node_id if session.selected_node is not None else None
    )
    original_camera = camera_bookmark(
        app.camera, app.camera.view(), v.sample.context.model_camera_id
    )
    original_geometry = geometry_view(app.backend)
    original_bookmarks = panel.bookmark_directory
    original_name, original_index, original_error = (
        panel._bookmark_name,
        panel._bookmark_index,
        panel._bookmark_error,
    )
    report = {"directions": [], "geometry": [], "bookmarks": [], "sensors": []}
    owned_transaction = False

    def reveal(key):
        return _reveal_camera_control(v, key)

    def input_name(name):
        click(reveal("camera-bookmark-name"))
        modifier = g.Key.mod_super if g.get_io().config_mac_osx_behaviors else g.Key.mod_ctrl
        event("add_key_event", modifier, True)
        event("add_key_event", g.Key.a, True)
        event("add_key_event", g.Key.a, False)
        event("add_key_event", modifier, False)
        event("add_input_characters_utf8", name)

    try:
        page("Cameras")
        v.values["camera-size"] = "Wide"
        v.frame()
        camera_window = _camera_content_window()
        original_scroll = camera_window.scroll.y
        g.internal.set_scroll_y(camera_window, 0)
        v.frame()
        for name in ("Front", "Back", "Left", "Right", "Top", "Bottom", "ISO"):
            click(reveal("camera-" + name))
            app.camera.advance(1.0, app.camera_out)
            expected = PRESETS[name.lower()]
            actual = (app.camera.yaw, app.camera.pitch)
            yaw_error = abs((actual[0] - expected[0] + 180) % 360 - 180)
            if yaw_error > 1e-4 or abs(actual[1] - expected[1]) > 1e-4:
                raise AssertionError((name, actual, expected))
            report["directions"].append({"name": name, "yaw": actual[0], "pitch": actual[1]})
        save("camera-directions")
        with TemporaryDirectory(prefix="camera-capture-", dir=v.output) as directory:
            panel.bookmark_directory = Path(directory)
            panel._bookmark_name = ""
            v.frame()
            heading = "section-" + v.tr("camera bookmarks")
            reveal(heading)
            opened = "camera-bookmark-name" in v.ui.hits
            if not opened:
                click(v.ui.hits[heading])
            reveal("camera-bookmark-name")
            camera_window = _camera_content_window()
            g.internal.set_scroll_y(camera_window, camera_window.scroll_max.y)
            save("camera-bookmarks-empty")
            for direction in ("Front", "ISO"):
                app.camera.set_preset(direction.lower())
                app.camera.advance(1.0, app.camera_out)
                name = v.tr(direction)
                input_name(name)
                click(reveal("camera-bookmark-save"))
                if name not in list_named_snapshots(panel.bookmark_directory):
                    raise AssertionError("Camera bookmark was not saved: " + name)
                report["bookmarks"].append(name)
            reveal("camera-bookmark-open/" + report["bookmarks"][-1])
            save("camera-bookmarks-populated")
            key = "camera-bookmark-actions/" + report["bookmarks"][0]
            click(reveal(key))
            save("camera-bookmarks-menu")
            event("add_key_event", g.Key.escape, True)
            event("add_key_event", g.Key.escape, False)
            v.values["camera-size"] = "Narrow"
            v.frame()
            reveal("camera-bookmark-open/" + report["bookmarks"][-1])
            save("camera-bookmarks-narrow")
            if not opened:
                click(reveal(heading))
        g.internal.set_scroll_y(camera_window, original_scroll)
        page("Menus")
        click(v.hits["sample-menu-Add"])
        save("menu-add")

        page("Inspector")
        joint_node = next(node for node in session.nodes if node.name == "hinge_limited")
        if not session.submit(cmd.SelectNode(joint_node.node_id)).ok:
            raise AssertionError("Could not select the sample hinge joint")
        v.selected["Inspector tabs"] = 3
        v.values["inspector-size"] = "Wide"
        v.frame()
        joint = next(joint for joint in session.joints if joint.name == "hinge_limited")
        if not session.submit(cmd.BeginEditTransaction("Guide joint limit capture")).ok:
            raise AssertionError("Could not begin the sample joint edit")
        owned_transaction = True
        for limited in (False, True):
            result = session.submit(
                cmd.SetJointProperties(
                    joint.joint_id, joint.axis, limited, joint.range, joint.damping, joint.stiffness
                )
            )
            if not result.ok:
                raise AssertionError(result.message)
            save("joint-limited" if limited else "joint-unlimited")
            current = next(item for item in session.joints if item.name == "hinge_limited")
            if current.limited != limited:
                raise AssertionError("Sample joint limit change was not applied")
        result = session.submit(cmd.CancelEditTransaction())
        owned_transaction = False
        if not result.ok:
            raise AssertionError(result.message)

        color_node = next(node for node in session.nodes if node.name == "free_box")
        if not session.submit(cmd.SelectNode(color_node.node_id)).ok:
            raise AssertionError("Could not select the sample geometry")
        v.selected["Inspector tabs"] = 2
        v.frame()
        original_color = list(app.projection.entity["rgba"])
        click(v.ui.hits["material-color"])
        try:
            click(v.ui.hits["color-editor-picker"])
            if list(app.projection.entity["rgba"]) == original_color:
                raise AssertionError("Color picker interaction did not change the sample")
            save("color-picker-edited")
            click(v.ui.hits["color-editor-original"])
            restored = list(app.projection.entity["rgba"])
            if any(abs(a - b) > 1e-6 for a, b in zip(restored, original_color, strict=True)):
                raise AssertionError("Original color did not restore the sample")
            if not session.editing:
                raise AssertionError("Restoring the original color ended its edit gesture")
            save("color-picker-restored")
        finally:
            event("add_key_event", g.Key.escape, True)
            event("add_key_event", g.Key.escape, False)
            v.frame()

        body = next(node for node in session.nodes if node.name == "hinge_body")
        if not session.submit(cmd.SelectNode(body.node_id)).ok:
            raise AssertionError("Could not select the sample body")
        v.selected["Inspector tabs"] = 1
        save("inertia-automatic")
        for field in ("mass", "position", "rotation", "inertia"):
            if "##body-derived-" + field not in v.ui.hits:
                raise AssertionError("Automatic inertia field is missing: " + field)

        page("Sensors")
        for name, suffix in (
            ("hinge_position_xyz", "xyz"),
            ("component_readings", "components"),
        ):
            sensors.sensor_index = next(
                index for index, sensor in enumerate(session.sensor_infos) if sensor.name == name
            )
            save("sensors-" + suffix)
            info = session.sensor_infos[sensors.sensor_index]
            if any(
                f"sensor-value-{info.sensor_id}-{index}" not in v.ui.hits
                for index in range(info.dim)
            ):
                raise AssertionError("Sensor components are not fully visible: " + name)
            report["sensors"].append({"name": name, "dimension": info.dim})

        set_geometry_view(app.backend, GEOMETRY_VIEWS[-1])
        for mode in GEOMETRY_VIEWS:
            page("Settings")
            v.selected["Settings navigation"] = CATEGORIES.index("MuJoCo Visuals")
            v.frame()
            if any("settings-geometry-" + view.value not in v.ui.hits for view in GEOMETRY_VIEWS):
                raise AssertionError("Content geometry choices are incomplete")
            click(v.ui.hits["settings-geometry-" + mode.value])
            if geometry_view(app.backend) != mode:
                raise AssertionError("Content geometry selection did not reach the backend")
            save("content-geometry-" + mode.value)
            page("Scene")
            save("scene-geometry-" + mode.value)
            if any("scene-geometry-" + view.value not in v.ui.hits for view in GEOMETRY_VIEWS):
                raise AssertionError("Scene geometry choices are incomplete")
            if geometry_view(app.backend) != mode:
                raise AssertionError("Scene and content geometry selections disagree")
            report["geometry"].append(mode.value)
        (v.output / "context-interactions.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if owned_transaction:
            session.submit(cmd.CancelEditTransaction())
        result = session.submit(
            cmd.SelectNode(original_selection) if original_selection is not None else cmd.Select(0)
        )
        if not result.ok:
            raise AssertionError(result.message)
        apply_camera_bookmark(original_camera, app.camera, v.sample.context.select_model_camera)
        set_geometry_view(app.backend, original_geometry)
        sensors.sensor_index = original_sensor
        panel.bookmark_directory = original_bookmarks
        panel._bookmark_name, panel._bookmark_index, panel._bookmark_error = (
            original_name,
            original_index,
            original_error,
        )


def capture_telemetry(v):
    """Advance the isolated export session and capture actual telemetry at three widths."""
    from mojive import commands as cmd

    app, session = v.sample.app, v.sample.app.session
    v.animate = False

    def save(name):
        for _ in range(4):
            pixels = v.frame(readback=True)
        Image.fromarray(pixels[::-1]).save(v.output / (name + ".png"))

    v.page = "Sensors"
    panel = app.panels.get("Sensors")
    for name, suffix in (
        ("hinge_position", "scalar"),
        ("hinge_position_xyz", "xyz"),
        ("component_readings", "components"),
    ):
        panel.sensor_index = next(
            index for index, info in enumerate(session.sensor_infos) if info.name == name
        )
        for size in ("Wide", "Narrow"):
            v.values["sensor-size"] = size
            save("sensor-" + suffix + "-" + size.lower())
        if suffix == "components":
            window = next(
                window
                for window in g.get_current_context().windows
                if window.active
                and "Native specimen/Sensors" in window.name
                and "/Panel content" in window.name
            )
            ancestors = []
            ancestor = window.parent_window
            while ancestor is not None:
                ancestors.append((ancestor, ancestor.scroll.y))
                g.internal.set_scroll_y(ancestor, ancestor.scroll_max.y)
                ancestor = ancestor.parent_window
            for _ in range(4):
                v.frame()
            g.internal.set_scroll_y(window, window.scroll_max.y)
            save("sensor-components-narrow-scrolled")
            info = session.sensor_infos[panel.sensor_index]
            key = f"sensor-value-{info.sensor_id}-{info.dim - 1}"
            rect = v.ui.hits.get(key)
            clip = window.inner_clip_rect
            tail = {
                "key": key,
                "rect": rect,
                "clip": [clip.min.x, clip.min.y, clip.max.x, clip.max.y],
                "windows": [
                    {
                        "name": item.name,
                        "inner": [item.inner_rect.min.y, item.inner_rect.max.y],
                        "clip": [item.inner_clip_rect.min.y, item.inner_clip_rect.max.y],
                        "scroll": item.scroll.y,
                        "max": item.scroll_max.y,
                        "content_max_y": item.dc.cursor_max_pos.y,
                    }
                    for item in (window, *(item for item, _value in ancestors))
                ],
            }
            (v.output / "sensor-scroll-geometry.json").write_text(json.dumps(tail, indent=2) + "\n")
            if rect is None:
                raise AssertionError("The last sensor component cannot be reached by scrolling")
            if not (
                clip.min.x <= rect[0]
                and clip.min.y <= rect[1]
                and rect[0] + rect[2] <= clip.max.x
                and rect[1] + rect[3] <= clip.max.y
            ):
                raise AssertionError("The last sensor component is clipped after scrolling")
            g.internal.set_scroll_y(window, 0)
            for ancestor, original_scroll in ancestors:
                g.internal.set_scroll_y(ancestor, original_scroll)
            for _ in range(4):
                v.frame()

    v.page = "Curves"
    v.values["curve-size"] = "Wide"
    plot = app.panels.get("Plot")
    plot.source = "joint"
    plot.joint_index = next(
        index for index, joint in enumerate(session.joints) if joint.name == "hinge_limited"
    )
    v.frame()
    plot._angle.clear()
    plot._velocity.clear()
    save("curves-single-sample")
    actuator = next(item for item in session.actuators if item.name == "hinge_pos")
    result = session.submit(cmd.SetCtrl(actuator.ctrl_address, 0.7))
    if not result.ok:
        raise AssertionError(result.message)
    for _ in range(80):
        result = session.submit(cmd.Step(count=5))
        if not result.ok:
            raise AssertionError(result.message)
        v.frame()
    if plot._angle.filled < 80 or plot._angle.duration <= 0:
        raise AssertionError("Telemetry capture requires real advancing samples")
    samples = plot._angle.data[: plot._angle.filled]
    if max(samples) - min(samples) < 0.1:
        raise AssertionError("Telemetry capture requires visible joint motion")
    for size in ("Wide", "Medium", "Narrow"):
        v.values["curve-size"] = size
        save("curves-sampled-" + size.lower())
    sample_count, duration = plot._angle.filled, plot._angle.duration
    _capture_telemetry_controls(v, plot, save)
    (v.output / "telemetry-interactions.json").write_text(
        json.dumps(
            {
                "samples": sample_count,
                "duration": duration,
                "component": "Hinge angle",
                "actions": ["sensor-plot", "source-selection", "channel-visibility"],
            },
            indent=2,
        )
        + "\n"
    )


def _capture_telemetry_controls(v, plot, save):
    queued = []
    process = v.window._input.process_inputs

    def inputs():
        process()
        for method, args in queued:
            getattr(g.get_io(), method)(*args)
        queued.clear()

    def event(method, *args):
        queued.append((method, args))
        v.frame()

    def click(key):
        x, y, width, height = v.ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        for _ in range(3):
            v.frame()

    v.window._input.process_inputs = inputs
    try:
        v.page = "Sensors"
        v.values["sensor-size"] = "Wide"
        panel = v.sample.app.panels.get("Sensors")
        panel.sensor_index = 1
        # An inactive narrow child retains its previous overflow for one frame.
        # Let the native scrollbar budget settle before caching a click target.
        for _ in range(4):
            v.frame()
        click("sensor-plot")
        if plot.source != "sensor" or plot.sensor_index != panel.sensor_index:
            raise AssertionError("Sensor plot action did not preserve the selected sensor")
        v.page = "Curves"
        v.values["curve-size"] = "Wide"
        save("curves-sensor-link")
        click("plot-source-0")
        if plot.source != "joint":
            raise AssertionError("Curve source choice did not select joints")
        initial = plot.show_velocity
        click("plot-channels")
        save("curves-channels")
        click("plot-channels/choice/1")
        if plot.show_velocity == initial:
            raise AssertionError("Curve visibility choice did not toggle velocity")
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        click("plot-channels")
        click("plot-channels/choice/1")
        if plot.show_velocity != initial:
            raise AssertionError("Curve visibility choice did not restore velocity")
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
    finally:
        v.window._input.process_inputs = process
