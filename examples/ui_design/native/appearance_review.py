"""Capture native editor pages and check their visible control geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, NodeType
from mojive.ui.viewport_widgets import localized_viewport_labels

from .settings_panel import CATEGORIES


def _window_rect(name):
    window = imgui.internal.find_window_by_name(name)
    assert window is not None and window.active, ("Review window is not visible", name)
    return [window.pos.x, window.pos.y, window.size.x, window.size.y]


def _check_controls(controls, owner, *, tolerance=1.0):
    """Check one peer group; parent controls are checked separately."""
    x, y, w, h = owner
    checked = []
    for key, rect in controls.items():
        cx, cy, cw, ch = rect
        assert cw > 0 and ch > 0, ("Empty control", key, rect)
        assert (
            cx >= x - tolerance
            and cy >= y - tolerance
            and cx + cw <= x + w + tolerance
            and cy + ch <= y + h + tolerance
        ), ("Control exceeds its window", key, rect, owner)
        for other_key, (ox, oy, ow, oh) in checked:
            assert (
                cx + cw <= ox + tolerance
                or ox + ow <= cx + tolerance
                or cy + ch <= oy + tolerance
                or oy + oh <= cy + tolerance
            ), ("Peer controls overlap", key, other_key, rect, (ox, oy, ow, oh))
        checked.append((key, rect))
    return len(checked)


def _settings_size(study, width, height):
    window = imgui.internal.find_window_by_name("Settings")
    assert window is not None and window.dock_id == 0, "Settings must float for size review"
    viewport = imgui.get_main_viewport()
    s = study.s
    width = min(width * s, viewport.size.x - 32 * s)
    height = min(height * s, viewport.size.y - 32 * s)
    imgui.internal.set_window_size(window, (width, height))
    imgui.internal.set_window_pos(
        window,
        (
            viewport.pos.x + (viewport.size.x - width) / 2,
            viewport.pos.y + (viewport.size.y - height) / 2,
        ),
    )


def _inspector_size(study, width, height):
    window = imgui.internal.find_window_by_name("Inspector")
    assert window is not None and window.dock_id == 0
    viewport = imgui.get_main_viewport()
    s = study.s
    size = (
        min(width * s, viewport.work_size.x - 32 * s),
        min(height * s, viewport.work_size.y - 32 * s),
    )
    imgui.internal.set_window_size(window, size)
    imgui.internal.set_window_pos(
        window,
        (
            viewport.work_pos.x + (viewport.work_size.x - size[0]) / 2,
            viewport.work_pos.y + (viewport.work_size.y - size[1]) / 2,
        ),
    )


def _check_inertial_groups(study, controls, node_id):
    groups = []
    names = ("body_inertial_position", "body_inertial_rotation", "body_diagonal_inertia")
    for name in names:
        fields = [controls[f"##{name}_{index}_{node_id}-frame"] for index in range(3)]
        label = controls["vector-label-" + name]
        width = fields[0][2]
        for field in fields:
            assert abs(field[2] - width) <= study.s, ("Unequal vector field widths", name, fields)
        geometry = [(x - fields[0][0], y - fields[0][1], w, h) for x, y, w, h in fields]
        groups.append(geometry)
        above = fields[0][1] >= label[1] + label[3]
        assert above == (fields[0][0] <= label[0] + study.s), (
            "Vector label and field placement disagree",
            name,
            label,
            fields,
        )
    assert np.allclose(groups[0], groups[1], atol=study.s)
    assert np.allclose(groups[0], groups[2], atol=study.s)
    starts = [controls[f"##{name}_0_{node_id}-frame"][0] for name in names]
    assert np.allclose(starts, controls["##body-mass-frame"][0], atol=study.s), (
        "Scalar and vector fields must start at the same column",
        starts,
        controls["##body-mass-frame"],
    )
    return len(groups)


def capture_appearance(window, study, output, frames):
    """Render real Study pages without retaining scene or preference changes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames(window, study)
    app, session = study.preview, study.preview.session
    original = {
        "layout": imgui.save_ini_settings_to_memory(),
        "open": dict(study.open),
        "workspace": study.workspace,
        "left_active": study.left_active,
        "bottom_active": study.bottom_active,
        "search_visible": study.search_visible,
        "pose_search": study.navigation.joints._search,
        "pose_search_visible": study.navigation.pose_search_visible,
        "inspector_tab": study.inspector_tab,
        "pinned": study.pinned,
        "tool": app.tool,
        "category": study.settings._category,
        "language": app.localizer.language,
        "viewport_labels": app._viewport_labels,
        "selection": session.selected_node.node_id if session.selected_node is not None else None,
        "node_count": len(session.nodes),
    }
    states = {}
    created = False

    def set_language(language):
        app.localizer.set_language(language, persist=False)
        app._viewport_labels = localized_viewport_labels(app.localizer.text)

    def capture(name, *, inspector_fields=False, settings=False):
        snapshot = frames(window, study, output / f"{name}.png")
        controls = {key: list(rect) for key, rect in study.ui.hits.items()}
        checks = {}
        if inspector_fields:
            entity = app.entity
            numeric = {
                key: controls[key]
                for component in ("position", "rotation", "scale")
                for axis in "XYZ"
                if (key := f"##{entity['id']}{component}{axis}") in controls
            }
            assert len(numeric) == 9, ("Editable transform fields are missing", numeric)
            checks["inspector_fields"] = _check_controls(
                numeric, _window_rect("Inspector"), tolerance=study.s
            )
        if settings:
            navigation = {
                f"settings-{category}": controls[f"settings-{category}"]
                for category in CATEGORIES
                if f"settings-{category}" in controls
            }
            assert len(navigation) == len(CATEGORIES), (
                "Settings categories are missing",
                navigation,
            )
            owner = _window_rect("Settings")
            settings_window = imgui.internal.find_window_by_name("Settings")
            assert settings_window.dock_id == 0, "Settings must remain an independent window"
            assert settings_window.flags & imgui.WindowFlags_.no_title_bar, (
                "Settings must use one content header",
                settings_window.flags,
            )
            header = {key: controls[key] for key in ("settings-title", "settings-close")}
            checks["settings_header"] = _check_controls(header, owner, tolerance=study.s)
            checks["settings_categories"] = _check_controls(navigation, owner, tolerance=study.s)
            _check_controls(header | navigation, owner, tolerance=study.s)
            if study.settings._category == "General":
                workspace = {
                    key: controls[key]
                    for key in ("settings-sidebar", "settings-bottom", "settings-inspector")
                }
                checks["settings_panel_controls"] = _check_controls(
                    workspace, owner, tolerance=study.s
                )
                _check_controls(header | navigation | workspace, owner, tolerance=study.s)
            assert settings_window.scroll_max.y == 0, (
                "Settings must scroll its page, not its header"
            )
            viewport = imgui.get_main_viewport()
            _check_controls(
                {"Settings": owner},
                [viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y],
                tolerance=study.s,
            )
        states[name] = {
            "snapshot": snapshot,
            "language": app.localizer.language.value,
            "controls": controls,
            "checks": checks,
        }

    def capture_inspector_layouts():
        body = next(node for node in session.nodes if node.name == "hinge_body")
        joint = next(node for node in session.nodes if node.name == "hinge_limited")
        docked_layout = imgui.save_ini_settings_to_memory()
        study.command("float")
        frames(window, study)
        try:
            for language, suffix in (("en", "en"), ("zh_CN", "zh")):
                set_language(language)
                for width in (384, 424):
                    _inspector_size(study, width, 820)
                    assert session.submit(cmd.SelectNode(body.node_id)).ok
                    study.inspector_tab = "Properties"
                    frames(window, study, count=3)
                    capture(f"transform-inline-{width}-{suffix}", inspector_fields=True)
                    for component in ("position", "rotation", "scale"):
                        fields = [
                            study.ui.hits[f"##{body.node_id}{component}{axis}-frame"]
                            for axis in "XYZ"
                        ]
                        label = study.ui.hits[f"##{body.node_id}{component}-label"]
                        assert len({round(field[1], 3) for field in fields}) == 1
                        assert fields[0][0] >= label[0] + label[2] + 12 * study.s - 1
                for width, name in ((284, "compact"), (344, "standard"), (744, "wide")):
                    _inspector_size(study, width, 820)
                    for node, tab, kind in ((body, "Physics", "body"), (joint, "Joint", "joint")):
                        result = session.submit(cmd.SelectNode(node.node_id))
                        assert result.ok, result.message
                        study.inspector_tab = tab
                        frames(window, study, count=3)
                        content = next(
                            child
                            for child in imgui.get_current_context().windows
                            if "Inspector content" in str(child.name) and child.active
                        )
                        if kind == "joint":
                            content.state_storage.set_int(
                                content.get_id("native-section-advanced joint properties"), 0
                            )
                        imgui.internal.set_scroll_y(content, 0)
                        state_name = f"{kind}-{tab.lower()}-{name}-{suffix}"
                        capture(state_name)
                        if kind == "body":
                            states[state_name]["checks"]["inertial_groups"] = (
                                _check_inertial_groups(
                                    study, states[state_name]["controls"], node.node_id
                                )
                            )
                        else:
                            content.state_storage.set_int(
                                content.get_id("native-section-advanced joint properties"), 1
                            )
                            frames(window, study, count=3)
                            armature_y = study.ui.hits["##joint_armature"][1]
                            imgui.internal.set_scroll_y(
                                content,
                                content.scroll.y
                                + armature_y
                                - content.inner_clip_rect.min.y
                                - 28 * study.s,
                            )
                            capture(state_name + "-advanced")
                    if name == "standard":
                        free_ids = {
                            joint.joint_id for joint in session.joints if joint.type == "free"
                        }
                        free = next(
                            node
                            for node in session.nodes
                            if node.type is NodeType.JOINT and node.joint_index in free_ids
                        )
                        assert session.submit(cmd.SelectNode(free.node_id)).ok
                        study.inspector_tab = "Joint"
                        frames(window, study, count=3)
                        imgui.internal.set_scroll_y(content, 0)
                        content.state_storage.set_int(
                            content.get_id("native-section-advanced joint properties"), 0
                        )
                        capture(f"free-joint-standard-{suffix}")
        finally:
            imgui.load_ini_settings_from_memory(docked_layout)
            frames(window, study)

    def restore_selection():
        result = session.submit(
            cmd.SelectNode(original["selection"])
            if original["selection"] is not None
            else cmd.Select(0)
        )
        assert result.ok, result.message

    def capture_joint_poses():
        study.activate("Pose")
        study.navigation.pose_search_visible = True
        panel = study.panel
        layout = imgui.save_ini_settings_to_memory()

        def floating_pose(name, ctx):
            if name == "Pose":
                imgui.set_next_window_dock_id(0, imgui.Cond_.always)
                imgui.set_next_window_pos((48 * study.s, 80 * study.s), imgui.Cond_.always)
                imgui.set_next_window_size((744 * study.s, 620 * study.s), imgui.Cond_.always)
            return panel(name, ctx)

        try:
            for wide in (False, True):
                if wide:
                    study.panel = floating_pose
                for language, suffix in (("en", "en"), ("zh_CN", "zh")):
                    set_language(language)
                    for kind in ("ball", "free"):
                        joint = next(joint for joint in session.joints if joint.type == kind)
                        node = next(
                            node
                            for node in session.nodes
                            if node.type is NodeType.JOINT and node.joint_index == joint.joint_id
                        )
                        assert session.submit(cmd.SelectNode(node.node_id)).ok
                        study.navigation.joints._search = joint.name
                        name = f"pose-{kind}-{'wide-' if wide else ''}{suffix}"
                        capture(name)
                        controls = states[name]["controls"]
                        components = ("position", "rotation") if kind == "free" else ("rotation",)
                        fields = {
                            key: controls[key]
                            for component in components
                            for axis in "XYZ"
                            if (key := f"##pose-joint-{joint.joint_id}-{component}{axis}-frame")
                            in controls
                        }
                        assert len(fields) == 3 * len(components), (
                            "Joint pose fields are missing",
                            name,
                        )
                        states[name]["checks"]["pose_fields"] = _check_controls(
                            fields, _window_rect("Pose"), tolerance=study.s
                        )
        finally:
            study.panel = panel
            imgui.load_ini_settings_from_memory(layout)
            frames(window, study)
        study.navigation.joints._search = original["pose_search"]
        study.navigation.pose_search_visible = original["pose_search_visible"]
        restore_selection()
        set_language("en")

    try:
        set_language("en")
        study.open["Settings"] = False
        study.pinned = None
        study.inspector_tab = "Properties"
        study.preset("Edit")
        capture("edit")

        study.menus.dispatch("create:Box")
        created = len(session.nodes) > original["node_count"]
        assert created, ("Temporary box was not created", session.last_message)
        frames(window, study)
        entity = app.entity
        assert entity is not None and entity["posable"] and entity["scalable"], entity
        app.tool = "move"
        capture("inspector-editable", inspector_fields=True)
        study.inspector_tab = "Material"
        capture("inspector-material")
        study.inspector_tab = "Properties"
        result = session.submit(
            cmd.SetPose(
                session.selected_node.node_id,
                np.array([-12345678.125, 0.000123456789, 9876543.25]),
                np.eye(3),
            )
        )
        assert result.ok, result.message
        capture("inspector-long-numbers", inspector_fields=True)
        result = session.submit(cmd.Undo())
        assert result.ok, result.message
        result = session.submit(cmd.Undo())
        assert result.ok, result.message
        created = False
        assert len(session.nodes) == original["node_count"], "Temporary box was not removed"
        restore_selection()
        app.tool = original["tool"]

        camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
        geometry = next(
            node
            for node in session.nodes
            if node.type is NodeType.GEOM and session.geometry_properties(node.node_id) is not None
        )
        body = next(node for node in session.nodes if node.name == "ball_anchor")
        for language, suffix in (("en", "en"), ("zh_CN", "zh")):
            set_language(language)
            result = session.submit(cmd.SelectNode(camera.node_id))
            assert result.ok, result.message
            study.inspector_tab = "Properties"
            capture(f"camera-properties-{suffix}")
            app.camera_preview.set_enabled(True)
            capture(f"camera-preview-{suffix}")
            app.camera_preview.set_enabled(False)
            result = session.submit(cmd.SelectNode(geometry.node_id))
            assert result.ok, result.message
            study.inspector_tab = "Physics"
            capture(f"geometry-physics-{suffix}")
            result = session.submit(cmd.SelectNode(body.node_id))
            assert result.ok, result.message
            capture(f"body-physics-{suffix}")
        capture_inspector_layouts()
        set_language("zh_CN")
        selected_geometry = next(node for node in session.nodes if node.name == "free_box")
        assert session.submit(cmd.SelectNode(selected_geometry.node_id)).ok
        study.activate("Scene")
        frames(window, study, count=3)
        hit = study.ui.hits[f"entity-{selected_geometry.node_id}"]
        imgui.get_io().add_mouse_pos_event(hit[0] + hit[2] / 2, hit[1] + hit[3] / 2)
        capture("scene-selected-hover-zh")
        imgui.get_io().add_mouse_pos_event(-100, -100)
        set_language("en")
        study.inspector_tab = "Properties"
        restore_selection()

        for panel in ("Control", "Assets", "Pose", "Cameras", "Layers", "Sensors"):
            study.activate(panel)
            capture(panel.lower())
            if panel == "Control" and session.actuators:
                key = f"control-target-{session.actuators[0].ctrl_address}"
                x, y, width, height = study.ui.hits[key]
                io = imgui.get_io()
                io.add_mouse_pos_event(x + width / 2, y + height / 2)
                capture("control-hover")
                io.add_mouse_button_event(0, True)
                capture("control-press")
                io.add_mouse_button_event(0, False)
                io.add_mouse_pos_event(-100, -100)
                capture("control-selected")
                restore_selection()
            if panel == "Layers":
                set_language("zh_CN")
                capture("layers-zh")
                set_language("en")
            if panel == "Cameras":
                projection = study.ui.hits["projection"]
                assert (
                    study.ui.hits["camera-frame"][1] >= projection[1] + projection[3] + 12 * study.s
                )
                set_language("zh_CN")
                capture("cameras-zh")
                set_language("en")
        capture_joint_poses()
        study.activate("Scene")
        frames(window, study)
        for panel in ("Output", "Timeline", "Statistics"):
            study.activate(panel)
            capture(panel.lower())
            if panel == "Timeline":
                set_language("zh_CN")
                capture("timeline-zh")
                hit = study.ui.hits["timeline-settings"]
                imgui.get_io().add_mouse_pos_event(hit[0] + hit[2] / 2, hit[1] + hit[3] / 2)
                imgui.get_io().add_mouse_button_event(0, True)
                frames(window, study)
                imgui.get_io().add_mouse_button_event(0, False)
                capture("timeline-settings-zh")
                imgui.get_io().add_key_event(imgui.Key.escape, True)
                frames(window, study)
                imgui.get_io().add_key_event(imgui.Key.escape, False)
                imgui.get_io().add_mouse_pos_event(-100, -100)
                saved_take = session.active_state_take_id
                snapshot = session.submit(cmd.CaptureSceneSnapshot())
                assert snapshot.ok, snapshot.message
                session.set_threaded_physics(False)
                assert session.submit(cmd.StartStateTakeRecording()).ok
                take_id = session.active_state_take_id
                for _ in range(90):
                    session.tick(FrameNeeds.none(), wall_dt=1 / 60)
                assert session.submit(cmd.StopStateTakeRecording()).ok
                session.set_threaded_physics(True)
                editor = study.timeline_panel.editor
                times = session.state_take_times
                index = len(times) // 2
                dt = times[index + 1] - times[index]
                assert session.submit(cmd.SeekStateTake(index - 5)).ok
                frames(window, study, count=3)
                editor.set_follow_mode("off")
                editor.view_start, editor.view_end = times[index] - dt * 8, times[index] + dt * 8
                editor.view_needs_fit = False
                editor.take_selection = (index - 2, index + 2)
                capture("timeline-frame-selection-zh")
                assert "timeline-take-selection" in study.ui.hits
                assert session.submit(cmd.DeleteStateTakeFrames(index - 2, index + 2)).ok
                editor.take_selection = None
                capture("timeline-deleted-frames-zh")
                assert np.isclose(editor.view_end - editor.view_start, dt * 16)
                assert len(session.state_take_segments) == 2
                editor.view_start, editor.view_end = times[0] - dt, times[-1] + dt
                capture("timeline-deleted-overview-zh")
                assert session.submit(cmd.RemoveStateTake(take_id)).ok
                if saved_take >= 0:
                    assert session.submit(cmd.SelectStateTake(saved_take)).ok
                assert session.submit(cmd.RestoreSceneSnapshot(snapshot.entity_id)).ok
                assert session.submit(cmd.RemoveSceneSnapshot(snapshot.entity_id)).ok
                set_language("en")
        study.collapse_bottom()
        frames(window, study)
        for language, suffix in (("en", "en"), ("zh_CN", "zh")):
            set_language(language)
            path = (output / "camera-preview-zh.png").resolve()
            assert path.is_file()
            app.output.publish(
                f"{app.localizer.text('Saved capture to')} {path}",
                duration=60,
                copy_text=str(path),
            )
            capture(f"saved-file-receipt-{suffix}")
            notice = app._status_notice_bounds
            assert notice is not None and notice[3] - notice[1] < 100 * study.s, (
                notice,
                app._viewport_rect,
                app.output.active_status(),
                session.last_message,
                app.model_edits.active,
            )
            states[f"saved-file-receipt-{suffix}"]["checks"]["file_receipt_bounds"] = list(notice)
            app.output.active_status(now=float("inf"))
        study.activate("Settings")
        study.new_panel = "Settings"
        frames(window, study)
        _settings_size(study, 820, 620)
        for language, suffix in (("en", "en"), ("zh_CN", "zh")):
            set_language(language)
            for category in CATEGORIES:
                study.settings.show_category(category)
                slug = category.lower().replace(" ", "-")
                capture(f"settings-{slug}-{suffix}", settings=True)
        study.settings.show_category("General")
        _settings_size(study, 420, 600)
        capture("settings-narrow-zh", settings=True)
        set_language("en")
        capture("settings-narrow-en", settings=True)
        study.settings.show_category("Interaction")
        capture("settings-interaction-narrow-en", settings=True)
        set_language("zh_CN")
        study.settings.show_category("Rendering")
        capture("settings-rendering-narrow-zh", settings=True)
    finally:
        if created:
            result = session.submit(cmd.Undo())
            assert result.ok, result.message
        restore_selection()
        app.localizer.set_language(original["language"], persist=False)
        app._viewport_labels = original["viewport_labels"]
        app.tool = original["tool"]
        study.open.update(original["open"])
        study.workspace = original["workspace"]
        study.left_active = study.active = original["left_active"]
        study.bottom_active = original["bottom_active"]
        study.search_visible = original["search_visible"]
        study.navigation.joints._search = original["pose_search"]
        study.navigation.pose_search_visible = original["pose_search_visible"]
        study.bottom_focus = None
        study.inspector_tab = original["inspector_tab"]
        study.pinned = original["pinned"]
        study.settings.show_category(original["category"])
        imgui.load_ini_settings_from_memory(original["layout"])
        frames(window, study)

    report = {
        "imgui": imgui.get_version(),
        "scale": study.s,
        "states": states,
        "checks": (
            "Viewport overlays, Settings navigation, editable transform bounds and uniform "
            "inertial groups across Inspector widths, ball/free Pose fields, deleted take "
            "coverage and compact saved-file receipts passed."
        ),
        "method": "Real Study pages and Session authoring; language changes remain in memory.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(report["checks"], output)
