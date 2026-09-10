"""Verify deferred geometry previews and native primitive dimension gestures."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .. import commands as cmd
from .. import math3d
from ..composition import build_editor
from ..gizmo import SCREEN_RING_RADIUS, SIZE_PT, GizmoHandle, plane_corners, project, world_scale
from ..model_edits import model_edit_scope
from ..types import MeshShape
from .keyframe_timeline import drag, timeline_point
from .ui_runtime import _activate_panel, _click, _item_center, _park_cursor, _settle


def _dimension_drag(viewer, start, end, handle):
    io = imgui.get_io()
    io.add_mouse_pos_event(*start)
    _settle(viewer, 3)
    assert viewer.app.gizmo.hovered_handle is handle, viewer.app.gizmo.hovered_handle
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using
    io.add_mouse_pos_event(*end)
    _settle(viewer, 2)
    io.add_mouse_button_event(0, False)
    _settle(viewer, 3)


def capture_empty_editor(output: Path) -> dict:
    """Exercise a docked empty timeline and Entity > Create > Plane through native UI."""
    output.mkdir(parents=True, exist_ok=True)
    with build_editor(vsync=False, show_window=False, width=1600, height=1100) as viewer:
        app, session = viewer.app, viewer.session
        _settle(viewer, 6)
        _activate_panel(viewer, "Keyframes")
        panel = viewer.panels.get("Keyframes")
        assert not session.state_take_times
        panel._set_follow_mode("off")
        first, last = panel._view_start, panel._view_end
        span = last - first
        drag(
            viewer,
            timeline_point(viewer, first + span * 0.2),
            timeline_point(viewer, first + span * 0.8),
        )
        assert abs(panel._playhead - (first + span * 0.8)) < span * 0.002
        chosen = panel._playhead
        _park_cursor(viewer)
        _settle(viewer, 4)
        assert panel._playhead == chosen
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "empty-timeline.png")

        app.camera.pivot[:] = (0, 0, 0)
        app.camera.yaw, app.camera.pitch, app.camera.distance = -65, 35, 10
        for method, label in (
            ("begin_menu", "Entity"),
            ("begin_menu", "Create"),
            ("menu_item", "Plane"),
        ):
            _click(viewer, _item_center(viewer, method, app.localizer.text(label)))
        _park_cursor(viewer)
        _settle(viewer, 4)
        assert session.selected_node.name == "plane"
        assert app.model_edits.active and app.model_edits.compatible()
        assert session.adapter.primary.model.ngeom == 0
        before = viewer.capture_array()
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "pending-plane.png")
        _click(
            viewer,
            _item_center(viewer, "button", app.localizer.text("Apply") + "##apply_model_edits"),
        )
        deadline = time.monotonic() + 10
        while app.model_edits.active and time.monotonic() < deadline:
            viewer.sync()
        assert not app.model_edits.active, session.last_message
        _park_cursor(viewer)
        _settle(viewer, 4)
        after = viewer.capture_array()
        difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
        assert difference.mean() < 0.1
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "applied-plane.png")
        assert session.submit(cmd.Undo()).ok
        assert session.adapter.primary.model.ngeom == 0
        assert session.submit(cmd.Redo()).ok
        assert session.adapter.primary.model.ngeom == 1
        return {"playhead": chosen, "apply_mean_error": float(difference.mean())}


def capture(output: Path) -> dict:
    """Exercise the production editor paths and compare scene-only pixels across Apply."""
    output.mkdir(parents=True, exist_ok=True)
    with build_editor(vsync=False, show_window=False, width=1600, height=1100) as viewer:
        app, session = viewer.app, viewer.session
        _settle(viewer, 6)
        assert not app.live_model_updates
        primary = session.adapter.primary
        original = primary.model
        app.camera.pivot[:] = (0, 0, 0.6)
        app.camera.yaw, app.camera.pitch, app.camera.distance = -65, 25, 5
        with model_edit_scope(session, app._intercept_model_edit):
            app._add_scene_object(MeshShape.PLANE, "ground")
            app._add_model_primitive("capsule", "capsule")
            capsule = session.selected_node
            assert session.submit(cmd.SetPose(capsule.node_id, np.array((-0.8, 0, 0.9)), np.eye(3)))
            assert session.submit(cmd.SetGeometrySize(capsule.node_id, np.array((0.3, 0.3, 0.5))))
            assert session.submit(cmd.SetGeometryColor(capsule.node_id, (0.8, 0.55, 0.2, 1)))
            for shape, name, position, size, color in (
                (MeshShape.BOX, "box", (0.6, 0.6, 0.4), (0.4, 0.6, 0.4), (0.3, 0.5, 0.3, 1)),
                (
                    MeshShape.SPHERE,
                    "sphere",
                    (0.8, -0.4, 1),
                    (0.4, 0.4, 0.4),
                    (0.65, 0.25, 0.25, 1),
                ),
            ):
                app._add_scene_object(shape, name, size=size)
                owner = session.selected_node
                geometry = session.node(owner.children[0])
                assert session.submit(
                    cmd.SetPose(
                        owner.node_id, np.array(position), math3d.axis_angle_to_mat3((0, 0, 1), 0.3)
                    )
                )
                assert session.submit(cmd.SetGeometryColor(geometry.node_id, color))
        _park_cursor(viewer)
        _settle(viewer, 5)
        selected_object = session.selected
        assert primary.model is original and primary.model.ngeom == 0
        before = viewer.capture_array()
        Image.fromarray(before).save(output / "pending-scene.png")
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "pending-editor.png")
        _click(
            viewer,
            _item_center(viewer, "button", app.localizer.text("Apply") + "##apply_model_edits"),
        )
        deadline = time.monotonic() + 10
        while app.model_edits.active and time.monotonic() < deadline:
            viewer.sync()
        assert not app.model_edits.active, (app.model_edits.error, session.last_message)
        assert session.selected == selected_object
        _park_cursor(viewer)
        _settle(viewer, 5)
        after = viewer.capture_array()
        Image.fromarray(after).save(output / "applied-scene.png")
        difference = np.abs(after.astype(np.int16) - before.astype(np.int16))
        metrics = {
            "apply_mean_error": float(difference.mean()),
            "apply_max_error": int(difference.max()),
        }
        assert metrics["apply_mean_error"] < 0.1, metrics
        assert primary.model.ngeom == 2

        capsule = next(node for node in session.nodes if node.name == "capsule")
        for index, size in enumerate(((0.15, 0.15, 0.75), (0.6, 0.6, 0.1))):
            with model_edit_scope(session, app._intercept_model_edit):
                assert session.submit(cmd.SetGeometrySize(capsule.node_id, np.array(size)))
            _settle(viewer, 3)
            Image.fromarray(viewer.capture_array()).save(output / f"capsule-preview-{index}.png")
            _click(
                viewer,
                _item_center(
                    viewer, "button", app.localizer.text("Discard") + "##discard_model_edits"
                ),
            )
            _park_cursor(viewer)
            _settle(viewer, 3)
            np.testing.assert_allclose(viewer.capture_array(), after, atol=1)

        assert session.submit(cmd.SelectNode(capsule.node_id))
        app.camera.pivot[:] = (-0.8, 0, 0.9)
        app.camera.yaw, app.camera.pitch, app.camera.distance = 45, 30, 3
        viewer.set_gizmo_mode("dimensions")
        app.gizmo.enabled = True
        app._set_model_drop_notice("")
        _settle(viewer, 4)
        center = project(app.camera.view(), (app.gizmo._frame.position,), app._viewport_rect)[0, :2]
        original_model = primary.model
        _dimension_drag(viewer, center, center + np.array((30, -30)), GizmoHandle.SCREEN)
        assert primary.model is original_model and app.model_edits.active
        target, _ = app.gizmo._dimension_target(session, session.selected_node)
        ratios = target.size / np.array((0.3, 0.3, 0.5))
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)
        assert ratios[0] > 1
        metrics["capsule_uniform_factor"] = float(ratios[0])
        _park_cursor(viewer)
        _settle(viewer, 3)
        Image.fromarray(viewer.capture_array(surface="window")).save(output / "capsule-scaled.png")
        pending = viewer.capture_array()
        _click(
            viewer,
            _item_center(viewer, "button", app.localizer.text("Apply") + "##apply_model_edits"),
        )
        deadline = time.monotonic() + 10
        while app.model_edits.active and time.monotonic() < deadline:
            viewer.sync()
        assert not app.model_edits.active
        _park_cursor(viewer)
        _settle(viewer, 3)
        np.testing.assert_allclose(viewer.capture_array(), pending, atol=2)

        app._set_model_drop_notice("")
        box = next(node for node in session.nodes if node.name == "box")
        assert session.submit(cmd.SelectNode(box.node_id))
        for normal, handle in enumerate((GizmoHandle.YZ, GizmoHandle.ZX, GizmoHandle.XY)):
            target, _ = app.gizmo._dimension_target(session, session.selected_node)
            original_size = target.size.copy()
            rotation = math3d.axis_angle_to_mat3((0, 0, 1), 0.3)
            eye = np.full(3, 2.0)
            eye[normal] = 6
            offset = rotation @ eye
            app.camera.pivot[:] = (0.6, 0.6, 0.4)
            app.camera.yaw = float(np.degrees(np.arctan2(offset[1], offset[0])))
            app.camera.pitch = float(np.degrees(np.arcsin(offset[2] / np.linalg.norm(offset))))
            _settle(viewer, 4)
            frame, cam, rect = app.gizmo._frame, app.camera.view(), app._viewport_rect
            scale = world_scale(cam, frame.position, rect[3], SIZE_PT * viewer.window.style_scale)
            start_world = plane_corners(frame.position, frame.rotation, scale, normal).mean(axis=0)
            delta = np.full(3, 0.10)
            delta[normal] = 0
            start, end = project(cam, (start_world, start_world + rotation @ delta), rect)[:, :2]
            _dimension_drag(viewer, start, end, handle)
            target, _ = app.gizmo._dimension_target(session, session.selected_node)
            np.testing.assert_allclose(target.size, original_size + delta, atol=0.01)
            assert not app.model_edits.active
            metrics[f"box_{handle.name.lower()}_size"] = target.size.tolist()
            _park_cursor(viewer)
            _settle(viewer, 2)
            Image.fromarray(viewer.capture_array(surface="window")).save(
                output / f"box-{handle.name.lower()}-scaled.png"
            )
            assert session.submit(cmd.Undo())
            _settle(viewer, 3)
            target, _ = app.gizmo._dimension_target(session, session.selected_node)
            np.testing.assert_allclose(target.size, original_size)

        sphere = next(node for node in session.nodes if node.name == "sphere")
        assert session.submit(cmd.SelectNode(sphere.node_id))
        app.camera.pivot[:] = (0.8, -0.4, 1)
        app.camera.yaw, app.camera.pitch = -65, 25
        app.camera.distance = 3
        viewer.set_gizmo_mode("dimensions")
        app.gizmo.enabled = True
        app._set_model_drop_notice("")
        _settle(viewer, 4)
        target, reason = app.gizmo._dimension_target(session, sphere)
        assert target is not None, reason
        radius_before = float(target.size[0])
        center = project(app.camera.view(), (app.gizmo._frame.position,), app._viewport_rect)[0, :2]
        radius = SCREEN_RING_RADIUS * SIZE_PT * viewer.window.style_scale
        start = center + np.array((radius / np.sqrt(2), -radius / np.sqrt(2)))
        io = imgui.get_io()
        io.add_mouse_pos_event(*start)
        _settle(viewer, 3)
        assert app.gizmo._hovered is GizmoHandle.SCREEN
        Image.fromarray(viewer.capture_array(surface="window")).save(
            output / "sphere-ring-hover.png"
        )
        io.add_mouse_button_event(0, True)
        viewer.sync()
        assert app.gizmo.using
        io.add_mouse_pos_event(*(start + np.array((25, -25))))
        _settle(viewer, 2)
        io.add_mouse_button_event(0, False)
        _settle(viewer, 3)
        target, _ = app.gizmo._dimension_target(session, session.selected_node)
        assert target.size[0] > radius_before
        metrics["sphere_radius_before"] = radius_before
        metrics["sphere_radius_after"] = float(target.size[0])
        _park_cursor(viewer)
        _settle(viewer, 3)
        Image.fromarray(viewer.capture_array(surface="window")).save(
            output / "sphere-ring-dragged.png"
        )
        assert session.submit(cmd.Undo())
        _settle(viewer, 3)
        target, _ = app.gizmo._dimension_target(session, session.selected_node)
        assert np.isclose(target.size[0], radius_before)
        return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/scene-editing-audit"))
    args = parser.parse_args(argv)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    os.environ.setdefault("MOJIVE_LANGUAGE", "zh_CN")
    empty_editor = capture_empty_editor(args.output / "empty-editor")
    result = capture(args.output)
    result["empty_editor"] = empty_editor
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
