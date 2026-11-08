"""Focused hinge ticks stay distinct from the rendered rotation axis."""

import json
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.interaction.gizmo import (
    RING_RADIUS,
    SIZE_PT,
    GizmoHandle,
    project,
    screen_polygon_distance,
)
from mojive.scene.assets import resolve
from mojive.tools.native_ui_review import _NativeInput
from mojive.types import CameraView
from mojive.ui.gizmo import JOINT_RANGE_WIDTH_PT, _basis_from_z
from mojive.ui.gizmo.projection import _RotationDialProjector
from mojive.ui.gizmo.state import joint_slide_arrow_targets

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("joint_name", ("01_revolute_y", "02_prismatic_x"))
def test_joint_tick_projection_during_shift_drag_and_cancel(
    tmp_path, monkeypatch, backend_name, orthographic, joint_name
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1.25")
    output = Path("output/joint-tick-perspective") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    projection = "orthographic" if orthographic else "perspective"
    with (
        build_workspace(
            resolve("joint_gizmo"),
            paused=True,
            vsync=False,
            show_window=False,
            width=1400,
            height=900,
            config=ViewerConfig(layout=LayoutConfig(persistence=False)),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        joint = next(j for j in viewer.session.joints if j.name == joint_name)
        node = next(n for n in viewer.session.nodes if n.joint_index == joint.joint_id)
        assert viewer.session.submit(cmd.SelectNode(node.node_id))
        viewer.set_gizmo_mode("rotate")
        inputs.sync(10)
        gizmo = viewer.app.gizmo
        origin, basis = gizmo._frame.position.copy(), gizmo._frame.rotation.copy()
        viewer.set_camera(
            CameraView(
                eye=origin + (basis[:, 0] * 0.8 + basis[:, 2] * 0.6) * 0.8,
                target=origin.copy(),
                up=basis[:, 1],
                orthographic=orthographic,
                ortho_height=0.8,
            )
        )
        inputs.sync()
        view, rect = viewer.app._camera_view(), viewer.app._viewport_rect
        scale = viewer.window.style_scale
        dial = _RotationDialProjector(view, rect, origin, basis[:, 2], basis[:, 0], SIZE_PT * scale)
        center = project(view, (origin,), rect)[0, :2]

        def capture(state):
            pixels = viewer.capture_array(surface="window")
            density = viewer.window.pixel_scale
            radius = 180 * scale
            box = tuple(
                round(v * density)
                for v in (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                )
            )
            Image.fromarray(pixels).crop(box).save(
                output / f"{joint.type}-{projection}-{state}.png"
            )

        inputs.move((0, 0))
        capture("idle")
        if joint.type == "hinge":
            for angle in np.linspace(-1.5, 1.5, 32):
                cursor = dial.points(RING_RADIUS, angle)[0, :2]
                gizmo.update_hover(viewer.session, view, rect, tuple(cursor), style_scale=scale)
                if (
                    gizmo.hovered_handle is GizmoHandle.ROTATE_Z
                    and gizmo.hovered_joint_limit is None
                ):
                    break
            else:
                pytest.fail("No visible hinge ring segment can begin the drag")
            path = [
                dial.points(RING_RADIUS, angle + delta)[0, :2]
                for delta in np.linspace(0, np.deg2rad(25), 12)
            ]
        else:
            slide = gizmo._slide_range_projection(
                view, rect, scale, gizmo._joint_range, origin, basis
            )
            assert slide is not None
            cursor = joint_slide_arrow_targets(slide.current, slide.tangent, scale)[0]
            path = [cursor + slide.tangent * delta for delta in np.linspace(0, 70 * scale, 12)]
            gizmo.translation_snap_m = 0.01
        before = viewer.session.frame.qpos.copy()
        inputs.move(cursor)
        inputs.event("add_mouse_button_event", 0, True)
        assert gizmo.using
        inputs.event("add_key_event", imgui.Key.left_shift, True)
        for point in path:
            inputs.move(point, frames=1)
        assert gizmo.snapping
        assert not np.allclose(viewer.session.frame.qpos, before)
        capture("shift")
        inputs.escape()
        inputs.event("add_mouse_button_event", 0, False)
        inputs.event("add_key_event", imgui.Key.left_shift, False)
        inputs.sync()
        assert not gizmo.using
        np.testing.assert_allclose(viewer.session.frame.qpos, before)
        assert viewer.session.selected_node.node_id == node.node_id
        assert not viewer.session.editing and not viewer.session.can_undo
        # Cancellation consumes the held press; a fresh gesture can commit once.
        inputs.move(cursor)
        inputs.event("add_mouse_button_event", 0, True)
        for point in path:
            inputs.move(point, frames=1)
        assert gizmo.using
        capture("drag")
        inputs.event("add_mouse_button_event", 0, False)
        inputs.sync()
        assert viewer.session.can_undo
        assert viewer.session.submit(cmd.Undo()).ok
        inputs.sync()
        np.testing.assert_allclose(viewer.session.frame.qpos, before)
        assert not viewer.session.can_undo


@pytest.mark.parametrize("scale", [1.0, 1.2])
@pytest.mark.parametrize("orthographic", [False, True])
def test_joint_focus_keeps_rendered_ticks_clear_and_pickable(
    tmp_path, monkeypatch, backend_name, scale, orthographic
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    projection = "ortho" if orthographic else "perspective"
    output = Path("output/joint-focus-readability") / f"{backend_name}-{scale:g}-{projection}"
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    with build_workspace(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=1400,
        height=900,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        viewer.set_gizmo_mode("rotate")
        for _ in range(8):
            viewer.sync()
        for name, value, aligned_angle in (
            ("05_multi_revolute_z", -26.624, -26.624),
            ("01_revolute_y", 60.0, 120.0),
            ("06_precision_hinge_y", 1.0, None),
        ):
            joint = next(joint for joint in viewer.session.joints if joint.name == name)
            node = next(node for node in viewer.session.nodes if node.joint_index == joint.joint_id)
            assert viewer.session.submit(cmd.SetQpos(joint.qpos_adr, np.deg2rad(value)))
            assert viewer.session.submit(cmd.SelectNode(node.node_id))
            for _ in range(3):
                viewer.sync()
            gizmo = viewer.app.gizmo
            target, reason = gizmo._joint_target(viewer.session, node)
            assert target is not None, reason
            origin, basis = gizmo._target_pose(viewer.session, node, target)
            basis = _basis_from_z(basis[:, 2])
            if aligned_angle is None:
                direction = np.array((-1.0, 1.0, 1.0))
            else:
                angle = np.deg2rad(aligned_angle)
                direction = basis[:, 2] + basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)
            direction /= np.linalg.norm(direction)
            viewer.set_camera(
                CameraView(
                    eye=origin + direction * 1.5,
                    target=origin.copy(),
                    up=np.array((0.0, 0.0, 1.0)),
                    orthographic=orthographic,
                    ortho_height=1.5,
                )
            )
            for _ in range(3):
                viewer.sync()
            Image.fromarray(viewer.capture_array(surface="viewport")).save(
                output / f"{name}-before.png"
            )
            qpos = viewer.session.frame.qpos.copy()
            assert viewer.app.request_joint_focus(joint.joint_id)
            viewer.sync()
            viewer.app.camera.advance(1.0, viewer.app.camera_out)
            for _ in range(3):
                viewer.sync()
            view, rect = viewer.app._camera_view(), viewer.app._viewport_rect
            actual_scale = viewer.window.style_scale
            dial = gizmo._hinge_range_projection(view, rect, actual_scale, gizmo._joint_range)
            axis = gizmo._hinge_axis_projection(view, rect, actual_scale, origin, basis[:, 2])
            assert dial is not None and axis is not None
            clearances = {}
            for label, tick in (
                ("current", dial.current_tick),
                ("MIN", dial.lower_tick),
                ("MAX", dial.upper_tick),
            ):
                assert tick is not None
                clearance = (
                    min(
                        screen_polygon_distance(point, polygon)
                        for point in np.linspace(*tick, 41)
                        for polygon in axis.polygons
                    )
                    - JOINT_RANGE_WIDTH_PT * actual_scale * 0.5
                )
                clearances[label] = clearance
                assert clearance >= 2.0 * actual_scale, (name, label, clearance)
                if label != "current":
                    gizmo.update_hover(
                        viewer.session, view, rect, tuple(tick[1]), style_scale=actual_scale
                    )
                    assert gizmo.hovered_joint_limit is not None, (name, label)
                    assert gizmo.hovered_joint_limit.label.startswith(label)
            np.testing.assert_array_equal(viewer.session.frame.qpos, qpos)
            assert viewer.session.selected_node.node_id == node.node_id
            final_direction = viewer.app.camera.direction().copy()
            assert viewer.app.request_joint_focus(joint.joint_id)
            viewer.sync()
            viewer.app.camera.advance(1.0, viewer.app.camera_out)
            for _ in range(3):
                viewer.sync()
            assert viewer.app.camera.direction() == pytest.approx(final_direction, abs=1e-6)
            Image.fromarray(viewer.capture_array(surface="viewport")).save(
                output / f"{name}-after.png"
            )
            measurements.append({"joint": name, "clearance_pt": clearances})
    (output / "clearance.json").write_text(json.dumps(measurements, indent=2) + "\n")
