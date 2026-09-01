"""Overlay features draw through the Draw2D protocol, so a recording fake can
verify what would be painted — and in which order — without a window."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.gizmo import AXIS_COLORS, paint_order, plane_direction
from forge_viewer.render.backend import BackendCaps
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import CameraView
from forge_viewer.ui import viewcube as vc
from forge_viewer.ui.draw2d import (
    Draw2D,
    ImguiDraw2D,
    _anti_alias_fringe_outer,
    _capped_polyline_outline,
    _round_cap_polyline_outline,
)
from forge_viewer.ui.gizmo import ObjectGizmo

RECT = (0.0, 0.0, 800.0, 600.0)


class RecordingDraw2D:
    """Draw2D fake that records every call as a (name, args) tuple."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record

    def text_size(self, text: str) -> tuple[float, float]:
        return (6.0 * len(text), 12.0)


class CaptureBackend:
    caps = BackendCaps(name="capture", gizmo=True)

    def __init__(self) -> None:
        self.frame = None

    def set_gizmo(self, frame) -> bool:
        self.frame = frame
        return frame is not None


def camera() -> CameraView:
    return CameraView(
        eye=np.array((4.0, -6.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=RECT[2] / RECT[3],
    )


def axis_of(color) -> int:
    return int(np.argmin(np.linalg.norm(AXIS_COLORS[:, :3] - color[:3], axis=1)))


def test_imgui_adapter_covers_the_protocol_surface() -> None:
    missing = [
        name
        for name, _member in inspect.getmembers(Draw2D, inspect.isfunction)
        if not callable(getattr(ImguiDraw2D, name, None))
    ]
    assert not missing


def test_fill_fringe_expands_outward_for_both_polygon_windings() -> None:
    square = np.array(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)))
    expected = np.array(((-1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-1.0, 3.0)))

    assert _anti_alias_fringe_outer(square) == pytest.approx(expected)
    assert _anti_alias_fringe_outer(square[::-1]) == pytest.approx(expected[::-1])


def test_round_cap_polyline_is_one_capsule_silhouette() -> None:
    outline = np.asarray(_round_cap_polyline_outline(((0.0, 0.0), (10.0, 0.0)), 4.0))

    assert outline[:, 0].min() == pytest.approx(-2.0)
    assert outline[:, 0].max() == pytest.approx(12.0)
    assert outline[:, 1].min() == pytest.approx(-2.0)
    assert outline[:, 1].max() == pytest.approx(2.0)
    assert len(outline) > 4


def test_asymmetric_cap_polyline_is_flat_at_start_and_round_at_end() -> None:
    outline = np.asarray(
        _capped_polyline_outline(
            ((0.0, 0.0), (10.0, 0.0)),
            4.0,
            round_start=False,
            round_end=True,
        )
    )

    assert outline[:, 0].min() == pytest.approx(0.0)
    assert outline[:, 0].max() == pytest.approx(12.0)
    assert outline[:, 1].min() == pytest.approx(-2.0)
    assert outline[:, 1].max() == pytest.approx(2.0)


def test_round_cap_polyline_keeps_clockwise_screen_winding_for_a_reflex_arc() -> None:
    angles = np.linspace(0.0, np.radians(240.0), 80)
    path = np.column_stack((100.0 * np.cos(angles), 100.0 * np.sin(angles)))

    outline = np.asarray(_round_cap_polyline_outline(path, 4.0))
    signed_area = 0.5 * np.sum(
        outline[:, 0] * np.roll(outline[:, 1], -1) - outline[:, 1] * np.roll(outline[:, 0], -1)
    )

    assert signed_area > 0.0
    assert signed_area == pytest.approx(np.radians(240.0) * 100.0 * 4.0, rel=0.02)


def test_flat_gizmo_submits_handles_in_painter_order() -> None:
    gizmo = ObjectGizmo()
    cam = camera()
    scene = Scene()
    obj = scene.box(name="editable")
    session = Session(StaticSceneAdapter(scene))
    session.submit(cmd.Select(obj.object_id))
    assert gizmo.publish(
        CaptureBackend(),
        session,
        cam,
        RECT,
        ui_scale=1.0,
        style_scale=1.0,
        yielding=False,
        interactive=False,
    )
    overlay = RecordingDraw2D()
    gizmo.draw_overlay(cam, RECT, overlay, style_scale=1.0)

    names = [name for name, _args in overlay.calls]
    assert names == ["convex_fill"] * 3 + ["concave_fill"] * 3 + ["circle_filled"] * 2

    origin = np.zeros(3)
    rotation = np.eye(3)
    # Camera eye (4, -6, 3): the Y handle is farthest, the X handle nearest.
    planes = [args[1] for name, args in overlay.calls if name == "convex_fill"]
    expected_planes = paint_order(
        cam, origin, [plane_direction(rotation, axis) for axis in range(3)]
    )
    assert [axis_of(color) for color in planes] == list(expected_planes) == [0, 2, 1]

    arrows = [args[1] for name, args in overlay.calls if name == "concave_fill"]
    expected_arrows = paint_order(cam, origin, [rotation[:, axis] for axis in range(3)])
    assert [axis_of(color) for color in arrows] == list(expected_arrows) == [1, 2, 0]


def test_viewcube_submits_balls_back_to_front() -> None:
    cube = vc.ViewCube()
    cam = CameraView(
        eye=np.array((4.0, -4.0, 3.0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0.0, 0.0, 1.0), np.float32),
    )
    cube.update(cam, RECT, cursor=(-1000.0, -1000.0), style_scale=1.0)
    overlay = RecordingDraw2D()
    cube.draw(overlay, style_scale=1.0)

    expected: list[str] = []
    for ball in cube.balls:  # layout() is already sorted far-to-near
        if ball.alpha <= 0.0:
            continue
        expected.append("fringed_concave_fill" if ball.positive else "circle_filled")
        if not ball.positive:
            expected.append("circle")
        if vc._label_alpha(ball, False) > 0.0:
            expected.append("centered_label")
    assert [name for name, _args in overlay.calls] == expected

    labels = [args[0] for name, args in overlay.calls if name == "centered_label"]
    assert labels == [
        ball.label for ball in cube.balls if ball.alpha > 0.0 and vc._label_alpha(ball, False) > 0.0
    ]
