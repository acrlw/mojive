"""The native camera frame, image and action corners use measured offsets."""

from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.types import ViewportImage
from mojive.ui.camera_preview import CameraPreview
from mojive.ui.workspace.camera_preview import WorkspaceCameraPreview
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.menus import Menus
from tests.test_component_appearance import appearance as appearance


def _camera(ui, width=700):
    preview = CameraPreview()
    preview.set_enabled(True)
    preview._image = ViewportImage(19, 320, 180)
    app = SimpleNamespace(
        camera_preview=preview,
        viewport_layers=SimpleNamespace(viewport_ui=True),
        _viewport_rect=(20 * ui.s, 30 * ui.s, width * ui.s, 500 * ui.s),
    )
    workspace = SimpleNamespace(
        app=app,
        ui=ui,
        s=ui.s,
        rects={"overlays": []},
        input_overlays=[],
        window=SimpleNamespace(
            style_scale=ui.s,
            viewport_texture_ref=lambda image: imgui.ImTextureRef(image.texture_id),
        ),
    )
    workspace.menus = Menus(workspace, ())
    return WorkspaceCameraPreview(workspace)


def _records(runtime):
    return {record["key"]: record for record in runtime.observed}


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("width", (130, 700))
@pytest.mark.parametrize("radius", (0, 12, 20))
def test_camera_actual_image_and_close_button_are_concentric(appearance, width, radius):
    ui, runtime, update, frame = appearance
    update({"camera_preview": {"radius": radius}, "icon_button": {"radius": 23}})
    runtime.observe_components = True
    camera = _camera(ui, width)
    # Settle the child work rectangle before inspecting its native placement.
    for _ in range(3):
        runtime.observed.clear()
        camera.workspace.rects["overlays"] = []
        frame(lambda: camera.draw("Front camera"))
    records = _records(runtime)
    outer, header, image = (
        records[key] for key in ("camera-preview", "camera-preview-header", "camera-preview-image")
    )
    x, y, w, h = outer["hit"]
    cx, cy, cw, ch = ui.hits["camera-preview-close"]
    gap = native_unit(8 * ui.s)
    assert cy - y == pytest.approx(gap)
    assert x + w - cx - cw == pytest.approx(gap)
    assert header["actionRadius"] == pytest.approx(max(0, radius * ui.s - gap))
    assert image["cornerInsets"] == pytest.approx((gap,) * 3)
    assert image["radius"] == pytest.approx(max(0, radius * ui.s - gap))
    _dx, dy, dw, dh = image["content"]
    assert dw / dh == pytest.approx(16 / 9)
    assert camera.workspace.app.camera_preview.bounds == pytest.approx((x, y, x + w, y + h))
    assert all(value == int(value) for value in (x, y, w, h))
    assert dy + dh <= y + h - gap + 1e-5
    assert ch > 0 and len(header["actions"]) == (2 if width == 130 else 3)


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("native", (False, True))
def test_camera_corner_constraint_reaches_actual_active_action_paint(
    appearance, monkeypatch, native
):
    ui, _runtime, update, frame = appearance
    update({"camera_preview": {"radius": 12}, "icon_button": {"radius": 23}})
    if not native:
        ui.design = None
    camera = _camera(ui)
    camera.workspace.app.camera_preview._pinned = True
    painted = []
    original = ui.rect

    def capture(x, y, width, height, ink, r=0):
        painted.append(((x, y, width, height), r * ui.s))
        return original(x, y, width, height, ink, r)

    monkeypatch.setattr(ui, "rect", capture)
    for _ in range(3):
        painted.clear()
        camera.workspace.rects["overlays"] = []
        frame(lambda: camera.draw("Front camera"))
    pin = ui.hits["camera-preview-pin"]
    radii = [radius for rect, radius in painted if rect == pytest.approx(pin)]
    expected = 12 * ui.s - native_unit(8 * ui.s) if native else 6 * ui.s
    assert radii == pytest.approx([expected])
    assert camera._image_radius == pytest.approx(radii[0])
    # The rounded texture has a tessellated path, not the old four-corner quad.
    # Child IDs are implementation-defined; find the actual submitted child.
    children = imgui.internal.find_window_by_name("Appearance").dc.child_windows
    assert children
    child = children[len(children) - 1]
    vertices = child.draw_list.vtx_buffer
    assert len(vertices) > 4


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
def test_authored_camera_side_insets_resolve_to_one_adjacent_corner_offset(appearance):
    ui, runtime, update, frame = appearance
    update(
        {
            "camera_header": {"paddingTop": 7, "paddingRight": 10, "marginLeft": 2},
            "camera_body": {"marginBottom": 3},
        }
    )
    runtime.observe_components = True
    camera = _camera(ui)
    for _ in range(3):
        runtime.observed.clear()
        camera.workspace.rects["overlays"] = []
        frame(lambda: camera.draw("Front camera"))
    records = _records(runtime)
    parent, header, image = (
        records[key] for key in ("camera-preview", "camera-preview-header", "camera-preview-image")
    )
    gap = parent["cornerInset"]
    assert header["padding"] == pytest.approx((gap,) * 4)
    assert image["cornerInsets"] == pytest.approx((gap,) * 3)
    x, y, width, _height = parent["hit"]
    cx, cy, cw, _ch = ui.hits["camera-preview-close"]
    assert (cy - y, x + width - cx - cw) == pytest.approx((gap, gap))
    assert image["radius"] == pytest.approx(max(0, parent["radius"] - gap))


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("g3", (False, True))
@pytest.mark.parametrize("authored_small", (False, True))
def test_actual_camera_close_hover_and_frame_triangles_use_the_same_corner_offset(
    appearance, g3, authored_small, monkeypatch
):
    from copy import deepcopy
    from math import sqrt

    from mojive.geometry2d.curves import smooth_rect_points
    from mojive.ui.workspace import camera_preview as camera_module
    from mojive.ui.workspace.widgets import color
    from tests.test_menu_corner_offsets import _corner_point, _paint_covers

    ui, runtime, _update, frame = appearance
    panel_ink = (19, 67, 101, 255)
    monkeypatch.setattr(camera_module, "PANEL", panel_ink)
    spec = deepcopy(runtime.spec)
    spec["native"]["g3"] = g3
    if authored_small:
        spec["parts"].update(
            {"camera_preview": {"radius": 32}, "icon_button": {"width": 20, "height": 20}}
        )
    runtime.update_spec(spec)
    camera = _camera(ui)

    def paint():
        camera.workspace.rects["overlays"] = []
        camera.draw("joint_gizmos")

    for _ in range(3):
        frame(paint)
    x, y, width, height = ui.hits["camera-preview-close"]
    imgui.get_io().add_mouse_pos_event(x + width / 2, y + height / 2)
    for _ in range(3):
        frame(paint)
    parent = imgui.internal.find_window_by_name("Appearance")
    children = parent.dc.child_windows
    child = children[len(children) - 1]
    close = ui.hits["camera-preview-close"]
    outer, inset = camera._frame, camera._corner_inset
    inner = max(0, camera._radius - inset)
    if authored_small:
        assert close[2:] == pytest.approx((20 * ui.s,) * 2)
        assert inset > native_unit(8 * ui.s)
        assert camera._radius == pytest.approx(32 * ui.s)
        assert inner <= min(close[2:]) / 2 - 1
        assert camera._image_radius == pytest.approx(inner)
    else:
        assert (inset, inner) == pytest.approx((native_unit(8 * ui.s), 6 * ui.s))
    assert (close[0] + close[2] + inset, close[1] - inset) == pytest.approx((outer.right, outer.y))
    assert camera._radius == pytest.approx(inner + inset)
    assert _paint_covers(
        child.draw_list,
        (close[0] + close[2] / 2, close[1] + close[3] / 2),
        color((255, 255, 255, 12)),
    )
    for rect, radius, draw, ink in (
        (close, inner, child.draw_list, color((255, 255, 255, 12))),
        (outer.tuple(), camera._radius, parent.draw_list, color(ui.ink(panel_ink))),
    ):
        diagonal = radius * (1 - sqrt(0.5))
        if g3:
            x, y, width, height = rect
            points = smooth_rect_points(
                x,
                y,
                x + width,
                y + height,
                radius,
                smoothing=runtime.spec["native"]["g3_smoothing"],
            )
            candidates = []
            for a, b in zip(points, (*points[1:], points[0]), strict=True):
                start = a[0] + a[1] - x - width - y
                end = b[0] + b[1] - x - width - y
                if start * end <= 0 and start != end:
                    t = start / (start - end)
                    inset = x + width - a[0] - t * (b[0] - a[0])
                    if 0 <= inset <= radius:
                        candidates.append(inset)
            diagonal = min(candidates)
        # Probe actual filled triangles, away from the anti-alias fringe.
        assert not _paint_covers(draw, _corner_point(rect, True, False, diagonal - 0.8), ink)
        assert _paint_covers(draw, _corner_point(rect, True, False, diagonal + 0.8), ink)
