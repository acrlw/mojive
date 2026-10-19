"""Native resize input remains available without painting wedges over panel corners."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.tools.native_workspace import float_panel
from mojive.ui.workspace.window_chrome import floating_dock_root

pytestmark = pytest.mark.gpu

GRIP_COLORS = (
    (0.97, 0.07, 0.43, 1.0),
    (0.03, 0.93, 0.53, 1.0),
    (0.53, 0.13, 0.97, 1.0),
)


@pytest.fixture(params=(1, 1.2, 1.25))
def native_resize(request, backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(request.param))
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        vsync=False,
        width=1600,
        height=1000,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    events = []
    process = viewer.window._input.process_inputs

    def inputs():
        process()
        io = imgui.get_io()
        for method, args in events:
            getattr(io, method)(*args)
        events.clear()
        io.delta_time = 1 / 60

    monkeypatch.setattr(viewer.window._input, "process_inputs", inputs)

    def frame(count=1):
        for _ in range(count):
            viewer.sync()

    def event(method, *args):
        events.append((method, args))
        frame()

    try:
        frame(8)
        for role, color in zip(
            (imgui.Col_.resize_grip, imgui.Col_.resize_grip_hovered, imgui.Col_.resize_grip_active),
            GRIP_COLORS,
            strict=True,
        ):
            imgui.get_style().set_color_(role, color)
        yield SimpleNamespace(viewer=viewer, frame=frame, event=event, scale=request.param)
    finally:
        viewer.release()


def _ink(draw):
    # Include the AA fringe as well as opaque vertices, retaining only RGB.
    return {vertex.col & 0xFFFFFF for vertex in draw.vtx_buffer if vertex.col >> 24}


def _grip_ink():
    return {imgui.color_convert_float4_to_u32(color) & 0xFFFFFF for color in GRIP_COLORS}


def _owner(grouped):
    window = imgui.internal.find_window_by_name("Timeline")
    root = floating_dock_root(window)
    if grouped:
        assert root is not None and root.host_window is not None
        return root.host_window
    assert window.dock_id == 0 and root is None
    return window


def _bounds(window):
    return np.array((*window.pos, *window.size), dtype=float)


def _corner(window, left, scale):
    inset = 3 * scale
    return (
        window.pos.x + inset if left else window.pos.x + window.size.x - inset,
        window.pos.y + window.size.y - inset,
    )


@pytest.mark.parametrize("grouped", (False, True), ids=("solo", "floating-dock-host"))
def test_native_lower_corners_resize_without_filled_grip_vertices(native_resize, grouped):
    driver = native_resize
    float_panel(driver.viewer, "Timeline", with_panel="Output" if grouped else None)
    window = _owner(grouped)
    driver.frame(3)
    assert not _ink(window.draw_list).intersection(_grip_ink())
    for left in (True, False):
        before = _bounds(window)
        start = _corner(window, left, driver.scale)
        cursor = imgui.MouseCursor_.resize_nesw if left else imgui.MouseCursor_.resize_nwse
        grip_id = imgui.internal.get_window_resize_corner_id(window, int(left))
        driver.event("add_mouse_pos_event", *start)
        driver.frame(3)
        assert imgui.get_mouse_cursor() == cursor
        assert not _ink(window.draw_list).intersection(_grip_ink())
        driver.event("add_mouse_button_event", 0, True)
        assert imgui.get_current_context().active_id == grip_id
        assert not _ink(window.draw_list).intersection(_grip_ink())
        delta = np.array((-36 if left else 36, 28)) * driver.scale
        for fraction in (0.5, 1.0):
            driver.event("add_mouse_pos_event", *(np.array(start) + delta * fraction))
            assert imgui.get_mouse_cursor() == cursor
            assert not _ink(window.draw_list).intersection(_grip_ink())
        expected = before + np.array((delta[0] if left else 0, 0, abs(delta[0]), delta[1]))
        np.testing.assert_allclose(_bounds(window), expected, atol=1)
        # Reverse the same held gesture to prove inward resizing without a new hit target.
        driver.event("add_mouse_pos_event", *start)
        np.testing.assert_allclose(_bounds(window), before, atol=1)
        driver.event("add_mouse_button_event", 0, False)
        driver.frame(2)
        assert imgui.get_current_context().active_id != grip_id


def test_native_corner_double_click_still_finishes_auto_fit(native_resize):
    driver = native_resize
    float_panel(driver.viewer, "Timeline")
    window = _owner(False)
    start = _corner(window, False, driver.scale)
    grip_id = imgui.internal.get_window_resize_corner_id(window, 0)
    driver.event("add_mouse_pos_event", *start)
    driver.frame(3)
    driver.event("add_mouse_button_event", 0, True)
    assert imgui.get_current_context().active_id == grip_id
    driver.event("add_mouse_button_event", 0, False)
    driver.event("add_mouse_button_event", 0, True)
    assert imgui.get_io().mouse_clicked_count[0] == 2
    # The native auto-fit branch clears the resize ID while the second press is held.
    assert imgui.get_current_context().active_id == 0
    assert np.isfinite(_bounds(window)).all() and min(window.size) > 0
    after = _bounds(window)
    driver.event("add_mouse_pos_event", start[0] + 24, start[1] + 24)
    np.testing.assert_allclose(_bounds(window), after, atol=1)
    driver.event("add_mouse_button_event", 0, False)
    assert not _ink(window.draw_list).intersection(_grip_ink())


def test_production_dock_splitter_keeps_colored_hover_and_drag(native_resize):
    driver = native_resize
    driver.viewer.app.workspace.activate("Timeline")
    driver.frame(5)
    window = imgui.internal.find_window_by_name("Timeline")
    assert window.dock_id
    start = (
        window.pos.x + window.size.x * 0.5,
        window.pos.y - imgui.get_style().docking_separator_size * 0.5,
    )
    before = _bounds(window)

    def painted(color):
        packed = imgui.color_convert_float4_to_u32(color) & 0xFFFFFF
        return any(
            packed in _ink(owner.draw_list)
            for owner in imgui.get_current_context().windows
            if owner.active
        )

    driver.event("add_mouse_pos_event", *start)
    driver.frame(5)
    assert imgui.get_mouse_cursor() == imgui.MouseCursor_.resize_ns
    assert painted(GRIP_COLORS[1]), "Dock splitter must retain its native hover feedback"
    driver.event("add_mouse_button_event", 0, True)
    assert painted(GRIP_COLORS[2]), "Dock splitter must retain its native active feedback"
    driver.event("add_mouse_pos_event", start[0], start[1] - 48 * driver.scale)
    assert window.size.y > before[3] + 40 * driver.scale
    driver.event("add_mouse_button_event", 0, False)
