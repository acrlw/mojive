"""Real wl_pointer/wl_keyboard delivery in the private Weston acceptance target."""

from __future__ import annotations

import os
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.application.composition import build
from mojive.ui import window as wm

pytestmark = pytest.mark.skipif(
    "MOJIVE_TEST_INPUT_FD" not in os.environ, reason="Run make native-wayland-test"
)
OUTPUT = Path(os.environ.get("MOJIVE_WAYLAND_OUTPUT", "output/native-wayland"))
RENDERER = os.environ.get("MOJIVE_TEST_RENDERER", "bgfx")


class Input:
    def __init__(self, viewer):
        self.viewer = viewer
        self.socket = socket.socket(fileno=os.dup(int(os.environ["MOJIVE_TEST_INPUT_FD"])))
        self.socket.settimeout(10)
        self.offset = (0, 0)

    def step(self, frames=3):
        for _ in range(frames):
            self.viewer.sync()
            time.sleep(0.01)

    def send(self, command, *, step=True):
        self.socket.send(command.encode())
        assert self.socket.recv(1024) == b"ok", command
        if step:
            self.step()

    def move(self, x, y):
        self.send(f"move {round(x - self.offset[0])} {round(y - self.offset[1])}")

    def click(self, point):
        self.move(*point)
        self.send("button 272 1")
        self.send("button 272 0")

    def key(self, key):
        self.send(f"key {key} 1")
        self.send(f"key {key} 0")

    def chord(self, modifier, key):
        self.send(f"key {modifier} 1")
        self.key(key)
        self.send(f"key {modifier} 0")

    def item(self, function, label):
        original = getattr(imgui, function)
        points = []

        def record(name, *args, **kwargs):
            result = original(name, *args, **kwargs)
            if name == label:
                lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                points.append(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2))
            return result

        setattr(imgui, function, record)
        try:
            self.step(1)
        finally:
            setattr(imgui, function, original)
        assert points, label
        return points[-1]

    def capture(self, name):
        path = (OUTPUT / f"{RENDERER}-{name}").resolve()
        reference = self.viewer.capture_array(surface="window")
        Image.fromarray(reference).save(path.with_suffix(".png"))
        self.send(f"capture {path.with_suffix('.ppm')}", step=False)
        with Image.open(path.with_suffix(".ppm")) as image:
            rgb = np.array(image)
            image.save(path.with_name(path.name + "-compositor.png"))
        # This is compositor output, not the viewer's offscreen UI target.
        dx, dy = map(round, self.offset)
        height, width = rgb.shape[:2]
        reference = reference[max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)]
        presented = rgb[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
        error = np.abs(presented.astype(int) - reference.astype(int))
        assert error.mean() < 5, float(error.mean())
        assert np.mean(error.max(axis=2) > 32) < 0.02


def wait_message(process, expected, control):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        control.step(1)
        if select.select([process.stdout], [], [], 0)[0]:
            message = process.stdout.readline().decode().strip()
            assert message == expected, (message, process.poll())
            return
        assert process.poll() is None, process.stderr.read().decode()
    pytest.fail(f"Wayland helper did not report {expected!r}")


@pytest.fixture
def control(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MOJIVE_IMGUI_INI", str(tmp_path / "imgui.ini"))
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    wm._load_window_deps()
    with build(
        Path("assets/pick_scene.xml"),
        renderer=RENDERER,
        width=1400,
        height=1000,
        vsync=False,
        paused=True,
    ) as viewer:
        control = Input(viewer)
        try:
            control.step(8)
            assert wm.glfw.get_platform() == wm.glfw.PLATFORM_WAYLAND
            control.move(700, 400)
            control.offset = tuple(
                value - base
                for value, base in zip(
                    wm.glfw.get_cursor_pos(viewer.window._window), (700, 400), strict=True
                )
            )
            yield control
        finally:
            control.socket.close()


def test_pointer_selection_camera_wheel_and_keyboard(control):
    v = control.viewer
    # Pick a rendered object well inside the viewport using the ID product.
    ids = v.backend.target.read_ids(flip=True)
    labels, counts = np.unique(ids[ids != 0], return_counts=True)
    label = labels[np.argmax(counts)]
    ys, xs = np.nonzero(ids == label)
    index = len(xs) // 2
    x, y, w, h = v.app._viewport_rect
    point = (x + (xs[index] + 0.5) * w / ids.shape[1], y + (ys[index] + 0.5) * h / ids.shape[0])
    control.click(point)
    assert v.session.selected != 0
    control.key(1)  # Escape
    assert v.session.selected == 0
    center = (x + w * 0.55, y + h * 0.6)
    before = v.app.camera.view().view_matrix().copy()
    control.move(*center)
    control.send("button 272 1")
    for i in range(1, 6):
        control.move(center[0] + 12 * i, center[1] + 5 * i)
    control.send("button 272 0")
    assert not np.allclose(before, v.app.camera.view().view_matrix())
    distance = v.app.camera.distance
    control.send("axis 0 -20")
    assert v.app.camera.distance != distance
    control.key(57)  # Space
    assert not v.session.paused
    control.key(57)
    assert v.session.paused
    control.capture("navigation")


@pytest.mark.parametrize(
    ("left", "right", "attribute"), [(29, 97, "key_ctrl"), (42, 54, "key_shift")]
)
def test_both_modifier_keys_and_focus_loss(control, left, right, attribute):
    control.click((700, 400))
    control.send(f"key {left} 1")
    control.send(f"key {right} 1")
    control.send(f"key {left} 0")
    assert getattr(imgui.get_io(), attribute)
    control.send("focus 0")
    assert not wm.glfw.get_window_attrib(control.viewer.window._window, wm.glfw.FOCUSED)
    assert not getattr(imgui.get_io(), attribute)
    control.send(f"key {right} 0")
    control.send("focus 1")
    assert wm.glfw.get_window_attrib(control.viewer.window._window, wm.glfw.FOCUSED)
    control.key(1)
    assert not getattr(imgui.get_io(), attribute)


def test_text_clipboard_and_foreign_context_polling(control):
    v = control.viewer
    field = control.item("input_text_with_hint", "##filter")
    # Re-enter from the compositor background with no intervening motion.
    control.move(1400 + control.offset[0] - 1, 1000 + control.offset[1] - 1)
    control.move(*field)
    assert wm.glfw.get_cursor_pos(v.window._window) == pytest.approx(field, abs=1)
    assert tuple(imgui.get_io().mouse_pos) == pytest.approx(field, abs=1)
    control.click(field)
    control.key(30)  # A
    control.key(48)  # B
    hierarchy = v.app.panels.get("Hierarchy")
    assert hierarchy._filter == "ab"
    # A second ImGui context may be current when GLFW polls an event for v.
    foreign = imgui.create_context()
    try:
        control.send("button 272 1", step=False)
        control.send("button 272 0", step=False)
        control.send("key 107 1", step=False)  # End clears a possible double-click selection.
        control.send("key 107 0", step=False)
        control.send("key 46 1", step=False)
        control.send("key 46 0", step=False)
        time.sleep(0.02)
        imgui.set_current_context(foreign)
        wm.glfw.poll_events()
        control.step(5)
        assert hierarchy._filter == "abc"
        assert not foreign.input_events_queue
    finally:
        imgui.destroy_context(foreign)
        imgui.set_current_context(v.window._imgui_context)
    text = "关节 test λ"
    # An external Wayland client owns the selection; GLFW must receive it.
    copy = subprocess.Popen(
        [sys.executable, "tools/wayland/clipboard.py", "copy", text],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_message(copy, "ready", control)
        control.move(field[0] + 15, field[1])
        control.click(field)
        control.chord(29, 30)  # Ctrl+A
        control.chord(29, 47)  # Ctrl+V
        assert hierarchy._filter == text
        control.chord(29, 30)
        control.chord(29, 46)  # Ctrl+C
        paste = subprocess.Popen(
            [sys.executable, "tools/wayland/clipboard.py", "paste"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 5
        while paste.poll() is None and time.monotonic() < deadline:
            control.step(1)
        if paste.poll() is None:
            paste.kill()
        out, err = paste.communicate(timeout=5)
        assert paste.returncode == 0, err.decode()
        assert out.decode() == text
        control.capture("clipboard")
    finally:
        copy.terminate()
        copy.wait(timeout=5)


def test_external_file_drag_adds_model_from_unicode_path(control, tmp_path):
    source = tmp_path / "关节 sample.xml"
    source.write_bytes(Path("assets/joint_gizmo.xml").read_bytes())
    before = len(control.viewer.session.nodes)
    process = subprocess.Popen(
        [os.environ["MOJIVE_TEST_DRAG_SOURCE"], source.as_uri()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_message(process, "ready", control)
        control.move(720, 420)
        control.send("button 272 1")
        wait_message(process, "dragging", control)
        control.move(760, 460)
        control.send("button 272 0")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            control.step(1)
            if len(control.viewer.session.nodes) > before:
                break
        assert len(control.viewer.session.nodes) > before
        assert any("prismatic" in node.name for node in control.viewer.session.nodes)
        control.capture("file-drop")
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_two_visible_windows_route_input_and_survive_peer_close(control):
    first = control.viewer
    with build(
        Path("assets/joint_gizmo.xml"),
        renderer=RENDERER,
        width=1400,
        height=1000,
        vsync=False,
        paused=True,
    ) as second:
        peer = Input(second)
        peer.offset = control.offset
        try:
            peer.step(8)
            field = peer.item("input_text_with_hint", "##filter")
            peer.move(*field)
            # The first window polls native events while the second owns focus.
            peer.send("button 272 1", step=False)
            control.step(1)
            peer.step(1)
            peer.send("button 272 0", step=False)
            control.step(1)
            peer.step(1)
            peer.send("key 30 1", step=False)
            peer.send("key 30 0", step=False)
            control.step(2)
            peer.step(3)
            assert second.app.panels.get("Hierarchy")._filter == "a"
            assert first.app.panels.get("Hierarchy")._filter == ""
            peer.capture("peer")
        finally:
            peer.socket.close()
    control.step(8)
    control.click(control.item("input_text_with_hint", "##filter"))
    control.key(48)
    assert first.app.panels.get("Hierarchy")._filter == "b"
    control.capture("peer-closed")
