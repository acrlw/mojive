from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")

from forge_viewer import commands as cmd  # noqa: E402
from forge_viewer.bridge import DebugClient  # noqa: E402
from forge_viewer.composition import build_scene  # noqa: E402
from forge_viewer.demos import canvas_scene  # noqa: E402
from forge_viewer.types import DEFAULT_MATERIAL, Material  # noqa: E402


@pytest.fixture(scope="module")
def canvas():
    scene = canvas_scene()
    viewer = build_scene(scene, vsync=False, width=1100, height=720)
    try:
        for _ in range(8):
            viewer.sync()
        yield viewer, scene
    finally:
        viewer.release()


def snap(viewer) -> np.ndarray:
    px = viewer.window.read_frame()
    assert px is not None
    return np.asarray(px)[::-1][..., :3].copy()


def viewport_snap(viewer) -> np.ndarray:
    """Crop the window snapshot to the viewport panel.

    The docked stats panel redraws its frame-time plot every sync, so a
    whole-window diff measures UI churn instead of the scene edit — and the
    churn differs between the forge and wgpu frame loops.
    """
    image = snap(viewer)
    x, y, w, h = viewer.window.points_to_pixels(viewer.app._viewport_rect)
    x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
    return image[y0:y1, x0:x1]


def test_canvas_opens_without_importing_mujoco(canvas):
    viewer, _scene = canvas
    image = snap(viewer)

    assert "mujoco" not in sys.modules
    assert viewer.session.adapter.caps.name == "static"
    assert viewer.session.paused
    assert viewer.backend.stats.instances == 4
    assert float(image.std()) > 10.0


def test_canvas_selection_reaches_antialiased_outline(canvas):
    viewer, _scene = canvas
    target = next(n for n in viewer.session.nodes if n.name == "crate")
    viewer.session.submit(cmd.Select(target.object_id))
    viewer.sync()
    viewer.sync()
    image = snap(viewer).astype(np.int16)
    outline = np.array([255, 161, 51], np.int16)

    assert np.all(np.abs(image - outline) <= 3, axis=-1).sum() > 100


def test_canvas_pose_update_changes_the_window(canvas):
    viewer, scene = canvas
    before = snap(viewer)
    scene.object("ball").set_pose((1.4, -0.3, 0.42))
    viewer.sync()
    viewer.sync()
    after = snap(viewer)

    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 500


def test_canvas_material_edits_change_the_window(canvas):
    viewer, scene = canvas
    before = viewport_snap(viewer)
    crate = scene.object("crate")
    crate.set_color((0.1, 0.8, 0.9, 1.0))
    crate.set_material(Material(name="emissive", emission=0.65, specular=0.1))
    viewer.sync()
    viewer.sync()
    after = viewport_snap(viewer)

    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 150

    crate.set_color((0.92, 0.42, 0.18, 1.0))
    crate.set_material(DEFAULT_MATERIAL)
    viewer.sync()


def test_canvas_detects_dynamic_structure_changes(canvas):
    viewer, scene = canvas
    generation = viewer.session.structure_generation
    obj = scene.sphere(name="spawned", position=(0.0, 1.2, 0.25), size=(0.25, 0.25, 0.25))
    viewer.sync()

    assert viewer.session.structure_generation == generation + 1
    assert viewer.session.node_by_object_id(obj.object_id).name == "spawned"
    assert viewer.backend.stats.instances == 5


def test_canvas_receives_external_debug_draw(canvas):
    viewer, _scene = canvas
    before = viewer.backend.debug.stats().primitives
    with DebugClient(pid=os.getpid()) as client:
        client.send(
            op="line",
            layer="external-test",
            id="socket-line",
            a=[-1.0, 0.0, 1.0],
            b=[1.0, 0.0, 1.0],
            color=[1.0, 0.0, 1.0, 1.0],
            width_px=4.0,
        )
        deadline = time.monotonic() + 1.0
        while viewer.backend.debug.stats().primitives == before and time.monotonic() < deadline:
            viewer.sync()

    assert viewer.bridge.stats.applied >= 1
    assert viewer.backend.debug.stats().primitives == before + 1


def test_canvas_records_streaming_video(canvas, tmp_path):
    from imageio_ffmpeg import read_frames

    viewer, scene = canvas
    output = tmp_path / "canvas.mp4"

    def move(index, _viewer):
        scene.object("ball").set_pose((0.3 + 0.15 * index, -0.3, 0.42))

    viewer.record(output, frames=4, fps=24, before_frame=move)
    reader = read_frames(str(output))
    metadata = next(reader)
    count = sum(1 for _ in reader)
    reader.close()

    assert output.stat().st_size > 1000
    assert count == 4
    assert metadata["fps"] == pytest.approx(24.0)
