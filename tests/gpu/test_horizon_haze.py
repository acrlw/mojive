"""GPU coverage for MuJoCo horizon haze semantics."""

from __future__ import annotations

import numpy as np
import pytest

from forge_viewer.render.backend import RenderFlag
from forge_viewer.tools._harness import OffscreenHarness
from forge_viewer.types import CameraView
from forge_viewer.ui.viewport_widgets import (
    PLAYBACK_CHROME_SCALE,
    playback_size,
    viewport_chrome_scale,
)

pytestmark = pytest.mark.gpu


def test_mujoco_classic_skybox_depth_clips_far_infinite_plane(tmp_path):
    scene = tmp_path / "skybox-depth.xml"
    scene.write_text(
        """
        <mujoco>
          <visual><map zfar="30"/></visual>
          <asset>
            <texture type="skybox" builtin="gradient" width="64" height="384"
                     rgb1="0.2 0.4 0.7" rgb2="0.02 0.04 0.08"/>
            <texture name="grid" type="2d" builtin="flat" width="8" height="8"
                     rgb1="0.7 0.1 0.05"/>
            <material name="grid" texture="grid" texuniform="true"/>
          </asset>
          <worldbody>
            <geom name="floor" type="plane" size="0 0 .05" material="grid"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    with OffscreenHarness(scene, 320, 240) as harness:
        camera = CameraView(
            eye=np.array([0.0, -5.0, 1.0], np.float32),
            target=np.array([0.0, 0.0, 0.7], np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            near=0.01,
            far=50.0,
            aspect=4.0 / 3.0,
        )
        harness.camera = camera
        harness.backend.set_camera(camera)
        harness.backend.set_flag(RenderFlag.SKYBOX, True)
        harness.backend.set_flag(RenderFlag.HAZE, False)
        harness.step_and_render(0)
        visible = harness.backend.target.read_color(flip=True)[..., :3].copy()

        floor = next(node for node in harness.source.nodes if node.name == "floor")
        assert harness.backend._builder.set_visible(floor.node_id, False)
        harness.backend.set_render_scene(harness.backend._builder.scene)
        harness.step_and_render(0)
        hidden = harness.backend.target.read_color(flip=True)[..., :3].copy()

    height, width = visible.shape[:2]
    y, x = np.mgrid[0:height, 0:width]
    ndc_x = 2.0 * (x + 0.5) / width - 1.0
    ndc_y = 1.0 - 2.0 * (y + 0.5) / height
    forward = camera.target - camera.eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    tan_half = np.tan(camera.fov_y * 0.5)
    rays = (
        forward
        + ndc_x[..., None] * tan_half * camera.aspect * right
        + ndc_y[..., None] * tan_half * true_up
    )
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    plane_distance = -float(camera.eye[2]) / np.minimum(rays[..., 2], -1e-9)
    view_depth = plane_distance * np.sum(rays * forward, axis=2)
    cylinder_distance = (camera.far * 0.70) / np.maximum(
        np.hypot(rays[..., 0], rays[..., 1]), np.abs(rays[..., 2])
    )
    behind_skybox = (
        (rays[..., 2] < -1e-4)
        & (plane_distance > cylinder_distance * 1.05)
        & (view_depth < camera.far * 0.95)
    )
    assert np.count_nonzero(behind_skybox) > 100
    assert np.max(np.abs(visible.astype(np.int16) - hidden.astype(np.int16))[behind_skybox]) <= 1


def test_mujoco_haze_changes_sky_below_horizon_without_fogging_objects(tmp_path):
    scene = tmp_path / "horizon-haze.xml"
    scene.write_text(
        """
        <mujoco>
          <visual>
            <rgba haze="0.9 0.5 0.1 1"/>
            <map haze="0.3"/>
          </visual>
          <asset>
            <texture type="skybox" builtin="gradient" width="64" height="384"
                     rgb1="0.05 0.15 0.35" rgb2="0.3 0.55 0.9"/>
          </asset>
          <worldbody>
            <geom type="plane" size="0 0 .05" rgba=".2 .25 .3 1"/>
            <geom pos="0 0 .5" type="sphere" size=".5" rgba=".2 .8 .4 1"/>
            <camera pos="0 -4 4" xyaxes="1 0 0 0 .707 .707"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    with OffscreenHarness(scene, 320, 240) as harness:
        assert harness.backend.get_flag(RenderFlag.HAZE)
        assert harness.source.lights.horizon_haze_slices == 28
        camera = CameraView(
            eye=np.array([0.0, -5.0, 1.0], np.float32),
            target=np.array([0.0, 0.0, 0.7], np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            near=0.01,
            far=50.0,
        )
        harness.camera = camera
        harness.backend.set_camera(camera)
        harness.backend.set_flag(RenderFlag.SKYBOX, True)
        harness.backend.set_flag(RenderFlag.HAZE, False)
        harness.warmup(2)
        clear = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)

        harness.backend.set_flag(RenderFlag.HAZE, True)
        harness.step_and_render(0)
        haze_rgba = harness.backend.target.read_color(flip=True)
        haze = haze_rgba[..., :3].astype(np.int16)

    difference = np.max(np.abs(haze - clear), axis=2)
    sphere = (clear[..., 1] > clear[..., 0] + 20) & (clear[..., 1] > clear[..., 2] + 20)
    assert np.count_nonzero(sphere) > 100
    assert np.count_nonzero(difference > 5) > 100
    assert np.count_nonzero(difference[sphere]) == 0
    assert np.all(haze_rgba[..., 3] == 255)
    target = np.array([0.9, 0.5, 0.1]) * 255.0
    assert np.abs(haze.astype(np.float32) - target).max(axis=2).min() <= 2.0


def test_interactive_viewport_does_not_composite_haze_twice(tmp_path):
    pytest.importorskip("glfw")
    from forge_viewer.composition import build

    scene = tmp_path / "interactive-haze.xml"
    scene.write_text(
        """
        <mujoco>
          <visual>
            <rgba haze="0.45 0.55 0.65 1"/>
            <map haze="0.3"/>
          </visual>
          <asset>
            <texture type="skybox" builtin="gradient" width="64" height="384"
                     rgb1="0.08 0.18 0.32" rgb2="0.3 0.5 0.7"/>
          </asset>
          <worldbody>
            <geom type="plane" size="0 0 .05" rgba=".1 .15 .2 1"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    viewer = build(scene, paused=True, vsync=False, width=960, height=640)
    try:
        viewer.app.camera.pivot = (0.0, 0.0, 0.7)
        viewer.app.camera.distance = 5.0
        viewer.app.camera.yaw = -90.0
        viewer.app.camera.pitch = 5.0
        for _ in range(12):
            viewer.sync()

        target = viewer.backend.target.read_color(flip=True)
        window = np.asarray(viewer.window.read_frame())[::-1, :, :3]
        x, y, width, height = viewer.window.points_to_pixels(viewer.app._viewport_rect)
        x0, y0, x1, y1 = map(round, (x, y, x + width, y + height))
        viewport = window[y0:y1, x0:x1]

        assert viewport.shape == (*target.shape[:2], 3)
        assert np.all(target[..., 3] == 255)
        # This upper-left probe sits above the vertically centered Tool Column
        # and left of the playback capsule. Derive its right edge from current
        # playback geometry so chrome antialiasing cannot enter the sample when
        # the overlay proportions change.
        y0, y1 = 8, max(16, target.shape[0] // 8)
        style_scale = viewer.window.style_scale
        overlay_scale = viewer.app._viewport_overlay_scale
        point_to_pixel = target.shape[1] / viewer.app._viewport_rect[2]
        playback_scale = viewport_chrome_scale(
            style_scale,
            overlay_scale,
            PLAYBACK_CHROME_SCALE,
        )
        playback_width, _ = playback_size(
            playback_scale,
            viewer.app.viewport_chrome.playback_controls,
        )
        guard = 4.0 * style_scale
        x0 = round(guard * point_to_pixel)
        x1 = round(
            (viewer.app._viewport_rect[2] * 0.5 - playback_width * 0.5 - guard) * point_to_pixel
        )
        assert x1 > x0
        np.testing.assert_array_equal(
            viewport[y0:y1, x0:x1],
            target[y0:y1, x0:x1, :3],
        )
    finally:
        viewer.release()


def test_mujoco_haze_writes_depth_before_transparent_geometry(tmp_path):
    scene = tmp_path / "haze-transparent-depth.xml"
    scene.write_text(
        """
        <mujoco>
          <statistic center="0 3 0.5" extent="1"/>
          <visual>
            <map znear="0.01" zfar="10" haze="0.3"/>
            <rgba haze="0.15 0.25 0.35 1"/>
          </visual>
          <asset>
            <texture type="skybox" builtin="flat" rgb1="0 0 0"
                     width="32" height="256"/>
            <material name="probe" rgba="1 0 1 0.5" emission="1"/>
          </asset>
          <worldbody>
            <geom type="plane" size="0 0 0.05" rgba="0.05 0.05 0.05 1"/>
            <geom type="sphere" pos="0 6 0.4" size="0.25" material="probe"/>
            <camera name="probe" pos="0 0 1" xyaxes="1 0 0 0 0.1 0.995" fovy="45"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    with OffscreenHarness(scene, 640, 480) as harness:
        camera = harness.adapter.camera_view(0)
        assert camera is not None
        camera = camera.with_aspect(4.0 / 3.0)
        harness.camera = camera
        harness.backend.set_camera(camera)

        counts = []
        for enabled in (False, True):
            harness.backend.set_flag(RenderFlag.HAZE, enabled)
            harness.step_and_render(0)
            image = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)
            magenta = (image[..., 0] > image[..., 1] + 30) & (image[..., 2] > image[..., 1] + 30)
            counts.append(int(np.count_nonzero(magenta)))

    clear_count, haze_count = counts
    assert clear_count > 1_000
    assert clear_count * 0.25 < haze_count < clear_count * 0.65
