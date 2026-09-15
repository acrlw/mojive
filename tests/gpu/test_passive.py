"""Real-window ownership, asynchronous progress, and passive input exchange."""

import time
from pathlib import Path

import numpy as np
import pytest

from mojive import (
    CameraTrackingConfig,
    CameraView,
    PassiveAction,
    RecordingConfig,
    RecordingPhase,
    SharedImage,
    ViewerConfig,
    build,
    launch_passive,
)
from mojive import commands as cmd
from mojive.control.rpc import RpcClient, RpcError

pytestmark = [pytest.mark.gpu, pytest.mark.physics]
mujoco = pytest.importorskip("mujoco")


def test_existing_model_data_keep_an_external_clock():
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    with build(
        model=model, data=data, vsync=False, show_window=False, width=640, height=480
    ) as viewer:
        assert viewer.session.adapter.model is model
        assert viewer.session.adapter.data is data
        assert viewer.is_running()
        mujoco.mj_step(model, data, nstep=7)
        before = data.time
        assert not viewer.session.submit(cmd.Pause()).ok
        viewer.session.submit(cmd.Play())
        assert not viewer.session.submit(cmd.Step()).ok
        assert not viewer.session.submit(cmd.Reset()).ok
        for _ in range(4):
            viewer.sync()
        assert data.time == before
        image = viewer.capture_array()
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    assert not viewer.is_running()
    viewer.close()
    with pytest.raises(RuntimeError, match="closed"):
        viewer.sync()


def test_passive_rates_input_capture_and_shutdown(tmp_path):
    from mojive.ui import ToolHint

    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    with launch_passive(
        model,
        data,
        width=640,
        height=480,
        show_window=False,
    ) as viewer:
        assert viewer.model is model and viewer.data is data
        assert viewer.is_running()
        assert viewer.max_fps == 60
        viewer.configure_actions((PassiveAction("pause", "space"),))
        viewer.configure_tool_hints((ToolHint("key", "Space", "Pause"),), surface="scene")
        viewer.configure_tool_hints((ToolHint("key", "Space", "Resume"),), surface="status")
        viewer.configure_tool_hints(())
        initial = viewer.stats
        # Display must continue pumping events while the physics owner does no work.
        time.sleep(0.2)
        later = viewer.stats
        assert later["rendered_frames"] > initial["rendered_frames"]
        assert data.time == 0.0
        for index in range(50):
            mujoco.mj_step(model, data)
            viewer.sync(step=index + 1)
        expected = data.time
        image = viewer.capture_array()
        assert image.dtype == np.uint8 and image.shape[2] == 3
        assert np.ptp(image) > 50
        with SharedImage(image.shape) as target:
            for _ in range(2):
                metadata = viewer.capture_into(target)
                assert metadata["transport"] == "shared_memory"
                assert "image" not in metadata
                np.testing.assert_array_equal(target.array, image)
        assert data.time == expected
        socket_path = viewer.start_rpc(tmp_path / "passive.sock")
        with RpcClient(socket_path, timeout=5) as client:
            assert not client.describe_operations(name="resume")["operations"][0]["available"]
            with pytest.raises(RpcError):
                client.call("pause")
            with pytest.raises(RpcError):
                client.call("step")
            client.set_ctrl(np.full(model.nu, 0.15))
            time.sleep(0.04)
            viewer.sync()
            np.testing.assert_allclose(data.ctrl, 0.15)
            # Applied once: a later policy action must not be overwritten by a stale UI value.
            data.ctrl[:] = 0.05
            time.sleep(0.04)
            viewer.sync()
            np.testing.assert_allclose(data.ctrl, 0.05)
            rpc_image = client.capture_array(width=64, height=48)
            assert rpc_image.shape == (48, 64, 3)
            with SharedImage(rpc_image.shape) as target:
                client.capture_into(target)
                np.testing.assert_array_equal(target.array, rpc_image)
        viewer.stop_rpc()
        assert not Path(socket_path).exists()
        assert data.time == expected
        assert not list(tmp_path.glob("*.png"))
        worker = viewer._process
    assert not viewer.is_running() and not worker.is_alive()
    viewer.close()


def test_passive_debug_commands_reach_the_display():
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    with launch_passive(model, data, width=640, height=480, show_window=False) as viewer:
        viewer.set_camera(CameraView(eye=(0.0, -5.0, 2.0), target=(0.0, 0.0, 0.7)))
        baseline = viewer.capture_array(surface="viewport")
        baseline_primitives = viewer.stats["debug_commands"]["primitives"]
        viewer.publish_debug_commands(
            (
                {
                    "op": "arrow",
                    "layer": "policy.velocity",
                    "occlusion": "always",
                    "id": "target",
                    "a": [-1.5, 0.0, 1.2],
                    "b": [1.5, 0.0, 1.2],
                    "color": [0.2, 0.95, 0.3, 1.0],
                    "width_px": 12.0,
                },
            )
        )
        deadline = time.monotonic() + 2.0
        image = baseline
        while np.array_equal(image, baseline) and time.monotonic() < deadline:
            time.sleep(0.02)
            image = viewer.capture_array(surface="viewport")

        debug_stats = viewer.stats["debug_commands"]
        assert debug_stats["applied"] == 1
        assert debug_stats["dropped"] == debug_stats["invalid"] == debug_stats["queued"] == 0
        assert debug_stats["primitives"] == baseline_primitives + 1
        assert debug_stats["notes"] == []
        assert not np.array_equal(image, baseline)
        green = image[..., 1].astype(np.int16) - image[..., 0].astype(np.int16)
        assert green.max() > 80


def test_passive_startup_failure_is_reported_and_reaped():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(RuntimeError, match="renderer"):
        launch_passive(model, mujoco.MjData(model), renderer="missing", show_window=False)


def test_passive_body_tracking_uses_published_pose_without_stepping_physics(tmp_path):
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    with launch_passive(model, data, width=800, height=600, show_window=False) as viewer:
        viewer.set_camera(CameraView(eye=np.array((4, -6, 3)), target=np.array((0, 0, 1))))
        viewer.configure_tracking(CameraTrackingConfig(axes="xy", smoothing=0))
        viewer.track_body("free_body")
        socket_path = viewer.start_rpc(tmp_path / "tracking.sock")
        with RpcClient(socket_path, timeout=5) as client:
            data.qpos[:3] = (2, 3, 8)
            mujoco.mj_forward(model, data)
            viewer.capture_array()
            assert client.call("get_viewport_camera")["target"] == pytest.approx((2, 3, 1))
            viewer.configure_tracking(CameraTrackingConfig(axes="xyz", smoothing=0))
            viewer.capture_array()
            assert client.call("get_viewport_camera")["target"] == pytest.approx((2, 3, 8))
            viewer.track_body(None)
            data.qpos[:3] = (5, 6, 9)
            viewer.capture_array()
            assert client.call("get_viewport_camera")["target"] == pytest.approx((2, 3, 8))
        assert data.time == 0


def test_passive_video_lifecycle_uses_display_time_and_finalizes_on_close(tmp_path):
    import contextlib

    import imageio_ffmpeg

    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    output = tmp_path / "policy.mp4"
    with launch_passive(
        model,
        data,
        width=640,
        height=480,
        show_window=False,
        config=ViewerConfig(recording=RecordingConfig(countdown=0, run_simulation=True)),
    ) as viewer:
        viewer.configure_recording(RecordingConfig(fps=20, countdown=0, run_simulation=True))
        viewer.configure_actions((PassiveAction("pause", "space", "Pause policy", "toggle"),))
        viewer.set_status("Policy running", paused=False)
        assert viewer.poll_events() == ()
        height, width = viewer.capture_array(surface="window").shape[:2]
        assert viewer.start_recording(output, surface="window") == output
        deadline = time.monotonic() + 5
        while viewer.recording.frames < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert viewer.recording.frames >= 2
        assert data.time == 0  # run_simulation must not touch the caller's clock.
        assert viewer.pause_recording()
        paused = viewer.recording
        assert paused.phase is RecordingPhase.PAUSED
        for step in range(20):
            mujoco.mj_step(model, data)
            viewer.sync(step=step + 1)
            time.sleep(0.005)
        assert viewer.recording.frames == paused.frames
        assert data.time > 0
        assert viewer.resume_recording()
        time.sleep(0.12)
        assert viewer.recording.frames > paused.frames
        expected = data.time
        assert viewer.stop_recording() == output
        assert not viewer.recording.active and not viewer.recording.error
        assert data.time == expected
        # Closing the handle also explicitly completes an active encoder.
        last = tmp_path / "on-close.mp4"
        viewer.start_recording(last, surface="window", countdown=0)
        deadline = time.monotonic() + 5
        while viewer.recording.frames == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert viewer.recording.frames > 0
    for path in (output, last):
        with contextlib.closing(imageio_ffmpeg.read_frames(str(path))) as frames:
            metadata = next(frames)
            assert metadata["size"] == (width, height)
            assert len(list(frames)) > 0


def test_passive_reports_async_encoder_failure_and_can_record_again(tmp_path):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with launch_passive(
        model, mujoco.MjData(model), width=640, height=480, show_window=False
    ) as viewer:
        viewer.start_recording(tmp_path / "unsupported.extension", countdown=0, surface="window")
        deadline = time.monotonic() + 5
        while viewer.recording.active and time.monotonic() < deadline:
            time.sleep(0.02)
        assert viewer.recording.error
        with pytest.raises(RuntimeError):
            viewer.stop_recording()
        viewer.start_recording(tmp_path / "recovered.mp4", countdown=30, surface="window")
        assert not viewer.recording.error
        assert viewer.stop_recording() is None


def test_policy_owned_actuators_reject_writes_but_keep_capture_available(tmp_path):
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.2
    with launch_passive(
        model, data, width=640, height=480, show_window=False, control_writeback=False
    ) as viewer:
        socket_path = viewer.start_rpc(tmp_path / "policy.sock")
        with (
            RpcClient(socket_path, timeout=5) as client,
            pytest.raises(RpcError, match="write_ctrl"),
        ):
            client.set_ctrl(np.full(model.nu, 0.7))
        viewer.sync()
        np.testing.assert_array_equal(data.ctrl, 0.2)
        assert viewer.capture_array().shape[2] == 3
