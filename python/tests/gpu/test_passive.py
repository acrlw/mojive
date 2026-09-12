"""Real-window ownership, asynchronous progress, and passive input exchange."""

import time
from pathlib import Path

import numpy as np
import pytest

from mojive import CameraTrackingConfig, CameraView, SharedImage, build, launch_passive
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
