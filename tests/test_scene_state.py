"""Camera bookmark and scene snapshot tests."""

from __future__ import annotations

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.render.backend import RenderFlag
from forge_viewer.scene_state import (
    CAMERA_BOOKMARK_FORMAT,
    FORMAT_VERSION,
    SCENE_SNAPSHOT_FORMAT,
    apply_camera_bookmark,
    camera_bookmark,
    capture_scene,
    delete_named_snapshot,
    list_named_snapshots,
    load_named_snapshot,
    restore_scene,
    save_named_snapshot,
)
from forge_viewer.session import Session
from forge_viewer.ui.camera import OrbitCamera

pytestmark = [pytest.mark.integration, pytest.mark.physics]


MODEL = """
<mujoco model="snapshot">
  <worldbody>
    <light name="sun" pos="0 0 3"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".2 .2 .2" rgba=".2 .5 .8 1"/>
    </body>
  </worldbody>
  <keyframe><key name="rest" time=".25" qpos="0 0 1 1 0 0 0"/></keyframe>
</mujoco>
"""


class Backend:
    def __init__(self):
        self.flags = {RenderFlag.SHADOW: True, RenderFlag.TENDON: False}

    def render_options(self):
        return tuple(self.flags)

    def get_flag(self, flag):
        return self.flags[flag]

    def set_flag(self, flag, enabled):
        self.flags[flag] = bool(enabled)
        return True


@pytest.fixture
def state_rig(tmp_path):
    path = tmp_path / "snapshot.xml"
    path.write_text(MODEL)
    adapter = MuJoCoAdapter(path)
    session = Session(adapter, path)
    session.submit(cmd.Pause())
    yield session, Backend(), OrbitCamera()
    adapter.release()


def test_camera_bookmark_restores_projection_and_orbit_values():
    camera = OrbitCamera(
        pivot=[1.0, 2.0, 3.0], distance=7.0, yaw=42.0, pitch=-17.0, orthographic=True
    )
    saved = camera_bookmark(camera, camera.view(), source=-1)
    camera.pivot = np.zeros(3)
    camera.yaw = 0.0
    camera.pitch = 0.0
    camera.distance = 1.0
    view = apply_camera_bookmark(saved, camera)
    assert camera.pivot == pytest.approx([1.0, 2.0, 3.0])
    assert (camera.yaw, camera.pitch, camera.distance) == pytest.approx((42.0, -17.0, 7.0))
    assert view.orthographic
    assert saved["format"] == CAMERA_BOOKMARK_FORMAT


def test_camera_bookmark_requires_the_current_format_and_version():
    camera = OrbitCamera()
    saved = camera_bookmark(camera, camera.view())
    missing_format = saved.copy()
    missing_format.pop("format")
    with pytest.raises(ValueError, match="Unsupported camera bookmark format: None"):
        apply_camera_bookmark(missing_format, camera)

    saved["version"] = 2
    with pytest.raises(ValueError, match="Unsupported camera bookmark version: 2"):
        apply_camera_bookmark(saved, camera)


def test_scene_snapshot_restores_physics_view_flags_and_selection(state_rig):
    session, backend, camera = state_rig
    selected = next(node for node in session.nodes if node.name == "box")
    session.submit(cmd.Select(selected.object_id))
    session.adapter.data.qpos[:3] = [0.2, -0.3, 1.4]
    session.adapter.data.qvel[:3] = [1.0, 2.0, 3.0]
    session.adapter.data.time = 2.5
    camera.yaw = 33.0
    snapshot = capture_scene(session, backend, camera)
    assert snapshot["format"] == SCENE_SNAPSHOT_FORMAT
    assert snapshot["version"] == FORMAT_VERSION

    session.adapter.data.qpos[:] = 0.0
    session.adapter.data.qvel[:] = 0.0
    session.adapter.data.time = 0.0
    backend.set_flag(RenderFlag.SHADOW, False)
    session.submit(cmd.Select(0))
    camera.yaw = -80.0
    restore_scene(snapshot, session, backend, camera)

    assert session.adapter.data.qpos[:3] == pytest.approx([0.2, -0.3, 1.4])
    assert session.adapter.data.qvel[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert session.adapter.data.time == pytest.approx(2.5)
    assert backend.get_flag(RenderFlag.SHADOW)
    assert session.selected == selected.object_id
    assert camera.yaw == pytest.approx(33.0)


def test_scene_snapshot_preserves_a_non_pickable_node_selection(state_rig):
    session, backend, camera = state_rig
    node = next(item for item in session.nodes if not item.object_id)
    assert session.submit(cmd.SelectNode(node.node_id)).ok
    snapshot = capture_scene(session, backend, camera)
    session.submit(cmd.Select(0))

    restore_scene(snapshot, session, backend, camera)

    assert session.selected == 0
    assert session.selected_node is not None
    assert session.selected_node.node_id == node.node_id


def test_named_snapshot_storage_overwrites_lists_and_deletes(tmp_path):
    first = {"version": 1, "value": 1}
    second = {"version": 1, "value": 2}
    path = save_named_snapshot("pose one", first, tmp_path)
    assert path.name == "pose-one.json"
    save_named_snapshot("pose one", second, tmp_path)
    assert list_named_snapshots(tmp_path) == ["pose-one"]
    assert load_named_snapshot("pose-one", tmp_path) == second
    delete_named_snapshot("pose-one", tmp_path)
    assert list_named_snapshots(tmp_path) == []


def test_snapshot_rejects_a_different_model(state_rig):
    session, backend, camera = state_rig
    snapshot = capture_scene(session, backend, camera)
    snapshot["asset"] = "/different/model.xml"
    with pytest.raises(ValueError, match="different model"):
        restore_scene(snapshot, session, backend, camera)


def test_scene_snapshot_rejects_non_current_versions(state_rig):
    session, backend, camera = state_rig
    snapshot = capture_scene(session, backend, camera)
    snapshot.pop("format")
    snapshot["version"] = 1
    with pytest.raises(ValueError, match="Unsupported scene snapshot version: 1"):
        restore_scene(snapshot, session, backend, camera)

    snapshot["version"] = FORMAT_VERSION + 1
    with pytest.raises(
        ValueError, match=f"Unsupported scene snapshot version: {FORMAT_VERSION + 1}"
    ):
        restore_scene(snapshot, session, backend, camera)
