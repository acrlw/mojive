"""Scene-source conversion preserves geometry identity and retained mesh storage."""

from __future__ import annotations

import numpy as np
import pytest

from mojive.adapters.base import FrameNeeds
from mojive.scene.assets import resolve
from mojive.types import InstancePoseSource, MeshShape

mujoco = pytest.importorskip("mujoco")
from mojive.adapters.mujoco import MuJoCoAdapter  # noqa: E402

pytestmark = pytest.mark.physics


@pytest.mark.parametrize(
    "kind,shapes,scales",
    [
        ("sphere", [MeshShape.SPHERE], [[0.1, 0.1, 0.1]]),
        ("ellipsoid", [MeshShape.SPHERE], [[0.1, 0.2, 0.3]]),
        ("box", [MeshShape.BOX], [[0.1, 0.2, 0.3]]),
        ("cylinder", [MeshShape.CYLINDER], [[0.1, 0.1, 0.2]]),
        (
            "capsule",
            [MeshShape.CAPSULE_SHAFT, MeshShape.CAPSULE_CAP, MeshShape.CAPSULE_CAP],
            [[0.1, 0.1, 0.2], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        ),
    ],
)
def test_geom_and_site_share_primitives_but_keep_separate_identity(kind, shapes, scales):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <asset><material name="paint" rgba=".2 .4 .6 .8"/></asset>
      <worldbody><body name="moving"><freejoint/>
        <geom name="shape" type="{kind}" size=".1 .2 .3" material="paint"/>
        <site name="marker" type="{kind}" size=".1 .2 .3" material="paint"/>
        <site name="hidden" type="{kind}" size=".1 .2 .3" group="3"/>
      </body></worldbody>
    </mujoco>
    """)
    adapter = MuJoCoAdapter()
    try:
        adapter.load_model(model)
        source = adapter.scene_source()
        assert adapter.scene_source() is source
        assert source.nodes is adapter.nodes()
        part_count = len(shapes)
        assert source.instance_count == 2 * part_count
        assert [key.shape for key in source.geom_mesh] == shapes * 2
        np.testing.assert_allclose(source.geom_size, scales * 2)
        expected_locals = np.repeat(np.eye(4, dtype=np.float32)[None], part_count, axis=0)
        if kind == "capsule":
            expected_locals[1, 2, 3] = 0.2
            expected_locals[2, 2, 3] = -0.2
            expected_locals[2, 1, 1] = expected_locals[2, 2, 2] = -1
        np.testing.assert_array_equal(source.geom_local, np.tile(expected_locals, (2, 1, 1)))

        shape_node = next(node for node in source.nodes if node.name == "shape")
        marker_node = next(node for node in source.nodes if node.name == "marker")
        body_node = next(node for node in source.nodes if node.name == "moving")
        assert (
            source.geom_node.tolist()
            == [shape_node.node_id] * part_count + [marker_node.node_id] * part_count
        )
        assert (
            source.geom_object_id.tolist() == [body_node.object_id] * part_count + [0] * part_count
        )
        assert (
            source.geom_pose_source.tolist()
            == [InstancePoseSource.GEOM] * part_count + [InstancePoseSource.SITE] * part_count
        )
        assert source.geom_source.tolist() == [0] * (2 * part_count)
        assert (
            source.geom_segmentation.tolist()
            == [[0, int(mujoco.mjtObj.mjOBJ_GEOM)]] * part_count
            + [[0, int(mujoco.mjtObj.mjOBJ_SITE)]] * part_count
        )
        assert not source.geom_static.any()
        assert source.geom_material == [0] * (2 * part_count)
        assert source.materials[0].name == "paint"
        np.testing.assert_allclose(source.geom_rgba, [[0.2, 0.4, 0.6, 0.8]] * (2 * part_count))

        assert adapter.set_visual_group("site", 3, True)
        rebuilt = adapter.scene_source()
        assert rebuilt is not source
        assert rebuilt.instance_count == 3 * part_count
        assert rebuilt.geom_source[-part_count:].tolist() == [1] * part_count
    finally:
        adapter.release()


def test_deformable_mesh_buffers_are_shared_across_frames_and_replaced_with_source():
    adapter = MuJoCoAdapter(resolve("deformables"))
    try:
        source = adapter.scene_source()
        frame = adapter.frame(FrameNeeds(deformables=True))
        updates = frame.mesh_updates
        assert source.dynamic_meshes == updates.keys()
        for key in source.dynamic_meshes:
            assert source.meshes[key].positions is updates[key].positions
            assert source.meshes[key].normals is updates[key].normals

        before = {key: update.positions.copy() for key, update in updates.items()}
        adapter.step(2)
        assert adapter.frame(FrameNeeds(deformables=True)).mesh_updates is updates
        assert adapter.scene_source() is source
        assert any(not np.array_equal(before[key], updates[key].positions) for key in updates)

        # Visibility rebuilds the source; the old source remains a valid snapshot
        # with its own retained mesh buffers while the adapter advances the new one.
        assert adapter.set_visual_group("skin", 0, False)
        rebuilt = adapter.scene_source()
        rebuilt_updates = adapter.frame(FrameNeeds(deformables=True)).mesh_updates
        assert rebuilt is not source
        assert rebuilt_updates is not updates
        assert not any(key.shape is MeshShape.SKIN for key in rebuilt.dynamic_meshes)
        retained = {key: update.positions.copy() for key, update in updates.items()}
        for key in rebuilt.dynamic_meshes:
            assert rebuilt.meshes[key].positions is rebuilt_updates[key].positions
            assert rebuilt.meshes[key].positions is not source.meshes[key].positions
        adapter.step(2)
        adapter.frame(FrameNeeds(deformables=True))
        for key, positions in retained.items():
            np.testing.assert_array_equal(updates[key].positions, positions)
    finally:
        adapter.release()
