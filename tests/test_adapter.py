from __future__ import annotations

import gc
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest

from forge_viewer.adapters.base import FrameNeeds, JointVisualKind, NodeKind

pytestmark = pytest.mark.physics

mujoco = pytest.importorskip("mujoco", reason="需要装 mujoco 才能建真实世界")

from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter  # noqa: E402

FIXTURE_XML = """
<mujoco model="adapter_fixture">
  <compiler angle="radian"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8"
             rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <texture name="sky" type="skybox" builtin="gradient" width="4" height="4"
             rgb1=".3 .5 .7" rgb2="0 0 0"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
              specular=".3" shininess=".4" emission=".1" reflectance=".2"/>
    <mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 .05" material="grid"/>
    <geom name="wall" type="box" pos="2 0 .5" size=".1 1 .5"/>
    <!-- 不带关节的地标：MuJoCo 眼里它是"静态"（焊在世界上），但它**不是世界 body**，
         有自己的 object_id，点得中它。射线那条路的判据就落在这里 -->
    <body name="post" pos="-2 0 .5">
      <geom name="post_box" type="box" size=".2 .2 .5"/>
      <!-- 组 3 的包络球：画不出来，也不该挡住射线（它比 post_box 高 0.3 m，
           挡住了的话命中距离会短 0.3） -->
      <geom name="post_hull" type="sphere" size=".8" group="3"/>
    </body>
    <body name="free_body" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="cap" type="capsule" size=".05 .2" rgba="1 0 0 1"/>
      <geom name="head" type="sphere" size=".08" pos="0 0 .3"/>
      <site name="grip" pos="0 0 .3"/>
    </body>
    <body name="arm" pos="1 0 1">
      <joint name="h1" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
      <geom name="upper" type="mesh" mesh="tet"/>
      <body name="fore" pos="0 0 -.3">
        <joint name="h2" type="hinge" axis="0 1 0"/>
        <geom name="lower" type="box" size=".05 .05 .15"/>
        <geom name="hidden" type="sphere" size=".2" group="3"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="m1" joint="h1" ctrlrange="-1 1"/></actuator>
  <sensor><framepos name="root_pos" objtype="body" objname="free_body"/></sensor>
  <keyframe>
    <key name="pose" time="1.25" qpos="0 0 1 1 0 0 0 .25 -.5" ctrl=".75"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture(scope="module")
def fixture_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mjcf") / "adapter_fixture.xml"
    path.write_text(FIXTURE_XML)
    return path


@pytest.fixture
def adapter(fixture_path):
    a = MuJoCoAdapter(fixture_path)
    yield a
    a.release()


def test_scene_source_arrays_agree(adapter):

    src = adapter.scene_source()
    n = src.instance_count
    assert n > 0
    for name in (
        "geom_material",
        "geom_size",
        "geom_rgba",
        "geom_object_id",
        "geom_body",
        "geom_source",
        "geom_local",
        "geom_infinite_plane",
    ):
        assert len(getattr(src, name)) == n, f"{name} 长度 {len(getattr(src, name))} != {n}"
    assert src.geom_size.shape == (n, 3)
    assert src.geom_rgba.shape == (n, 4)
    assert src.geom_local.shape == (n, 4, 4)
    assert src.geom_object_id.dtype == np.uint32

    assert all(0 <= i < len(src.materials) for i in src.geom_material)


def test_bundled_test_scene_loads(fixture_path):

    resolve = pytest.importorskip("forge_viewer.assets", reason="assets 模块还没落地").resolve
    a = MuJoCoAdapter(resolve("test_scene"))
    src = a.scene_source()
    n = src.instance_count
    assert n > 0
    for name in ("geom_material", "geom_size", "geom_rgba", "geom_object_id", "geom_source"):
        assert len(getattr(src, name)) == n
    assert src.geom_infinite_plane.any(), "test_scene 的地板是无限平面"
    a.release()


def test_capsule_splits_into_three_instances(adapter):

    src = adapter.scene_source()
    m = adapter.model
    cap_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cap")
    rows = np.flatnonzero(src.geom_source == cap_geom)
    assert len(rows) == 3

    shapes = [src.geom_mesh[i].shape for i in rows]
    assert shapes.count("capsule_shaft") == 1
    assert shapes.count("capsule_cap") == 2

    assert len(set(src.geom_source[rows].tolist())) == 1

    caps = [i for i in rows if src.geom_mesh[i].shape == "capsule_cap"]
    offsets = sorted(float(src.geom_local[i][2, 3]) for i in caps)
    half = float(m.geom_size[cap_geom][1])
    assert offsets == pytest.approx([-half, half])
    for i in caps:
        assert src.geom_size[i] == pytest.approx([m.geom_size[cap_geom][0]] * 3)

    lo = next(i for i in caps if src.geom_local[i][2, 3] < 0)
    hi = next(i for i in caps if src.geom_local[i][2, 3] > 0)
    assert src.geom_local[hi][2, 2] > 0
    assert src.geom_local[lo][2, 2] < 0, "下端帽没翻过来"

    assert np.linalg.det(src.geom_local[lo][:3, :3]) > 0


def test_object_id_is_per_body(adapter):

    src = adapter.scene_source()
    m = adapter.model
    by_name = {
        name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("floor", "wall", "cap", "head", "lower")
    }

    def ids(geom_name):
        return set(src.geom_object_id[src.geom_source == by_name[geom_name]].tolist())

    assert ids("floor") == {0}, "地板属于世界 body"
    assert ids("wall") == {0}
    assert ids("cap") == ids("head"), "同一个 body 的两个 geom 必须同 id"
    assert ids("cap") != {0}
    assert ids("cap") != ids("lower"), "不同 body 必须不同 id"

    nodes = adapter.nodes()
    body_ids = {n.object_id for n in nodes if n.kind in (NodeKind.LINK, NodeKind.ROBOT)}
    assert ids("cap") <= body_ids


def test_hidden_group_geoms_are_skipped(adapter):

    src = adapter.scene_source()
    m = adapter.model
    hidden = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "hidden")
    assert hidden >= 0
    assert hidden not in src.geom_source.tolist()
    assert not any(n.name == "hidden" for n in adapter.nodes())


def test_infinite_plane_flagged(adapter):

    src = adapter.scene_source()
    rows = np.flatnonzero(src.geom_infinite_plane)
    assert len(rows) == 1
    assert src.geom_mesh[rows[0]].shape == "plane"
    assert src.geom_size[rows[0]][0] == 0.0
    assert src.geom_size[rows[0]][1] == 0.0


def test_mesh_without_texcoord_gets_zero_uv(adapter):

    src = adapter.scene_source()
    mesh_keys = [k for k in src.meshes if k.shape == "asset"]
    assert mesh_keys
    data = src.meshes[mesh_keys[0]]
    assert len(data.uvs) == len(data.positions)
    assert np.all(data.uvs == 0.0)
    assert data.indices.dtype == np.uint32
    assert data.triangle_count > 0


def test_textures_are_srgb_and_cube_is_six_faces(adapter):

    src = adapter.scene_source()
    assert set(src.textures) == {"grid", "sky"}
    assert all(t.srgb for t in src.textures.values())
    assert src.skybox == "sky"
    sky = src.textures["sky"]
    assert sky.pixels.ndim == 4 and sky.pixels.shape[0] == 6, "cube/skybox 是 6 张面"
    assert src.textures["grid"].pixels.ndim == 3


def test_material_and_lights(adapter):

    src = adapter.scene_source()
    grid = next(m for m in src.materials if m.name == "grid")
    assert grid.emission == pytest.approx(0.1)
    assert grid.specular == pytest.approx(0.3)
    assert grid.shininess == pytest.approx(0.4)
    assert grid.reflectance == pytest.approx(0.2)
    assert grid.texture == "grid"
    assert grid.tex_uniform is True
    assert grid.tex_repeat == pytest.approx([2.0, 2.0])

    lights = src.lights
    assert len(lights.lights) == 1
    assert lights.headlight is not None
    hl = adapter.model.vis.headlight

    assert lights.headlight.diffuse == pytest.approx(np.asarray(hl.diffuse))

    assert lights.ambient == pytest.approx(np.asarray(hl.ambient))

    node = next(node for node in adapter.nodes() if node.kind is NodeKind.LIGHT)
    assert node.object_id and node.light_index == 0
    edited = replace(lights.lights[0], diffuse=np.array([0.2, 0.3, 0.4], np.float32))
    assert adapter.set_light(0, edited)
    frame = adapter.frame(FrameNeeds())
    assert frame.lights is not None
    assert frame.lights.lights[0].diffuse == pytest.approx([0.2, 0.3, 0.4])


def test_forge_area_light_survives_mujoco_writeback(adapter, fixture_path):
    from forge_viewer import commands as cmd
    from forge_viewer.session import Session
    from forge_viewer.types import LightKind

    session = Session(adapter, fixture_path)
    source_light = session.source.lights.lights[0]
    edited = replace(source_light, kind=LightKind.AREA, area_radius=0.35)

    assert session.submit(cmd.SetLight(0, edited))
    assert session.source.lights.lights[0].kind is LightKind.AREA
    assert int(adapter.model.light_type[0]) == int(mujoco.mjtLightType.mjLIGHT_POINT)

    frame = session.tick(FrameNeeds())
    assert frame.lights.lights[0].kind is LightKind.AREA
    assert frame.lights.lights[0].position == pytest.approx(adapter.data.light_xpos[0])


def test_initial_values_come_from_the_model(adapter):

    m = adapter.model
    expect = np.asarray(m.qpos0, np.float32).copy()
    adapter.step(200)
    assert not np.allclose(adapter.data.qpos, expect), "跑了 200 步还没动，判据挡不住东西"
    src = adapter.scene_source()
    assert src.initial_qpos == pytest.approx(expect)
    assert len(src.initial_ctrl) == m.nu
    assert np.all(src.initial_ctrl == 0.0)


def test_frame_produces_only_what_is_needed(adapter):

    f = adapter.frame(FrameNeeds(qvel=False))
    assert f.qvel is None
    assert f.qpos is None
    assert f.contacts is None
    assert f.sensors is None
    assert f.geom_xpos is not None and f.geom_xmat is not None

    f = adapter.frame(FrameNeeds(qvel=True, qpos=True))
    assert f.qvel is not None and len(f.qvel) == adapter.model.nv
    assert f.qpos is not None and len(f.qpos) == adapter.model.nq

    f = adapter.frame(FrameNeeds.none())
    assert f.geom_xpos is None, "连位姿都没申报时不该去取"


def test_diagnostic_metadata_and_frame_match_mujoco(adapter):
    source = adapter.scene_source().diagnostics
    model, data = adapter.model, adapter.data
    expected_kinds = np.asarray(
        [
            JointVisualKind.FREE,
            JointVisualKind.HINGE,
            JointVisualKind.HINGE,
        ],
        np.uint8,
    )
    assert source.joint_kinds == pytest.approx(expected_kinds)
    assert source.joint_visible.tolist() == [True] * model.njnt
    assert source.joint_length == pytest.approx(model.stat.meansize * model.vis.scale.jointlength)
    assert source.joint_width == pytest.approx(model.stat.meansize * model.vis.scale.jointwidth)
    assert source.joint_rgba == pytest.approx(model.vis.rgba.joint)

    expected_com = np.flatnonzero(np.asarray(model.body_parentid[1:]) == 0) + 1
    assert source.com_bodies == pytest.approx(expected_com)
    assert source.com_radius == pytest.approx(model.stat.meansize * model.vis.scale.com)

    moving = np.flatnonzero(np.asarray(model.body_dofnum) > 0)
    assert source.inertia_bodies == pytest.approx(moving)
    mass = np.asarray(model.body_mass[moving])
    assert 8.0 * np.prod(source.scaled_inertia_sizes, axis=1) == pytest.approx(mass / 1000.0)

    assert adapter.frame(FrameNeeds.none()).diagnostics is None
    frame = adapter.frame(FrameNeeds(poses=False, diagnostics=True)).diagnostics
    assert frame.joint_xpos == pytest.approx(data.xanchor)
    assert frame.joint_xaxis == pytest.approx(data.xaxis)
    assert frame.subtree_com == pytest.approx(data.subtree_com)
    assert frame.body_xipos == pytest.approx(data.xipos)
    assert frame.body_ximat == pytest.approx(data.ximat.reshape(model.nbody, 3, 3))


def test_pose_fetch_allocates_nothing(adapter):

    needs = FrameNeeds()
    adapter.frame(needs)
    buf_ids = {id(adapter._geom_xpos_buf), id(adapter._geom_xmat_buf)}

    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    start, _ = tracemalloc.get_traced_memory()
    for _ in range(100):
        adapter.frame(needs)
        buf_ids.add(id(adapter._geom_xpos_buf))
        buf_ids.add(id(adapter._geom_xmat_buf))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(buf_ids) == 2, "暂存数组被换过对象——那就是每帧在分配"
    assert peak - start < 16 * 1024, f"取姿态峰值 {peak - start} B"
    assert current - start < 8 * 1024


def test_slow_path_agrees_with_fast_path(adapter):

    assert adapter.fast_pose, "本机应当走整段拷贝那条"
    adapter.step(20)
    fast = adapter.frame(FrameNeeds())
    fast_pos, fast_mat = fast.geom_xpos.copy(), fast.geom_xmat.copy()

    adapter._fast_pose = False
    slow = adapter.frame(FrameNeeds())
    assert np.array_equal(slow.geom_xpos, fast_pos)
    assert np.array_equal(slow.geom_xmat, fast_mat)
    adapter._fast_pose = True

    assert np.allclose(fast_pos, adapter.data.geom_xpos, atol=1e-6)


def test_frame_poses_track_the_physics(adapter):

    first = adapter.frame(FrameNeeds()).geom_xpos.copy()
    adapter.step(100)
    later = adapter.frame(FrameNeeds()).geom_xpos
    assert not np.allclose(first, later)


def test_set_qpos_takes_a_qpos_address(adapter):

    joints = {j.name: j for j in adapter.joints()}
    assert joints["root"].qpos_adr == 0 and joints["root"].dof == 6
    assert joints["h1"].qpos_adr == 7

    assert adapter.set_qpos(1, 0.5)
    qpos = adapter.data.qpos
    assert qpos[1] == pytest.approx(0.5), "写的应当是 qpos[1]"
    assert qpos[7] == pytest.approx(0.0), "第 1 个关节（h1）不该被碰到"

    assert adapter.set_qpos(joints["h1"].qpos_adr, 0.3)
    assert adapter.data.qpos[7] == pytest.approx(0.3)
    assert not adapter.set_qpos(9999, 0.0), "越界要如实报失败，不能静默"


def test_set_qpos_on_joint_types_scene():

    resolve = pytest.importorskip("forge_viewer.assets", reason="assets 模块还没落地").resolve
    a = MuJoCoAdapter(resolve("joint_types"))
    joints = {j.name: j for j in a.joints()}
    assert joints["free"].qpos_adr == 0 and joints["free"].dof == 6
    ball = joints["ball"]
    assert ball.qpos_adr == 7, "free joint 占 7 个 qpos，ball 从第 7 个开始"

    before = a.data.qpos.copy()
    assert a.set_qpos(2, 0.75)
    assert a.data.qpos[2] == pytest.approx(0.75)

    slide = joints["slide"]
    assert a.data.qpos[slide.qpos_adr] == pytest.approx(before[slide.qpos_adr])

    body_kinds = (NodeKind.WORLD, NodeKind.ROBOT, NodeKind.LINK)
    nodes = {n.name: n for n in a.nodes() if n.kind in body_kinds}
    assert nodes["free_body"].posable
    assert nodes["free_chain"].posable
    for name in ("ball_body", "slide_body", "hinge_body", "chain_root", "chain_0"):
        assert not nodes[name].posable, f"{name} 是被关节驱动的，手柄不该对它出现"
    a.release()


def test_set_ctrl_clips_to_range(adapter):
    assert adapter.set_ctrl(0, 5.0)
    assert adapter.data.ctrl[0] == pytest.approx(1.0), "超出 ctrlrange 要夹住"
    assert not adapter.set_ctrl(7, 0.0)


def test_set_pose_only_on_free_bodies(adapter):

    nodes = {n.name: n for n in adapter.nodes()}
    assert nodes["free_body"].posable
    assert not nodes["arm"].posable
    assert not nodes["fore"].posable
    assert not nodes["world"].posable

    adapter.step(100)
    assert np.linalg.norm(adapter.data.qvel[:6]) > 1e-3

    ok = adapter.set_pose(nodes["free_body"].node_id, np.array([1.0, 2.0, 3.0]), np.eye(3))
    assert ok
    assert adapter.data.qpos[:3] == pytest.approx([1.0, 2.0, 3.0])

    assert adapter.data.qvel[:6] == pytest.approx(np.zeros(6))
    assert not adapter.set_pose(nodes["arm"].node_id, np.zeros(3), np.eye(3))


def test_perturb_uses_mujoco_viewer_mass_scaled_spring(adapter):

    nodes = {n.name: n for n in adapter.nodes()}
    body = nodes["free_body"].body_index
    target = np.asarray(adapter.data.xpos[body]).copy() + np.array([0.1, 0.0, 0.0])
    assert adapter.apply_perturb(nodes["free_body"].node_id, target, np.eye(3), "translate")
    applied = np.asarray(adapter.data.xfrc_applied[body])
    assert applied[0] > 0.0
    assert np.linalg.norm(applied[:3]) > 0.1 * float(adapter.model.body_mass[body])

    adapter.clear_perturb()
    assert np.all(adapter.data.xfrc_applied == 0.0)
    assert not adapter.apply_perturb(nodes["world"].node_id, target, np.eye(3), "translate")


def test_raycast_treats_the_world_body_as_a_miss(adapter):

    nodes = {n.name: n for n in adapter.nodes()}

    obj, dist = adapter.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    assert obj == nodes["free_body"].object_id
    assert 0 < dist < 3.0

    obj, dist = adapter.raycast(np.array([8.0, 8.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    assert obj == 0
    assert dist == float("inf")

    obj, dist = adapter.raycast(np.array([-2.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    assert obj == nodes["post"].object_id != 0, "不带关节的 body 也该点得中"
    assert dist == pytest.approx(2.0, abs=1e-6), "被没画出来的碰撞几何挡住了"

    obj_a, dist_a = adapter.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    obj_b, dist_b = adapter.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -7.0]))
    assert obj_a == obj_b
    assert dist_a == pytest.approx(dist_b)


def test_caps_report_what_is_not_there(adapter):

    caps = adapter.caps
    assert caps.name == "mujoco"
    assert caps.inverse_kinematics is False
    assert (
        caps.write_pose
        and caps.write_qpos
        and caps.perturb
        and caps.raycast
        and caps.reload
        and caps.keyframes
        and caps.sensors
    )
    assert adapter.set_paused(True)


def test_keyframes_are_listed_and_restore_the_complete_state(adapter, fixture_path):
    from forge_viewer import commands as cmd
    from forge_viewer.session import Session

    keys = adapter.keyframes()
    assert [(k.keyframe_id, k.name, k.time) for k in keys] == [(0, "pose", 1.25)]

    adapter.data.qpos[:] = 0.0
    adapter.data.ctrl[:] = 0.0
    assert adapter.load_keyframe(0)
    assert adapter.data.qpos[7:9] == pytest.approx([0.25, -0.5])
    assert adapter.data.ctrl == pytest.approx([0.75])
    assert adapter.data.time == pytest.approx(1.25)

    session = Session(adapter, fixture_path)
    assert session.submit(cmd.Pause()).ok
    result = session.submit(cmd.LoadKeyframe(0))
    assert result.ok and session.active_keyframe == 0
    assert session.keyframes[0].name == "pose"
    assert adapter.data.qpos[7:9] == pytest.approx([0.25, -0.5])


def test_sensor_metadata_addresses_the_frame_values(adapter):
    sensors = adapter.sensors()
    assert len(sensors) == 1
    sensor = sensors[0]
    assert (sensor.name, sensor.kind, sensor.data_adr, sensor.dim) == (
        "root_pos",
        "mjSENS_FRAMEPOS",
        0,
        3,
    )
    frame = adapter.frame(FrameNeeds(poses=False, sensors=True))
    assert frame.sensors[sensor.data_adr : sensor.data_adr + sensor.dim] == pytest.approx(
        adapter.data.sensordata[sensor.data_adr : sensor.data_adr + sensor.dim]
    )


def test_load_failure_keeps_the_original_message(tmp_path):

    bad = tmp_path / "bad.xml"
    bad.write_text('<mujoco><worldbody><geom type="nonsense" size="1"/></worldbody></mujoco>')
    with pytest.raises(RuntimeError) as err:
        MuJoCoAdapter(bad)
    text = str(err.value)
    assert "bad.xml" in text
    assert "nonsense" in text, f"MuJoCo 的原文被吞掉了：{text}"

    missing = tmp_path / "nope.xml"
    with pytest.raises(RuntimeError):
        MuJoCoAdapter(missing)


def test_reload_and_reset(adapter, fixture_path):
    adapter.step(50)
    moved = adapter.data.qpos.copy()
    adapter.reset()
    assert adapter.data.qpos == pytest.approx(adapter.model.qpos0)
    assert not np.allclose(moved, adapter.data.qpos)

    old_model = adapter.model
    adapter.reload()
    assert adapter.model is not old_model
    assert adapter.scene_source().instance_count > 0

    #

    assert len(adapter._geom_xpos_buf) == adapter.model.ngeom
    assert adapter._mj_geom_xpos is adapter.data.geom_xpos
    assert adapter._mj_geom_xmat3.base is adapter.data.geom_xmat
    adapter.step(50)
    f = adapter.frame(FrameNeeds())
    assert np.allclose(f.geom_xpos, adapter.data.geom_xpos, atol=1e-6)
    assert np.linalg.norm(f.geom_xpos - adapter.model.geom_pos) > 1e-4, "50 步之后该动了"


def test_camera_hint_frames_the_scene(adapter):
    cam = adapter.camera_hint()
    assert cam is not None
    extent = adapter.model.stat.extent
    assert cam.near < cam.far
    assert cam.near == pytest.approx(adapter.model.vis.map.znear * extent)
    assert 0.5 * extent < np.linalg.norm(cam.eye - cam.target) < 5.0 * extent


def test_mujoco_visuals_cover_heightfield_sites_and_tendon():
    from forge_viewer.assets import resolve
    from forge_viewer.mujoco_audit import audit_model
    from forge_viewer.render.builder import SceneSourceBuilder
    from forge_viewer.types import CameraView, InstancePoseSource, MeshShape

    a = MuJoCoAdapter(resolve("mujoco_visuals"))
    try:
        src = a.scene_source()
        assert audit_model(a.model)["unsupported"] == 0
        height_rows = [
            i for i, key in enumerate(src.geom_mesh) if key.shape is MeshShape.HEIGHTFIELD
        ]
        assert len(height_rows) == 1
        mesh = src.meshes[src.geom_mesh[height_rows[0]]]
        assert mesh.triangle_count > 20
        assert np.isfinite(mesh.normals).all()

        site_rows = np.flatnonzero(src.geom_pose_source == int(InstancePoseSource.SITE))
        assert len(site_rows) >= 3
        assert all(src.geom_node[i] >= 0 for i in site_rows)

        assert len(a._mj_wrap_points) == a.data.wrap_xpos.size // 3
        frame = a.frame(FrameNeeds(poses=True, tendons=True, actuator=True))
        assert frame.tendon_segments is not None and len(frame.tendon_segments) >= 1
        assert frame.tendon_widths is not None
        assert frame.tendon_widths.shape == (len(frame.tendon_segments),)
        assert np.all(frame.tendon_widths > 0.0)
        assert frame.actuator_activation is not None
        assert src.actuator_tendon.tolist() == [0]
        assert src.actuator_tendon_scale == pytest.approx(a.model.vis.map.actuatortendon)
        assert np.all(
            np.linalg.norm(frame.tendon_segments[:, 1] - frame.tendon_segments[:, 0], axis=1) > 0
        )

        builder = SceneSourceBuilder()
        scene = builder.set_source(src, CameraView())
        builder.update(frame)
        row = int(site_rows[0])
        rendered = scene.transforms[builder.write_index[row]][:3, 3]
        source_site = int(src.geom_source[row])
        assert rendered == pytest.approx(frame.site_xpos[source_site])
    finally:
        a.release()


def test_tendon_material_matches_mujocos_final_color_and_scalars(tmp_path):
    path = tmp_path / "tendon_material.xml"
    path.write_text(
        """<mujoco>
  <asset>
    <texture name="tendon_tex" type="2d" builtin="checker" width="8" height="8"/>
    <material name="tendon_mat" texture="tendon_tex" rgba=".1 .8 .2 .6" emission=".2"
              specular=".7" shininess=".4" reflectance=".3"/>
  </asset>
  <worldbody><site name="a"/><site name="b" pos="1 0 0"/></worldbody>
  <tendon>
    <spatial name="material_color" material="tendon_mat">
      <site site="a"/><site site="b"/>
    </spatial>
    <spatial name="local_color" material="tendon_mat" rgba=".9 .1 .2 .8">
      <site site="a"/><site site="b"/>
    </spatial>
  </tendon>
</mujoco>"""
    )
    a = MuJoCoAdapter(path)
    try:
        source = a.scene_source()
        assert source.tendon_rgba[0] == pytest.approx((0.1, 0.8, 0.2, 0.6))
        assert source.tendon_rgba[1] == pytest.approx((0.9, 0.1, 0.2, 0.8))
        material = source.materials[int(source.tendon_material[0])]
        assert (material.emission, material.specular, material.shininess) == pytest.approx(
            (0.2, 0.7, 0.4)
        )
        assert material.reflectance == pytest.approx(0.3)
        assert material.texture == "tendon_tex"
    finally:
        a.release()


def test_deformables_match_mujocos_abstract_visualization():

    from forge_viewer.assets import resolve
    from forge_viewer.mujoco_audit import audit_model
    from forge_viewer.render.builder import SceneSourceBuilder
    from forge_viewer.types import InstancePoseSource, MeshKey, MeshShape

    a = MuJoCoAdapter(resolve("deformables"))
    try:
        src = a.scene_source()
        assert len(src.dynamic_meshes) == a.model.nflex + a.model.nskin == 4
        assert audit_model(a.model)["unsupported"] == 0
        assert sum(src.geom_pose_source == int(InstancePoseSource.WORLD)) == 4
        deformable_nodes = [
            node for node in a.nodes() if node.kind in (NodeKind.FLEX, NodeKind.SKIN)
        ]
        deformable_ids = {node.object_id for node in deformable_nodes}
        assert len(deformable_ids) == 4
        assert 0 not in deformable_ids
        world_instances = src.geom_pose_source == int(InstancePoseSource.WORLD)
        assert set(src.geom_object_id[world_instances].tolist()) == deformable_ids

        hinge = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_JOINT, "skin_tip_hinge")
        assert a.set_qpos(int(a.model.jnt_qposadr[hinge]), np.deg2rad(35.0))
        a.step(2)
        frame = a.frame(FrameNeeds(poses=True, deformables=True))
        assert frame.mesh_updates is not None
        updates = frame.mesh_updates

        ref = mujoco.MjvScene(a.model, maxgeom=64)
        option = mujoco.MjvOption()
        camera = mujoco.MjvCamera()
        mujoco.mjv_updateScene(
            a.model,
            a.data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            ref,
        )

        for flex_id in range(a.model.nflex):
            update = frame.mesh_updates[MeshKey(MeshShape.FLEX, flex_id)]
            if int(a.model.flex_dim[flex_id]) == 1:
                assert len(update.positions) > 0
                assert np.isfinite(update.positions).all()
                assert np.linalg.norm(update.normals, axis=1) == pytest.approx(1.0)
                continue
            face_adr = int(ref.flexfaceadr[flex_id])
            face_num = int(ref.flexfaceused[flex_id])
            positions = ref.flexface[9 * face_adr : 9 * (face_adr + face_num)].reshape(-1, 3)
            normals = ref.flexnormal[9 * face_adr : 9 * (face_adr + face_num)].reshape(-1, 3)
            assert update.positions == pytest.approx(positions, abs=2e-6)
            assert update.normals == pytest.approx(normals, abs=2e-6)

        skin = frame.mesh_updates[MeshKey(MeshShape.SKIN, 0)]
        adr, count = int(ref.skinvertadr[0]), int(ref.skinvertnum[0])
        assert skin.positions == pytest.approx(
            ref.skinvert[3 * adr : 3 * (adr + count)].reshape(-1, 3), abs=2e-6
        )
        assert skin.normals == pytest.approx(
            ref.skinnormal[3 * adr : 3 * (adr + count)].reshape(-1, 3), abs=2e-6
        )

        builder = SceneSourceBuilder()
        scene = builder.set_source(src)
        builder.update(frame)
        for i, pose_source in enumerate(src.geom_pose_source):
            if pose_source == int(InstancePoseSource.WORLD):
                assert scene.transforms[builder.write_index[i]] == pytest.approx(np.eye(4))

        assert a.frame(FrameNeeds.none()).mesh_updates is None
        assert a.frame(FrameNeeds(deformables=True)).mesh_updates is updates
    finally:
        a.release()


def test_deformables_contribute_to_session_bounds():
    from forge_viewer.assets import resolve
    from forge_viewer.session import Session

    session = Session(MuJoCoAdapter(resolve("deformables")))
    try:
        source = session.source
        points = np.concatenate([source.meshes[key].positions for key in source.dynamic_meshes])
        lo, hi = session.bounds()
        assert np.all(lo <= points.min(axis=0))
        assert np.all(hi >= points.max(axis=0))
        assert lo[0] < -1.5  # regular geoms only reach x=-1; the cloth extends farther
    finally:
        session.release()


def test_mujoco_model_cameras_follow_forward_kinematics():
    from forge_viewer.assets import resolve

    a = MuJoCoAdapter(resolve("mujoco_visuals"))
    try:
        cameras = {c.name: c.camera_id for c in a.cameras()}
        assert set(cameras) == {"overview", "ball_camera"}
        fixed = a.camera_view(cameras["overview"])
        mounted = a.camera_view(cameras["ball_camera"])
        assert fixed is not None and mounted is not None
        assert np.linalg.norm(fixed.forward()) == pytest.approx(1.0)
        assert abs(float(np.dot(fixed.forward(), fixed.up))) < 1e-5

        ball = next(n for n in a.nodes() if n.name == "ball")
        delta = np.array([0.4, -0.2, 0.3])
        assert a.set_pose(ball.node_id, a.data.xpos[ball.body_index] + delta, np.eye(3))
        moved = a.camera_view(cameras["ball_camera"])
        assert moved is not None
        assert moved.eye - mounted.eye == pytest.approx(delta)
        assert a.camera_view(999) is None
    finally:
        a.release()


def test_mujoco_geom_groups_rebuild_scene_nodes_and_raycast_mask():
    from forge_viewer.assets import resolve

    a = MuJoCoAdapter(resolve("mujoco_visuals"))
    try:
        before_instances = a.scene_source().instance_count
        before_nodes = len(a.nodes())
        revision = a.structure_revision
        groups = {item.category: item.visible for item in a.visual_groups()}
        assert groups["geom"] == (True, True, True, False, False, False)
        assert groups["site"] == (True, True, True, False, False, False)
        assert a.set_visual_group("geom", 3, True)
        assert a.structure_revision == revision + 1
        assert {item.category: item.visible for item in a.visual_groups()}["geom"][3]
        assert a._ray_geomgroup.tolist() == [1, 1, 1, 1, 0, 0]
        assert a.scene_source().instance_count > before_instances
        assert len(a.nodes()) == before_nodes + 1
        assert any(n.name == "collision_debug" for n in a.nodes())
        source = a.scene_source()
        assert source.tendon_visible.tolist() == [True]
        assert source.actuator_visible.tolist() == [True]
        assert a.set_visual_group("tendon", 0, False)
        assert a.scene_source().tendon_visible.tolist() == [False]
        assert a.scene_source().actuator_visible.tolist() == [True]
        assert a.set_visual_group("actuator", 0, False)
        assert a.scene_source().actuator_visible.tolist() == [False]
        assert a.set_visual_group("site", 0, False)
        assert not any(n.kind is NodeKind.SITE for n in a.nodes())
        assert not a.set_visual_group("geom", 6, True)
        assert not a.set_visual_group("unknown", 0, True)
    finally:
        a.release()
