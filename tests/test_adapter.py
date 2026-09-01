from __future__ import annotations

import gc
import tracemalloc
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mojive.adapters.base import (
    ActuatorVisualType,
    BvhType,
    FrameNeeds,
    JointVisualType,
    NodeType,
)
from mojive.types import ShadingModel

pytestmark = pytest.mark.physics

mujoco = pytest.importorskip("mujoco", reason="MuJoCo is required")

from mojive.adapters.mujoco_adapter import MuJoCoAdapter  # noqa: E402

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
    <!-- A jointless landmark welded to the world with its own pickable object id. -->
    <body name="post" pos="-2 0 .5">
      <geom name="post_box" type="box" size=".2 .2 .5"/>
      <!-- Group 3 collision envelope; hidden geoms do not participate in visual raycasts. -->
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
        assert len(getattr(src, name)) == n
    assert src.geom_size.shape == (n, 3)
    assert src.geom_rgba.shape == (n, 4)
    assert src.geom_local.shape == (n, 4, 4)
    assert src.geom_object_id.dtype == np.uint32

    assert all(0 <= i < len(src.materials) for i in src.geom_material)


def test_bundled_test_scene_loads(fixture_path):

    resolve = pytest.importorskip("mojive.assets", reason="asset registry unavailable").resolve
    a = MuJoCoAdapter(resolve("test_scene"))
    src = a.scene_source()
    n = src.instance_count
    assert n > 0
    for name in ("geom_material", "geom_size", "geom_rgba", "geom_object_id", "geom_source"):
        assert len(getattr(src, name)) == n
    assert src.geom_infinite_plane.any()
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
    assert src.geom_local[lo][2, 2] < 0

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

    assert ids("floor") == {0}
    assert ids("wall") == {0}
    assert ids("cap") == ids("head")
    assert ids("cap") != {0}
    assert ids("cap") != ids("lower")

    nodes = adapter.nodes()
    body_ids = {n.object_id for n in nodes if n.type in (NodeType.LINK, NodeType.ROBOT)}
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
    assert src.shading_model is ShadingModel.MUJOCO_CLASSIC
    assert set(src.textures) == {"grid", "sky"}
    assert all(t.srgb for t in src.textures.values())
    assert src.skybox == "sky"
    sky = src.textures["sky"]
    assert sky.pixels.shape == (6, 4, 4, 3)
    assert src.textures["grid"].pixels.ndim == 3


def test_skybox_selection_writes_back_to_the_editable_spec(adapter):
    sky_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_TEXTURE, "sky")

    assert adapter.set_skybox(None)
    assert int(adapter.model.tex_type[sky_id]) == int(mujoco.mjtTexture.mjTEXTURE_CUBE)
    assert int(adapter._root_spec.compile().tex_type[sky_id]) == int(
        mujoco.mjtTexture.mjTEXTURE_CUBE
    )

    assert adapter.set_skybox("sky")
    assert int(adapter.model.tex_type[sky_id]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    assert int(adapter._root_spec.compile().tex_type[sky_id]) == int(
        mujoco.mjtTexture.mjTEXTURE_SKYBOX
    )
    assert not adapter.set_skybox("grid")


def test_square_cube_texture_repeats_one_image_on_every_face(tmp_path):
    from PIL import Image

    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    Image.fromarray(pixels, "RGB").save(tmp_path / "cube.png")
    path = tmp_path / "cube.xml"
    path.write_text(
        """
        <mujoco>
          <asset><texture name="cube" type="cube" file="cube.png"/></asset>
          <worldbody><geom type="box" size=".1 .1 .1"/></worldbody>
        </mujoco>
        """
    )

    cube_adapter = MuJoCoAdapter(path)
    try:
        texture = cube_adapter.scene_source().textures["cube"]
        assert texture.pixels.shape == (6, 8, 8, 3)
        assert np.all(texture.pixels == texture.pixels[0])
    finally:
        cube_adapter.release()


def test_flex_uv_indices_can_reference_texture_coordinates_beyond_vertex_count():
    from types import SimpleNamespace

    from mojive.adapters.mujoco_deformables import _SurfaceFlex

    surface = object.__new__(_SurfaceFlex)
    surface._corner_ids = np.array([0, 1, 2, 0, 2, 1], np.int32)
    surface._smooth = False
    model = SimpleNamespace(
        flex_texcoordadr=np.array([0], np.int32),
        nflextexcoord=4,
        flex_texcoord=np.array(((0, 0), (1, 0), (0, 1), (1, 1)), np.float32),
        flex_elemdataadr=np.array([0], np.int32),
        flex_elemtexcoord=np.array([0, 1, 3], np.int32),
        flex_vertnum=np.array([3], np.int32),
    )

    uvs = surface._uvs(model, 0, np.array([[0, 1, 2]]), np.empty((0, 2)), 2)

    assert uvs == pytest.approx(model.flex_texcoord[[0, 1, 3, 0, 3, 1]])


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

    material_id = src.materials.index(grid)
    edited_material = replace(
        grid,
        rgba=np.array([0.15, 0.25, 0.35, 0.8], np.float32),
        emission=0.35,
        specular=0.65,
        shininess=0.75,
        reflectance=0.45,
        texture=None,
        tex_repeat=np.array([3.0, 4.0], np.float32),
        tex_uniform=False,
    )
    assert adapter.set_material(material_id, edited_material)
    assert adapter.model.mat_emission[material_id] == pytest.approx(0.35)
    assert adapter.model.mat_specular[material_id] == pytest.approx(0.65)
    assert adapter.model.mat_shininess[material_id] == pytest.approx(0.75)
    assert adapter.model.mat_reflectance[material_id] == pytest.approx(0.45)
    assert adapter.model.mat_texrepeat[material_id] == pytest.approx([3.0, 4.0])
    assert adapter.model.mat_texid[material_id, 1] == -1
    recompiled = adapter._root_spec.compile()
    assert recompiled.mat_rgba[material_id] == pytest.approx(edited_material.rgba)
    assert recompiled.mat_emission[material_id] == pytest.approx(0.35)
    assert recompiled.mat_texrepeat[material_id] == pytest.approx([3.0, 4.0])
    assert recompiled.mat_texid[material_id, 1] == -1

    instance = next(i for i, value in enumerate(src.geom_node) if value >= 0)
    node_id = int(src.geom_node[instance])
    color = np.array([0.15, 0.3, 0.6, 0.75], np.float32)
    assert adapter.set_geometry_color(node_id, color)
    geom_id = int(src.geom_source[instance])
    assert adapter.model.geom_rgba[geom_id] == pytest.approx(color)

    lights = src.lights
    assert len(lights.lights) == 1
    assert lights.headlight is not None
    assert lights.horizon_haze_slices == int(adapter.model.vis.quality.numslices)
    hl = adapter.model.vis.headlight

    assert lights.headlight.diffuse == pytest.approx(np.asarray(hl.diffuse))

    assert lights.ambient == pytest.approx(np.asarray(hl.ambient))

    node = next(node for node in adapter.nodes() if node.type is NodeType.LIGHT)
    assert node.object_id and node.light_index == 0
    edited = replace(lights.lights[0], diffuse=np.array([0.2, 0.3, 0.4], np.float32))
    assert adapter.set_light(0, edited)
    frame = adapter.frame(FrameNeeds())
    assert frame.lights is not None
    assert frame.lights.lights[0].diffuse == pytest.approx([0.2, 0.3, 0.4])


def test_material_texture_roundtrip_preserves_relative_mesh_directory(tmp_path):
    from mojive import commands as cmd
    from mojive.session import Session

    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetrahedron.obj").write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
""",
        encoding="utf-8",
    )
    path = tmp_path / "relative_mesh.xml"
    path.write_text(
        """
        <mujoco>
          <compiler meshdir="meshes"/>
          <asset>
            <mesh name="tetrahedron" file="tetrahedron.obj" scale=".1 .1 .1"/>
            <texture name="checker" type="2d" builtin="checker" width="8" height="8"/>
            <material name="ground" texture="checker"/>
          </asset>
          <worldbody>
            <geom name="mesh" type="mesh" mesh="tetrahedron"/>
            <geom name="floor" type="plane" size="0 0 .05" material="ground"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    relative_adapter = MuJoCoAdapter(path)
    session = Session(relative_adapter, path)
    try:
        source = session.source
        assert source is not None
        material_id = next(
            index for index, material in enumerate(source.materials) if material.name == "ground"
        )
        material = source.materials[material_id]

        assert session.submit(cmd.SetMaterial(material_id, replace(material, texture=None))).ok
        assert Path(relative_adapter._root_spec.modelfiledir) == tmp_path
        assert "tetrahedron.obj" in relative_adapter.scene_model_source(0)
        assert session.submit(cmd.SetMaterial(material_id, material)).ok
        assert relative_adapter._root_spec.compile().nmesh == 1
    finally:
        session.release()


def test_target_light_uses_mujoco_world_pose_and_updates_with_its_target(tmp_path):
    path = tmp_path / "target_light.xml"
    path.write_text(
        """
        <mujoco>
          <worldbody>
            <light name="target" mode="targetbodycom" target="body" pos="0 -4 3"/>
            <body name="body" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size=".2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    target_adapter = MuJoCoAdapter(path)
    try:
        initial = target_adapter.scene_source().lights.lights[0]
        assert initial.position == pytest.approx(target_adapter.data.light_xpos[0])
        assert initial.direction == pytest.approx(target_adapter.data.light_xdir[0])
        assert initial.direction != pytest.approx(target_adapter.model.light_dir[0])

        target_adapter.data.qpos[0] = 1.0
        mujoco.mj_forward(target_adapter.model, target_adapter.data)
        frame = target_adapter.frame(FrameNeeds())

        assert frame.lights is not None
        assert frame.lights.lights[0].direction == pytest.approx(target_adapter.data.light_xdir[0])
        assert frame.lights.lights[0].direction != pytest.approx(initial.direction)
    finally:
        target_adapter.release()


def test_trackcom_light_edit_updates_while_simulation_is_paused(tmp_path):
    path = tmp_path / "tracking_light.xml"
    path.write_text(
        """
        <mujoco>
          <worldbody>
            <light name="tracking" mode="trackcom" pos="0 0 2"/>
            <body name="body" pos="1 0 1">
              <freejoint/>
              <geom type="sphere" size=".2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    tracking_adapter = MuJoCoAdapter(path)
    try:
        light = tracking_adapter.scene_source().lights.lights[0]
        edited_position = np.array((0.75, -0.5, 2.25), np.float32)

        assert tracking_adapter.set_light(0, replace(light, position=edited_position))

        frame = tracking_adapter.frame(FrameNeeds())
        assert frame.lights is not None
        assert frame.lights.lights[0].position == pytest.approx(edited_position)
        assert tracking_adapter.data.light_xpos[0] == pytest.approx(edited_position)
    finally:
        tracking_adapter.release()


def test_opengl_area_light_survives_mujoco_writeback(adapter, fixture_path):
    from mojive import commands as cmd
    from mojive.session import Session
    from mojive.types import LightType

    session = Session(adapter, fixture_path)
    source_light = session.source.lights.lights[0]
    edited = replace(source_light, type=LightType.AREA, area_radius=0.35)

    assert session.submit(cmd.SetLight(0, edited))
    assert session.source.lights.lights[0].type is LightType.AREA
    assert int(adapter.model.light_type[0]) == int(mujoco.mjtLightType.mjLIGHT_POINT)

    frame = session.tick(FrameNeeds())
    assert frame.lights.lights[0].type is LightType.AREA
    assert frame.lights.lights[0].position == pytest.approx(adapter.data.light_xpos[0])


def test_image_light_preserves_its_cube_texture_and_intensity(tmp_path):
    from mojive.types import LightType

    path = tmp_path / "image_light.xml"
    path.write_text(
        """
        <mujoco>
          <asset>
            <texture name="studio" type="cube" builtin="gradient" width="8" height="48"
                     rgb1="1 .4 .1" rgb2=".1 .3 1"/>
          </asset>
          <worldbody>
            <light name="environment" type="image" texture="studio" intensity="7500"/>
            <geom type="sphere" size=".2"/>
          </worldbody>
        </mujoco>
        """
    )
    image_adapter = MuJoCoAdapter(path)
    try:
        light = image_adapter.scene_source().lights.lights[0]
        texture = image_adapter.scene_source().textures["studio"]
        assert light.type is LightType.IMAGE
        assert light.texture == "studio"
        assert light.intensity == pytest.approx(7500.0)
        assert texture.pixels.shape == (6, 8, 8, 3)

        edited = replace(light, intensity=3200.0)
        assert image_adapter.set_light(0, edited)
        assert image_adapter.model.light_intensity[0] == pytest.approx(3200.0)
        assert image_adapter.frame(FrameNeeds()).lights.lights[0].intensity == pytest.approx(3200.0)
        assert image_adapter.set_light(0, replace(edited, type=LightType.POINT, texture=None))
        assert image_adapter.model.light_texid[0] == -1
        assert image_adapter.set_light(0, edited)
    finally:
        image_adapter.release()


def test_attached_image_light_writes_its_local_texture_name(tmp_path):
    from mojive import commands as cmd
    from mojive.session import Session

    root = tmp_path / "root.xml"
    root.write_text('<mujoco model="root"><worldbody/></mujoco>')
    child = tmp_path / "child.xml"
    child.write_text(
        """
        <mujoco model="child">
          <asset>
            <texture name="studio" type="cube" builtin="gradient" width="8" height="48"
                     rgb1="1 .4 .1" rgb2=".1 .3 1"/>
          </asset>
          <worldbody>
            <light name="environment" type="image" texture="studio" intensity="7500"/>
            <geom type="sphere" size=".2"/>
          </worldbody>
        </mujoco>
        """
    )

    image_adapter = MuJoCoAdapter(root)
    session = Session(image_adapter, root)
    try:
        assert session.submit(cmd.Pause())
        assert session.submit(cmd.AddSceneModel(child, np.zeros(3, np.float32)))
        light = session.source.lights.lights[0]
        assert light.texture == "opengl_1_studio"
        assert image_adapter.set_light(0, replace(light, intensity=3200.0))

        exported = tmp_path / "exported.xml"
        image_adapter.export_mjcf(exported, session.source, session.frame)
        restored = MuJoCoAdapter(exported)
        try:
            assert restored.scene_source().lights.lights[0].intensity == pytest.approx(3200.0)
        finally:
            restored.release()
    finally:
        image_adapter.release()


def test_initial_values_come_from_the_model(adapter):

    m = adapter.model
    expect = np.asarray(m.qpos0, np.float32).copy()
    adapter.step(200)
    assert not np.allclose(adapter.data.qpos, expect)
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
    assert f.geom_xpos is None


def test_diagnostic_metadata_and_frame_match_mujoco(adapter):
    source = adapter.scene_source().diagnostics
    model, data = adapter.model, adapter.data
    expected_kinds = np.asarray(
        [
            JointVisualType.FREE,
            JointVisualType.HINGE,
            JointVisualType.HINGE,
        ],
        np.uint8,
    )
    assert source.joint_types == pytest.approx(expected_kinds)
    assert source.joint_visible.tolist() == [True] * model.njnt
    assert source.joint_length == pytest.approx(model.stat.meansize * model.vis.scale.jointlength)
    assert source.joint_width == pytest.approx(model.stat.meansize * model.vis.scale.jointwidth)
    assert source.joint_rgba == pytest.approx(model.vis.rgba.joint)

    expected_com = np.flatnonzero(np.asarray(model.body_parentid[1:]) == 0) + 1
    assert source.com_bodies == pytest.approx(expected_com)
    assert source.com_radius == pytest.approx(model.stat.meansize * model.vis.scale.com)

    moving = np.flatnonzero(
        (np.asarray(model.body_dofnum) > 0) & (np.asarray(model.body_mass) > 0.0)
    )
    assert source.inertia_bodies == pytest.approx(moving)
    mass = np.asarray(model.body_mass[moving])
    assert 8.0 * np.prod(source.scaled_inertia_sizes, axis=1) == pytest.approx(mass / 1000.0)

    assert source.actuator_visual_types.tolist() == [ActuatorVisualType.HINGE]
    assert source.actuator_visual_actuators.tolist() == [0]
    assert source.actuator_visual_sizes[0] == pytest.approx(
        (
            model.stat.meansize * model.vis.scale.actuatorwidth,
            model.stat.meansize * model.vis.scale.actuatorwidth,
            model.stat.meansize * model.vis.scale.actuatorlength,
        )
    )

    assert adapter.frame(FrameNeeds.none()).diagnostics is None
    frame = adapter.frame(FrameNeeds(poses=False, diagnostics=True)).diagnostics
    assert frame.joint_xpos == pytest.approx(data.xanchor)
    assert frame.joint_xaxis == pytest.approx(data.xaxis)
    assert frame.subtree_com == pytest.approx(data.subtree_com)
    assert frame.body_xipos == pytest.approx(data.xipos)
    assert frame.body_ximat == pytest.approx(data.ximat.reshape(model.nbody, 3, 3))
    joint = int(model.actuator_trnid[0, 0])
    assert frame.actuator_xpos[0] == pytest.approx(data.xanchor[joint])
    assert frame.actuator_xmat[0, :, 2] == pytest.approx(data.xaxis[joint])


def test_massless_free_parent_is_excluded_from_inertia_visuals(tmp_path):
    path = tmp_path / "massless_parent.xml"
    path.write_text(
        """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <body><geom type="sphere" size="0.1"/></body>
    </body>
  </worldbody>
</mujoco>
"""
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        instance = MuJoCoAdapter(path)
    try:
        root = instance.model.body("root").id
        assert root not in instance.scene_source().diagnostics.inertia_bodies
    finally:
        instance.release()


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

    assert len(buf_ids) == 2
    assert peak - start < 16 * 1024
    assert current - start < 8 * 1024


def test_slow_path_agrees_with_fast_path(adapter):

    assert adapter.fast_pose
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
    assert qpos[1] == pytest.approx(0.5)
    assert qpos[7] == pytest.approx(0.0)

    assert adapter.set_qpos(joints["h1"].qpos_adr, 0.3)
    assert adapter.data.qpos[7] == pytest.approx(0.3)
    assert not adapter.set_qpos(9999, 0.0)


def test_set_qpos_on_joint_types_scene():

    resolve = pytest.importorskip("mojive.assets", reason="asset registry unavailable").resolve
    a = MuJoCoAdapter(resolve("joint_types"))
    joints = {j.name: j for j in a.joints()}
    assert joints["free"].qpos_adr == 0 and joints["free"].dof == 6
    ball = joints["ball"]
    assert ball.qpos_adr == 7

    before = a.data.qpos.copy()
    assert a.set_qpos(2, 0.75)
    assert a.data.qpos[2] == pytest.approx(0.75)

    slide = joints["slide"]
    assert a.data.qpos[slide.qpos_adr] == pytest.approx(before[slide.qpos_adr])

    body_kinds = (NodeType.WORLD, NodeType.ROBOT, NodeType.LINK)
    nodes = {n.name: n for n in a.nodes() if n.type in body_kinds}
    assert nodes["free_body"].posable
    assert nodes["free_chain"].posable
    assert nodes["chain_root"].posable
    for name in ("ball_body", "slide_body", "hinge_body", "chain_0"):
        assert not nodes[name].posable
    a.release()


def test_set_ctrl_clips_to_range(adapter):
    assert adapter.set_ctrl(0, 5.0)
    assert adapter.data.ctrl[0] == pytest.approx(1.0)
    assert not adapter.set_ctrl(7, 0.0)


def test_set_pose_on_free_body_resets_velocity(adapter):

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


def test_mocap_bodies_use_the_shared_pose_editing_contract():
    from mojive.assets import resolve

    adapter = MuJoCoAdapter(resolve("mocap_equality"))
    try:
        target = next(node for node in adapter.nodes() if node.name == "mocap_target")
        assert target.posable

        position = np.array([0.4, -0.25, 1.6])
        rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        assert adapter.set_pose(target.node_id, position, rotation)
        assert adapter.data.mocap_pos[0] == pytest.approx(position)
        assert adapter.data.xpos[target.body_index] == pytest.approx(position)
        assert adapter.data.xmat[target.body_index].reshape(3, 3) == pytest.approx(rotation)
    finally:
        adapter.release()


def test_equality_constraints_are_listed_and_switchable():
    from mojive import commands as cmd
    from mojive.assets import resolve
    from mojive.session import Session

    adapter = MuJoCoAdapter(resolve("mocap_equality"))
    session = Session(adapter)
    try:
        constraints = session.equality_constraints
        assert [(item.name, item.type, item.enabled) for item in constraints] == [
            ("mocap_weld", "mjEQ_WELD", True)
        ]
        assert session.submit(cmd.SetEqualityEnabled(0, False))
        assert not adapter.data.eq_active[0]
        assert not session.equality_constraints[0].enabled
        assert session.submit(cmd.Reset())
        assert adapter.data.eq_active[0]
        assert session.equality_constraints[0].enabled
        assert not session.submit(cmd.SetEqualityEnabled(2, True))
    finally:
        session.release()


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
    assert obj == nodes["post"].object_id != 0
    assert dist == pytest.approx(2.0, abs=1e-6)

    obj_a, dist_a = adapter.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]))
    obj_b, dist_b = adapter.raycast(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -7.0]))
    assert obj_a == obj_b
    assert dist_a == pytest.approx(dist_b)


def test_caps_report_what_is_not_there(adapter):

    caps = adapter.caps
    assert caps.name == "mujoco"
    assert (
        caps.write_pose
        and caps.write_qpos
        and caps.perturb
        and caps.raycast
        and caps.reload
        and caps.keyframes
        and caps.sensors
        and caps.equality_constraints
    )
    assert adapter.set_paused(True)


def test_keyframes_are_listed_and_restore_the_complete_state(adapter, fixture_path):
    from mojive import commands as cmd
    from mojive.session import Session

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


def test_session_state_take_replays_mujoco_frames_without_recompiling(adapter, fixture_path):
    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.session import Session

    session = Session(adapter, fixture_path)
    generation = session.structure_generation
    timestep = adapter.timestep()
    assert session.submit(cmd.StartStateTakeRecording())
    session.tick(FrameNeeds(), wall_dt=timestep)
    session.tick(FrameNeeds(), wall_dt=timestep)
    assert session.submit(cmd.StopStateTakeRecording())

    assert len(session.state_take_times) == 3
    assert session.structure_generation == generation
    assert not session.dirty
    final_time = adapter.data.time
    assert session.submit(cmd.SeekStateTake(0))
    assert adapter.data.time < final_time
    assert session.submit(cmd.SeekStateTake(2))
    assert adapter.data.time == pytest.approx(final_time)


def test_sensor_metadata_addresses_the_frame_values(adapter):
    sensors = adapter.sensors()
    assert len(sensors) == 1
    sensor = sensors[0]
    assert (sensor.name, sensor.type, sensor.data_adr, sensor.dim) == (
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
    assert "nonsense" in text

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

    assert len(adapter._geom_xpos_buf) == adapter.model.ngeom
    assert adapter._mj_geom_xpos is adapter.data.geom_xpos
    assert adapter._mj_geom_xmat3.base is adapter.data.geom_xmat
    adapter.step(50)
    f = adapter.frame(FrameNeeds())
    assert np.allclose(f.geom_xpos, adapter.data.geom_xpos, atol=1e-6)
    assert np.linalg.norm(f.geom_xpos - adapter.model.geom_pos) > 1e-4


def test_mjspec_model_composition_preserves_matching_state(tmp_path):
    from mojive import commands as cmd
    from mojive.session import Session

    root = tmp_path / "root.xml"
    root.write_text(
        """
        <mujoco model="root">
          <worldbody>
            <body name="arm" pos="0 0 1">
              <joint name="hinge" type="hinge"/>
              <geom type="capsule" size=".08 .3"/>
            </body>
          </worldbody>
          <actuator><motor name="motor" joint="hinge"/></actuator>
        </mujoco>
        """
    )
    child = tmp_path / "payload.xml"
    child.write_text(
        """
        <mujoco model="payload">
          <worldbody>
            <body name="payload" pos=".2 0 0">
              <freejoint name="root"/>
              <geom type="sphere" size=".1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    adapter = MuJoCoAdapter(root)
    session = Session(adapter, root)
    assert session.submit(cmd.Pause())
    adapter.data.qpos[0] = 0.37
    adapter.data.qvel[0] = -0.2
    adapter.data.ctrl[0] = 0.6
    adapter.data.time = 2.5

    added = session.submit(cmd.AddSceneModel(child, (1.0, 2.0, 3.0)))
    assert added.ok and added.entity_id > 0
    assert [item.name for item in session.scene_models] == ["root", "payload"]
    assert adapter.data.qpos[0] == pytest.approx(0.37)
    assert adapter.data.qvel[0] == pytest.approx(-0.2)
    assert adapter.data.ctrl[0] == pytest.approx(0.6)
    assert adapter.data.time == pytest.approx(2.5)
    assert adapter.model.body("opengl_1_payload").id > 0
    model_node = next(node for node in session.nodes if node.type is NodeType.MODEL)
    assert model_node.name == "payload" and model_node.model_id == added.entity_id
    assert any(session.node(child).name == "opengl_1_payload" for child in model_node.children)
    model_center, model_half = session.node_world_bounds(model_node.node_id)
    assert model_center == pytest.approx((1.2, 2.0, 3.0))
    assert model_half == pytest.approx((0.1, 0.1, 0.1))

    removed = session.submit(cmd.RemoveSceneModel(added.entity_id))
    assert removed.ok
    assert [item.name for item in session.scene_models] == ["root"]
    assert adapter.data.qpos[0] == pytest.approx(0.37)
    assert adapter.data.qvel[0] == pytest.approx(-0.2)
    assert adapter.data.ctrl[0] == pytest.approx(0.6)
    assert adapter.data.time == pytest.approx(2.5)
    adapter.release()


def test_session_loads_mjcf_and_urdf_without_losing_the_current_model_on_failure(tmp_path):
    from mojive import commands as cmd
    from mojive.assets import resolve
    from mojive.session import Session

    session = Session(MuJoCoAdapter(resolve("empty")), resolve("empty"))
    try:
        adapter = session.adapter
        original_load = adapter.load

        def load_while_paused(path):
            assert session.paused
            return original_load(path)

        adapter.load = load_while_paused
        generation = session.structure_generation
        for name in ("test_scene.xml", "test_scene.urdf"):
            assert session.submit(cmd.Play())
            assert not session.paused
            path = resolve(name)
            result = session.submit(cmd.LoadAsset(path))
            assert result.ok, result.message
            assert session.paused
            assert session.asset_path == path
            assert session.source.instance_count > 0
            assert session.structure_generation > generation
            generation = session.structure_generation

        current_path = session.asset_path
        current_source = session.source
        invalid = tmp_path / "invalid.xml"
        invalid.write_text("<mujoco><worldbody>", encoding="utf-8")
        assert session.submit(cmd.Play())
        result = session.submit(cmd.LoadAsset(invalid))
        assert not result.ok
        assert session.paused
        assert "invalid.xml" in result.message
        assert session.asset_path == current_path
        assert session.source is current_source
    finally:
        session.release()


def test_urdf_loader_normalizes_repeated_compiler_mesh_directory(tmp_path: Path) -> None:
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "tetra.obj").write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 3 2",
                "f 1 2 4",
                "f 1 4 3",
                "f 2 3 4",
            )
        ),
        encoding="utf-8",
    )
    path = tmp_path / "redundant-meshdir.urdf"
    path.write_text(
        """<robot name="redundant_meshdir">
  <mujoco><compiler meshdir="meshes" discardvisual="false"/></mujoco>
  <link name="base">
    <visual><geometry><mesh filename="meshes/tetra.obj"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    adapter = MuJoCoAdapter(path)

    assert adapter.model.nmesh == 1


def test_urdf_loader_preserves_an_explicit_repeated_mesh_directory(tmp_path: Path) -> None:
    from mojive.adapters.mujoco_adapter import _normalized_urdf_source

    meshes = tmp_path / "meshes" / "meshes"
    meshes.mkdir(parents=True)
    mesh_source = "\n".join(
        (
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "v 0 0 1",
            "f 1 3 2",
            "f 1 2 4",
            "f 1 4 3",
            "f 2 3 4",
        )
    )
    (meshes / "tetra.obj").write_text(mesh_source, encoding="utf-8")
    (tmp_path / "meshes" / "tetra.obj").write_text(mesh_source, encoding="utf-8")
    path = tmp_path / "explicit-repeated-meshdir.urdf"
    path.write_text(
        """<robot name="explicit_repeated_meshdir">
  <mujoco><compiler meshdir="meshes" discardvisual="false"/></mujoco>
  <link name="base">
    <visual><geometry><mesh filename="meshes/tetra.obj"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    assert _normalized_urdf_source(path) is None
    adapter = MuJoCoAdapter(path)

    assert adapter.model.nmesh == 1


@pytest.mark.parametrize(
    "filename",
    ("package://assets/tetra.obj", "meshes/tetra.obj"),
)
def test_urdf_loader_resolves_unambiguous_local_mesh_relocations(
    tmp_path: Path, filename: str
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tetra.obj").write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
""",
        encoding="utf-8",
    )
    path = tmp_path / "relocated.urdf"
    path.write_text(
        f"""<robot name="relocated">
  <mujoco><compiler discardvisual="false"/></mujoco>
  <link name="base">
    <visual><geometry><mesh filename="{filename}"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    adapter = MuJoCoAdapter(path)

    assert adapter.model.nmesh == 1


def test_urdf_loader_does_not_guess_between_ambiguous_meshes(tmp_path: Path) -> None:
    from mojive.adapters.mujoco_adapter import _normalized_urdf_source

    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir()
        (directory / "part.stl").write_bytes(b"not needed for path normalization")
    path = tmp_path / "ambiguous.urdf"
    path.write_text(
        """<robot name="ambiguous">
  <link name="base"><visual><geometry><mesh filename="missing/part.stl"/></geometry></visual></link>
</robot>
""",
        encoding="utf-8",
    )

    assert _normalized_urdf_source(path) is None


def test_urdf_loader_scales_only_tiny_positive_definite_inertia(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from mojive.adapters.mujoco_adapter import _normalized_urdf_source

    path = tmp_path / "tiny-inertia.urdf"
    path.write_text(
        """<robot name="tiny_inertia">
  <mujoco><compiler discardvisual="false"/></mujoco>
  <link name="base">
    <inertial>
      <mass value="6.55e-8"/>
      <inertia ixx="1.64e-15" ixy="0" ixz="0"
               iyy="1.64e-15" iyz="0" izz="1.64e-15"/>
    </inertial>
    <visual><geometry><box size=".01 .01 .01"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    normalized = _normalized_urdf_source(path)
    assert normalized is not None
    adapter = MuJoCoAdapter(path)

    assert adapter.model.ngeom == 1
    inertia = ET.fromstring(normalized).find("link/inertial/inertia")
    assert inertia is not None
    assert [float(inertia.attrib[name]) for name in ("ixx", "iyy", "izz")] == pytest.approx(
        [1e-14] * 3
    )


def test_camera_hint_frames_the_scene(adapter):
    cam = adapter.camera_hint()
    assert cam is not None
    extent = adapter.model.stat.extent
    assert cam.near < cam.far
    assert cam.near == pytest.approx(adapter.model.vis.map.znear * extent)
    assert cam.far >= 200.0 * extent
    assert 0.5 * extent < np.linalg.norm(cam.eye - cam.target) < 5.0 * extent


def test_session_indexes_selected_body_joints_and_requires_pause() -> None:
    from mojive import commands as cmd
    from mojive.assets import resolve
    from mojive.session import Session

    session = Session(MuJoCoAdapter(resolve("joint_types")))
    joint = session.joints[0]

    assert joint in session.joints_for_body(joint.body)
    for actuator in session.actuators_for_joint(joint.joint_id):
        assert actuator.joint == joint.joint_id
    assert not session.submit(cmd.SetQpos(joint.qpos_adr, 0.1))
    assert session.submit(cmd.Pause())
    assert session.submit(cmd.SetQpos(joint.qpos_adr, 0.1))


def test_qpos_batch_validates_atomically_and_forwards_once(monkeypatch) -> None:
    from mojive import commands as cmd
    from mojive.assets import resolve
    from mojive.session import Session

    adapter = MuJoCoAdapter(resolve("joint_types"))
    session = Session(adapter)
    assert session.submit(cmd.Pause())
    ball = next(joint for joint in session.joints if joint.type == "ball")
    indices = np.arange(ball.qpos_adr, ball.qpos_adr + 4, dtype=np.intp)
    before = np.asarray(adapter.data.qpos[indices]).copy()

    assert not session.submit(cmd.SetQposBatch(indices, np.array((1.0, 0.0, 0.0))))
    assert adapter.data.qpos[indices] == pytest.approx(before)
    assert not session.submit(cmd.SetQposBatch(np.array((indices[0], indices[0])), np.ones(2)))
    assert adapter.data.qpos[indices] == pytest.approx(before)

    calls = 0
    real_forward = mujoco.mj_forward

    def count_forward(model, data):
        nonlocal calls
        calls += 1
        return real_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", count_forward)
    values = np.array((np.cos(0.2), 0.0, 0.0, np.sin(0.2)))
    assert session.submit(cmd.SetQposBatch(indices, values))
    assert calls == 1
    assert adapter.data.qpos[indices] == pytest.approx(values)


def test_mujoco_visuals_cover_heightfield_sites_and_tendon():
    from mojive.assets import resolve
    from mojive.mujoco_audit import audit_model
    from mojive.render.builder import SceneSourceBuilder
    from mojive.types import CameraView, InstancePoseSource, MeshShape

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


def test_collision_mesh_exposes_mujoco_compiled_convex_hull():
    from mojive.assets import resolve
    from mojive.types import MeshShape

    adapter = MuJoCoAdapter(resolve("convex_hull"))
    try:
        source = adapter.scene_source()
        assert len(source.geom_convex_mesh) == source.instance_count == 1
        original = source.geom_mesh[0]
        hull = source.geom_convex_mesh[0]
        assert original.shape is MeshShape.ASSET
        assert hull.shape is MeshShape.CONVEX_HULL
        assert original in source.meshes and hull in source.meshes
        assert source.meshes[original].triangle_count == 20
        assert source.meshes[hull].triangle_count == 16
    finally:
        adapter.release()


def test_actuator_visual_metadata_and_controls_follow_mujoco_addresses():
    from mojive.assets import resolve
    from mojive.mujoco_audit import audit_model

    a = MuJoCoAdapter(resolve("actuator_visuals"))
    try:
        model = a.model
        actuators = a.actuators()
        source = a.scene_source()
        assert len(actuators) == model.nactuator == 4
        assert [item.ctrl_address for item in actuators] == model.actuator_ctrladr.tolist()
        assert [item.ctrl_count for item in actuators] == model.actuator_ctrlnum.tolist()
        assert [item.act_address for item in actuators] == model.actuator_actadr.tolist()
        assert [item.act_count for item in actuators] == model.actuator_actnum.tolist()
        assert source.actuator_ctrl_address.tolist() == model.actuator_ctrladr.tolist()
        assert source.actuator_ctrl_range == pytest.approx(model.actuator_ctrlrange)

        frame = a.frame(FrameNeeds(poses=False, actuator=True, diagnostics=True))
        assert frame.actuator_activation.shape == (model.nactuator,)
        assert frame.diagnostics.actuator_xpos.shape[0] == len(
            source.diagnostics.actuator_visual_types
        )

        address = actuators[1].ctrl_address
        assert a.set_ctrl(address, 5.0)
        assert a.data.ctrl[address] == pytest.approx(actuators[1].ctrl_range[1])
        findings = audit_model(model)["findings"]
        visual = next(item for item in findings if item["feature"] == "actuator visualization")
        assert visual["status"] == "supported"
        assert visual["count"] == model.nactuator
    finally:
        a.release()


def test_slider_crank_visuals_match_mujoco_linkage_geometry():
    from mojive.assets import resolve

    a = MuJoCoAdapter(resolve("slider_crank"))
    try:
        source = a.scene_source().diagnostics
        frame = a.frame(FrameNeeds(poses=True, actuator=True, diagnostics=True)).diagnostics
        assert source.slider_crank_actuators.tolist() == [0, 1]
        assert source.slider_crank_width == pytest.approx(
            a.model.stat.meansize * a.model.vis.scale.slidercrank
        )
        assert frame.slider_crank_broken.tolist() == [False, True]
        valid_rod = np.linalg.norm(
            frame.slider_crank_points[0, 2] - frame.slider_crank_points[0, 1]
        )
        assert valid_rod == pytest.approx(a.model.actuator_cranklength[0])
        broken_rod = np.linalg.norm(
            frame.slider_crank_points[1, 2] - frame.slider_crank_points[1, 1]
        )
        assert broken_rod > a.model.actuator_cranklength[1]
    finally:
        a.release()


def test_contact_force_components_and_autoconnect_segments_match_mujoco():
    from mojive.assets import resolve

    contacts = MuJoCoAdapter(resolve("mujoco_visuals"))
    chain = MuJoCoAdapter(resolve("joint_types"))
    try:
        contacts.scene_source()
        contacts.step(400)
        frame = contacts.frame(FrameNeeds(poses=True, contacts=True, diagnostics=True))
        assert len(frame.contacts) > 0
        assert frame.contact_forces.shape == (len(frame.contacts), 2, 3)
        combined = np.linalg.norm(frame.contact_forces.sum(axis=1), axis=1)
        assert combined == pytest.approx(frame.contacts[:, 6])
        assert np.linalg.norm(frame.contact_forces[:, 1], axis=1).max() > 0.0

        source = chain.scene_source().diagnostics
        chain_frame = chain.frame(FrameNeeds(poses=True, diagnostics=True)).diagnostics
        expected = sum(
            1 + int(chain.model.body_jntnum[body])
            for body in range(1, chain.model.nbody)
            if int(chain.model.body_parentid[body]) != 0
        )
        assert len(chain_frame.autoconnect_segments) == expected
        assert source.autoconnect_width == pytest.approx(
            chain.model.stat.meansize * chain.model.vis.scale.connect
        )
    finally:
        contacts.release()
        chain.release()


def test_island_colors_match_mujocos_visualizer():
    from mojive.assets import resolve

    adapter = MuJoCoAdapter(resolve("mujoco_visuals"))
    try:
        source = adapter.scene_source()
        adapter.step(400)
        frame = adapter.frame(FrameNeeds(poses=True, contacts=True, tendons=True, islands=True))

        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        reference = mujoco.MjvScene(adapter.model, maxgeom=256)
        mujoco.mjv_updateScene(
            adapter.model,
            adapter.data,
            option,
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            reference,
        )

        geom_colors = {
            int(geom.objid): np.asarray(geom.rgba).copy()
            for geom in reference.geoms[: reference.ngeom]
            if int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
        }
        moving = np.flatnonzero(source.instance_island_body >= 0)
        assert len(moving)
        for instance in moving:
            geom = int(source.geom_source[instance])
            assert frame.island_rgba[instance] == pytest.approx(geom_colors[geom], abs=2e-6)

        tendon_colors = np.asarray(
            [
                np.asarray(geom.rgba).copy()
                for geom in reference.geoms[: reference.ngeom]
                if int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_TENDON)
            ]
        )
        contact_colors = np.asarray(
            [
                np.asarray(geom.rgba).copy()
                for geom in reference.geoms[: reference.ngeom]
                if int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_UNKNOWN)
            ]
        )
        assert frame.tendon_island_rgba == pytest.approx(tendon_colors, abs=2e-6)
        assert frame.contact_island_rgba == pytest.approx(contact_colors, abs=2e-6)
    finally:
        adapter.release()


@pytest.mark.parametrize(
    ("asset", "flag", "bvh_type", "depth", "show_inactive"),
    (
        ("joint_types", mujoco.mjtVisFlag.mjVIS_BODYBVH, BvhType.BODY, 1, False),
        ("dense_mesh", mujoco.mjtVisFlag.mjVIS_MESHBVH, BvhType.MESH, 2, True),
        ("deformables", mujoco.mjtVisFlag.mjVIS_MESHBVH, BvhType.FLEX, 2, False),
    ),
)
def test_bvh_boxes_match_mujocos_visualizer(asset, flag, bvh_type, depth, show_inactive):
    from mojive.assets import resolve

    adapter = MuJoCoAdapter(resolve(asset))
    try:
        if show_inactive:
            adapter.model.vis.global_.bvactive = 0
        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True, bvh=True)).diagnostics
        source = adapter.scene_source().diagnostics

        option = mujoco.MjvOption()
        option.flags[:] = 0
        option.flags[flag] = True
        option.bvh_depth = depth
        reference = mujoco.MjvScene(adapter.model, maxgeom=200_000)
        mujoco.mjv_updateScene(
            adapter.model,
            adapter.data,
            option,
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            reference,
        )
        boxes = [
            geom
            for geom in reference.geoms[: reference.ngeom]
            if int(geom.type) == int(mujoco.mjtGeom.mjGEOM_LINEBOX)
        ]

        selected = (source.bvh_type == int(bvh_type)) & (
            (source.bvh_depth == depth) | (source.bvh_leaf & (source.bvh_depth < depth))
        )
        records = np.flatnonzero(selected)
        assert len(records) == len(boxes) > 0
        assert frame.bvh_centers[records] == pytest.approx(
            np.asarray([geom.pos for geom in boxes]), abs=2e-6
        )
        assert frame.bvh_matrices[records] == pytest.approx(
            np.asarray([geom.mat for geom in boxes]).reshape(-1, 3, 3), abs=2e-6
        )
        assert frame.bvh_sizes[records] == pytest.approx(
            np.asarray([geom.size for geom in boxes]), abs=2e-6
        )
        colors = np.where(frame.bvh_active[records, None], source.bvh_active_rgba, source.bvh_rgba)
        assert colors == pytest.approx(np.asarray([geom.rgba for geom in boxes]), abs=2e-6)
    finally:
        adapter.release()


def test_bvh_metadata_is_materialized_only_when_a_frame_requests_it():
    from mojive.assets import resolve
    from mojive.session import Session

    adapter = MuJoCoAdapter(resolve("dense_mesh"))
    session = Session(adapter)
    try:
        generation = session.structure_generation

        assert not len(session.source.diagnostics.bvh_type)

        frame = session.tick(FrameNeeds(poses=True, diagnostics=True, bvh=True))

        assert session.structure_generation == generation + 1
        assert len(session.source.diagnostics.bvh_type) > 0
        assert len(frame.diagnostics.bvh_centers) == len(session.source.diagnostics.bvh_type)
    finally:
        session.release()


def test_interpolated_flex_control_cage_matches_mujocos_visualizer(tmp_path):
    path = tmp_path / "interpolated_flex.xml"
    path.write_text(
        """<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="soft" type="grid" dim="3" count="5 4 3" spacing=".1 .1 .1"
              pos="0 0 1" radius=".005" mass="1" dof="trilinear">
      <contact selfcollide="none"/>
      <elasticity young="100" poisson=".3"/>
    </flexcomp>
  </worldbody>
</mujoco>"""
    )
    adapter = MuJoCoAdapter(path)
    try:
        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True, bvh=True)).diagnostics
        source = adapter.scene_source().diagnostics

        option = mujoco.MjvOption()
        option.flags[:] = 0
        option.flags[mujoco.mjtVisFlag.mjVIS_MESHBVH] = True
        reference = mujoco.MjvScene(adapter.model, maxgeom=256)
        mujoco.mjv_updateScene(
            adapter.model,
            adapter.data,
            option,
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            reference,
        )
        lines = [
            geom
            for geom in reference.geoms[: reference.ngeom]
            if int(geom.type) == int(mujoco.mjtGeom.mjGEOM_LINE)
        ]
        starts = np.asarray([geom.pos for geom in lines])
        ends = np.asarray(
            [geom.pos + geom.mat.reshape(3, 3)[:, 2] * geom.size[2] for geom in lines]
        )

        assert source.bvh_control_count == len(lines) == 12
        assert frame.bvh_control_segments[:, 0] == pytest.approx(starts, abs=2e-6)
        assert frame.bvh_control_segments[:, 1] == pytest.approx(ends, abs=2e-6)
        assert source.bvh_control_rgba == pytest.approx(lines[0].rgba, abs=2e-6)
    finally:
        adapter.release()


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

    from mojive.assets import resolve
    from mojive.mujoco_audit import audit_model
    from mojive.render.builder import SceneSourceBuilder
    from mojive.types import InstancePoseSource, InstanceVisual, MeshKey, MeshShape

    a = MuJoCoAdapter(resolve("deformables"))
    try:
        src = a.scene_source()
        surface_count = sum(int(a.model.flex_dim[i]) > 1 for i in range(a.model.nflex))
        assert len(src.dynamic_meshes) == a.model.nflex + a.model.nskin + surface_count == 6
        assert audit_model(a.model)["unsupported"] == 0
        assert sum(src.geom_pose_source == int(InstancePoseSource.WORLD)) == 6
        deformable_nodes = [
            node for node in a.nodes() if node.type in (NodeType.FLEX, NodeType.SKIN)
        ]
        deformable_ids = {node.object_id for node in deformable_nodes}
        assert len(deformable_ids) == 4
        assert 0 not in deformable_ids
        world_instances = src.geom_pose_source == int(InstancePoseSource.WORLD)
        assert set(src.geom_object_id[world_instances].tolist()) == deformable_ids
        assert sum(src.geom_visual == int(InstanceVisual.FLEX_FACE)) == surface_count
        assert sum(src.geom_visual == int(InstanceVisual.FLEX_SKIN)) == surface_count
        assert sum(src.geom_visual == int(InstanceVisual.FLEX_EDGE)) == 1
        assert sum(src.geom_visual == int(InstanceVisual.SKIN)) == a.model.nskin
        assert len(src.flex_vertex_indices) == a.model.nflexvert
        assert len(src.flex_edges) == a.model.nflexedge
        assert src.flex_vertex_rgba.shape == (a.model.nflexvert, 4)
        assert src.flex_edge_rgba.shape == (a.model.nflexedge, 4)
        assert src.flex_vertex_ranges.shape == (a.model.nflex, 2)
        expected_static = np.zeros(src.instance_count, bool)
        posed = src.geom_pose_source != int(InstancePoseSource.WORLD)
        expected_static[posed] = a.model.body_weldid[src.geom_body[posed]] == 0
        assert np.array_equal(src.geom_static, expected_static)

        hinge = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_JOINT, "skin_tip_hinge")
        assert a.set_qpos(int(a.model.jnt_qposadr[hinge]), np.deg2rad(35.0))
        a.step(2)
        frame = a.frame(FrameNeeds(poses=True, deformables=True))
        assert frame.mesh_updates is not None
        assert frame.flex_vertices == pytest.approx(a.data.flexvert_xpos, abs=1e-6)
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

        option.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = False
        option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
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
            if int(a.model.flex_dim[flex_id]) == 1:
                continue
            update = updates[MeshKey(MeshShape.FLEX_FACE, flex_id)]
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

        island_frame = a.frame(FrameNeeds(poses=True, deformables=True, islands=True))
        option.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = True
        mujoco.mjv_updateScene(
            a.model,
            a.data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            ref,
        )
        flex_colors = {
            int(geom.objid): np.asarray(geom.rgba).copy()
            for geom in ref.geoms[: ref.ngeom]
            if int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_FLEX)
        }
        assert set(flex_colors) == set(range(a.model.nflex))
        for flex_id, color in flex_colors.items():
            assert island_frame.flex_island_rgba[flex_id] == pytest.approx(color, abs=2e-6)

        builder = SceneSourceBuilder()
        scene = builder.set_source(src)
        assert scene.count == src.instance_count - surface_count
        builder.update(frame)
        dynamic_rows = np.flatnonzero(np.isin(scene.object_id, list(deformable_ids)))
        assert len(dynamic_rows) == a.model.nflex + a.model.nskin
        for row in dynamic_rows:
            assert scene.transforms[row] == pytest.approx(np.eye(4))

        assert builder.set_visual_options(
            static=True,
            skin=True,
            flex_face=True,
            flex_skin=False,
        )
        assert builder.scene.count == scene.count

        empty = a.frame(FrameNeeds.none())
        assert empty.mesh_updates is None
        assert empty.flex_vertices is None
        assert a.frame(FrameNeeds(deformables=True)).mesh_updates is updates
    finally:
        a.release()


def test_deformables_contribute_to_session_bounds():
    from mojive.assets import resolve
    from mojive.session import Session

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
    from mojive.assets import resolve

    a = MuJoCoAdapter(resolve("mujoco_visuals"))
    try:
        cameras = {c.name: c.camera_id for c in a.cameras()}
        assert set(cameras) == {"overview", "calibrated_shift", "ball_camera"}
        camera_nodes = [node for node in a.nodes() if node.type is NodeType.CAMERA]
        assert [node.camera_index for node in camera_nodes] == [0, 1, 2]
        assert [node.object_id for node in camera_nodes] == [
            camera.object_id for camera in a.cameras()
        ]
        diagnostic_frame = a.frame(FrameNeeds(poses=True, diagnostics=True))
        assert diagnostic_frame.cameras is not None
        assert len(diagnostic_frame.cameras) == a.model.ncam
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


def test_mujoco_camera_intrinsics_preserve_principal_point(tmp_path):
    from mojive.mujoco_audit import audit_model

    path = tmp_path / "intrinsic_camera.xml"
    path.write_text(
        """<mujoco><worldbody>
  <camera name="calibrated" sensorsize=".036 .024" focal=".05 .04"
          principal=".003 -.002" resolution="1920 1080"/>
</worldbody></mujoco>"""
    )
    a = MuJoCoAdapter(path)
    try:
        view = a.camera_view(0)
        assert view.uses_intrinsics()
        assert view.focal_length == pytest.approx([0.05, 0.04])
        assert view.sensor_size == pytest.approx([0.036, 0.024])
        assert view.principal_offset == pytest.approx([0.003, -0.002])
        projection = view.proj_matrix()
        assert projection[0, 2] == pytest.approx(1.0 / 6.0)
        assert projection[1, 2] == pytest.approx(1.0 / 6.0)
        report = audit_model(a.model)
        finding = next(
            item for item in report["findings"] if item["feature"] == "camera principal point"
        )
        assert finding["status"] == "supported"
    finally:
        a.release()


def test_mujoco_camera_switch_to_fov_clears_persisted_intrinsics(tmp_path):
    path = tmp_path / "intrinsic_camera.xml"
    path.write_text(
        """<mujoco><worldbody>
  <camera name="calibrated" sensorsize=".036 .024" focal=".05 .04"
          principal=".003 -.002" resolution="1920 1080"/>
</worldbody></mujoco>"""
    )
    adapter = MuJoCoAdapter(path)
    try:
        original = adapter.camera_view(0)
        fov_view = replace(
            original,
            fov_y=float(np.deg2rad(60.0)),
            focal_length=np.zeros(2, np.float32),
            sensor_size=np.zeros(2, np.float32),
            principal_offset=np.zeros(2, np.float32),
        )

        assert adapter.set_camera_view(0, fov_view)
        recompiled = adapter._root_spec.compile()

        assert adapter.camera_view(0).uses_intrinsics() is False
        assert recompiled.cam_sensorsize[0] == pytest.approx(np.zeros(2))
        assert recompiled.cam_fovy[0] == pytest.approx(60.0)
    finally:
        adapter.release()


def test_many_lights_audit_matches_the_renderer_capacity():
    from mojive.assets import resolve
    from mojive.mujoco_audit import audit_model

    adapter = MuJoCoAdapter(resolve("many_lights"))
    try:
        report = audit_model(adapter.model)
        assert not any(item["feature"] == "lights" for item in report["findings"])
    finally:
        adapter.release()


def test_site_and_camera_rangefinders_publish_generic_diagnostics():
    from mojive.assets import resolve

    a = MuJoCoAdapter(resolve("rangefinder"))
    try:
        frame = a.frame(FrameNeeds(poses=True, diagnostics=True))
        diagnostics = frame.diagnostics
        assert diagnostics is not None
        assert len(diagnostics.rangefinder_lines) == 8
        assert diagnostics.rangefinder_lines.tolist() == [True, False, *([True] * 6)]
        assert diagnostics.rangefinder_points.tolist() == [True, False, *([True] * 6)]
        assert diagnostics.rangefinder_normal_arrows.tolist() == [
            True,
            False,
            *([True] * 6),
        ]
        assert diagnostics.rangefinder_starts[0] == pytest.approx([-0.18, 0.0, 2.0])
        assert diagnostics.rangefinder_ends[0] == pytest.approx([-0.18, 0.0, 0.0])
        assert diagnostics.rangefinder_normals[0] == pytest.approx([0.0, 0.0, 1.0])
        assert diagnostics.rangefinder_starts[2:] == pytest.approx(
            np.repeat([[0.0, 0.0, 2.25]], 6, axis=0)
        )

        start = int(a.model.sensor_adr[2])
        output = a.data.sensordata[start:].reshape(6, 7)
        assert diagnostics.rangefinder_ends[2:] == pytest.approx(output[:, 1:4])
        assert diagnostics.rangefinder_normals[2:] == pytest.approx(output[:, 4:7])
        assert np.allclose(np.linalg.norm(diagnostics.rangefinder_normals[2:], axis=1), 1.0)
    finally:
        a.release()


def test_connect_and_weld_constraints_publish_mujoco_endpoint_markers():
    from mojive.assets import resolve

    a = MuJoCoAdapter(resolve("constraints"))
    try:
        source = a.scene_source().diagnostics
        frame = a.frame(FrameNeeds(poses=True, diagnostics=True)).diagnostics
        assert frame is not None
        assert source.constraint_radius == pytest.approx(
            a.model.stat.meansize * a.model.vis.scale.constraint
        )
        assert source.constraint_connect_rgba == pytest.approx(a.model.vis.rgba.connect)
        assert source.constraint_rgba == pytest.approx(a.model.vis.rgba.constraint)
        assert frame.constraint_visible.tolist() == [True, True]
        assert frame.constraint_starts[0] == pytest.approx(a.data.site_xpos[0])
        assert frame.constraint_ends[0] == pytest.approx(a.data.site_xpos[1])
        first, second = a.model.eq_obj1id[1], a.model.eq_obj2id[1]
        assert frame.constraint_starts[1] == pytest.approx(
            a.data.xpos[first] + a.data.xmat[first].reshape(3, 3) @ a.model.eq_data[1, 3:6]
        )
        assert frame.constraint_ends[1] == pytest.approx(
            a.data.xpos[second] + a.data.xmat[second].reshape(3, 3) @ a.model.eq_data[1, :3]
        )

        assert a.set_equality_enabled(0, False)
        frame = a.frame(FrameNeeds(poses=True, diagnostics=True)).diagnostics
        assert frame.constraint_visible.tolist() == [False, True]
    finally:
        a.release()


def test_mujoco_geom_groups_rebuild_scene_nodes_and_raycast_mask():
    from mojive.assets import resolve

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
        assert not any(n.type is NodeType.SITE for n in a.nodes())
        assert not a.set_visual_group("geom", 6, True)
        assert not a.set_visual_group("unknown", 0, True)
    finally:
        a.release()
