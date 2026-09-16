"""Geometry roles and display presets preserve physics and shared resources."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mojive import GeometryRole, GeometryView
from mojive.adapters.base import FrameNeeds
from mojive.render.builder import SceneSourceBuilder
from mojive.types import MeshShape

mujoco = pytest.importorskip("mujoco")
from mojive.adapters.mujoco import MuJoCoAdapter  # noqa: E402
from mojive.adapters.mujoco.geometry import geometry_roles  # noqa: E402

pytestmark = pytest.mark.physics


@pytest.fixture
def adapter():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="separate" pos="-1 0 1"><freejoint/>
        <geom name="appearance" type="box" size=".4 .4 .4" contype="0" conaffinity="0"/>
        <geom name="proxy" size=".25" group="3" rgba="0 0 0 0"/>
      </body>
      <body name="shared" pos="1 0 1"><freejoint/><geom name="shared_geom" size=".3"/></body>
    </worldbody></mujoco>
    """)
    a = MuJoCoAdapter()
    a.load_model(model)
    yield a
    a.release()


def set_view(builder, view):
    return builder.set_visual_options(
        static=True,
        skin=True,
        flex_face=False,
        flex_skin=True,
        visual_geometry=view in (GeometryView.VISUAL, GeometryView.BOTH),
        collision_geometry=view in (GeometryView.COLLISION, GeometryView.BOTH),
    )


def test_automatic_roles_keep_shared_geom_and_do_not_use_group_numbers(adapter):
    roles, collision = geometry_roles(adapter.model)
    assert roles.tolist() == [3, 1, 2, 3]
    assert collision.tolist() == [True, False, True, True]
    adapter.model.geom_group[:] = 0
    assert geometry_roles(adapter.model)[0].tolist() == roles.tolist()


def test_fixed_bodies_and_moving_welds_share_only_their_own_appearance():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="decoration" size=".1" contype="0" conaffinity="0"/>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body><geom size=".2" contype="0" conaffinity="0"/><geom size=".1"/></body>
      <body><geom size=".1"/></body>
      <body><freejoint/><geom size=".1"/>
        <body><geom size=".2" contype="0" conaffinity="0"/></body>
      </body>
    </worldbody></mujoco>
    """)
    assert geometry_roles(model)[0].tolist() == [1, 3, 1, 2, 3, 2, 1]


@pytest.mark.parametrize(
    "view,expected",
    [
        (GeometryView.DEFAULT, {0, 1, 3}),
        (GeometryView.VISUAL, {0, 1, 3}),
        (GeometryView.COLLISION, {0, 2, 3}),
        (GeometryView.BOTH, {0, 1, 2, 3}),
    ],
)
def test_modes_filter_same_instances_for_rendering_and_ids(adapter, view, expected):
    source = adapter.scene_source()
    model, data = adapter.model, adapter.data
    state = (data.time, data.qpos.copy(), data.qvel.copy(), model.geom_group.copy())
    builder = SceneSourceBuilder()
    builder.set_source(source)
    set_view(builder, view)
    scene = builder.update(adapter.frame(FrameNeeds()))
    assert set(scene.segmentation[:, 0]) == expected
    if view is GeometryView.COLLISION:
        assert np.all(scene.colors[:, 3] == 1)
    assert adapter.scene_source() is source and adapter.model is model and adapter.data is data
    assert data.time == state[0]
    np.testing.assert_array_equal(data.qpos, state[1])
    np.testing.assert_array_equal(data.qvel, state[2])
    np.testing.assert_array_equal(model.geom_group, state[3])
    assert not set_view(builder, view)


def test_switch_back_restores_groups_and_manual_visibility(adapter):
    source = adapter.scene_source()
    builder = SceneSourceBuilder()
    builder.set_source(source)
    initial = builder.scene.segmentation.copy()
    for view in (
        GeometryView.BOTH,
        GeometryView.COLLISION,
        GeometryView.VISUAL,
        GeometryView.DEFAULT,
    ):
        set_view(builder, view)
    np.testing.assert_array_equal(builder.scene.segmentation, initial)
    node = next(n for n in source.nodes if n.name == "shared_geom")
    builder.set_visible(node.node_id, False)
    set_view(builder, GeometryView.COLLISION)
    assert 3 not in builder.scene.segmentation[:, 0]


def test_switch_preserves_last_published_world_poses_without_another_frame(adapter):
    builder = SceneSourceBuilder()
    builder.set_source(adapter.scene_source())
    scene = builder.update(adapter.frame(FrameNeeds()))
    original = scene.transforms.copy()
    set_view(builder, GeometryView.COLLISION)
    for geom, x in ((2, -1), (3, 1)):
        row = np.flatnonzero(builder.scene.segmentation[:, 0] == geom)[0]
        np.testing.assert_allclose(builder.scene.transforms[row, :3, 3], (x, 0, 1))
    set_view(builder, GeometryView.DEFAULT)
    np.testing.assert_array_equal(builder.scene.transforms, original)


def test_link_view_is_local_and_follow_scene_restores_the_global_mode(adapter):
    from mojive import commands as cmd
    from mojive.session import Session

    session = Session(adapter)
    link = next(n for n in session.nodes if n.name == "separate")
    # Display-only commands must not interrupt a running simulation driver.
    session._simulation_driver = object()
    try:
        model = adapter.model
        assert session.submit(cmd.SetGeometryView(link.node_id, GeometryView.COLLISION)).ok
        builder = SceneSourceBuilder()
        builder.set_source(session.source)
        set_view(builder, GeometryView.VISUAL)
        assert set(builder.scene.segmentation[:, 0]) == {0, 2, 3}
        assert adapter.model is model
        generation = session.structure_generation
        session.submit(cmd.SetGeometryView(link.node_id, GeometryView.COLLISION))
        assert session.structure_generation == generation
        session.submit(cmd.SetGeometryView(link.node_id, GeometryView.DEFAULT))
        assert session.node(link.node_id).geometry_view is None
        builder.set_source(session.source)
        assert set(builder.scene.segmentation[:, 0]) == {0, 1, 3}
    finally:
        session._simulation_driver = None


def test_link_view_does_not_change_child_links(adapter):
    source = adapter.scene_source()
    parent = next(n for n in source.nodes if n.name == "separate")
    child = next(n for n in source.nodes if n.name == "shared")
    parent.geometry_view = GeometryView.COLLISION
    child.parent = parent.node_id
    builder = SceneSourceBuilder()
    builder.set_source(source)
    set_view(builder, GeometryView.VISUAL)
    shared_row = np.flatnonzero(builder.scene.segmentation[:, 0] == 3)[0]
    # The child's shared sphere keeps its original material, not the amber override.
    np.testing.assert_allclose(builder.scene.colors[shared_row], np.r_[np.full(3, 0.5**2.2), 1])


def test_both_view_keeps_repeated_comparison_meshes_in_instanced_batches(adapter):
    from mojive.adapters.worlds import WorldInstances

    worlds = WorldInstances(
        adapter.scene_source(), adapter.frame(FrameNeeds()), np.zeros((100, 3), np.float32)
    )
    builder = SceneSourceBuilder()
    builder.set_source(worlds.scene_source())
    set_view(builder, GeometryView.BOTH)
    assert not builder.scene.transparent_buckets
    assert builder.scene.bucket_count() < 10
    assert np.any(builder.scene.colors[:, 3] < 0)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_both_keeps_transparent_shared_primitives_visible_as_collision(adapter, alpha):
    source = adapter.scene_source()
    source.geom_rgba[3, 3] = alpha
    builder = SceneSourceBuilder()
    builder.set_source(source)
    set_view(builder, GeometryView.BOTH)
    rows = np.flatnonzero(builder.scene.segmentation[:, 0] == 3)
    assert len(rows) == 1
    assert builder.scene.colors[rows[0], 3] == pytest.approx(-3.35)
    set_view(builder, GeometryView.VISUAL)
    rows = np.flatnonzero(builder.scene.segmentation[:, 0] == 3)
    assert builder.scene.colors[rows[0], 3] == alpha


@pytest.mark.parametrize("mode", ["visual", "collision", "both"])
def test_explicit_views_never_pick_a_hidden_body_through_physics_fallback(mode):
    from mojive.render.backend import NullBackend, RenderFlag
    from mojive.ui.app.viewport import _Viewport

    backend = NullBackend()
    backend._flags[RenderFlag.VISUAL_GEOMETRY] = mode in ("visual", "both")
    backend._flags[RenderFlag.COLLISION_GEOMETRY] = mode in ("collision", "both")
    app = SimpleNamespace(
        backend=backend,
        _viewport_rect=(0, 0, 100, 100),
        _viewport_image=None,
        viewport_layers=SimpleNamespace(helpers=False),
        session=SimpleNamespace(source=SimpleNamespace(nodes=[])),
    )
    # The fake intentionally has no physics session or nearest-body fallback.
    assert _Viewport._pick_at(app, (50, 50)) == 0


def test_link_view_never_picks_hidden_geometry_through_physics_fallback():
    from mojive.render.backend import NullBackend
    from mojive.ui.app.viewport import _Viewport

    app = SimpleNamespace(
        backend=NullBackend(),
        _viewport_rect=(0, 0, 100, 100),
        _viewport_image=None,
        viewport_layers=SimpleNamespace(helpers=False),
        session=SimpleNamespace(
            source=SimpleNamespace(nodes=[SimpleNamespace(geometry_view=GeometryView.COLLISION)])
        ),
    )
    assert _Viewport._pick_at(app, (50, 50)) == 0


def test_explicit_contact_pair_with_zero_masks_is_collision():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="a" size=".2" contype="0" conaffinity="0"/>
      <body><freejoint/><geom name="b" size=".2" contype="0" conaffinity="0"/></body>
    </worldbody><contact><pair geom1="a" geom2="b"/></contact></mujoco>
    """)
    roles, collision = geometry_roles(model)
    assert collision.all() and np.all(roles == int(GeometryRole.BOTH))


@pytest.mark.parametrize(
    "tag,role",
    [
        ("mojive_visual_geoms", GeometryRole.VISUAL),
        ("mojive_collision_geoms", GeometryRole.COLLISION),
        ("mojive_shared_geoms", GeometryRole.BOTH),
    ],
)
def test_explicit_presentation_role_preserves_collision_settings(tag, role):
    m = mujoco.MjModel.from_xml_string(f'''
    <mujoco><custom><text name="{tag}" data="[&quot;shape&quot;]"/></custom>
      <worldbody><geom name="shape" size=".2"/></worldbody></mujoco>''')
    before = m.geom_contype.copy()
    assert geometry_roles(m)[0].tolist() == [int(role)]
    np.testing.assert_array_equal(m.geom_contype, before)


def test_unknown_or_conflicting_roles_are_not_silently_ignored():
    for texts in (
        '<text name="mojive_visual_geoms" data="shape"/>',
        '<text name="mojive_visual_geoms" data="[1]"/>',
        '<text name="mojive_visual_geoms" data="[&quot;typo&quot;]"/>',
        '<text name="mojive_visual_geoms" data="[&quot;shape&quot;]"/><text name="mojive_shared_geoms" data="[&quot;shape&quot;]"/>',
    ):
        m = mujoco.MjModel.from_xml_string(f"""<mujoco><custom>{texts}</custom>
          <worldbody><geom name="shape" size=".2"/></worldbody></mujoco>""")
        with pytest.raises(ValueError):
            geometry_roles(m)


def test_shared_nonconvex_mesh_uses_two_shapes_only_in_both_view():
    a = MuJoCoAdapter(Path(__file__).parents[1] / "assets/convex_hull.xml")
    try:
        builder = SceneSourceBuilder()
        builder.set_source(a.scene_source())
        for view, shapes in (
            (GeometryView.VISUAL, [MeshShape.ASSET]),
            (GeometryView.COLLISION, [MeshShape.CONVEX_HULL]),
            (GeometryView.BOTH, [MeshShape.ASSET, MeshShape.CONVEX_HULL]),
        ):
            set_view(builder, view)
            assert {key.shape for key, _ in builder.scene.bucket_keys} == set(shapes)
            assert builder.scene.count == len(shapes)
            assert np.all(builder.scene.segmentation[:, 0] == 0)
    finally:
        a.release()


def test_zero_mask_contact_mesh_uses_hull_only_in_collision_view():
    xml = (Path(__file__).parents[1] / "assets/convex_hull.xml").read_text()
    xml = xml.replace('rgba="0.22', 'contype="0" conaffinity="0" rgba="0.22')
    xml = xml.replace(
        "</worldbody>",
        '<body pos="0 0 2"><freejoint/><geom name="ball" size=".1"/></body></worldbody>',
    ).replace(
        "</mujoco>",
        '<contact><pair geom1="concave_collision_mesh" geom2="ball"/></contact></mujoco>',
    )
    adapter = MuJoCoAdapter()
    try:
        adapter.load_model(mujoco.MjModel.from_xml_string(xml))
        source = adapter.scene_source()
        assert source.geom_role[0] == int(GeometryRole.BOTH)
        assert source.geom_mesh[0].shape is MeshShape.ASSET
        assert source.geom_collision_mesh[0].shape is MeshShape.CONVEX_HULL
        # Keep the separate MuJoCo Convex Hull visualization flag compatible.
        assert source.geom_convex_mesh[0] == source.geom_mesh[0]
    finally:
        adapter.release()
