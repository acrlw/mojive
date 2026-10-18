"""Low-contrast ground is a reversible render recipe over backend-neutral scene data."""

from copy import deepcopy
from dataclasses import asdict, fields

import numpy as np
import pytest

from mojive.adapters.base import SceneFrame, SceneSource
from mojive.render.builder import SceneSourceBuilder
from mojive.types import (
    GeometryRole,
    GeometryStyle,
    GeometryView,
    Material,
    MeshKey,
    MeshShape,
    TextureData,
    TextureType,
)


def source_and_frame():
    material = Material(
        name="shared_checker",
        rgba=np.array((0.8, 0.6, 0.3, 1), np.float32),
        texture="checker",
        tex_repeat=np.array((3, 2), np.float32),
        tex_uniform=True,
        emission=0.1,
        specular=0.6,
        shininess=0.8,
        reflectance=0.3,
    )
    pixels = np.array([[[240, 240, 240], [20, 20, 20]], [[20, 20, 20], [240, 240, 240]]], np.uint8)
    source = SceneSource(
        materials=[material],
        textures={"checker": TextureData("checker", TextureType.TWO_D, pixels)},
        geom_mesh=[MeshKey(MeshShape.BOX), *[MeshKey(MeshShape.PLANE)] * 3],
        geom_material=[0, 0, 0, -1],
        geom_size=np.array(((1, 2, 1), (0, 0, 1), (2, 3, 0.1), (0, 0, 1)), np.float32),
        geom_rgba=np.array(
            ((0.7, 0.1, 0.2, 1), (0.2, 0.8, 0.4, 1), (0.4, 0.3, 0.9, 1), (0.9, 0.7, 0.1, 1)),
            np.float32,
        ),
        geom_object_id=np.arange(101, 105, dtype=np.uint32),
        geom_source=np.arange(4, dtype=np.int32),
        geom_local=np.tile(np.eye(4, dtype=np.float32), (4, 1, 1)),
        geom_infinite_plane=np.array((False, True, False, True)),
        geom_role=np.full(4, int(GeometryRole.BOTH), np.uint8),
        initial_qpos=np.array((1, 2, 3), np.float32),
    )
    frame = SceneFrame(
        time=1.25,
        step=125,
        geom_xpos=np.array(((1, 2, 3), (0, 0, 0), (2, 0, 1), (0, 0, -1)), np.float32),
        geom_xmat=np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)),
        qpos=np.array((0.1, -0.7, 2), np.float32),
        qvel=np.array((0.5, 1.2, 0), np.float32),
        ctrl=np.array((0.3,), np.float32),
    )
    return source, frame


def appearance(builder, quiet, *, view=GeometryView.DEFAULT, island=False):
    return builder.set_visual_options(
        static=True,
        skin=True,
        flex_face=False,
        flex_skin=True,
        island=island,
        visual_geometry=view in (GeometryView.VISUAL, GeometryView.BOTH),
        collision_geometry=view in (GeometryView.COLLISION, GeometryView.BOTH),
        geometry_style=GeometryStyle(quiet_ground=quiet),
    )


def row_for(scene, object_id):
    return int(np.flatnonzero(scene.object_id == object_id)[0])


def material_for(scene, object_id):
    row = row_for(scene, object_id)
    return scene.materials[scene.bucket_keys[scene.bucket[row]][1]]


def object_columns(scene):
    order = np.argsort(scene.object_id)
    return {
        name: getattr(scene, name)[order].copy()
        for name in ("object_id", "transforms", "colors", "material", "tex_coef", "cube_coef")
    }


@pytest.mark.parametrize("alpha", (1.0, 0.4))
def test_quiet_ground_preserves_source_assets_physics_and_ordinary_shared_material(alpha):
    source, frame = source_and_frame()
    source.geom_rgba[1, 3] = alpha
    original, original_frame = deepcopy(source), deepcopy(frame)
    original_material, original_texture = source.materials[0], source.textures["checker"]
    builder = SceneSourceBuilder()
    builder.set_source(source)
    baseline = object_columns(builder.update(frame))
    assert appearance(builder, True)
    quiet = builder.scene
    quiet.validate()
    for object_id in (101, 103):
        row = row_for(quiet, object_id)
        offset = object_id - 101
        for name in baseline:
            np.testing.assert_array_equal(getattr(quiet, name)[row], baseline[name][offset])
        assert material_for(quiet, object_id) is original_material
    for object_id in (102, 104):
        row = row_for(quiet, object_id)
        assert material_for(quiet, object_id).texture is None
        np.testing.assert_allclose(
            quiet.colors[row, :3], np.array((0.32, 0.34, 0.36), np.float32) ** 2.2
        )
        assert quiet.colors[row, 3] == source.geom_rgba[object_id - 101, 3]
        np.testing.assert_array_equal(quiet.material[row, 1:], 0)
    assert source.materials[0] is original_material
    assert source.textures["checker"] is original_texture
    np.testing.assert_array_equal(original_texture.pixels, original.textures["checker"].pixels)
    for item in fields(original_material):
        np.testing.assert_equal(
            getattr(original_material, item.name), getattr(original.materials[0], item.name)
        )
    for item in fields(source):
        if isinstance(getattr(source, item.name), np.ndarray):
            np.testing.assert_array_equal(getattr(source, item.name), getattr(original, item.name))
    for item in fields(frame):
        np.testing.assert_equal(getattr(frame, item.name), getattr(original_frame, item.name))

    assert appearance(builder, False)
    restored = object_columns(builder.scene)
    for name in baseline:
        np.testing.assert_array_equal(restored[name], baseline[name])
    assert material_for(builder.scene, 102) is original_material
    assert not appearance(builder, False)


def test_frame_rgba_updates_keep_quiet_ground_but_restore_current_scene_colors_on_toggle():
    source, frame = source_and_frame()
    builder = SceneSourceBuilder()
    builder.set_source(source)
    appearance(builder, True)
    builder.update(frame)
    ground = {
        object_id: builder.scene.colors[row_for(builder.scene, object_id)].copy()
        for object_id in (102, 104)
    }
    for intensity in (0.2, 0.7, 0.9):
        rgba = np.tile(np.array((intensity, 0.1, 0.8, 1), np.float32), (4, 1))
        original = rgba.copy()
        frame.time += 0.1
        frame.step += 10
        scene = builder.update(frame, instance_rgba=rgba)
        for object_id in (102, 104):
            np.testing.assert_array_equal(
                scene.colors[row_for(scene, object_id)], ground[object_id]
            )
        for object_id in (101, 103):
            np.testing.assert_allclose(
                scene.colors[row_for(scene, object_id), :3], rgba[0, :3] ** 2.2
            )
        np.testing.assert_array_equal(rgba, original)
        revision = scene.visual_revision
        assert builder.update(frame, instance_rgba=rgba).visual_revision == revision

    appearance(builder, False)
    expected = rgba.copy()
    expected[:, :3] **= 2.2
    np.testing.assert_allclose(object_columns(builder.scene)["colors"], expected)
    assert material_for(builder.scene, 102).texture == "checker"
    appearance(builder, True)
    builder.update(frame, instance_rgba=None)
    for object_id in (102, 104):
        np.testing.assert_array_equal(
            builder.scene.colors[row_for(builder.scene, object_id)], ground[object_id]
        )


@pytest.mark.parametrize("view", (GeometryView.COLLISION, GeometryView.BOTH))
def test_collision_ground_treatment_takes_precedence(view):
    source, frame = source_and_frame()
    source.geom_role[1] = int(GeometryRole.COLLISION)
    source.geom_role[3] = int(GeometryRole.VISUAL)
    builder = SceneSourceBuilder()
    builder.set_source(source)
    appearance(builder, False, view=view)
    before = builder.update(frame).colors[row_for(builder.scene, 102)].copy()
    before_material = builder.scene.material[row_for(builder.scene, 102)].copy()
    appearance(builder, True, view=view)
    scene = builder.update(frame, instance_rgba=np.ones((4, 4), np.float32))
    row = row_for(scene, 102)
    np.testing.assert_array_equal(scene.colors[row], before)
    np.testing.assert_array_equal(scene.material[row], before_material)
    np.testing.assert_allclose(
        scene.colors[row, :3], np.asarray(GeometryStyle().collision_color, np.float32) ** 2.2
    )
    assert material_for(scene, 102).texture is None


def test_island_colors_take_precedence_and_still_accept_dynamic_updates():
    source, frame = source_and_frame()
    source.instance_island_body = np.array((0, 0, 0, -1), np.int32)
    builder = SceneSourceBuilder()
    builder.set_source(source)
    appearance(builder, True, island=True)
    for rgba in (np.full((4, 4), 0.3, np.float32), np.full((4, 4), 0.8, np.float32)):
        rgba[:, 3] = 1
        scene = builder.update(frame, instance_rgba=rgba)
        for object_id in (101, 102, 103):
            row = row_for(scene, object_id)
            np.testing.assert_allclose(scene.colors[row, :3], rgba[object_id - 101, :3] ** 2.2)
            assert material_for(scene, object_id).texture is None
        np.testing.assert_allclose(
            scene.colors[row_for(scene, 104), :3], np.array((0.32, 0.34, 0.36), np.float32) ** 2.2
        )
    appearance(builder, True, island=False)
    np.testing.assert_allclose(
        builder.scene.colors[row_for(builder.scene, 102), :3],
        np.array((0.32, 0.34, 0.36), np.float32) ** 2.2,
    )


def test_ground_setting_defaults_to_authored_appearance_and_round_trips_without_coercion():
    assert GeometryStyle().quiet_ground is False
    style = GeometryStyle(quiet_ground=True)
    assert GeometryStyle(**asdict(style)) == style
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError, match="quiet_ground"):
            GeometryStyle(quiet_ground=value)
