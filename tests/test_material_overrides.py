"""Material edits bind to their owner's layout, never to a recycled array index."""

from dataclasses import replace

import pytest

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene import Scene
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.types import Light, Material


class _ViewerMaterialAdapter(StaticSceneAdapter):
    def set_material(self, material_index, material):
        return False


def test_material_overlay_preserves_public_light_handle_updates():
    scene = Scene()
    scene.box()
    light = scene.add_light("key", Light())
    session = Session(_ViewerMaterialAdapter(scene))
    try:
        assert session.submit(cmd.SetMaterial(0, Material(name="viewer paint"))).ok
        light.set(replace(light.value, intensity=2.0))
        assert session.source.lights.lights[0].intensity == 2.0
        assert session.tick(FrameNeeds()).lights.lights[0].intensity == 2.0
    finally:
        session.release()


def test_viewer_material_override_survives_frames_but_not_unverified_layouts():
    scene = Scene()
    original = scene.box(material=Material(name="paint", roughness=0.8))
    session = Session(_ViewerMaterialAdapter(scene))
    try:
        edited = replace(session.source.materials[0], roughness=0.2)
        assert session.submit(cmd.SetMaterial(0, edited)).ok
        assert scene.source.materials[0].roughness == 0.8
        for _ in range(2):
            session.tick(FrameNeeds(poses=True))
            assert session.source.materials[0].roughness == 0.2
        assert session.scene_overrides.materials

        # Insertion retains the original material instance and its geometry identity.
        scene.box(material=Material(name="paint", roughness=0.6))
        assert session.sync_structure()
        assert session.source.materials[0].roughness == 0.2
        assert session.scene_overrides.materials

        # A replacement reuses the name and index, but neither target survives.
        scene.remove(original.object_id)
        assert session.sync_structure()
        assert session.source.materials[0].roughness == 0.6
        assert not session.scene_overrides.materials
        assert "material identities could not be verified" in session.last_message
        assert session.last_message_level == "warning"
    finally:
        session.release()


def test_viewer_material_override_reindexes_only_with_surviving_geometry():
    scene = Scene()
    first = scene.box(material=Material(name="first"))
    paint = Material(name="paint", roughness=0.8)
    original = scene.box(material=paint)
    session = Session(_ViewerMaterialAdapter(scene))
    try:
        edited = replace(paint, roughness=0.2)
        assert session.submit(cmd.SetMaterial(1, edited)).ok
        scene.remove(first.object_id)
        assert session.sync_structure()
        assert session.source.materials[0] is edited
        assert session.scene_overrides.materials == {0: edited}

        # Reusing even the same shared Material cannot reconnect a deleted
        # geometry identity to a new object that happens to occupy its slot.
        scene.remove(original.object_id)
        scene.box(material=paint)
        assert session.sync_structure()
        assert session.source.materials[0] is paint
        assert not session.scene_overrides.materials
    finally:
        session.release()


def test_material_invalidation_keeps_warning_when_a_command_reports_success(monkeypatch):
    messages = []
    monkeypatch.setattr("mojive.session.source.log.warning", messages.append)
    scene = Scene()
    item = scene.box()
    session = Session(_ViewerMaterialAdapter(scene))
    try:
        assert session.submit(cmd.SetMaterial(0, Material(name="viewer paint"))).ok
        result = session.submit(cmd.RemoveSceneEntity(item.object_id))
        assert result.ok
        assert session.last_message == result.message
        assert any("material identities could not be verified" in message for message in messages)
        assert not session.scene_overrides.materials
    finally:
        session.release()


def test_undo_rebinds_viewer_materials_only_after_restoring_their_exact_checkpoint():
    scene = Scene()
    scene.box(material=Material(name="paint", roughness=0.8))
    session = Session(_ViewerMaterialAdapter(scene))
    try:
        for roughness in (0.2, 0.4):
            assert session.submit(
                cmd.SetMaterial(0, replace(session.source.materials[0], roughness=roughness))
            ).ok
        for command, expected in ((cmd.Undo(), 0.2), (cmd.Redo(), 0.4), (cmd.Undo(), 0.2)):
            assert session.submit(command).ok
            assert session.source.materials[0].roughness == expected
            session.tick(FrameNeeds())
            assert session.source.materials[0].roughness == expected
        scene.box()
        assert session.sync_structure()
        assert session.source.materials[0].roughness == 0.2
    finally:
        session.release()


@pytest.mark.physics
def test_material_writeback_survives_asset_reindexing_and_undo():
    pytest.importorskip("mujoco")
    from mojive.adapters.mujoco import MuJoCoAdapter

    primary = MuJoCoAdapter(resolve("empty"))
    session = None
    try:
        assert primary.set_scene_model_xml(
            0,
            """
        <mujoco><asset>
          <material name="A" rgba="1 0 0 1"/>
          <material name="B" rgba="0 1 0 1"/>
          <material name="C" rgba="0 0 1 1"/>
        </asset><worldbody>
          <geom name="b" type="box" size=".2 .2 .2" material="B"/>
          <geom name="c" type="box" size=".2 .2 .2" pos="1 0 0" material="C"/>
        </worldbody></mujoco>
        """,
        )
        session = Session(WorkspaceAdapter(primary))
        assert session.submit(cmd.Pause()).ok
        assert session.submit(
            cmd.SetMaterial(1, replace(session.source.materials[1], roughness=0.17))
        ).ok
        assert not session.scene_overrides.materials
        assert session.submit(cmd.RemoveModelAsset(0, "material", "A")).ok
        for command, names in (
            (None, ["B", "C"]),
            (cmd.Undo(), ["A", "B", "C"]),
            (cmd.Redo(), ["B", "C"]),
        ):
            if command is not None:
                assert session.submit(command).ok
            assert [material.name for material in session.source.materials[: len(names)]] == names
            b = next(material for material in session.source.materials if material.name == "B")
            assert b.roughness == pytest.approx(0.17)
            c_index = next(
                i for i, material in enumerate(session.source.materials) if material.name == "C"
            )
            assert session.source.geom_material[1] == c_index
    finally:
        if session is not None:
            session.release()
        else:
            primary.release()
