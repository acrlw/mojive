"""Adapter read, command and pose-buffer ownership boundaries."""

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import AdapterCaps, AdapterCommandError, FrameNeeds, SceneAdapterBase
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.scene import Scene
from mojive.session import Session


def test_default_read_boundary_is_reentrant_and_command_guard_checks_revision():
    adapter = SceneAdapterBase()
    with (
        adapter.scene_read(),
        adapter.scene_read(),
        adapter.command_context(adapter.structure_revision),
    ):
        pass
    with (
        pytest.raises(AdapterCommandError, match="structure changed"),
        adapter.command_context(adapter.structure_revision + 1),
    ):
        pytest.fail("A stale command reached its body")


def test_workspace_checks_composite_revision_before_mapping_to_primary():
    class Primary(StaticSceneAdapter):
        def command_context(self, revision):
            pytest.fail("A stale workspace revision was mapped to a newer primary")

    primary = Scene()
    adapter = WorkspaceAdapter(Primary(primary))
    checked = adapter.structure_revision
    primary.box()
    with (
        pytest.raises(AdapterCommandError, match="structure changed"),
        adapter.command_context(checked),
    ):
        pytest.fail("A stale command reached its body")


def test_structure_sync_preserves_pending_steps_without_advancing_physics():
    steps = []

    class Simulation(StaticSceneAdapter):
        caps = AdapterCaps(simulation=True)

        def step(self, count=1):
            steps.append(count)

    scene = Scene()
    scene.box()
    session = Session(Simulation(scene))
    try:
        assert session.submit(cmd.Pause()).ok
        assert session.submit(cmd.Step(3)).ok
        scene.sphere()
        assert session.sync_structure()
        assert session.source.instance_count == 2
        assert not steps
        assert not session.sync_structure()
        session.tick(FrameNeeds())
        assert steps == [3]
    finally:
        session.release()


def test_empty_workspace_borrows_primary_pose_arrays():
    primary = Scene()
    primary.box()
    adapter = WorkspaceAdapter(StaticSceneAdapter(primary))
    try:
        for _ in range(2):
            frame = adapter.frame(FrameNeeds())
            for name in ("geom_xpos", "geom_xmat", "body_xpos", "body_xmat"):
                assert getattr(frame, name) is getattr(primary.frame, name)
    finally:
        adapter.release()


def test_workspace_reuses_owned_pose_buffers_and_handles_layout_changes():
    primary = Scene()
    main = primary.box(position=(1, 0, 0))
    authored = Scene()
    extra = authored.sphere(position=(2, 0, 0))
    adapter = WorkspaceAdapter(StaticSceneAdapter(primary), authored)
    try:
        first = adapter.frame(FrameNeeds())
        main.set_pose((3, 0, 0))
        extra.set_pose((4, 0, 0))
        second = adapter.frame(FrameNeeds())
        for name in ("geom_xpos", "geom_xmat", "body_xpos", "body_xmat"):
            assert getattr(first, name) is getattr(second, name)
            assert not np.shares_memory(getattr(second, name), getattr(primary.frame, name))
            assert not np.shares_memory(getattr(second, name), getattr(authored.frame, name))
        np.testing.assert_array_equal(second.geom_xpos[:, 0], [3, 4])
        np.testing.assert_array_equal(second.body_xpos[:, 0], [0, 3, 4])

        authored.remove(extra.object_id)
        empty = adapter.frame(FrameNeeds())
        assert empty.geom_xpos is primary.frame.geom_xpos
        authored.box(position=(5, 0, 0))
        restored = adapter.frame(FrameNeeds())
        assert restored.geom_xpos is first.geom_xpos
        np.testing.assert_array_equal(restored.geom_xpos[:, 0], [3, 5])
        np.testing.assert_array_equal(primary.frame.geom_xpos[:, 0], [3])
        authored.box(position=(6, 0, 0))
        enlarged = adapter.frame(FrameNeeds())
        assert enlarged.geom_xpos is not restored.geom_xpos
        np.testing.assert_array_equal(enlarged.geom_xpos[:, 0], [3, 5, 6])
    finally:
        adapter.release()
