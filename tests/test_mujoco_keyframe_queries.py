"""Keyframe queries reuse committed local models without reading stale document data."""

from dataclasses import replace

import pytest

from mojive import commands as cmd
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.session import Session

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.physics


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "keys.xml"
    path.write_text("""<mujoco><worldbody><body><joint name="joint" type="slide"/>
        <geom size="0.1"/></body></worldbody><keyframe>
        <key name="first" time="1" qpos="0.1"/><key name="second" time="2" qpos="0.2"/>
        </keyframe></mujoco>""")
    return path


def test_key_queries_reuse_one_local_compilation_and_release_it(model_path, monkeypatch):
    adapter = MuJoCoAdapter(model_path)
    compile_spec = mujoco.MjSpec.compile
    compilations = []

    def compile_once(spec, *args, **kwargs):
        compilations.append(spec)
        return compile_spec(spec, *args, **kwargs)

    monkeypatch.setattr(mujoco.MjSpec, "compile", compile_once)
    try:
        for _ in range(3):
            assert adapter.keyframe_properties(0).qpos == (0.1,)
            assert adapter.keyframe_properties(1).qpos == (0.2,)
        assert len(compilations) == 1
        adapter.release()
        assert adapter._keyframe_model is None
    finally:
        adapter.release()


def test_key_queries_follow_edit_undo_and_reload(model_path):
    session = Session(WorkspaceAdapter(MuJoCoAdapter(model_path)))
    try:
        assert session.submit(cmd.Pause())
        before = session.keyframe_properties(0)
        changed = replace(before, time=3.5, qpos=(0.75,))
        assert session.submit(cmd.SetModelKeyframe(**changed.__dict__))
        assert session.keyframe_properties(0) == changed
        assert session.submit(cmd.Undo())
        assert session.keyframe_properties(0) == before
        assert session.submit(cmd.Redo())
        assert session.keyframe_properties(0) == changed
        assert session.submit(cmd.Reload())
        assert session.keyframe_properties(0) == before
    finally:
        session.release()


def test_key_queries_switch_models_and_bypass_pending_batch_state(model_path):
    adapter = MuJoCoAdapter(model_path)
    try:
        model_id = adapter.add_scene_model(model_path, (2, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        attached = next(key for key in adapter.keyframes() if key.model_id == model_id)
        assert adapter.keyframe_properties(0).model_id == 0
        assert adapter.keyframe_properties(attached.keyframe_id).model_id == model_id
        before = adapter.keyframe_properties(0)
        with adapter.model_edit_batch():
            assert adapter.set_keyframe_properties(replace(before, time=7))
            assert adapter.keyframe_properties(0).time == 7
            assert adapter.set_keyframe_properties(replace(before, time=8))
            assert adapter.keyframe_properties(0).time == 8
        assert adapter.keyframe_properties(0).time == 8
        assert adapter.keyframe_properties(attached.keyframe_id).time == 1
    finally:
        adapter.release()
