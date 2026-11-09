"""Copied keyframes retain model state and paste through one atomic edit boundary."""

from dataclasses import replace

import pytest

from mojive import commands as cmd
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.session import Session
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.panels import PanelContext

pytestmark = pytest.mark.physics


@pytest.fixture
def clipboard(tmp_path):
    path = tmp_path / "keys.xml"
    path.write_text("""<mujoco><worldbody><body><joint name="j"/>
      <geom size=".1"/></body></worldbody><actuator><motor joint="j"/></actuator>
      <keyframe><key name="a" time="1" qpos=".2" qvel=".3" ctrl=".4"/>
      <key name="b" time="1.5" qpos=".5" qvel=".6" ctrl=".7"/></keyframe></mujoco>""")
    session = Session(MuJoCoAdapter(path), path)
    session.submit(cmd.Pause())
    editor = TimelineEditor()
    editor.set_model(0)
    editor.selected_keyframes = {key.keyframe_id for key in session.keyframes}
    ctx = PanelContext(session, None)
    try:
        yield editor, ctx
    finally:
        session.release()


def test_paste_preserves_copied_state_offsets_and_one_undo(clipboard):
    editor, ctx = clipboard
    editor.copy_keyframes(ctx)
    expected = editor._clipboard
    editor.delete_selection(ctx, True)
    editor.playhead = 3
    editor.paste_keyframes(ctx)
    assert not editor.error
    actual = tuple(
        ctx.session.keyframe_properties(key.keyframe_id) for key in ctx.session.keyframes
    )
    assert len(actual) == 2
    for original, pasted in zip(expected, actual, strict=True):
        assert pasted == replace(
            original, keyframe_id=pasted.keyframe_id, name=pasted.name, time=original.time + 2
        )
    assert editor.selected_id in {key.keyframe_id for key in actual}
    assert ctx.session.submit(cmd.Undo()).ok
    assert not ctx.session.keyframes
    assert ctx.session.submit(cmd.Redo()).ok
    assert len(ctx.session.keyframes) == 2


def test_failed_second_insert_rolls_back_first_and_preserves_selection(clipboard, monkeypatch):
    editor, ctx = clipboard
    editor.copy_keyframes(ctx)
    before = ctx.session.keyframes
    add = ctx.session.adapter.add_model_keyframe
    calls = []

    def insert(model_id, name, *, state=None):
        calls.append(name)
        return add(model_id, name, state=state) if len(calls) == 1 else -1

    monkeypatch.setattr(ctx.session.adapter, "add_model_keyframe", insert)
    editor.paste_keyframes(ctx)
    assert editor.error
    assert len(calls) == 2
    assert ctx.session.keyframes == before
    assert editor.selected_keyframes == {key.keyframe_id for key in before}


def test_clipboard_is_model_and_document_scoped(clipboard):
    editor, ctx = clipboard
    editor.copy_keyframes(ctx)
    assert editor.can_paste_keyframes(ctx)
    editor.set_model(12)
    assert not editor.can_paste_keyframes(ctx)
    editor.set_model(0)
    editor._clipboard_document = "previous document"
    assert not editor.can_paste_keyframes(ctx)


def test_duplicate_places_copies_after_selection_without_loading_it(clipboard):
    editor, ctx = clipboard
    before = ctx.session.adapter.capture_state().qpos.copy()
    editor.paste_keyframes(ctx, duplicate=True)
    assert not editor.error
    assert [key.time for key in ctx.session.keyframes] == pytest.approx([1, 1.5, 1.6, 2.1])
    assert (ctx.session.adapter.capture_state().qpos == before).all()
