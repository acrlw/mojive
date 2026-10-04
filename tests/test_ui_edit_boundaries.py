"""UI operation boundaries stay coherent across model queues and drawing frames."""

import pytest

from mojive import commands as cmd
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.config import ViewerConfig
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.panels import PanelContext


@pytest.mark.physics
@pytest.mark.parametrize("mode", ["synchronous", "live", "deferred"])
@pytest.mark.parametrize("fail_second", [False, True])
def test_multi_key_delete_is_one_atomic_user_operation(tmp_path, monkeypatch, mode, fail_second):
    path = tmp_path / "keys.xml"
    path.write_text(
        '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody>'
        '<keyframe><key name="first"/><key name="second"/></keyframe></mujoco>'
    )
    adapter = MuJoCoAdapter(path)
    session = Session(adapter)
    try:
        assert session.submit(cmd.Pause()).ok
        original = session.keyframes
        editor = TimelineEditor()
        editor.edit_lane = "model"
        editor.selected_keyframes = {key.keyframe_id for key in original}
        monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
        app = ViewerApp(
            session, NullBackend(), config=ViewerConfig(live_model_updates=mode == "live")
        )
        ctx = PanelContext(
            session,
            None,
            queue_model_edits=app._queue_model_edits if mode != "synchronous" else None,
        )
        compiles = []
        compile_model = adapter._compile_composed_model

        def compile_counted():
            if not adapter._model_edit_batch_depth:
                compiles.append(True)
            return compile_model()

        monkeypatch.setattr(adapter, "_compile_composed_model", compile_counted)
        remove = adapter.remove_model_keyframe
        if fail_second:
            monkeypatch.setattr(
                adapter, "remove_model_keyframe", lambda key: False if key == 0 else remove(key)
            )
        editor.delete_selection(ctx, True)
        if mode == "live":
            assert len(app._model_load_queue) == 1
            assert session.keyframes == original
            job = app._model_load_queue.pop()
            job.completed(app._load_model(job.command))
        elif mode == "deferred":
            assert session.keyframes == original
            assert compiles == []
            draft = app.model_edits
            draft.applying = True
            app._finish_pending_model_edits(
                session.apply_model_edits(draft.resolve_commands(session))
            )
        if fail_second:
            assert session.keyframes == original
            assert not session.submit(cmd.Undo()).ok
            if mode == "deferred":
                assert app.model_edits.active and app.model_edits.error
            else:
                assert editor.error
                assert editor.selected_keyframes == {key.keyframe_id for key in original}
        else:
            assert not session.keyframes
            assert len(compiles) == 1
            assert session.submit(cmd.Undo()).ok
            assert session.keyframes == original
            assert not session.submit(cmd.Undo()).ok
            assert session.submit(cmd.Redo()).ok
            assert not session.keyframes
    finally:
        session.release()


@pytest.mark.physics
@pytest.mark.parametrize("exception", [False, True])
def test_failed_multi_stage_retains_prior_draft_and_reports_once(tmp_path, monkeypatch, exception):
    path = tmp_path / "draft.xml"
    path.write_text('<mujoco><worldbody><geom name="box" size=".1"/></worldbody></mujoco>')
    session = Session(MuJoCoAdapter(path))
    try:
        assert session.submit(cmd.Pause()).ok
        monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
        app = ViewerApp(session, NullBackend(), config=ViewerConfig(live_model_updates=False))
        draft = app.model_edits
        assert draft.stage(cmd.AddModelKeyframe(0, "kept")).ok
        stage = draft.stage

        def reject_second(command):
            if command.name == "second":
                if exception:
                    raise RuntimeError("staging failed")
                return cmd.CommandResult.bad("staging failed")
            return stage(command)

        monkeypatch.setattr(draft, "stage", reject_second)
        results = []
        commands = (cmd.AddModelKeyframe(0, "first"), cmd.AddModelKeyframe(0, "second"))
        if exception:
            with pytest.raises(RuntimeError, match="staging failed"):
                app._queue_model_edits(commands, results.append)
            assert results == []
        else:
            app._queue_model_edits(commands, results.append)
            assert len(results) == 1 and not results[0].ok
        assert draft.model_keyframe_names(0) == {"kept"}
        # A later gesture can start: failure released its temporary checkpoint.
        assert draft.stage_many((cmd.AddModelKeyframe(0, "third"),)).ok
        assert draft.model_keyframe_names(0) == {"kept", "third"}
    finally:
        session.release()
