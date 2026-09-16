"""Gestures make explicit decisions without owning scene or playback state."""

from mojive.interaction.timeline import TimelineMarkerIndex
from mojive.interaction.timeline_edit import TimelineEditorState


def test_read_only_selection_has_no_retime_but_edit_release_commits_once():
    state = TimelineEditorState()
    state.select_key(2, 1, additive=False, editable=False, mouse_x=10, marker_x=10)
    assert state.selected_keyframes == {2}
    assert state.preview_key(20, 2, 3) is None
    assert state.release(editable=False) == (None, False)
    state.select_key(2, 1, additive=False, editable=True, mouse_x=10, marker_x=11)
    assert state.preview_key(11, 1.1, 3) is None
    assert state.preview_key(30, 3, 3) == 3
    assert state.release(editable=True) == ((2, 3), False)
    assert state.release(editable=True) == (None, False)


def test_escape_and_lost_edit_capability_discard_held_preview():
    for cancel in (False, True):
        state = TimelineEditorState()
        state.select_key(2, 1, additive=False, editable=True, mouse_x=10, marker_x=10)
        state.preview_key(30, 3, 3)
        if cancel:
            state.cancel_key()
        assert state.release(editable=cancel) == (None, False)


def test_selection_uses_index_and_take_selection_remains_independent():
    state = TimelineEditorState(take_selection=(3, 4), selected_keyframes={9})
    index = TimelineMarkerIndex(((2, 1), (3, 2), (4, 3)))
    state.begin_empty("model", on_track=True, playing=False, time=1, additive=True)
    state.select_to(2, index, ())
    assert state.selected_keyframes == {2, 3, 9}
    assert state.take_selection == (3, 4)
    state.release(editable=True)
    state.begin_empty("ruler", on_track=False, playing=True, time=1, additive=False)
    assert state.release(editable=False) == (None, True)
    assert state.release(editable=False) == (None, False)


def test_metadata_cache_is_independent_of_visibility_and_preserves_model_view():
    from dataclasses import replace

    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds, KeyframeInfo
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.scene import Scene
    from mojive.session import Session

    class Presets(StaticSceneAdapter):
        caps = replace(StaticSceneAdapter.caps, keyframes=True)
        presets = (KeyframeInfo(1, "first", 1, 0),)

        def keyframes(self):
            return self.presets

    scene = Scene()
    scene.box()
    adapter = Presets(scene)
    session = Session(adapter)
    try:
        revision, keys = session.keyframe_revision, session.model_keyframes(0)
        assert session.submit(cmd.SetVisible(session.nodes[0].node_id, False)).ok
        assert session.keyframe_revision == revision and session.model_keyframes(0) is keys
        scene.box()
        session.tick(FrameNeeds.none(), wall_dt=0)
        assert session.keyframe_revision == revision and session.model_keyframes(0) is keys
        adapter.presets = [KeyframeInfo(1, "renamed", 2, 0)]
        scene.box()
        session.tick(FrameNeeds.none(), wall_dt=0)
        assert session.keyframe_revision > revision
        assert session.model_keyframes(0)[0].time == 2
    finally:
        session.release()
