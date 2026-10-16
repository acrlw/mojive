"""Every drag update remains atomic and visible; durability is completed on release."""

from unittest.mock import Mock

import pytest

from mojive.design.editor import SpecificationEditor
from mojive.design.store import SpecificationStore, read_spec


def test_drag_updates_publish_immediately_and_sync_once_at_release(tmp_path, monkeypatch):
    sync = Mock()
    monkeypatch.setattr("mojive.design.store.os.fsync", sync)
    store = SpecificationStore(tmp_path / "spec.json")
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    original = editor.spec["parts"]["switch"]["switchHeight"]
    path = ("parts", "switch", "switchHeight")
    for height in (26, 28, 30):
        assert editor.change(path, height, gesture=True)
        assert store.latest()["spec"]["parts"]["switch"]["switchHeight"] == height
    assert editor.pending_sync
    assert sync.call_count == 0
    assert len(editor.undo_stack) == 1
    editor.end_gesture()
    assert sync.call_count == 1
    assert not editor.pending_sync
    editor.undo()
    assert store.latest()["spec"]["parts"]["switch"]["switchHeight"] == original
    assert sync.call_count == 2


def test_failed_sync_keeps_pending_state_for_retry(tmp_path, monkeypatch):
    store = SpecificationStore(tmp_path / "spec.json")
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    editor.change(("parts", "switch", "switchHeight"), 26, gesture=True)
    sync = Mock(side_effect=[OSError("disk failed"), None])
    monkeypatch.setattr("mojive.design.store.os.fsync", sync)
    with pytest.raises(OSError, match="disk failed"):
        editor.end_gesture()
    assert editor.pending_sync
    assert read_spec(store.live_path)["parts"]["switch"]["switchHeight"] == 26
    editor.end_gesture()
    assert not editor.pending_sync
