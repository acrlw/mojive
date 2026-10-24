"""Reusable native field buffers preserve nesting, paint order and borrowed lifetimes."""

from types import SimpleNamespace

import pytest

from mojive.ui.interaction_paint import FieldLayer


class Splitter:
    def __init__(self):
        self.draw = None

    def split(self, draw, count):
        assert self.draw is None and count == 2
        self.draw = draw

    def set_current_channel(self, draw, channel):
        assert draw is self.draw
        draw.append(channel)

    def merge(self, draw):
        assert draw is self.draw
        draw.append("merge")
        self.draw = None


def test_fields_reuse_only_inactive_splitters_and_release_draw_lists():
    created, scratch = [], []

    def create():
        splitter = Splitter()
        created.append(splitter)
        return splitter

    api = SimpleNamespace(ImDrawListSplitter=create)
    for _ in range(100):
        parent, popup = [], []
        with FieldLayer(api, parent, scratch) as outer:
            with FieldLayer(api, popup, scratch) as inner:
                assert inner.splitter is not outer.splitter
                inner.background()
            assert inner.draw is inner.splitter is None
            outer.background()
        assert outer.draw is outer.splitter is None
        assert parent == popup == [1, 0, "merge"]
    assert len(created) == len(scratch) == 2


def test_field_exception_merges_and_releases_borrowed_list():
    scratch, draw = [], []
    api = SimpleNamespace(ImDrawListSplitter=Splitter)
    with pytest.raises(ValueError, match="paint failed"), FieldLayer(api, draw, scratch) as layer:
        layer.background()
        raise ValueError("paint failed")
    assert draw == [1, 0, "merge"]
    assert layer.draw is layer.splitter is None
    assert len(scratch) == 1
