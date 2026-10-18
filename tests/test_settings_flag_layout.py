"""Rendering flags retain adjacent labels, responsive columns and backend commands."""

from copy import deepcopy
from dataclasses import replace
from itertools import pairwise

import pytest

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.render.backend import RenderFlag
from mojive.ui.panels.settings_pages.rendering import flag_groups, render_flag_label
from tests import test_ui_design_settings

settings_fixture = test_ui_design_settings.settings


@pytest.fixture
def flags(settings_fixture, monkeypatch, tmp_path):
    settings = settings_fixture
    settings.ui.design = DesignPresentation(
        settings.ui, DesignRuntime(SpecificationStore(tmp_path / "flag-design.json"))
    )
    settings.settings._category = "Rendering"
    labels = {}
    paint_text = settings.ui.component_text

    def observe(part, rect, text, *args, **kwargs):
        if part == "form_label":
            labels[text] = rect
        return paint_text(part, rect, text, *args, **kwargs)

    monkeypatch.setattr(settings.ui, "component_text", observe)
    settings.frame(2)
    return settings, labels


def test_renderer_flag_checkbox_stays_next_to_its_own_label(flags):
    settings, labels = flags
    positions = []
    for flag in flag_groups()[-1][1]:
        label = labels[render_flag_label(flag, settings.ctx.tr, localized=True)]
        x, y, width, height = settings.ui.hits["##rf_" + flag.value]
        assert label.x - (x + width) == pytest.approx(12 * settings.s, abs=0.001)
        assert label.y + label.height / 2 == pytest.approx(y + height / 2)
        assert label.width >= settings.ui.measure(
            render_flag_label(flag, settings.ctx.tr, localized=True),
            14,
            translate=False,
            font_resolved=True,
        )
        positions.append((x, y, label))
    for (_, y, label), (next_x, next_y, _) in pairwise(positions):
        if y == next_y:
            assert next_x - label.right >= 24 * settings.s - 0.5
    # The same table wraps its four renderer options in a narrow Settings page.
    assert len({y for _, y, _ in positions}) >= 2


def test_flag_spacing_edits_preserve_separate_item_and_column_gaps(flags):
    settings, labels = flags
    runtime = settings.ui.design.runtime
    spec = deepcopy(runtime.spec)
    spec["parts"]["visual_group_item"] = {"columnGap": 7}
    spec["parts"]["visual_group"] = {"columnGap": 31}
    runtime.update_spec(spec)
    settings.frame()
    positions = []
    for flag in flag_groups()[-1][1]:
        label = labels[render_flag_label(flag, settings.ctx.tr, localized=True)]
        x, y, width, _height = settings.ui.hits["##rf_" + flag.value]
        assert label.x - (x + width) == pytest.approx(7 * settings.s, abs=0.001)
        positions.append((x, y, label))
    for (_, y, label), (next_x, next_y, _) in pairwise(positions):
        if y == next_y:
            assert next_x - label.right == pytest.approx(31 * settings.s, abs=0.001)


def test_renderer_flag_hit_region_routes_supported_toggle_and_keeps_unsupported_disabled(
    flags, monkeypatch
):
    settings, _ = flags
    backend = settings.ctx.backend
    backend.caps = replace(backend.caps, render_flags=frozenset({RenderFlag.OUTLINE}))
    writes = []

    def set_flag(flag, value):
        backend._flags[flag] = value
        writes.append((flag, value))
        return True

    monkeypatch.setattr(backend, "set_flag", set_flag)
    settings.frame()
    settings.click("##rf_outline")
    assert writes == [(RenderFlag.OUTLINE, True)]
    settings.click("##rf_outline")
    assert writes[-1] == (RenderFlag.OUTLINE, False)
    settings.click("##rf_tonemap")
    assert len(writes) == 2
