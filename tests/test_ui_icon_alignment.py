"""Manual offsets and automatic comparisons share candidate geometry without repeated work."""

import ast

import numpy as np
import pytest

from mojive.tools.ui_feasibility import icon_alignment
from mojive.tools.ui_feasibility.state import ProbeState
from mojive.tools.ui_feasibility.tuning import _geometry_values_text, _icon_values_text
from mojive.ui.icons import ICON_FAMILIES, production_icon_offset

NAMES = tuple(name for _, icons in ICON_FAMILIES for _, name in icons)


@pytest.mark.parametrize("size", (14.0, 24.0, 56.0, 112.0))
def test_default_library_matches_production_placement_and_shape_at_each_size(size):
    from mojive.ui.icons import draw_concept_icon, draw_icon
    from tests.test_ui_icon_concepts import _RecordingDraw

    state = ProbeState()
    for name in NAMES:
        reference, candidate = _RecordingDraw(), _RecordingDraw()
        center, color = (120.0, 80.0), (1.0,) * 4
        draw_icon(reference, center, size, name, color)
        draw_concept_icon(
            candidate,
            state.icon_center(name, center, size),
            size,
            name,
            color,
            padding=state.icon_padding_for_glyph(name),
            stroke_width=state.icon_stroke_for_glyph(name),
            alignment=state.icon_alignment_for_glyph(name),
            mouse_width=state.concept_mouse_width(),
            tuning=state.icon_tuning(),
            rotate_ring_gap_ratio=state.rotate_ring_gap_ratio,
            rotate_ring_cap=state.rotate_ring_cap,
        )
        assert candidate.bounds == pytest.approx(reference.bounds), name
        assert candidate.widths == pytest.approx(reference.widths), name


def test_all_library_candidates_have_finite_centroids_and_independent_offsets():
    state = ProbeState(icon_offsets_by_glyph={"helper-camera": (0.4, -1.2)})
    for name in NAMES:
        center = state.icon_centroid(name)
        assert np.isfinite(center).all() and np.max(np.abs(center)) < 12
        assert state.icon_offset(name) == (
            (0.4, -1.2) if name == "helper-camera" else production_icon_offset(name)
        )
    state.icon_auto_align = True
    for strength in (0.0, 1.0, 2.5):
        state.icon_alignment_strength = strength
        for name in NAMES:
            centroid = np.asarray(state.icon_centroid(name))
            assert np.add(centroid, state.icon_offset(name)) == pytest.approx(
                centroid * (1 - strength)
            )
    state.icon_auto_align = False
    assert state.icon_offset("helper-camera") == (0.4, -1.2)
    assert state.icon_offsets_by_glyph == {"helper-camera": (0.4, -1.2)}


def test_translation_strength_and_preview_size_reuse_measurement_but_shape_changes_recompute():
    icon_alignment.candidate_centroid.cache_clear()
    state = ProbeState(icon_auto_align=True)
    center = state.icon_centroid("helper-camera")
    for index in range(20):
        state.icon_alignment_strength = index / 10
        state.icon_offsets_by_glyph["helper-camera"] = (index / 10, -1.0)
        for size in (14, 24, 56, 112):
            state.icon_center("helper-camera", (20, 30), size)
    assert icon_alignment.candidate_centroid.cache_info().misses == 1
    state.set_icon_padding_for_glyph("helper-camera", 3.0)
    assert state.icon_centroid("helper-camera") != center
    assert icon_alignment.candidate_centroid.cache_info().misses == 2
    state.set_icon_alignment_for_glyph("helper-camera", "circle")
    state.icon_centroid("helper-camera")
    assert icon_alignment.candidate_centroid.cache_info().misses == 3


@pytest.mark.parametrize("export", (_geometry_values_text, _icon_values_text))
def test_exports_keep_all_manual_offsets_while_previewing_automatic_alignment(export, monkeypatch):
    state = ProbeState(
        icon_offsets_by_glyph={"helper-camera": (0.4, -1.2), "status-warning": (-0.1, 0.3)},
        icon_auto_align=True,
        icon_alignment_strength=2.5,
        apply_icon_offsets=False,
        link_mirrored_icon_offsets=True,
    )

    def unexpected_measurement(*_args):
        pytest.fail("Exporting stored parameters must not rasterize the library")

    monkeypatch.setattr(state, "icon_centroid", unexpected_measurement)
    fields = {
        key: ast.literal_eval(value)
        for key, value in (
            line.removesuffix(",").split("=", 1) for line in export(state).splitlines()
        )
    }
    assert fields["icon_auto_align"] is True
    assert fields["icon_alignment_strength"] == 2.5
    assert fields["icon_offset_grid_units"] == 24
    assert fields["apply_icon_offsets"] is False
    assert fields["link_mirrored_icon_offsets"] is True
    for name in NAMES:
        assert fields[f"icon_manual_offset_{name.replace('-', '_')}"] == state.icon_manual_offset(
            name
        )


@pytest.mark.parametrize(("left", "right"), icon_alignment.MIRROR_PAIRS)
def test_mirrored_edits_and_resets_preserve_the_pair_and_other_glyphs(left, right):
    state = ProbeState(link_mirrored_icon_offsets=True)
    state.set_icon_manual_offset("helper-camera", (0.1, -0.8))
    state.set_icon_manual_offset(left, (0.5, -0.7))
    assert state.icon_manual_offset(right) == (-0.5, -0.7)
    state.set_icon_manual_offset(right, (-0.2, 0.3))
    assert state.icon_manual_offset(left) == (0.2, 0.3)
    state.link_mirrored_icon_offsets = False
    state.set_icon_manual_offset(left, (0.1, -0.2))
    assert state.icon_manual_offset(right) == (-0.2, 0.3)
    state.link_mirrored_icon_offsets = True
    state.set_icon_manual_offset(right, None)
    assert state.icon_manual_offset(left) == production_icon_offset(left)
    assert state.icon_manual_offset(right) == production_icon_offset(right)
    assert state.icon_manual_offset("helper-camera") == (0.1, -0.8)


@pytest.mark.parametrize("automatic", (False, True))
def test_preview_switch_bypasses_translation_without_erasing_edits_or_measuring(
    automatic, monkeypatch
):
    state = ProbeState(icon_auto_align=automatic)
    state.set_icon_manual_offset("helper-camera", (0.5, -0.8))
    applied = state.icon_center("helper-camera", (30, 40), 48)
    state.apply_icon_offsets = False

    def unexpected_measurement(*_args):
        pytest.fail("Disabled offsets must not perform automatic measurement")

    with monkeypatch.context() as patch:
        patch.setattr(state, "icon_centroid", unexpected_measurement)
        assert state.icon_center("helper-camera", (30, 40), 48) == (30, 40)
    assert state.icon_manual_offset("helper-camera") == (0.5, -0.8)
    state.apply_icon_offsets = True
    assert state.icon_center("helper-camera", (30, 40), 48) == applied
