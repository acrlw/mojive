"""Optical diagnostics must preserve production geometry and text placement."""

import json

import numpy as np
import pytest

from mojive.tools.ui_feasibility.optical import (
    FAMILIES,
    SPECIMENS,
    OpticalStudy,
    _gaussian_blur,
    _ink_mask,
    silhouette,
)
from mojive.ui.icons import ICON_FAMILIES, draw_icon_label
from tests.test_ui_icon_concepts import _RecordingDraw


@pytest.mark.parametrize("label", ("Camera", "相机", ""))
def test_icon_offset_preserves_text_and_translates_only_glyph(label):
    class LabelDraw(_RecordingDraw):
        def __init__(self):
            super().__init__()
            self.labels = []

        def text_size(self, text):
            return len(text) * 8.0, 16.0

        def text_ink_bounds(self, text):
            return 0.0, 3.0, len(text) * 8.0, 13.0

        def text(self, *args, **kwargs):
            self.labels.append((args, kwargs))

    current, candidate = LabelDraw(), LabelDraw()
    for draw, offset in ((current, (0.0, 0.0)), (candidate, (1.5, -0.75))):
        draw_icon_label(
            draw, (0, 0), (160, 30), (1.0,) * 4, 1.0, "helper-camera", label, icon_offset=offset
        )
    assert candidate.labels == current.labels
    assert np.asarray(candidate.bounds) - current.bounds == pytest.approx((1.5, -0.75) * 2)


@pytest.mark.parametrize("sigma", (0.0, 1.2, 3.0, 6.0))
def test_diagnostics_cover_review_glyphs_without_clipping_blur(sigma):
    for _, name in SPECIMENS:
        mask, centroid, (_, radius) = silhouette(name, sigma)
        assert mask.max() > 0
        assert not mask[[0, -1]].any()
        assert not mask[:, [0, -1]].any()
        assert np.isfinite(centroid).all() and 0 < radius < 40
    # A symmetric shape has no preferred translation; alpha-weighting must preserve that.
    assert silhouette("playback-pause")[1] == pytest.approx((0, 0), abs=0.01)


def test_gaussian_spreads_an_impulse_without_moving_its_centroid():
    alpha = np.zeros((81, 81))
    alpha[32, 43] = 1
    y, x = np.indices(alpha.shape)
    for sigma in (1.2, 3.0, 6.0):
        blurred = _gaussian_blur(alpha, sigma)
        assert blurred.sum() == pytest.approx(1)
        assert (blurred * x).sum() == pytest.approx(43)
        assert (blurred * y).sum() == pytest.approx(32)
        assert (blurred * (x - 43) ** 2).sum() == pytest.approx(sigma**2, rel=0.002)
        assert blurred[32, 44] / blurred[32, 43] == pytest.approx(np.exp(-0.5 / sigma**2))


def test_threshold_changes_the_circle_without_reblurring_or_moving_ink():
    low, centroid, (_, low_radius) = silhouette("helper-camera", 3.0, 0.05)
    high, high_centroid, (_, high_radius) = silhouette("helper-camera", 3.0, 0.8)
    assert low is high
    assert centroid == high_centroid
    assert low_radius > high_radius


def test_offset_export_is_per_glyph_and_does_not_change_reference():
    study = OpticalStudy(offsets={"helper-camera": (0.0, -0.81)})
    assert study.offset("helper-camera", False) == (0.0, 0.0)
    assert study.offset("helper-light") == (0.0, 0.0)
    assert json.loads(study.export()) == {
        "grid_units": 24.0,
        "alignment": "manual",
        "icon_offsets": {"helper-camera": [0.0, -0.81]},
        "link_mirrored_offsets": False,
    }


def test_components_cover_every_library_glyph_except_severity_without_duplicates():
    expected = {name for _, icons in ICON_FAMILIES for _, name in icons} - {
        "status-info",
        "status-warning",
        "status-error",
    }
    assert {name for _, name in SPECIMENS} == expected
    assert len(SPECIMENS) == len(expected)
    assert all(icons for icons in FAMILIES.values())


def test_components_link_offsets_from_either_side_without_changing_auto_measurements():
    study = OpticalStudy(link_mirrored_offsets=True)
    study.set_offset("playback-previous", (0.4, -0.6))
    assert study.offset("playback-next") == (-0.4, -0.6)
    study.set_offset("playback-next", (-0.2, 0.3))
    assert study.offset("playback-previous") == (0.2, 0.3)
    study.auto_align = True
    assert study.effective_offset("playback-next") == pytest.approx(
        -np.asarray(_ink_mask("playback-next")[1])
    )
    study.auto_align = False
    study.set_offset("playback-next", None)
    assert study.offset("playback-previous") == study.offset("playback-next") == (0, 0)


@pytest.mark.parametrize("strength", (0.0, 0.5, 1.0, 1.5, 2.5))
def test_alignment_strength_scales_correction_and_preserves_manual_offsets(strength):
    study = OpticalStudy(
        offsets={"helper-camera": (0.25, -0.81)},
        auto_align=True,
        alignment_strength=strength,
    )
    for _, name in SPECIMENS:
        center = _ink_mask(name)[1]
        assert np.add(center, study.effective_offset(name)) == pytest.approx(
            np.multiply(center, 1 - strength)
        )
        assert study.effective_offset(name, False) == (0, 0)
    automatic = study.effective_offset("helper-camera")
    study.sigma, study.threshold = 6, 0.8
    assert study.effective_offset("helper-camera") == automatic
    exported = json.loads(study.export())
    assert exported["alignment"] == "weighted-centroid"
    assert exported["alignment_strength"] == strength
    assert set(exported["icon_offsets"]) == {name for _, name in SPECIMENS}
    assert exported["icon_offsets"]["helper-camera"] == list(automatic)
    study.auto_align = False
    assert study.effective_offset("helper-camera") == (0.25, -0.81)
    assert study.offsets == {"helper-camera": (0.25, -0.81)}
