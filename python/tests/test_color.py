from __future__ import annotations

import numpy as np
import pytest

from mojive.render.opengl import color


def _wrong_ambient_in_linear_domain(a, gain=2.0):

    return gain * color.srgb_to_linear(a)


def _wrong_tonemap_on_luma(rgb, knee: float = color.KNEE):

    rgb = np.asarray(rgb, np.float64)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    headroom = 1.0 - knee
    mapped = knee + color.softroll(luma - knee, headroom)
    scale = np.where(luma > knee, mapped / np.maximum(luma, 1e-12), 1.0)
    return rgb * scale[..., None]


# ---------------------------------------------------------------- sRGB
def test_srgb_roundtrip_is_exact():
    x = np.linspace(0.0, 1.0, 4097)
    assert np.abs(color.linear_to_srgb(color.srgb_to_linear(x)) - x).max() < 1e-6
    assert np.abs(color.srgb_to_linear(color.linear_to_srgb(x)) - x).max() < 1e-6


def test_srgb_is_not_a_pure_power_curve():

    x = 0.05
    exact = float(color.srgb_to_linear(x))
    approx = x**2.2
    assert abs(exact - approx) / exact > 0.1


def test_ambient_gain_lives_in_display_domain():

    a = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.5])
    assert np.allclose(
        color.ambient_linear(a, gain=2.0), color.srgb_to_linear(np.clip(2.0 * a, 0.0, 1.0))
    )

    k = color.AMBIENT_GAIN
    assert np.allclose(color.ambient_linear(a), color.srgb_to_linear(np.clip(k * a, 0.0, 1.0)))


def test_ambient_wrong_domain_is_off_by_a_factor_of_two():

    table = {0.2: 2.007, 0.25: 2.104, 0.3: 2.175, 0.4: 2.272, 0.5: 2.336}
    for a, expected in table.items():
        ratio = float(color.ambient_linear(a, gain=2.0) / _wrong_ambient_in_linear_domain(a))
        assert ratio == pytest.approx(expected, abs=0.01)

    a = np.array([0.15, 0.25, 0.35, 0.45])
    assert not np.allclose(color.ambient_linear(a, gain=2.0), _wrong_ambient_in_linear_domain(a))


def test_ambient_saturates_instead_of_overflowing():

    assert float(color.ambient_linear(0.6, gain=2.0)) == pytest.approx(1.0)
    assert float(color.ambient_linear(0.9, gain=2.0)) == pytest.approx(1.0)

    assert float(color.ambient_linear(1.4)) == pytest.approx(1.0)


def test_tonemap_is_identity_below_the_knee():

    rgb = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.8, 0.4, 0.2], [0.79, 0.79, 0.79]], np.float64
    )
    assert np.array_equal(color.tonemap(rgb), rgb)


def test_tonemap_preserves_hue():

    rgb = np.array([1.5, 0.8, 1.1])
    before = rgb[0] / rgb[2]

    ok = np.clip(color.tonemap(rgb), 0.0, 1.0)
    assert ok.max() <= 1.0
    assert ok[0] / ok[2] == pytest.approx(before, abs=1e-5)

    bad = np.clip(_wrong_tonemap_on_luma(rgb), 0.0, 1.0)
    assert bad[0] / bad[2] == pytest.approx(1.0, abs=0.01)
    assert abs(bad[0] / bad[2] - before) > 0.3


def test_tonemap_never_reaches_one():

    peaks = color.tonemap(np.array([[2.0, 0.0, 0.0], [10.0, 5.0, 1.0], [1e4, 0.0, 0.0]])).max(
        axis=-1
    )
    assert (peaks < 1.0).all()
    assert (peaks > color.KNEE).all()


def test_tonemap_is_continuous_at_the_knee():

    lo = color.tonemap(np.array([color.KNEE - 1e-6, 0.0, 0.0]))[0]
    hi = color.tonemap(np.array([color.KNEE + 1e-6, 0.0, 0.0]))[0]
    assert abs(hi - lo) < 1e-5


def test_exposure_one_saturates_far_less_than_two():

    rng = np.random.default_rng(0)
    hdr = rng.lognormal(mean=-1.2, sigma=0.9, size=(256, 256, 3))

    def saturated(exposure: float) -> int:
        return int((color.to_u8(color.finish(hdr, exposure)) >= 255).sum())

    one, two = saturated(1.0), saturated(2.0)
    assert color.EXPOSURE == 1.0
    assert two > 10 * one
    assert one < hdr.size * 0.0005


def test_flat_ambient_shading_matches_the_measured_reference():

    rgba = 0.95

    measured = {0.10: 24, 0.20: 48, 0.35: 85, 0.50: 121}
    tolerance = {0.10: 7, 0.20: 4, 0.35: 2, 0.50: 2}
    for ambient, reference in measured.items():
        got = int(color.shade_flat(color.srgb_to_linear(rgba), ambient)[0])
        assert abs(got - reference) <= tolerance[ambient]


def test_ambient_gain_is_what_the_reference_actually_does():

    assert pytest.approx(1.0) == color.AMBIENT_GAIN

    got = int(color.shade_flat(color.srgb_to_linear(0.95), 0.5)[0])
    assert abs(got - 121) <= 2
    assert abs(got - 242) > 100


def test_tonemap_only_bites_above_the_knee():

    dim = np.full(3, 0.30)
    assert np.allclose(color.finish(dim, tonemap_on=True), color.finish(dim, tonemap_on=False))

    bright = np.full(3, 1.60)
    with_knee = int(color.to_u8(color.finish(bright, tonemap_on=True))[0])
    without = int(color.to_u8(color.finish(bright, tonemap_on=False))[0])
    assert without == 255
    assert with_knee < 255
