"""Numeric display policy does not mutate the stored or entered value."""

import pytest

from mojive.ui.numeric_text import fit_numeric_text, numeric_edit_format, numeric_text


@pytest.mark.parametrize(
    "value,format,expected",
    (
        (12.34, "%.5f kg", "12.34000 kg"),
        (-0.0, "mass: %.5f", "mass: 0.00000"),
        (-0.00001, "%.3f", "0.000"),
        (-0.0, "%+8.3f", "  +0.000"),
        (-3.0, "%.3f", "-3.000"),
        (1.0, "%.3f", "1.000"),
        (255.0, "%.0f", "255"),
        (0.00288568, "%.6g", "0.00288568"),
        (10, "%d px", "10 px"),
    ),
)
def test_numeric_caption_keeps_units_and_normalizes_zero(value, format, expected):
    assert numeric_text(value, format) == expected


@pytest.mark.parametrize("value", (1234567890123.456, -1234567890123.456, 0.000123456789))
def test_numeric_caption_compacts_only_when_the_fixed_box_requires_it(value):
    def measure(text):
        return len(text), 1

    full = numeric_text(value, "%.9f")
    assert fit_numeric_text(value, "%.9f", 100, measure) == full
    compact = fit_numeric_text(value, "%.9f", 9, measure)
    assert len(compact) <= 9
    assert float(compact) == pytest.approx(value, rel=0.005)


@pytest.mark.parametrize("value", (1e308, -1e308))
def test_scientific_fallback_keeps_complete_exponents_units_and_field_budget(value):
    shown = fit_numeric_text(value, "mass: %.3f kg", 20, lambda text: (len(text), 1))
    assert shown.startswith("mass: ") and shown.endswith(" kg")
    number = shown[6:-3]
    assert "e308" in number and len(shown) <= 20
    assert float(number) == pytest.approx(value)
    assert fit_numeric_text(value, "%.3f", 2, lambda text: (len(text), 1)) == "…"


def test_numeric_text_entry_uses_roundtrip_precision_and_retains_units():
    assert numeric_edit_format("%.3f m") == "%.9g m"
    assert numeric_edit_format("mass: %.2f") == "mass: %.9g"
    assert numeric_edit_format("%d") == "%d"


def test_exact_zero_remains_readable_when_its_authored_precision_cannot_fit():
    def measure(text):
        return len(text), 1

    assert fit_numeric_text(-0.0, "%.3f", 5, measure) == "0.000"
    assert fit_numeric_text(0.0, "%.3f", 3, measure) == "0.0"
    assert fit_numeric_text(0.0, "%.3f", 1, measure) == "0"
