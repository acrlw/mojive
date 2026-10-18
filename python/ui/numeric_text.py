"""Numeric captions fit fixed fields without changing the edited value."""

import math
import re
from functools import lru_cache


@lru_cache(maxsize=128)
def _format_parts(format):
    return re.fullmatch(r"(.*?)(%[-+ #0]*\d*(?:\.(\d+))?([eEfFgGdiu]))(.*)", format)


def numeric_text(value, format):
    """Keep authored precision and units while normalizing displayed negative zero."""
    parts = _format_parts(format)
    if parts is None:
        return format % value
    prefix, token, _, _, suffix = parts.groups()
    number = token % value
    if number.lstrip().startswith("-") and float(number) == 0:
        number = token % 0
    return prefix + number + suffix


def fit_numeric_text(value, format, width, measure):
    """Prefer authored precision, then scientific notation inside the same box.

    This affects inactive ink only. Native text entry and stored values retain
    their own precision, and callers can expose the full value in a tooltip.
    """
    shown = numeric_text(value, format)
    if measure(shown)[0] <= width:
        return shown
    parts = _format_parts(format)
    if parts is None or parts[4] in "diu" or not math.isfinite(value):
        return shown
    prefix, _, precision, kind, suffix = parts.groups()
    if value == 0 and kind in "fF":
        for digits in range(max(0, int(precision or 6) - 1), -1, -1):
            candidate = prefix + f"{0:.{digits}f}" + suffix
            if measure(candidate)[0] <= width:
                return candidate
    for digits in range(min(9, int(precision or 6)), -1, -1):
        mantissa, exponent = f"{value:.{digits}e}".split("e")
        candidate = prefix + mantissa + "e" + str(int(exponent)) + suffix
        if measure(candidate)[0] <= width:
            return candidate
    return "…"


def numeric_edit_format(format):
    """Expose float32's round-trip precision while native text entry is active."""
    parts = _format_parts(format)
    if parts is None or parts[4] in "diu":
        return format
    return parts[1] + "%.9g" + parts[5]
