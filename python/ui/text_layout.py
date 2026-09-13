"""Backend-neutral placement and elision for single-line UI labels."""

from __future__ import annotations

from mojive.ui.paint_protocol import Draw2D


def text_line_y(draw: Draw2D, center_y: float) -> float:
    """Align a text row to one cap-height reference, independent of word contents."""

    ink = draw.text_ink_bounds("H")
    center = (ink[1] + ink[3]) * 0.5 if ink else draw.text_size("H")[1] * 0.5
    return center_y - center


def fit_text(draw: Draw2D, value: str, max_width: float, *, middle: bool = False) -> str:
    """Fit a single line with an ellipsis instead of cutting a glyph at the edge."""
    text = " ".join(str(value).split())
    if draw.text_size(text)[0] <= max_width:
        return text

    def elided(length):
        if middle:
            return (
                f"{text[: (length + 1) // 2].rstrip()}…{text[len(text) - length // 2 :].lstrip()}"
            )
        return f"{text[:length].rstrip()}…"

    lo, hi = 0, len(text) - 1
    while lo < hi:
        count = (lo + hi + 1) // 2
        if draw.text_size(elided(count))[0] <= max_width:
            lo = count
        else:
            hi = count - 1
    return elided(lo) if lo else ""
