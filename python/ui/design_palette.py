"""Stable source inks for painters, resolved per workspace by the design runtime."""

from mojive.design.themes import PRESETS

PALETTE = PRESETS["dark"]


def rgba8(role: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = PALETTE[role].removeprefix("#")
    opacity = int(value[6:8], 16) if len(value) == 8 else 255
    return (
        *(int(value[index : index + 2], 16) for index in (0, 2, 4)),
        round(opacity * alpha / 255),
    )


def rgba(role: str) -> tuple[float, float, float, float]:
    return tuple(channel / 255 for channel in rgba8(role))
