"""Resolve concentric painted corners from actual native content insets."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NestedCorners:
    padding: tuple[float, float, float, float]
    radius: float
    inner_radius: float
    authored_radius: bool
    parent_constrained: bool


def nested_corners(style, scale, padding, inner_radius):
    """Keep a uniform offset; an authored outer radius constrains its child."""
    inset = max(0, *padding)
    authored = style.explicit("radius")
    constrained = getattr(style, "origins", {}).get("radius") == "parent_corner"
    outer = (
        max(0, style.get("radius", 0) * scale)
        if authored or constrained
        else max(0, inner_radius) + inset
    )
    return NestedCorners((inset,) * 4, outer, max(0, outer - inset), authored, constrained)
