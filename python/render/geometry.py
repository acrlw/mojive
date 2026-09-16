"""Geometry presentation presets over independent renderer flags."""

from mojive.types import GeometryView as GeometryView

from .backend import RenderBackend, RenderFlag

_FLAGS = (RenderFlag.VISUAL_GEOMETRY, RenderFlag.COLLISION_GEOMETRY)
_VIEWS = tuple(GeometryView)


def geometry_view(backend: RenderBackend) -> GeometryView:
    return _VIEWS[int(backend.get_flag(_FLAGS[0])) + 2 * int(backend.get_flag(_FLAGS[1]))]


def set_geometry_view(backend: RenderBackend, view: GeometryView | str) -> bool:
    """Select a preset without changing physics, groups, or other render flags."""
    view = GeometryView(view)
    if not all(backend.caps.supports(flag) for flag in _FLAGS):
        return False
    return backend.set_geometry_view(view)


def set_geometry_flags(flags: dict, view: GeometryView | str) -> bool:
    """Update both preset bits before rebuilding any render scene."""
    bits = _VIEWS.index(GeometryView(view))
    changed = False
    for i, flag in enumerate(_FLAGS):
        value = bool(bits & (1 << i))
        changed |= flags.get(flag, False) != value
        flags[flag] = value
    return changed
