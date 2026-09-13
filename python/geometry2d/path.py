"""Immutable authored paths and their short-lived mutable builder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .style import finite_tuple


class PathVerb(IntEnum):
    MOVE = 0
    LINE = 1
    QUADRATIC = 2
    CUBIC = 3
    ARC = 4
    CLOSE = 5


_COORDINATES = {
    PathVerb.MOVE: 2,
    PathVerb.LINE: 2,
    PathVerb.QUADRATIC: 4,
    PathVerb.CUBIC: 6,
    PathVerb.ARC: 7,
    PathVerb.CLOSE: 0,
}


@dataclass(frozen=True, slots=True)
class Path2D:
    """Local-coordinate path with explicit nonzero or evenodd fill semantics.

    Open subpaths are implicitly closed for filling and remain open for
    stroking. After close, a new subpath must start with move. Arc coordinates
    are center X/Y, radius X/Y, rotation, start angle, and signed sweep, in
    radians. An arc connects the current point to its first point with a line.
    """

    commands: tuple[tuple[int, tuple[float, ...]], ...] = ()
    fill_rule: str = "nonzero"
    control_bounds: tuple[float, float, float, float] | None = field(
        init=False, compare=False, repr=False
    )

    def __post_init__(self):
        if self.fill_rule not in ("nonzero", "evenodd"):
            raise ValueError("Fill rule must be nonzero or evenodd")
        validated = []
        active = False
        for verb, coordinates in self.commands:
            verb = PathVerb(verb)
            values = finite_tuple(coordinates, _COORDINATES[verb], verb.name)
            if verb == PathVerb.MOVE:
                active = True
            elif not active:
                raise ValueError("A subpath must start with move")
            if verb == PathVerb.ARC and (values[2] < 0 or values[3] < 0):
                raise ValueError("Arc radii must be nonnegative")
            if verb == PathVerb.CLOSE:
                active = False
            validated.append((int(verb), values))
        object.__setattr__(self, "commands", tuple(validated))
        bounds = None
        for verb, values in validated:
            if verb == PathVerb.ARC:
                cx, cy, rx, ry, rotation, _, _ = values
                c, s = math.cos(rotation), math.sin(rotation)
                ex, ey = math.hypot(rx * c, ry * s), math.hypot(rx * s, ry * c)
                points = ((cx - ex, cy - ey), (cx + ex, cy + ey))
            else:
                points = zip(values[::2], values[1::2], strict=True)
            for x, y in points:
                bounds = (
                    (x, y, x, y)
                    if bounds is None
                    else (
                        min(x, bounds[0]),
                        min(y, bounds[1]),
                        max(x, bounds[2]),
                        max(y, bounds[3]),
                    )
                )
        object.__setattr__(self, "control_bounds", bounds)

    def packed(self) -> np.ndarray:
        """Owned contiguous float64 [N, 8] command buffer for one native call."""
        values = np.zeros((len(self.commands), 8), dtype=np.float64)
        for row, (verb, coordinates) in zip(values, self.commands, strict=True):
            row[0] = verb
            row[1 : len(coordinates) + 1] = coordinates
        return values


class PathBuilder2D:
    """Fluent builder; finish returns a snapshot and does not reset the builder."""

    def __init__(self, *, fill_rule: str = "nonzero"):
        self._commands = []
        self._fill_rule = Path2D(fill_rule=fill_rule).fill_rule
        self._active = False

    def _append(self, verb, coordinates):
        values = finite_tuple(coordinates, _COORDINATES[verb], verb.name)
        if verb != PathVerb.MOVE and not self._active:
            raise ValueError("A subpath must start with move")
        if verb == PathVerb.ARC and (values[2] < 0 or values[3] < 0):
            raise ValueError("Arc radii must be nonnegative")
        self._commands.append((int(verb), values))
        self._active = verb != PathVerb.CLOSE
        return self

    def move_to(self, x: float, y: float) -> PathBuilder2D:
        return self._append(PathVerb.MOVE, (x, y))

    def line_to(self, x: float, y: float) -> PathBuilder2D:
        return self._append(PathVerb.LINE, (x, y))

    def quadratic_to(self, cx: float, cy: float, x: float, y: float) -> PathBuilder2D:
        return self._append(PathVerb.QUADRATIC, (cx, cy, x, y))

    def cubic_to(
        self, cx1: float, cy1: float, cx2: float, cy2: float, x: float, y: float
    ) -> PathBuilder2D:
        return self._append(PathVerb.CUBIC, (cx1, cy1, cx2, cy2, x, y))

    def arc(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        start: float,
        sweep: float,
        *,
        rotation: float = 0.0,
    ) -> PathBuilder2D:
        return self._append(PathVerb.ARC, (cx, cy, rx, ry, rotation, start, sweep))

    def close(self) -> PathBuilder2D:
        return self._append(PathVerb.CLOSE, ())

    def finish(self) -> Path2D:
        return Path2D(tuple(self._commands), self._fill_rule)
