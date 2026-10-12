"""Control geometry shared by panel presenters and native input behavior."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RailStyle:
    """Logical rail geometry and the attached numeric/unit field treatment."""

    thumb_size: float = 10
    track_size: float = 3
    entry_width: float = 68
    unit_width: float = 34
    minimum_track_width: float = 0
    unit_accent: bool = True
    degree_unit: str = "deg"
