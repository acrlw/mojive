"""MuJoCo scene adapter with private modules grouped by responsibility."""

from .adapter import MuJoCoAdapter
from .constants import DEFAULT_GEOM_GROUPS, VISUAL_GROUP_CATEGORIES

__all__ = ["DEFAULT_GEOM_GROUPS", "VISUAL_GROUP_CATEGORIES", "MuJoCoAdapter"]
