"""Compatibility imports; new code uses :mod:`mojive.adapters.mujoco`."""

from .mujoco import DEFAULT_GEOM_GROUPS, VISUAL_GROUP_CATEGORIES, MuJoCoAdapter

__all__ = ["DEFAULT_GEOM_GROUPS", "VISUAL_GROUP_CATEGORIES", "MuJoCoAdapter"]
