"""Immutable model composition and preview state records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..base import (
    PhysicsState,
)


@dataclass(frozen=True)
class _AttachedModel:
    model_id: int
    name: str
    path: Path
    prefix: str
    position: np.ndarray
    rotation: np.ndarray
    spec: object
    edited: bool = False


@dataclass
class _ModelTransformPreview:
    """Render-only placement while an attached model gizmo is active."""

    model_id: int
    previous_position: np.ndarray
    previous_rotation: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    delta_rotation: np.ndarray
    body_indices: np.ndarray
    geom_indices: np.ndarray
    site_indices: np.ndarray
    joint_indices: np.ndarray
    light_mask: np.ndarray
    camera_mask: np.ndarray
    point_input: np.ndarray
    point_output: np.ndarray
    matrix_input: np.ndarray
    matrix_output: np.ndarray


@dataclass(frozen=True)
class _NamedModelState:
    joints: dict[str, tuple[np.ndarray, np.ndarray]]
    actuators: dict[str, tuple[np.ndarray, np.ndarray]]
    mocap: dict[str, tuple[np.ndarray, np.ndarray]]
    equality: dict[str, bool]
    time: float


@dataclass(frozen=True)
class _ModelComponentEntry:
    component_id: int
    subtype: str
    name: str
    signature: str


@dataclass(frozen=True)
class _CompositionEditState:
    models: tuple[_AttachedModel, ...]
    physics: PhysicsState
    root_spec: object
    root_edited: bool
    geometry_object_ids: dict[tuple[int, str], int]
    next_geometry_object_id: int
    component_entries: dict[tuple[int, str], tuple[_ModelComponentEntry, ...]]
