"""Read composed hierarchy state without depending on viewer interaction code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import math3d
from .adapters.base import NodeType, SceneNode
from .types import CameraView

if TYPE_CHECKING:
    from .session import Session


def node_hierarchy_visible(session: Session, node: SceneNode) -> bool:
    """Resolve local and ancestor visibility, independent of renderer flags or occlusion."""
    while node is not None:
        if not node.visible:
            return False
        node = session.node(node.parent) if node.parent >= 0 else None
    return True


def node_geometry_indices(session: Session, node: SceneNode) -> np.ndarray:
    """Return render instances in a hierarchy subtree, including hidden geometry."""
    if session.source is None:
        return np.empty(0, np.intp)
    descendants = set()
    pending = [node.node_id]
    while pending:
        node_id = pending.pop()
        if node_id in descendants:
            continue
        descendants.add(node_id)
        child = session.node(node_id)
        if child is not None:
            pending.extend(child.children)
    return np.flatnonzero(np.isin(session.source.geom_node, list(descendants)))


def node_world_pose(session: Session, node: SceneNode) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one hierarchy node's current world-space position and orientation."""

    frame = session.frame
    if node.type is NodeType.MODEL:
        info = next((item for item in session.scene_models if item.model_id == node.model_id), None)
        if info is not None:
            return (
                np.asarray(info.position, np.float64).reshape(3),
                np.asarray(info.rotation, np.float64).reshape(3, 3),
            )
    if node.type is NodeType.LIGHT:
        lights = frame.lights or (session.source.lights if session.source is not None else None)
        if lights is not None and 0 <= node.light_index < len(lights.lights):
            light = lights.lights[node.light_index]
            return (
                np.asarray(light.position, np.float64).reshape(3),
                np.asarray(math3d.direction_basis(light.direction), np.float64),
            )
    if node.type is NodeType.CAMERA:
        view = camera_for_node(session, node)
        if view is not None:
            return (
                np.asarray(view.eye, np.float64).reshape(3),
                np.asarray(math3d.camera_rotation(view), np.float64),
            )
    if node.type is NodeType.SITE:
        i = int(node.site_index)
        pos = np.zeros(3, np.float64)
        mat = np.eye(3, dtype=np.float64)
        if frame.site_xpos is not None and 0 <= i < len(frame.site_xpos):
            pos = np.asarray(frame.site_xpos[i], np.float64).reshape(3)
        if frame.site_xmat is not None and 0 <= i < len(frame.site_xmat):
            mat = np.asarray(frame.site_xmat[i], np.float64).reshape(3, 3)
        return pos, mat
    if node.type is NodeType.GEOM:
        i = int(node.geom_index)
        pos = np.zeros(3, np.float64)
        mat = np.eye(3, dtype=np.float64)
        if frame.geom_xpos is not None and 0 <= i < len(frame.geom_xpos):
            pos = np.asarray(frame.geom_xpos[i], np.float64).reshape(3)
        if frame.geom_xmat is not None and 0 <= i < len(frame.geom_xmat):
            mat = np.asarray(frame.geom_xmat[i], np.float64).reshape(3, 3)
        return pos, mat
    i = int(node.body_index)
    pos = np.zeros(3, np.float64)
    mat = np.eye(3, dtype=np.float64)
    if frame.body_xpos is not None and 0 <= i < len(frame.body_xpos):
        pos = np.asarray(frame.body_xpos[i], np.float64).reshape(3)
    if frame.body_xmat is not None and 0 <= i < len(frame.body_xmat):
        mat = np.asarray(frame.body_xmat[i], np.float64).reshape(3, 3)
    return pos, mat


def camera_for_node(session: Session, node: SceneNode) -> CameraView | None:
    """Resolve a hierarchy camera against the current composed scene."""
    if not 0 <= node.camera_index < len(session.cameras):
        return None
    camera_id = session.cameras[node.camera_index].camera_id
    view = session.camera_view(camera_id)
    if view is not None:
        return view
    frame = session.frame
    if frame.cameras is not None and node.camera_index < len(frame.cameras):
        return frame.cameras[node.camera_index]
    return None
