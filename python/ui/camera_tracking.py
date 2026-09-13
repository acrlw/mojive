"""World-space camera following, independent of simulation clocks and target rotation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..adapters.base import NodeType
from ..config import CameraTrackingConfig

if TYPE_CHECKING:
    from ..adapters.base import SceneNode
    from ..session import Session
    from .camera import OrbitCamera


_TRACKABLE_TYPES = frozenset(
    (NodeType.ROBOT, NodeType.LINK, NodeType.JOINT, NodeType.GEOM, NodeType.SITE)
)


def can_track_node(node: SceneNode | None) -> bool:
    """Return whether a hierarchy entry has a rendered world-position stream."""
    return node is not None and node.type in _TRACKABLE_TYPES


def tracking_position(session: Session, node: SceneNode) -> np.ndarray | None:
    """Read the displayed pose, avoiding physics-thread access and pose allocations."""
    frame = session.frame
    if node.type is NodeType.GEOM:
        positions, index = frame.geom_xpos, node.geom_index
    elif node.type is NodeType.SITE:
        positions, index = frame.site_xpos, node.site_index
    else:
        positions, index = frame.body_xpos, node.body_index
    if positions is None or not 0 <= index < len(positions):
        return None
    position = positions[index]
    if math.isfinite(position[0]) and math.isfinite(position[1]) and math.isfinite(position[2]):
        return position
    return None


class CameraTracker:
    """Translate an orbit camera toward one target with exponential position damping."""

    def __init__(self, config: CameraTrackingConfig = CameraTrackingConfig()) -> None:
        self.config = config
        self.node_id: int | None = None
        self._offset = np.zeros(3, np.float64)
        self._last_pivot = np.zeros(3, np.float64)
        self._delta = np.zeros(3, np.float64)

    def start(self, node_id: int, camera: OrbitCamera) -> None:
        """Acquire a target smoothly from the current view, with no initial jump."""
        self.node_id = node_id
        self._offset.fill(0)
        np.copyto(self._last_pivot, camera.pivot)
        camera._stop_anim()

    def stop(self) -> None:
        """Leave the current camera view in place and release the target."""
        self.node_id = None

    def advance(self, camera: OrbitCamera, position: np.ndarray, dt: float) -> bool:
        """Follow the latest displayed target without changing distance or orientation."""
        if self.node_id is None:
            return False
        # Preserve intentional panning as a composition offset. Orbit and dolly
        # leave the pivot unchanged and therefore do not affect this offset.
        np.subtract(camera.pivot, self._last_pivot, out=self._delta)
        self._offset += self._delta
        np.add(position, self._offset, out=self._delta)
        self._delta -= camera.pivot
        if self.config.axes == "xy":
            self._delta[2] = 0
        smoothing = self.config.smoothing
        alpha = 1.0 if smoothing == 0 else -math.expm1(-math.log(2) * max(0.0, dt) / smoothing)
        self._delta *= alpha
        changed = bool(np.dot(self._delta, self._delta) > 1e-18)
        if changed:
            camera.translate(self._delta)
        np.copyto(self._last_pivot, camera.pivot)
        return changed
