"""Session commands: transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    NodeType,
)
from mojive.commands import CommandResult
from mojive.scene.geometry import scale_vector

if TYPE_CHECKING:
    from .. import Session


def select(self: Session, c: cmd.Select) -> CommandResult:
    node = self.node_by_object_id(c.object_id)
    if c.object_id and node is None:
        return CommandResult.bad(f"Unknown object_id={c.object_id}")
    self._selected = int(c.object_id)
    self._selected_node_id = node.node_id if node is not None else -1
    self._selection_revision += 1
    return CommandResult.good(node.name if node else "Selection cleared")


def select_node(self: Session, c: cmd.SelectNode) -> CommandResult:
    node = self.node(self._command_node_id(c, c.node_key))
    if node is None:
        return CommandResult.bad(self._command_node_error(c, c.node_key, "node"))
    self._selected_node_id = node.node_id
    self._selected = int(node.object_id)
    self._selection_revision += 1
    return CommandResult.good(node.name)


def set_visible(self: Session, c: cmd.SetVisible) -> CommandResult:
    node = self.node(c.node_id)
    if node is None:
        return CommandResult.bad(f"Unknown node_id={c.node_id}")
    node.visible = c.visible
    if self._source is not None:
        source_node = next((item for item in self._source.nodes if item.node_id == c.node_id), None)
        if source_node is not None:
            source_node.visible = c.visible
    self._structure_generation += 1
    return CommandResult.good("")


def set_visual_group(self: Session, c: cmd.SetVisualGroup) -> CommandResult:
    caps = self._adapter.caps
    if not caps.visual_groups:
        return CommandResult.bad(f"{caps.name} does not expose visual groups")
    ok = self._adapter.set_visual_group(c.category, c.group, c.visible)
    return (
        CommandResult.good("")
        if ok
        else CommandResult.bad(f"visual group {c.category}:{c.group} is unavailable")
    )


def set_pose(self: Session, c: cmd.SetPose) -> CommandResult:
    caps = self._adapter.caps
    if not caps.write_pose:
        return CommandResult.bad(f"{caps.name} does not support pose editing")
    if not self._paused:
        return CommandResult.bad("physics is running; pause to move things")
    node = self.node(c.node_id)
    if node is None:
        return CommandResult.bad(f"Unknown node_id={c.node_id}")
    if not node.posable:
        message = (
            "This link is joint-driven; use its viewport gizmo or the Joints panel"
            if node.type in (NodeType.LINK, NodeType.ROBOT)
            else "This entity has no editable transform"
        )
        return CommandResult.bad(message)
    ok = self._adapter.set_pose(c.node_id, c.position, c.rotation)
    return CommandResult.good("") if ok else CommandResult.bad("Pose update failed")


def set_scale(self: Session, c: cmd.SetScale) -> CommandResult:
    if not self._paused:
        return CommandResult.bad("Pause simulation before scaling geometry")
    target = self.scale_target(c.node_id)
    if target is None:
        return CommandResult.bad("This entity does not support local geometry scaling")
    try:
        scale = scale_vector(c.scale)
    except (TypeError, ValueError) as error:
        return CommandResult.bad(str(error))
    if not self._adapter.set_scale(target.node_id, scale):
        return CommandResult.bad("Geometry scale update failed")
    self._refresh_structure()
    return CommandResult.good("")


def set_qpos(self: Session, c: cmd.SetQpos) -> CommandResult:
    caps = self._adapter.caps
    if not caps.write_qpos:
        return CommandResult.bad(f"{caps.name} does not support joint editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing joints")
    ok = self._adapter.set_qpos(c.index, c.value)
    if ok:
        self._frame_history_dirty = True
    return CommandResult.good("") if ok else CommandResult.bad(f"Joint {c.index} update failed")


def set_qpos_batch(self: Session, c: cmd.SetQposBatch) -> CommandResult:
    caps = self._adapter.caps
    if not caps.write_qpos:
        return CommandResult.bad(f"{caps.name} does not support joint editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing joints")
    raw_indices = np.asarray(c.indices).reshape(-1)
    if not np.issubdtype(raw_indices.dtype, np.integer):
        return CommandResult.bad("Joint batch indices must be integers")
    indices = raw_indices.astype(np.intp, copy=False)
    values = np.asarray(c.values, np.float64).reshape(-1)
    if not len(indices) or len(indices) != len(values):
        return CommandResult.bad("Joint batch indices and values must have equal lengths")
    if len(np.unique(indices)) != len(indices):
        return CommandResult.bad("Joint batch indices must be unique")
    if not np.all(np.isfinite(values)):
        return CommandResult.bad("Joint batch values must be finite")
    ok = self._adapter.set_qpos_batch(indices, values)
    if ok:
        self._frame_history_dirty = True
    return CommandResult.good("") if ok else CommandResult.bad("Joint batch update failed")
