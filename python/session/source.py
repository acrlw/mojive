"""Session: source."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    ENVIRONMENT_OBJECT_ID,
    ActuatorInfo,
    FrameNeeds,
    JointInfo,
    NodeType,
    PhysicsOption,
    PhysicsOptions,
    ReplayControl,
    ReplayInfo,
    RolloutSync,
    RolloutSyncInfo,
    SceneNode,
    WorldSelection,
    WorldSelectionInfo,
)
from mojive.commands import Query
from mojive.scene.bounds import SceneBounds
from mojive.types import (
    Bounds,
    CameraView,
)

from .state import (
    _apply_geometry_color_overrides,
)


class _Source:
    """Private source methods of Session; state belongs to its owner."""

    @property
    def replay_info(self) -> ReplayInfo | None:
        """Read the local replay clock, independent of the physics clock."""
        if not self._adapter.caps.supports("replay.control"):
            return None
        return cast(ReplayControl, self._adapter).replay_info()

    @property
    def rollout_sync_info(self) -> RolloutSyncInfo | None:
        """Read manual synchronization status without network work."""
        if not self._adapter.caps.supports("replay.sync"):
            return None
        return cast(RolloutSync, self._adapter).rollout_sync_info()

    @property
    def world_selection(self) -> WorldSelectionInfo | None:
        """Return preview world identities when the adapter supports selection."""
        if not self._adapter.caps.supports("world.selection"):
            return None
        return cast(WorldSelection, self._adapter).world_selection()

    @property
    def physics_options(self) -> tuple[PhysicsOption, ...]:
        """World-wide simulation settings, or an empty tuple when unsupported."""
        if not self._adapter.caps.supports("physics.options"):
            return ()
        return cast(PhysicsOptions, self._adapter).physics_options()

    def query(self, q: Query):
        """Evaluate a read-only pick, node lookup, or bounds query."""
        if isinstance(q, cmd.Pick):
            if not self._adapter.caps.raycast:
                return (0, float("inf"))
            return self._adapter.raycast(q.origin, q.direction)
        if isinstance(q, cmd.NodeAt):
            return self._by_object_id.get(int(q.object_id))
        if isinstance(q, cmd.Bounds):
            return self.bounds()
        raise TypeError(f"Unknown query: {type(q).__name__}")

    def bounds(self) -> Bounds:
        """Return world-space minimum/maximum bounds for finite scene geometry."""
        if self.source is not None:
            if self._scene_bounds is None:
                self._scene_bounds = SceneBounds(self.source, self._mesh_bounds_cache)
            bounds = self._scene_bounds.world(self.frame)
            if bounds is not None:
                return bounds
        return Bounds(np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32))

    def camera_hint(self) -> CameraView | None:
        """Return the adapter camera suggested for initial framing."""
        return self._adapter.camera_hint()

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Return a scene override or adapter camera by stable ID."""
        i = int(camera_id)
        if i in self._scene_overrides.cameras:
            return self._scene_overrides.cameras[i]
        return self._adapter.camera_view(i) if self._adapter.caps.model_cameras else None

    def _retain_scene_override(self, writeback: bool) -> bool:
        caps = self._adapter.caps
        return not writeback or caps.external_clock or caps.model_composition

    def visual_groups(self):
        """Return numbered visual group states exposed by the adapter."""
        return self._adapter.visual_groups() if self._adapter.caps.visual_groups else ()

    def _refresh_structure(self, *, installed: bool = False) -> None:
        """Rebuild session structure from the adapter.

        A rebuild batch installs one compiled model at its end, so intermediate refreshes
        stay cheap. ``installed`` marks the point where a created element exists and the
        commands that address it need its node ID.
        """

        if getattr(self, "_applying_model_edits", False) and not installed:
            return
        world_visibility = {
            (node.type, node.object_id): node.visible
            for node in self._nodes
            if self._adapter.caps.supports("world.selection")
            and (node.object_id or node.type is NodeType.WORLD)
        }
        selected_before = self._by_node_id.get(self._selected_node_id)
        self._mesh_bounds_cache.clear()
        self._scene_bounds = None
        self._source = self._adapter.scene_source()
        if self._scene_overrides.environment is not None:
            self._source.lights = self._source.lights.with_environment(
                self._scene_overrides.environment
            )
        for material_index, material in self._scene_overrides.materials.items():
            if material_index < len(self._source.materials):
                self._source.materials[material_index] = material
        if self._scene_overrides.geometry_colors:
            # Hierarchy indices can shift when model geometry is inserted before scene
            # objects. Resolve retained colors by object identity across rebuilds and Undo.
            by_object = dict(zip(self._source.geom_object_id, self._source.geom_node, strict=True))
            by_name = {(n.model_id, n.type, n.name): n.node_id for n in self._source.nodes}
            colors, targets = {}, {}
            for node_id, color in self._scene_overrides.geometry_colors.items():
                target = self._scene_overrides.geometry_color_targets.get(node_id)
                if target is not None:
                    node_id = by_object.get(target[0]) if target[0] else by_name.get(target[1:])
                if node_id is not None:
                    colors[node_id] = color
                    if target is not None:
                        targets[node_id] = target
            self._scene_overrides.geometry_colors = colors
            self._scene_overrides.geometry_color_targets = targets
            _apply_geometry_color_overrides(self._source, colors)
        self._nodes = [
            replace(node, children=list(node.children)) for node in self._adapter.nodes()
        ]
        if self._scene_overrides.lights:
            light_nodes = {
                node.object_id: node
                for node in self._nodes
                if node.object_id > 0 and node.light_index >= 0
            }
            lights = list(self._source.lights.lights)
            for override in self._scene_overrides.lights.values():
                node = light_nodes.get(override.object_id) if override.object_id > 0 else None
                index = node.light_index if node is not None else override.light_index
                if 0 <= index < len(lights):
                    lights[index] = override.light
            self._source.lights = replace(self._source.lights, lights=tuple(lights))
        if not any(node.type is NodeType.ENVIRONMENT for node in self._nodes):
            parent = next(
                (node for node in self._nodes if node.type is NodeType.WORLD and node.parent < 0),
                None,
            )
            node_id = max((node.node_id for node in self._nodes), default=-1) + 1
            environment = SceneNode(
                node_id,
                "environment",
                NodeType.ENVIRONMENT,
                parent=parent.node_id if parent is not None else -1,
                object_id=ENVIRONMENT_OBJECT_ID,
            )
            self._nodes.append(environment)
            if parent is not None:
                parent.children.append(node_id)
        for node in self._nodes:
            if 0 <= node.light_index < len(self._source.lights.lights):
                node.visible = self._source.lights.lights[node.light_index].active
        self._refresh_joint_metadata()
        self._actuators = self._adapter.actuators()
        actuators_by_joint: dict[int, list[ActuatorInfo]] = {}
        for actuator in self._actuators:
            actuators_by_joint.setdefault(int(actuator.joint), []).append(actuator)
        self._actuators_by_joint = {
            joint: tuple(actuators) for joint, actuators in actuators_by_joint.items()
        }
        self._cameras = self._adapter.cameras() if self._adapter.caps.model_cameras else []
        self._camera_slot_by_id = {
            camera.camera_id: slot for slot, camera in enumerate(self._cameras)
        }
        if self._scene_overrides.cameras:
            cameras = list(self._source.cameras)
            for camera_id, camera in self._scene_overrides.cameras.items():
                slot = self._camera_slot(camera_id)
                if 0 <= slot < len(cameras):
                    cameras[slot] = camera
            self._source.cameras = tuple(cameras)
        self._model_keyframes.refresh(self._adapter if self._adapter.caps.keyframes else None)
        self._sensor_infos = self._adapter.sensors() if self._adapter.caps.sensors else []
        self._equality_constraints = (
            self._adapter.equality_constraints() if self._adapter.caps.equality_constraints else []
        )
        if self._active_keyframe != -1 and self._keyframe_slot(self._active_keyframe) < 0:
            self._active_keyframe = -1
        self._by_node_id = {n.node_id: n for n in self._nodes}
        self._by_object_id = {n.object_id: n for n in self._nodes if n.object_id}
        self._unlocked_entity_gizmos.intersection_update(self._by_object_id)
        if selected_before is not None and selected_before.source_editable:
            self._restore_model_selection(
                selected_before.model_id,
                selected_before.type,
                selected_before.source_name or selected_before.name,
            )
        elif self._selected:
            selected = self._by_object_id.get(self._selected)
            if selected is None:
                self._selected = 0
                self._selected_node_id = -1
            else:
                self._selected_node_id = selected.node_id
        elif (selected := self.node(self._selected_node_id)) is None or selected.object_id:
            self._selected_node_id = -1
        if (self._state_takes or self._frame_history) and self._adapter.caps.state_snapshots:
            state = self._adapter.capture_state()
            signature = None if state is None else self._physics_state_signature(state)
            if any(
                take.frames and signature != take.signature for take in self._state_takes.values()
            ):
                self._clear_state_takes()
                self._publish_message(
                    "Cleared recorded takes after the simulation state layout changed",
                    level="warning",
                    duration=5.0,
                )
            if self._frame_history and signature != self._frame_history_signature:
                self._clear_frame_history()
        if self._scene_snapshots and any(
            snapshot.structure_revision != self._adapter.structure_revision
            for snapshot in self._scene_snapshots.values()
        ):
            self._clear_scene_snapshots()
        self._adapter_revision = self._adapter.structure_revision
        for node in (*self._nodes, *self._source.nodes):
            key = (node.type, node.object_id)
            if key in world_visibility:
                node.visible = world_visibility[key]
        self._structure_generation += 1
        self._frame = self._adapter.frame(FrameNeeds())
        self._sync_equality_state()
        self._compose_lights()
        self._compose_cameras()

    def _restore_model_selection(self, model_id: int, node_type: NodeType, name: str) -> None:
        """Rebind an editable model element without reusing compiled body/node indices."""
        body_types = (NodeType.LINK, NodeType.ROBOT)
        matches = [
            node
            for node in self._nodes
            if node.model_id == model_id
            and (node.source_name or node.name) == name
            and (node.type is node_type or (node.type in body_types and node_type in body_types))
        ]
        selected = matches[0] if len(matches) == 1 else None
        self._selected = selected.object_id if selected is not None else 0
        self._selected_node_id = selected.node_id if selected is not None else -1

    def refresh_control_metadata(self) -> None:
        """Refresh parameters in an unchanged joint/actuator layout.

        Panel row caches retain metadata objects until a structural refresh, so
        update their live parameters in place without resetting UI state.
        """
        joints, actuators = self._adapter.joints(), self._adapter.actuators()
        if [j.joint_id for j in joints] != [j.joint_id for j in self._joints] or [
            a.actuator_id for a in actuators
        ] != [a.actuator_id for a in self._actuators]:
            raise ValueError("Control layout changed; refresh scene structure first")
        for current, updated in zip(self._joints, joints, strict=True):
            current.limited = updated.limited
            current.range = updated.range
            current.axis = updated.axis
            current.damping = updated.damping
            current.stiffness = updated.stiffness
        for current, updated in zip(self._actuators, actuators, strict=True):
            current.ctrl_range = updated.ctrl_range
            current.ctrl_limited = updated.ctrl_limited
            current.gain = updated.gain

    def _refresh_joint_metadata(self) -> None:
        """Refresh joint lookup tables without rebuilding stable scene geometry."""
        self._joints = self._adapter.joints()
        joints_by_body: dict[int, list[JointInfo]] = {}
        for joint in self._joints:
            joints_by_body.setdefault(int(joint.body), []).append(joint)
        self._joints_by_body = {body: tuple(joints) for body, joints in joints_by_body.items()}

    def _compose_lights(self) -> None:
        """Combine scene light settings with backend-driven transforms.

        A physics backend may move a body-attached light and publish its world
        position/direction in ``SceneFrame``.  Color, intensity, range, fog and
        every other render setting still come from the Mojive scene.
        """
        if self._source is None:
            return
        configured = self._source.lights
        driven = self._frame.lights
        if driven is None or driven is configured or len(driven.lights) != len(configured.lights):
            self._frame.lights = configured
            return
        lights = tuple(
            light
            if light is dynamic
            else replace(light, position=dynamic.position, direction=dynamic.direction)
            for light, dynamic in zip(configured.lights, driven.lights, strict=True)
        )
        self._frame.lights = replace(configured, lights=lights)

    def _sync_equality_state(self) -> None:
        values = self._frame.equality_enabled
        if values is None or len(values) != len(self._equality_constraints):
            return
        for i, enabled in enumerate(values):
            if self._equality_constraints[i].enabled != bool(enabled):
                self._equality_constraints[i] = replace(
                    self._equality_constraints[i], enabled=bool(enabled)
                )

    def _compose_cameras(self) -> None:
        if self._source is None:
            return
        driven = self._frame.cameras
        if not self._scene_overrides.cameras:
            if driven is None or len(driven) != len(self._source.cameras):
                self._frame.cameras = self._source.cameras
            return
        cameras = list(
            driven
            if driven is not None and len(driven) == len(self._source.cameras)
            else self._source.cameras
        )
        for camera_id, camera in self._scene_overrides.cameras.items():
            slot = self._camera_slot(camera_id)
            if 0 <= slot < len(cameras):
                cameras[slot] = camera
        self._frame.cameras = tuple(cameras)

    def _camera_slot(self, camera_id: int) -> int:
        return self._camera_slot_by_id.get(int(camera_id), -1)

    def _keyframe_slot(self, keyframe_id: int) -> int:
        return self._model_keyframes.slot(keyframe_id)

    def _equality_slot(self, constraint_id: int) -> int:
        return next(
            (
                slot
                for slot, constraint in enumerate(self._equality_constraints)
                if constraint.constraint_id == constraint_id
            ),
            -1,
        )
