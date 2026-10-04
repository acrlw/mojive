"""Session: source."""

from __future__ import annotations

from contextlib import nullcontext
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
from mojive.log import get_logger
from mojive.scene.bounds import SceneBounds
from mojive.types import (
    Bounds,
    CameraView,
)

from .state import (
    _apply_geometry_color_overrides,
    _MaterialTarget,
)

log = get_logger("session")


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

    def _material_target(self, index: int, source=None, *, revision=None) -> _MaterialTarget:
        source = self._source if source is None else source
        return _MaterialTarget(
            self._adapter_revision if revision is None else revision,
            source.materials[index],
            frozenset(
                int(object_id)
                for object_id, material_index in zip(
                    source.geom_object_id, source.geom_material, strict=True
                )
                if material_index == index and object_id
            ),
        )

    def _rebind_material_overrides(self) -> None:
        """Rebind indices only after restoring their exact document checkpoint."""
        source = self._adapter.scene_source()
        self._scene_overrides.material_targets = {
            index: self._material_target(index, source, revision=self._adapter.structure_revision)
            for index in self._scene_overrides.materials
            if 0 <= index < len(source.materials)
        }

    def _apply_material_overrides(self) -> None:
        overrides = self._scene_overrides
        revision = self._adapter.structure_revision
        materials, targets = {}, {}
        invalid = False
        for index, material in overrides.materials.items():
            target = overrides.material_targets.get(index)
            if target is not None and target.revision != revision:
                # Object identity is proof for retained in-process materials;
                # equal values or names cannot identify a replaced resource.
                candidates = [
                    candidate
                    for candidate, value in enumerate(self._source.materials)
                    if value is target.material
                    and self._material_target(candidate).object_ids & target.object_ids
                ]
                index = candidates[0] if len(candidates) == 1 else -1
            if target is None or not 0 <= index < len(self._source.materials):
                invalid = True
                continue
            targets[index] = self._material_target(index, revision=revision)
            materials[index] = material
        overrides.materials, overrides.material_targets = materials, targets
        for index, material in materials.items():
            self._source.materials[index] = material
        if invalid:
            message = (
                "Cleared viewer material overrides after the scene structure changed; "
                "their material identities could not be verified"
            )
            # A command success message may replace the transient UI warning;
            # retain the invalidation evidence in the application's Output log.
            log.warning(message)
            self._publish_message(message, level="warning", duration=5.0)

    def visual_groups(self):
        """Return numbered visual group states exposed by the adapter."""
        return self._adapter.visual_groups() if self._adapter.caps.visual_groups else ()

    def _scene_read(self):
        # Preserve structural adapters published before the optional read boundary.
        return getattr(self._adapter, "scene_read", nullcontext)()

    def sync_structure(self) -> bool:
        """Install external structure changes without advancing simulation or playback."""
        with self._scene_read():
            prepare = getattr(self._adapter, "prepare_frame", None)
            if prepare is not None:
                prepare(FrameNeeds.none())
            if self._adapter.structure_revision == self._adapter_revision:
                return False
            self._refresh_structure()
            return True

    def _refresh_structure(self, *, installed: bool = False) -> None:
        """Rebuild session structure from the adapter.

        A rebuild batch installs one compiled model at its end, so intermediate refreshes
        stay cheap. ``installed`` marks the point where a created element exists and the
        commands that address it need its node ID.
        """

        if getattr(self, "_applying_model_edits", False) and not installed:
            return
        with self._scene_read():
            self._install_structure()

    def _install_structure(self) -> None:
        """Rebind local state before publishing; the caller holds ``scene_read``."""
        world_visibility = {
            (node.type, node.object_id): node.visible
            for node in self._nodes
            if self._adapter.caps.supports("world.selection")
            and (node.object_id or node.type is NodeType.WORLD)
        }
        selected_before = self._by_node_id.get(self._selected_node_id)

        self._mesh_bounds_cache.clear()
        self._scene_bounds = None
        source = self._adapter.scene_source()
        self._adapter_source = source
        # Borrow the adapter-owned source unless a viewer material overlay needs
        # a derived table. The owner's table identifies resources and checkpoints;
        # ordinary borrowers retain live updates made through public scene handles.
        self._source = (
            replace(source, materials=list(source.materials))
            if self._scene_overrides.materials
            else source
        )
        self._apply_material_overrides()
        self._rebind_geometry_color_overrides()

        self._nodes = [
            replace(node, children=list(node.children)) for node in self._adapter.nodes()
        ]
        self._source.lights = self._configured_lights()
        self._ensure_environment_node()
        for node in self._nodes:
            if 0 <= node.light_index < len(self._source.lights.lights):
                node.visible = self._source.lights.lights[node.light_index].active

        self._refresh_structure_metadata()
        self._by_node_id = {n.node_id: n for n in self._nodes}
        self._by_object_id = {n.object_id: n for n in self._nodes if n.object_id}
        self._restore_structure_selection(selected_before)
        self._invalidate_incompatible_recordings()

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

    def _rebind_geometry_color_overrides(self) -> None:
        """Resolve retained instance colors after hierarchy indices move."""
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

    def _ensure_environment_node(self) -> None:
        """Give every Session an editable environment node, including render-only scenes."""
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

    def _refresh_structure_metadata(self) -> None:
        """Refresh capability-owned metadata and its joint/camera lookup tables."""
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

    def _restore_structure_selection(self, selected_before: SceneNode | None) -> None:
        """Restore selection by authored identity or a surviving scene object ID."""
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

    def _invalidate_incompatible_recordings(self) -> None:
        """Retain recordings only when their physics layout or structure still matches."""
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

    def _configured_lights(self):
        # Keep the adapter-owned source reference so public scene light handles
        # remain live, while viewer overlays stay confined to the Session copy.
        configured = self._adapter_source.lights
        if self._scene_overrides.environment is not None:
            configured = configured.with_environment(self._scene_overrides.environment)
        if self._scene_overrides.lights:
            light_nodes = {
                node.object_id: node
                for node in self._nodes
                if node.object_id > 0 and node.light_index >= 0
            }
            lights = list(configured.lights)
            for override in self._scene_overrides.lights.values():
                node = light_nodes.get(override.object_id) if override.object_id > 0 else None
                index = node.light_index if node is not None else override.light_index
                if 0 <= index < len(lights):
                    lights[index] = override.light
            configured = replace(configured, lights=tuple(lights))
        return configured

    def _compose_lights(self) -> None:
        """Combine scene light settings with backend-driven transforms.

        A physics backend may move a body-attached light and publish its world
        position/direction in ``SceneFrame``.  Color, intensity, range, fog and
        every other render setting still come from the Mojive scene.
        """
        if self._source is None:
            return
        configured = self._configured_lights()
        self._source.lights = configured
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
