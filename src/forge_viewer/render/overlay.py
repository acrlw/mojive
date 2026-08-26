"""Backend-neutral publishing of scene diagnostics into the DebugDraw store.

Port of ``ForgeBackend._publish_*``: pure CPU work that turns a ``SceneFrame``
(joints, COM, inertia, actuators, rangefinders, constraints, BVH, contacts,
flex debug, camera/light icons, labels, and coordinate frames) into retained
``DebugDraw`` primitives.  Both render backends run the same publisher from
``update()``; only the GPU upload differs (``render.forge.passes.debug`` vs
``render.webgpu.passes.debug``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import math3d
from ..adapters.base import (
    ActuatorVisualType,
    BvhType,
    JointVisualType,
    SceneFrame,
    SceneSource,
)
from ..types import CameraView, LightType
from .backend import FrameMode, LabelMode, RenderFlag
from .debugdraw import DebugDraw, Occlusion

_BOX_CORNERS = np.array(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (1, 1, 1),
    ],
    np.float32,
)
_BOX_EDGES = np.array(
    (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
    np.intp,
)


@dataclass(frozen=True)
class OverlayState:
    """Per-frame overlay inputs owned by the calling backend."""

    camera: CameraView
    viewport_height: int
    selected: int
    label_mode: LabelMode
    frame_mode: FrameMode
    bvh_depth: int


class OverlayPublisher:
    """Writes per-frame debug overlays into a retained ``DebugDraw`` store.

    The flag dict is the backend's live mapping, so overlay reads always see
    the current frame's switches.  Reusable packing buffers (contact ends,
    actuator palette) survive across frames to avoid transient allocations.
    """

    def __init__(self, debug: DebugDraw, flags: dict[RenderFlag, bool]) -> None:
        self.debug = debug
        self._flags = flags
        self._source: SceneSource | None = None
        self._actuator_palette = np.zeros((0, 4), np.float32)
        self._contact_ends = np.zeros((0, 3), np.float32)

    @property
    def actuator_palette(self) -> np.ndarray:
        """Ctrl/activation-colored actuator palette, filled during publish."""
        return self._actuator_palette

    def get_flag(self, flag: RenderFlag) -> bool:
        return self._flags.get(flag, False)

    def set_scene(self, source: SceneSource) -> None:
        self._source = source
        self._actuator_palette = np.zeros((len(source.actuator_tendon), 4), np.float32)
        self.debug.layer("physics.joints").clear()
        self.debug.layer("physics.com").clear()
        self.debug.layer("physics.inertia").clear()
        self.debug.layer("physics.actuators").clear()
        self.debug.layer("scene.cameras", Occlusion.GHOST).clear()
        self.debug.layer("scene.lights", Occlusion.GHOST).clear()
        self.debug.layer("physics.rangefinders", Occlusion.GHOST).clear()
        self.debug.layer("physics.constraints", Occlusion.DEPTH).clear()
        self.debug.layer("physics.bvh", Occlusion.DEPTH).clear()

    def publish(self, frame: SceneFrame, state: OverlayState) -> None:
        self._publish_diagnostics(frame, state)
        self._publish_scene_icons(frame, state)
        self._publish_flex_debug(frame)
        self._publish_labels(frame, state)
        self._publish_frames(frame, state)
        self._publish_contacts(frame)

    def _publish_contacts(self, frame: SceneFrame) -> None:
        contacts = frame.contacts
        contact_source = self._source.diagnostics
        points = self.debug.layer("physics.contact.points", Occlusion.ALWAYS)
        forces = self.debug.layer("physics.contact.forces", Occlusion.GHOST)
        if contacts is None or not len(contacts):
            points.erase("contacts")
            forces.clear()
        else:
            if self.get_flag(RenderFlag.CONTACTPOINT):
                color = (
                    frame.contact_island_rgba
                    if self.get_flag(RenderFlag.ISLAND) and frame.contact_island_rgba is not None
                    else contact_source.contact_point_rgba
                )
                points.points("contacts", contacts[:, :3], color, 4.0)
            else:
                points.erase("contacts")
            if self.get_flag(RenderFlag.CONTACTFORCE):
                n = len(contacts)
                needed = 2 * n if self.get_flag(RenderFlag.CONTACTSPLIT) else n
                if needed > len(self._contact_ends):
                    cap = max(needed, 2 * len(self._contact_ends), 64)
                    self._contact_ends = np.zeros((cap, 3), np.float32)
                forces.clear()
                components = frame.contact_forces
                if components is None:
                    self._contact_ends[:n] = contacts[:, :3]
                    self._contact_ends[:n] += (
                        contacts[:, 3:6] * contacts[:, 6:7] * contact_source.contact_force_scale
                    )
                    forces.arrows(
                        "forces",
                        contacts[:, :3],
                        self._contact_ends[:n],
                        contact_source.contact_force_rgba,
                        2.0,
                    )
                elif self.get_flag(RenderFlag.CONTACTSPLIT):
                    scale = contact_source.contact_force_scale
                    self._contact_ends[:n] = contacts[:, :3] + components[:, 0] * scale
                    self._contact_ends[n : 2 * n] = contacts[:, :3] + components[:, 1] * scale
                    forces.arrows(
                        "normal",
                        contacts[:, :3],
                        self._contact_ends[:n],
                        contact_source.contact_force_rgba,
                        2.0,
                    )
                    forces.arrows(
                        "friction",
                        contacts[:, :3],
                        self._contact_ends[n : 2 * n],
                        contact_source.contact_friction_rgba,
                        2.0,
                    )
                else:
                    scale = contact_source.contact_force_scale
                    self._contact_ends[:n] = contacts[:, :3] + components.sum(axis=1) * scale
                    forces.arrows(
                        "forces",
                        contacts[:, :3],
                        self._contact_ends[:n],
                        contact_source.contact_force_rgba,
                        2.0,
                    )
            else:
                forces.clear()

    def _publish_flex_debug(self, frame: SceneFrame) -> None:
        vertices = frame.flex_vertices
        points = self.debug.layer("deformable.flex.vertices", Occlusion.ALWAYS)
        edges = self.debug.layer("deformable.flex.edges", Occlusion.ALWAYS)
        if vertices is None:
            points.clear()
            edges.clear()
            return
        source = self._source
        if self.get_flag(RenderFlag.FLEXVERT) and len(source.flex_vertex_indices):
            indices = source.flex_vertex_indices
            color = source.flex_vertex_rgba
            if self.get_flag(RenderFlag.ISLAND) and frame.flex_island_rgba is not None:
                color = frame.flex_island_rgba[source.flex_vertex_owner]
            points.points("vertices", vertices[indices], color, 3.5)
        else:
            points.clear()
        if self.get_flag(RenderFlag.FLEXEDGE) and len(source.flex_edges):
            topology = source.flex_edges
            color = source.flex_edge_rgba
            if self.get_flag(RenderFlag.ISLAND) and frame.flex_island_rgba is not None:
                color = frame.flex_island_rgba[source.flex_edge_owner]
            edges.lines(
                "edges",
                vertices[topology[:, 0]],
                vertices[topology[:, 1]],
                color,
                1.4,
            )
        else:
            edges.clear()

    def _publish_labels(self, frame: SceneFrame, state: OverlayState) -> None:
        layer = self.debug.layer("scene.labels", Occlusion.GHOST)
        layer.clear()
        mode = state.label_mode
        source = self._source
        if mode is LabelMode.NONE:
            return
        if mode is LabelMode.BODY and frame.body_xpos is not None:
            _draw_labels(layer, mode, source.body_names, frame.body_xpos)
        elif mode is LabelMode.JOINT and frame.diagnostics is not None:
            _draw_labels(layer, mode, source.joint_names, frame.diagnostics.joint_xpos)
        elif mode is LabelMode.GEOM and frame.geom_xpos is not None:
            _draw_labels(layer, mode, source.geom_names, frame.geom_xpos)
        elif mode is LabelMode.SITE and frame.site_xpos is not None:
            _draw_labels(layer, mode, source.site_names, frame.site_xpos)
        elif mode is LabelMode.CAMERA:
            cameras = frame.cameras if frame.cameras is not None else source.cameras
            _draw_labels(layer, mode, source.camera_names, [view.eye for view in cameras])
        elif mode is LabelMode.LIGHT:
            lights = frame.lights if frame.lights is not None else source.lights
            _draw_labels(
                layer, mode, source.light_names, [light.position for light in lights.lights]
            )
        elif mode is LabelMode.TENDON:
            self._draw_tendon_labels(layer, frame)
        elif mode is LabelMode.ACTUATOR and frame.diagnostics is not None:
            seen: set[int] = set()
            for record, actuator in enumerate(source.diagnostics.actuator_visual_actuators):
                actuator = int(actuator)
                if actuator in seen or actuator >= len(source.actuator_names):
                    continue
                seen.add(actuator)
                layer.text(
                    f"{mode.value}:{actuator}",
                    frame.diagnostics.actuator_xpos[record],
                    source.actuator_names[actuator],
                )
        elif mode is LabelMode.CONSTRAINT and frame.diagnostics is not None:
            dynamic = frame.diagnostics
            for index in np.flatnonzero(dynamic.constraint_visible):
                if index < len(source.constraint_names):
                    anchor = 0.5 * (
                        dynamic.constraint_starts[index] + dynamic.constraint_ends[index]
                    )
                    layer.text(f"{mode.value}:{index}", anchor, source.constraint_names[index])
        elif mode is LabelMode.FLEX and frame.flex_vertices is not None:
            for index, (start, count) in enumerate(source.flex_vertex_ranges):
                if count and index < len(source.flex_names):
                    anchor = frame.flex_vertices[start : start + count].mean(axis=0)
                    layer.text(f"{mode.value}:{index}", anchor, source.flex_names[index])
        elif mode in (LabelMode.CONTACT_POINT, LabelMode.CONTACT_FORCE):
            contacts = frame.contacts
            if contacts is not None:
                for index, contact in enumerate(contacts):
                    text = str(index)
                    if mode is LabelMode.CONTACT_FORCE:
                        text = f"{contact[6]:.3g} N"
                    layer.text(f"{mode.value}:{index}", contact[:3], text)
        elif mode is LabelMode.SELECTION and state.selected:
            node = next((node for node in source.nodes if node.object_id == state.selected), None)
            if node is not None and frame.body_xpos is not None and node.body_index >= 0:
                layer.text("selection", frame.body_xpos[node.body_index], node.name)

    def _draw_tendon_labels(self, layer, frame: SceneFrame) -> None:
        source = self._source
        if frame.tendon_segments is None or frame.tendon_ids is None:
            return
        for tendon, name in enumerate(source.tendon_names):
            matches = frame.tendon_ids == tendon
            if np.any(matches):
                anchor = frame.tendon_segments[matches].mean(axis=(0, 1))
                layer.text(f"tendon:{tendon}", anchor, name)

    def _publish_frames(self, frame: SceneFrame, state: OverlayState) -> None:
        layer = self.debug.layer("scene.frames", Occlusion.GHOST)
        layer.clear()
        mode = state.frame_mode
        source = self._source
        length = source.debug_frame_length
        if mode is FrameMode.NONE:
            return
        if mode is FrameMode.WORLD:
            layer.frame("world", np.eye(4, dtype=np.float32), length)
        elif mode is FrameMode.BODY and frame.body_xpos is not None:
            _draw_frames(layer, mode, frame.body_xpos, frame.body_xmat, length)
        elif mode is FrameMode.GEOM and frame.geom_xpos is not None:
            _draw_frames(layer, mode, frame.geom_xpos, frame.geom_xmat, length)
        elif mode is FrameMode.SITE and frame.site_xpos is not None:
            _draw_frames(layer, mode, frame.site_xpos, frame.site_xmat, length)
        elif mode is FrameMode.CAMERA:
            cameras = frame.cameras if frame.cameras is not None else source.cameras
            for index, view in enumerate(cameras):
                forward = math3d.normalize(np.asarray(view.target) - np.asarray(view.eye))
                right = math3d.normalize(np.cross(forward, np.asarray(view.up)))
                up = math3d.normalize(np.cross(right, forward))
                rotation = np.column_stack((right, up, -forward)).astype(np.float32)
                layer.frame(
                    f"{mode.value}:{index}", math3d.compose(view.eye, rotation, 1.0), length
                )
        elif mode is FrameMode.LIGHT:
            lights = frame.lights if frame.lights is not None else source.lights
            for index, light in enumerate(lights.lights):
                layer.frame(
                    f"{mode.value}:{index}",
                    math3d.compose(light.position, _axis_rotation(light.direction), 1.0),
                    length,
                )
        elif mode is FrameMode.CONTACT and frame.contacts is not None:
            for index, contact in enumerate(frame.contacts):
                layer.frame(
                    f"{mode.value}:{index}",
                    math3d.compose(contact[:3], _axis_rotation(contact[3:6]), 1.0),
                    length,
                )

    def _publish_diagnostics(self, frame: SceneFrame, state: OverlayState) -> None:
        joints = self.debug.layer("physics.joints", Occlusion.DEPTH)
        com = self.debug.layer("physics.com", Occlusion.DEPTH)
        inertia = self.debug.layer("physics.inertia", Occlusion.DEPTH)
        actuators = self.debug.layer("physics.actuators", Occlusion.DEPTH)
        rangefinders = self.debug.layer("physics.rangefinders", Occlusion.GHOST)
        constraints = self.debug.layer("physics.constraints", Occlusion.DEPTH)
        autoconnect = self.debug.layer("physics.autoconnect", Occlusion.DEPTH)
        bvh = self.debug.layer("physics.bvh", Occlusion.DEPTH)
        dynamic = frame.diagnostics
        source = self._source.diagnostics
        if dynamic is None:
            joints.clear()
            com.clear()
            inertia.clear()
            actuators.clear()
            rangefinders.clear()
            constraints.clear()
            autoconnect.clear()
            bvh.clear()
            return

        if self.get_flag(RenderFlag.JOINT):
            joint_radius = 3.0 * source.joint_width
            identity = np.eye(3, dtype=np.float32)
            for joint in np.flatnonzero(source.joint_visible):
                position = dynamic.joint_xpos[joint]
                visual_type = JointVisualType(int(source.joint_types[joint]))
                if visual_type is JointVisualType.FREE:
                    joints.box(
                        f"free:{joint}",
                        math3d.compose(position, identity, np.full(3, joint_radius)),
                        source.joint_rgba,
                    )
                elif visual_type is JointVisualType.BALL:
                    joints.sphere(
                        f"ball:{joint}",
                        math3d.compose(position, identity, np.full(3, joint_radius)),
                        source.joint_rgba,
                    )
                else:
                    transform = math3d.compose(
                        position,
                        _axis_rotation(dynamic.joint_xaxis[joint]),
                        (2.0 * source.joint_width, 2.0 * source.joint_width, source.joint_length),
                    )
                    if visual_type is JointVisualType.SLIDE:
                        joints.solid_double_arrow(f"slide:{joint}", transform, source.joint_rgba)
                    else:
                        joints.solid_arrow(f"hinge:{joint}", transform, source.joint_rgba)
        else:
            joints.clear()

        if self.get_flag(RenderFlag.COM):
            for body in source.com_bodies:
                com.sphere(
                    f"body:{body}",
                    math3d.compose(
                        dynamic.subtree_com[body],
                        np.eye(3, dtype=np.float32),
                        np.full(3, source.com_radius),
                    ),
                    source.com_rgba,
                )
        else:
            com.clear()

        if self.get_flag(RenderFlag.INERTIA):
            sizes = (
                source.scaled_inertia_sizes
                if self.get_flag(RenderFlag.SCLINERTIA)
                else source.inertia_sizes
            )
            for index, body in enumerate(source.inertia_bodies):
                inertia.box(
                    f"body:{body}",
                    math3d.compose(
                        dynamic.body_xipos[body], dynamic.body_ximat[body], sizes[index]
                    ),
                    source.inertia_rgba,
                )
        else:
            inertia.clear()

        self._publish_actuator_visuals(frame, actuators)
        self._publish_rangefinders(frame, rangefinders)
        self._publish_constraints(frame, constraints)
        self._publish_bvh(frame, bvh, state)
        if self.get_flag(RenderFlag.AUTOCONNECT):
            for index, segment in enumerate(dynamic.autoconnect_segments):
                _draw_capsule_between(
                    autoconnect,
                    f"segment:{index}",
                    segment[0],
                    segment[1],
                    source.autoconnect_width,
                    source.autoconnect_rgba,
                )
        else:
            autoconnect.clear()

    def _publish_bvh(self, frame: SceneFrame, layer, state: OverlayState) -> None:
        source = self._source.diagnostics
        dynamic = frame.diagnostics
        show_body = self.get_flag(RenderFlag.BODYBVH)
        show_mesh = self.get_flag(RenderFlag.MESHBVH)
        if dynamic is None or not (show_body or show_mesh):
            layer.clear()
            return

        bvh_type = source.bvh_type
        selected = np.zeros(len(bvh_type), bool)
        if show_body:
            selected |= bvh_type == int(BvhType.BODY)
        if show_mesh:
            selected |= bvh_type != int(BvhType.BODY)
        selected &= (source.bvh_depth == state.bvh_depth) | (
            source.bvh_leaf & (source.bvh_depth < state.bvh_depth)
        )
        if source.bvh_active_highlight:
            selected &= ~((bvh_type == int(BvhType.MESH)) & ~dynamic.bvh_active)
        records = np.flatnonzero(selected)
        layer.clear()
        if len(records):
            local = dynamic.bvh_sizes[records, None, :] * _BOX_CORNERS[None, :, :]
            corners = dynamic.bvh_centers[records, None, :] + np.einsum(
                "nij,nkj->nki", dynamic.bvh_matrices[records], local
            )
            starts = corners[:, _BOX_EDGES[:, 0]].reshape(-1, 3)
            ends = corners[:, _BOX_EDGES[:, 1]].reshape(-1, 3)
            colors = np.repeat(source.bvh_rgba[None], len(records), axis=0)
            colors[dynamic.bvh_active[records]] = source.bvh_active_rgba
            layer.lines("boxes", starts, ends, np.repeat(colors, len(_BOX_EDGES), axis=0), 1.5)
        if show_mesh and len(dynamic.bvh_control_segments):
            segments = dynamic.bvh_control_segments
            layer.lines(
                "control_cages",
                segments[:, 0],
                segments[:, 1],
                source.bvh_control_rgba,
                3.0,
            )

    def _publish_constraints(self, frame: SceneFrame, layer) -> None:
        dynamic = frame.diagnostics
        if not self.get_flag(RenderFlag.CONSTRAINT) or dynamic is None:
            layer.clear()
            return
        layer.clear()
        source = self._source.diagnostics
        identity = np.eye(3, dtype=np.float32)
        scale = np.full(3, source.constraint_radius, np.float32)
        for equality in np.flatnonzero(dynamic.constraint_visible):
            layer.sphere(
                f"connect:{equality}",
                math3d.compose(dynamic.constraint_starts[equality], identity, scale),
                source.constraint_connect_rgba,
            )
            layer.sphere(
                f"constraint:{equality}",
                math3d.compose(dynamic.constraint_ends[equality], identity, scale),
                source.constraint_rgba,
            )

    def _publish_rangefinders(self, frame: SceneFrame, layer) -> None:
        dynamic = frame.diagnostics
        if not self.get_flag(RenderFlag.RANGEFINDER) or dynamic is None:
            layer.clear()
            return
        source = self._source.diagnostics
        lines = dynamic.rangefinder_lines
        points = dynamic.rangefinder_points
        normals = dynamic.rangefinder_normal_arrows
        layer.lines(
            "rays",
            dynamic.rangefinder_starts[lines],
            dynamic.rangefinder_ends[lines],
            source.rangefinder_rgba,
            1.6,
        )
        layer.points(
            "hits",
            dynamic.rangefinder_ends[points],
            source.rangefinder_rgba,
            4.0,
        )
        starts = dynamic.rangefinder_ends[normals]
        layer.arrows(
            "normals",
            starts,
            starts + dynamic.rangefinder_normals[normals] * source.rangefinder_normal_length,
            source.rangefinder_rgba,
            1.6,
        )

    def _publish_actuator_visuals(self, frame: SceneFrame, layer) -> None:
        source = self._source.diagnostics
        dynamic = frame.diagnostics
        if not self.get_flag(RenderFlag.ACTUATOR) or frame.ctrl is None or dynamic is None:
            layer.clear()
            return

        palette = self.fill_actuator_palette(frame)
        for record, actuator in enumerate(source.actuator_visual_actuators):
            actuator = int(actuator)
            if not self._source.actuator_visible[actuator]:
                continue
            visual_type = ActuatorVisualType(int(source.actuator_visual_types[record]))
            position = dynamic.actuator_xpos[record]
            rotation = dynamic.actuator_xmat[record]
            size = source.actuator_visual_sizes[record]
            color = palette[actuator]
            transform = math3d.compose(position, rotation, size)
            ident = f"actuator:{record}"
            if visual_type is ActuatorVisualType.SLIDE:
                layer.solid_double_arrow(ident, transform, color)
            elif visual_type is ActuatorVisualType.HINGE:
                layer.solid_arrow(ident, transform, color)
            elif visual_type in (
                ActuatorVisualType.BALL,
                ActuatorVisualType.SPHERE,
                ActuatorVisualType.ELLIPSOID,
            ):
                layer.sphere(ident, transform, color)
            elif visual_type in (ActuatorVisualType.FREE, ActuatorVisualType.BOX):
                layer.box(ident, transform, color)
            elif visual_type is ActuatorVisualType.CYLINDER:
                layer.cylinder(ident, transform, color)
            else:
                _draw_capsule(layer, ident, position, rotation, size, color)

        for record, actuator in enumerate(source.slider_crank_actuators):
            actuator = int(actuator)
            if not self._source.actuator_visible[actuator]:
                continue
            slider, joint, crank = dynamic.slider_crank_points[record]
            color = source.slider_crank_rgba
            _draw_cylinder_between(
                layer,
                f"slider-crank:{record}:slider",
                slider,
                joint,
                source.slider_crank_width,
                color,
            )
            rod_color = (
                source.slider_crank_broken_rgba if dynamic.slider_crank_broken[record] else color
            )
            _draw_capsule_between(
                layer,
                f"slider-crank:{record}:rod",
                joint,
                crank,
                source.slider_crank_width * 0.5,
                rod_color,
            )

    def _publish_scene_icons(self, frame: SceneFrame, state: OverlayState) -> None:
        cameras = self.debug.layer("scene.cameras", Occlusion.GHOST)
        lights = self.debug.layer("scene.lights", Occlusion.GHOST)
        source = self._source

        if self.get_flag(RenderFlag.CAMERA):
            views = frame.cameras if frame.cameras is not None else source.cameras
            for index, view in enumerate(views):
                self._draw_camera_icon(cameras, index, view, source.diagnostics.camera_rgba, state)
        else:
            cameras.clear()

        if self.get_flag(RenderFlag.LIGHT):
            light_set = frame.lights if frame.lights is not None else source.lights
            for index, light in enumerate(light_set.lights):
                if light.active:
                    self._draw_light_icon(
                        lights, index, light, source.diagnostics.light_rgba, state
                    )
                else:
                    lights.erase(f"light:{index}:point")
                    lights.erase(f"light:{index}:direction")
        else:
            lights.clear()

    def _draw_camera_icon(self, layer, index: int, view: CameraView, color, state) -> None:
        eye = np.asarray(view.eye, np.float32)
        forward = math3d.normalize(np.asarray(view.target, np.float32) - eye)
        right = math3d.normalize(np.cross(forward, np.asarray(view.up, np.float32)))
        up = math3d.normalize(np.cross(right, forward))
        length = self._icon_world_size(eye, 26.0, state)
        center = eye + forward * length
        half_height = length * 0.45
        half_width = half_height * min(max(float(view.aspect), 0.75), 1.8)
        corners = np.stack(
            (
                center - right * half_width - up * half_height,
                center + right * half_width - up * half_height,
                center + right * half_width + up * half_height,
                center - right * half_width + up * half_height,
            )
        )
        starts = np.concatenate((np.repeat(eye[None], 4, axis=0), corners), axis=0)
        ends = np.concatenate((corners, np.roll(corners, -1, axis=0)), axis=0)
        layer.lines(f"camera:{index}", starts, ends, color, 1.8)

    def _draw_light_icon(self, layer, index: int, light, color, state) -> None:
        position = np.asarray(light.position, np.float32)
        layer.point(f"light:{index}:point", position, color, 6.0)
        if light.type in (LightType.POINT, LightType.IMAGE):
            layer.erase(f"light:{index}:direction")
            return
        direction = math3d.normalize(np.asarray(light.direction, np.float32))
        length = self._icon_world_size(position, 30.0, state)
        layer.arrow(
            f"light:{index}:direction",
            position,
            position + direction * length,
            color,
            2.0,
            start_mask_px=7.0,
        )

    def _icon_world_size(self, position: np.ndarray, pixels: float, state: OverlayState) -> float:
        camera = state.camera
        if camera.orthographic:
            return float(camera.ortho_height) * float(pixels) / max(state.viewport_height, 1)
        depth = abs(float(np.dot(position - camera.eye, camera.forward())))
        depth = max(depth, float(camera.near), 1e-4)
        world_per_pixel = (
            2.0 * depth * np.tan(float(camera.fov_y) * 0.5) / max(state.viewport_height, 1)
        )
        return float(world_per_pixel * pixels)

    def fill_actuator_palette(self, frame: SceneFrame) -> np.ndarray:
        """Fill and return the per-actuator RGBA palette for this frame."""
        source = self._source
        use_activation = self.get_flag(RenderFlag.ACTIVATION)
        for i, out in enumerate(self._actuator_palette):
            if source.actuator_ctrl_limited[i]:
                rmin, rmax = source.actuator_ctrl_range[i]
            elif use_activation and source.actuator_act_limited[i]:
                rmin, rmax = source.actuator_act_range[i]
            else:
                rmin, rmax = -1.0, 1.0
            if rmin >= 0.0:
                low, middle, high = -1.0, float(rmin), float(rmax)
            elif rmax <= 0.0:
                low, middle, high = float(rmin), float(rmax), 1.0
            else:
                low, middle, high = float(rmin), 0.0, float(rmax)
            value = float(frame.ctrl[source.actuator_ctrl_address[i]])
            if (
                use_activation
                and source.actuator_dynamic[i]
                and frame.actuator_activation is not None
            ):
                value = float(frame.actuator_activation[i])
            value = min(max(value, low), high)
            if value <= middle:
                weight = (middle - value) / max(middle - low, 1e-15)
                out[:] = weight * source.actuator_rgba[0] + (1.0 - weight) * source.actuator_rgba[1]
            else:
                weight = (value - middle) / max(high - middle, 1e-15)
                out[:] = (1.0 - weight) * source.actuator_rgba[1] + weight * source.actuator_rgba[2]
        return self._actuator_palette


def _draw_labels(layer, mode: LabelMode, names, positions) -> None:
    for index, (name, position) in enumerate(zip(names, positions, strict=False)):
        layer.text(f"{mode.value}:{index}", position, name)


def _draw_frames(layer, mode: FrameMode, positions, rotations, length: float) -> None:
    if rotations is None:
        return
    for index, (position, rotation) in enumerate(zip(positions, rotations, strict=False)):
        layer.frame(f"{mode.value}:{index}", math3d.compose(position, rotation, 1.0), length)


def _draw_capsule(layer, ident, position, rotation, size, color) -> None:
    radius, half_length = float(size[0]), float(size[2])
    layer.cylinder(
        f"{ident}:shaft",
        math3d.compose(position, rotation, (radius, radius, half_length)),
        color,
    )
    offset = rotation[:, 2] * half_length
    sphere_scale = np.full(3, radius, np.float32)
    layer.sphere(
        f"{ident}:cap-",
        math3d.compose(position - offset, rotation, sphere_scale),
        color,
    )
    layer.sphere(
        f"{ident}:cap+",
        math3d.compose(position + offset, rotation, sphere_scale),
        color,
    )


def _draw_cylinder_between(layer, ident, start, end, radius, color) -> None:
    position, rotation, half_length = _connector_pose(start, end)
    layer.cylinder(
        ident,
        math3d.compose(position, rotation, (radius, radius, half_length)),
        color,
    )


def _draw_capsule_between(layer, ident, start, end, radius, color) -> None:
    position, rotation, half_length = _connector_pose(start, end)
    _draw_capsule(layer, ident, position, rotation, (radius, radius, half_length), color)


def _connector_pose(start, end) -> tuple[np.ndarray, np.ndarray, float]:
    start = np.asarray(start, np.float32)
    end = np.asarray(end, np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    rotation = _axis_rotation(delta) if length else np.eye(3, dtype=np.float32)
    return (start + end) * 0.5, rotation, length * 0.5


def _axis_rotation(axis: np.ndarray) -> np.ndarray:
    z = np.asarray(axis, np.float32)
    z /= np.linalg.norm(z)
    reference = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(z[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], np.float32)
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z)).astype(np.float32)
