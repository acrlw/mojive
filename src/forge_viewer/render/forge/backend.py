from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import moderngl
import numpy as np

from ... import math3d
from ...adapters.base import ActuatorVisualKind, JointVisualKind, SceneFrame, SceneSource
from ...gizmo import GizmoFrame
from ...log import get_logger
from ...types import CameraView, LightKind, MeshKey, ViewportImage
from ..backend import BackendCaps, DebugView, FrameMode, LabelMode, RenderFlag, RenderStats
from ..debugdraw import Occlusion
from ..scene import RenderScene
from .context import ContextCaps, attach
from .instances import InstanceStore
from .passes.base import PassContext, RenderPass, ShadowResult
from .passes.present import PresentPass
from .programs import ProgramCache
from .registry import PASS_ORDER, register_pass, registered
from .resources import MeshStore, TextureStore
from .state_guard import GLStateGuard, bind_default_framebuffer
from .targets import RenderTarget
from .timing import FrameTiming

log = get_logger("backend")


PassFactory = Callable[[], RenderPass]


class ForgeBackend:
    def __init__(
        self,
        gl_context: moderngl.Context | None = None,
        width: int = 1280,
        height: int = 720,
        samples: int = 4,
        shader_dir: Path | None = None,
    ) -> None:
        self.ctx, self.gl_caps = attach(gl_context)
        self.guard = GLStateGuard()
        self.programs = ProgramCache(self.ctx, shader_dir)
        self.meshes = MeshStore(self.ctx)
        self.textures = TextureStore(self.ctx)
        self.instances = InstanceStore(self.ctx)
        samples = min(samples, max(self.gl_caps.max_samples, 1))
        self.target = RenderTarget(self.ctx, width, height, samples)
        self.timing = FrameTiming(self.ctx, enabled=self.gl_caps.timer_query != "none")
        self.stats = RenderStats()

        from . import passes as _passes_pkg

        _passes_pkg.load_all()
        self.pass_load_failures = _passes_pkg.failed()
        for name, why in self.pass_load_failures.items():
            log.error("Pass {} failed to load: {}", name, why)

        factories = registered()
        self._passes: dict[str, RenderPass] = {}
        for name in PASS_ORDER:
            if name == "present":
                self._passes[name] = PresentPass()
            elif name in factories:
                self._passes[name] = factories[name]()
        self.debug = getattr(self._passes.get("debug"), "draw", None)

        self._scene: RenderScene | None = None
        self._source: SceneSource | None = None
        self._builder = None
        self._camera = CameraView()
        self._selected = 0
        self._gizmo: GizmoFrame | None = None
        self._debug_view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE
        self._flags: dict[RenderFlag, bool] = dict.fromkeys(self._supported_flags(), True)
        self._flags[RenderFlag.WIREFRAME] = False
        self._flags[RenderFlag.FOG] = False
        self._flags[RenderFlag.HAZE] = False
        self._flags[RenderFlag.CONTACTPOINT] = False
        self._flags[RenderFlag.CONTACTFORCE] = False
        self._flags[RenderFlag.CONTACTSPLIT] = False
        self._flags[RenderFlag.AUTOCONNECT] = False
        self._flags[RenderFlag.ACTUATOR] = False
        self._flags[RenderFlag.ACTIVATION] = False
        self._flags[RenderFlag.JOINT] = False
        self._flags[RenderFlag.COM] = False
        self._flags[RenderFlag.INERTIA] = False
        self._flags[RenderFlag.SCLINERTIA] = False
        self._flags[RenderFlag.CAMERA] = False
        self._flags[RenderFlag.LIGHT] = False
        self._flags[RenderFlag.RANGEFINDER] = False
        self._flags[RenderFlag.CONSTRAINT] = False
        self._flags[RenderFlag.FLEXFACE] = False
        self._flags[RenderFlag.FLEXVERT] = False
        # MuJoCo mjv_defaultOption() enables tendon paths by default.
        self._flags[RenderFlag.TENDON] = True
        self._contact_ends = np.zeros((0, 3), np.float32)
        self._actuator_palette = np.zeros((0, 4), np.float32)
        self._tendon_actuator = np.zeros(0, np.int32)
        self._capsule_segments = np.zeros((0, 2, 3), np.float32)
        self._capsule_widths = np.zeros(0, np.float32)
        self._capsule_colors = np.zeros((0, 4), np.float32)
        self._capsule_materials = np.zeros((0, 4), np.float32)
        self._capsule_material_ids = np.zeros(0, np.int32)
        self._capsule_transparent = np.zeros(0, bool)
        self._bucket_meshes: list = []
        self._structure_generation = -1
        self._program_generation = -1
        self.caps = self._build_caps()
        self._hot_reload = False

    # ------------------------------------------------------------------
    def _supported_flags(self) -> frozenset[RenderFlag]:

        flags = {
            RenderFlag.CULL_FACE,
            RenderFlag.TEXTURE,
            RenderFlag.TRANSPARENT,
            RenderFlag.MSAA,
            RenderFlag.TONEMAP,
            RenderFlag.WIREFRAME,
            RenderFlag.FOG,
            RenderFlag.HAZE,
            RenderFlag.STATIC,
            RenderFlag.SKIN,
            RenderFlag.FLEXFACE,
            RenderFlag.FLEXSKIN,
        }
        if "shadow" in self._passes:
            flags.add(RenderFlag.SHADOW)
        if "skybox" in self._passes:
            flags.add(RenderFlag.SKYBOX)
        if "outline" in self._passes:
            flags.add(RenderFlag.OUTLINE)
        if "reflect" in self._passes:
            flags.add(RenderFlag.REFLECTION)
        if "debug" in self._passes:
            flags |= {
                RenderFlag.CONTACTPOINT,
                RenderFlag.CONTACTFORCE,
                RenderFlag.CONTACTSPLIT,
                RenderFlag.AUTOCONNECT,
                RenderFlag.TENDON,
                RenderFlag.ACTUATOR,
                RenderFlag.ACTIVATION,
                RenderFlag.JOINT,
                RenderFlag.COM,
                RenderFlag.INERTIA,
                RenderFlag.SCLINERTIA,
                RenderFlag.CAMERA,
                RenderFlag.LIGHT,
                RenderFlag.RANGEFINDER,
                RenderFlag.CONSTRAINT,
                RenderFlag.FLEXVERT,
                RenderFlag.FLEXEDGE,
            }
        return frozenset(flags)

    def _build_caps(self) -> BackendCaps:
        views = {DebugView.SHADED, DebugView.SEGMENT, DebugView.IDCOLOR}
        if "opaque" in self._passes:
            views |= {
                DebugView.ALBEDO,
                DebugView.NORMAL,
                DebugView.DEPTH,
                DebugView.OVERDRAW,
                DebugView.WIREFRAME,
            }
        return BackendCaps(
            name="forge",
            gpu_pick=True,
            debug_draw=self.debug is not None,
            render_flags=self._supported_flags(),
            debug_views=frozenset(views),
            label_modes=frozenset(LabelMode),
            frame_modes=frozenset(FrameMode),
            capture=True,
            orthographic=True,
            shadows="shadow" in self._passes,
            outline="outline" in self._passes,
            gizmo="gizmo" in self._passes,
            pass_timing=True,
            gpu_timing=self.timing.gpu_available,
            msaa_samples=self.target.samples,
            id_msaa=self.target.id_multisample,
            gl_version=self.gl_caps.version,
            renderer=self.gl_caps.renderer,
            notes=self.gl_caps.notes,
        )

    # ------------------------------------------------------------------
    def set_scene(self, source: SceneSource) -> None:

        from ..builder import SceneSourceBuilder
        from ..mesh import all_builtin

        self._source = source
        self._tendon_visible = source.tendon_visible
        self._actuator_visible = source.actuator_visible
        self._material_values = np.asarray(
            [
                (mat.emission, mat.specular, mat.shininess, mat.reflectance)
                for mat in source.materials
            ],
            np.float32,
        )
        self._tendon_material_table = tuple(source.materials)
        self._tendon_actuator = np.full(len(source.tendon_rgba), -1, np.int32)
        for actuator, tendon in enumerate(source.actuator_tendon):
            if 0 <= tendon < len(self._tendon_actuator):
                self._tendon_actuator[tendon] = actuator
        self._actuator_palette = np.zeros((len(source.actuator_tendon), 4), np.float32)
        self.meshes.sync({**all_builtin(), **source.meshes})
        self.textures.sync(source.textures, source.skybox)
        self._builder = SceneSourceBuilder()
        self._builder.set_source(source, self._camera)
        self._sync_instance_visibility()
        self._structure_generation = -1
        if self.debug is not None:
            self.debug.layer("physics.joints").clear()
            self.debug.layer("physics.com").clear()
            self.debug.layer("physics.inertia").clear()
            self.debug.layer("physics.actuators").clear()
            self.debug.layer("scene.cameras", Occlusion.GHOST).clear()
            self.debug.layer("scene.lights", Occlusion.GHOST).clear()
            self.debug.layer("physics.rangefinders", Occlusion.GHOST).clear()
            self.debug.layer("physics.constraints", Occlusion.DEPTH).clear()

    def set_render_scene(self, scene: RenderScene) -> None:

        self._scene = scene
        gen = self.programs.generation
        need = self.instances.needs_rebuild(scene, gen) or self._structure_generation < 0
        if need:
            self._bucket_meshes = [self.meshes.get(k) for k, _ in scene.bucket_keys]
            prog = self._scene_program()
            if prog is not None:
                self.instances.rebuild(scene, prog, self._bucket_meshes, gen)
            self._structure_generation = 0
            self._program_generation = gen

    def _scene_program(self) -> moderngl.Program | None:

        op = self._passes.get("opaque")
        getter = getattr(op, "scene_program", None)
        return getter(self) if callable(getter) else None

    def update(self, frame: SceneFrame) -> None:

        if self._builder is None:
            return
        self.meshes.update(frame.mesh_updates)
        self.set_render_scene(self._builder.update(frame, self._camera))
        self._publish_frame_visuals(frame)

    def _publish_frame_visuals(self, frame: SceneFrame) -> None:

        self._publish_tendons(frame)
        if self.debug is None:
            return
        self._publish_diagnostics(frame)
        self._publish_scene_icons(frame)
        self._publish_flex_debug(frame)
        self._publish_labels(frame)
        self._publish_frames(frame)
        contacts = frame.contacts
        contact_source = self._source.diagnostics
        points = self.debug.layer("physics.contact.points", Occlusion.ALWAYS)
        forces = self.debug.layer("physics.contact.forces", Occlusion.GHOST)
        if contacts is None or not len(contacts):
            points.erase("contacts")
            forces.clear()
        else:
            if self.get_flag(RenderFlag.CONTACTPOINT):
                points.points("contacts", contacts[:, :3], contact_source.contact_point_rgba, 4.0)
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
            points.points("vertices", vertices[indices], source.flex_vertex_rgba, 3.5)
        else:
            points.clear()
        if self.get_flag(RenderFlag.FLEXEDGE) and len(source.flex_edges):
            topology = source.flex_edges
            edges.lines(
                "edges",
                vertices[topology[:, 0]],
                vertices[topology[:, 1]],
                source.flex_edge_rgba,
                1.4,
            )
        else:
            edges.clear()

    def _publish_labels(self, frame: SceneFrame) -> None:
        layer = self.debug.layer("scene.labels", Occlusion.GHOST)
        layer.clear()
        mode = self._label_mode
        source = self._source
        if mode is LabelMode.NONE:
            return
        if mode is LabelMode.BODY and frame.body_xpos is not None:
            self._draw_labels(layer, mode, source.body_names, frame.body_xpos)
        elif mode is LabelMode.JOINT and frame.diagnostics is not None:
            self._draw_labels(layer, mode, source.joint_names, frame.diagnostics.joint_xpos)
        elif mode is LabelMode.GEOM and frame.geom_xpos is not None:
            self._draw_labels(layer, mode, source.geom_names, frame.geom_xpos)
        elif mode is LabelMode.SITE and frame.site_xpos is not None:
            self._draw_labels(layer, mode, source.site_names, frame.site_xpos)
        elif mode is LabelMode.CAMERA:
            cameras = frame.cameras if frame.cameras is not None else source.cameras
            self._draw_labels(layer, mode, source.camera_names, [view.eye for view in cameras])
        elif mode is LabelMode.LIGHT:
            lights = frame.lights if frame.lights is not None else source.lights
            self._draw_labels(
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
        elif mode is LabelMode.SELECTION and self._selected:
            node = next((node for node in source.nodes if node.object_id == self._selected), None)
            if node is not None and frame.body_xpos is not None and node.body_index >= 0:
                layer.text("selection", frame.body_xpos[node.body_index], node.name)

    @staticmethod
    def _draw_labels(layer, mode: LabelMode, names, positions) -> None:
        for index, (name, position) in enumerate(zip(names, positions, strict=False)):
            layer.text(f"{mode.value}:{index}", position, name)

    def _draw_tendon_labels(self, layer, frame: SceneFrame) -> None:
        source = self._source
        if frame.tendon_segments is None or frame.tendon_ids is None:
            return
        for tendon, name in enumerate(source.tendon_names):
            matches = frame.tendon_ids == tendon
            if np.any(matches):
                anchor = frame.tendon_segments[matches].mean(axis=(0, 1))
                layer.text(f"tendon:{tendon}", anchor, name)

    def _publish_frames(self, frame: SceneFrame) -> None:
        layer = self.debug.layer("scene.frames", Occlusion.GHOST)
        layer.clear()
        mode = self._frame_mode
        source = self._source
        length = source.debug_frame_length
        if mode is FrameMode.NONE:
            return
        if mode is FrameMode.WORLD:
            layer.frame("world", np.eye(4, dtype=np.float32), length)
        elif mode is FrameMode.BODY and frame.body_xpos is not None:
            self._draw_frames(layer, mode, frame.body_xpos, frame.body_xmat, length)
        elif mode is FrameMode.GEOM and frame.geom_xpos is not None:
            self._draw_frames(layer, mode, frame.geom_xpos, frame.geom_xmat, length)
        elif mode is FrameMode.SITE and frame.site_xpos is not None:
            self._draw_frames(layer, mode, frame.site_xpos, frame.site_xmat, length)
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
                    math3d.compose(light.position, self._axis_rotation(light.direction), 1.0),
                    length,
                )
        elif mode is FrameMode.CONTACT and frame.contacts is not None:
            for index, contact in enumerate(frame.contacts):
                layer.frame(
                    f"{mode.value}:{index}",
                    math3d.compose(contact[:3], self._axis_rotation(contact[3:6]), 1.0),
                    length,
                )

    @staticmethod
    def _draw_frames(layer, mode: FrameMode, positions, rotations, length: float) -> None:
        if rotations is None:
            return
        for index, (position, rotation) in enumerate(zip(positions, rotations, strict=False)):
            layer.frame(f"{mode.value}:{index}", math3d.compose(position, rotation, 1.0), length)

    def _publish_diagnostics(self, frame: SceneFrame) -> None:
        joints = self.debug.layer("physics.joints", Occlusion.DEPTH)
        com = self.debug.layer("physics.com", Occlusion.DEPTH)
        inertia = self.debug.layer("physics.inertia", Occlusion.DEPTH)
        actuators = self.debug.layer("physics.actuators", Occlusion.DEPTH)
        rangefinders = self.debug.layer("physics.rangefinders", Occlusion.GHOST)
        constraints = self.debug.layer("physics.constraints", Occlusion.DEPTH)
        autoconnect = self.debug.layer("physics.autoconnect", Occlusion.DEPTH)
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
            return

        if self.get_flag(RenderFlag.JOINT):
            joint_radius = 3.0 * source.joint_width
            identity = np.eye(3, dtype=np.float32)
            for joint in np.flatnonzero(source.joint_visible):
                position = dynamic.joint_xpos[joint]
                kind = JointVisualKind(int(source.joint_kinds[joint]))
                if kind is JointVisualKind.FREE:
                    joints.box(
                        f"free:{joint}",
                        math3d.compose(position, identity, np.full(3, joint_radius)),
                        source.joint_rgba,
                    )
                elif kind is JointVisualKind.BALL:
                    joints.sphere(
                        f"ball:{joint}",
                        math3d.compose(position, identity, np.full(3, joint_radius)),
                        source.joint_rgba,
                    )
                else:
                    transform = math3d.compose(
                        position,
                        self._axis_rotation(dynamic.joint_xaxis[joint]),
                        (2.0 * source.joint_width, 2.0 * source.joint_width, source.joint_length),
                    )
                    if kind is JointVisualKind.SLIDE:
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
        if self.get_flag(RenderFlag.AUTOCONNECT):
            for index, segment in enumerate(dynamic.autoconnect_segments):
                self._draw_capsule_between(
                    autoconnect,
                    f"segment:{index}",
                    segment[0],
                    segment[1],
                    source.autoconnect_width,
                    source.autoconnect_rgba,
                )
        else:
            autoconnect.clear()

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

        self._fill_actuator_palette(frame)
        for record, actuator in enumerate(source.actuator_visual_actuators):
            actuator = int(actuator)
            if not self._source.actuator_visible[actuator]:
                continue
            kind = ActuatorVisualKind(int(source.actuator_visual_kinds[record]))
            position = dynamic.actuator_xpos[record]
            rotation = dynamic.actuator_xmat[record]
            size = source.actuator_visual_sizes[record]
            color = self._actuator_palette[actuator]
            transform = math3d.compose(position, rotation, size)
            ident = f"actuator:{record}"
            if kind is ActuatorVisualKind.SLIDE:
                layer.solid_double_arrow(ident, transform, color)
            elif kind is ActuatorVisualKind.HINGE:
                layer.solid_arrow(ident, transform, color)
            elif kind in (
                ActuatorVisualKind.BALL,
                ActuatorVisualKind.SPHERE,
                ActuatorVisualKind.ELLIPSOID,
            ):
                layer.sphere(ident, transform, color)
            elif kind in (ActuatorVisualKind.FREE, ActuatorVisualKind.BOX):
                layer.box(ident, transform, color)
            elif kind is ActuatorVisualKind.CYLINDER:
                layer.cylinder(ident, transform, color)
            else:
                self._draw_capsule(layer, ident, position, rotation, size, color)

        for record, actuator in enumerate(source.slider_crank_actuators):
            actuator = int(actuator)
            if not self._source.actuator_visible[actuator]:
                continue
            slider, joint, crank = dynamic.slider_crank_points[record]
            color = source.slider_crank_rgba
            self._draw_cylinder_between(
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
            self._draw_capsule_between(
                layer,
                f"slider-crank:{record}:rod",
                joint,
                crank,
                source.slider_crank_width * 0.5,
                rod_color,
            )

    def _publish_scene_icons(self, frame: SceneFrame) -> None:
        cameras = self.debug.layer("scene.cameras", Occlusion.GHOST)
        lights = self.debug.layer("scene.lights", Occlusion.GHOST)
        source = self._source

        if self.get_flag(RenderFlag.CAMERA):
            views = frame.cameras if frame.cameras is not None else source.cameras
            for index, view in enumerate(views):
                self._draw_camera_icon(cameras, index, view, source.diagnostics.camera_rgba)
        else:
            cameras.clear()

        if self.get_flag(RenderFlag.LIGHT):
            light_set = frame.lights if frame.lights is not None else source.lights
            for index, light in enumerate(light_set.lights):
                if light.active:
                    self._draw_light_icon(lights, index, light, source.diagnostics.light_rgba)
                else:
                    lights.erase(f"light:{index}:point")
                    lights.erase(f"light:{index}:direction")
        else:
            lights.clear()

    def _draw_camera_icon(self, layer, index: int, view: CameraView, color) -> None:
        eye = np.asarray(view.eye, np.float32)
        forward = math3d.normalize(np.asarray(view.target, np.float32) - eye)
        right = math3d.normalize(np.cross(forward, np.asarray(view.up, np.float32)))
        up = math3d.normalize(np.cross(right, forward))
        length = self._icon_world_size(eye, 26.0)
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

    def _draw_light_icon(self, layer, index: int, light, color) -> None:
        position = np.asarray(light.position, np.float32)
        layer.point(f"light:{index}:point", position, color, 6.0)
        if light.kind is LightKind.POINT:
            layer.erase(f"light:{index}:direction")
            return
        direction = math3d.normalize(np.asarray(light.direction, np.float32))
        length = self._icon_world_size(position, 30.0)
        layer.arrow(
            f"light:{index}:direction",
            position,
            position + direction * length,
            color,
            2.0,
            start_mask_px=7.0,
        )

    def _icon_world_size(self, position: np.ndarray, pixels: float) -> float:
        camera = self._camera
        if camera.orthographic:
            return float(camera.ortho_height) * float(pixels) / max(self.target.height, 1)
        depth = abs(float(np.dot(position - camera.eye, camera.forward())))
        depth = max(depth, float(camera.near), 1e-4)
        world_per_pixel = (
            2.0 * depth * np.tan(float(camera.fov_y) * 0.5) / max(self.target.height, 1)
        )
        return float(world_per_pixel * pixels)

    @staticmethod
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

    @classmethod
    def _draw_cylinder_between(cls, layer, ident, start, end, radius, color) -> None:
        position, rotation, half_length = cls._connector_pose(start, end)
        layer.cylinder(
            ident,
            math3d.compose(position, rotation, (radius, radius, half_length)),
            color,
        )

    @classmethod
    def _draw_capsule_between(cls, layer, ident, start, end, radius, color) -> None:
        position, rotation, half_length = cls._connector_pose(start, end)
        cls._draw_capsule(layer, ident, position, rotation, (radius, radius, half_length), color)

    @classmethod
    def _connector_pose(cls, start, end) -> tuple[np.ndarray, np.ndarray, float]:
        start = np.asarray(start, np.float32)
        end = np.asarray(end, np.float32)
        delta = end - start
        length = float(np.linalg.norm(delta))
        rotation = cls._axis_rotation(delta) if length else np.eye(3, dtype=np.float32)
        return (start + end) * 0.5, rotation, length * 0.5

    @staticmethod
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

    def _publish_tendons(self, frame: SceneFrame) -> None:
        tendon_pass = self._passes.get("tendon")
        update = getattr(tendon_pass, "update", None)
        clear = getattr(tendon_pass, "clear", None)
        segments, ids, widths = (
            frame.tendon_segments,
            frame.tendon_ids,
            frame.tendon_widths,
        )
        if (
            not callable(update)
            or segments is None
            or ids is None
            or widths is None
            or not len(segments)
        ):
            if callable(clear):
                clear()
            return

        base_indices = (
            np.flatnonzero(self._tendon_visible[ids])
            if self.get_flag(RenderFlag.TENDON)
            else np.zeros(0, np.intp)
        )
        base_count = len(base_indices)
        actuator_indices = np.zeros(0, np.intp)
        segment_actuators = np.zeros(0, np.int32)
        if self.get_flag(RenderFlag.ACTUATOR) and frame.ctrl is not None:
            segment_actuators = self._tendon_actuator[ids]
            available = segment_actuators >= 0
            available[available] &= self._actuator_visible[segment_actuators[available]]
            actuator_indices = np.flatnonzero(available)
        total = base_count + len(actuator_indices)
        if not total:
            clear()
            return

        if total > len(self._capsule_widths):
            capacity = max(total, 2 * len(self._capsule_widths), 64)
            self._capsule_segments = np.zeros((capacity, 2, 3), np.float32)
            self._capsule_widths = np.zeros(capacity, np.float32)
            self._capsule_colors = np.zeros((capacity, 4), np.float32)
            self._capsule_materials = np.zeros((capacity, 4), np.float32)
            self._capsule_material_ids = np.zeros(capacity, np.int32)
            self._capsule_transparent = np.zeros(capacity, bool)

        if base_count:
            self._capsule_segments[:base_count] = segments[base_indices]
            self._capsule_widths[:base_count] = widths[base_indices]
            np.take(
                self._source.tendon_rgba,
                ids[base_indices],
                axis=0,
                out=self._capsule_colors[:base_count],
                mode="clip",
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[base_indices]],
                axis=0,
                out=self._capsule_materials[:base_count],
                mode="clip",
            )
            self._capsule_material_ids[:base_count] = self._source.tendon_material[
                ids[base_indices]
            ]

        if len(actuator_indices):
            self._fill_actuator_palette(frame)
            start = base_count
            stop = start + len(actuator_indices)
            self._capsule_segments[start:stop] = segments[actuator_indices]
            self._capsule_widths[start:stop] = (
                widths[actuator_indices] * self._source.actuator_tendon_scale
            )
            np.take(
                self._actuator_palette,
                segment_actuators[actuator_indices],
                axis=0,
                out=self._capsule_colors[start:stop],
            )
            np.take(
                self._material_values,
                self._source.tendon_material[ids[actuator_indices]],
                axis=0,
                out=self._capsule_materials[start:stop],
                mode="clip",
            )
            self._capsule_material_ids[start:stop] = self._source.tendon_material[
                ids[actuator_indices]
            ]

        np.less(
            self._capsule_colors[:total, 3],
            1.0,
            out=self._capsule_transparent[:total],
        )

        update(
            self._capsule_segments[:total],
            self._capsule_widths[:total],
            self._capsule_colors[:total],
            self._capsule_materials[:total],
            self._capsule_material_ids[:total],
            self._capsule_transparent[:total],
            self._tendon_material_table,
        )

    def _fill_actuator_palette(self, frame: SceneFrame) -> None:

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

    def set_camera(self, camera: CameraView) -> None:
        self._camera = camera

    def resize(self, width: int, height: int) -> None:
        if (width, height) == self.target.size:
            return
        self.target.release()
        self.target = RenderTarget(self.ctx, width, height, self.target.samples)
        self.caps = self._build_caps()

    def highlight(self, object_id: int) -> None:
        self._selected = int(object_id)

    def set_gizmo(self, gizmo: GizmoFrame | None) -> bool:
        if gizmo is not None and "gizmo" not in self._passes:
            return False
        self._gizmo = gizmo
        return True

    def configure_text(
        self,
        primary: str = "",
        primary_index: int = 0,
        fallback: str = "",
        fallback_index: int = 0,
        size_px: float = 14.0,
    ) -> None:
        debug = self._passes.get("debug")
        if debug is not None:
            debug.configure_text(primary, primary_index, fallback, fallback_index, size_px)

    # ------------------------------------------------------------------
    def render(self, frame: SceneFrame | None = None) -> ViewportImage | None:

        if self._scene is None and self._builder is not None and frame is not None:
            self.update(frame)
        scene = self._scene
        if scene is None or not self.gl_caps.usable:
            return None

        t0 = time.perf_counter()
        with self.guard:
            if self._hot_reload:
                self.programs.reload_changed()
                if self.programs.generation != self._program_generation:
                    self.set_render_scene(scene)

            ctx = self._make_context(scene)
            self.instances.draw_calls = 0
            self.instances.upload(scene)

            for name in PASS_ORDER:
                p = self._passes.get(name)
                if p is None:
                    continue

                if not p.prepare(ctx):
                    continue
                with self.timing.scope(name):
                    p.execute(ctx)

            self.timing.collect()

            bind_default_framebuffer(self.ctx)

        self.stats.draw_calls = self.instances.draw_calls
        self.stats.instances = scene.count
        self.stats.buckets = scene.bucket_count()
        self.stats.triangles = self.instances.triangles(scene)
        self.stats.cpu_ms = self.timing.cpu_table()
        self.stats.gpu_ms = self.timing.gpu_table()
        self.stats.frame_cpu_ms = (time.perf_counter() - t0) * 1000.0
        present = self._passes.get("present")
        return getattr(present, "image", None)

    def _make_context(self, scene: RenderScene) -> PassContext:
        cam = self._camera.with_aspect(self.target.width / max(self.target.height, 1))
        view = cam.view_matrix()
        proj = cam.proj_matrix()
        return PassContext(
            ctx=self.ctx,
            target=self.target,
            scene=scene,
            camera=cam,
            view=view,
            proj=proj,
            view_proj=(proj @ view).astype(np.float32),
            instances=self.instances,
            programs=self.programs,
            textures=self.textures,
            meshes=self._bucket_meshes,
            timing=self.timing,
            flags=self._flags,
            debug_view=self._debug_view,
            debug=self.debug,
            selected_id=self._selected,
            gizmo=self._gizmo,
            time=time.monotonic(),
            shadow=ShadowResult(),
        )

    # ------------------------------------------------------------------
    def pick(self, x: int, y: int) -> int:

        return self.target.read_id(int(x), int(y))

    def capture(
        self,
        path: Path,
        camera: CameraView | None = None,
        size: tuple[int, int] | None = None,
    ) -> bool:

        from PIL import Image

        saved_target = None
        saved_camera = self._camera
        try:
            if size is not None:
                w, h = int(size[0]), int(size[1])
                cap = self.gl_caps.max_texture_size or 16384
                if max(w, h) > cap:
                    log.error("Capture size {}x{} exceeds GL_MAX_TEXTURE_SIZE {}", w, h, cap)
                    return False
                saved_target, self.target = (
                    self.target,
                    RenderTarget(self.ctx, w, h, self.target.samples, self.target.id_layout),
                )
            if camera is not None:
                self._camera = camera
            if size is not None or camera is not None:
                self.render()
            img = self.target.read_color(flip=True)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img, "RGBA").save(Path(path))
            return True
        finally:
            self._camera = saved_camera
            if saved_target is not None:
                self.target.release()
                self.target = saved_target

    def set_flag(self, flag: RenderFlag, value: bool) -> bool:
        if flag not in self.caps.render_flags:
            return False
        self._flags[flag] = bool(value)
        if flag in {
            RenderFlag.STATIC,
            RenderFlag.SKIN,
            RenderFlag.FLEXFACE,
            RenderFlag.FLEXSKIN,
        }:
            self._sync_instance_visibility()
        return True

    def _sync_instance_visibility(self) -> None:
        if self._builder is None:
            return
        changed = self._builder.set_visual_options(
            static=self.get_flag(RenderFlag.STATIC),
            skin=self.get_flag(RenderFlag.SKIN),
            flex_face=self.get_flag(RenderFlag.FLEXFACE),
            flex_skin=self.get_flag(RenderFlag.FLEXSKIN),
        )
        if changed:
            self.set_render_scene(self._builder.scene)

    def get_flag(self, flag: RenderFlag) -> bool:
        return self._flags.get(flag, False)

    def set_debug_view(self, view: DebugView) -> bool:
        if view not in self.caps.debug_views:
            return False
        self._debug_view = view
        return True

    def get_debug_view(self) -> DebugView:

        return self._debug_view

    def set_label_mode(self, mode: LabelMode) -> bool:
        if mode not in self.caps.label_modes:
            return False
        self._label_mode = mode
        return True

    def get_label_mode(self) -> LabelMode:
        return self._label_mode

    def set_frame_mode(self, mode: FrameMode) -> bool:
        if mode not in self.caps.frame_modes:
            return False
        self._frame_mode = mode
        return True

    def get_frame_mode(self) -> FrameMode:
        return self._frame_mode

    @property
    def debug_view(self) -> DebugView:
        return self._debug_view

    def render_options(self) -> tuple[RenderFlag, ...]:
        return tuple(sorted(self.caps.render_flags, key=lambda f: f.value))

    def enable_hot_reload(self, on: bool = True) -> None:
        self._hot_reload = bool(on)

    def mesh_triangle_counts(self) -> dict[MeshKey, int]:
        return self.meshes.triangle_counts()

    def release(self) -> None:
        for p in self._passes.values():
            p.release()
        self.instances.release()
        self.meshes.release()
        self.textures.release()
        self.programs.release()
        self.timing.release()
        self.target.release()

    # ------------------------------------------------------------------
    def describe(self) -> str:

        c = self.gl_caps
        lines = [
            f"forge  GL {c.version}  ({c.renderer})",
            f"  core profile      : {c.core_profile}",
            f"  MSAA              : {self.target.samples}× (max {c.max_samples})",
            f"  ID buffer layout  : {self.target.id_layout} (samples={self.target.id_samples})",
            f"  instance strategy : {self.instances.strategy}",
            f"  GPU timing        : {c.timer_query}",
            f"  line width limit  : {c.line_width_max:g}",
            f"  loaded passes     : {', '.join(n for n in PASS_ORDER if n in self._passes)}",
        ]
        lines += [f"  · {n}" for n in c.notes]
        return "\n".join(lines)


def forge_context_caps(ctx: moderngl.Context) -> ContextCaps:
    from .context import probe

    return probe(ctx)


__all__ = [
    "PASS_ORDER",
    "ForgeBackend",
    "register_pass",
    "registered",
]
