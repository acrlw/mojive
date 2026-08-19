"""Forge render-pass orchestration and frame submission."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import moderngl
import numpy as np

from ...adapters.base import SceneFrame, SceneSource
from ...gizmo import GizmoFrame
from ...log import get_logger
from ...types import CameraView, MeshKey, ViewportImage
from ..backend import BackendCaps, DebugView, FrameMode, LabelMode, RenderFlag, RenderStats
from ..debugdraw import DebugDraw
from ..overlay import OverlayPublisher, OverlayState
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
        self._background = (0.13, 0.14, 0.16, 1.0)
        self._selected = 0
        self._include_transparent_ids = False
        self._gizmo: GizmoFrame | None = None
        self._debug_view = DebugView.SHADED
        self._label_mode = LabelMode.NONE
        self._frame_mode = FrameMode.NONE
        self._flags: dict[RenderFlag, bool] = dict.fromkeys(self._supported_flags(), True)
        self._flags[RenderFlag.WIREFRAME] = False
        self._flags[RenderFlag.ADDITIVE] = False
        self._flags[RenderFlag.FOG] = False
        self._flags[RenderFlag.HAZE] = False
        self._flags[RenderFlag.CONTACTPOINT] = False
        self._flags[RenderFlag.CONTACTFORCE] = False
        self._flags[RenderFlag.CONTACTSPLIT] = False
        self._flags[RenderFlag.ISLAND] = False
        self._flags[RenderFlag.CONVEXHULL] = False
        self._flags[RenderFlag.AUTOCONNECT] = False
        self._flags[RenderFlag.ACTUATOR] = False
        self._flags[RenderFlag.ACTIVATION] = False
        self._flags[RenderFlag.JOINT] = False
        self._flags[RenderFlag.COM] = False
        self._flags[RenderFlag.INERTIA] = False
        self._flags[RenderFlag.SCLINERTIA] = False
        self._flags[RenderFlag.BODYBVH] = False
        self._flags[RenderFlag.MESHBVH] = False
        self._flags[RenderFlag.CAMERA] = False
        self._flags[RenderFlag.LIGHT] = False
        self._flags[RenderFlag.RANGEFINDER] = False
        self._flags[RenderFlag.CONSTRAINT] = False
        self._flags[RenderFlag.FLEXFACE] = False
        self._flags[RenderFlag.FLEXVERT] = False
        # MuJoCo mjv_defaultOption() enables tendon paths by default.
        self._flags[RenderFlag.TENDON] = True
        # Overlay publishing is backend-neutral; when the debug pass failed to
        # load the publisher writes into a private store nobody reads.
        self._overlay = OverlayPublisher(
            self.debug if self.debug is not None else DebugDraw(), self._flags
        )
        self._tendon_actuator = np.zeros(0, np.int32)
        self._capsule_segments = np.zeros((0, 2, 3), np.float32)
        self._capsule_widths = np.zeros(0, np.float32)
        self._capsule_colors = np.zeros((0, 4), np.float32)
        self._capsule_materials = np.zeros((0, 4), np.float32)
        self._capsule_material_ids = np.zeros(0, np.int32)
        self._capsule_transparent = np.zeros(0, bool)
        self._bucket_meshes: list = []
        self._bvh_depth = 0
        self._structure_generation = -1
        self._program_generation = -1
        self.caps = self._build_caps()
        self._hot_reload = False

    def _supported_flags(self) -> frozenset[RenderFlag]:
        flags = {
            RenderFlag.CULL_FACE,
            RenderFlag.TEXTURE,
            RenderFlag.TRANSPARENT,
            RenderFlag.ADDITIVE,
            RenderFlag.MSAA,
            RenderFlag.TONEMAP,
            RenderFlag.WIREFRAME,
            RenderFlag.FOG,
            RenderFlag.HAZE,
            RenderFlag.STATIC,
            RenderFlag.SKIN,
            RenderFlag.FLEXFACE,
            RenderFlag.FLEXSKIN,
            RenderFlag.ISLAND,
            RenderFlag.CONVEXHULL,
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
                RenderFlag.BODYBVH,
                RenderFlag.MESHBVH,
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
        self._island_tendon_material_table = tuple(
            replace(material, texture=None) for material in source.materials
        )
        self._tendon_actuator = np.full(len(source.tendon_rgba), -1, np.int32)
        for actuator, tendon in enumerate(source.actuator_tendon):
            if 0 <= tendon < len(self._tendon_actuator):
                self._tendon_actuator[tendon] = actuator
        self._overlay.set_scene(source)
        self.meshes.sync({**all_builtin(), **source.meshes})
        self.textures.sync(source.textures, source.skybox)
        self._builder = SceneSourceBuilder()
        self._builder.set_source(source, self._camera)
        self._sync_instance_visibility()
        self._structure_generation = -1

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
        island_rgba = frame.island_rgba if self.get_flag(RenderFlag.ISLAND) else None
        self.set_render_scene(self._builder.update(frame, self._camera, island_rgba))
        self._publish_tendons(frame)
        if self.debug is not None:
            self._overlay.publish(frame, self._overlay_state())

    def _overlay_state(self) -> OverlayState:
        return OverlayState(
            camera=self._camera,
            viewport_height=self.target.height,
            selected=self._selected,
            label_mode=self._label_mode,
            frame_mode=self._frame_mode,
            bvh_depth=self._bvh_depth,
        )

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
            tendon_rgba = self._source.tendon_rgba
            if self.get_flag(RenderFlag.ISLAND) and frame.tendon_island_rgba is not None:
                tendon_rgba = frame.tendon_island_rgba
            np.take(
                tendon_rgba,
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
            palette = self._overlay.fill_actuator_palette(frame)
            start = base_count
            stop = start + len(actuator_indices)
            self._capsule_segments[start:stop] = segments[actuator_indices]
            self._capsule_widths[start:stop] = (
                widths[actuator_indices] * self._source.actuator_tendon_scale
            )
            np.take(
                palette,
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

        material_table = (
            self._island_tendon_material_table
            if self.get_flag(RenderFlag.ISLAND)
            else self._tendon_material_table
        )
        update(
            self._capsule_segments[:total],
            self._capsule_widths[:total],
            self._capsule_colors[:total],
            self._capsule_materials[:total],
            self._capsule_material_ids[:total],
            self._capsule_transparent[:total],
            material_table,
        )

    def set_camera(self, camera: CameraView) -> None:
        self._camera = camera

    def set_background(self, rgba: tuple[float, float, float, float]) -> None:
        self._background = tuple(float(channel) for channel in rgba)

    def resize(self, width: int, height: int) -> None:
        if (width, height) == self.target.size:
            return
        self.target.release()
        self.target = RenderTarget(self.ctx, width, height, self.target.samples)
        self.caps = self._build_caps()

    def highlight(self, object_id: int) -> None:
        self._selected = int(object_id)

    def set_transparent_id_rendering(self, enabled: bool) -> None:
        self._include_transparent_ids = bool(enabled)

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
        self._update_light_stats(scene)
        present = self._passes.get("present")
        return getattr(present, "image", None)

    def _update_light_stats(self, scene: RenderScene) -> None:
        from .passes.base import schedule_lights

        schedule = schedule_lights(scene.lights)
        self.stats.notes = {
            "scene lights": (f"{len(schedule.lights)} active, {schedule.deferred_lights} deferred"),
            "shadow casters": (
                f"{schedule.selected_shadow_count} active, {schedule.deferred_shadows} deferred"
            ),
        }

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
            background=self._background,
            include_transparent_ids=self._include_transparent_ids,
            gizmo=self._gizmo,
            time=time.monotonic(),
            shadow=ShadowResult(),
        )

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
            RenderFlag.ISLAND,
            RenderFlag.CONVEXHULL,
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
            island=self.get_flag(RenderFlag.ISLAND),
            convex_hull=self.get_flag(RenderFlag.CONVEXHULL),
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

    def set_bvh_depth(self, depth: int) -> bool:
        self._bvh_depth = max(int(depth), 0)
        return True

    def get_bvh_depth(self) -> int:
        return self._bvh_depth

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
