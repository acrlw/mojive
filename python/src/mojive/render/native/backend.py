"""Python scene adaptation for the C++ renderer; no graphics API leaks into callers."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...types import CameraView, LightType, MeshKey, MeshShape, TextureType, ViewportImage
from ..backend import (
    BackendCaps,
    DebugView,
    FrameMode,
    LabelMode,
    RenderFlag,
    RenderProduct,
    RenderRequest,
    RenderStats,
    ShadowQuality,
)
from ..builder import SceneSourceBuilder
from ..debugdraw import PRIMITIVE_MESH, DebugDraw, DrawPath, Occlusion
from ..dependencies import lights_key
from ..gizmo_plan import _MESHES, GizmoPlanner
from ..mesh import builtin_mesh, gizmo_mesh
from ..overlay import OverlayPublisher, OverlayState
from ..tendon import TendonPublisher, TendonScene
from ..text import TextLayout
from ..texture import srgb_to_linear_u8
from .device import acquire_device

_NO_VISUALS = np.empty((0, 4), np.float32)
_NO_VISUALS.flags.writeable = False

_DRAW_PATHS = {path: index for index, path in enumerate(DrawPath)}
_OCCLUSIONS = {mode: index for index, mode in enumerate(Occlusion)}


class NativeTarget:
    """Readback retains the shared top/bottom orientation and NumPy contracts."""

    def __init__(self, backend, width, height, samples):
        self.backend = backend
        self.width, self.height, self.samples = width, height, samples
        self.handle = backend.runtime.create_target(width, height, samples, backend._scene_handle)
        self.texture = backend.runtime.target_texture(self.handle)
        self.frame = None
        self._depth_camera = None

    def _read(self, product, flip, out=None):
        if self.frame is None:
            raise RuntimeError("Render a frame before requesting readback")
        dtype, channels = self.backend._read_specs[product]
        shape = (self.height, self.width, *channels)
        if out is not None and (out.shape != shape or out.dtype != dtype):
            raise ValueError(f"Expected {dtype} destination with shape {shape}")
        if out is not None and not out.flags.writeable:
            raise ValueError("Readback destination is read-only")
        image = (
            out
            if out is not None and flip and out.flags.c_contiguous
            else np.empty(shape, dtype=dtype)
        )
        state = self.backend.runtime.read_into(self.frame, product, image)
        if state != self.backend.api.ReadbackState.READY:
            raise RuntimeError("Readback was canceled by a scene or target change")
        if image is out:
            return out
        return self._deliver(image, flip, out)

    @staticmethod
    def _deliver(image, flip, out):
        if not flip:
            image = image[::-1].copy()
        if out is None:
            return image
        if out.shape != image.shape or out.dtype != image.dtype:
            raise ValueError(f"Expected {image.dtype} destination with shape {image.shape}")
        np.copyto(out, image)
        return out

    def _read_async(self, product, flip, out):
        if self.frame is None:
            raise RuntimeError("Render a frame before requesting readback")
        runtime = self.backend.runtime
        api = self.backend.api
        # Copy commands capture this frame before the next submission can replace it.
        ticket = runtime.readback(self.frame, product)

        def complete():
            result = runtime.wait(ticket)
            if result.state != api.ReadbackState.READY:
                raise RuntimeError("Readback was canceled by a scene or target change")
            image = result.image
            if out is not None:
                np.copyto(out, image if flip else image[::-1], casting="unsafe")
                return out
            return self._deliver(image, flip, None)

        future = self.backend.device.readbacks.submit(complete)
        self.backend._pending_readbacks.add(future)
        future.add_done_callback(self.backend._pending_readbacks.discard)
        return future

    def read_rgb_async(self, flip=True, out=None):
        return self._read_async(self.backend.api.Product.COLOR, flip, out)

    def read_metric_depth_async(self, flip=True, out=None):
        return self._read_async(self.backend.api.Product.METRIC_DEPTH, flip, out)

    def read_segmentation_async(self, flip=True, out=None):
        return self._read_async(self.backend.api.Product.SEGMENTATION, flip, out)

    def read_rgb(self, flip=True, out=None):
        return self._read(self.backend.api.Product.COLOR, flip, out)

    def read_color(self, flip=True):
        return self._read(self.backend.api.Product.RGBA, flip)

    def read_metric_depth(self, flip=True, out=None):
        return self._read(self.backend.api.Product.METRIC_DEPTH, flip, out)

    def read_depth(self, flip=True):
        depth = self.read_metric_depth(flip)
        near, far, orthographic = self._depth_camera
        if orthographic:
            return (depth - near) / (far - near)
        return far / (far - near) * (1 - near / depth)

    def read_ids(self, flip=False):
        return self._read(self.backend.api.Product.OBJECT_ID, flip)

    def read_segmentation(self, flip=True, out=None):
        return self._read(self.backend.api.Product.SEGMENTATION, flip, out)


class NativeBackend:
    """Native rendering with explicit capabilities and independent scene ownership."""

    def __init__(self, width=640, height=480, samples=1):
        # Match the portable color target policy; MuJoCo may request arbitrary
        # counts (for example 24). Data products remain single-sampled.
        samples = max(value for value in (1, 2, 4, 8) if value <= max(1, samples))
        self.device = acquire_device()
        self.api, self.runtime = self.device.api, self.device.runtime
        self._read_specs = {
            self.api.Product.COLOR: (np.dtype("uint8"), (3,)),
            self.api.Product.RGBA: (np.dtype("uint8"), (4,)),
            self.api.Product.OBJECT_ID: (np.dtype("uint32"), ()),
            self.api.Product.SEGMENTATION: (np.dtype("int32"), (2,)),
            self.api.Product.METRIC_DEPTH: (np.dtype("float32"), ()),
        }
        self._data_products = {
            RenderProduct.OBJECT_ID: self.api.Product.OBJECT_ID,
            RenderProduct.SEGMENTATION: self.api.Product.SEGMENTATION,
            RenderProduct.METRIC_DEPTH: self.api.Product.METRIC_DEPTH,
        }
        self._closed = False
        self._hot_reload = False
        self._camera = CameraView()
        self._gizmo = None
        self._gizmo_planner = GizmoPlanner()
        self._gizmo_meshes = {}
        self._builder = SceneSourceBuilder()
        self._scene = self._builder.scene
        self._external_scene = False
        self._triangle_count = 0
        self._revision = 0
        self._sequence = 0
        self._uploaded = None
        self._uploaded_visuals = None
        self._last_frame = None
        self._render_state_dirty = False
        self._source = None
        self._mesh_indices = {}
        self._style = self.api.SceneStyle()
        self._style.transparent_ids = False
        self._style.cull_face = True
        self._lighting_source = None
        self._textures = {}
        self._debug_view = DebugView.SHADED
        self._selected = self._outlined = 0
        self._scene_handle = None
        self._pending_readbacks = set()
        try:
            self._scene_handle = self.runtime.create_scene(self.api.SceneSource())
            self.target = NativeTarget(self, max(1, width), max(1, height), max(1, samples))
        except Exception:
            self.release()
            raise
        self.runtime.configure(self._scene_handle, self._style)
        self.debug = DebugDraw()
        self._text = TextLayout()
        self._glyph_atlas = None
        self._debug_meshes = {}
        self._label_mode, self._frame_mode, self._bvh_depth = LabelMode.NONE, FrameMode.NONE, 0
        self.stats = RenderStats()
        self._flags = dict.fromkeys(
            (
                RenderFlag.STATIC,
                RenderFlag.SKIN,
                RenderFlag.FLEXSKIN,
                RenderFlag.FLEXEDGE,
                RenderFlag.TEXTURE,
                RenderFlag.CULL_FACE,
                RenderFlag.TRANSPARENT,
                RenderFlag.TONEMAP,
                RenderFlag.HAZE,
                RenderFlag.MSAA,
                RenderFlag.SHADOW,
                RenderFlag.SKYBOX,
                RenderFlag.REFLECTION,
                RenderFlag.OUTLINE,
            ),
            True,
        )
        self._overlay = OverlayPublisher(self.debug, self._flags)
        self._tendons = TendonScene()
        self._tendon_publisher = TendonPublisher(self._tendons, self._overlay, self.get_flag)
        self._surface_meshes = {}
        self._surface_records = np.zeros((0, 32), np.float32)
        self._flags[RenderFlag.TENDON] = True
        self.caps = BackendCaps(
            name="bgfx",
            renderer=self.runtime.capabilities.device,
            gpu_pick=True,
            shadows=True,
            outline=True,
            gizmo=True,
            debug_draw=True,
            pass_timing=True,
            label_modes=frozenset(LabelMode),
            frame_modes=frozenset(FrameMode),
            capture=True,
            orthographic=True,
            msaa_samples=self.target.samples,
            render_flags=frozenset(RenderFlag),
            debug_views=frozenset(DebugView),
            notes=(
                "Object ID and metric depth use single-sampled scene geometry",
                "Color MSAA rounds down to 1x, 2x, 4x, or 8x",
            ),
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Native renderer is closed")

    def set_scene(self, source):
        self._require_open()
        self._source = source
        self._last_frame = None
        self._uploaded_visuals = None
        self._overlay.set_scene(source)
        self._tendon_publisher.set_scene(source)
        self._tendons.clear()
        self._builder.set_source(source, self._camera)
        self._uploaded = None
        self._lighting_source = None
        self._sync_scene()

    def set_render_scene(self, scene):
        self._require_open()
        # Unmanaged RenderScene callers do not advance upload revisions.
        if scene is not self._scene or not scene.structure_revision:
            self._uploaded = None
        self._sync_scene(scene)

    def _sync_scene(self, scene=None):
        self._external_scene = scene is not None
        scene = self._builder.scene if scene is None else scene
        self._scene = scene
        structure_key = (scene.structure_revision, scene.identity_revision)
        if structure_key != self._uploaded:
            source = self.api.SceneSource()
            self._revision += 1
            source.revision = self._revision
            source.infinite_planes = scene.infinite_planes
            source.planar_kinds = [
                {MeshShape.PLANE: 1, MeshShape.BOX: 2}.get(scene.bucket_keys[int(b)][0].shape, 0)
                for b in scene.bucket
            ]
            source.linear_colors = scene.shading_model.value == "linear"
            source.extent, source.shadow_clip, source.center = (
                scene.scene_extent,
                scene.shadow_clip,
                tuple(scene.scene_center),
            )
            keys = list(dict.fromkeys(key for key, _ in scene.bucket_keys))
            self._mesh_indices = {key: index for index, key in enumerate(keys)}
            triangle_counts = {}
            for mesh_key in keys:
                mesh = self._source.meshes.get(mesh_key) if self._source else None
                if mesh is None:
                    mesh = builtin_mesh(mesh_key)
                triangle_counts[mesh_key] = len(mesh.indices) // 3
                index = source.add_mesh(mesh.positions, mesh.normals, mesh.indices)
                source.set_mesh_texcoords(index, mesh.uvs)
            self._triangle_count = scene.triangle_count(triangle_counts)
            self._gizmo_meshes = {}
            for name in _MESHES:
                mesh = (
                    builtin_mesh(
                        MeshKey(MeshShape.DISK if name == "trackball" else MeshShape.SPHERE)
                    )
                    if name in {"trackball", "center"}
                    else gizmo_mesh(name)
                )
                self._gizmo_meshes[name] = source.add_mesh(
                    mesh.positions, mesh.normals, mesh.indices
                )
            self._surface_meshes = {}
            for shape in (MeshShape.CAPSULE_SHAFT, MeshShape.CAPSULE_CAP):
                key = MeshKey(shape)
                mesh = builtin_mesh(key)
                index = source.add_mesh(mesh.positions, mesh.normals, mesh.indices)
                source.set_mesh_texcoords(index, mesh.uvs)
                self._surface_meshes[key] = index
            self._debug_meshes = {}
            for key in PRIMITIVE_MESH.values():
                mesh = builtin_mesh(key)
                self._debug_meshes[key] = source.add_mesh(
                    mesh.positions, mesh.normals, mesh.indices
                )
            slots = np.array(
                [self._mesh_indices[scene.bucket_keys[int(b)][0]] for b in scene.bucket], np.uint32
            )
            texture_items = list((self._source.textures if self._source else {}).items())

            def prepare(item):
                name, texture = item
                pixels, srgb = texture.pixels, texture.srgb
                if srgb and pixels.shape[-1] < 3:
                    # Match the R8/RG8 linear fallback used by the other backends.
                    pixels = srgb_to_linear_u8(pixels)
                    srgb = False
                if texture.type is TextureType.TWO_D:
                    pixels = pixels[None]
                index = source.add_texture_pixels(
                    np.ascontiguousarray(pixels), texture.type is not TextureType.TWO_D, srgb
                )
                return name, index

            # The resizer releases the GIL; only the completed texture insertion
            # takes it again. Keep CPU preprocessing bounded and GPU submission on
            # its owner thread. Small texture sets avoid worker startup overhead.
            if len(texture_items) > 1 and sum(t.pixels.nbytes for _, t in texture_items) > 4 << 20:
                with ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="mojive-textures"
                ) as pool:
                    textures = dict(pool.map(prepare, texture_items))
            else:
                textures = dict(map(prepare, texture_items))
            self._textures = textures
            self._lighting_source = None
            materials = []
            for material in scene.materials:
                native = self.api.Material()
                native.texture = textures.get(material.texture, -1)
                native.emission, native.specular, native.shininess = (
                    material.emission,
                    material.specular,
                    material.shininess,
                )
                materials.append(native)
            source.materials = materials
            source.set_instances(slots, scene.object_id, scene.segmentation, scene.colors)
            source.set_material_indices(
                np.array([scene.bucket_keys[int(b)][1] for b in scene.bucket], np.uint32)
            )
            source.set_visuals(scene.material, scene.cube_coef)
            self.runtime.set_scene(source, self._scene_handle)
            self._uploaded = structure_key
            self._uploaded_visuals = None
        visual_key = (scene.pose_revision, scene.visual_revision)
        if visual_key != self._uploaded_visuals:
            attributes_changed = (
                self._uploaded_visuals is None or scene.visual_revision != self._uploaded_visuals[1]
            )
            self._sequence += 1
            self.runtime.update_visuals(
                self._scene_handle,
                scene.transforms,
                scene.tex_coef,
                scene.colors if attributes_changed else _NO_VISUALS,
                scene.material if attributes_changed else _NO_VISUALS,
                scene.cube_coef if attributes_changed else _NO_VISUALS,
                self._revision,
                self._sequence,
            )
            self._uploaded_visuals = visual_key
        self._sync_lighting()

    def _sync_lighting(self):
        lights = self._scene.lights
        key = lights_key(lights)
        if lights is self._lighting_source and key == self._lighting_key:
            return
        native = self.api.Lighting()
        native.enabled = True
        native.horizon_haze = lights.horizon_haze
        native.haze_slices = lights.horizon_haze_slices
        native.haze_density = lights.haze_density
        native.ambient = tuple(lights.ambient)
        native.fog = (
            lights.fog_start,
            lights.fog_end,
            0,
            0 if lights.horizon_haze else lights.haze_density,
        )
        native.fog_color, native.haze_color = tuple(lights.fog_color), tuple(lights.haze_color)
        if lights.headlight is not None and lights.headlight.active:
            native.headlight_diffuse = (*lights.headlight.diffuse, 1)
            native.headlight_specular = tuple(lights.headlight.specular)
        items = []
        for light in lights.lights:
            if not light.active:
                continue
            if light.type is LightType.IMAGE:
                native.image_texture = self._textures.get(light.texture, -1)
                native.image_intensity = max(light.intensity, 0) if native.image_texture >= 0 else 0
                continue
            row = self.api.Light()
            for field in ("position", "direction", "diffuse", "specular", "attenuation"):
                setattr(row, field, tuple(getattr(light, field)))
            row.type, row.cutoff, row.exponent, row.range = (
                int(light.type),
                light.cutoff,
                light.exponent,
                light.range,
            )
            row.radius, row.cast_shadow = light.area_radius, light.cast_shadow
            items.append(row)
        native.lights = items[:100]
        native.skybox_texture = self._textures.get(self._source.skybox, -1) if self._source else -1
        self.runtime.set_lighting(self._scene_handle, native)
        self._lighting_source = lights
        self._lighting_key = key

    def update(self, frame):
        self._require_open()
        if self._source is None:
            return
        self._last_frame = frame
        self._render_state_dirty = False
        self._builder.update(
            frame, self._camera, frame.island_rgba if self.get_flag(RenderFlag.ISLAND) else None
        )
        self._tendon_publisher.update(frame)
        self._overlay.publish(
            frame,
            OverlayState(
                self._camera,
                self.target.height,
                self._selected,
                self._label_mode,
                self._frame_mode,
                self._bvh_depth,
            ),
        )
        self._sync_scene()
        for key, mesh in (frame.mesh_updates or {}).items():
            index = self._mesh_indices.get(key)
            if index is not None:
                self.runtime.update_mesh(index, mesh.positions, mesh.normals, self._scene_handle)

    def _sync_overlays(self):
        packet = self.api.OverlayFrame()
        camera = self._camera
        view, proj = camera.view_matrix(), camera.proj_matrix()
        draws = []
        for name, (compare, write, cull), model, color, mask in self._gizmo_planner.plan(
            self._gizmo, camera, proj @ view, proj, self.target.height
        ):
            row = self.api.OverlayDraw()
            row.mesh, row.transform, row.color = self._gizmo_meshes[name], model, tuple(color)
            row.mask_radius = mask
            row.depth_test, row.depth_write, row.cull_face = (
                compare != "always",
                write,
                cull == "back",
            )
            draws.append(row)
        packet.gizmos = draws
        surface = self._tendons.scene
        count = surface.count
        if count:
            if len(self._surface_records) < count:
                self._surface_records = np.zeros(
                    (max(count, 2 * len(self._surface_records)), 32), np.float32
                )
            records = self._surface_records[:count]
            records[:, :12] = surface.transforms[:, :3].reshape(count, 12)
            records[:, 12:16], records[:, 16:20] = surface.colors, surface.tex_coef
            records[:, 20:24], records[:, 24:28] = surface.material, surface.cube_coef
            packet.set_surfaces(records)
            batches = []

            def batch(bucket, start, stop, transparent):
                key, material_id = surface.bucket_keys[bucket]
                row = self.api.SurfaceBatch()
                row.mesh, row.start, row.count = self._surface_meshes[key], start, stop - start
                row.texture = self._textures.get(surface.materials[material_id].texture, -1)
                row.transparent = transparent
                batches.append(row)

            for bucket in surface.opaque_buckets:
                batch(bucket, *surface.bucket_ranges[bucket], False)
            for bucket in surface.transparent_draw_order(camera.eye):
                batch(bucket, *surface.bucket_ranges[bucket], True)
            packet.surface_batches = batches

        def pack(frame):
            batches = []
            for path in DrawPath:
                if frame.counts[path]:
                    packet.set_stream(_DRAW_PATHS[path], frame.stream(path))
            for batch in frame.active():
                row = self.api.DebugBatch()
                row.path, row.occlusion = (
                    _DRAW_PATHS[batch.path],
                    _OCCLUSIONS[batch.occlusion],
                )
                row.start, row.count = batch.start, batch.count
                if batch.mesh is not None:
                    row.mesh = self._debug_meshes[batch.mesh]
                batches.append(row)
            if self._text.prepare(frame.texts, frame.text_count):
                if self._text.atlas_dirty or self._glyph_atlas is None:
                    width, height = self._text.atlas_size
                    mask = np.frombuffer(self._text.pixels, np.uint8).reshape(height, width)
                    rgba = np.repeat(mask[..., None], 4, axis=2)
                    replacement = self.runtime.upload_texture(rgba)
                    if self._glyph_atlas is not None:
                        self.runtime.destroy_texture(self._glyph_atlas)
                    self._glyph_atlas = replacement
                    self._text.mark_uploaded()
                packet.glyph_atlas = self._glyph_atlas
                packet.set_stream(8, self._text.records[: self._text.count])
                for batch in self._text.batches():
                    row = self.api.DebugBatch()
                    row.path, row.occlusion = 8, _OCCLUSIONS[batch.occlusion]
                    row.start, row.count = batch.start, batch.count
                    batches.append(row)
            packet.debug = batches

        self.debug.render_frame(pack)
        self.runtime.set_overlays(self._scene_handle, packet)

    def set_camera(self, camera):
        self._camera = camera.with_aspect(self.target.width / self.target.height)
        self._render_state_dirty = True

    def set_background(self, rgba):
        self._style.background = tuple(rgba)
        self.runtime.configure(self._scene_handle, self._style)

    def set_transparent_id_rendering(self, enabled):
        self._style.transparent_ids = bool(enabled)
        self.runtime.configure(self._scene_handle, self._style)

    def render(self, frame=None, request=None):
        self._require_open()
        if self._hot_reload:
            self.device.shaders.check()
        started = time.perf_counter()
        if frame is not None:
            self.update(frame)
        elif self._render_state_dirty and self._last_frame is not None:
            self.update(self._last_frame)
        if self._external_scene:
            self._sync_scene(self._scene)
        elif frame is None:
            self._sync_lighting()
        camera = self.api.CameraView()
        camera.view = self._camera.view_matrix()
        camera.projection = self._camera.proj_matrix()
        camera.far_plane = self._camera.far
        camera.near_plane = self._camera.near
        camera.focus = tuple(self._camera.target)
        camera.revision = self._sequence
        request = request or RenderRequest.viewport()
        color = request.needs(RenderProduct.COLOR)
        if color:
            self._sync_overlays()
        self.target.frame = self.runtime.render(
            self.target.handle,
            camera,
            color,
            bool(request.products & ~RenderProduct.COLOR),
            self._data_products.get(request.products & ~RenderProduct.COLOR),
        )
        self.target._depth_camera = (camera.near_plane, camera.far_plane, self._camera.orthographic)
        self.stats.instances = self._scene.count
        self.stats.triangles = self._triangle_count
        statistics = self.target.frame.statistics
        self.stats.cpu_ms = statistics.cpu_ms
        self.stats.gpu_ms = statistics.gpu_pass_ms
        if self.stats.gpu_ms and not self.caps.gpu_timing:
            self.caps = replace(self.caps, gpu_timing=True)
        self.stats.notes["timing"] = "Native pass submission CPU time; delayed GPU samples"
        self.stats.notes["cpu submission"] = statistics.cpu_submission
        self.stats.notes["gpu submission"] = statistics.gpu_submission
        self.stats.draw_calls = statistics.draw_calls
        for name in ("shadow", "reflection"):
            self.stats.notes[f"{name} cache"] = (
                "rendered"
                if getattr(statistics, f"{name}_rendered")
                else "reused"
                if getattr(statistics, f"{name}_reused")
                else "disabled"
            )
        self.stats.notes["culled instances"] = statistics.culled_instances
        self.stats.notes["shadow instances"] = statistics.shadow_instances
        self.stats.notes["culled shadow instances"] = statistics.culled_shadow_instances
        self.stats.buckets = self._scene.bucket_count()
        self.stats.frame_cpu_ms = (time.perf_counter() - started) * 1000
        texture = self.target.texture
        return ViewportImage(texture.id, self.target.width, self.target.height, False, texture)

    def resize(self, width, height):
        self._require_open()
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) != (self.target.width, self.target.height):
            self.runtime.resize(self.target.handle, width, height)
            self.target.width, self.target.height = width, height
            self.target.frame = None
            self.set_camera(self._camera)

    def pick(self, x, y):
        if self.target.frame is None or not (
            0 <= x < self.target.width and 0 <= y < self.target.height
        ):
            return 0
        region = self.api.Region()
        region.x, region.y = int(x), self.target.height - 1 - int(y)
        region.width = region.height = 1
        result = self.runtime.wait(
            self.runtime.readback(self.target.frame, self.api.Product.OBJECT_ID, region)
        )
        return int(result.image[0, 0]) if result.state == self.api.ReadbackState.READY else 0

    def highlight(self, object_id, *, xray=False, fill=True, outline=True):
        object_id = int(object_id)
        self._selected = object_id if fill else 0
        self._outlined = object_id if outline else 0
        self._style.selected_id, self._style.selection_fill = object_id, fill
        self._style.selection_outline, self._style.selection_xray = outline, xray
        self.runtime.configure(self._scene_handle, self._style)

    def set_gizmo(self, gizmo):
        self._gizmo = gizmo
        return True

    def set_flag(self, flag, value):
        flag = RenderFlag(flag)
        if flag not in self.caps.render_flags:
            return False
        self._flags[flag] = bool(value)
        self._render_state_dirty = True
        style_names = {
            RenderFlag.WIREFRAME: "wireframe",
            RenderFlag.TEXTURE: "textures",
            RenderFlag.CULL_FACE: "cull_face",
            RenderFlag.TRANSPARENT: "transparent",
            RenderFlag.ADDITIVE: "additive",
            RenderFlag.TONEMAP: "tonemap",
            RenderFlag.FOG: "fog",
            RenderFlag.HAZE: "haze",
            RenderFlag.MSAA: "msaa",
            RenderFlag.SHADOW: "shadows",
            RenderFlag.SKYBOX: "skybox",
            RenderFlag.REFLECTION: "reflections",
            RenderFlag.OUTLINE: "outline",
        }
        if flag in style_names:
            setattr(self._style, style_names[flag], bool(value))
            self.runtime.configure(self._scene_handle, self._style)
            return True
        if self._source is not None:
            self._builder.set_visual_options(
                static=self.get_flag(RenderFlag.STATIC),
                skin=self.get_flag(RenderFlag.SKIN),
                flex_face=self.get_flag(RenderFlag.FLEXFACE),
                flex_skin=self.get_flag(RenderFlag.FLEXSKIN),
                convex_hull=self.get_flag(RenderFlag.CONVEXHULL),
                island=self.get_flag(RenderFlag.ISLAND),
            )
            self._sync_scene()
        return True

    def get_flag(self, flag):
        return self._flags.get(flag, False)

    def set_shadow_quality(self, quality):
        try:
            quality = ShadowQuality(quality)
        except ValueError:
            return False
        self._style.shadow_quality = list(ShadowQuality).index(quality)
        self.runtime.configure(self._scene_handle, self._style)
        return True

    def get_shadow_quality(self):
        return list(ShadowQuality)[self._style.shadow_quality]

    def set_debug_view(self, view):
        view = DebugView(view)
        modes = {
            DebugView.SHADED: 0,
            DebugView.ALBEDO: 1,
            DebugView.NORMAL: 2,
            DebugView.DEPTH: 3,
            DebugView.OVERDRAW: 4,
            DebugView.WIREFRAME: 5,
            DebugView.SEGMENT: 6,
            DebugView.IDCOLOR: 7,
        }
        if view not in modes:
            return False
        self._debug_view = view
        self._style.debug_view = modes[view]
        self.runtime.configure(self._scene_handle, self._style)
        return True

    def get_debug_view(self):
        return self._debug_view

    def configure_text(
        self, primary="", primary_index=0, fallback="", fallback_index=0, size_px=14
    ):
        self._text.configure(primary, primary_index, fallback, fallback_index, size_px)

    def set_label_mode(self, mode):
        self._label_mode = LabelMode(mode)
        self._render_state_dirty = True
        return True

    def get_label_mode(self):
        return self._label_mode

    def set_frame_mode(self, mode):
        self._frame_mode = FrameMode(mode)
        self._render_state_dirty = True
        return True

    def get_frame_mode(self):
        return self._frame_mode

    def set_bvh_depth(self, depth):
        self._bvh_depth = max(int(depth), 0)
        self._render_state_dirty = True
        return True

    def get_bvh_depth(self):
        return self._bvh_depth

    def render_options(self):
        return tuple(flag for flag in RenderFlag if flag in self.caps.render_flags)

    def create_peer(self, width, height):
        peer = NativeBackend(width, height, self.target.samples)
        peer.enable_hot_reload(self._hot_reload)
        peer.set_shadow_quality(self.get_shadow_quality())
        return peer

    def enable_hot_reload(self, on=True):
        self._require_open()
        self._hot_reload = bool(on)
        if self._hot_reload:
            self.device.shaders.enable()

    def capture(self, path, camera=None, size=None):
        from PIL import Image

        self._require_open()
        width, height = size or (self.target.width, self.target.height)
        view = (camera or self._camera).with_aspect(width / height)
        previous_target, previous_camera = self.target, self._camera
        temporary = NativeTarget(self, width, height, self.target.samples)
        try:
            self.target = temporary
            self.set_camera(view)
            self.render(request=RenderRequest.color())
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(temporary.read_rgb()).save(path)
            return True
        finally:
            self.target = previous_target
            self.set_camera(previous_camera)
            self.runtime.destroy(temporary.handle)

    def describe(self):
        return f"bgfx / {self.runtime.capabilities.backend} / C++ render owner"

    def release(self):
        if self._closed:
            return
        self._closed = True
        if self.device.in_readback_worker():
            # A callback must not occupy a pool worker while waiting for queued
            # reads. The cleanup thread retains this scene's device reference.
            threading.Thread(target=self._finish_release, name="mojive-render-close").start()
        else:
            self._finish_release()

    def _finish_release(self):
        wait(tuple(self._pending_readbacks))
        try:
            if getattr(self, "_glyph_atlas", None) is not None:
                self.runtime.destroy_texture(self._glyph_atlas)
            if self._scene_handle is not None:
                self.runtime.destroy_scene(self._scene_handle)
        finally:
            self.device.release()
