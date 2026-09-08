"""Python scene adaptation for the C++ renderer; no graphics API leaks into callers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ...types import CameraView, ViewportImage
from ..backend import (
    BackendCaps,
    DebugView,
    FrameMode,
    LabelMode,
    RenderFlag,
    RenderStats,
    ShadowQuality,
)
from ..builder import SceneSourceBuilder
from ..mesh import builtin_mesh
from .device import acquire_device


class NativeTarget:
    """Readback retains the shared top/bottom orientation and NumPy contracts."""

    def __init__(self, backend, width, height, samples):
        self.backend = backend
        self.width, self.height, self.samples = width, height, samples
        self.handle = backend.runtime.create_target(width, height, samples, backend._scene_handle)
        self.frame = None

    def _read(self, product, flip, out=None):
        if self.frame is None:
            raise RuntimeError("Render a frame before requesting readback")
        result = self.backend.runtime.wait(self.backend.runtime.readback(self.frame, product))
        if result.state != self.backend.api.ReadbackState.READY:
            raise RuntimeError("Readback was canceled by a scene or target change")
        image = result.image
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

        return self.backend.device.readbacks.submit(complete)

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
        camera = self.backend._camera
        if camera.orthographic:
            return (depth - camera.near) / (camera.far - camera.near)
        return camera.far / (camera.far - camera.near) * (1 - camera.near / depth)

    def read_ids(self, flip=False):
        return self._read(self.backend.api.Product.OBJECT_ID, flip)

    def read_segmentation(self, flip=True, out=None):
        return self._read(self.backend.api.Product.SEGMENTATION, flip, out)


class NativeBackend:
    """Native candidate with explicit capabilities and independent scene ownership."""

    def __init__(self, width=640, height=480, samples=1):
        self.device = acquire_device()
        self.api, self.runtime = self.device.api, self.device.runtime
        self._closed = False
        self._camera = CameraView()
        self._builder = SceneSourceBuilder()
        self._revision = 0
        self._sequence = 0
        self._uploaded = None
        self._source = None
        self._mesh_indices = {}
        self._style = self.api.SceneStyle()
        self._tint_colors = np.empty((0, 4), np.float32)
        self._selected = 0
        self._fill = True
        self._scene_handle = None
        try:
            self._scene_handle = self.runtime.create_scene(self.api.SceneSource())
            self.target = NativeTarget(self, max(1, width), max(1, height), max(1, samples))
        except Exception:
            self.release()
            raise
        self.debug = None
        self.stats = RenderStats()
        self._flags = dict.fromkeys(
            (RenderFlag.STATIC, RenderFlag.SKIN, RenderFlag.FLEXSKIN, RenderFlag.TEXTURE), True
        )
        self.caps = BackendCaps(
            name="bgfx",
            renderer=self.runtime.capabilities.device,
            gpu_pick=True,
            capture=True,
            orthographic=True,
            msaa_samples=max(1, samples),
            render_flags=frozenset(
                (
                    RenderFlag.STATIC,
                    RenderFlag.SKIN,
                    RenderFlag.FLEXSKIN,
                    RenderFlag.FLEXFACE,
                    RenderFlag.CONVEXHULL,
                    RenderFlag.TEXTURE,
                )
            ),
            debug_views=frozenset((DebugView.SHADED,)),
            notes=(
                "Native preview: production lighting, shadows and diagnostic overlays are pending",
            ),
        )

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Native renderer is closed")

    def set_scene(self, source):
        self._require_open()
        self._source = source
        self._builder.set_source(source, self._camera)
        self._uploaded = None
        self._sync_scene()

    def _sync_scene(self):
        scene = self._builder.scene
        key = (scene.structure_revision, scene.identity_revision)
        if key != self._uploaded:
            source = self.api.SceneSource()
            self._revision += 1
            source.revision = self._revision
            source.linear_colors = scene.shading_model.value == "linear"
            keys = list(dict.fromkeys(key for key, _ in scene.bucket_keys))
            self._mesh_indices = {key: index for index, key in enumerate(keys)}
            for mesh_key in keys:
                mesh = self._source.meshes.get(mesh_key) if self._source else None
                if mesh is None:
                    mesh = builtin_mesh(mesh_key)
                index = source.add_mesh(mesh.positions, mesh.normals, mesh.indices)
                source.set_mesh_texcoords(index, mesh.uvs)
            slots = np.array(
                [self._mesh_indices[scene.bucket_keys[int(b)][0]] for b in scene.bucket], np.uint32
            )
            textures = {}
            for name, texture in (self._source.textures if self._source else {}).items():
                if texture.type.value != "2d":
                    continue
                pixels = texture.pixels
                rgba = np.full((*pixels.shape[:2], 4), 255, np.uint8)
                rgba[..., : pixels.shape[2]] = pixels
                from PIL import Image

                image = Image.fromarray(rgba)
                levels = [rgba.reshape(-1)]
                while image.width > 1 or image.height > 1:
                    image = image.resize(
                        (max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.BOX
                    )
                    levels.append(np.asarray(image).reshape(-1))
                textures[name] = source.add_texture_mips(
                    rgba.shape[1], rgba.shape[0], np.concatenate(levels)
                )
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
            self.runtime.set_scene(source, self._scene_handle)
            self._uploaded = key
        colors = scene.colors
        if self._selected and self._fill:
            if self._tint_colors.shape != colors.shape:
                self._tint_colors = np.empty_like(colors)
            np.copyto(self._tint_colors, colors)
            mask = scene.object_id == self._selected
            self._tint_colors[mask, :3] = colors[mask, :3] * 0.65 + np.array(
                [0.35, 0.2275, 0.035], np.float32
            )
            colors = self._tint_colors
        self._sequence += 1
        self.runtime.update_textured(
            self._scene_handle,
            scene.transforms,
            scene.tex_coef,
            colors,
            self._revision,
            self._sequence,
        )

    def update(self, frame):
        self._require_open()
        if self._source is None:
            return
        self._builder.update(frame, self._camera)
        self._sync_scene()
        for key, mesh in (frame.mesh_updates or {}).items():
            index = self._mesh_indices.get(key)
            if index is not None:
                self.runtime.update_mesh(index, mesh.positions, mesh.normals, self._scene_handle)

    def set_camera(self, camera):
        self._camera = camera

    def set_background(self, rgba):
        self._style.background = tuple(rgba)
        self.runtime.configure(self._scene_handle, self._style)

    def set_transparent_id_rendering(self, enabled):
        self._style.transparent_ids = bool(enabled)
        self.runtime.configure(self._scene_handle, self._style)

    def render(self, frame=None, request=None):
        self._require_open()
        started = time.perf_counter()
        if frame is not None:
            self.update(frame)
        camera = self.api.CameraView()
        camera.view = self._camera.view_matrix()
        camera.projection = self._camera.proj_matrix()
        camera.far_plane = self._camera.far
        camera.revision = self._sequence
        self.target.frame = self.runtime.render(self.target.handle, camera)
        self.stats.instances = self._builder.scene.count
        self.stats.draw_calls = 2 * self._builder.scene.bucket_count()
        self.stats.frame_cpu_ms = (time.perf_counter() - started) * 1000
        texture = self.runtime.target_texture(self.target.handle)
        return ViewportImage(texture.id, self.target.width, self.target.height, False, texture)

    def resize(self, width, height):
        self._require_open()
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) != (self.target.width, self.target.height):
            self.runtime.resize(self.target.handle, width, height)
            self.target.width, self.target.height = width, height
            self.target.frame = None

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
        if (self._selected, self._fill) != (object_id, fill):
            self._selected, self._fill = object_id, fill
            if self._source is not None:
                self._sync_scene()

    def set_gizmo(self, gizmo):
        return False

    def set_flag(self, flag, value):
        flag = RenderFlag(flag)
        if flag not in self.caps.render_flags:
            return False
        self._flags[flag] = bool(value)
        if flag == RenderFlag.TEXTURE:
            self._style.textures = bool(value)
            self.runtime.configure(self._scene_handle, self._style)
            return True
        if self._source is not None:
            self._builder.set_visual_options(
                static=self.get_flag(RenderFlag.STATIC),
                skin=self.get_flag(RenderFlag.SKIN),
                flex_face=self.get_flag(RenderFlag.FLEXFACE),
                flex_skin=self.get_flag(RenderFlag.FLEXSKIN),
                convex_hull=self.get_flag(RenderFlag.CONVEXHULL),
            )
            self._sync_scene()
        return True

    def get_flag(self, flag):
        return self._flags.get(flag, False)

    def set_shadow_quality(self, quality):
        return False

    def get_shadow_quality(self):
        return ShadowQuality.BALANCED

    def set_debug_view(self, view):
        return DebugView(view) == DebugView.SHADED

    def get_debug_view(self):
        return DebugView.SHADED

    def set_label_mode(self, mode):
        return LabelMode(mode) == LabelMode.NONE

    def get_label_mode(self):
        return LabelMode.NONE

    def set_frame_mode(self, mode):
        return FrameMode(mode) == FrameMode.NONE

    def get_frame_mode(self):
        return FrameMode.NONE

    def set_bvh_depth(self, depth):
        return False

    def get_bvh_depth(self):
        return 0

    def render_options(self):
        return tuple(flag for flag in RenderFlag if flag in self.caps.render_flags)

    def create_peer(self, width, height):
        return NativeBackend(width, height, self.target.samples)

    def capture(self, path, camera=None, size=None):
        from PIL import Image

        self._require_open()
        width, height = size or (self.target.width, self.target.height)
        view = (camera or self._camera).with_aspect(width / height)
        native = self.api.CameraView()
        native.view, native.projection = view.view_matrix(), view.proj_matrix()
        native.far_plane = view.far
        target = self.runtime.create_target(width, height, self.target.samples, self._scene_handle)
        try:
            frame = self.runtime.render(target, native)
            result = self.runtime.wait(self.runtime.readback(frame, self.api.Product.COLOR))
            if result.state != self.api.ReadbackState.READY:
                raise RuntimeError("Capture was canceled by a scene change")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.image).save(path)
            return True
        finally:
            self.runtime.destroy(target)

    def describe(self):
        return f"bgfx / {self.runtime.capabilities.backend} / C++ render owner"

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._scene_handle is not None:
                self.runtime.destroy_scene(self._scene_handle)
        finally:
            self.device.release()
