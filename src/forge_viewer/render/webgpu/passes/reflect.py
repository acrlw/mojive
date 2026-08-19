"""Planar reflection render pass for the webgpu backend.

Port of ``render.forge.passes.reflect.ReflectPass`` (up to four planes detected
from negative-capable reflectance on PLANE/BOX shapes, deduped by plane
equation).  WebGPU translations:

- per-plane ``rgba16float`` color textures + one shared ``depth24plus``
  texture, single-sampled — forge's f2 color textures and shared depth FBO;
- the mirrored view (``view @ mirror``) flips triangle winding, so the scene
  pipeline variant for reflections uses ``front_face="cw"`` instead of
  ``gl.front_face``;
- ``gl_ClipDistance0`` becomes a fragment discard on the plane equation in the
  frame uniforms (``clip_plane``); the main pass binds the (0,0,0,1) no-op;
- reflection rendering writes linear color (``flags.y`` linear-out, forge's
  ``u_linear_out``); the main pass adds ``reflectance * reflected`` before
  atmosphere/tonemap;
- one frame-uniform buffer holds one block per plane at 512-byte stride
  (384-byte frame block + ``minUniformBufferOffsetAlignment``), because
  ``queue.write_buffer`` is submit-ordered and cannot retarget a single
  binding between passes inside one encoder.
"""

from __future__ import annotations

import numpy as np
import wgpu

from .... import math3d as M
from ....types import CameraView, MeshShape
from ...backend import RenderFlag
from ...scene import RenderScene
from ..instances import InstanceStore
from ..lighting import LIGHTS_BYTES, LightUniforms
from ..targets import FRAME_BYTES, FRAME_DTYPE, proj_matrix_wgpu

MAX_REFLECTION_PLANES = 4
REFLECT_STRIDE = 512  # FRAME_BYTES (384) rounded up to 256-byte binding alignment

REFLECT_FRAME_DTYPE = np.dtype(
    {
        "names": FRAME_DTYPE.names,
        "formats": [FRAME_DTYPE.fields[name][0] for name in FRAME_DTYPE.names],
        "offsets": [FRAME_DTYPE.fields[name][1] for name in FRAME_DTYPE.names],
        "itemsize": REFLECT_STRIDE,
    }
)


class _PlaneGroup:
    def __init__(self, plane, indices, buckets) -> None:
        self.plane = plane
        self.indices = indices
        self.buckets = buckets


class ReflectPass:
    """Renders the mirrored scene into one color texture per reflection plane."""

    name = "reflect"

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        # Group 4 of the scene pipeline: the four reflection colors + sampler.
        self.sample_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                }
                for i in range(MAX_REFLECTION_PLANES)
            ]
            + [
                {
                    "binding": MAX_REFLECTION_PLANES,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                }
            ]
        )
        # forge: filter LINEAR, repeat off (clamp to edge).
        self._sampler = device.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )
        self._uniforms = device.create_buffer(
            size=MAX_REFLECTION_PLANES * REFLECT_STRIDE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._arena = np.zeros(MAX_REFLECTION_PLANES, REFLECT_FRAME_DTYPE)

        self.colors: list[wgpu.GPUTexture] = []
        self._views: list[wgpu.GPUTextureView] = []
        self.depth: wgpu.GPUTexture | None = None
        self._depth_view: wgpu.GPUTextureView | None = None
        self._size = (0, 0)
        self._fallback: wgpu.GPUTextureView | None = None
        self._fallback_group: wgpu.GPUBindGroup | None = None

        self._groups: tuple[_PlaneGroup, ...] = ()
        self._mirrored_eyes = np.zeros((MAX_REFLECTION_PLANES, 3), np.float32)
        self._transparent = True
        self._additive = False
        self._encoded_scene = None
        self._encoded_reflectance: dict[int, float] = {}
        self._group0_cache: dict[tuple, wgpu.GPUBindGroup] = {}
        self._sample_groups: dict[tuple, wgpu.GPUBindGroup] = {}

    # -- plane detection (verbatim port of the forge classmethods) -------------

    @classmethod
    def find_planes(cls, scene: RenderScene) -> tuple[_PlaneGroup, ...]:
        if scene.count == 0 or len(scene.material) == 0 or not scene.bucket_keys:
            return ()
        reflectance = np.asarray(scene.material[:, 3])
        candidates = [
            int(index)
            for index in np.argsort(-reflectance)
            if reflectance[index] > 0.0
            and 0 <= int(scene.bucket[index]) < len(scene.bucket_keys)
            and scene.bucket_keys[int(scene.bucket[index])][0].shape
            in (MeshShape.PLANE, MeshShape.BOX)
        ]
        groups: list[_PlaneGroup] = []
        for index in candidates:
            plane = cls._plane_equation(scene, index)
            if plane is None:
                continue
            group = next(
                (
                    item
                    for item in groups
                    if np.allclose(item.plane, plane, atol=1e-5)
                    or np.allclose(item.plane, -np.asarray(plane), atol=1e-5)
                ),
                None,
            )
            if group is not None:
                group.indices.append(index)
                group.buckets.add(int(scene.bucket[index]))
            elif len(groups) < MAX_REFLECTION_PLANES:
                groups.append(_PlaneGroup(plane, [index], {int(scene.bucket[index])}))
        return groups

    @staticmethod
    def _plane_equation(scene: RenderScene, index: int) -> tuple[float, float, float, float] | None:
        transform = np.asarray(scene.transforms[index], np.float64)
        try:
            normal = np.linalg.inv(transform[:3, :3]).T @ np.array([0.0, 0.0, 1.0])
        except np.linalg.LinAlgError:
            return None
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            return None
        normal /= length
        shape = scene.bucket_keys[int(scene.bucket[index])][0].shape
        point = transform[:3, 3]
        if shape is MeshShape.BOX:
            point = point + transform[:3, 2]
        d = -float(np.dot(normal, point))
        return (float(normal[0]), float(normal[1]), float(normal[2]), d)

    # -- reflectance channel encoding (verbatim port) ---------------------------

    def _restore_reflectance(self, scene: RenderScene) -> None:
        if self._encoded_scene is scene:
            for index, value in self._encoded_reflectance.items():
                if index < len(scene.material):
                    scene.material[index, 3] = value
        self._encoded_scene = None
        self._encoded_reflectance.clear()

    def _encode_reflectance(self, scene: RenderScene, groups: tuple[_PlaneGroup, ...]) -> None:
        self._encoded_scene = scene
        for layer, group in enumerate(groups):
            for index in group.indices:
                value = float(scene.material[index, 3])
                shape = scene.bucket_keys[int(scene.bucket[index])][0].shape
                top_face = 1.0 if shape is MeshShape.BOX else 0.0
                self._encoded_reflectance[index] = value
                scene.material[index, 3] = -(4.0 * layer + 2.0 * top_face + value)

    # -- targets -----------------------------------------------------------------

    def _ensure_target(self, width: int, height: int, count: int) -> bool:
        if width <= 0 or height <= 0:
            return False
        if self._size == (width, height) and len(self._views) == count:
            return True
        self._release_targets()
        self.depth = self._device.create_texture(
            size=(width, height, 1),
            format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self._depth_view = self.depth.create_view()
        for _ in range(count):
            color = self._device.create_texture(
                size=(width, height, 1),
                format="rgba16float",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self.colors.append(color)
            self._views.append(color.create_view())
        self._size = (width, height)
        return True

    def _release_targets(self) -> None:
        for tex in self.colors:
            tex.destroy()
        if self.depth is not None:
            self.depth.destroy()
        self.colors = []
        self._views = []
        self.depth = None
        self._depth_view = None
        self._size = (0, 0)
        self._sample_groups.clear()

    def _fallback_view(self) -> wgpu.GPUTextureView:
        if self._fallback is None:
            tex = self._device.create_texture(
                size=(1, 1, 1), format="rgba16float", usage=wgpu.TextureUsage.TEXTURE_BINDING
            )
            self._fallback = tex.create_view()
        return self._fallback

    # -- per-frame scheduling ------------------------------------------------------

    def prepare(
        self,
        scene: RenderScene,
        camera: CameraView,
        flags: dict[RenderFlag, bool],
        width: int,
        height: int,
    ) -> bool:
        """Detect planes and encode reflectance; False means no pass this frame."""
        self._restore_reflectance(scene)
        self._groups = ()
        if not flags.get(RenderFlag.REFLECTION, True):
            return False
        eye = np.asarray(camera.eye, np.float64)
        groups = tuple(
            group
            for group in self.find_planes(scene)
            if float(np.dot(group.plane[:3], eye) + group.plane[3]) > 1e-4
        )
        if not groups:
            return False
        if not self._ensure_target(width, height, len(groups)):
            return False
        self._groups = groups
        self._transparent = flags.get(RenderFlag.TRANSPARENT, True)
        self._additive = flags.get(RenderFlag.ADDITIVE, False)
        self._encode_reflectance(scene, groups)
        return True

    def write_frames(
        self,
        camera: CameraView,
        light_count: int,
        image_light: tuple[float, float],
        pack,
    ) -> None:
        """Pack one mirrored frame block per plane into the uniform buffer."""
        eye = np.asarray(camera.eye, np.float64)
        view = np.asarray(camera.view_matrix(), np.float64)
        proj = np.asarray(proj_matrix_wgpu(camera), np.float64)
        arena = self._arena
        for k, group in enumerate(self._groups):
            normal = np.array(group.plane[:3], np.float64)
            point = -normal * group.plane[3]
            mirror = np.asarray(M.mirror(point, normal), np.float64)
            # forge ReflectPass._set_plane: the view matrix is mirrored; the
            # projection is unchanged.
            m_view = view @ mirror
            m_eye = (mirror @ np.append(eye, 1.0))[:3]
            self._mirrored_eyes[k] = m_eye
            pack(
                arena[k],
                camera,
                m_view.astype(np.float32),
                (proj @ m_view).astype(np.float32),
                m_eye,
                light_count,
                image_light,
                group.plane,
                (0.0, 0.0),
                1.0,
            )
        self._device.queue.write_buffer(self._uniforms, 0, arena[: len(self._groups)].tobytes())

    # -- bind groups -----------------------------------------------------------------

    def _group0(
        self, layout, plane: int, instances: InstanceStore, lights: LightUniforms
    ) -> wgpu.GPUBindGroup:
        key = (plane, id(instances.buffer), id(lights.buffer))
        group = self._group0_cache.get(key)
        if group is None:
            group = self._device.create_bind_group(
                layout=layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._uniforms,
                            "offset": plane * REFLECT_STRIDE,
                            "size": FRAME_BYTES,
                        },
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": instances.buffer,
                            "offset": 0,
                            "size": instances.capacity * 128,
                        },
                    },
                    {
                        "binding": 2,
                        "resource": {"buffer": lights.buffer, "offset": 0, "size": LIGHTS_BYTES},
                    },
                ],
            )
            self._group0_cache[key] = group
        return group

    def sample_group(self) -> wgpu.GPUBindGroup:
        """Main-pass group 4: live reflection colors, 1x1 fallback for idle slots."""
        views = [
            self._views[i] if i < len(self._views) else self._fallback_view()
            for i in range(MAX_REFLECTION_PLANES)
        ]
        key = tuple(id(view) for view in views)
        group = self._sample_groups.get(key)
        if group is None:
            group = self._device.create_bind_group(
                layout=self.sample_layout,
                entries=[{"binding": i, "resource": view} for i, view in enumerate(views)]
                + [{"binding": MAX_REFLECTION_PLANES, "resource": self._sampler}],
            )
            self._sample_groups[key] = group
        return group

    def fallback_group(self) -> wgpu.GPUBindGroup:
        """Group 4 bound inside the reflection pass itself (never sampled)."""
        if self._fallback_group is None:
            self._fallback_group = self._device.create_bind_group(
                layout=self.sample_layout,
                entries=[
                    {"binding": i, "resource": self._fallback_view()}
                    for i in range(MAX_REFLECTION_PLANES)
                ]
                + [{"binding": MAX_REFLECTION_PLANES, "resource": self._sampler}],
            )
        return self._fallback_group

    # -- rendering ----------------------------------------------------------------

    def execute(
        self,
        encoder: wgpu.GPUCommandEncoder,
        scene: RenderScene,
        group0_layout,
        instances: InstanceStore,
        lights: LightUniforms,
        draw_buckets,
    ) -> tuple[int, int]:
        """Encode one pass per plane; returns (draw calls, instances drawn)."""
        excluded = set().union(*(group.buckets for group in self._groups))
        calls = 0
        drawn = 0
        for k in range(len(self._groups)):
            plane_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._views[k],
                        "clear_value": (0.0, 0.0, 0.0, 1.0),
                        "load_op": "clear",
                        "store_op": "store",
                    }
                ],
                depth_stencil_attachment={
                    "view": self._depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
            )
            group0 = self._group0(group0_layout, k, instances, lights)
            opaque = tuple(b for b in scene.opaque_buckets if b not in excluded)
            c, d = draw_buckets(plane_pass, group0, opaque, "opaque")
            calls += c
            drawn += d
            if scene.transparent_buckets and self._transparent:
                blend = "additive" if self._additive else "alpha"
                order = tuple(
                    b
                    for b in scene.transparent_draw_order(self._mirrored_eyes[k])
                    if b not in excluded
                )
                c, d = draw_buckets(plane_pass, group0, order, blend)
                calls += c
                drawn += d
            plane_pass.end()
        return calls, drawn

    def release(self) -> None:
        self._release_targets()
        self._uniforms.destroy()
        self._groups = ()
        self._encoded_scene = None
        self._encoded_reflectance.clear()
        self._group0_cache.clear()
        self._sample_groups.clear()
        self._fallback = None
        self._fallback_group = None
