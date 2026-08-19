"""Minimal wgpu-py scene backend spike for forge-viewer.

Consumes the renderer-neutral ``RenderScene`` contract (bucketed instances,
row-major transforms) exactly like ``ForgeBackend`` does, so this file alone
measures the real porting surface: mesh upload, per-instance storage, one
MRT geometry pass (color + linear depth + object id), and CPU readback.

Deliberately out of scope (noted in the spike report): textures, materials
table, transparency sorting, MSAA resolve, shadows, outline, skybox, tendons,
debug draw, gizmo.
"""

from __future__ import annotations

import numpy as np
import wgpu

from forge_viewer.render.scene import RenderScene

WGSL = """
struct Camera {
    view: mat4x4f,
    proj: mat4x4f,
    light_dir: vec4f,   // world-space direction the light travels
    light_diffuse: vec4f,
    light_specular: vec4f,
    ambient: vec4f,
    eye: vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct Instance {
    model: mat4x4f,
    color: vec4f,       // linear rgba
    object_id: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) normal: vec3f,
    @location(1) world: vec3f,
    @location(2) color: vec4f,
    @location(3) view_z: f32,
    @location(4) @interpolate(flat) object_id: u32,
};

@vertex
fn vs_main(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    let inst = instances[instance_index];
    let world = inst.model * vec4f(position, 1.0);
    let view = cam.view * world;
    var out: VsOut;
    out.clip = cam.proj * view;
    // Uniform scale is the common case here; normalize in the fragment stage.
    out.normal = (inst.model * vec4f(normal, 0.0)).xyz;
    out.world = world.xyz;
    out.color = inst.color;
    out.view_z = view.z;
    out.object_id = inst.object_id;
    return out;
}

fn shade(normal: vec3f, world: vec3f, base: vec3f) -> vec3f {
    let n = normalize(normal);
    let l = normalize(-cam.light_dir.xyz);
    let v = normalize(cam.eye.xyz - world);
    let h = normalize(l + v);
    let ndl = max(dot(n, l), 0.0);
    let spec = 0.3 * pow(max(dot(n, h), 0.0), 32.0);
    let rgb = cam.ambient.rgb * base + cam.light_diffuse.rgb * base * ndl
        + cam.light_specular.rgb * spec;
    return pow(max(rgb, vec3f(0.0)), vec3f(1.0 / 2.2));
}

struct FsOut {
    @location(0) color: vec4f,
    @location(1) depth: f32,   // metric, meters from the camera plane
    @location(2) object_id: u32,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = vec4f(shade(in.normal, in.world, in.color.rgb), 1.0);
    out.depth = -in.view_z;
    out.object_id = in.object_id;
    return out;
}

@fragment
fn fs_color_only(in: VsOut) -> @location(0) vec4f {
    return vec4f(shade(in.normal, in.world, in.color.rgb), 1.0);
}
"""

# (FsIn was renamed to VsOut above; no patch-up needed.)
WGSL = WGSL.replace("in: FsIn", "in: VsOut")

_INSTANCE_DTYPE = np.dtype(
    [
        ("model", "(4,4)f4"),  # column-major upload, like InstanceStore.pack
        ("color", "(4,)f4"),
        ("object_id", "u4"),
        ("pad", "(3,)f4"),
    ]
)
assert _INSTANCE_DTYPE.itemsize == 96


def perspective_wgpu(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective with WebGPU clip conventions (z in [0, 1])."""
    f = 1.0 / np.tan(fov_y * 0.5)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = (far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _aligned_row_bytes(width: int, bpp: int) -> int:
    raw = width * bpp
    return (raw + 255) // 256 * 256


class WgpuSceneBackend:
    def __init__(self, width: int, height: int, background=(0.13, 0.14, 0.16, 1.0)) -> None:
        self.device = wgpu.utils.get_default_device()
        self.width = int(width)
        self.height = int(height)
        self.background = background
        self._meshes: dict[object, tuple[wgpu.GPUBuffer, wgpu.GPUBuffer, int]] = {}
        self._instance_buffer: wgpu.GPUBuffer | None = None
        self._instance_capacity = 0

        adapter_info = self.device.adapter.info
        self.adapter_name = f"{adapter_info.vendor} {adapter_info.device}"

        module = self.device.create_shader_module(code=WGSL)
        self._camera_buffer = self.device.create_buffer(
            size=5 * 64 + 3 * 16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 24,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": module,
                "entry_point": "fs_main",
                "targets": [
                    {"format": "rgba8unorm"},
                    {"format": "r32float"},
                    {"format": "r32uint"},
                ],
            },
            primitive={
                "topology": "triangle-list",
                "front_face": "ccw",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
            multisample={"count": 1},
        )
        self._make_targets()
        self._module = module
        self._surface_pipelines: dict[str, wgpu.GPURenderPipeline] = {}
        self._surface_zbuf: dict[tuple[int, int], wgpu.GPUTexture] = {}
        self._vertex_buffers_dict = {  # shared by both pipelines
            "buffers": [
                {
                    "array_stride": 24,
                    "step_mode": "vertex",
                    "attributes": [
                        {"format": "float32x3", "offset": 0, "shader_location": 0},
                        {"format": "float32x3", "offset": 12, "shader_location": 1},
                    ],
                }
            ]
        }

    def _make_targets(self) -> None:
        size = (self.width, self.height, 1)
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self._color_tex = self.device.create_texture(
            size=size, format="rgba8unorm", usage=usage
        )
        self._depth_tex = self.device.create_texture(size=size, format="r32float", usage=usage)
        self._id_tex = self.device.create_texture(size=size, format="r32uint", usage=usage)
        self._zbuf = self.device.create_texture(
            size=size, format="depth24plus", usage=wgpu.TextureUsage.RENDER_ATTACHMENT
        )

    def set_meshes(self, meshes: dict) -> None:
        for key, mesh in meshes.items():
            if key in self._meshes:
                continue
            n = len(mesh.positions)
            vertices = np.empty((n, 6), np.float32)
            vertices[:, 0:3] = mesh.positions
            vertices[:, 3:6] = mesh.normals
            vbo = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(vertices),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
            ibo = self.device.create_buffer_with_data(
                data=np.ascontiguousarray(mesh.indices, np.uint32),
                usage=wgpu.BufferUsage.INDEX,
            )
            self._meshes[key] = (vbo, ibo, len(mesh.indices))

    def set_camera_uniforms(self, camera, lights) -> None:
        view = np.ascontiguousarray(camera.view_matrix().T)
        proj = np.ascontiguousarray(
            perspective_wgpu(camera.fov_y, camera.aspect, camera.near, camera.far).T
        )
        directional = next(
            (lt for lt in lights.lights if lt.active and int(lt.kind) == 0), None
        )
        if directional is None:
            light_dir = np.asarray(camera.target, np.float32) - np.asarray(camera.eye, np.float32)
            light_dir = light_dir / max(np.linalg.norm(light_dir), 1e-12)
            diffuse = np.array([0.4, 0.4, 0.4], np.float32)
            specular = np.array([0.5, 0.5, 0.5], np.float32)
        else:
            light_dir = np.asarray(directional.direction, np.float32)
            light_dir = light_dir / max(np.linalg.norm(light_dir), 1e-12)
            diffuse = np.asarray(directional.diffuse, np.float32)
            specular = np.asarray(directional.specular, np.float32)
        ambient = np.asarray(lights.ambient, np.float32)
        eye = np.asarray(camera.eye, np.float32)

        def pad3(v):
            return np.array([v[0], v[1], v[2], 0.0], np.float32)

        q = self.device.queue
        q.write_buffer(self._camera_buffer, 0, view)
        q.write_buffer(self._camera_buffer, 64, proj)
        q.write_buffer(self._camera_buffer, 128, pad3(light_dir))
        q.write_buffer(self._camera_buffer, 144, pad3(diffuse))
        q.write_buffer(self._camera_buffer, 160, pad3(specular))
        q.write_buffer(self._camera_buffer, 176, pad3(ambient))
        q.write_buffer(self._camera_buffer, 192, pad3(eye))

    def _ensure_instances(self, count: int) -> None:
        if count <= self._instance_capacity:
            return
        capacity = max(count, self._instance_capacity * 2, 64)
        self._instance_buffer = self.device.create_buffer(
            size=capacity * _INSTANCE_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._instance_capacity = capacity

    def upload_instances(self, scene: RenderScene) -> None:
        n = scene.count
        self._ensure_instances(max(n, 1))
        if n == 0:
            return
        data = np.zeros(n, _INSTANCE_DTYPE)
        data["model"] = scene.transforms.transpose(0, 2, 1)
        data["color"] = scene.colors
        data["object_id"] = scene.object_id
        self.device.queue.write_buffer(self._instance_buffer, 0, data)

    def render(self) -> None:
        encoder = self.device.create_command_encoder()
        color_attachments = [
            {
                "view": self._color_tex.create_view(),
                "clear_value": self.background,
                "load_op": "clear",
                "store_op": "store",
            },
            {
                "view": self._depth_tex.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 0.0),
                "load_op": "clear",
                "store_op": "store",
            },
            {
                "view": self._id_tex.create_view(),
                "clear_value": (0, 0, 0, 0),
                "load_op": "clear",
                "store_op": "store",
            },
        ]
        pass_encoder = encoder.begin_render_pass(
            color_attachments=color_attachments,
            depth_stencil_attachment={
                "view": self._zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        pass_encoder.set_pipeline(self.pipeline)
        pass_encoder.set_bind_group(0, self._make_bind_group(self.pipeline))
        self._draw_buckets(pass_encoder)
        pass_encoder.end()
        self.device.queue.submit([encoder.finish()])

    def _make_bind_group(self, pipeline):
        return self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self._camera_buffer, "offset": 0, "size": 208}},
                {
                    "binding": 1,
                    "resource": {"buffer": self._instance_buffer, "offset": 0, "size": self._instance_capacity * 96},
                },
            ],
        )

    def _draw_buckets(self, pass_encoder) -> None:
        scene = self._scene
        for bucket, (start, stop) in enumerate(scene.bucket_ranges):
            if stop <= start:
                continue
            key = scene.bucket_keys[bucket][0]
            entry = self._meshes.get(key)
            if entry is None:
                continue
            vbo, ibo, index_count = entry
            pass_encoder.set_vertex_buffer(0, vbo)
            pass_encoder.set_index_buffer(ibo, "uint32")
            pass_encoder.draw_indexed(index_count, stop - start, 0, 0, start)

    def _surface_pipeline(self, format: str) -> wgpu.GPURenderPipeline:
        pipeline = self._surface_pipelines.get(format)
        if pipeline is None:
            pipeline = self.device.create_render_pipeline(
                layout="auto",
                vertex={
                    "module": self._module,
                    "entry_point": "vs_main",
                    **self._vertex_buffers_dict,
                },
                fragment={
                    "module": self._module,
                    "entry_point": "fs_color_only",
                    "targets": [{"format": format}],
                },
                primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
                depth_stencil={
                    "format": "depth24plus",
                    "depth_write_enabled": True,
                    "depth_compare": "less",
                },
                multisample={"count": 1},
            )
            self._surface_pipelines[format] = pipeline
        return pipeline

    def draw_to_view(self, view, width: int, height: int, format: str) -> None:
        """Render the current scene (color only) into an external texture view."""
        key = (int(width), int(height))
        zbuf = self._surface_zbuf.get(key)
        if zbuf is None:
            zbuf = self.device.create_texture(
                size=(key[0], key[1], 1),
                format="depth24plus",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self._surface_zbuf[key] = zbuf
        pipeline = self._surface_pipeline(format)
        encoder = self.device.create_command_encoder()
        pass_encoder = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "clear_value": self.background,
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            depth_stencil_attachment={
                "view": zbuf.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        pass_encoder.set_pipeline(pipeline)
        pass_encoder.set_bind_group(0, self._make_bind_group(pipeline))
        self._draw_buckets(pass_encoder)
        pass_encoder.end()
        self.device.queue.submit([encoder.finish()])

    def draw_scene(self, scene: RenderScene) -> None:
        self._scene = scene
        self.upload_instances(scene)
        self.set_camera_uniforms(scene.camera, scene.lights)
        self.render()

    def _read_texture(self, texture, dtype, shape) -> np.ndarray:
        bpp = np.dtype(dtype).itemsize * (shape[2] if len(shape) > 2 else 1)
        row_bytes = _aligned_row_bytes(self.width, bpp)
        data = self.device.queue.read_texture(
            {"texture": texture, "origin": (0, 0, 0)},
            {"bytes_per_row": row_bytes, "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        raw = np.frombuffer(data, np.uint8).reshape(self.height, row_bytes)
        trimmed = raw[:, : self.width * bpp]
        channels = bpp // np.dtype(dtype).itemsize
        image = trimmed.view(dtype).reshape(self.height, self.width, channels)
        if channels == 1:
            image = image[..., 0]
        return np.ascontiguousarray(image)

    def read_color(self) -> np.ndarray:
        return self._read_texture(self._color_tex, np.uint8, (self.height, self.width, 4))

    def read_linear_depth(self) -> np.ndarray:
        return self._read_texture(self._depth_tex, np.float32, (self.height, self.width, 1))

    def read_ids(self) -> np.ndarray:
        return self._read_texture(self._id_tex, np.uint32, (self.height, self.width, 1))

    def release(self) -> None:
        for vbo, ibo, _ in self._meshes.values():
            vbo.destroy()
            ibo.destroy()
        self._meshes.clear()
