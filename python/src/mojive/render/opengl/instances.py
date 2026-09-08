"""GPU mesh resources and lifecycle-aware instanced draw storage."""

from __future__ import annotations

import enum

import moderngl
import numpy as np

from . import gl_native as G

Attribute = tuple[str, str, int, int, int, int]

POSE_ATTRIBUTES: tuple[Attribute, ...] = (
    ("in_model0", "4f", 16, 4, 0, G.GL_FLOAT),
    ("in_model1", "4f", 16, 4, 16, G.GL_FLOAT),
    ("in_model2", "4f", 16, 4, 32, G.GL_FLOAT),
    ("in_model3", "4f", 16, 4, 48, G.GL_FLOAT),
)
VISUAL_ATTRIBUTES: tuple[Attribute, ...] = (
    ("in_color", "4f", 16, 4, 0, G.GL_FLOAT),
    ("in_material", "4f", 16, 4, 16, G.GL_FLOAT),
    ("in_texcoef", "4f", 16, 4, 32, G.GL_FLOAT),
    ("in_cubecoef", "4f", 16, 4, 48, G.GL_FLOAT),
)
IDENTITY_ATTRIBUTES: tuple[Attribute, ...] = (
    ("in_object_id", "1u", 4, 1, 0, G.GL_UNSIGNED_INT),
    ("in_reflection_info", "1u", 4, 1, 4, G.GL_UNSIGNED_INT),
    ("in_segment_id", "1i", 4, 1, 8, G.GL_INT),
    ("in_segment_type", "1i", 4, 1, 12, G.GL_INT),
)
INSTANCE_ATTRIBUTES = (*POSE_ATTRIBUTES, *VISUAL_ATTRIBUTES, *IDENTITY_ATTRIBUTES)

POSE_BYTES = 64
VISUAL_BYTES = 64
IDENTITY_BYTES = 16
INSTANCE_WORDS = 36
INSTANCE_BYTES = POSE_BYTES + VISUAL_BYTES + IDENTITY_BYTES

INSTANCE_STREAMS: tuple[tuple[str, tuple[Attribute, ...], int], ...] = (
    ("pose", POSE_ATTRIBUTES, POSE_BYTES),
    ("visual", VISUAL_ATTRIBUTES, VISUAL_BYTES),
    ("identity", IDENTITY_ATTRIBUTES, IDENTITY_BYTES),
)

MESH_ATTRIBUTES: tuple[Attribute, ...] = (
    ("in_position", "3f", 12, 3, 0, G.GL_FLOAT),
    ("in_normal", "3f", 12, 3, 12, G.GL_FLOAT),
    ("in_uv", "2f", 8, 2, 24, G.GL_FLOAT),
)


def build_layout(
    program: moderngl.Program,
    entries: tuple[Attribute, ...],
    per_instance: bool = False,
) -> tuple[str, tuple[str, ...]]:
    parts: list[str] = []
    names: list[str] = []
    for name, fmt, nbytes, *_ in entries:
        if name in program:
            parts.append(fmt)
            names.append(name)
        else:
            parts.append(f"{nbytes}x")
    layout = " ".join(parts)
    return (layout + "/i" if per_instance else layout), tuple(names)


def instance_content(program, buffers) -> list[tuple]:
    """Build moderngl content entries for only the streams a shader consumes."""

    content = []
    for buffer, (_name, attributes, _stride) in zip(buffers, INSTANCE_STREAMS, strict=True):
        layout, names = build_layout(program, attributes, per_instance=True)
        if names:
            content.append((buffer, layout, *names))
    return content


def _changed_runs(mask: np.ndarray):
    if not len(mask) or not bool(np.any(mask)):
        return
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    yield from zip(edges[::2], edges[1::2], strict=True)


class Strategy(enum.StrEnum):
    SHARED = "shared"
    PER_BUCKET = "per_bucket"


class GpuMesh:
    __slots__ = ("_vertices", "ibo", "index_count", "triangle_count", "vbo")

    def __init__(self, ctx: moderngl.Context, positions, normals, uvs, indices) -> None:
        n = len(positions)
        self._vertices = np.empty((n, 8), np.float32)
        self._vertices[:, 0:3] = positions
        self._vertices[:, 3:6] = normals
        self._vertices[:, 6:8] = uvs
        self.vbo = ctx.buffer(self._vertices.tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(indices, np.uint32).tobytes())
        self.index_count = len(indices)
        self.triangle_count = self.index_count // 3

    def update(self, positions: np.ndarray, normals: np.ndarray) -> None:
        shape = self._vertices[:, :3].shape
        if positions.shape != shape or normals.shape != shape:
            raise ValueError(
                f"dynamic mesh vertex shape changed: expected {shape}, "
                f"got {positions.shape} / {normals.shape}"
            )
        np.copyto(self._vertices[:, :3], positions, casting="unsafe")
        np.copyto(self._vertices[:, 3:6], normals, casting="unsafe")
        self.vbo.write(self._vertices)

    def release(self) -> None:
        self.vbo.release()
        self.ibo.release()


class InstanceStore:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._gl = G.native()
        self.strategy = Strategy.SHARED if self._gl.has_attrib_pointer else Strategy.PER_BUCKET
        self.pose_buffer: moderngl.Buffer | None = None
        self.visual_buffer: moderngl.Buffer | None = None
        self.identity_buffer: moderngl.Buffer | None = None
        self.capacity = 0
        self.count = 0
        self._vaos: list[moderngl.VertexArray | None] = []
        self._bucket_buffers: list[tuple[moderngl.Buffer, ...]] = []
        self._ranges: tuple[tuple[int, int], ...] = ()
        self._program: moderngl.Program | None = None
        self._generation = -1
        self._meshes: list[GpuMesh | None] = []
        self._keys = ()
        self._last_structure = -1
        self._last_scene = None
        self._last_pose = -1
        self._last_visual = -1
        self._last_identity = -1
        self._valid = False
        self.uploaded_bytes = 0
        self.uploaded_streams: dict[str, int] = {}

        self._pose = np.zeros((0, 16), np.float32)
        self._visual = np.zeros((0, 16), np.float32)
        self._identity = np.zeros((0, 4), np.uint32)
        self.draw_calls = 0

    @property
    def buffer(self) -> moderngl.Buffer | None:
        """Compatibility alias for transform-only consumers."""

        return self.pose_buffer

    @property
    def buffers(self) -> tuple[moderngl.Buffer, moderngl.Buffer, moderngl.Buffer]:
        assert self.pose_buffer is not None
        assert self.visual_buffer is not None
        assert self.identity_buffer is not None
        return self.pose_buffer, self.visual_buffer, self.identity_buffer

    @property
    def reflection_info(self) -> np.ndarray:
        return self._identity[: self.count, 1]

    def _ensure_capacity(self, count: int) -> bool:
        if count <= self.capacity and self.pose_buffer is not None:
            return False
        new_cap = max(count, self.capacity * 2, 64)
        for buffer in (self.pose_buffer, self.visual_buffer, self.identity_buffer):
            if buffer is not None:
                buffer.release()
        self.pose_buffer = self.ctx.buffer(reserve=new_cap * POSE_BYTES)
        self.visual_buffer = self.ctx.buffer(reserve=new_cap * VISUAL_BYTES)
        self.identity_buffer = self.ctx.buffer(reserve=new_cap * IDENTITY_BYTES)
        self.capacity = new_cap
        self._pose = np.zeros((new_cap, 16), np.float32)
        self._visual = np.zeros((new_cap, 16), np.float32)
        self._identity = np.zeros((new_cap, 4), np.uint32)
        self._valid = False
        return True

    def ensure_capacity(self, count: int) -> bool:
        """Ensure pass-independent instance streams exist before VAO creation."""

        return self._ensure_capacity(max(int(count), 1))

    def invalidate_upload(self) -> None:
        self._valid = False

    @staticmethod
    def _attributes(program, entries):
        return tuple(
            (InstanceStore._location(program, name), comps, off, gl_type)
            for name, _fmt, _nbytes, comps, off, gl_type in entries
        )

    def _bind_shared_offsets(self, vao, program, start: int) -> bool:
        for buffer, (_name, entries, stride) in zip(self.buffers, INSTANCE_STREAMS, strict=True):
            if not self._gl.rebind_instance_attributes(
                vao.glo,
                buffer.glo,
                stride,
                start * stride,
                self._attributes(program, entries),
            ):
                return False
        return True

    def rebuild(self, scene, program, meshes, generation: int) -> None:
        self._release_vaos()
        self._ensure_capacity(max(scene.count, 1))
        self._program = program
        self._generation = generation
        self._meshes = meshes
        self._keys = scene.bucket_keys
        self._ranges = scene.bucket_ranges

        mesh_layout, mesh_names = build_layout(program, MESH_ATTRIBUTES)
        if self.strategy is Strategy.SHARED:
            for b, (start, _stop) in enumerate(scene.bucket_ranges):
                mesh = meshes[b] if b < len(meshes) else None
                if mesh is None:
                    self._vaos.append(None)
                    continue
                vao = self.ctx.vertex_array(
                    program,
                    [
                        (mesh.vbo, mesh_layout, *mesh_names),
                        *instance_content(program, self.buffers),
                    ],
                    mesh.ibo,
                    index_element_size=4,
                )
                if not self._bind_shared_offsets(vao, program, start):
                    self.strategy = Strategy.PER_BUCKET
                    self._release_vaos()
                    self.rebuild(scene, program, meshes, generation)
                    return
                self._vaos.append(vao)
        else:
            for b, (start, stop) in enumerate(scene.bucket_ranges):
                mesh = meshes[b] if b < len(meshes) else None
                count = max(1, stop - start)
                buffers = tuple(
                    self.ctx.buffer(reserve=count * stride)
                    for _name, _entries, stride in INSTANCE_STREAMS
                )
                self._bucket_buffers.append(buffers)
                if mesh is None:
                    self._vaos.append(None)
                    continue
                self._vaos.append(
                    self.ctx.vertex_array(
                        program,
                        [(mesh.vbo, mesh_layout, *mesh_names), *instance_content(program, buffers)],
                        mesh.ibo,
                        index_element_size=4,
                    )
                )

    @staticmethod
    def _location(program: moderngl.Program, name: str) -> int:
        try:
            return int(program[name].location)
        except KeyError:
            return -1

    def pack(self, scene, reflection_info: np.ndarray | None = None) -> np.ndarray:
        """Return the three streams in one diagnostic record array."""

        n = scene.count
        raw = np.zeros((n, INSTANCE_WORDS), np.uint32)
        floats = raw.view(np.float32)
        if n:
            floats[:, :16] = scene.transforms.transpose(0, 2, 1).reshape(n, 16)
            floats[:, 16:20] = scene.colors
            floats[:, 20:24] = scene.material
            floats[:, 24:28] = scene.tex_coef
            floats[:, 28:32] = scene.cube_coef
            raw[:, 32] = scene.object_id
            if reflection_info is not None:
                raw[:, 33] = np.asarray(reflection_info, np.uint32).reshape(n)
            raw[:, 34:36] = (
                scene.segmentation.view(np.uint32)
                if scene.segmentation.shape == (n, 2)
                else np.uint32(0xFFFFFFFF)
            )
        return raw

    def _write_shared(self, name, buffer, data, changed, stride) -> None:
        written = 0
        for start, stop in _changed_runs(changed):
            payload = data[start:stop]
            buffer.write(payload, int(start) * stride)
            written += payload.nbytes
        if written:
            self.uploaded_streams[name] = written
            self.uploaded_bytes += written

    def _write_bucket_stream(self, stream: int, data: np.ndarray) -> int:
        written = 0
        for b, (start, stop) in enumerate(self._ranges):
            if stop > start and b < len(self._bucket_buffers):
                payload = data[start:stop]
                self._bucket_buffers[b][stream].write(payload)
                written += payload.nbytes
        return written

    def _write_stream(self, stream, name, buffer, data, changed, stride) -> None:
        if self.strategy is Strategy.SHARED:
            self._write_shared(name, buffer, data, changed, stride)
            return
        if not bool(np.any(changed)):
            return
        written = self._write_bucket_stream(stream, data)
        if written:
            self.uploaded_streams[name] = written
            self.uploaded_bytes += written

    def upload(self, scene, reflection_info: np.ndarray | None = None) -> None:
        n = scene.count
        grew = self._ensure_capacity(max(n, 1))
        self.count = n
        self.uploaded_bytes = 0
        self.uploaded_streams = {}
        if n == 0:
            self._last_scene = scene
            self._last_structure = scene.structure_revision
            self._last_pose = scene.pose_revision
            self._last_visual = scene.visual_revision
            self._last_identity = scene.identity_revision
            self._valid = True
            return
        full = (
            grew
            or not self._valid
            or scene is not self._last_scene
            or scene.structure_revision != self._last_structure
        )

        if full or not scene.pose_revision or scene.pose_revision != self._last_pose:
            source = scene.transforms.transpose(0, 2, 1).reshape(n, 16)
            changed = np.ones(n, bool) if full else np.any(self._pose[:n] != source, axis=1)
            self._pose[:n] = source
            assert self.pose_buffer is not None
            self._write_stream(0, "pose", self.pose_buffer, self._pose, changed, POSE_BYTES)

        if full or not scene.visual_revision or scene.visual_revision != self._last_visual:
            changed = np.ones(n, bool) if full else np.zeros(n, bool)
            for start, source in (
                (0, scene.colors),
                (4, scene.material),
                (8, scene.tex_coef),
                (12, scene.cube_coef),
            ):
                if not full:
                    changed |= np.any(self._visual[:n, start : start + 4] != source, axis=1)
                self._visual[:n, start : start + 4] = source
            assert self.visual_buffer is not None
            self._write_stream(1, "visual", self.visual_buffer, self._visual, changed, VISUAL_BYTES)

        reflect = np.asarray(reflection_info, np.uint32) if reflection_info is not None else None
        if reflect is None or reflect.size != n:
            reflect = np.zeros(n, np.uint32)
        else:
            reflect = reflect.reshape(n)
        identity_dirty = (
            full
            or not scene.identity_revision
            or scene.identity_revision != self._last_identity
            or not np.array_equal(self._identity[:n, 1], reflect)
        )
        if identity_dirty:
            changed = np.ones(n, bool) if full else self._identity[:n, 1] != reflect
            segmentation = (
                scene.segmentation.view(np.uint32)
                if scene.segmentation.shape == (n, 2)
                else np.full((n, 2), np.uint32(0xFFFFFFFF), np.uint32)
            )
            if not full and (
                not scene.identity_revision or scene.identity_revision != self._last_identity
            ):
                changed |= self._identity[:n, 0] != scene.object_id
                changed |= np.any(self._identity[:n, 2:4] != segmentation, axis=1)
            self._identity[:n, 0] = scene.object_id
            self._identity[:n, 1] = reflect
            self._identity[:n, 2:4] = segmentation
            assert self.identity_buffer is not None
            self._write_stream(
                2, "identity", self.identity_buffer, self._identity, changed, IDENTITY_BYTES
            )

        self._last_structure = scene.structure_revision
        self._last_scene = scene
        self._last_pose = scene.pose_revision
        self._last_visual = scene.visual_revision
        self._last_identity = scene.identity_revision
        self._valid = True

    def stream_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._pose, self._visual, self._identity

    def draw(self, bucket: int, instances: int | None = None) -> int:
        if not (0 <= bucket < len(self._vaos)):
            return 0
        vao = self._vaos[bucket]
        if vao is None:
            return 0
        start, stop = self._ranges[bucket]
        count = (stop - start) if instances is None else instances
        if count <= 0:
            return 0
        vao.render(moderngl.TRIANGLES, instances=count)
        self.draw_calls += 1
        return count

    def vao(self, bucket: int) -> moderngl.VertexArray | None:
        return self._vaos[bucket] if 0 <= bucket < len(self._vaos) else None

    def triangles(self, scene, meshes=None) -> int:
        total = 0
        meshes = self._meshes if meshes is None else meshes
        for b, (start, stop) in enumerate(scene.bucket_ranges):
            mesh = meshes[b] if b < len(meshes) else None
            if mesh is not None:
                total += mesh.triangle_count * (stop - start)
        return total

    def needs_rebuild(self, scene, generation: int) -> bool:
        return (
            generation != self._generation
            or scene.bucket_keys != self._keys
            or scene.bucket_ranges != self._ranges
            or scene.count > self.capacity
        )

    def _release_vaos(self) -> None:
        for vao in self._vaos:
            if vao is not None:
                vao.release()
        self._vaos.clear()
        for buffers in self._bucket_buffers:
            for buffer in buffers:
                buffer.release()
        self._bucket_buffers.clear()

    def release(self) -> None:
        self._release_vaos()
        for buffer in (self.pose_buffer, self.visual_buffer, self.identity_buffer):
            if buffer is not None:
                buffer.release()
        self.pose_buffer = None
        self.visual_buffer = None
        self.identity_buffer = None
        self.capacity = 0
        self.count = 0
        self._last_scene = None
