from __future__ import annotations

import enum

import moderngl
import numpy as np

from ..scene import INSTANCE_FLOATS, RenderScene
from . import gl_native as G

INSTANCE_ATTRIBUTES: tuple[tuple[str, str, int, int, int, int], ...] = (
    ("in_model0", "4f", 16, 4, 0, G.GL_FLOAT),
    ("in_model1", "4f", 16, 4, 16, G.GL_FLOAT),
    ("in_model2", "4f", 16, 4, 32, G.GL_FLOAT),
    ("in_model3", "4f", 16, 4, 48, G.GL_FLOAT),
    ("in_color", "4f", 16, 4, 64, G.GL_FLOAT),
    ("in_material", "4f", 16, 4, 80, G.GL_FLOAT),
    ("in_texcoef", "4f", 16, 4, 96, G.GL_FLOAT),
    ("in_object_id", "1u", 4, 1, 112, G.GL_UNSIGNED_INT),
)
INSTANCE_WORDS = INSTANCE_FLOATS + 1
INSTANCE_BYTES = INSTANCE_WORDS * 4  # 116
MESH_ATTRIBUTES: tuple[tuple[str, str, int, int, int, int], ...] = (
    ("in_position", "3f", 12, 3, 0, G.GL_FLOAT),
    ("in_normal", "3f", 12, 3, 12, G.GL_FLOAT),
    ("in_uv", "2f", 8, 2, 24, G.GL_FLOAT),
)


def build_layout(
    program: moderngl.Program,
    entries: tuple[tuple[str, str, int, int, int, int], ...],
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
        self.buffer: moderngl.Buffer | None = None
        self.capacity = 0
        self._vaos: list[moderngl.VertexArray] = []
        self._bucket_buffers: list[moderngl.Buffer] = []
        self._ranges: tuple[tuple[int, int], ...] = ()
        self._program: moderngl.Program | None = None
        self._generation = -1
        self._meshes: list[GpuMesh | None] = []

        self._raw = np.zeros((0, INSTANCE_WORDS), np.uint32)
        self._staging = self._raw.view(np.float32)
        self.draw_calls = 0

    # ------------------------------------------------------------------
    def _ensure_capacity(self, count: int) -> bool:

        if count <= self.capacity and self.buffer is not None:
            return False
        new_cap = max(count, self.capacity * 2, 64)
        if self.buffer is not None:
            self.buffer.release()
        self.buffer = self.ctx.buffer(reserve=new_cap * INSTANCE_BYTES)
        self.capacity = new_cap
        self._raw = np.zeros((new_cap, INSTANCE_WORDS), np.uint32)
        self._staging = self._raw.view(np.float32)
        return True

    def rebuild(
        self,
        scene: RenderScene,
        program: moderngl.Program,
        meshes: list[GpuMesh | None],
        generation: int,
    ) -> None:

        self._release_vaos()
        self._ensure_capacity(max(scene.count, 1))
        self._program = program
        self._generation = generation
        self._meshes = meshes
        self._ranges = scene.bucket_ranges

        assert self.buffer is not None
        mesh_layout, mesh_names = build_layout(program, MESH_ATTRIBUTES)
        inst_layout, inst_names = build_layout(program, INSTANCE_ATTRIBUTES, per_instance=True)

        if self.strategy is Strategy.SHARED:
            attrs = tuple(
                (self._location(program, name), comps, off, gl_type)
                for name, _fmt, _nb, comps, off, gl_type in INSTANCE_ATTRIBUTES
            )
            for b, (start, _stop) in enumerate(scene.bucket_ranges):
                mesh = meshes[b] if b < len(meshes) else None
                if mesh is None:
                    self._vaos.append(None)  # type: ignore[arg-type]
                    continue
                vao = self.ctx.vertex_array(
                    program,
                    [
                        (mesh.vbo, mesh_layout, *mesh_names),
                        (self.buffer, inst_layout, *inst_names),
                    ],
                    mesh.ibo,
                    index_element_size=4,
                )

                ok = self._gl.rebind_instance_attributes(
                    vao.glo, self.buffer.glo, INSTANCE_BYTES, start * INSTANCE_BYTES, attrs
                )
                if not ok:
                    self.strategy = Strategy.PER_BUCKET
                    self._release_vaos()
                    self.rebuild(scene, program, meshes, generation)
                    return
                self._vaos.append(vao)
        else:
            for b, (start, stop) in enumerate(scene.bucket_ranges):
                mesh = meshes[b] if b < len(meshes) else None
                buf = self.ctx.buffer(reserve=max(1, stop - start) * INSTANCE_BYTES)
                self._bucket_buffers.append(buf)
                if mesh is None:
                    self._vaos.append(None)  # type: ignore[arg-type]
                    continue
                self._vaos.append(
                    self.ctx.vertex_array(
                        program,
                        [
                            (mesh.vbo, mesh_layout, *mesh_names),
                            (buf, inst_layout, *inst_names),
                        ],
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

    # ------------------------------------------------------------------
    def pack(self, scene: RenderScene) -> np.ndarray:

        n = scene.count
        if n == 0:
            return self._raw[:0]
        if n > len(self._raw):
            self._raw = np.zeros((max(n, self.capacity), INSTANCE_WORDS), np.uint32)
            self._staging = self._raw.view(np.float32)
        dst = self._staging[:n]

        dst[:, 0:16] = scene.transforms.transpose(0, 2, 1).reshape(n, 16)
        dst[:, 16:20] = scene.colors
        dst[:, 20:24] = scene.material
        dst[:, 24:28] = scene.tex_coef

        self._raw[:n, 28] = scene.object_id
        return self._raw[:n]

    def upload(self, scene: RenderScene) -> None:

        data = self.pack(scene)
        if len(data) == 0 or self.buffer is None:
            return
        if self.strategy is Strategy.SHARED:
            self.buffer.write(data)
            return
        for b, (start, stop) in enumerate(self._ranges):
            if stop > start and b < len(self._bucket_buffers):
                self._bucket_buffers[b].write(data[start:stop])

    # ------------------------------------------------------------------
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

    def triangles(self, scene: RenderScene) -> int:
        total = 0
        for b, (start, stop) in enumerate(scene.bucket_ranges):
            mesh = self._meshes[b] if b < len(self._meshes) else None
            if mesh is not None:
                total += mesh.triangle_count * (stop - start)
        return total

    def needs_rebuild(self, scene: RenderScene, generation: int) -> bool:
        return (
            generation != self._generation
            or scene.bucket_ranges != self._ranges
            or scene.count > self.capacity
        )

    # ------------------------------------------------------------------
    def _release_vaos(self) -> None:
        for vao in self._vaos:
            if vao is not None:
                vao.release()
        self._vaos.clear()
        for buf in self._bucket_buffers:
            buf.release()
        self._bucket_buffers.clear()

    def release(self) -> None:
        self._release_vaos()
        if self.buffer is not None:
            self.buffer.release()
            self.buffer = None
        self.capacity = 0
