"""Renderer-facing scene data and draw buckets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import CameraView, LightSet, Material, MeshKey, ShadingModel

# transform 16 + color 4 + material 4 + tex_coef 4 + cube_coef 4 = 32
INSTANCE_FLOATS = 32
INSTANCE_STRIDE = INSTANCE_FLOATS * 4 + 4  # + object_id(uint32)

BACKGROUND_ID = np.uint32(0)


@dataclass
class RenderScene:
    count: int = 0

    transforms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    material: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    tex_coef: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    cube_coef: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    object_id: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.uint32))

    bucket: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))

    bucket_keys: tuple[tuple[MeshKey, int], ...] = ()
    bucket_ranges: tuple[tuple[int, int], ...] = ()
    opaque_buckets: tuple[int, ...] = ()
    transparent_buckets: tuple[int, ...] = ()

    camera: CameraView = field(default_factory=CameraView)
    lights: LightSet = field(default_factory=LightSet)
    materials: tuple[Material, ...] = ()
    shading_model: ShadingModel = ShadingModel.LINEAR

    scene_extent: float = 1.0

    shadow_clip: float = 1.0

    scene_center: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    selected_id: int = 0

    infinite_planes: tuple[int, ...] = ()

    def bucket_count(self) -> int:
        return len(self.bucket_keys)

    def triangle_count(self, mesh_tri_counts: dict[MeshKey, int]) -> int:
        total = 0
        for b, (start, stop) in enumerate(self.bucket_ranges):
            total += mesh_tri_counts.get(self.bucket_keys[b][0], 0) * (stop - start)
        return total

    def validate(self) -> None:
        n = self.count
        for name in (
            "transforms",
            "colors",
            "material",
            "tex_coef",
            "cube_coef",
            "object_id",
            "bucket",
        ):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match count {n}")
        if self.transforms.shape[1:] != (4, 4):
            raise ValueError(f"transforms must have shape (N, 4, 4), got {self.transforms.shape}")
        if self.transforms.dtype != np.float32:
            raise ValueError("transforms must use float32")
        if self.object_id.dtype != np.uint32:
            raise ValueError("object_id must use uint32")

        if len(self.bucket_ranges) != len(self.bucket_keys):
            raise ValueError("bucket_ranges and bucket_keys must have equal lengths")
        cursor = 0
        for i, (start, stop) in enumerate(self.bucket_ranges):
            if start != cursor:
                raise ValueError(f"bucket {i} starts at {start}; expected {cursor}")
            if stop < start:
                raise ValueError(f"bucket {i} has an invalid range [{start}, {stop})")
            cursor = stop
        if cursor != n:
            raise ValueError(f"bucket ranges cover {cursor} instances; count is {n}")

        seen = set(self.opaque_buckets) | set(self.transparent_buckets)
        if len(self.opaque_buckets) + len(self.transparent_buckets) != len(seen):
            raise ValueError("a bucket appears in both opaque and transparent sets")
        if seen != set(range(len(self.bucket_keys))):
            raise ValueError("every bucket must be opaque or transparent")

        for _, matid in self.bucket_keys:
            if not 0 <= matid < len(self.materials):
                raise ValueError(f"material id {matid} exceeds table size {len(self.materials)}")

    def transparent_draw_order(self, eye: np.ndarray | None = None) -> tuple[int, ...]:
        if not self.transparent_buckets:
            return ()
        eye = np.asarray(self.camera.eye if eye is None else eye, np.float32)
        keyed = []
        for b in self.transparent_buckets:
            start, stop = self.bucket_ranges[b]
            if stop <= start:
                keyed.append((-np.inf, b))
                continue
            centers = self.transforms[start:stop, :3, 3]
            d = float(np.max(np.linalg.norm(centers - eye, axis=1)))
            keyed.append((d, b))
        keyed.sort(key=lambda t: -t[0])
        return tuple(b for _, b in keyed)


class SceneBuilder:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._materials: list[Material] = []
        self._mat_index: dict[int, int] = {}
        self.write_index: np.ndarray = np.zeros(0, np.int32)

    def material_id(self, mat: Material) -> int:
        token = id(mat)
        if token not in self._mat_index:
            self._mat_index[token] = len(self._materials)
            self._materials.append(mat)
        return self._mat_index[token]

    def add(
        self,
        mesh: MeshKey,
        matid: int,
        transform: np.ndarray,
        color: np.ndarray,
        material: np.ndarray,
        object_id: int,
        tex_coef: np.ndarray | None = None,
        cube_coef: np.ndarray | None = None,
        infinite_plane: bool = False,
    ) -> int:
        self._rows.append(
            {
                "key": (mesh, matid),
                "transform": np.asarray(transform, np.float32).reshape(4, 4),
                "color": np.asarray(color, np.float32).reshape(4),
                "material": np.asarray(material, np.float32).reshape(4),
                "tex_coef": (
                    np.array([1.0, 1.0, 0.0, 0.0], np.float32)
                    if tex_coef is None
                    else np.asarray(tex_coef, np.float32).reshape(4)
                ),
                "cube_coef": (
                    np.zeros(4, np.float32)
                    if cube_coef is None
                    else np.asarray(cube_coef, np.float32).reshape(4)
                ),
                "object_id": np.uint32(object_id),
                "infinite_plane": infinite_plane,
            }
        )
        return len(self._rows) - 1

    def build(
        self,
        camera: CameraView,
        lights: LightSet,
        scene_extent: float,
        scene_center: np.ndarray,
        shadow_clip: float = 1.0,
        shading_model: ShadingModel = ShadingModel.LINEAR,
    ) -> RenderScene:
        n = len(self._rows)

        ident: list[tuple[MeshKey, int, bool, int]] = []
        seen: dict[tuple[MeshKey, int, bool, int], int] = {}
        row_bucket = np.empty(n, np.int32)
        for i, row in enumerate(self._rows):
            transparent = float(row["color"][3]) < 1.0
            key = (*row["key"], transparent, i if transparent else -1)
            if key not in seen:
                seen[key] = len(ident)
                ident.append(key)
            row_bucket[i] = seen[key]

        order_of_bucket = sorted(range(len(ident)), key=lambda b: (ident[b][2], b))
        remap = np.empty(len(ident), np.int32)
        for new_b, old_b in enumerate(order_of_bucket):
            remap[old_b] = new_b
        row_bucket = remap[row_bucket]
        ordered = [ident[b] for b in order_of_bucket]

        order = np.argsort(row_bucket, kind="stable")

        write_index = np.empty(n, np.int32)
        write_index[order] = np.arange(n, dtype=np.int32)
        self.write_index = write_index

        scene = RenderScene(count=n)
        scene.bucket = row_bucket[order].astype(np.int32)
        scene.transforms = (
            np.stack([r["transform"] for r in self._rows])[order]
            if n
            else np.zeros((0, 4, 4), np.float32)
        )
        scene.colors = (
            np.stack([r["color"] for r in self._rows])[order] if n else np.zeros((0, 4), np.float32)
        )
        scene.material = (
            np.stack([r["material"] for r in self._rows])[order]
            if n
            else np.zeros((0, 4), np.float32)
        )
        scene.tex_coef = (
            np.stack([r["tex_coef"] for r in self._rows])[order]
            if n
            else np.zeros((0, 4), np.float32)
        )
        scene.cube_coef = (
            np.stack([r["cube_coef"] for r in self._rows])[order]
            if n
            else np.zeros((0, 4), np.float32)
        )
        scene.object_id = (
            np.array([r["object_id"] for r in self._rows], np.uint32)[order]
            if n
            else np.zeros(0, np.uint32)
        )
        scene.infinite_planes = tuple(
            int(write_index[i]) for i, r in enumerate(self._rows) if r["infinite_plane"]
        )

        bounds = np.searchsorted(scene.bucket, np.arange(len(ordered) + 1))
        scene.bucket_ranges = tuple(
            (int(bounds[b]), int(bounds[b + 1])) for b in range(len(ordered))
        )
        scene.bucket_keys = tuple((mesh, matid) for mesh, matid, _, _ in ordered)
        scene.opaque_buckets = tuple(b for b, k in enumerate(ordered) if not k[2])
        scene.transparent_buckets = tuple(b for b, k in enumerate(ordered) if k[2])
        scene.materials = tuple(self._materials)
        scene.shading_model = shading_model
        scene.camera = camera
        scene.lights = lights
        scene.scene_extent = float(scene_extent)
        scene.shadow_clip = float(shadow_clip)
        scene.scene_center = np.asarray(scene_center, np.float32)
        scene.validate()
        return scene
