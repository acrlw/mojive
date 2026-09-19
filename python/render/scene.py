"""Renderer-facing scene data and draw buckets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import CameraView, LightSet, Material, MeshKey, ShadingModel

# transform 16 + color 4 + material 4 + tex_coef 4 + cube_coef 4 = 32;
# identity adds object, reflection routing, and two segmentation words.
INSTANCE_FLOATS = 32
INSTANCE_STRIDE = INSTANCE_FLOATS * 4 + 16

BACKGROUND_ID = np.uint32(0)


@dataclass
class RenderScene:
    count: int = 0

    transforms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4), np.float32))
    # Negative alpha encodes ordered coverage: -(1 + opacity) for visual surfaces,
    # -(3 + opacity) for complementary collision surfaces. Both remain instanced.
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

    # Appended lifecycle and semantic fields preserve the legacy positional
    # constructor while giving managed scene builders an explicit upload contract.
    structure_revision: int = 0
    pose_revision: int = 0
    visual_revision: int = 0
    identity_revision: int = 0
    segmentation: np.ndarray = field(default_factory=lambda: np.full((0, 2), -1, np.int32))

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
            "segmentation",
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
        if self.segmentation.shape != (n, 2) or self.segmentation.dtype != np.int32:
            raise ValueError("segmentation must use int32 with shape (N, 2)")

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
        buckets = np.asarray(self.transparent_buckets, np.intp)
        starts = np.fromiter(
            (self.bucket_ranges[int(bucket)][0] for bucket in buckets),
            np.intp,
            count=len(buckets),
        )
        stops = np.fromiter(
            (self.bucket_ranges[int(bucket)][1] for bucket in buckets),
            np.intp,
            count=len(buckets),
        )
        if np.all(stops - starts == 1):
            offsets = self.transforms[starts, :3, 3] - eye
            distance2 = np.einsum("ij,ij->i", offsets, offsets)
            order = np.argsort(-distance2, kind="stable")
            return tuple(int(bucket) for bucket in buckets[order])
        keyed = []
        for b in buckets:
            b = int(b)
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
        segmentation: tuple[int, int] | np.ndarray = (-1, -1),
        coverage: bool = False,
        collision_coverage: bool = False,
    ) -> int:
        color = np.asarray(color, np.float32).reshape(4).copy()
        if coverage:
            color[3] = -((3.0 if collision_coverage else 1.0) + np.clip(color[3], 0.0, 1.0))
        self._rows.append(
            {
                "key": (mesh, matid),
                "transform": np.asarray(transform, np.float32).reshape(4, 4),
                "color": color,
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
                "segmentation": np.asarray(segmentation, np.int32).reshape(2),
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
        scene = RenderScene(
            count=n,
            camera=camera,
            lights=lights,
            scene_extent=float(scene_extent),
            scene_center=np.asarray(scene_center, np.float32),
            shadow_clip=float(shadow_clip),
            shading_model=shading_model,
            infinite_planes=tuple(i for i, row in enumerate(self._rows) if row["infinite_plane"]),
        )
        for column, field_name, dtype, shape in (
            ("transforms", "transform", np.float32, (n, 4, 4)),
            ("colors", "color", np.float32, (n, 4)),
            ("material", "material", np.float32, (n, 4)),
            ("tex_coef", "tex_coef", np.float32, (n, 4)),
            ("cube_coef", "cube_coef", np.float32, (n, 4)),
            ("object_id", "object_id", np.uint32, (n,)),
            ("segmentation", "segmentation", np.int32, (n, 2)),
        ):
            setattr(
                scene, column, np.asarray([r[field_name] for r in self._rows], dtype).reshape(shape)
            )
        return self.build_columns(
            [r["key"][0] for r in self._rows],
            np.asarray([r["key"][1] for r in self._rows], np.intp),
            scene,
        )

    def build_columns(
        self, meshes: list[MeshKey], material_ids: np.ndarray, scene: RenderScene
    ) -> RenderScene:
        """Bucket instance columns without constructing a Python object for every row."""
        n = scene.count
        # Buckets represent GPU bindings, not logical materials. Keep first-use
        # order and one bucket per transparent row for stable depth sorting.
        mesh_ids: dict[MeshKey, int] = {}
        mesh_slots = np.fromiter(
            (mesh_ids.setdefault(mesh, len(mesh_ids)) for mesh in meshes), np.int64, count=n
        )
        bindings: dict[str | None, int] = {}
        material_bindings = np.asarray(
            [bindings.setdefault(mat.texture, len(bindings)) for mat in self._materials], np.int64
        )
        valid = (material_ids >= 0) & (material_ids < len(self._materials))
        if not np.all(valid):
            invalid = int(material_ids[np.flatnonzero(~valid)[0]])
            raise ValueError(f"material id {invalid} exceeds table size {len(self._materials)}")
        keys = mesh_slots * max(1, len(bindings)) + material_bindings[material_ids]
        transparent = (scene.colors[:, 3] >= 0) & (scene.colors[:, 3] < 1)
        opaque_rows = np.flatnonzero(~transparent)
        transparent_rows = np.flatnonzero(transparent)
        _, first, inverse = np.unique(keys[opaque_rows], return_index=True, return_inverse=True)
        first_order = np.argsort(first)
        remap = np.empty(len(first), np.int32)
        remap[first_order] = np.arange(len(first), dtype=np.int32)
        row_bucket = np.empty(n, np.int32)
        row_bucket[opaque_rows] = remap[inverse]
        row_bucket[transparent_rows] = np.arange(len(first), len(first) + len(transparent_rows))
        representatives = np.concatenate((opaque_rows[first[first_order]], transparent_rows))
        order = np.argsort(row_bucket, kind="stable")
        self.write_index = np.empty(n, np.int32)
        self.write_index[order] = np.arange(n, dtype=np.int32)
        for name in (
            "transforms",
            "colors",
            "material",
            "tex_coef",
            "cube_coef",
            "object_id",
            "segmentation",
        ):
            setattr(scene, name, getattr(scene, name)[order])
        scene.bucket = row_bucket[order]
        scene.infinite_planes = tuple(int(self.write_index[i]) for i in scene.infinite_planes)
        bounds = np.searchsorted(scene.bucket, np.arange(len(representatives) + 1))
        scene.bucket_ranges = tuple(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))
        scene.bucket_keys = tuple((meshes[i], int(material_ids[i])) for i in representatives)
        scene.opaque_buckets = tuple(range(len(first)))
        scene.transparent_buckets = tuple(range(len(first), len(representatives)))
        scene.materials = tuple(self._materials)
        scene.validate()
        return scene
