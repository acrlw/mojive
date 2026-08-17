from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..types import DEFAULT_MATERIAL, InstancePoseSource, InstanceVisual, MeshKey, MeshShape
from .scene import RenderScene, SceneBuilder

if TYPE_CHECKING:
    from ..adapters.base import SceneFrame, SceneSource
    from ..types import CameraView

try:
    from .mesh import builtin_mesh
except ImportError:
    builtin_mesh = None


# |---|---|---|---|
# | 1 | 0.5 | 0.5 | 1.0 |
# | 2 | 1.0 | 1.0 | 2.0 |
# | 4 | 2.0 | 2.0 | 4.0 |


TEXUNIFORM_SCALE = 0.5


def texuniform_coef(extent_u: float, extent_v: float, repeat: np.ndarray) -> tuple[float, float]:

    return (
        TEXUNIFORM_SCALE * float(repeat[0]) * float(extent_u),
        TEXUNIFORM_SCALE * float(repeat[1]) * float(extent_v),
    )


@dataclass(frozen=True)
class BuilderStats:
    instances: int = 0
    buckets: int = 0
    triangles: int = 0
    hidden: int = 0
    notes: tuple[str, ...] = ()


@dataclass
class _InfinitePlane:
    row: int

    slot: int

    axis_x: bool
    axis_y: bool

    period_u: float
    period_v: float

    repeat_u: float
    repeat_v: float
    half_x: float = 0.0
    half_y: float = 0.0


class SceneSourceBuilder:
    """Build a RenderScene from stable source data and the current frame."""

    def __init__(self) -> None:
        self._source: SceneSource | None = None
        self._scene = RenderScene()
        self._write_index = np.zeros(0, np.intp)
        self._src_geom = np.zeros(0, np.intp)
        self._source_instances = np.zeros(0, np.intp)
        self._base_colors = np.zeros((0, 4), np.float32)
        self._color_stage = np.zeros((0, 4), np.float32)
        self._colors_overridden = False

        self._overrides: dict[int, bool] = {}
        self._show_static = True
        self._show_skin = True
        self._show_flex_face = False
        self._show_flex_skin = True
        self._show_island = False
        self._show_convex_hull = False
        self._planes: list[_InfinitePlane] = []
        self._tri_counts: dict[MeshKey, int] = {}
        self._notes: tuple[str, ...] = ()
        self._hidden_count = 0
        self._geom_count = 0
        self._geom_sources = np.zeros(0, np.intp)
        self._site_rows = np.zeros(0, np.intp)
        self._site_sources = np.zeros(0, np.intp)
        self._site_rot = np.zeros((0, 3, 3), np.float32)
        self._site_pos = np.zeros((0, 3), np.float32)
        self._world_rows = np.zeros(0, np.intp)
        self._identity3 = np.eye(3, dtype=np.float32)

        self._w_rot = np.zeros((0, 3, 3), np.float32)
        self._w_pos = np.zeros((0, 3), np.float32)
        self._ls_rot = np.zeros((0, 3, 3), np.float32)

        self._ls_pos = np.zeros((0, 3), np.float32)
        self._out_rot = np.zeros((0, 3, 3), np.float32)
        self._out_pos = np.zeros((0, 3), np.float32)
        self._stage = np.zeros((0, 4, 4), np.float32)

    @property
    def scene(self) -> RenderScene:
        return self._scene

    @property
    def source(self) -> SceneSource | None:
        return self._source

    @property
    def write_index(self) -> np.ndarray:

        return self._write_index

    def set_source(self, source: SceneSource, camera: CameraView | None = None) -> RenderScene:

        self._source = source
        self._overrides = {}
        return self._build(camera)

    def set_visible(self, node_id: int, visible: bool) -> bool:

        if self._source is None:
            return False
        if not any(n.node_id == node_id for n in self._source.nodes):
            return False
        if self._overrides.get(node_id, True) == visible:
            return False
        self._overrides[node_id] = visible
        self._build(self._scene.camera)
        return True

    def rebuild(self, camera: CameraView | None = None) -> RenderScene:
        return self._build(camera if camera is not None else self._scene.camera)

    def set_visual_options(
        self,
        *,
        static: bool,
        skin: bool,
        flex_face: bool,
        flex_skin: bool,
        island: bool = False,
        convex_hull: bool = False,
    ) -> bool:
        options = (
            bool(static),
            bool(skin),
            bool(flex_face),
            bool(flex_skin),
            bool(island),
            bool(convex_hull),
        )
        current = (
            self._show_static,
            self._show_skin,
            self._show_flex_face,
            self._show_flex_skin,
            self._show_island,
            self._show_convex_hull,
        )
        if options == current:
            return False
        (
            self._show_static,
            self._show_skin,
            self._show_flex_face,
            self._show_flex_skin,
            self._show_island,
            self._show_convex_hull,
        ) = options
        self.rebuild()
        return True

    def _build(self, camera: CameraView | None) -> RenderScene:
        src = self._source
        if src is None:
            self._scene = RenderScene()
            return self._scene

        keep = self._visible_instances()
        self._hidden_count = int(np.count_nonzero(~keep))
        materials = src.materials or [DEFAULT_MATERIAL]
        untextured = tuple(replace(material, texture=None) for material in materials)

        sb = SceneBuilder()
        slots: list[int] = []
        source_instances: list[int] = []
        pose_sources: list[int] = []
        ls: list[np.ndarray] = []
        planes: list[_InfinitePlane] = []

        for i in range(src.instance_count):
            if not keep[i]:
                continue
            mat_index = src.geom_material[i] if i < len(src.geom_material) else 0
            mat = materials[mat_index] if 0 <= mat_index < len(materials) else DEFAULT_MATERIAL
            rgba = src.geom_rgba[i]
            if (
                self._show_island
                and i < len(src.instance_island_body)
                and int(src.instance_island_body[i]) >= 0
            ):
                mat = untextured[mat_index] if 0 <= mat_index < len(untextured) else untextured[0]
                rgba = rgba.copy()
                rgba[3] = 1.0
            matid = sb.material_id(mat)
            size = np.asarray(src.geom_size[i], np.float32)
            key: MeshKey = (
                src.geom_convex_mesh[i]
                if self._show_convex_hull and len(src.geom_convex_mesh) == src.instance_count
                else src.geom_mesh[i]
            )
            infinite = bool(src.geom_infinite_plane[i]) if len(src.geom_infinite_plane) else False

            local = (
                np.asarray(src.geom_local[i], np.float32)
                if len(src.geom_local) > i
                else np.eye(4, dtype=np.float32)
            )

            ls_i = local.copy()
            ls_i[:3, :3] = local[:3, :3] * size[None, :]

            tex = self._tex_coef(mat, size, infinite, key, local)
            slot = sb.add(
                mesh=key,
                matid=matid,
                transform=ls_i,
                color=self._linear_color(rgba),
                material=np.array(
                    [
                        mat.emission,
                        mat.specular,
                        mat.shininess,
                        mat.reflectance if key.shape is MeshShape.PLANE else 0.0,
                    ],
                    np.float32,
                ),
                object_id=int(src.geom_object_id[i]),
                tex_coef=tex,
                infinite_plane=infinite,
            )
            slots.append(int(src.geom_source[i]) if len(src.geom_source) > i else i)
            source_instances.append(i)
            pose_sources.append(
                int(src.geom_pose_source[i])
                if len(src.geom_pose_source) > i
                else int(InstancePoseSource.GEOM)
            )
            ls.append(ls_i)
            if infinite:
                if not np.allclose(local, np.eye(4)):
                    raise ValueError("infinite planes must use identity local transforms")
                repeat = np.asarray(mat.tex_repeat, np.float32)

                pu = 2.0 / max(float(repeat[0]), 1e-6)
                pv = 2.0 / max(float(repeat[1]), 1e-6)
                planes.append(
                    _InfinitePlane(
                        row=0,
                        slot=slot,
                        axis_x=float(size[0]) == 0.0,
                        axis_y=float(size[1]) == 0.0,
                        period_u=pu,
                        period_v=pv,
                        repeat_u=float(repeat[0]),
                        repeat_v=float(repeat[1]),
                    )
                )

        cam = camera if camera is not None else self._scene.camera
        lights = src.lights
        self._scene = sb.build(cam, lights, src.scene_extent, src.scene_center, src.shadow_clip)
        self._write_index = sb.write_index.astype(np.intp)
        self._source_instances = np.asarray(source_instances, np.intp)
        self._base_colors = self._scene.colors.copy()
        self._color_stage = np.zeros((self._scene.count, 4), np.float32)
        self._colors_overridden = False

        n = self._scene.count
        self._src_geom = np.array(slots, np.intp) if n else np.zeros(0, np.intp)
        pose = np.asarray(pose_sources, np.uint8)
        self._site_rows = np.flatnonzero(pose == int(InstancePoseSource.SITE))
        self._world_rows = np.flatnonzero(pose == int(InstancePoseSource.WORLD))
        self._site_sources = self._src_geom[self._site_rows]
        self._geom_sources = self._src_geom[pose == int(InstancePoseSource.GEOM)]
        self._site_rot = np.zeros((len(self._site_rows), 3, 3), np.float32)
        self._site_pos = np.zeros((len(self._site_rows), 3), np.float32)
        self._ls_rot = np.stack([m[:3, :3] for m in ls]) if n else np.zeros((0, 3, 3), np.float32)
        self._ls_pos = np.stack([m[:3, 3] for m in ls]) if n else np.zeros((0, 3), np.float32)
        self._w_rot = np.zeros((n, 3, 3), np.float32)
        self._w_pos = np.zeros((n, 3), np.float32)
        self._out_rot = np.zeros((n, 3, 3), np.float32)
        self._out_pos = np.zeros((n, 3), np.float32)
        self._stage = np.zeros((n, 4, 4), np.float32)
        self._stage[:, 3, 3] = 1.0
        if self._world_rows.size:
            self._w_rot[self._world_rows] = self._identity3

        for p in planes:
            p.row = int(self._write_index[p.slot])
        self._planes = planes

        for p in self._planes:
            if p.axis_x:
                p.half_x = _snap_up(float(cam.far), p.period_u)
                self._ls_rot[p.slot, 0, 0] = p.half_x
                self._scene.tex_coef[p.row, 0] = TEXUNIFORM_SCALE * p.repeat_u * 2.0 * p.half_x
            if p.axis_y:
                p.half_y = _snap_up(float(cam.far), p.period_v)
                self._ls_rot[p.slot, 1, 1] = p.half_y
                self._scene.tex_coef[p.row, 1] = TEXUNIFORM_SCALE * p.repeat_v * 2.0 * p.half_y
            self._scene.transforms[p.row, 0, 0] = p.half_x
            self._scene.transforms[p.row, 1, 1] = p.half_y
        self._tri_counts = self._mesh_triangles()
        self._geom_count = 0
        return self._scene

    def _visible_instances(self) -> np.ndarray:

        src = self._source
        n = src.instance_count
        keep = np.ones(n, bool)
        if len(src.geom_static) == n and not self._show_static:
            keep &= ~src.geom_static
        if len(src.geom_visual) == n:
            visual = src.geom_visual
            keep &= (visual != int(InstanceVisual.SKIN)) | self._show_skin
            keep &= (visual != int(InstanceVisual.FLEX_SKIN)) | self._show_flex_skin
            show_face = self._show_flex_face and not self._show_flex_skin
            keep &= (visual != int(InstanceVisual.FLEX_FACE)) | show_face
        if not src.nodes or n == 0:
            return keep

        by_id = {node.node_id: node for node in src.nodes}
        cache: dict[int, bool] = {}

        def effective(node_id: int) -> bool:
            if node_id in cache:
                return cache[node_id]
            node = by_id.get(node_id)
            if node is None:
                return True
            own = node.visible and self._overrides.get(node_id, True)
            cache[node_id] = own and (node.parent < 0 or effective(node.parent))
            return cache[node_id]

        geom_nodes: dict[int, list[int]] = {}
        body_nodes: dict[int, int] = {}
        for node in src.nodes:
            if node.kind == "geom":
                geom_nodes.setdefault(node.body_index, []).append(node.node_id)
            elif node.kind in ("world", "robot", "link"):
                body_nodes.setdefault(node.body_index, node.node_id)

        order: dict[int, dict[int, int]] = {}
        for i in range(n):
            if len(src.geom_node) > i and int(src.geom_node[i]) >= 0:
                keep[i] &= effective(int(src.geom_node[i]))
                continue
            body = int(src.geom_body[i]) if len(src.geom_body) > i else -1
            gsrc = int(src.geom_source[i]) if len(src.geom_source) > i else i
            seen = order.setdefault(body, {})
            if gsrc not in seen:
                seen[gsrc] = len(seen)
            k = seen[gsrc]
            candidates = geom_nodes.get(body, ())
            if k < len(candidates):
                keep[i] &= effective(candidates[k])
            elif body in body_nodes:
                keep[i] &= effective(body_nodes[body])
        return keep

    def _tex_coef(
        self, mat, size: np.ndarray, infinite: bool, key: MeshKey, local: np.ndarray
    ) -> np.ndarray:
        """Return texture scale and offset as (u_scale, v_scale, u_offset, v_offset)."""
        if mat.texture is None:
            coef = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
        elif not mat.tex_uniform and not infinite:
            repeat = np.asarray(mat.tex_repeat, np.float32)
            coef = np.array([repeat[0], repeat[1], 0.0, 0.0], np.float32)
        else:
            repeat = np.asarray(mat.tex_repeat, np.float32)
            u, v = texuniform_coef(2.0 * float(size[0]), 2.0 * float(size[1]), repeat)
            coef = np.array([u, v, 0.0, 0.0], np.float32)

        if key.shape is MeshShape.CAPSULE_CAP and float(local[2, 2]) < 0.0:
            coef[3] = coef[1]
            coef[1] = -coef[1]
        return coef

    @staticmethod
    def _linear_color(rgba) -> np.ndarray:

        c = np.asarray(rgba, np.float32).reshape(4).copy()
        c[:3] = np.power(np.clip(c[:3], 0.0, 1.0), 2.2, dtype=np.float32)
        return c

    def _mesh_triangles(self) -> dict[MeshKey, int]:

        counts: dict[MeshKey, int] = {}
        notes: list[str] = []
        src = self._source
        if src is not None:
            for key, data in src.meshes.items():
                counts[key] = data.triangle_count
        if builtin_mesh is None:
            notes.append("built-in mesh triangles are unavailable in statistics")
        else:
            for key, _matid in self._scene.bucket_keys:
                if key in counts:
                    continue
                try:
                    counts[key] = builtin_mesh(key).triangle_count
                except KeyError as exc:
                    notes.append(str(exc))
        self._notes = tuple(notes)
        return counts

    def update(
        self,
        frame: SceneFrame,
        camera: CameraView | None = None,
        instance_rgba: np.ndarray | None = None,
    ) -> RenderScene:

        scene = self._scene
        if camera is not None:
            scene.camera = camera
        if frame.lights is not None:
            scene.lights = frame.lights
        self._update_colors(instance_rgba)
        if scene.count == 0 or frame.geom_xpos is None or frame.geom_xmat is None:
            return scene

        xpos, xmat = frame.geom_xpos, frame.geom_xmat
        if len(xpos) != self._geom_count:
            self._geom_count = len(xpos)
            if self._geom_sources.size and int(self._geom_sources.max()) >= len(xpos):
                raise ValueError(
                    f"frame contains {len(xpos)} geoms; scene references geom "
                    f"{int(self._geom_sources.max())}"
                )

        if len(xpos):
            np.take(xmat, self._src_geom, axis=0, out=self._w_rot, mode="clip")
            np.take(xpos, self._src_geom, axis=0, out=self._w_pos, mode="clip")
        else:
            self._w_rot.fill(0.0)
            self._w_pos.fill(0.0)
        if self._site_rows.size and frame.site_xpos is not None and frame.site_xmat is not None:
            np.take(frame.site_xmat, self._site_sources, axis=0, out=self._site_rot, mode="clip")
            np.take(frame.site_xpos, self._site_sources, axis=0, out=self._site_pos, mode="clip")
            self._w_rot[self._site_rows] = self._site_rot
            self._w_pos[self._site_rows] = self._site_pos
        if self._world_rows.size:
            self._w_rot[self._world_rows] = self._identity3
            self._w_pos[self._world_rows] = 0.0

        if self._planes:
            self._update_infinite_planes(scene)

        np.matmul(self._w_rot, self._ls_rot, out=self._out_rot)
        np.matmul(self._w_rot, self._ls_pos[:, :, None], out=self._out_pos[:, :, None])
        np.add(self._out_pos, self._w_pos, out=self._out_pos)
        self._stage[:, :3, :3] = self._out_rot
        self._stage[:, :3, 3] = self._out_pos

        scene.transforms[self._write_index] = self._stage
        return scene

    def _update_colors(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            if self._colors_overridden:
                np.copyto(self._scene.colors, self._base_colors)
                self._colors_overridden = False
            return
        np.take(rgba, self._source_instances, axis=0, out=self._color_stage)
        np.power(
            np.clip(self._color_stage[:, :3], 0.0, 1.0),
            2.2,
            out=self._color_stage[:, :3],
        )
        self._scene.colors[self._write_index] = self._color_stage
        self._colors_overridden = True

    def _update_infinite_planes(self, scene: RenderScene) -> None:

        cam = scene.camera
        far = float(cam.far)
        ex, ey, ez = (float(v) for v in np.asarray(cam.eye, np.float32).reshape(3))
        for p in self._planes:
            j = p.slot
            dx = ex - self._w_pos.item(j, 0)
            dy = ey - self._w_pos.item(j, 1)
            dz = ez - self._w_pos.item(j, 2)

            lx = (
                self._w_rot.item(j, 0, 0) * dx
                + self._w_rot.item(j, 1, 0) * dy
                + self._w_rot.item(j, 2, 0) * dz
            )
            ly = (
                self._w_rot.item(j, 0, 1) * dx
                + self._w_rot.item(j, 1, 1) * dy
                + self._w_rot.item(j, 2, 1) * dz
            )
            row = p.row
            if p.axis_x:
                p.half_x = _snap_up(abs(lx) + far, p.period_u)
                self._ls_rot[j, 0, 0] = p.half_x
                scene.tex_coef[row, 0] = TEXUNIFORM_SCALE * p.repeat_u * 2.0 * p.half_x
            if p.axis_y:
                p.half_y = _snap_up(abs(ly) + far, p.period_v)
                self._ls_rot[j, 1, 1] = p.half_y
                scene.tex_coef[row, 1] = TEXUNIFORM_SCALE * p.repeat_v * 2.0 * p.half_y

    def stats(self) -> BuilderStats:
        scene = self._scene
        return BuilderStats(
            instances=scene.count,
            buckets=scene.bucket_count(),
            triangles=scene.triangle_count(self._tri_counts),
            hidden=self._hidden_count,
            notes=self._notes,
        )

    def infinite_plane_half_extents(self) -> tuple[tuple[float, float], ...]:

        return tuple((p.half_x, p.half_y) for p in self._planes)

    def infinite_plane_periods(self) -> tuple[tuple[float, float], ...]:
        return tuple((p.period_u, p.period_v) for p in self._planes)


def _snap_up(value: float, period: float) -> float:

    if period <= 0.0:
        return value
    return float(np.ceil(value / period) * period)
