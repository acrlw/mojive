"""Render scene construction from stable sources and dynamic frames."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..types import (
    DEFAULT_MATERIAL,
    GeometryRole,
    GeometryStyle,
    GeometryView,
    InstancePoseSource,
    InstanceVisual,
    MeshKey,
    MeshShape,
    ShadingModel,
    TextureType,
)
from .scene import RenderScene, SceneBuilder

if TYPE_CHECKING:
    from ..adapters.base import SceneFrame, SceneSource
    from ..types import CameraView

try:
    from .mesh import builtin_mesh
except ImportError:
    builtin_mesh = None


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
        self._preserved_color_rows = np.zeros(0, np.intp)
        self._preserved_colors = np.zeros((0, 4), np.float32)
        self._color_stage = np.zeros((0, 4), np.float32)
        self._colors_overridden = False

        self._overrides: dict[int, bool] = {}
        self._show_static = True
        self._show_skin = True
        self._show_flex_face = False
        self._show_flex_skin = True
        self._show_island = False
        self._show_convex_hull = False
        self._geometry_view = 0
        self._geometry_style = GeometryStyle()
        self._last_frame = None
        self._last_instance_rgba = None
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
        self._structure_revision = 0
        self._pose_revision = 0
        self._visual_revision = 0
        self._identity_revision = 0

        self._w_rot = np.zeros((0, 3, 3), np.float32)
        self._w_pos = np.zeros((0, 3), np.float32)
        self._ls_rot = np.zeros((0, 3, 3), np.float32)
        self._ls_pos = np.zeros((0, 3), np.float32)
        self._ls_scale = np.zeros((0, 3), np.float32)
        self._local_rotation_diagonal = False
        self._local_position_zero = False
        self._all_static = False
        self._last_w_rot = np.zeros((0, 3, 3), np.float32)
        self._last_w_pos = np.zeros((0, 3), np.float32)
        self._world_pose_valid = False
        self._out_rot = np.zeros((0, 3, 3), np.float32)
        self._out_pos = np.zeros((0, 3), np.float32)
        self._stage = np.zeros((0, 4, 4), np.float32)
        self._last_stage = np.zeros((0, 4, 4), np.float32)
        self._pose_valid = False

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
        self._last_frame = None
        self._last_instance_rgba = None
        return self._build(camera)

    def set_visible(self, node_id: int, visible: bool) -> bool:
        if self._source is None:
            return False
        if not any(n.node_id == node_id for n in self._source.nodes):
            return False
        if self._overrides.get(node_id, True) == visible:
            return False
        self._overrides[node_id] = visible
        self.rebuild()
        return True

    def rebuild(self, camera: CameraView | None = None) -> RenderScene:
        scene = self._build(camera if camera is not None else self._scene.camera)
        if self._last_frame is not None:
            return self.update(self._last_frame, scene.camera, self._last_instance_rgba)
        return scene

    def set_visual_options(
        self,
        *,
        static: bool,
        skin: bool,
        flex_face: bool,
        flex_skin: bool,
        island: bool = False,
        convex_hull: bool = False,
        visual_geometry: bool = False,
        collision_geometry: bool = False,
        geometry_style: GeometryStyle | None = None,
    ) -> bool:
        options = (
            bool(static),
            bool(skin),
            bool(flex_face),
            bool(flex_skin),
            bool(island),
            bool(convex_hull),
            int(visual_geometry) | (int(collision_geometry) << 1),
            self._geometry_style if geometry_style is None else geometry_style,
        )
        current = (
            self._show_static,
            self._show_skin,
            self._show_flex_face,
            self._show_flex_skin,
            self._show_island,
            self._show_convex_hull,
            self._geometry_view,
            self._geometry_style,
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
            self._geometry_view,
            self._geometry_style,
        ) = options
        self.rebuild()
        return True

    def _build(self, camera: CameraView | None) -> RenderScene:
        src = self._source
        if src is None:
            self._scene = RenderScene()
            return self._scene

        self._instance_views = self._geometry_views()
        keep = self._visible_instances()
        self._hidden_count = int(np.count_nonzero(~keep))
        materials = src.materials or [DEFAULT_MATERIAL]
        untextured = tuple(replace(material, texture=None) for material in materials)
        colors = self._linear_color(src.geom_rgba)
        style = self._geometry_style
        collision_rgba = (*style.collision_color, 1.0)
        collision_color = self._linear_color(collision_rgba)
        count = src.instance_count
        if len(src.geom_local) >= count:
            local_transforms = np.asarray(src.geom_local[:count], np.float32)
        else:
            local_transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy()
            local_transforms[: len(src.geom_local)] = src.geom_local
        scaled_transforms = local_transforms.copy()
        scaled_transforms[:, :3, :3] *= np.asarray(src.geom_size, np.float32)[:, None, :]
        material_values: dict[tuple[int, bool], np.ndarray] = {}
        default_tex = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
        reflected_tex = np.array([1.0, -1.0, 0.0, 1.0], np.float32)
        no_cube = np.zeros(4, np.float32)

        sb = SceneBuilder()
        slots: list[int] = []
        source_instances: list[int] = []
        pose_sources: list[int] = []
        ls: list[np.ndarray] = []
        planes: list[_InfinitePlane] = []

        for i, collision in self._geometry_instances(keep):
            view = int(self._instance_views[i])
            mat_index = src.geom_material[i] if i < len(src.geom_material) else 0
            mat = materials[mat_index] if 0 <= mat_index < len(materials) else DEFAULT_MATERIAL
            rgba = src.geom_rgba[i]
            color = colors[i]
            if collision:
                mat = DEFAULT_MATERIAL
                rgba = np.array(collision_rgba, np.float32)
                color = collision_color
                if view == int(GeometryRole.BOTH):
                    rgba[3] = style.collision_opacity
            elif view == int(GeometryRole.BOTH) and (
                len(src.geom_role) == src.instance_count
                and (
                    int(src.geom_role[i]) == int(GeometryRole.VISUAL)
                    or (
                        len(src.geom_collision_mesh) == src.instance_count
                        and src.geom_collision_mesh[i] != src.geom_mesh[i]
                    )
                )
            ):
                rgba = rgba.copy()
                rgba[3] *= style.visual_opacity
            if (
                self._show_island
                and not view
                and i < len(src.instance_island_body)
                and int(src.instance_island_body[i]) >= 0
            ):
                mat = untextured[mat_index] if 0 <= mat_index < len(untextured) else untextured[0]
                rgba = rgba.copy()
                rgba[3] = 1.0
            matid = sb.material_id(mat)
            size = np.asarray(src.geom_size[i], np.float32)
            key: MeshKey = src.geom_mesh[i]
            if collision and len(src.geom_collision_mesh) == src.instance_count:
                key = src.geom_collision_mesh[i]
            elif (
                not view
                and self._show_convex_hull
                and len(src.geom_convex_mesh) == src.instance_count
            ):
                key = src.geom_convex_mesh[i]
            infinite = bool(src.geom_infinite_plane[i]) if len(src.geom_infinite_plane) else False

            local = local_transforms[i]
            ls_i = scaled_transforms[i]
            if (
                collision
                and view == int(GeometryRole.BOTH)
                and (int(src.geom_role[i]) & int(GeometryRole.VISUAL))
            ):
                # Shared mesh/hull faces are often coplanar. Separate only the
                # translucent comparison overlay; collision-only stays exact.
                ls_i = ls_i.copy()
                ls_i[:3, :3] *= 1.005

            if mat.texture is None:
                tex = (
                    reflected_tex
                    if key.shape is MeshShape.CAPSULE_CAP and float(local[2, 2]) < 0.0
                    else default_tex
                )
                cube = no_cube
            else:
                tex = self._tex_coef(mat, size, infinite, key, local)
                cube = self._cube_coef(src, mat, size, key, local)
            planar = key.shape in (MeshShape.PLANE, MeshShape.BOX)
            material_key = (matid, planar)
            if material_key not in material_values:
                material_values[material_key] = np.array(
                    [mat.emission, mat.specular, mat.shininess, mat.reflectance if planar else 0.0],
                    np.float32,
                )
            if color[3] != rgba[3]:
                color = color.copy()
                color[3] = rgba[3]
            slot = sb.add(
                mesh=key,
                matid=matid,
                transform=ls_i,
                color=color,
                material=material_values[material_key],
                object_id=int(src.geom_object_id[i]),
                segmentation=(
                    src.geom_segmentation[i]
                    if len(src.geom_segmentation) == src.instance_count
                    else (-1, -1)
                ),
                tex_coef=tex,
                cube_coef=cube,
                infinite_plane=infinite,
                coverage=view == int(GeometryRole.BOTH) and float(rgba[3]) < 1,
                collision_coverage=collision,
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
        self._scene = sb.build(
            cam,
            lights,
            src.scene_extent,
            src.scene_center,
            src.shadow_clip,
            getattr(src, "shading_model", ShadingModel.LINEAR),
        )
        self._write_index = sb.write_index.astype(np.intp)
        self._source_instances = np.asarray(source_instances, np.intp)
        self._base_colors = self._scene.colors.copy()
        self._preserved_color_rows = np.flatnonzero(self._instance_views[self._source_instances])
        self._preserved_colors = self._base_colors[self._write_index[self._preserved_color_rows]]
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
        local_matrices = np.asarray(ls, np.float32).reshape(n, 4, 4)
        self._ls_rot = local_matrices[:, :3, :3].copy()
        self._ls_pos = local_matrices[:, :3, 3].copy()
        self._ls_scale = (
            np.diagonal(self._ls_rot, axis1=1, axis2=2).copy()
            if n
            else np.zeros((0, 3), np.float32)
        )
        # Primitive size and capsule cap reflection are diagonal local transforms.
        # Record that invariant once so the frame path can use column scaling.
        off_diagonal = ~np.eye(3, dtype=bool)
        self._local_rotation_diagonal = bool(n == 0 or not np.any(self._ls_rot[:, off_diagonal]))
        self._local_position_zero = bool(n == 0 or not np.any(self._ls_pos))
        self._all_static = bool(
            n
            and len(src.geom_static) == src.instance_count
            and np.all(src.geom_static[self._source_instances])
        )
        self._w_rot = np.zeros((n, 3, 3), np.float32)
        self._w_pos = np.zeros((n, 3), np.float32)
        self._last_w_rot = np.zeros((n, 3, 3), np.float32)
        self._last_w_pos = np.zeros((n, 3), np.float32)
        self._world_pose_valid = False
        self._out_rot = np.zeros((n, 3, 3), np.float32)
        self._out_pos = np.zeros((n, 3), np.float32)
        self._stage = np.zeros((n, 4, 4), np.float32)
        self._stage[:, 3, 3] = 1.0
        self._last_stage = np.zeros_like(self._stage)
        self._pose_valid = False
        if self._world_rows.size:
            self._w_rot[self._world_rows] = self._identity3

        for p in planes:
            p.row = int(self._write_index[p.slot])
        self._planes = planes

        for p in self._planes:
            if p.axis_x:
                p.half_x = _snap_up(float(cam.far), p.period_u)
                self._ls_rot[p.slot, 0, 0] = p.half_x
                self._ls_scale[p.slot, 0] = p.half_x
                self._scene.tex_coef[p.row, 0] = TEXUNIFORM_SCALE * p.repeat_u * 2.0 * p.half_x
            if p.axis_y:
                p.half_y = _snap_up(float(cam.far), p.period_v)
                self._ls_rot[p.slot, 1, 1] = p.half_y
                self._ls_scale[p.slot, 1] = p.half_y
                self._scene.tex_coef[p.row, 1] = TEXUNIFORM_SCALE * p.repeat_v * 2.0 * p.half_y
            self._scene.transforms[p.row, 0, 0] = p.half_x
            self._scene.transforms[p.row, 1, 1] = p.half_y
        self._structure_revision += 1
        self._pose_revision += 1
        self._visual_revision += 1
        self._identity_revision += 1
        self._scene.structure_revision = self._structure_revision
        self._scene.pose_revision = self._pose_revision
        self._scene.visual_revision = self._visual_revision
        self._scene.identity_revision = self._identity_revision
        self._tri_counts = self._mesh_triangles()
        self._geom_count = 0
        return self._scene

    def _geometry_views(self) -> np.ndarray:
        src = self._source
        views = np.full(src.instance_count, self._geometry_view, np.uint8)
        overrides = {node.node_id: node.geometry_view for node in src.nodes if node.geometry_view}
        if not overrides:
            return views
        by_id = {node.node_id: node for node in src.nodes}
        modes = tuple(GeometryView)
        for i, node_id in enumerate(src.geom_node):
            node = by_id.get(int(node_id))
            if node is not None:
                view = overrides.get(node.node_id, overrides.get(node.parent))
                if view is not None:
                    views[i] = modes.index(GeometryView(view))
        return views

    def _geometry_instances(self, keep):
        src = self._source
        for i in np.flatnonzero(keep):
            view = int(self._instance_views[i])
            if view == int(GeometryRole.COLLISION):
                yield i, True
            elif view != int(GeometryRole.BOTH):
                yield i, False
            else:
                role = int(src.geom_role[i]) if len(src.geom_role) else int(GeometryRole.VISUAL)
                distinct_mesh = (
                    len(src.geom_collision_mesh) == src.instance_count
                    and src.geom_collision_mesh[i] != src.geom_mesh[i]
                )
                # An invisible/shared primitive still needs an opaque collision shape.
                shared_transparent = (
                    role == int(GeometryRole.BOTH) and not distinct_mesh and src.geom_rgba[i, 3] < 1
                )
                if role & int(GeometryRole.VISUAL) and not shared_transparent:
                    yield i, False
                if role & int(GeometryRole.COLLISION) and (
                    not role & int(GeometryRole.VISUAL) or distinct_mesh or shared_transparent
                ):
                    yield i, True

    def _visible_instances(self) -> np.ndarray:
        src = self._source
        n = src.instance_count
        keep = np.ones(n, bool)
        explicit = self._instance_views != 0
        roles = src.geom_role if len(src.geom_role) == n else int(GeometryRole.VISUAL)
        keep &= ~explicit | ((roles & self._instance_views) != 0)
        if len(src.geom_group_visible) == n:
            keep &= explicit | src.geom_group_visible
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
            path: set[int] = set()
            current = node_id
            visible = True
            while current >= 0 and current not in cache:
                if current in path:
                    raise ValueError(f"Cycle in scene node parents at node {current}")
                node = by_id.get(current)
                if node is None:
                    break
                path.add(current)
                if not node.visible or not self._overrides.get(current, True):
                    visible = False
                    break
                current = node.parent
            visible = visible and cache.get(current, True)
            for current in path:
                cache[current] = visible
            return visible

        geom_nodes: dict[int, list[int]] = {}
        body_nodes: dict[int, int] = {}
        for node in src.nodes:
            if node.type == "geom":
                geom_nodes.setdefault(node.body_index, []).append(node.node_id)
            elif node.type in ("world", "robot", "link"):
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
        """Return texture mapping parameters for one primitive instance."""
        src = self._source
        texture = src.textures.get(mat.texture) if src is not None else None
        if (
            src is not None
            and src.shading_model == ShadingModel.MUJOCO_CLASSIC
            and texture is not None
            and texture.type is TextureType.TWO_D
            and not infinite
            and key.shape
            in {
                MeshShape.PLANE,
                MeshShape.BOX,
                MeshShape.SPHERE,
                MeshShape.CYLINDER,
                MeshShape.CAPSULE_CAP,
            }
        ):
            # MuJoCo's generated 2D coordinates project object X/Y. The half-unit
            # phase is constant, independent of repetition, and T reverses Y.
            scale = np.asarray(mat.tex_repeat, np.float32) * np.array([0.5, -0.5], np.float32)
            if mat.tex_uniform:
                scale *= size[:2]
            if key.shape is MeshShape.CAPSULE_CAP and float(local[2, 2]) < 0.0:
                scale[1] *= -1.0
            return np.array([scale[0], scale[1], 2.0, 0.0], np.float32)
        if mat.texture is None:
            coef = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
        elif not mat.tex_uniform and not infinite:
            repeat = np.asarray(mat.tex_repeat, np.float32)
            coef = np.array([repeat[0], repeat[1], 0.0, 0.0], np.float32)
        else:
            repeat = np.asarray(mat.tex_repeat, np.float32)
            u, v = texuniform_coef(2.0 * float(size[0]), 2.0 * float(size[1]), repeat)
            box_axes = 1.0 if key.shape is MeshShape.BOX else 0.0
            coef = np.array([u, v, box_axes, 0.0], np.float32)

        if key.shape is MeshShape.CAPSULE_CAP and float(local[2, 2]) < 0.0:
            coef[3] = coef[1]
            coef[1] = -coef[1]
        return coef

    @staticmethod
    def _cube_coef(src, mat, size: np.ndarray, key: MeshKey, local: np.ndarray) -> np.ndarray:
        """Return MuJoCo object-linear cubemap coordinates for one primitive part."""
        texture = src.textures.get(mat.texture) if mat.texture is not None else None
        if texture is None or texture.type is not TextureType.CUBE:
            return np.zeros(4, np.float32)

        scale = np.asarray(size if mat.tex_uniform else np.ones(3), np.float32).copy()
        offset = 0.0
        if key.shape is MeshShape.CAPSULE_CAP:
            cap_offset = float(local[2, 3])
            if mat.tex_uniform:
                offset = cap_offset
            else:
                radius = max(abs(float(size[0])), 1e-7)
                offset = cap_offset / radius
            if float(local[2, 2]) < 0.0:
                scale[1:] *= -1.0
        return np.array([scale[0], scale[1], scale[2], offset], np.float32)

    @staticmethod
    def _linear_color(rgba) -> np.ndarray:
        c = np.asarray(rgba, np.float32).copy()
        c[..., :3] = np.power(np.clip(c[..., :3], 0.0, 1.0), 2.2, dtype=np.float32)
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
        self._last_frame = frame
        self._last_instance_rgba = instance_rgba
        scene = self._scene
        if camera is not None:
            scene.camera = camera
        if frame.lights is not None:
            scene.lights = frame.lights
        visual_changed = self._update_colors(instance_rgba)
        if scene.count == 0 or frame.geom_xpos is None or frame.geom_xmat is None:
            if visual_changed:
                self._visual_revision += 1
                scene.visual_revision = self._visual_revision
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

        plane_changed = False
        if self._planes:
            plane_changed = self._update_infinite_planes(scene)
            visual_changed = plane_changed or visual_changed

        if self._all_static:
            # Static inputs normally repeat forever, but previews may still change
            # their authoritative frame poses. Compare those poses before skipping.
            world_pose_changed = (
                not self._world_pose_valid
                or not np.array_equal(self._last_w_rot, self._w_rot)
                or not np.array_equal(self._last_w_pos, self._w_pos)
            )
            if not world_pose_changed and not plane_changed:
                if visual_changed:
                    self._visual_revision += 1
                    scene.visual_revision = self._visual_revision
                return scene
            np.copyto(self._last_w_rot, self._w_rot)
            np.copyto(self._last_w_pos, self._w_pos)
            self._world_pose_valid = True

        if self._local_rotation_diagonal:
            np.multiply(self._w_rot, self._ls_scale[:, None, :], out=self._out_rot)
        else:
            np.matmul(self._w_rot, self._ls_rot, out=self._out_rot)
        if self._local_position_zero:
            np.copyto(self._out_pos, self._w_pos)
        else:
            np.matmul(self._w_rot, self._ls_pos[:, :, None], out=self._out_pos[:, :, None])
            np.add(self._out_pos, self._w_pos, out=self._out_pos)
        self._stage[:, :3, :3] = self._out_rot
        self._stage[:, :3, 3] = self._out_pos

        pose_changed = not self._pose_valid or not np.array_equal(self._last_stage, self._stage)
        if pose_changed:
            scene.transforms[self._write_index] = self._stage
            np.copyto(self._last_stage, self._stage)
            self._pose_valid = True
            self._pose_revision += 1
            scene.pose_revision = self._pose_revision
        if visual_changed:
            self._visual_revision += 1
            scene.visual_revision = self._visual_revision
        return scene

    def _update_colors(self, rgba: np.ndarray | None) -> bool:
        if rgba is None or len(self._preserved_color_rows) == len(self._color_stage):
            if self._colors_overridden:
                np.copyto(self._scene.colors, self._base_colors)
                self._colors_overridden = False
                return True
            return False
        np.take(rgba, self._source_instances, axis=0, out=self._color_stage)
        np.power(
            np.clip(self._color_stage[:, :3], 0.0, 1.0),
            2.2,
            out=self._color_stage[:, :3],
        )
        self._color_stage[self._preserved_color_rows] = self._preserved_colors
        changed = not np.array_equal(self._scene.colors[self._write_index], self._color_stage)
        if changed:
            self._scene.colors[self._write_index] = self._color_stage
        self._colors_overridden = True
        return changed

    def _update_infinite_planes(self, scene: RenderScene) -> bool:
        cam = scene.camera
        far = float(cam.far)
        ex, ey, ez = (float(v) for v in np.asarray(cam.eye, np.float32).reshape(3))
        changed = False
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
                half_x = _snap_up(abs(lx) + far, p.period_u)
                changed = changed or half_x != p.half_x
                p.half_x = half_x
                self._ls_rot[j, 0, 0] = p.half_x
                self._ls_scale[j, 0] = p.half_x
                scene.tex_coef[row, 0] = TEXUNIFORM_SCALE * p.repeat_u * 2.0 * p.half_x
            if p.axis_y:
                half_y = _snap_up(abs(ly) + far, p.period_v)
                changed = changed or half_y != p.half_y
                p.half_y = half_y
                self._ls_rot[j, 1, 1] = p.half_y
                self._ls_scale[j, 1] = p.half_y
                scene.tex_coef[row, 1] = TEXUNIFORM_SCALE * p.repeat_v * 2.0 * p.half_y
        return changed

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
