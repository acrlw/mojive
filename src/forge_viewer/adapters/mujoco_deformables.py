"""MuJoCo flex and skin mesh construction and updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import math3d
from ..types import InstanceVisual, MeshData, MeshKey, MeshShape, MeshUpdate

_CABLE_SIDES = 10


def _normalize_rows(values: np.ndarray, scratch: np.ndarray, lengths: np.ndarray) -> None:
    np.multiply(values, values, out=scratch)
    np.sum(scratch, axis=1, out=lengths)
    np.sqrt(lengths, out=lengths)
    np.maximum(lengths, 1e-12, out=lengths)
    values /= lengths[:, None]


def _cross_rows(a: np.ndarray, b: np.ndarray, out: np.ndarray, scratch: np.ndarray) -> None:
    np.multiply(a[:, 1], b[:, 2], out=out[:, 0])
    np.multiply(a[:, 2], b[:, 1], out=scratch[:, 0])
    out[:, 0] -= scratch[:, 0]
    np.multiply(a[:, 2], b[:, 0], out=out[:, 1])
    np.multiply(a[:, 0], b[:, 2], out=scratch[:, 1])
    out[:, 1] -= scratch[:, 1]
    np.multiply(a[:, 0], b[:, 1], out=out[:, 2])
    np.multiply(a[:, 1], b[:, 0], out=scratch[:, 2])
    out[:, 2] -= scratch[:, 2]


@dataclass
class DeformableMesh:
    key: MeshKey
    mesh: MeshData
    update_data: MeshUpdate
    group: int
    matid: int
    rgba: np.ndarray
    visual: InstanceVisual

    def update(self, data) -> None:
        raise NotImplementedError


class _SurfaceFlex(DeformableMesh):
    def __init__(self, model, data, flex_id: int, *, smooth: bool) -> None:
        self.flex_id = flex_id
        dim = int(model.flex_dim[flex_id])
        self._dim = dim
        nv = int(model.flex_vertnum[flex_id])
        elem_adr = int(model.flex_elemdataadr[flex_id])
        nelem = int(model.flex_elemnum[flex_id])
        elements = np.asarray(
            model.flex_elem[elem_adr : elem_adr + nelem * (dim + 1)], np.int32
        ).reshape(nelem, dim + 1)
        shell_adr = int(model.flex_shelldataadr[flex_id])
        nshell = int(model.flex_shellnum[flex_id])
        shells = np.asarray(
            model.flex_shell[shell_adr : shell_adr + nshell * dim], np.int32
        ).reshape(nshell, dim)
        self._shells = np.ascontiguousarray(shells, np.int32)
        self._nelem = nelem

        self._smooth = smooth
        if dim == 2 and smooth:
            top = elements[:, :3]
            bottom = top[:, (0, 2, 1)]
            side_a = np.stack([shells[:, 0], shells[:, 1], shells[:, 1]], axis=1)
            side_b = np.stack([shells[:, 1], shells[:, 0], shells[:, 0]], axis=1)
            faces = np.concatenate(
                [
                    np.stack([top, bottom], axis=1).reshape(-1, 3),
                    np.stack([side_a, side_b], axis=1).reshape(-1, 3),
                ]
            )
            top_sign = np.ones_like(top, np.float32)
            bottom_sign = -np.ones_like(bottom, np.float32)
            side_a_sign = np.tile(np.array([1.0, -1.0, 1.0], np.float32), (nshell, 1))
            side_b_sign = np.tile(np.array([-1.0, 1.0, -1.0], np.float32), (nshell, 1))
            signs = np.concatenate(
                [
                    np.stack([top_sign, bottom_sign], axis=1).reshape(-1, 3),
                    np.stack([side_a_sign, side_b_sign], axis=1).reshape(-1, 3),
                ]
            )
            normal_faces = top
        elif dim == 3 and smooth:
            faces = shells[:, :3]
            signs = np.ones_like(faces, np.float32)
            normal_faces = faces
        elif dim == 2:
            top = elements[:, :3]
            bottom = top[:, (0, 2, 1)]
            faces = np.stack((top, bottom), axis=1).reshape(-1, 3)
            signs = np.ones_like(faces, np.float32)
            normal_faces = faces
        else:
            elem_adr = int(model.flex_elemadr[flex_id])
            layers = np.asarray(model.flex_elemlayer[elem_adr : elem_adr + nelem])
            tetra = elements[layers == 0]
            faces = np.stack(
                (
                    tetra[:, (0, 1, 2)],
                    tetra[:, (0, 2, 3)],
                    tetra[:, (0, 3, 1)],
                    tetra[:, (1, 3, 2)],
                ),
                axis=1,
            ).reshape(-1, 3)
            signs = np.ones_like(faces, np.float32)
            normal_faces = faces

        self._faces = np.ascontiguousarray(faces, np.int32)
        self._corner_ids = self._faces.reshape(-1)
        self._signs = signs.reshape(-1, 1)
        self._normal_faces = np.ascontiguousarray(normal_faces, np.int32)
        self._radius = float(model.flex_radius[flex_id])
        self._flat_skin = bool(model.flex_flatskin[flex_id])
        self._vert_adr = int(model.flex_vertadr[flex_id])

        ncorner, nface = len(self._corner_ids), len(self._faces)
        self._base = np.zeros((nv, 3), np.float32)
        self._vertex_normals = np.zeros((nv, 3), np.float32)
        self._vertex_scratch = np.zeros((nv, 3), np.float32)
        self._vertex_lengths = np.zeros(nv, np.float32)
        self._normal_p0 = np.zeros((len(normal_faces), 3), np.float32)
        self._normal_e1 = np.zeros_like(self._normal_p0)
        self._normal_e2 = np.zeros_like(self._normal_p0)
        self._normal_cross = np.zeros_like(self._normal_p0)
        self._normal_scratch = np.zeros_like(self._normal_p0)
        self._normal_lengths = np.zeros(len(normal_faces), np.float32)
        self._side_p0 = np.zeros((nshell, 3), np.float32)
        self._side_edge = np.zeros((nshell, 3), np.float32)
        self._side_ref = np.zeros((nshell, 3), np.float32)
        self._side_normal = np.zeros((nshell, 3), np.float32)
        self._side_scratch = np.zeros((nshell, 3), np.float32)
        self._side_lengths = np.zeros(nshell, np.float32)
        self._corner_scratch = np.zeros((ncorner, 3), np.float32)
        positions = np.zeros((ncorner, 3), np.float32)
        normals = np.zeros_like(positions)
        self._tri_normals = normals.reshape(nface, 3, 3)

        uvs = self._uvs(model, flex_id, elements, shells, dim)
        mesh = MeshData(positions, normals, uvs, np.arange(ncorner, dtype=np.uint32))
        super().__init__(
            key=MeshKey(MeshShape.FLEX if smooth else MeshShape.FLEX_FACE, flex_id),
            mesh=mesh,
            update_data=MeshUpdate(positions, normals),
            group=int(model.flex_group[flex_id]),
            matid=int(model.flex_matid[flex_id]),
            rgba=np.asarray(model.flex_rgba[flex_id], np.float32).copy(),
            visual=InstanceVisual.FLEX_SKIN if smooth else InstanceVisual.FLEX_FACE,
        )
        self.update(data)

    def _uvs(self, model, flex_id, elements, shells, dim: int) -> np.ndarray:
        adr = int(model.flex_texcoordadr[flex_id])
        if adr < 0:
            return np.zeros((len(self._corner_ids), 2), np.float32)
        count = int(model.flex_vertnum[flex_id])
        tex = np.asarray(model.flex_texcoord[adr : adr + count], np.float32)
        if dim == 3:
            return np.ascontiguousarray(tex[self._corner_ids], np.float32)
        eadr = int(model.flex_elemdataadr[flex_id])
        tids = np.asarray(
            model.flex_elemtexcoord[eadr : eadr + len(elements) * 3], np.int32
        ).reshape(-1, 3)
        bottom = tids[:, (0, 2, 1)]
        if not self._smooth:
            ids = np.stack((tids, bottom), axis=1).reshape(-1)
            return np.ascontiguousarray(tex[ids], np.float32)
        side_a = np.stack([shells[:, 0], shells[:, 1], shells[:, 1]], axis=1)
        side_b = np.stack([shells[:, 1], shells[:, 0], shells[:, 0]], axis=1)
        ids = np.concatenate(
            [
                np.stack([tids, bottom], axis=1).reshape(-1, 3),
                np.stack([side_a, side_b], axis=1).reshape(-1, 3),
            ]
        ).reshape(-1)
        return np.ascontiguousarray(tex[ids], np.float32)

    def update(self, data) -> None:
        src = data.flexvert_xpos[self._vert_adr : self._vert_adr + len(self._base)]
        np.copyto(self._base, src, casting="unsafe")
        self._vertex_normals.fill(0.0)
        tri = self._normal_faces
        np.take(self._base, tri[:, 0], axis=0, out=self._normal_p0)
        np.take(self._base, tri[:, 1], axis=0, out=self._normal_e1)
        np.take(self._base, tri[:, 2], axis=0, out=self._normal_e2)
        self._normal_e1 -= self._normal_p0
        self._normal_e2 -= self._normal_p0
        _cross_rows(self._normal_e1, self._normal_e2, self._normal_cross, self._normal_p0)
        _normalize_rows(self._normal_cross, self._normal_scratch, self._normal_lengths)
        if not self._smooth:
            np.take(self._base, self._corner_ids, axis=0, out=self.mesh.positions)
            self._tri_normals[:] = self._normal_cross[:, None, :]
            if self._radius:
                np.multiply(self.mesh.normals, self._radius, out=self._corner_scratch)
                np.add(self.mesh.positions, self._corner_scratch, out=self.mesh.positions)
            return
        for corner in range(3):
            np.add.at(self._vertex_normals, tri[:, corner], self._normal_cross)
        _normalize_rows(self._vertex_normals, self._vertex_scratch, self._vertex_lengths)

        np.take(self._base, self._corner_ids, axis=0, out=self.mesh.positions)
        np.take(self._vertex_normals, self._corner_ids, axis=0, out=self.mesh.normals)
        np.multiply(self.mesh.normals, self._signs, out=self.mesh.normals)
        if self._radius:
            np.multiply(self.mesh.normals, self._radius, out=self._corner_scratch)
            np.add(self.mesh.positions, self._corner_scratch, out=self.mesh.positions)

        if self._dim == 3:
            if self._flat_skin:
                self._tri_normals[:] = self._normal_cross[:, None, :]
            return

        if self._flat_skin:
            self._tri_normals[: 2 * self._nelem : 2] = self._normal_cross[:, None, :]
            self._tri_normals[1 : 2 * self._nelem : 2] = -self._normal_cross[:, None, :]

        side_start = 2 * self._nelem
        np.take(self._base, self._shells[:, 0], axis=0, out=self._side_p0)
        np.take(self._base, self._shells[:, 1], axis=0, out=self._side_edge)
        self._side_edge -= self._side_p0
        np.take(self._vertex_normals, self._shells[:, 1], axis=0, out=self._side_ref)
        _cross_rows(self._side_edge, self._side_ref, self._side_normal, self._side_scratch)
        _normalize_rows(self._side_normal, self._side_scratch, self._side_lengths)
        self._tri_normals[side_start::2] = self._side_normal[:, None, :]
        np.take(self._vertex_normals, self._shells[:, 0], axis=0, out=self._side_ref)
        _cross_rows(self._side_edge, self._side_ref, self._side_normal, self._side_scratch)
        _normalize_rows(self._side_normal, self._side_scratch, self._side_lengths)
        self._tri_normals[side_start + 1 :: 2] = self._side_normal[:, None, :]


class _CableFlex(DeformableMesh):
    def __init__(self, model, data, flex_id: int) -> None:
        self.flex_id = flex_id
        self._vert_adr = int(model.flex_vertadr[flex_id])
        nv = int(model.flex_vertnum[flex_id])
        edge_adr = int(model.flex_edgeadr[flex_id])
        ne = int(model.flex_edgenum[flex_id])
        self._edges = np.asarray(model.flex_edge[edge_adr : edge_adr + ne], np.int32).copy()
        self._radius = float(model.flex_radius[flex_id])
        self._base = np.zeros((nv, 3), np.float32)
        self._p0 = np.zeros((ne, 3), np.float32)
        self._p1 = np.zeros((ne, 3), np.float32)
        self._tangent = np.zeros((ne, 3), np.float32)
        self._u = np.zeros((ne, 3), np.float32)
        self._v = np.zeros((ne, 3), np.float32)
        self._ref = np.zeros((ne, 3), np.float32)
        self._scratch = np.zeros((ne, 3), np.float32)
        self._scratch2 = np.zeros((ne, 3), np.float32)
        self._lengths = np.zeros(ne, np.float32)
        self._near_z = np.zeros(ne, bool)
        self._angles = np.linspace(0.0, 2.0 * np.pi, _CABLE_SIDES, endpoint=False)

        positions = np.zeros((ne * 2 * _CABLE_SIDES, 3), np.float32)
        normals = np.zeros_like(positions)
        self._rings = positions.reshape(ne, 2, _CABLE_SIDES, 3)
        self._normal_rings = normals.reshape(ne, 2, _CABLE_SIDES, 3)
        u = np.tile(np.arange(_CABLE_SIDES, dtype=np.float32) / _CABLE_SIDES, ne * 2)
        v = np.tile(np.repeat(np.array([0.0, 1.0], np.float32), _CABLE_SIDES), ne)
        indices: list[int] = []
        for edge in range(ne):
            base = edge * 2 * _CABLE_SIDES
            for side in range(_CABLE_SIDES):
                nxt = (side + 1) % _CABLE_SIDES
                a, b = base + side, base + nxt
                c, d = base + _CABLE_SIDES + side, base + _CABLE_SIDES + nxt
                indices.extend((a, c, d, a, d, b))
        mesh = MeshData(
            positions,
            normals,
            np.stack([u, v], axis=1),
            np.asarray(indices, np.uint32),
        )
        super().__init__(
            key=MeshKey(MeshShape.FLEX, flex_id),
            mesh=mesh,
            update_data=MeshUpdate(positions, normals),
            group=int(model.flex_group[flex_id]),
            matid=int(model.flex_matid[flex_id]),
            rgba=np.asarray(model.flex_rgba[flex_id], np.float32).copy(),
            visual=InstanceVisual.FLEX_EDGE,
        )
        self.update(data)

    def update(self, data) -> None:
        src = data.flexvert_xpos[self._vert_adr : self._vert_adr + len(self._base)]
        np.copyto(self._base, src, casting="unsafe")
        np.take(self._base, self._edges[:, 0], axis=0, out=self._p0)
        np.take(self._base, self._edges[:, 1], axis=0, out=self._p1)
        np.subtract(self._p1, self._p0, out=self._tangent)
        _normalize_rows(self._tangent, self._scratch, self._lengths)
        self._ref.fill(0.0)
        self._ref[:, 2] = 1.0
        np.abs(self._tangent[:, 2], out=self._lengths)
        np.greater(self._lengths, 0.9, out=self._near_z)
        self._ref[self._near_z, 1] = 1.0
        self._ref[self._near_z, 2] = 0.0
        _cross_rows(self._tangent, self._ref, self._u, self._scratch)
        _normalize_rows(self._u, self._scratch, self._lengths)
        _cross_rows(self._tangent, self._u, self._v, self._scratch)
        for side, angle in enumerate(self._angles):
            np.multiply(self._u, np.cos(angle), out=self._scratch)
            np.multiply(self._v, np.sin(angle), out=self._scratch2)
            self._scratch += self._scratch2
            self._normal_rings[:, 0, side] = self._scratch
            self._normal_rings[:, 1, side] = self._scratch
            np.multiply(self._scratch, self._radius, out=self._scratch2)
            np.add(self._p0, self._scratch2, out=self._rings[:, 0, side])
            np.add(self._p1, self._scratch2, out=self._rings[:, 1, side])


class _Skin(DeformableMesh):
    def __init__(self, model, data, skin_id: int) -> None:
        self.skin_id = skin_id
        va, nv = int(model.skin_vertadr[skin_id]), int(model.skin_vertnum[skin_id])
        fa, nf = int(model.skin_faceadr[skin_id]), int(model.skin_facenum[skin_id])
        self._base = np.asarray(model.skin_vert[va : va + nv], np.float32).copy()
        self._faces = np.asarray(model.skin_face[fa : fa + nf], np.int32).copy()
        positions = np.zeros((nv, 3), np.float32)
        normals = np.zeros_like(positions)
        ta = int(model.skin_texcoordadr[skin_id])
        uvs = (
            np.asarray(model.skin_texcoord[ta : ta + nv], np.float32).copy()
            if ta >= 0
            else np.zeros((nv, 2), np.float32)
        )
        mesh = MeshData(positions, normals, uvs, self._faces.reshape(-1).astype(np.uint32))

        bones = range(
            int(model.skin_boneadr[skin_id]),
            int(model.skin_boneadr[skin_id]) + int(model.skin_bonenum[skin_id]),
        )
        self._bones: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        max_vertices = 0
        for bone in bones:
            adr, count = int(model.skin_bonevertadr[bone]), int(model.skin_bonevertnum[bone])
            ids = np.asarray(model.skin_bonevertid[adr : adr + count], np.int32).copy()
            weights = np.asarray(model.skin_bonevertweight[adr : adr + count], np.float32).copy()
            bind_pos = np.asarray(model.skin_bonebindpos[bone], np.float32).copy()
            bind_rot = math3d.quat_to_mat3(model.skin_bonebindquat[bone])
            self._bones.append(
                (int(model.skin_bonebodyid[bone]), ids, weights[:, None], bind_pos, bind_rot)
            )
            max_vertices = max(max_vertices, count)
        self._bone_src = np.zeros((max_vertices, 3), np.float32)
        self._bone_dst = np.zeros_like(self._bone_src)
        self._rot = np.zeros((3, 3), np.float32)
        self._translate = np.zeros(3, np.float32)
        self._face_p0 = np.zeros((nf, 3), np.float32)
        self._face_e1 = np.zeros((nf, 3), np.float32)
        self._face_e2 = np.zeros((nf, 3), np.float32)
        self._face_normals = np.zeros((nf, 3), np.float32)
        self._normal_scratch = np.zeros((nv, 3), np.float32)
        self._normal_lengths = np.zeros(nv, np.float32)
        self._inflate_scratch = np.zeros((nv, 3), np.float32)
        self._inflate = float(model.skin_inflate[skin_id])
        self._xmat = data.xmat.reshape(-1, 3, 3)
        super().__init__(
            key=MeshKey(MeshShape.SKIN, skin_id),
            mesh=mesh,
            update_data=MeshUpdate(positions, normals),
            group=int(model.skin_group[skin_id]),
            matid=int(model.skin_matid[skin_id]),
            rgba=np.asarray(model.skin_rgba[skin_id], np.float32).copy(),
            visual=InstanceVisual.SKIN,
        )
        self.update(data)

    def update(self, data) -> None:
        self.mesh.positions.fill(0.0)
        for body, ids, weights, bind_pos, bind_rot in self._bones:
            n = len(ids)
            np.matmul(self._xmat[body], bind_rot.T, out=self._rot)
            np.matmul(self._rot, bind_pos, out=self._translate)
            self._translate *= -1.0
            self._translate += data.xpos[body]
            np.take(self._base, ids, axis=0, out=self._bone_src[:n])
            np.matmul(self._bone_src[:n], self._rot.T, out=self._bone_dst[:n])
            self._bone_dst[:n] += self._translate
            self._bone_dst[:n] *= weights
            np.add.at(self.mesh.positions, ids, self._bone_dst[:n])

        self.mesh.normals.fill(0.0)
        tri = self._faces
        np.take(self.mesh.positions, tri[:, 0], axis=0, out=self._face_p0)
        np.take(self.mesh.positions, tri[:, 1], axis=0, out=self._face_e1)
        np.take(self.mesh.positions, tri[:, 2], axis=0, out=self._face_e2)
        self._face_e1 -= self._face_p0
        self._face_e2 -= self._face_p0
        _cross_rows(self._face_e1, self._face_e2, self._face_normals, self._face_p0)
        for corner in range(3):
            np.add.at(self.mesh.normals, tri[:, corner], self._face_normals)
        _normalize_rows(self.mesh.normals, self._normal_scratch, self._normal_lengths)
        if self._inflate:
            np.multiply(self.mesh.normals, self._inflate, out=self._inflate_scratch)
            np.add(self.mesh.positions, self._inflate_scratch, out=self.mesh.positions)


def build_deformables(
    model, data, visible_flex_groups: set[int], visible_skin_groups: set[int]
) -> list[DeformableMesh]:
    result: list[DeformableMesh] = []
    for flex_id in range(model.nflex):
        if int(model.flex_group[flex_id]) not in visible_flex_groups:
            continue
        if int(model.flex_dim[flex_id]) == 1:
            result.append(_CableFlex(model, data, flex_id))
        else:
            result.append(_SurfaceFlex(model, data, flex_id, smooth=True))
            result.append(_SurfaceFlex(model, data, flex_id, smooth=False))
    for skin_id in range(model.nskin):
        if int(model.skin_group[skin_id]) in visible_skin_groups:
            result.append(_Skin(model, data, skin_id))
    return result


def update_deformables(specs: list[DeformableMesh], data) -> None:
    for spec in specs:
        spec.update(data)
