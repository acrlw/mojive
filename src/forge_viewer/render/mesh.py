from __future__ import annotations

import numpy as np

from ..gizmo import (
    AXIS_HEAD_HALF_PT,
    AXIS_HEAD_LENGTH_PT,
    AXIS_SHAFT_HALF_PT,
    AXIS_START,
    PLANE_INNER,
    PLANE_OUTER,
    RING_RADIUS,
    RING_TUBE,
    SCREEN_RING_EDGE_TUBE,
    SCREEN_RING_RADIUS,
    SCREEN_RING_TUBE,
    SIZE_PT,
)
from ..types import MeshData, MeshKey, MeshShape

CIRCLE_SEGMENTS = 32


SPHERE_RINGS = 16


CAP_RINGS = SPHERE_RINGS // 2


ARROW_SEGMENTS = 16


TUBE_UV_V0, TUBE_UV_V1 = 0.0, 1.0
CAPSULE_SHAFT_V0, CAPSULE_SHAFT_V1 = 0.25, 0.75
CAPSULE_CAP_V0, CAPSULE_CAP_V1 = 0.75, 1.0


# Asset, heightfield and deformable meshes come from the loaded scene rather than a generator.
BUILTIN_SHAPES: tuple[MeshShape, ...] = tuple(
    s
    for s in MeshShape
    if s not in (MeshShape.ASSET, MeshShape.HEIGHTFIELD, MeshShape.FLEX, MeshShape.SKIN)
)


def _finish(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
) -> MeshData:

    pos = np.array(positions, np.float32, order="C")
    nrm = np.array(normals, np.float32, order="C")
    uv = np.array(uvs, np.float32, order="C")
    idx = np.array(indices, np.uint32, order="C")
    for arr in (pos, nrm, uv, idx):
        arr.flags.writeable = False
    return MeshData(positions=pos, normals=nrm, uvs=uv, indices=idx)


def _ring_angles(segments: int) -> np.ndarray:

    return np.linspace(0.0, 2.0 * np.pi, segments + 1)


def _grid_indices(rows: int, cols: int) -> np.ndarray:

    i = np.arange(rows - 1)[:, None]
    j = np.arange(cols - 1)[None, :]
    a = i * cols + j
    b = a + 1
    c = a + cols
    d = c + 1
    return np.stack([a, b, d, a, d, c], axis=-1).reshape(-1).astype(np.uint32)


def _drop_degenerate(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:

    tri = indices.reshape(-1, 3)
    p = positions[tri]
    area2 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return tri[area2 > 1e-9].reshape(-1).astype(np.uint32)


def _cylinder_side(
    z0: float, z1: float, segments: int, v0: float, v1: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    th = _ring_angles(segments)
    cos, sin = np.cos(th), np.sin(th)
    u = th / (2.0 * np.pi)
    zero = np.zeros_like(cos)

    pos = np.stack(
        [
            np.stack([cos, sin, np.full_like(cos, z0)], axis=1),
            np.stack([cos, sin, np.full_like(cos, z1)], axis=1),
        ]
    ).reshape(-1, 3)
    nrm = np.tile(np.stack([cos, sin, zero], axis=1), (2, 1))
    uvs = np.concatenate(
        [
            np.stack([u, np.full_like(u, v0)], axis=1),
            np.stack([u, np.full_like(u, v1)], axis=1),
        ]
    )
    return pos, nrm, uvs, _grid_indices(2, segments + 1)


def _cap_disk(
    z: float, segments: int, up: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    th = _ring_angles(segments)
    cos, sin = np.cos(th), np.sin(th)
    n = len(th)

    center = np.array([[0.0, 0.0, z]])
    rim = np.stack([cos, sin, np.full_like(cos, z)], axis=1)
    pos = np.concatenate([center, rim])

    axis = 1.0 if up else -1.0
    nrm = np.tile(np.array([[0.0, 0.0, axis]]), (n + 1, 1))
    su = 0.5 * axis
    uvs = np.concatenate(
        [
            np.array([[0.5, 0.5]]),
            np.stack([0.5 + su * cos, 0.5 - 0.5 * sin], axis=1),
        ]
    )

    j = np.arange(segments)
    if up:
        tri = np.stack([np.zeros_like(j), j + 1, j + 2], axis=1)
    else:
        tri = np.stack([np.zeros_like(j), j + 2, j + 1], axis=1)
    return pos, nrm, uvs, tri.reshape(-1).astype(np.uint32)


def _merge(*parts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    pos, nrm, uvs, idx, offset = [], [], [], [], 0
    for p, n, t, i in parts:
        pos.append(p)
        nrm.append(n)
        uvs.append(t)
        idx.append(np.asarray(i, np.uint32) + offset)
        offset += len(p)
    return (
        np.concatenate(pos),
        np.concatenate(nrm),
        np.concatenate(uvs),
        np.concatenate(idx).astype(np.uint32),
    )


def _sphere_band(
    phi0: float, phi1: float, rings: int, segments: int, v0: float, v1: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    phi = np.linspace(phi0, phi1, rings + 1)[:, None]
    th = _ring_angles(segments)[None, :]
    shape = (rings + 1, segments + 1)

    z = np.broadcast_to(-np.cos(phi), shape)
    r = np.sin(phi)
    x = np.broadcast_to(r * np.cos(th), shape)
    y = np.broadcast_to(r * np.sin(th), shape)
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    u = np.broadcast_to(th / (2.0 * np.pi), shape)
    v = np.broadcast_to(np.linspace(v0, v1, rings + 1)[:, None], shape)
    uvs = np.stack([u, v], axis=-1).reshape(-1, 2)

    idx = _drop_degenerate(pos, _grid_indices(rings + 1, segments + 1))
    return pos, pos.copy(), uvs, idx


def _make_sphere() -> MeshData:
    return _finish(*_sphere_band(0.0, np.pi, SPHERE_RINGS, CIRCLE_SEGMENTS, 0.0, 1.0))


def _make_box() -> MeshData:

    faces = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    )
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    pos, nrm, uvs, idx = [], [], [], []
    for f, (n, ua, va) in enumerate(faces):
        n, ua, va = np.array(n, float), np.array(ua, float), np.array(va, float)
        for s, t in corners:
            pos.append(n + s * ua + t * va)
            nrm.append(n)
            uvs.append((0.5 * (s + 1.0), 0.5 * (t + 1.0)))
        base = 4 * f
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return _finish(np.array(pos), np.array(nrm), np.array(uvs), np.array(idx, np.uint32))


def _make_plane() -> MeshData:

    quad = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
    pos = np.stack([quad[:, 0], quad[:, 1], np.zeros(4)], axis=1)
    nrm = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    s, t = 0.5 * quad[:, 0], 0.5 * quad[:, 1]
    uvs = np.stack([0.5 + s, 0.5 - t], axis=1)
    return _finish(pos, nrm, uvs, np.array([0, 1, 2, 0, 2, 3], np.uint32))


def _make_cylinder() -> MeshData:

    return _finish(
        *_merge(
            _cylinder_side(-1.0, 1.0, CIRCLE_SEGMENTS, 0.0, 1.0),
            _cap_disk(1.0, CIRCLE_SEGMENTS, up=True),
            _cap_disk(-1.0, CIRCLE_SEGMENTS, up=False),
        )
    )


def _cone_side(
    z_base: float, z_apex: float, segments: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    th = _ring_angles(segments)
    cos, sin = np.cos(th), np.sin(th)
    height = z_apex - z_base

    slope = 1.0 / height
    base_n = np.stack([cos, sin, np.full_like(cos, slope)], axis=1)
    base_n /= np.linalg.norm(base_n, axis=1, keepdims=True)
    base_p = np.stack([cos, sin, np.full_like(cos, z_base)], axis=1)
    base_uv = np.stack([th / (2.0 * np.pi), np.zeros_like(th)], axis=1)

    mid = 0.5 * (th[:-1] + th[1:])
    apex_n = np.stack([np.cos(mid), np.sin(mid), np.full_like(mid, slope)], axis=1)
    apex_n /= np.linalg.norm(apex_n, axis=1, keepdims=True)
    apex_p = np.tile(np.array([[0.0, 0.0, z_apex]]), (segments, 1))
    apex_uv = np.stack([mid / (2.0 * np.pi), np.ones_like(mid)], axis=1)

    j = np.arange(segments)
    tri = np.stack([j, j + 1, len(th) + j], axis=1).reshape(-1)
    return (
        np.concatenate([base_p, apex_p]),
        np.concatenate([base_n, apex_n]),
        np.concatenate([base_uv, apex_uv]),
        tri.astype(np.uint32),
    )


def _make_cone() -> MeshData:

    return _finish(
        *_merge(
            _cone_side(-1.0, 1.0, CIRCLE_SEGMENTS),
            _cap_disk(-1.0, CIRCLE_SEGMENTS, up=False),
        )
    )


def _make_disk() -> MeshData:

    return _finish(*_cap_disk(0.0, CIRCLE_SEGMENTS, up=True))


def _make_tube() -> MeshData:

    return _finish(*_cylinder_side(-1.0, 1.0, CIRCLE_SEGMENTS, TUBE_UV_V0, TUBE_UV_V1))


def _make_capsule_shaft() -> MeshData:

    return _finish(*_cylinder_side(-1.0, 1.0, CIRCLE_SEGMENTS, CAPSULE_SHAFT_V0, CAPSULE_SHAFT_V1))


def _make_capsule_cap() -> MeshData:

    return _finish(
        *_sphere_band(
            0.5 * np.pi, np.pi, CAP_RINGS, CIRCLE_SEGMENTS, CAPSULE_CAP_V0, CAPSULE_CAP_V1
        )
    )


def _make_arrow_shaft() -> MeshData:

    return _finish(
        *_merge(
            _cylinder_side(0.0, 1.0, ARROW_SEGMENTS, 0.0, 1.0),
            _cap_disk(1.0, ARROW_SEGMENTS, up=True),
            _cap_disk(0.0, ARROW_SEGMENTS, up=False),
        )
    )


def _make_arrow_head() -> MeshData:

    return _finish(
        *_merge(
            _cone_side(0.0, 1.0, ARROW_SEGMENTS),
            _cap_disk(0.0, ARROW_SEGMENTS, up=False),
        )
    )


_SOLID_ARROW_HEAD_BASE = 0.65
_SOLID_ARROW_SHAFT_RADIUS = 0.5
_SOLID_ARROW_HEAD_RADIUS = 1.0


def _solid_arrow_head():
    cone = _scale_xy(
        _cone_side(_SOLID_ARROW_HEAD_BASE, 1.0, ARROW_SEGMENTS),
        _SOLID_ARROW_HEAD_RADIUS,
    )
    positions, normals, uvs, indices = cone
    radial = normals[:, :2]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    slope = _SOLID_ARROW_HEAD_RADIUS / (1.0 - _SOLID_ARROW_HEAD_BASE)
    normals[:] = np.column_stack((radial, np.full(len(radial), slope)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return _merge(
        _annulus(
            _SOLID_ARROW_HEAD_BASE,
            _SOLID_ARROW_SHAFT_RADIUS,
            _SOLID_ARROW_HEAD_RADIUS,
            ARROW_SEGMENTS,
            up=False,
        ),
        (positions, normals, uvs, indices),
    )


def _mirror_z(part):
    positions, normals, uvs, indices = part
    positions = positions.copy()
    normals = normals.copy()
    positions[:, 2] *= -1.0
    normals[:, 2] *= -1.0
    indices = indices.reshape(-1, 3)[:, ::-1].reshape(-1)
    return positions, normals, uvs, indices


def _make_solid_arrow() -> MeshData:
    return _finish(
        *_merge(
            _scale_xy(
                _cylinder_side(0.0, _SOLID_ARROW_HEAD_BASE, ARROW_SEGMENTS, 0.0, 1.0),
                _SOLID_ARROW_SHAFT_RADIUS,
            ),
            _scale_xy(_cap_disk(0.0, ARROW_SEGMENTS, up=False), _SOLID_ARROW_SHAFT_RADIUS),
            _solid_arrow_head(),
        )
    )


def _make_solid_double_arrow() -> MeshData:
    positive_head = _solid_arrow_head()
    return _finish(
        *_merge(
            _scale_xy(
                _cylinder_side(
                    -_SOLID_ARROW_HEAD_BASE,
                    _SOLID_ARROW_HEAD_BASE,
                    ARROW_SEGMENTS,
                    0.0,
                    1.0,
                ),
                _SOLID_ARROW_SHAFT_RADIUS,
            ),
            positive_head,
            _mirror_z(positive_head),
        )
    )


def _scale_xy(part, radius: float):
    p, n, uv, idx = part
    p = np.asarray(p).copy()
    p[:, :2] *= float(radius)
    return p, n, uv, idx


def _annulus(z: float, inner: float, outer: float, segments: int, *, up: bool):

    th = _ring_angles(segments)
    unit = np.stack((np.cos(th), np.sin(th)), axis=1)
    pos = np.concatenate(
        (
            np.column_stack((unit * outer, np.full(len(unit), z))),
            np.column_stack((unit * inner, np.full(len(unit), z))),
        )
    )
    normal = np.array((0.0, 0.0, 1.0 if up else -1.0))
    nrm = np.tile(normal, (len(pos), 1))
    uv = np.tile(np.array((0.5, 0.5)), (len(pos), 1))
    j = np.arange(segments)
    outer_j, outer_n = j, j + 1
    inner_j, inner_n = len(unit) + j, len(unit) + j + 1
    if up:
        tri = np.stack((outer_j, outer_n, inner_n, outer_j, inner_n, inner_j), axis=1)
    else:
        tri = np.stack((outer_j, inner_n, outer_n, outer_j, inner_j, inner_n), axis=1)
    return pos, nrm, uv, tri.reshape(-1).astype(np.uint32)


def _make_gizmo_arrow() -> MeshData:

    shaft_radius = AXIS_SHAFT_HALF_PT / SIZE_PT
    head_radius = AXIS_HEAD_HALF_PT / SIZE_PT
    head_base = 1.0 - AXIS_HEAD_LENGTH_PT / SIZE_PT
    cone = _scale_xy(_cone_side(head_base, 1.0, ARROW_SEGMENTS), head_radius)
    p, n, uv, idx = cone
    slope = head_radius / (1.0 - head_base)
    side = n[:, 2] > 0.0
    xy = n[side, :2]
    xy /= np.linalg.norm(xy, axis=1, keepdims=True)
    n[side] = np.column_stack((xy, np.full(len(xy), slope)))
    n[side] /= np.linalg.norm(n[side], axis=1, keepdims=True)
    cone = p, n, uv, idx
    return _finish(
        *_merge(
            _scale_xy(
                _cylinder_side(AXIS_START, head_base, ARROW_SEGMENTS, 0.0, 1.0), shaft_radius
            ),
            _scale_xy(_cap_disk(AXIS_START, ARROW_SEGMENTS, up=False), shaft_radius),
            _annulus(head_base, shaft_radius, head_radius, ARROW_SEGMENTS, up=False),
            cone,
        )
    )


def _make_gizmo_plane() -> MeshData:
    a, b = PLANE_INNER, PLANE_OUTER
    pos = np.array(((a, a, 0.0), (b, a, 0.0), (b, b, 0.0), (a, b, 0.0)))
    nrm = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))
    uv = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    return _finish(pos, nrm, uv, np.array((0, 1, 2, 0, 2, 3), np.uint32))


def _make_gizmo_ring(
    radius: float = RING_RADIUS, *, half: bool = False, tube: float = RING_TUBE
) -> MeshData:

    major, minor = (32 if half else 64), 8
    sweep = np.pi if half else 2.0 * np.pi
    th = np.linspace(0.0, sweep, major + 1)[:, None]
    ph = np.linspace(0.0, 2.0 * np.pi, minor + 1)[None, :]
    radial = radius + tube * np.cos(ph)
    x = radial * np.cos(th)
    y = radial * np.sin(th)
    z = np.broadcast_to(tube * np.sin(ph), x.shape)
    pos = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    nrm = np.stack(
        (
            np.cos(ph) * np.cos(th),
            np.cos(ph) * np.sin(th),
            np.broadcast_to(np.sin(ph), x.shape),
        ),
        axis=-1,
    ).reshape(-1, 3)
    uv = np.stack(
        (
            np.broadcast_to(th / sweep, x.shape),
            np.broadcast_to(ph / (2.0 * np.pi), x.shape),
        ),
        axis=-1,
    ).reshape(-1, 2)
    indices = _grid_indices(major + 1, minor + 1).reshape(-1, 3)[:, ::-1].reshape(-1)
    return _finish(pos, nrm, uv, indices)


_GENERATORS = {
    MeshShape.SPHERE: _make_sphere,
    MeshShape.BOX: _make_box,
    MeshShape.PLANE: _make_plane,
    MeshShape.CYLINDER: _make_cylinder,
    MeshShape.CONE: _make_cone,
    MeshShape.DISK: _make_disk,
    MeshShape.TUBE: _make_tube,
    MeshShape.CAPSULE_SHAFT: _make_capsule_shaft,
    MeshShape.CAPSULE_CAP: _make_capsule_cap,
    MeshShape.ARROW_SHAFT: _make_arrow_shaft,
    MeshShape.ARROW_HEAD: _make_arrow_head,
    MeshShape.ARROW: _make_solid_arrow,
    MeshShape.DOUBLE_ARROW: _make_solid_double_arrow,
}


_CACHE: dict[MeshShape, MeshData] = {}
_GIZMO_CACHE: dict[str, MeshData] = {}


def builtin_mesh(key: MeshKey) -> MeshData:

    shape = key.shape
    mesh = _CACHE.get(shape)
    if mesh is None:
        gen = _GENERATORS.get(shape)
        if gen is None:
            raise KeyError(
                f"{shape!r} is not a built-in shape. Available shapes: "
                f"{[s.value for s in BUILTIN_SHAPES]}"
            )
        mesh = gen()
        _CACHE[shape] = mesh
    return mesh


def all_builtin() -> dict[MeshKey, MeshData]:

    return {MeshKey(shape=s): builtin_mesh(MeshKey(shape=s)) for s in BUILTIN_SHAPES}


def gizmo_mesh(name: str) -> MeshData:

    mesh = _GIZMO_CACHE.get(name)
    if mesh is not None:
        return mesh
    generators = {
        "arrow": _make_gizmo_arrow,
        "plane": _make_gizmo_plane,
        "ring": _make_gizmo_ring,
        "half_ring": lambda: _make_gizmo_ring(half=True),
        "screen_ring": lambda: _make_gizmo_ring(SCREEN_RING_RADIUS, tube=SCREEN_RING_TUBE),
        "screen_ring_edge": lambda: _make_gizmo_ring(
            SCREEN_RING_RADIUS, tube=SCREEN_RING_EDGE_TUBE
        ),
    }
    try:
        mesh = generators[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown gizmo mesh {name!r}. Available meshes: {tuple(generators)}"
        ) from exc
    _GIZMO_CACHE[name] = mesh
    return mesh
