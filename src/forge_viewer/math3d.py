from __future__ import annotations

import numpy as np

Vec3 = np.ndarray
Mat4 = np.ndarray


def identity() -> Mat4:
    return np.eye(4, dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n if n > eps else np.zeros_like(v)).astype(np.float32)


def look_at(eye, target, up) -> Mat4:

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    n = np.linalg.norm(f)
    f = f / n if n > 1e-12 else np.array([0.0, 0.0, -1.0])
    s = np.cross(f, up)
    n = np.linalg.norm(s)

    s = (
        s / n
        if n > 1e-9
        else normalize(
            np.cross(f, [1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.cross(f, [0.0, 1.0, 0.0])
        ).astype(np.float64)
    )
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m.astype(np.float32)


def perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4:

    f = 1.0 / np.tan(fov_y * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m.astype(np.float32)


def perspective_intrinsics(
    focal_length, sensor_size, principal_offset, near: float, far: float
) -> Mat4:
    focal = np.asarray(focal_length, np.float64).reshape(2)
    sensor = np.asarray(sensor_size, np.float64).reshape(2)
    principal = np.asarray(principal_offset, np.float64).reshape(2)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 * focal[0] / sensor[0]
    m[1, 1] = 2.0 * focal[1] / sensor[1]
    m[0, 2] = 2.0 * principal[0] / sensor[0]
    m[1, 2] = -2.0 * principal[1] / sensor[1]
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m.astype(np.float32)


def orthographic(height: float, aspect: float, near: float, far: float) -> Mat4:

    h = max(height, 1e-6) * 0.5
    w = h * max(aspect, 1e-6)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1.0 / w
    m[1, 1] = 1.0 / h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m.astype(np.float32)


def ortho_box(left, right, bottom, top, near, far) -> Mat4:

    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m.astype(np.float32)


def compose(position, rotation_3x3, scale) -> Mat4:

    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = np.asarray(rotation_3x3, dtype=np.float32).reshape(3, 3) * np.asarray(
        scale, dtype=np.float32
    )
    m[:3, 3] = position
    return m


def quat_to_mat3(q) -> np.ndarray:

    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 1e-12 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def axis_angle_to_mat3(axis, angle: float) -> np.ndarray:
    a = normalize(axis).astype(np.float64)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float32,
    )


def rotvec_to_mat3(rotvec) -> np.ndarray:

    rv = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float32)
    return axis_angle_to_mat3(rv / angle, angle)


def mat3_to_quat(m) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.array(q, dtype=np.float32)


def euler_xyz_to_mat3(angles) -> np.ndarray:

    x, y, z = (float(v) for v in angles)
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    return np.array(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        np.float32,
    )


def mat3_to_euler_xyz(m) -> np.ndarray:

    m = np.asarray(m, np.float64).reshape(3, 3)
    y = float(np.arcsin(np.clip(-m[2, 0], -1.0, 1.0)))
    if abs(float(np.cos(y))) > 1e-7:
        x = float(np.arctan2(m[2, 1], m[2, 2]))
        z = float(np.arctan2(m[1, 0], m[0, 0]))
    else:
        x = float(np.arctan2(-m[1, 2], m[1, 1]))
        z = 0.0
    return np.array((x, y, z), np.float32)


def transform_points(m: Mat4, pts: np.ndarray) -> np.ndarray:

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    return pts @ m[:3, :3].T + m[:3, 3]


def invert_rigid(m: Mat4) -> Mat4:

    r = m[:3, :3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ m[:3, 3]
    return out


def to_gl(m: Mat4) -> np.ndarray:

    return np.ascontiguousarray(np.asarray(m, dtype=np.float32).T)


def unproject_ray(
    ndc_x: float, ndc_y: float, view: Mat4, proj: Mat4
) -> tuple[np.ndarray, np.ndarray]:

    inv = np.linalg.inv((proj @ view).astype(np.float64))
    near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    near = near[:3] / near[3]
    far = far[:3] / far[3]
    return near.astype(np.float32), normalize(far - near)


def mirror(point, normal) -> np.ndarray:

    n = np.asarray(normal, np.float64).reshape(3)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        return np.eye(4, dtype=np.float32)
    n = n / length
    d = float(np.dot(n, np.asarray(point, np.float64).reshape(3)))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] -= 2.0 * np.outer(n, n)
    out[:3, 3] = 2.0 * d * n
    return out.astype(np.float32)


def inverse_perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:

    f = 1.0 / np.tan(fov_y * 0.5)
    a = max(aspect, 1e-6)
    n, fr = float(near), float(far)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = a / f
    m[1, 1] = 1.0 / f
    m[2, 3] = -1.0
    m[3, 2] = (n - fr) / (2.0 * fr * n)
    m[3, 3] = (n + fr) / (2.0 * fr * n)
    return m.astype(np.float32)


def inverse_orthographic_box(left, right, bottom, top, near, far) -> np.ndarray:

    rl, rr, rb, rt, rn, rf = (float(v) for v in (left, right, bottom, top, near, far))
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = (rr - rl) * 0.5
    m[1, 1] = (rt - rb) * 0.5
    m[2, 2] = (rf - rn) * -0.5
    m[0, 3] = (rr + rl) * 0.5
    m[1, 3] = (rt + rb) * 0.5
    m[2, 3] = (rf + rn) * 0.5
    m[3, 3] = 1.0
    return m.astype(np.float32)
