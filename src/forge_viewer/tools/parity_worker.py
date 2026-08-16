from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VIEWPOINTS: tuple[tuple[str, float, float, float], ...] = (
    ("front", 90.0, -15.0, 2.6),
    ("quarter", 45.0, -25.0, 2.6),
    ("side", 0.0, -10.0, 2.6),
    ("high", 30.0, -60.0, 2.2),
    ("low", 200.0, -5.0, 3.0),
)


def render_all(scene: Path, out_dir: Path, width: int, height: int) -> dict:
    import mujoco
    from PIL import Image

    m = mujoco.MjModel.from_xml_path(str(scene))

    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(width))
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(height))

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(m, height=height, width=width)
    views: list[dict] = []
    try:
        for name, azimuth, elevation, dist_factor in VIEWPOINTS:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = m.stat.center
            cam.distance = float(dist_factor) * float(m.stat.extent)
            cam.azimuth = float(azimuth)
            cam.elevation = float(elevation)

            renderer.update_scene(d, cam)
            img = renderer.render()
            path = out_dir / f"{name}.ref.png"
            Image.fromarray(np.ascontiguousarray(img), "RGB").save(path)

            gl = renderer.scene.camera
            pos = (np.array(gl[0].pos) + np.array(gl[1].pos)) * 0.5
            fwd = np.array(gl[0].forward, dtype=np.float64)
            up = np.array(gl[0].up, dtype=np.float64)
            near = float(gl[0].frustum_near)
            far = float(gl[0].frustum_far)
            top = float(gl[0].frustum_top)
            fov_y_deg = float(np.degrees(2.0 * np.arctan2(top, near)))

            views.append(
                {
                    "name": name,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "distance": cam.distance,
                    "image": str(path),
                    "eye": pos.tolist(),
                    "forward": fwd.tolist(),
                    "up": up.tolist(),
                    "near": near,
                    "far": far,
                    "fov_y_deg": fov_y_deg,
                }
            )
    finally:
        renderer.close()

    default_rgba = np.array([0.5, 0.5, 0.5, 1.0], np.float32)
    geoms: list[dict] = []
    for g in range(m.ngeom):
        rgba = np.array(m.geom_rgba[g], np.float32)
        mat = int(m.geom_matid[g])
        if mat >= 0 and np.allclose(rgba, default_rgba):
            rgba = np.array(m.mat_rgba[mat], np.float32)
        geoms.append(
            {
                "index": g,
                "pos": np.array(d.geom_xpos[g], np.float64).tolist(),
                "rgba": rgba.tolist(),
                "size": np.array(m.geom_size[g], np.float64).tolist(),
                "type": int(m.geom_type[g]),
            }
        )

    return {
        "scene": str(scene),
        "width": width,
        "height": height,
        "extent": float(m.stat.extent),
        "center": np.array(m.stat.center).tolist(),
        "renderer": "mjr_",
        "views": views,
        "geoms": geoms,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 4:
        print("用法：parity_worker <场景> <输出目录> <宽> <高>", file=sys.stderr)
        return 2
    scene, out_dir, width, height = Path(argv[0]), Path(argv[1]), int(argv[2]), int(argv[3])
    doc = render_all(scene, out_dir, width, height)

    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
