"""Calibrate Forge output against reference images."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCENE = """<mujoco model="calib">
  <compiler angle="degree"/>
  <visual>
    <headlight ambient="{ha} {ha} {ha}" diffuse="{hd} {hd} {hd}" specular="0 0 0"/>
    <map znear="0.05" zfar="50"/>
    <global fovy="45" offwidth="256" offheight="256"/>
    <quality shadowsize="0"/>
  </visual>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 5" dir="0 0 -1"
           diffuse="{ld} {ld} {ld}" specular="0 0 0" ambient="{la} {la} {la}"
           castshadow="false"/>
    <geom name="panel" type="box" pos="0 0 0" size="2 2 0.05"
          rgba="{alb} {alb} {alb} 1" material=""/>
  </worldbody>
</mujoco>
"""


TEXTURED = """<mujoco model="tex">
  <compiler angle="degree"/>
  <asset>
    <texture name="flat" type="2d" builtin="flat" width="64" height="64"
             rgb1="{g} {g} {g}" rgb2="{g} {g} {g}"/>
    <material name="m" texture="flat" texrepeat="1 1" texuniform="false"/>
  </asset>
  <visual>
    <headlight ambient="{a} {a} {a}" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.05" zfar="50"/>
    <global fovy="45" offwidth="256" offheight="256"/>
    <quality shadowsize="0"/>
  </visual>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0 -1" diffuse="0 0 0" specular="0 0 0"
           ambient="0 0 0" castshadow="false"/>
    <geom type="box" pos="0 0 0" size="2 2 0.05" material="m"/>
  </worldbody>
</mujoco>"""


REFLECT = """<mujoco model="refl">
  <compiler angle="degree"/>
  <asset>
    <material name="floor" reflectance="{r}" rgba="0.35 0.35 0.35 1"/>
  </asset>
  <visual>
    <headlight ambient="0.30 0.30 0.30" diffuse="0.70 0.70 0.70" specular="0 0 0"/>
    <map znear="0.05" zfar="50"/>
    <global fovy="45" offwidth="512" offheight="512"/>
    <quality shadowsize="0"/>
  </visual>
  <worldbody>
    <light directional="true" pos="0 0 5" dir="0 0 -1"
           diffuse="0 0 0" specular="0 0 0" ambient="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"/>
    <geom name="box" type="box" pos="0 0 0.7" size="0.35 0.35 0.35" rgba="1 0.12 0.08 1"/>
  </worldbody>
</mujoco>"""

REFLECT_CAMERA = "cam.lookat[:]=[0,0,0.35]; cam.distance=4.0; cam.azimuth=90.0; cam.elevation=-18.0"


def _reference_image(path: Path, out: Path, camera: str) -> np.ndarray:
    code = (
        "import sys,numpy as np,mujoco\n"
        f"m=mujoco.MjModel.from_xml_path({str(path)!r})\n"
        "m.vis.global_.offwidth=512; m.vis.global_.offheight=512\n"
        "d=mujoco.MjData(m); mujoco.mj_forward(m,d)\n"
        "cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)\n"
        "cam.type=mujoco.mjtCamera.mjCAMERA_FREE\n"
        f"{camera}\n"
        "r=mujoco.Renderer(m,512,512); r.update_scene(d,cam); img=r.render(); r.close()\n"
        f"np.save({str(out)!r}, img)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Reference renderer failed: {proc.stderr}")
    return np.load(str(out) if str(out).endswith(".npy") else str(out) + ".npy")


def _write_scene(path: Path, *, ha: float, hd: float, ld: float, la: float, alb: float) -> None:
    path.write_text(SCENE.format(ha=ha, hd=hd, ld=ld, la=la, alb=alb), encoding="utf-8")


def _reference_center(path: Path) -> tuple[float, float, float]:
    code = (
        "import sys,json,numpy as np,mujoco\n"
        f"m=mujoco.MjModel.from_xml_path({str(path)!r})\n"
        "m.vis.global_.offwidth=256; m.vis.global_.offheight=256\n"
        "d=mujoco.MjData(m); mujoco.mj_forward(m,d)\n"
        "cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)\n"
        "cam.type=mujoco.mjtCamera.mjCAMERA_FREE\n"
        "cam.lookat[:]=[0,0,0]; cam.distance=6.0; cam.azimuth=90.0; cam.elevation=-90.0\n"
        "r=mujoco.Renderer(m,256,256); r.update_scene(d,cam); img=r.render(); r.close()\n"
        "c=img[112:144,112:144].reshape(-1,3).mean(axis=0)\n"
        "print(json.dumps([float(x) for x in c]))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Reference renderer failed: {proc.stderr}")
    return tuple(json.loads(proc.stdout))  # type: ignore[return-value]


def _forge_center(path: Path) -> tuple[float, float, float]:
    from ..types import CameraView
    from ._harness import OffscreenHarness

    with OffscreenHarness(path, 256, 256) as h:
        h.camera = CameraView(
            eye=np.array([0.0, 0.0, 6.0], np.float32),
            target=np.zeros(3, np.float32),
            up=np.array([0.0, 1.0, 0.0], np.float32),
            fov_y=float(np.radians(45.0)),
            near=0.05,
            far=50.0,
        )
        h.backend.set_camera(h.camera)
        h.warmup(4)
        h.step_and_render(0)
        img = h.backend.target.read_color(flip=True)[..., :3]
        return tuple(float(x) for x in img[112:144, 112:144].reshape(-1, 3).mean(axis=0))


def sweep(name: str, cases: list[dict], tmp: Path) -> list[tuple]:
    rows = []
    for i, kw in enumerate(cases):
        path = tmp / f"{name}_{i}.xml"
        _write_scene(path, **kw)
        ref = _reference_center(path)
        got = _forge_center(path)
        rows.append((kw, ref, got))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="calibrate", description="Calibrate forge lighting against MuJoCo"
    )
    ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("\n=== 1. Directional diffuse: ambient=0, albedo=0.8 ===")
        print(f"   {'light diffuse':>14}{'reference R':>12}{'forge R':>10}{'ratio':>8}")
        diffuse_cases = [
            {"ha": 0.0, "hd": 0.0, "ld": d, "la": 0.0, "alb": 0.8} for d in (0.2, 0.4, 0.6, 0.8)
        ]
        for kw, ref, got in sweep("diff", diffuse_cases, tmp):
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {kw['ld']:>14.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}")

        print("\n=== 2. Ambient sweep: diffuse=0, albedo=0.95 ===")
        print("   Reference relation: output = 2 × ambient × albedo × 255.")
        print(
            f"   {'headlight amb':>14}{'light amb':>11}{'reference R':>12}"
            f"{'forge R':>10}{'ratio':>8}{'prediction':>14}"
        )
        amb_cases = [
            {"ha": a, "hd": 0.0, "ld": 0.0, "la": 0.0, "alb": 0.95} for a in (0.1, 0.2, 0.35, 0.5)
        ]
        for kw, ref, got in sweep("amb", amb_cases, tmp):
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            pred = 2.0 * kw["ha"] * kw["alb"] * 255.0
            print(
                f"   {kw['ha']:>14.2f}{kw['la']:>11.2f}{ref[0]:>10.1f}{got[0]:>10.1f}"
                f"{r:>8.3f}{pred:>14.1f}"
            )

        print("\n=== 3. Headlight diffuse ===")
        print(f"   {'headlight diffuse':>18}{'reference R':>12}{'forge R':>10}{'ratio':>8}")
        hd_cases = [
            {"ha": 0.0, "hd": h, "ld": 0.0, "la": 0.0, "alb": 0.8} for h in (0.2, 0.4, 0.6, 0.8)
        ]
        for kw, ref, got in sweep("hd", hd_cases, tmp):
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {kw['hd']:>18.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}")

        print("\n=== 4. Textured surface lighting ===")
        print(
            f"   {'ambient':>10}{'reference R':>12}{'forge R':>10}{'ratio':>8}{'a×texel×255':>13}"
        )
        for a in (0.25, 0.5, 1.0):
            path = tmp / f"tex_{a}.xml"
            path.write_text(TEXTURED.format(a=a, g=0.6), encoding="utf-8")
            ref = _reference_center(path)
            got = _forge_center(path)
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {a:>10.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}{a * 0.6 * 255:>13.1f}")

        print("\n=== 5. Ambient source: headlight and scene light ===")
        print(f"   {'configuration':>28}{'reference R':>12}{'forge R':>10}")
        where_cases = [
            ({"ha": 0.4, "hd": 0.0, "ld": 0.0, "la": 0.0, "alb": 0.95}, "headlight.ambient=0.4"),
            ({"ha": 0.0, "hd": 0.0, "ld": 0.0, "la": 0.4, "alb": 0.95}, "light.ambient=0.4"),
            ({"ha": 0.4, "hd": 0.0, "ld": 0.0, "la": 0.4, "alb": 0.95}, "both=0.4"),
            ({"ha": 0.0, "hd": 0.0, "ld": 0.0, "la": 0.0, "alb": 0.95}, "both=0"),
        ]
        for kw, label in where_cases:
            path = tmp / f"where_{label}.xml".replace(" ", "_")
            _write_scene(path, **kw)
            ref = _reference_center(path)
            got = _forge_center(path)
            print(f"   {label:>28}{ref[0]:>10.1f}{got[0]:>10.1f}")

        print("\n=== 6. Planar reflection composition ===")
        print(f"   {'reflectance':>12}{'reflection':>22}{'far floor':>22}{'floor+r×box':>20}")

        shots = {}
        for i, r in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            path = tmp / f"refl_{i}.xml"
            path.write_text(REFLECT.format(r=r), encoding="utf-8")
            shots[r] = _reference_image(path, tmp / f"refl_{i}.npy", REFLECT_CAMERA).astype(float)

        base, top = shots[0.0], shots[1.0]
        diff = np.abs(top - base).sum(axis=2)
        diff[: diff.shape[0] // 2] = 0.0
        iy, ix = np.unravel_index(int(np.argmax(diff)), diff.shape)
        fy, fx = diff.shape[0] - 40, 40
        box = base[base.shape[0] // 2 - 30, base.shape[1] // 2]

        for r, img in shots.items():
            spot = img[iy, ix]
            far = img[fy, fx]
            pred = base[iy, ix] + r * box
            print(
                f"   {r:>12.2f}{spot.astype(int)!s:>22}{far.astype(int)!s:>22}"
                f"{np.minimum(pred, 255).astype(int)!s:>20}"
            )
        print("   Result: additive floor + reflectance × reflected color.")
        print("   Implementation: shaders/scene_body.glsl; gate: tests/gpu/test_reflection.py")

    print("\n  Calibration decisions are documented in docs/DECISIONS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
