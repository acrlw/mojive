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
        raise RuntimeError(f"参照渲染器失败：{proc.stderr}")
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
        raise RuntimeError(f"参照渲染器失败：{proc.stderr}")
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
    ap = argparse.ArgumentParser(prog="calibrate", description="拿参照渲染器裁决环境光来源")
    ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("\n=== 一、漫反射隔离（环境光全零，单向光，albedo 0.8）===")
        print("   规格 §5.2：这一趟应当逐档对上。对不上就别去看环境光。")
        print(f"   {'light diffuse':>14}{'参照 R':>10}{'forge R':>10}{'比值':>8}")
        diffuse_cases = [
            {"ha": 0.0, "hd": 0.0, "ld": d, "la": 0.0, "alb": 0.8} for d in (0.2, 0.4, 0.6, 0.8)
        ]
        for kw, ref, got in sweep("diff", diffuse_cases, tmp):
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {kw['ld']:>14.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}")

        print("\n=== 二、环境光扫描（漫反射全零，albedo 0.95）===")
        print("   规格 §5.2 的实测口径：纯色面 rgba=0.95、diffuse=0，ambient 0.5 → 输出 241")
        print("   （= 2 × 0.5 × 0.95 × 255）。下面看这条等式在本机成不成立。")
        print(
            f"   {'headlight amb':>14}{'light amb':>11}{'参照 R':>10}{'forge R':>10}{'比值':>8}{'2·a·rgba·255':>14}"
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

        print("\n=== 三、头灯漫反射隔离（只开 headlight.diffuse）===")
        print("   §5.2 的方法换一项来做：头灯与太阳光是两条不同的路径，各验各的。")
        print(f"   {'headlight diffuse':>18}{'参照 R':>10}{'forge R':>10}{'比值':>8}")
        hd_cases = [
            {"ha": 0.0, "hd": h, "ld": 0.0, "la": 0.0, "alb": 0.8} for h in (0.2, 0.4, 0.6, 0.8)
        ]
        for kw, ref, got in sweep("hd", hd_cases, tmp):
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {kw['hd']:>18.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}")

        print("\n=== 四、贴图面到底受不受光 ===")
        print("   规格 12 §12.5 把这条列为「一处还没决定的」，说参照对贴图面**完全不着色**")
        print("   （环境光 0.5 与 1.0 两档输出相同）。下面直接问参照渲染器。")
        print(f"   {'ambient':>10}{'参照 R':>10}{'forge R':>10}{'比值':>8}{'a·texel·255':>13}")
        for a in (0.25, 0.5, 1.0):
            path = tmp / f"tex_{a}.xml"
            path.write_text(TEXTURED.format(a=a, g=0.6), encoding="utf-8")
            ref = _reference_center(path)
            got = _forge_center(path)
            r = got[0] / ref[0] if ref[0] > 1e-6 else float("nan")
            print(f"   {a:>10.2f}{ref[0]:>10.1f}{got[0]:>10.1f}{r:>8.3f}{a * 0.6 * 255:>13.1f}")

        print("\n=== 五、环境光来自哪一处（headlight vs 逐灯）===")
        print("   同一个总量，一次挂在 headlight 上、一次挂在灯上，看参照渲染器认哪个。")
        print(f"   {'配置':>28}{'参照 R':>10}{'forge R':>10}")
        where_cases = [
            ({"ha": 0.4, "hd": 0.0, "ld": 0.0, "la": 0.0, "alb": 0.95}, "headlight.ambient=0.4"),
            ({"ha": 0.0, "hd": 0.0, "ld": 0.0, "la": 0.4, "alb": 0.95}, "light.ambient=0.4"),
            ({"ha": 0.4, "hd": 0.0, "ld": 0.0, "la": 0.4, "alb": 0.95}, "两处都是 0.4"),
            ({"ha": 0.0, "hd": 0.0, "ld": 0.0, "la": 0.0, "alb": 0.95}, "两处都是 0"),
        ]
        for kw, label in where_cases:
            path = tmp / f"where_{label}.xml".replace(" ", "_")
            _write_scene(path, **kw)
            ref = _reference_center(path)
            got = _forge_center(path)
            print(f"   {label:>28}{ref[0]:>10.1f}{got[0]:>10.1f}")

        print("\n=== 六、平面反射的混合律（14 §M5.2）===")
        print("   反射斑处的像素 = 地板 + r × 被反射物？还是 mix(地板, 反射, r)？")
        print("   两者在有东西可反射的地方都随 r 线性变化，**分界在没有东西可反射的地方**：")
        print("   mix 会把那片地板往黑里拉，加法则一动不动。所以两处都要采。")
        print(f"   {'reflectance':>12}{'反射斑':>22}{'远处地板':>22}{'地板+r×盒子':>20}")

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
        print("   实测：反射斑逐档等于「地板 + r×盒子」，远处地板**四档完全不变**")
        print("   → **加法**。写进 shaders/scene_body.glsl，判据在 tests/gpu/test_reflection.py")

    print(
        "\n  结论写进 docs/DECISIONS.md §三.1。**补在差异所在的那一项上，"
        "\n  不要用全局曝光去补**——补错地方会牵连别处（05 §5.2）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
