from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OUT = Path("out/parity")


def _gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _edges(g: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    return np.hypot(gx, gy)


def edge_iou(a: np.ndarray, b: np.ndarray, threshold: float = 24.0) -> float:

    ea = _edges(_gray(a)) > threshold
    eb = _edges(_gray(b)) > threshold
    union = np.count_nonzero(ea | eb)
    return float(np.count_nonzero(ea & eb) / union) if union else 1.0


def block_luma_diff(a: np.ndarray, b: np.ndarray, blocks: int = 16) -> float:

    ga, gb = _gray(a), _gray(b)
    h, w = ga.shape
    bh, bw = max(h // blocks, 1), max(w // blocks, 1)
    diffs = [
        abs(ga[i : i + bh, j : j + bw].mean() - gb[i : i + bh, j : j + bw].mean())
        for i in range(0, h - bh + 1, bh)
        for j in range(0, w - bw + 1, bw)
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def _project(world: np.ndarray, view: dict, width: int, height: int) -> tuple[int, int] | None:

    eye = np.asarray(view["eye"], np.float64)
    fwd = np.asarray(view["forward"], np.float64)
    up = np.asarray(view["up"], np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    rel = np.asarray(world, np.float64) - eye
    z = float(np.dot(rel, fwd))
    if z <= view["near"]:
        return None
    tan_half = np.tan(np.radians(view["fov_y_deg"]) * 0.5)
    ndc_y = float(np.dot(rel, true_up)) / (z * tan_half)
    ndc_x = float(np.dot(rel, right)) / (z * tan_half * (width / height))
    if not (-1.0 <= ndc_x <= 1.0 and -1.0 <= ndc_y <= 1.0):
        return None
    return int((ndc_x * 0.5 + 0.5) * width), int((0.5 - ndc_y * 0.5) * height)


def texture_cell_agreement(
    a: np.ndarray, b: np.ndarray, geoms: list[dict], view: dict, width: int, height: int
) -> tuple[int, int, str]:

    palette = np.array([g["rgba"][:3] for g in geoms], np.float32)
    if len(palette) == 0:
        return 0, 0, "没有可判的格子"

    def classify(img: np.ndarray, px: int, py: int) -> int | None:
        patch = img[max(py - 2, 0) : py + 3, max(px - 2, 0) : px + 3, :3].astype(np.float32)
        if patch.size == 0:
            return None
        flat = patch.reshape(-1, 3)

        if float(flat.std(axis=0).max()) > 18.0:
            return None
        mean = flat.mean(axis=0)

        norm = mean / max(float(np.linalg.norm(mean)), 1e-6)
        ref = palette / np.maximum(np.linalg.norm(palette, axis=1, keepdims=True), 1e-6)
        d = np.linalg.norm(ref - norm, axis=1)
        best = int(np.argmin(d))
        return best if d[best] < 0.25 else None

    agree = total = 0
    for g in geoms:
        hit = _project(np.asarray(g["pos"]), view, width, height)
        if hit is None:
            continue
        px, py = hit
        ca, cb = classify(a, px, py), classify(b, px, py)
        if ca is None or cb is None:
            continue
        total += 1
        agree += int(ca == cb)
    note = f"{agree}/{total}" if total else "无可判点"
    return agree, total, note


def triptych(forge: np.ndarray, reference: np.ndarray, path: Path) -> None:

    from PIL import Image

    h = min(forge.shape[0], reference.shape[0])
    w = min(forge.shape[1], reference.shape[1])
    a, b = forge[:h, :w, :3], reference[:h, :w, :3]
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([a, b, d], axis=1), "RGB").save(path)


@dataclass
class ViewScore:
    name: str
    edge_iou: float
    block_diff: float
    cells: str
    triptych: Path | None = None
    notes: list[str] = field(default_factory=list)


def run_reference(scene: Path, out_dir: Path, width: int, height: int) -> dict:

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge_viewer.tools.parity_worker",
            str(scene),
            str(out_dir),
            str(width),
            str(height),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"参照渲染器子进程失败（退出码 {proc.returncode}）：\n{proc.stderr}")
    return json.loads(proc.stdout)


def render_forge(scene: Path, doc: dict, out_dir: Path) -> dict[str, np.ndarray]:

    from PIL import Image

    from ..types import CameraView
    from ._harness import OffscreenHarness

    width, height = doc["width"], doc["height"]
    shots: dict[str, np.ndarray] = {}
    with OffscreenHarness(scene, width, height) as h:
        h.warmup(6)
        for v in doc["views"]:
            eye = np.asarray(v["eye"], np.float32)
            fwd = np.asarray(v["forward"], np.float64)
            fwd = (fwd / np.linalg.norm(fwd)).astype(np.float32)
            h.camera = CameraView(
                eye=eye,
                target=(eye + fwd * float(v["distance"])).astype(np.float32),
                up=np.asarray(v["up"], np.float32),
                fov_y=float(np.radians(v["fov_y_deg"])),
                near=float(v["near"]),
                far=float(v["far"]),
                aspect=width / height,
            )
            h.backend.set_camera(h.camera)
            h.step_and_render(0)
            img = h.backend.target.read_color(flip=True)[..., :3].copy()
            shots[v["name"]] = img
            Image.fromarray(img, "RGB").save(out_dir / f"{v['name']}.forge.png")
    return shots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parity", description="与参照渲染器同机位对拍")
    ap.add_argument("scene", nargs="?", default="parity_scene")
    ap.add_argument("-o", "--out", default=str(OUT))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    from PIL import Image

    from ..assets import resolve

    scene = resolve(args.scene)
    out_dir = Path(args.out) / scene.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = run_reference(scene, out_dir, args.width, args.height)
    forge_shots = render_forge(scene, doc, out_dir)

    scores: list[ViewScore] = []
    for v in doc["views"]:
        ref = np.asarray(Image.open(v["image"]).convert("RGB"))
        got = forge_shots[v["name"]]
        _agree, _total, cells = texture_cell_agreement(
            got, ref, doc.get("geoms", []), v, doc["width"], doc["height"]
        )
        tri = out_dir / f"{v['name']}.triptych.png"
        triptych(got, ref, tri)
        scores.append(
            ViewScore(
                name=v["name"],
                edge_iou=edge_iou(got, ref),
                block_diff=block_luma_diff(got, ref),
                cells=cells,
                triptych=tri,
            )
        )

    print(f"\n{scene.stem}   {doc['width']}×{doc['height']}   参照 = {doc['renderer']}（子进程）")
    print(f"  {'机位':<10}{'边缘 IoU':>10}{'分块亮度差':>12}{'采到哪一格':>12}")
    for s in scores:
        print(f"  {s.name:<10}{s.edge_iou:>10.3f}{s.block_diff:>12.1f}{s.cells:>12}")
    iou = float(np.mean([s.edge_iou for s in scores]))
    blk = float(np.mean([s.block_diff for s in scores]))
    agree_n = sum(int(s.cells.split("/")[0]) for s in scores if "/" in s.cells)
    agree_d = sum(int(s.cells.split("/")[1]) for s in scores if "/" in s.cells)
    print(f"  {'均值':<10}{iou:>10.3f}{blk:>12.1f}{f'{agree_n}/{agree_d}':>12}")
    print(f"\n  三联图（forge | 参照 | 差异）：{out_dir.resolve()}")
    print(
        "\n  **判据不是「块差越小越好」**：参照渲染器是 2010 年前后的固定管线，"
        "\n  相当一部分行为是当年的实现约束、不是设计。每一处差异只能落进"
        "\n  「我们错了」（修）或「我们更好」（登记进 docs/RENDERER.md，接受数字变差）两类。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
