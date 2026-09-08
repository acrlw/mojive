"""Compare continuous camera motion on the reported native rendering regression scene."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mojive import CameraView, RenderProduct, SceneRenderer, ShadingModel
from mojive.adapters.base import FrameNeeds
from mojive.assets import resolve
from mojive.backends import make_adapter


def camera_views(near):
    """Exercise close panning, oblique orbiting, clipping, and camera restoration."""

    def orbit(target, distance, yaw, pitch):
        yaw, pitch = np.deg2rad([yaw, pitch])
        direction = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
            np.float32,
        )
        target = np.array(target, np.float32)
        return CameraView(eye=target + direction * distance, target=target, near=near, far=509)

    for i, offset in enumerate(np.linspace(-0.08, 0.08, 17)):
        yield f"pan-{i:02}", orbit((offset, 1 + offset / 3, 0.15), 1, -55, 26.1)
    for i, yaw in enumerate(np.linspace(28.6, 388.6, 25)):
        yield f"orbit-{i:02}", orbit((0, 0, 0.2), 3.029, yaw, 8.9)
    for i, distance in enumerate(np.geomspace(0.318, 3, 9)):
        yield f"zoom-{i:02}", orbit((0, 1, 0.15), distance, -55, 26.1)


def run(output, width=1200, height=940):
    output.mkdir(parents=True, exist_ok=True)
    adapter = make_adapter("mujoco", resolve("test_scene"))
    needs = FrameNeeds(poses=True)
    adapter.prepare_frame(needs)
    source, frame = adapter.scene_source(), adapter.frame(needs)
    views = list(camera_views(adapter.camera_hint().near))
    report = {}
    failures = []
    try:
        for shading in ShadingModel:
            for samples in (0, 4):
                group = output / f"{shading.value}-msaa{samples}"
                group.mkdir(exist_ok=True)
                source = replace(source, shading_model=shading)
                rows = {}
                for backend in ("opengl", "bgfx"):
                    with SceneRenderer(
                        source, renderer=backend, width=width, height=height, samples=samples
                    ) as renderer:
                        first = None
                        for name, camera in [*views, views[0]]:
                            renderer.update(frame, camera=camera)
                            rgb = renderer.render()
                            ids = renderer.render(product=RenderProduct.OBJECT_ID)
                            if first is None:
                                first = rgb.copy()
                            elif name == views[0][0]:
                                np.testing.assert_array_equal(
                                    first, rgb, err_msg=f"{group.name}/{backend}/restore"
                                )
                            rows.setdefault(name, {})[backend] = (rgb, ids)
                            Image.fromarray(rgb).save(group / f"{name}-{backend}.png")
                previous = None
                previews = []
                for name, _camera in views:
                    (a, ids), (b, native_ids) = rows[name]["opengl"], rows[name]["bgfx"]
                    error = np.abs(a.astype(float) - b)
                    # Measure each object independently so a large floor cannot hide a bad face.
                    objects = {}
                    for object_id in np.unique(ids):
                        if not object_id:
                            continue
                        mask = (ids == object_id) & (native_ids == object_id)
                        if mask.sum() >= 32:
                            objects[int(object_id)] = float(error[mask].mean())
                    delta = b.astype(float) - a
                    temporal = 0 if previous is None else float(np.abs(delta - previous).mean())
                    previous = delta
                    metrics = {
                        "color_mean": float(error.mean()),
                        "color_p99": float(np.percentile(error, 99)),
                        "object_color_mean": objects,
                        "temporal_error": temporal,
                        "id_disagreement": float(np.mean(ids != native_ids)),
                    }
                    key = f"{group.name}/{name}"
                    report[key] = metrics
                    if (
                        metrics["color_mean"] >= 1
                        or metrics["color_p99"] > 5
                        or max(objects.values(), default=0) > 1.5
                        or temporal > 1
                        or metrics["id_disagreement"] >= 0.001
                    ):
                        failures.append(key)
                    if name in ("pan-00", "pan-08", "orbit-00", "orbit-12", "zoom-00"):
                        image = Image.fromarray(np.concatenate([a, b], axis=1))
                        image.save(group / f"{name}-comparison.png")
                    thumb = Image.fromarray(np.concatenate([a, b], axis=1))
                    thumb.thumbnail((960, 376))
                    ImageDraw.Draw(thumb).text(
                        (8, 8), f"{name} | OpenGL (left) / bgfx (right)", fill="white"
                    )
                    previews.append(thumb)
                previews[0].save(
                    group / "motion.webp",
                    save_all=True,
                    append_images=previews[1:],
                    duration=80,
                    loop=0,
                    lossless=True,
                )
                print(group.name, "checked", len(views), "frames", flush=True)
    finally:
        adapter.release()
    (output / "report.json").write_text(
        json.dumps({"frames": report, "failures": failures}, indent=2) + "\n"
    )
    if failures:
        raise SystemExit(f"Motion parity failed: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/native-motion-parity"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=940)
    args = parser.parse_args()
    run(args.output, args.width, args.height)


if __name__ == "__main__":
    main()
