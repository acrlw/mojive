"""Compare live MuJoCo models across all three renderers, including mesh updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from mojive import RenderFlag, RenderProduct, SceneRenderer
from mojive.adapters.base import FrameNeeds, NodeType
from mojive.assets import resolve
from mojive.backends import make_adapter
from mojive.tools.native_parity import compare


def run(output, humanoids=None):
    paths = [resolve(name) for name in ("joint_gizmo", "deformables", "interpolated_flex")]
    if humanoids:
        paths.append(humanoids)
    report = {}
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        captures = {}
        for backend in ("opengl", "wgpu", "bgfx"):
            adapter = make_adapter("mujoco", path)
            try:
                needs = FrameNeeds(poses=True, deformables=True, tendons=True)
                adapter.prepare_frame(needs)
                source = adapter.scene_source()
                skin_ids = [
                    node.object_id for node in adapter.nodes() if node.type is NodeType.SKIN
                ]
                with SceneRenderer(
                    source,
                    width=480,
                    height=360,
                    renderer=backend,
                    samples=0,
                    camera=adapter.camera_hint(),
                ) as renderer:
                    for pose in (0, 1, 0):
                        if path.stem == "deformables":
                            joint = mujoco.mj_name2id(
                                adapter.model, mujoco.mjtObj.mjOBJ_JOINT, "skin_tip_hinge"
                            )
                            adapter.set_qpos(
                                int(adapter.model.jnt_qposadr[joint]), np.deg2rad(40 * pose)
                            )
                        else:
                            mujoco.mj_resetData(adapter.model, adapter.data)
                            if pose:
                                mujoco.mj_step(adapter.model, adapter.data, nstep=20)
                            mujoco.mj_forward(adapter.model, adapter.data)
                        frame = adapter.frame(needs)
                        for view in ("shaded", "wireframe"):
                            renderer.set_debug_view(view)
                            renderer.set_flag(RenderFlag.SHADOW, view == "shaded")
                            renderer.set_flag(RenderFlag.REFLECTION, view == "shaded")
                            renderer.update(frame)
                            capture = {
                                product: renderer.render(product=product)
                                for product in RenderProduct
                            }
                            renderer.render()
                            with renderer._current():
                                capture["alpha"] = renderer._backend.target.read_color(flip=True)[
                                    ..., 3
                                ]
                            key = f"{path.stem}-{view}-{pose}"
                            previous = captures.setdefault(key, {}).get(backend)
                            if previous is not None:
                                Image.fromarray(capture[RenderProduct.COLOR]).save(
                                    output / f"{key}-{backend}-restored.png"
                                )
                                for product in RenderProduct:
                                    np.testing.assert_array_equal(
                                        capture[product],
                                        previous[product],
                                        err_msg=f"{key}/{backend}/{product}",
                                    )
                            captures[key][backend] = capture
                            if path.stem == "deformables":
                                assert (
                                    np.isin(capture[RenderProduct.OBJECT_ID], skin_ids).sum() > 100
                                )
                            Image.fromarray(capture[RenderProduct.COLOR]).save(
                                output / f"{key}-{backend}.png"
                            )
                    if path.stem == "deformables":
                        rest = captures[f"{path.stem}-wireframe-0"][backend][RenderProduct.COLOR]
                        moved = captures[f"{path.stem}-wireframe-1"][backend][RenderProduct.COLOR]
                        assert (
                            np.count_nonzero(
                                np.max(np.abs(rest.astype(float) - moved), axis=2) > 10
                            )
                            > 50
                        )
            finally:
                adapter.release()
        for key, rows in captures.items():
            report[key] = {}
            for backend in ("wgpu", "bgfx"):
                metrics = compare(rows["opengl"], rows[backend])
                metrics["alpha_mean"] = float(
                    np.abs(rows[backend]["alpha"].astype(float) - rows["opengl"]["alpha"]).mean()
                )
                report[key][backend] = metrics
            Image.fromarray(
                np.concatenate(
                    [rows[name][RenderProduct.COLOR] for name in ("opengl", "wgpu", "bgfx")], axis=1
                )
            ).save(output / f"{key}-comparison.png")
            print(key, json.dumps(report[key]), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = {
        key: rows["bgfx"]
        for key, rows in report.items()
        if rows["bgfx"]["color_mean"] >= 1
        or rows["bgfx"]["color_p99"] > 5
        or rows["bgfx"]["id_disagreement"] >= 0.001
        or rows["bgfx"]["segmentation_disagreement"] >= 0.001
        or rows["bgfx"]["depth_p99"] >= 1e-4
        or rows["bgfx"]["alpha_mean"] > 1
    }
    if failures:
        raise SystemExit(f"Native model parity failed: {json.dumps(failures)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanoids-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/native-model-parity"))
    args = parser.parse_args()
    run(args.output, args.humanoids_model)


if __name__ == "__main__":
    main()
