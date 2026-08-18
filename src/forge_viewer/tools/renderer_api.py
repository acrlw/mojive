"""Generate MuJoCo Renderer compatibility reference images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("output/renderer-api")
MAX_RGB_MAE = 3.0
MAX_DEPTH_P95_M = 0.025
MIN_SEGMENTATION_AGREEMENT = 0.999

_MODEL = """
<mujoco>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality offsamples="4"/>
  </visual>
  <worldbody>
    <light pos="-2 -3 4"/>
    <camera name="acceptance" pos="0 -4 2"
            xyaxes="1 0 0 0 .4472136 .8944272"/>
    <geom type="plane" size="4 4 .1" rgba=".12 .18 .24 1"/>
    <geom pos="-.8 0 .4" type="sphere" size=".4" rgba=".1 .55 .9 1"/>
    <geom pos=".8 0 .45" type="box" size=".4 .3 .45" rgba=".9 .3 .12 1"/>
    <site name="marker" pos="0 .15 .7" type="sphere" size=".14" rgba=".2 .9 .4 1"/>
    <geom pos="0 .55 .55" type="capsule" size=".18 .5" euler="0 90 0"
          rgba=".85 .75 .2 .4"/>
  </worldbody>
</mujoco>
"""


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, np.uint8), "RGB").save(path)


def _save_depth(path: Path, depth: np.ndarray) -> None:
    values = np.asarray(depth, np.float32)
    visible = values[np.isfinite(values)]
    ceiling = float(np.percentile(visible, 95)) if visible.size else 1.0
    normalized = 1.0 - np.clip(values / max(ceiling, 1e-6), 0.0, 1.0)
    Image.fromarray(np.asarray(normalized * 255.0, np.uint8), "L").save(path)


def _save_segmentation(path: Path, segmentation: np.ndarray) -> None:
    pairs = np.asarray(segmentation, np.int32)
    image = np.zeros((*pairs.shape[:2], 3), np.uint8)
    valid = pairs[..., 0] >= 0
    object_id = pairs[..., 0].astype(np.uint32)
    object_type = pairs[..., 1].astype(np.uint32)
    image[..., 0] = ((object_id * 73 + object_type * 17) & 255).astype(np.uint8)
    image[..., 1] = ((object_id * 151 + object_type * 29) & 255).astype(np.uint8)
    image[..., 2] = ((object_id * 199 + object_type * 47) & 255).astype(np.uint8)
    image[~valid] = 0
    Image.fromarray(image, "RGB").save(path)


def _render_forge(model, data):
    from ..renderer import Renderer

    with Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="acceptance")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    return rgb, depth, segmentation


def _render_mujoco(mujoco, model, data):
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="acceptance")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
    return rgb, depth, segmentation


def main() -> int:
    import mujoco

    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(_MODEL)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    forge = _render_forge(model, data)
    reference = _render_mujoco(mujoco, model, data)

    _save_rgb(OUT / "rgb-forge.png", forge[0])
    _save_rgb(OUT / "rgb-mujoco.png", reference[0])
    _save_rgb(
        OUT / "rgb-absolute-difference.png",
        np.abs(forge[0].astype(np.int16) - reference[0].astype(np.int16)).astype(np.uint8),
    )
    _save_depth(OUT / "depth-forge.png", forge[1])
    _save_depth(OUT / "depth-mujoco.png", reference[1])
    _save_segmentation(OUT / "segmentation-forge.png", forge[2])
    _save_segmentation(OUT / "segmentation-mujoco.png", reference[2])

    common = (forge[1] < np.max(forge[1])) & (reference[1] < np.max(reference[1]))
    depth_error = np.abs(forge[1][common] - reference[1][common])
    segmentation_agreement = float(np.mean(np.all(forge[2] == reference[2], axis=2)))
    report = {
        "rgb_mean_absolute_error": float(
            np.mean(np.abs(forge[0].astype(np.float32) - reference[0].astype(np.float32)))
        ),
        "depth_mean_absolute_error_m": float(np.mean(depth_error)) if depth_error.size else 0.0,
        "depth_p95_absolute_error_m": (
            float(np.percentile(depth_error, 95)) if depth_error.size else 0.0
        ),
        "forge_segmentation_pairs": len(np.unique(forge[2].reshape(-1, 2), axis=0)),
        "mujoco_segmentation_pairs": len(np.unique(reference[2].reshape(-1, 2), axis=0)),
        "segmentation_pixel_agreement": segmentation_agreement,
    }
    passed = (
        report["rgb_mean_absolute_error"] <= MAX_RGB_MAE
        and report["depth_p95_absolute_error_m"] <= MAX_DEPTH_P95_M
        and report["forge_segmentation_pairs"] == report["mujoco_segmentation_pairs"]
        and segmentation_agreement >= MIN_SEGMENTATION_AGREEMENT
    )
    report["passed"] = passed
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUT.resolve())
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
