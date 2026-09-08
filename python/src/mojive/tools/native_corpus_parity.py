"""Load and visually compare local MuJoCo model collections against OpenGL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .mujoco_model_suite import (
    ModelAuditResult,
    ModelAuditStatus,
    build_report,
    capture_name,
    discover_models,
    run_suite,
)


def run(roots, output, camera_count=8, width=480, height=360, compare_only=False, samples=0):
    output.mkdir(parents=True, exist_ok=True)
    models = discover_models(roots)
    if not models:
        raise SystemExit("No model documents found")
    results = {}
    for backend in ("opengl", "bgfx"):
        if compare_only:
            saved = json.loads((output / f"{backend}.json").read_text())
            results[backend] = {row["path"]: ModelAuditResult(**row) for row in saved["results"]}
            continue
        rows = run_suite(
            models,
            backend=backend,
            jobs=1,
            camera_count=camera_count,
            width=width,
            height=height,
            dynamic_frames=1,
            load_only=False,
            capture_dir=output / backend,
            capture_samples=samples,
        )
        results[backend] = {row.path: row for row in rows}
        (output / f"{backend}.json").write_text(
            json.dumps(build_report(roots, rows), indent=2) + "\n"
        )
    report, failures, thumbnails, ambiguities, empty_documents = {}, [], [], [], []
    skipped = {
        ModelAuditStatus.SKIPPED_DEPENDENCY,
        ModelAuditStatus.SKIPPED_FRAGMENT,
        ModelAuditStatus.SKIPPED_UNSUPPORTED_ASSET,
    }
    for path in models:
        name = capture_name(path)
        a, b = results["opengl"][str(path)], results["bgfx"][str(path)]
        if a.status != ModelAuditStatus.PASSED or b.status != ModelAuditStatus.PASSED:
            report[str(path)] = {
                "opengl_status": a.status,
                "bgfx_status": b.status,
                "opengl_message": a.message,
                "bgfx_message": b.message,
            }
            if a.status not in skipped or b.status not in skipped or a.status != b.status:
                failures.append(str(path))
            continue
        views = []
        worst, worst_image = -1, None
        for index in range(a.rendered_views):
            file = f"{index:02}"
            reference = np.array(Image.open(output / "opengl" / name / f"{file}.png"))
            native = np.array(Image.open(output / "bgfx" / name / f"{file}.png"))
            with np.load(output / "opengl" / name / f"{file}.npz") as data:
                reference_ids = data["segmentation"]
                reference_depth = data.get("depth")
                near, far = (float(data["near"]), float(data["far"])) if "near" in data else (0, 1)
            with np.load(output / "bgfx" / name / f"{file}.npz") as data:
                native_ids = data["segmentation"]
                native_depth = data.get("depth")
            error = np.abs(native.astype(float) - reference)
            # Compare material colors away from rasterization boundaries. MSAA
            # coverage differs between APIs; keep the full-image error separately.
            interior = np.all(reference_ids == native_ids, axis=2)
            interior[[0, -1], :] = False
            interior[:, [0, -1]] = False
            reference16 = reference.astype(np.int16)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    interior &= np.all(
                        reference_ids == np.roll(reference_ids, (dy, dx), (0, 1)), axis=2
                    )
                    interior &= np.all(native_ids == np.roll(native_ids, (dy, dx), (0, 1)), axis=2)
                    gradient = np.abs(reference16 - np.roll(reference16, (dy, dx), (0, 1)))
                    interior &= gradient.max(axis=2) <= 32
            objects = {}
            words = reference_ids[interior].astype(np.uint32).astype(np.uint64)
            packed, inverse = np.unique(words[:, 0] | (words[:, 1] << 32), return_inverse=True)
            identity = np.column_stack((packed & 0xFFFFFFFF, packed >> 32)).astype(np.int32)
            counts = np.bincount(inverse)
            totals = np.bincount(inverse, weights=error[interior].mean(axis=1))
            for pair, count, total in zip(identity, counts, totals, strict=True):
                if pair[0] >= 0 and count >= 32:
                    objects[str(tuple(map(int, pair)))] = float(total / count)
            disagreement = np.any(reference_ids != native_ids, axis=2)
            tied = np.zeros_like(disagreement)
            clipped = np.zeros_like(disagreement)
            if reference_depth is not None and native_depth is not None:
                # Intersecting/coplanar faces can resolve to different IDs after
                # 32-bit projection and depth quantization. Preserve the raw ID
                # error and classify only pairs within four depth-buffer ULPs.
                da, db = [
                    far / (far - near) * (1 - near / np.maximum(value.astype(float), near))
                    for value in (reference_depth, native_depth)
                ]
                tolerance = 4 * np.spacing(np.maximum(da, db).astype(np.float32))
                equal_depth = np.abs(da - db) <= tolerance
                foreground = (reference_ids[..., 0] >= 0) & (native_ids[..., 0] >= 0)
                tied = disagreement & foreground & equal_depth
                clipped = (
                    disagreement & ~foreground & equal_depth & (np.minimum(da, db) >= 1 - tolerance)
                )
            unresolved = disagreement & ~tied & ~clipped
            row = {
                "color_mean": float(error.mean()),
                "color_p99": float(np.percentile(error, 99)),
                "color_interior_p99": float(np.percentile(error[interior], 99))
                if interior.any()
                else 0.0,
                "interior_pixels": int(interior.sum()),
                "segmentation_disagreement": float(disagreement.mean()),
                "depth_tied_identity": float(tied.mean()),
                "far_clip_identity": float(clipped.mean()),
                "segmentation_unresolved": float(unresolved.mean()),
                "object_color_mean": objects,
            }
            views.append(row)
            if disagreement.mean() >= 0.001 and unresolved.mean() < 0.001:
                ambiguities.append(f"{path}/view-{file}")
            score = max(objects.values(), default=row["color_mean"])
            if score > worst:
                worst, worst_image = (
                    score,
                    Image.fromarray(np.concatenate([reference, native], axis=1)),
                )
            if (
                row["color_mean"] >= 1
                or row["color_interior_p99"] > 5
                or score > 1.5
                or row["segmentation_unresolved"] >= 0.001
            ):
                failures.append(f"{path}/view-{file}")
        empty = not (b.geom_count or b.flex_count or b.skin_count)
        report[str(path)] = {
            "views": views,
            "visible_objects": b.visible_objects,
            "empty_document": empty,
        }
        if empty:
            # Empty-scene and include-only documents still undergo loading and
            # background checks, but are not counted as rendered robot models.
            empty_documents.append(str(path))
            continue
        worst_image.save(output / f"{name}-comparison.png")
        thumb = worst_image.copy()
        thumb.thumbnail((480, 180))
        tile = Image.new("RGB", (480, 210), (24, 28, 32))
        tile.paste(thumb, (0, 30))
        ImageDraw.Draw(tile).text((5, 5), f"{path.parent.name}/{path.stem}"[:68], fill="white")
        thumbnails.append(tile)
    for previous in output.glob("contact-*.png"):
        previous.unlink()
    for start in range(0, len(thumbnails), 18):
        tiles = thumbnails[start : start + 18]
        sheet = Image.new("RGB", (1440, ((len(tiles) + 2) // 3) * 210), (24, 28, 32))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * 480, (i // 3) * 210))
        sheet.save(output / f"contact-{start // 18:02}.png")
    result = {
        "samples": samples,
        "rendered_models": len(thumbnails),
        "empty_documents": empty_documents,
        "models": report,
        "failures": failures,
        "depth_ambiguous_views": ambiguities,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Compared {len(thumbnails)} renderable models and {len(empty_documents)} empty documents; "
        f"{len(failures)} failing model/views",
        flush=True,
    )
    if failures:
        raise SystemExit("Corpus parity failed; inspect report.json and comparison images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/native-corpus-parity"))
    parser.add_argument("--camera-count", type=int, default=8)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--compare-only", action="store_true", help="compare existing captures without rendering"
    )
    parser.add_argument("--samples", type=int, default=0, choices=(0, 2, 4))
    args = parser.parse_args()
    run(
        args.roots,
        args.output,
        args.camera_count,
        args.width,
        args.height,
        args.compare_only,
        args.samples,
    )


if __name__ == "__main__":
    main()
