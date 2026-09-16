"""Capture MJCF and URDF model-loading references."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

from PIL import Image

from mojive.app.composition import build
from mojive.scene.assets import resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture runtime MJCF and URDF loading")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/model-loading"))
    parser.add_argument("--offscreen", action="store_true", help="Capture one model without a UI")
    parser.add_argument("--asset", type=Path, help="Model to capture with --offscreen")
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"))
    parser.add_argument("--profile", action="store_true", help="Profile offscreen resource loading")
    args = parser.parse_args(argv)
    if (args.asset or args.profile) and not args.offscreen:
        parser.error("--asset and --profile require --offscreen")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.offscreen:
        _capture_offscreen(args.asset or resolve("test_scene.xml"), args)
        return 0

    viewer = build(
        resolve("empty"), paused=True, vsync=False, width=1400, height=900, renderer=args.renderer
    )
    try:
        _capture(viewer, args.output / "empty.png")
        viewer.window._file_drag_active = True
        _capture(viewer, args.output / "drop-hover.png")
        viewer.window._on_file_drop(None, [str(resolve("test_scene.xml"))])
        _capture(viewer, args.output / "mjcf.png")
        for name, output_name in (("test_scene.urdf", "urdf.png"),):
            result = viewer.app.load_model(resolve(name))
            if not result.ok:
                raise RuntimeError(result.message)
            _capture(viewer, args.output / output_name)
    finally:
        viewer.release()

    for name in ("empty.png", "drop-hover.png", "mjcf.png", "urdf.png"):
        path = args.output / name
        print(path.resolve())
    return 0


def _capture_offscreen(path: Path, args) -> None:
    from mojive import SceneRenderer
    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.mujoco import MuJoCoAdapter

    # An initialized empty renderer matches runtime model replacement. Device
    # startup is excluded; the image readback includes completion of GPU work.
    with SceneRenderer(width=960, height=640, renderer=args.renderer) as renderer:
        started = time.perf_counter()
        adapter = MuJoCoAdapter(path)
        try:
            source = adapter.scene_source()
            prepared = time.perf_counter()
            profile = cProfile.Profile() if args.profile else None
            if profile is not None:
                profile.enable()
            renderer.set_scene(source)
            if profile is not None:
                profile.disable()
            ready = time.perf_counter()
            renderer.update(
                adapter.frame(FrameNeeds(poses=True, deformables=True)),
                camera=adapter.camera_hint(),
            )
            pixels = renderer.render()
            readable = time.perf_counter()
            Image.fromarray(pixels).save(args.output / "model.png")
            report = {
                "asset": str(path.resolve()),
                "backend": renderer._backend.caps.name,
                "device": renderer._backend.caps.renderer,
                "profiled": profile is not None,
                "source_s": prepared - started,
                "resources_s": ready - prepared,
                "first_image_readable_s": readable - started,
                "textures": len(source.textures),
                "texture_bytes": sum(texture.pixels.nbytes for texture in source.textures.values()),
                "meshes": len(source.meshes),
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if profile is not None:
                profile.dump_stats(args.output / "resources.prof")
                with (args.output / "resources.txt").open("w") as stream:
                    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
                        "cumtime"
                    ).print_stats(35)
            print(json.dumps(report, indent=2))
        finally:
            adapter.release()


def _capture(viewer, path: Path) -> None:
    for _ in range(4):
        viewer.sync()
    pixels = viewer.window.read_frame()[::-1, :, :3]
    Image.fromarray(pixels, "RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
