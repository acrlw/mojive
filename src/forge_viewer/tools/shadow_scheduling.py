"""Capture and report deterministic scene-light and shadow-slot scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..assets import resolve
from ..render.backend import RenderFlag
from ..render.forge.passes.base import schedule_lights
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/shadow-scheduling"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    with OffscreenHarness(resolve("many_lights"), args.width, args.height) as harness:
        schedule = schedule_lights(harness.source.lights)
        harness.backend.set_flag(RenderFlag.SHADOW, False)
        harness.warmup(3)
        harness.save_png(args.output / "shadows-off.png")
        harness.backend.set_flag(RenderFlag.SHADOW, True)
        harness.step_and_render(0)
        harness.save_png(args.output / "scheduled-shadows.png")
        report = {
            "active_lights": schedule.active_count,
            "selected_lights": len(schedule.lights),
            "deferred_lights": schedule.deferred_lights,
            "selected_shadow_casters": schedule.selected_shadow_count,
            "deferred_shadow_casters": schedule.deferred_shadows,
            "directional_shadow_index": schedule.directional_shadow,
            "local_shadow_indices": list(schedule.local_shadows),
            "stats": harness.backend.stats.notes,
        }

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for path in sorted(args.output.iterdir()):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
