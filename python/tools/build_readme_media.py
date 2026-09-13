"""Copy unmodified production captures into the README media directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="ui_runtime capture directory")
    parser.add_argument("--showcase", type=Path, required=True, help="showcase renderer capture")
    parser.add_argument("--output", type=Path, default=Path("docs/images/readme"))
    args = parser.parse_args(argv)

    captures = {
        "hero.png": args.runtime / "d4-running-rotation-perturb.png",
        "joint-authoring.png": args.runtime / "joint-revolute-focus-oblique.png",
        "rendering.png": args.showcase,
    }
    missing = [path for path in captures.values() if not path.is_file()]
    if missing:
        parser.error("missing capture input(s): " + ", ".join(str(path) for path in missing))

    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in captures.items():
        destination = args.output / name
        shutil.copyfile(source, destination)
        print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
