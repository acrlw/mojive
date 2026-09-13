"""Generate reviewed production icon placement without runtime fitting."""

import argparse
import json
from dataclasses import asdict, astuple
from pathlib import Path

from mojive.ui import icons


def generate_presets() -> dict:
    """Measure every production style through the independent dynamic painter."""
    presets = {}
    for _family, entries in icons.ICON_FAMILIES:
        for _label, name in entries:
            style = icons.production_icon_style(name)
            options = asdict(style)
            options["tuning"] = style.tuning
            presets[name] = {
                "style": asdict(style),
                "layout": icons._icon_layout(name, **options),
                "metrics": astuple(icons.icon_metrics(name, **options)),
            }
    return presets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed presets")
    args = parser.parse_args()
    target = Path(icons.__file__).with_name("icon_presets.json")
    data = generate_presets()
    # One entry per line keeps this generated numeric table compact and reviewable.
    content = (
        "{\n"
        + ",\n".join(
            f"  {json.dumps(name)}: {json.dumps(value, separators=(',', ':'))}"
            for name, value in sorted(data.items())
        )
        + "\n}\n"
    )
    if args.check:
        if target.read_text() != content:
            raise SystemExit("Icon presets are stale; run make icon-presets")
    else:
        target.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
