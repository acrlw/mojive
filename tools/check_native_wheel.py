"""Install the local native wheel separately and render without development paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    destination = Path("output/native-wheel-install").resolve()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--reinstall",
            "--target",
            str(destination),
            str(args.wheel),
        ],
        check=True,
    )
    script = """
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
import mojive
from mojive import Scene, SceneRenderer, RenderProduct
from PIL import Image
assert Path(mojive.__file__).is_relative_to(Path(sys.argv[1]))
scene = Scene()
box = scene.box(color=(.8,.2,.1,1))
with SceneRenderer(scene.source, width=160, height=120, renderer='bgfx', samples=0) as renderer:
    renderer.update(scene.frame)
    color = renderer.render()
    ids = renderer.render(product=RenderProduct.OBJECT_ID)
    assert (ids == box.object_id).sum() > 100
    assert color.std()>10
    Image.fromarray(color).save(sys.argv[2])
    import mojive._native as native
    assert Path(native.__file__).is_relative_to(Path(sys.argv[1]))
    print(json.dumps({'package':mojive.__file__,'extension':native.__file__,'shape':color.shape,'backend':renderer._backend.describe()}))
"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("MOJIVE_NATIVE")}
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(destination), "output/native-wheel/installed.png"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (destination / "mojive/nativeLicenses/stbImageResize.txt").is_file()
    assert (destination / "mojive/nativeLicenses/meshoptimizer.txt").is_file()
    report = json.loads(result.stdout)
    Path("output/native-wheel/installed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
