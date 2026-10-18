"""Install the local native wheel separately and render without development paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def verify_package_resources(package):
    """Check installed UI resources using only their packaged masters and manifests."""
    workspace = package / "ui/workspace"
    source = (workspace / "icons.json").read_bytes()
    icons = json.loads(source)
    meshes = json.loads((workspace / "icon_meshes.json").read_text())
    if (
        meshes["version"] != 1
        or meshes["source_sha256"] != hashlib.sha256(source).hexdigest()
        or set(meshes["icons"]) != set(icons)
    ):
        raise ValueError("Installed icon meshes do not match their packaged masters")
    scene = json.loads((package / "interaction/resources/scene_icons.json").read_text())
    if scene != {name: icons[name] for name in ("camera", "light")}:
        raise ValueError("Installed scene icons do not match their packaged masters")
    fonts = package / "text/resources/fonts"
    manifest = json.loads((fonts / "manifest.json").read_text())
    for name in ("MiSans-Regular.ttf", "MiSans-Semibold.ttf"):
        if hashlib.sha256((fonts / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f"Installed font checksum does not match: {name}")
    for path in (
        fonts / "MiSans-License.pdf",
        fonts / "README.md",
        workspace / "licenses/lucide.txt",
        workspace / "licenses/tabler.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"icons": len(icons), "scene_icons": len(scene), "font_faces": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    destination = Path("build/wheel_install").resolve()
    Path("output/native-wheel").mkdir(parents=True, exist_ok=True)
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
    resources = verify_package_resources(destination / "mojive")
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
    assert (destination / "mojive/native_licenses/stbImageResize.txt").is_file()
    assert (destination / "mojive/native_licenses/meshoptimizer.txt").is_file()
    assert (destination / "mojive/native_licenses/libtess2.txt").is_file()
    report = json.loads(result.stdout)
    report["ui_resources"] = resources
    Path("output/native-wheel/installed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
