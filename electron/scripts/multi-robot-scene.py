"""Compose an MJCF stress fixture without changing the source model or its assets."""

import argparse
import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco


def resolve_assets(spec, directory):
    for assets, field in ((spec.meshes, "meshdir"), (spec.textures, "texturedir"),
                          (spec.hfields, ""), (spec.skins, "meshdir")):
        for asset in assets:
            if asset.file and not Path(asset.file).is_absolute():
                compiler = getattr(asset, "compiler", spec.compiler)
                base = directory / str(getattr(compiler, field, "")) if field else directory
                asset.file = str((base / asset.file).resolve())
    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.spacing <= 0:
        parser.error("count and spacing must be positive")
    source = args.model.resolve()
    scene = mujoco.MjSpec.from_file(str(source))
    resolve_assets(scene, source.parent)
    # Keep one common environment. Each attached robot retains its complete dynamics,
    # actuators and full-resolution visual/collision assets, with unique object names.
    robot = scene.copy()
    for element in [*robot.worldbody.geoms, *robot.worldbody.lights, *robot.worldbody.cameras]:
        robot.delete(element)
    for key in list(robot.keys):
        robot.delete(key)
    for key in list(scene.keys):
        scene.delete(key)
    columns = int(args.count ** 0.5 + 0.999)
    for index in range(1, args.count):
        frame = scene.worldbody.add_frame(pos=[index % columns * args.spacing,
                                               index // columns * args.spacing, 0])
        scene.attach(robot.copy(), prefix=f"robot_{index}_", frame=frame)
    scene.modelname = f"{source.stem}-{args.count}-robots"
    model = scene.compile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the fixture portable through the same relative-resource sandbox as an
    # ordinary imported MJCF. Shared files are copied once, without modifying them.
    xml = ET.fromstring(scene.to_xml())
    assets = args.output.parent / "assets"
    assets.mkdir(exist_ok=True)
    for element in xml.findall("asset/*[@file]"):
        source_file = Path(element.attrib["file"])
        name = hashlib.sha256(str(source_file).encode()).hexdigest()[:12] + "_" + source_file.name
        destination = assets / name
        if not destination.exists():
            shutil.copy2(source_file, destination)
        element.set("file", f"assets/{name}")
    # Attachment namespaces assets as well as bodies. Identical immutable assets
    # should be shared across instances, not compiled into N copies of each mesh.
    asset_root = xml.find("asset")
    for kind in ("texture", "material", "mesh", "hfield"):
        canonical, aliases = {}, {}
        for element in list(asset_root.findall(kind)) if asset_root is not None else []:
            if "name" not in element.attrib:
                continue
            name = element.attrib["name"]
            signature = (tuple(sorted((k, v) for k, v in element.attrib.items() if k != "name")),
                         tuple(ET.tostring(child) for child in element))
            if signature in canonical:
                aliases[name] = canonical[signature]
                asset_root.remove(element)
            else:
                canonical[signature] = name
        for element in xml.iter():
            if element.get(kind) in aliases:
                element.set(kind, aliases[element.get(kind)])
    ET.indent(xml)
    args.output.write_text(ET.tostring(xml, encoding="unicode"))
    print(f"{args.output}: {model.nbody} bodies, {model.ngeom} geoms, {model.nv} velocities")


if __name__ == "__main__":
    main()
