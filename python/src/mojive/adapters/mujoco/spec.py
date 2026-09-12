"""Editable MJCF and URDF loading, serialization and metadata."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ... import math3d
from ...types import (
    MATERIAL_TEXTURE_ROLES,
    CameraView,
    LightSet,
)
from .constants import _MOJIVE_AMBIENT_NUMERIC, _MOJIVE_HAZE_NUMERIC, _URDF_MIN_POSITIVE_INERTIA
from .engine import mujoco
from .schema import _MODEL_COMPONENT_CATEGORIES


def _load_editable_spec(path: Path):
    is_urdf = path.suffix.lower() == ".urdf"
    source = _normalized_urdf_source(path) if is_urdf else None
    spec = (
        mujoco.MjSpec.from_string(source)
        if source is not None
        else mujoco.MjSpec.from_file(str(path))
    )
    if is_urdf:
        spec.modelfiledir = str(path.parent)
        spec = mujoco.MjSpec.from_string(spec.to_xml())
        spec.modelfiledir = str(path.parent)
    return spec


def _stabilize_editable_asset_paths(spec, root: ET.Element) -> None:
    """Keep expanded include/model assets resolvable after MjSpec serialization.

    MuJoCo keeps only one compiler mesh/texturedir on a composed MjSpec. Assets
    expanded from ``asset/model`` declarations can therefore retain a child
    filename while losing that child's directory. Resolve only missing relative
    files with one unambiguous match below the model directory, then store a
    portable model-relative path. Ordinary models whose compiler directory still
    resolves the file are left unchanged.
    """

    model_directory = Path(str(spec.modelfiledir or ".")).resolve()
    compiler = root.find("compiler")
    meshdir = "" if compiler is None else str(compiler.attrib.get("meshdir", ""))
    texturedir = "" if compiler is None else str(compiler.attrib.get("texturedir", ""))
    asset = root.find("asset")
    directories = {
        "mesh": meshdir,
        "hfield": "",
        "skin": meshdir,
        "texture": texturedir,
    }
    compiler_fields = {
        "mesh": "meshdir",
        "skin": "meshdir",
        "texture": "texturedir",
    }

    def portable_path(candidate: Path) -> str:
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(model_directory).as_posix()
        except ValueError:
            return str(resolved)

    unresolved: list[tuple[ET.Element, str]] = []
    for element in asset or ():
        filename = str(element.attrib.get("file", "")).strip()
        if not filename or Path(filename).is_absolute():
            continue
        directory = directories.get(element.tag, "")
        candidates = (
            model_directory / directory / filename,
            model_directory / filename,
        )
        if any(candidate.is_file() for candidate in candidates):
            continue
        finder = getattr(spec, element.tag, None)
        semantic = finder(element.attrib.get("name", "")) if finder is not None else None
        compiler_field = compiler_fields.get(element.tag, "")
        semantic_compiler = getattr(semantic, "compiler", None)
        semantic_directory = (
            str(getattr(semantic_compiler, compiler_field, "")) if compiler_field else ""
        )
        semantic_candidate = model_directory / semantic_directory / filename
        if semantic_directory and semantic_candidate.is_file():
            element.set("file", portable_path(semantic_candidate))
            continue
        unresolved.append((element, filename))
    if not unresolved:
        return

    names = {Path(filename).name for _element, filename in unresolved}
    matches: dict[str, list[Path]] = {name: [] for name in names}
    for candidate in model_directory.rglob("*"):
        if candidate.is_file() and candidate.name in matches:
            matches[candidate.name].append(candidate.resolve())
    for element, filename in unresolved:
        candidates = matches.get(Path(filename).name, ())
        if len(candidates) != 1:
            continue
        element.set("file", portable_path(candidates[0]))


def _normalized_urdf_source(path: Path) -> str | None:
    """Repair deterministic local URDF resource paths that MuJoCo cannot resolve.

    Some exported URDFs set ``meshdir="meshes"`` while also storing filenames as
    ``meshes/foo.stl``. MuJoCo applies both values and looks below
    ``meshes/meshes``. Preserve that explicit path when it exists; otherwise use
    the compatible shorter path only when its asset exists on disk. ROS package
    URIs and uniquely relocated local meshes are resolved only when one existing
    file is unambiguous. Positive-definite inertia tensors entirely below MuJoCo's
    numerical acceptance floor are scaled uniformly, preserving their shape while
    avoiding a compiler rejection for physically negligible decorative links.
    """

    root = ET.fromstring(path.read_bytes())
    compiler = root.find("./mujoco/compiler")
    meshdir = "" if compiler is None else str(compiler.attrib.get("meshdir", ""))
    directory = meshdir.strip().replace("\\", "/").rstrip("/")
    safe_directory = not (
        directory.startswith("/")
        or (len(directory) >= 2 and directory[1] == ":")
        or ".." in directory.split("/")
    )
    prefix = directory + "/" if directory and safe_directory else ""
    mesh_root = path.parent / directory if directory and safe_directory else path.parent

    def package_asset(filename: str) -> Path | None:
        if not filename.startswith("package://"):
            return None
        parts = tuple(part for part in filename[len("package://") :].split("/") if part)
        if not parts or ".." in parts:
            return None
        parents = (path.parent, *tuple(path.parents)[:6])
        candidates = [path.parent.joinpath(*parts)]
        for parent in parents:
            if len(parts) > 1:
                candidates.append(parent.joinpath(*parts[1:]))
            candidates.append(parent.joinpath(*parts))
            if parent.name == parts[0] and len(parts) > 1:
                candidates.append(parent.joinpath(*parts[1:]))
        existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
        return next(iter(existing)) if len(existing) == 1 else None

    def unique_local_asset(filename: str) -> Path | None:
        name = Path(filename).name
        if not name:
            return None
        exact = {
            parent.joinpath(filename).resolve()
            for parent in (path.parent, *tuple(path.parents)[:4])
            if parent.joinpath(filename).is_file()
        }
        if len(exact) == 1:
            return next(iter(exact))
        if exact:
            return None
        matches = {
            candidate.resolve() for candidate in path.parent.rglob(name) if candidate.is_file()
        }
        return next(iter(matches)) if len(matches) == 1 else None

    changed = False
    for mesh in root.iter("mesh"):
        filename = str(mesh.attrib.get("filename", "")).replace("\\", "/")
        while filename.startswith("./"):
            filename = filename[2:]
        package_path = package_asset(filename)
        if package_path is not None:
            mesh.attrib["filename"] = str(package_path)
            changed = True
            continue
        if filename.startswith("package://") or Path(filename).is_absolute():
            continue
        explicit_path = mesh_root / filename
        if explicit_path.is_file():
            continue
        if prefix and filename.startswith(prefix):
            shorter = filename[len(prefix) :]
            compatible_path = mesh_root / shorter
            if shorter and ".." not in shorter.split("/") and compatible_path.is_file():
                mesh.attrib["filename"] = shorter
                changed = True
                continue
        local_path = unique_local_asset(filename)
        if local_path is not None:
            mesh.attrib["filename"] = str(local_path)
            changed = True

    inertia_fields = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    for inertia in root.iter("inertia"):
        try:
            ixx, ixy, ixz, iyy, iyz, izz = (float(inertia.attrib[name]) for name in inertia_fields)
        except (KeyError, ValueError):
            continue
        matrix = np.array(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)), np.float64)
        eigenvalues = np.linalg.eigvalsh(matrix)
        largest = float(eigenvalues[-1])
        if float(eigenvalues[0]) <= 0.0 or not 0.0 < largest < _URDF_MIN_POSITIVE_INERTIA:
            continue
        scale = _URDF_MIN_POSITIVE_INERTIA / largest
        for name in inertia_fields:
            inertia.attrib[name] = f"{float(inertia.attrib[name]) * scale:.17g}"
        changed = True
    return ET.tostring(root, encoding="unicode") if changed else None


def _component_xml(spec) -> tuple[ET.Element, str]:
    xml = spec.to_xml()
    root = ET.fromstring(xml)
    # MuJoCo 3.11 serializes the input-only e_potential/e_kinetic sensor tags
    # without their e_ prefix, then rejects that spelling on the next parse.
    # Normalize the generated XML before the structured editor reparses it.
    sensor = root.find("sensor")
    for element in sensor or ():
        if element.tag in {"potential", "kinetic"}:
            element.tag = f"e_{element.tag}"
    _restore_material_texture_layers(spec, root)
    _stabilize_editable_asset_paths(spec, root)
    return root, xml


def _restore_material_texture_layers(spec, root: ET.Element) -> None:
    """Restore texture-role declarations omitted by MuJoCo 3.11 XML output.

    MjSpec keeps the complete inherited texture-role vector for every default
    class, but ``to_xml()`` only emits the legacy RGB texture attribute. Compare
    each class with its parent so inherited roles remain inherited rather than
    being materialized as child overrides.
    """

    root_default = root.find("default")
    if root_default is None:
        return
    empty = ("",) * len(MATERIAL_TEXTURE_ROLES)

    def visit(element: ET.Element, inherited: tuple[str, ...], *, main: bool = False) -> None:
        class_name = str(element.attrib.get("class", "")).strip()
        spec_default = spec.default if main else spec.find_default(class_name)
        if spec_default is None:
            current = inherited
        else:
            textures = tuple(str(value) for value in spec_default.material.textures)
            current = (*textures, *empty[len(textures) :])[: len(empty)]

        explicit = tuple(
            (role, texture)
            for role, texture, parent_texture in zip(
                MATERIAL_TEXTURE_ROLES,
                current,
                inherited,
                strict=True,
            )
            if texture and texture != parent_texture
        )
        material = element.find("material")
        if material is not None:
            material.attrib.pop("texture", None)
            for child in tuple(material):
                if child.tag == "layer":
                    material.remove(child)
        if explicit:
            material = material if material is not None else ET.SubElement(element, "material")
            for role, texture in explicit:
                ET.SubElement(material, "layer", {"role": role, "texture": texture})
        elif material is not None and not material.attrib and not len(material):
            element.remove(material)

        for child in element.findall("default"):
            visit(child, current)

    visit(root_default, empty, main=True)

    asset = root.find("asset")
    for element in asset.findall("material") if asset is not None else ():
        name = str(element.attrib.get("name", "")).strip()
        material_spec = spec.material(name) if name else None
        if material_spec is None:
            continue
        class_name = str(element.attrib.get("class", "")).strip()
        default_spec = spec.find_default(class_name) if class_name else spec.default
        inherited = (
            tuple(str(value) for value in default_spec.material.textures)
            if default_spec is not None
            else empty
        )
        inherited = (*inherited, *empty[len(inherited) :])[: len(empty)]
        textures = tuple(str(value) for value in material_spec.textures)
        current = (*textures, *empty[len(textures) :])[: len(empty)]
        explicit = tuple(
            (role, texture)
            for role, texture, default_texture in zip(
                MATERIAL_TEXTURE_ROLES,
                current,
                inherited,
                strict=True,
            )
            if texture and texture != default_texture
        )
        element.attrib.pop("texture", None)
        for child in tuple(element):
            if child.tag == "layer":
                element.remove(child)
        for role, texture in explicit:
            ET.SubElement(element, "layer", {"role": role, "texture": texture})


def _editable_spec_xml(spec) -> str:
    root, _xml = _component_xml(spec)
    return _serialize_component_xml(root)


def _component_section(root: ET.Element, category: str, *, create: bool = False):
    if category not in _MODEL_COMPONENT_CATEGORIES:
        raise ValueError(f"Unsupported model component category: {category}")
    section = root.find(category)
    if section is None and create:
        section = ET.SubElement(root, category)
    return section


def _serialize_component_xml(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _format_mjcf_values(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


_RAY_NORMAL = 1 << 4


def _mjcf_name(prefix: str, value: str, index: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return f"{prefix}_{index}_{token or 'entity'}"


def _camera_quaternion(camera: CameraView) -> np.ndarray:
    forward = math3d.normalize(np.asarray(camera.target, np.float64) - camera.eye)
    up = math3d.normalize(np.asarray(camera.up, np.float64))
    right = math3d.normalize(np.cross(forward, up))
    if not np.any(right):
        right = math3d.normalize(np.cross(forward, np.array((0.0, 0.0, 1.0))))
    up = math3d.normalize(np.cross(right, forward))
    return math3d.mat3_to_quat(np.column_stack((right, up, -forward)))


def _decode_text_names(data: str) -> tuple[str, ...]:
    try:
        values = json.loads(data)
    except (TypeError, ValueError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(value for value in values if isinstance(value, str) and value)


def _spec_text_names(spec, name: str) -> tuple[str, ...]:
    if spec is None:
        return ()
    item = spec.text(name)
    return _decode_text_names(item.data) if item is not None else ()


def _set_text_names(spec, name: str, values) -> None:
    if spec is None:
        return
    previous = spec.text(name)
    names = sorted(set(values))
    if not names:
        if previous is not None:
            spec.delete(previous)
        return
    data = json.dumps(names, ensure_ascii=True, separators=(",", ":"))
    if previous is None:
        spec.add_text(name=name, data=data)
    else:
        previous.data = data


def _compiled_text_names(model, name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    out = []
    for index in range(model.ntext):
        compiled_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, index) or ""
        if compiled_name == name:
            prefix = ""
        elif compiled_name.endswith(name):
            prefix = compiled_name[: -len(name)]
        else:
            continue
        start = int(model.text_adr[index])
        stop = start + int(model.text_size[index])
        raw = bytes(model.text_data[start:stop]).split(b"\0", 1)[0]
        out.append((prefix, _decode_text_names(raw.decode("utf-8", errors="replace"))))
    return tuple(out)


def _numeric_values(model, name: str) -> np.ndarray | None:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if index < 0:
        return None
    start = int(model.numeric_adr[index])
    stop = start + int(model.numeric_size[index])
    return np.asarray(model.numeric_data[start:stop], np.float64)


def _set_numeric(spec, name: str, values) -> None:
    previous = spec.numeric(name)
    if previous is not None:
        spec.delete(previous)
    spec.add_numeric(name=name, data=np.asarray(values, np.float64).reshape(-1))


def _apply_environment(spec, lights: LightSet, extent: float) -> None:
    headlight = lights.headlight
    spec.visual.headlight.active = headlight is not None and headlight.active
    if headlight is not None:
        spec.visual.headlight.ambient = headlight.ambient
        spec.visual.headlight.diffuse = headlight.diffuse
        spec.visual.headlight.specular = headlight.specular
    spec.visual.rgba.fog = (*np.asarray(lights.fog_color, np.float64), 1.0)
    scale = max(float(extent), 1e-9)
    spec.visual.map.fogstart = lights.fog_start / scale
    spec.visual.map.fogend = lights.fog_end / scale
    spec.visual.rgba.haze = (*np.asarray(lights.haze_color, np.float64), 1.0)
    spec.visual.map.haze = lights.haze_density
    spec.visual.quality.numslices = max(3, int(lights.horizon_haze_slices))
    _set_numeric(spec, _MOJIVE_AMBIENT_NUMERIC, lights.ambient)
    _set_numeric(
        spec,
        _MOJIVE_HAZE_NUMERIC,
        (float(lights.horizon_haze), float(lights.horizon_haze_slices)),
    )


def _relative_pose(position, rotation, parent_position, parent_rotation):
    parent_rotation = np.asarray(parent_rotation, np.float64).reshape(3, 3)
    local_rotation = parent_rotation.T @ np.asarray(rotation, np.float64).reshape(3, 3)
    local_position = parent_rotation.T @ (
        np.asarray(position, np.float64).reshape(3)
        - np.asarray(parent_position, np.float64).reshape(3)
    )
    return local_position, local_rotation
