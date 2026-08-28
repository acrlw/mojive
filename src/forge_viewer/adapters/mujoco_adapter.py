"""MuJoCo scene adapter and simulation state integration."""

from __future__ import annotations

import json
import re
import shutil
import warnings
import xml.etree.ElementTree as ET
from colorsys import hsv_to_rgb
from copy import deepcopy
from dataclasses import dataclass, replace
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .. import math3d
from ..commands import (
    AddModelElementEdit,
    ModelEdit,
    ModelElementRef,
    RemoveModelElementEdit,
    RenameModelElementEdit,
)
from ..types import (
    DEFAULT_HEADLIGHT,
    MATERIAL_TEXTURE_ROLES,
    CameraView,
    InstancePoseSource,
    InstanceVisual,
    Light,
    LightSet,
    LightType,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    ShadingModel,
    TextureData,
    TextureType,
)
from .base import (
    CAMERA_OBJECT_BASE,
    GEOMETRY_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    MODEL_OBJECT_BASE,
    ActuatorInfo,
    ActuatorVisualType,
    AdapterCaps,
    BodyProperties,
    BvhType,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    EqualityConstraintInfo,
    FrameNeeds,
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    JointVisualType,
    KeyframeInfo,
    KeyframeProperties,
    ModelAssetInfo,
    ModelComponentField,
    ModelComponentInfo,
    ModelComponentPathItem,
    ModelPropertyGroup,
    NodeType,
    PhysicsState,
    SceneAdapterBase,
    SceneFrame,
    SceneModelInfo,
    SceneNode,
    SceneSaveOptions,
    SceneSource,
    SensorInfo,
    SiteProperties,
    VisualGroupInfo,
)
from .mujoco_deformables import build_deformables, update_deformables

try:
    import mujoco
except ImportError as exc:
    mujoco = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


DEFAULT_GEOM_GROUPS: tuple[int, ...] = (0, 1, 2)
VISUAL_GROUP_CATEGORIES = ("geom", "site", "joint", "tendon", "actuator", "flex", "skin")


_GEOM_RGBA_DEFAULT = np.array([0.5, 0.5, 0.5, 1.0], np.float32)


_TEXROLE_RGB = 1
_TEXROLE_RGBA = 8
_FORGE_AMBIENT_NUMERIC = "forge_viewer.environment.ambient"
_FORGE_HAZE_NUMERIC = "forge_viewer.environment.horizon_haze"
_FORGE_AREA_LIGHTS_TEXT = "forge_viewer.light.area"
_URDF_MIN_POSITIVE_INERTIA = 1e-14

_BVH_POSE_BODY = 0
_BVH_POSE_GEOM = 1
_BVH_POSE_DYNAMIC = 2


_ACTUATOR_POSE_JOINT_AXIS = 0
_ACTUATOR_POSE_JOINT_BODY = 1
_ACTUATOR_POSE_SITE = 2
_ACTUATOR_POSE_GEOM = 3

_MODEL_COMPONENT_CATEGORIES = (
    "contact",
    "actuator",
    "sensor",
    "tendon",
    "equality",
    "custom",
)
_OBJECT_REFERENCE_TAGS = {
    "body": ("body",),
    "joint": ("joint",),
    "geom": ("geom",),
    "site": ("site",),
    "camera": ("camera",),
    "light": ("light",),
    "flex": ("flex",),
    "mesh": ("mesh",),
    "skin": ("skin",),
    "hfield": ("hfield",),
    "texture": ("texture",),
    "material": ("material",),
    "pair": ("pair",),
    "exclude": ("exclude",),
    "equality": ("equality",),
    "tendon": ("fixed", "spatial"),
    "actuator": ("actuator",),
    "sensor": ("sensor",),
    "numeric": ("numeric",),
    "text": ("text",),
    "tuple": ("tuple",),
    "key": ("key",),
}


def _mjcf_schema_attributes() -> dict[tuple[str, ...], tuple[str, ...]]:
    """Read attribute inventories from the exact linked MuJoCo version."""
    if mujoco is None:
        return {}
    pattern = re.compile(r"^(\s*)(\S+)\s+\([^)]*\)\s*(.*)$")
    attributes: dict[tuple[str, ...], list[str]] = {}
    stack: list[str] = []
    current: tuple[str, ...] | None = None
    for line in mujoco.mj_printSchema(False, True).splitlines():
        match = pattern.match(line)
        if match is not None:
            indent, tag, values = match.groups()
            depth = len(indent) // 3
            stack = stack[:depth]
            current = (*stack, tag)
            stack.append(tag)
            attributes.setdefault(current, []).extend(values.split())
        elif current is not None and line.strip():
            attributes[current].extend(line.split())
    return {path: tuple(dict.fromkeys(values)) for path, values in attributes.items()}


_MJCF_SCHEMA_ATTRIBUTES = _mjcf_schema_attributes()
_GLOBAL_PROPERTY_GROUPS = (
    ("global:compiler", "Compiler", ("compiler",)),
    ("global:compiler/lengthrange", "Actuator Length Range", ("compiler", "lengthrange")),
    ("global:option", "Simulation Options", ("option",)),
    ("global:option/flag", "Simulation Flags", ("option", "flag")),
    ("global:size", "Memory and User Data", ("size",)),
    ("global:statistic", "Model Statistics", ("statistic",)),
    ("global:visual/global", "Visual / Global", ("visual", "global")),
    ("global:visual/quality", "Visual / Quality", ("visual", "quality")),
    ("global:visual/headlight", "Visual / Headlight", ("visual", "headlight")),
    ("global:visual/map", "Visual / Map", ("visual", "map")),
    ("global:visual/scale", "Visual / Scale", ("visual", "scale")),
    ("global:visual/rgba", "Visual / Colors", ("visual", "rgba")),
)
_DEFAULT_PROPERTY_TYPES = tuple(
    path[-1]
    for path in _MJCF_SCHEMA_ATTRIBUTES
    if len(path) == 3
    and path[:2] == ("mujoco", "default")
    and path[-1] not in {"default", "plugin"}
)
_ASSET_PROPERTY_FIELDS = {
    "material": tuple(
        field
        for field in _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", "material"), ())
        if field not in {"name", "texture"}
    ),
    "texture": tuple(
        field
        for field in _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", "texture"), ())
        if field not in {"file", "name"}
    ),
    "mesh": tuple(
        field
        for field in _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", "mesh"), ())
        if field not in {"face", "file", "name", "normal", "texcoord", "vertex"}
    ),
    "hfield": tuple(
        field
        for field in _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", "hfield"), ())
        if field not in {"elevation", "file", "name"}
    ),
}
_MODEL_ASSET_TYPES = ("material", "texture", "mesh", "hfield", "skin", "model")
_MODEL_ASSET_REFERENCE_FIELDS = {
    "material": ("material",),
    "texture": ("texture",),
    "mesh": ("mesh",),
    "hfield": ("hfield",),
    "model": ("model",),
}
_MODEL_PROPERTY_CHOICES = {
    ("global:compiler", "angle"): ("", "degree", "radian"),
    ("global:compiler", "conflict"): ("", "warning", "error"),
    ("global:compiler", "inertiafromgeom"): ("", "false", "true", "auto"),
    ("global:option", "integrator"): ("", "Euler", "RK4", "implicit", "implicitfast", "exact"),
    ("global:option", "cone"): ("", "pyramidal", "elliptic"),
    ("global:option", "jacobian"): ("", "dense", "sparse", "auto"),
    ("global:option", "solver"): ("", "PGS", "CG", "Newton"),
}
_BOOLEAN_PROPERTY_FIELDS = {
    "active",
    "alignfree",
    "autolimits",
    "balanceinertia",
    "bvactive",
    "discardvisual",
    "ellipsoidinertia",
    "fitaabb",
    "fusestatic",
    "orthographic",
    "saveinertial",
    "smoothnormal",
    "strippath",
    "hflip",
    "vflip",
    "texuniform",
    "usethread",
}
_ACTUATOR_TRANSMISSION_FIELDS = (
    "joint",
    "jointinparent",
    "tendon",
    "site",
    "refsite",
    "body",
    "gear",
    "ctrlrange",
    "forcerange",
    "group",
    "armature",
    "damping",
    "delay",
    "nsample",
    "interp",
    "lengthrange",
)
_SENSOR_COMMON_FIELDS = ("cutoff", "noise", "delay", "nsample", "interp", "interval")
_SENSOR_SITE_TYPES = (
    "touch",
    "accelerometer",
    "velocimeter",
    "gyro",
    "force",
    "torque",
    "magnetometer",
    "rangefinder",
    "camprojection",
)
_SENSOR_JOINT_TYPES = ("jointpos", "jointvel", "jointactuatorfrc")
_SENSOR_BALL_TYPES = ("ballquat", "ballangvel")
_SENSOR_JOINT_LIMIT_TYPES = ("jointlimitpos", "jointlimitvel", "jointlimitfrc")
_SENSOR_TENDON_TYPES = ("tendonpos", "tendonvel", "tendonactuatorfrc")
_SENSOR_TENDON_LIMIT_TYPES = ("tendonlimitpos", "tendonlimitvel", "tendonlimitfrc")
_SENSOR_ACTUATOR_TYPES = ("actuatorpos", "actuatorvel", "actuatorfrc")
_SENSOR_FRAME_TYPES = (
    "framepos",
    "framequat",
    "framexaxis",
    "frameyaxis",
    "framezaxis",
    "framelinvel",
    "frameangvel",
)
_SENSOR_FRAME_ACCEL_TYPES = ("framelinacc", "frameangacc")
_SENSOR_SUBTREE_TYPES = ("subtreecom", "subtreelinvel", "subtreeangmom")
_COMPONENT_OPTIONAL_FIELDS = {
    "contact": (),
    "actuator": _ACTUATOR_TRANSMISSION_FIELDS,
    "sensor": _SENSOR_COMMON_FIELDS,
    "tendon": (
        "width",
        "stiffness",
        "damping",
        "frictionloss",
        "springlength",
        "range",
        "limited",
        "margin",
        "group",
        "material",
        "rgba",
        "solreflimit",
        "solimplimit",
        "solreffriction",
        "solimpfriction",
        "actuatorfrclimited",
        "actuatorfrcrange",
        "armature",
    ),
    "equality": ("active", "solref", "solimp", "polycoef"),
    "custom": (),
}
_COMPONENT_SUBTYPE_OPTIONAL_FIELDS = {
    ("contact", "pair"): (
        "geom1",
        "geom2",
        "condim",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
        "margin",
        "gap",
        "adhesion",
    ),
    ("contact", "exclude"): ("body1", "body2"),
    ("actuator", "general"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "dyntype",
        "gaintype",
        "biastype",
        "dynprm",
        "gainprm",
        "biasprm",
        "actdim",
        "actearly",
        "actrange",
        "actlimited",
    ),
    ("actuator", "position"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "kp",
        "kv",
        "dampratio",
        "timeconst",
        "inheritrange",
    ),
    ("actuator", "velocity"): (*_ACTUATOR_TRANSMISSION_FIELDS, "kv"),
    ("actuator", "intvelocity"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "kp",
        "kv",
        "dampratio",
        "inheritrange",
        "actrange",
        "actlimited",
    ),
    ("actuator", "orientation"): (
        "joint",
        "site",
        "refsite",
        "ctrlrange",
        "forcerange",
        "group",
        "kp",
        "kv",
        "dampratio",
        "input",
        "delay",
        "nsample",
        "interp",
    ),
    ("actuator", "damper"): (*_ACTUATOR_TRANSMISSION_FIELDS, "kv"),
    ("actuator", "cylinder"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "timeconst",
        "area",
        "diameter",
        "bias",
    ),
    ("actuator", "muscle"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "timeconst",
        "range",
        "force",
        "scale",
        "lmin",
        "lmax",
        "vmax",
        "fpmax",
        "fvmax",
        "tausmooth",
    ),
    ("actuator", "adhesion"): (
        "body",
        "ctrlrange",
        "forcerange",
        "group",
        "gain",
        "delay",
        "nsample",
        "interp",
    ),
    ("actuator", "dcmotor"): (
        *_ACTUATOR_TRANSMISSION_FIELDS,
        "motorconst",
        "resistance",
        "inductance",
        "cogging",
        "lugre",
        "saturation",
        "thermal",
        "controller",
        "input",
        "nominal",
    ),
    ("equality", "connect"): (
        "body1",
        "body2",
        "site1",
        "site2",
        "anchor",
        "active",
        "solref",
        "solimp",
    ),
    ("equality", "weld"): (
        "body1",
        "body2",
        "site1",
        "site2",
        "anchor",
        "relpose",
        "torquescale",
        "active",
        "solref",
        "solimp",
    ),
    ("equality", "joint"): (
        "joint1",
        "joint2",
        "polycoef",
        "active",
        "solref",
        "solimp",
    ),
    ("equality", "tendon"): (
        "tendon1",
        "tendon2",
        "polycoef",
        "active",
        "solref",
        "solimp",
    ),
    ("equality", "flex"): ("flex", "active", "solref", "solimp"),
    ("equality", "flexvert"): ("flex", "active", "solref", "solimp"),
    ("equality", "flexstrain"): ("flex", "cell", "active", "solref", "solimp"),
    **{
        ("sensor", subtype): ("site", *_SENSOR_COMMON_FIELDS)
        for subtype in _SENSOR_SITE_TYPES
        if subtype not in {"rangefinder", "camprojection"}
    },
    ("sensor", "rangefinder"): ("site", "camera", "data", *_SENSOR_COMMON_FIELDS),
    ("sensor", "camprojection"): ("site", "camera", *_SENSOR_COMMON_FIELDS),
    **{
        ("sensor", subtype): ("joint", *_SENSOR_COMMON_FIELDS)
        for subtype in (*_SENSOR_JOINT_TYPES, *_SENSOR_BALL_TYPES, *_SENSOR_JOINT_LIMIT_TYPES)
    },
    **{
        ("sensor", subtype): ("tendon", *_SENSOR_COMMON_FIELDS)
        for subtype in (*_SENSOR_TENDON_TYPES, *_SENSOR_TENDON_LIMIT_TYPES)
    },
    **{
        ("sensor", subtype): ("actuator", *_SENSOR_COMMON_FIELDS)
        for subtype in _SENSOR_ACTUATOR_TYPES
    },
    **{
        ("sensor", subtype): (
            "objtype",
            "objname",
            "reftype",
            "refname",
            *_SENSOR_COMMON_FIELDS,
        )
        for subtype in _SENSOR_FRAME_TYPES
    },
    **{
        ("sensor", subtype): ("objtype", "objname", *_SENSOR_COMMON_FIELDS)
        for subtype in _SENSOR_FRAME_ACCEL_TYPES
    },
    **{("sensor", subtype): ("body", *_SENSOR_COMMON_FIELDS) for subtype in _SENSOR_SUBTREE_TYPES},
    ("sensor", "insidesite"): (
        "site",
        "objtype",
        "objname",
        *_SENSOR_COMMON_FIELDS,
    ),
    **{
        ("sensor", subtype): ("body1", "body2", "geom1", "geom2", *_SENSOR_COMMON_FIELDS)
        for subtype in ("distance", "normal", "fromto")
    },
    ("sensor", "contact"): (
        "body1",
        "body2",
        "geom1",
        "geom2",
        "site",
        "subtree1",
        "subtree2",
        "data",
        "num",
        "reduce",
        *_SENSOR_COMMON_FIELDS,
    ),
    **{
        ("sensor", subtype): _SENSOR_COMMON_FIELDS
        for subtype in ("e_potential", "e_kinetic", "clock")
    },
    ("sensor", "tactile"): ("geom", "mesh", "delay", "nsample", "interp", "interval"),
    ("sensor", "user"): (
        "objtype",
        "objname",
        "datatype",
        "needstage",
        "dim",
        "cutoff",
        "noise",
    ),
}
_REFERENCE_ELEMENT = {
    "actuator": "actuator",
    "actuator1": "actuator",
    "actuator2": "actuator",
    "body": "body",
    "body1": "body",
    "body2": "body",
    "camera": "camera",
    "cranksite": "site",
    "flex": "flex",
    "geom": "geom",
    "geom1": "geom",
    "geom2": "geom",
    "joint": "joint",
    "jointinparent": "joint",
    "joint1": "joint",
    "joint2": "joint",
    "mesh": "mesh",
    "refsite": "site",
    "site": "site",
    "site1": "site",
    "site2": "site",
    "slidersite": "site",
    "subtree1": "body",
    "subtree2": "body",
    "tendon": "tendon",
    "tendon1": "tendon",
    "tendon2": "tendon",
}

_FLEX_COPY_FIELDS = (
    "contype",
    "conaffinity",
    "condim",
    "priority",
    "friction",
    "solmix",
    "solref",
    "solimp",
    "margin",
    "gap",
    "dim",
    "radius",
    "size",
    "internal",
    "flatskin",
    "selfcollide",
    "passive",
    "activelayers",
    "group",
    "edgestiffness",
    "edgedamping",
    "rgba",
    "young",
    "poisson",
    "damping",
    "thickness",
    "elastic2d",
    "cellcount",
    "order",
    "elem",
    "texcoord",
    "elemtexcoord",
    "info",
)


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
    return root, xml


def _component_section(root: ET.Element, category: str, *, create: bool = False):
    if category not in _MODEL_COMPONENT_CATEGORIES:
        raise ValueError(f"Unsupported model component category: {category}")
    section = root.find(category)
    if section is None and create:
        section = ET.SubElement(root, category)
    return section


def _named_elements(root: ET.Element, tag: str) -> tuple[str, ...]:
    if tag in _MODEL_COMPONENT_CATEGORIES:
        section = root.find(tag)
        return (
            tuple(
                value
                for element in section or ()
                if (value := str(element.attrib.get("name", "")).strip())
            )
            if section is not None
            else ()
        )
    return tuple(
        value
        for element in root.iter(tag)
        if (value := str(element.attrib.get("name", "")).strip())
    )


def _object_reference_names(root: ET.Element, object_type: str) -> tuple[str, ...]:
    tags = _OBJECT_REFERENCE_TAGS.get(str(object_type).lower(), ())
    names = tuple(dict.fromkeys(name for tag in tags for name in _named_elements(root, tag)))
    if object_type == "body":
        return ("world", *names)
    return names


def _field_choices(root: ET.Element, name: str, attributes: dict[str, str]) -> tuple[str, ...]:
    if name == "class":
        return tuple(
            value
            for element in root.iter("default")
            if (value := str(element.attrib.get("class", "")).strip())
        )
    if name in {"objtype", "reftype"}:
        return tuple(
            object_type
            for object_type in _OBJECT_REFERENCE_TAGS
            if _object_reference_names(root, object_type)
        )
    if name == "objname":
        return _object_reference_names(root, attributes.get("objtype", ""))
    if name == "refname":
        return _object_reference_names(root, attributes.get("reftype", ""))
    target = _REFERENCE_ELEMENT.get(name)
    return _named_elements(root, target) if target else ()


def _component_fields(
    root: ET.Element, category: str, subtype: str, attributes: dict[str, str]
) -> tuple[ModelComponentField, ...]:
    values = {name: value for name, value in attributes.items() if name != "name"}
    curated = _COMPONENT_SUBTYPE_OPTIONAL_FIELDS.get(
        (category, subtype), _COMPONENT_OPTIONAL_FIELDS[category]
    )
    schema = _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", category, subtype), ())
    optional = tuple(dict.fromkeys((*curated, *schema)))
    for name in optional:
        if name != "name":
            values.setdefault(name, "")

    tri_state = {
        "actlimited",
        "actuatorfrclimited",
        "ctrllimited",
        "forcelimited",
        "limited",
    }

    def choices(name: str) -> tuple[str, ...]:
        references = _field_choices(root, name, values)
        if references:
            return ("", *references)
        if name in tri_state:
            return ("", "false", "true", "auto")
        if name in _BOOLEAN_PROPERTY_FIELDS:
            return ("", "false", "true")
        return ()

    return tuple(ModelComponentField(name, value, choices(name)) for name, value in values.items())


def _component_path_presets(
    root: ET.Element, category: str, subtype: str
) -> tuple[ModelComponentPathItem, ...]:
    if category == "custom" and subtype == "tuple":
        object_types = _field_choices(root, "objtype", {})
        return tuple(
            ModelComponentPathItem(
                "element",
                (
                    ModelComponentField("objtype", object_type, object_types),
                    ModelComponentField(
                        "objname",
                        names[0],
                        names,
                    ),
                    ModelComponentField("prm", "0"),
                ),
            )
            for object_type in object_types
            if (names := _object_reference_names(root, object_type))
        )
    if category != "tendon":
        return ()
    if subtype == "fixed":
        joints = _named_elements(root, "joint")
        return (
            ModelComponentPathItem(
                "joint",
                (
                    ModelComponentField("joint", joints[0] if joints else "", joints),
                    ModelComponentField("coef", "1"),
                ),
            ),
        )
    sites = _named_elements(root, "site")
    geoms = _named_elements(root, "geom")
    presets = []
    if sites:
        presets.append(
            ModelComponentPathItem("site", (ModelComponentField("site", sites[0], sites),))
        )
    if geoms:
        presets.append(
            ModelComponentPathItem(
                "geom",
                (
                    ModelComponentField("geom", geoms[0], geoms),
                    ModelComponentField("sidesite", sites[0] if sites else "", sites),
                ),
            )
        )
    presets.append(ModelComponentPathItem("pulley", (ModelComponentField("divisor", "2"),)))
    return tuple(presets)


def _component_path_fields(
    root: ET.Element,
    category: str,
    subtype: str,
    child: ET.Element,
) -> tuple[ModelComponentField, ...]:
    values = dict(child.attrib)
    if category == "custom" and subtype == "tuple":
        for name in _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "custom", "tuple", "element"), ()):
            values.setdefault(name, "")
    return tuple(
        ModelComponentField(name, value, _field_choices(root, name, values))
        for name, value in values.items()
    )


def _model_property_choices(
    root: ET.Element,
    group_id: str,
    field: str,
    attributes: dict[str, str],
) -> tuple[str, ...]:
    choices = _MODEL_PROPERTY_CHOICES.get((group_id, field))
    if choices is not None:
        return choices
    if group_id == "global:option/flag":
        return ("", "enable", "disable")
    if group_id.startswith("asset:texture:"):
        texture_choices = {
            "type": ("", "2d", "cube", "skybox"),
            "colorspace": ("", "auto", "srgb", "linear"),
            "builtin": ("", "gradient", "checker", "flat"),
            "mark": ("", "none", "edge", "cross", "random"),
        }
        if field in texture_choices:
            return texture_choices[field]
    if field in _BOOLEAN_PROPERTY_FIELDS:
        return ("", "true", "false")
    references = _field_choices(root, field, attributes)
    return ("", *references) if references else ()


def _model_property_fields(
    root: ET.Element,
    group_id: str,
    names: tuple[str, ...],
    attributes: dict[str, str],
) -> tuple[ModelComponentField, ...]:
    return tuple(
        ModelComponentField(
            name,
            attributes.get(name, ""),
            _model_property_choices(root, group_id, name, attributes),
        )
        for name in names
    )


def _model_asset_references(
    root: ET.Element, asset_type: str, name: str, target: ET.Element
) -> tuple[str, ...]:
    """Return stable human-readable references to one model-local asset."""

    reference_fields = _MODEL_ASSET_REFERENCE_FIELDS.get(asset_type, ())
    if not reference_fields or not name:
        return ()
    references: list[str] = []
    tag_indices: dict[str, int] = {}
    for element in root.iter():
        tag_indices[element.tag] = tag_indices.get(element.tag, 0) + 1
        if element is target:
            continue
        if not any(element.attrib.get(field) == name for field in reference_fields):
            continue
        element_name = str(element.attrib.get("name", "")).strip()
        label = (
            f"{element.tag} {element_name}"
            if element_name
            else f"{element.tag} #{tag_indices[element.tag]}"
        )
        references.append(label)
    return tuple(references)


def _model_asset_element(
    root: ET.Element, asset_type: str, name: str
) -> tuple[ET.Element | None, ET.Element | None]:
    asset = root.find("asset")
    if asset is None:
        return None, None
    target = next(
        (
            element
            for element in asset.findall(asset_type)
            if str(element.attrib.get("name", "")).strip() == name
        ),
        None,
    )
    return asset, target


def _ensure_model_asset_section(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is not None:
        return asset
    asset = ET.Element("asset")
    children = tuple(root)
    insertion = next(
        (
            index
            for index, child in enumerate(children)
            if child.tag
            in {
                "worldbody",
                "deformable",
                "contact",
                "equality",
                "tendon",
                "actuator",
                "sensor",
                "keyframe",
            }
        ),
        len(children),
    )
    root.insert(insertion, asset)
    return asset


def _find_or_create_xml_path(root: ET.Element, path: tuple[str, ...]) -> ET.Element:
    element = root
    for tag in path:
        child = element.find(tag)
        if child is None:
            child = ET.SubElement(element, tag)
        element = child
    return element


def _serialize_component_xml(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _format_mjcf_values(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _grid_node(start: int, ny: int, nz: int, i: int, j: int, k: int) -> int:
    return start + i * ny * nz + j * nz + k


def _grid_boundary(nx: int, ny: int, nz: int, i: int, j: int, k: int) -> bool:
    return i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1)


_RAY_FIELD_WIDTHS = (1, 3, 3, 3, 3, 1)
_RAY_DIST = 1 << 0
_RAY_DIR = 1 << 1
_RAY_ORIGIN = 1 << 2
_RAY_POINT = 1 << 3
_RAY_NORMAL = 1 << 4


def _object_names(model, object_type, count: int, prefix: str) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or f"{prefix}{index}" for index in range(count)
    )


@dataclass(frozen=True)
class _RangefinderSpec:
    sensor: int
    fields: int
    ray_count: int
    stride: int
    frame_offset: int
    object_type: int
    object_id: int


@dataclass(frozen=True)
class _AttachedModel:
    model_id: int
    name: str
    path: Path
    prefix: str
    position: np.ndarray
    rotation: np.ndarray
    spec: object
    edited: bool = False


@dataclass
class _ModelTransformPreview:
    """Render-only placement while an attached model gizmo is active."""

    model_id: int
    previous_position: np.ndarray
    previous_rotation: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    delta_rotation: np.ndarray
    body_indices: np.ndarray
    geom_indices: np.ndarray
    site_indices: np.ndarray
    joint_indices: np.ndarray
    light_mask: np.ndarray
    camera_mask: np.ndarray
    point_input: np.ndarray
    point_output: np.ndarray
    matrix_input: np.ndarray
    matrix_output: np.ndarray


@dataclass(frozen=True)
class _NamedModelState:
    joints: dict[str, tuple[np.ndarray, np.ndarray]]
    actuators: dict[str, tuple[np.ndarray, np.ndarray]]
    mocap: dict[str, tuple[np.ndarray, np.ndarray]]
    equality: dict[str, bool]
    time: float


@dataclass(frozen=True)
class _ModelComponentEntry:
    component_id: int
    subtype: str
    name: str
    signature: str


@dataclass(frozen=True)
class _CompositionEditState:
    models: tuple[_AttachedModel, ...]
    physics: PhysicsState
    root_spec: object
    root_edited: bool
    geometry_object_ids: dict[tuple[int, str], int]
    next_geometry_object_id: int
    component_entries: dict[tuple[int, str], tuple[_ModelComponentEntry, ...]]


class MuJoCoAdapter(SceneAdapterBase):
    """Expose MuJoCo model structure, simulation state, and authoring through scene contracts."""

    def __init__(self, path: Path | None = None) -> None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(
                f"MuJoCo is not installed: {_IMPORT_ERROR}. Install the [mujoco] optional dependency."
            )
        self.caps = AdapterCaps(
            name="mujoco",
            simulation=True,
            asset_loading=True,
            write_pose=True,
            write_qpos=True,
            perturb=True,
            raycast=True,
            state_snapshots=True,
            contacts=True,
            model_cameras=True,
            keyframes=True,
            sensors=True,
            equality_constraints=True,
            visual_groups=True,
            reload=True,
            model_composition=True,
            topology_editing=True,
            model_properties=True,
            model_assets=True,
        )
        self._m = None
        self._d = None
        self._path: Path | None = None
        self._root_path: Path | None = None
        self._root_spec = None
        self._root_edited = False
        self._attached_models: list[_AttachedModel] = []
        self._model_transform_preview: _ModelTransformPreview | None = None
        self._next_model_id = 1
        self._structure_revision = 0
        self._notes: list[str] = []

        self._geom_xpos_buf = np.zeros((0, 3), np.float32)
        self._geom_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((0, 3), np.float32)
        self._site_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((0, 3), np.float32)
        self._body_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._diagnostic_frame = DiagnosticFrame()
        self._qpos_buf = np.zeros(0, np.float32)
        self._qvel_buf = np.zeros(0, np.float32)
        self._ctrl_buf = np.zeros(0, np.float32)
        self._equality_enabled_buf = np.zeros(0, bool)
        self._sensor_buf = np.zeros(0, np.float32)
        self._contact_buf = np.zeros((0, 7), np.float32)
        self._contact_force = np.zeros(6, np.float64)
        self._contact_view = self._contact_buf
        self._contact_force_buf = np.zeros((0, 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf
        self._contact_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._contact_island_rgba_view = self._contact_island_rgba_buf
        self._island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._body_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_segments = np.zeros((0, 2, 3), np.float32)
        self._tendon_ids = np.zeros(0, np.int32)
        self._tendon_widths = np.zeros(0, np.float32)
        self._actuator_visual_pose_types = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)
        self._slider_crank_actuators = np.zeros(0, np.int32)
        self._bvh_pose_type = np.zeros(0, np.uint8)
        self._bvh_pose_source = np.zeros(0, np.int32)
        self._bvh_global_index = np.zeros(0, np.int32)
        self._bvh_local_center = np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.zeros((0, 3), np.float32)
        self._bvh_control_body = np.zeros((0, 2), np.int32)
        self._bvh_control_local = np.zeros((0, 2, 3), np.float32)
        self._bvh_source_ready = False
        self._rangefinder_specs: tuple[_RangefinderSpec, ...] = ()

        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_wrap_objects = None
        self._mj_body_xpos = None
        self._mj_body_xmat3 = None
        self._fast_pose = False

        self._ray_pnt = np.zeros(3, np.float64)
        self._ray_vec = np.zeros(3, np.float64)
        self._ray_geomid = np.zeros(1, np.int32)

        defaults = np.array([g in DEFAULT_GEOM_GROUPS for g in range(6)], dtype=bool)
        self._visual_groups = {name: defaults.copy() for name in VISUAL_GROUP_CATEGORIES}
        self._ray_geomgroup = self._visual_groups["geom"].astype(np.uint8)

        self._perturb = mujoco.MjvPerturb()
        self._perturb_body = -1
        self._perturb_jac = np.zeros((3, 0), np.float64)
        self._perturb_jac_m2 = np.zeros((3, 0), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(0, np.float64)
        self._perturb_quat = np.zeros(4, np.float64)

        self._frame = SceneFrame()
        self._source: SceneSource | None = None
        self._nodes: list[SceneNode] = []
        self._node_body: dict[int, int] = {}
        self._node_model: dict[int, int] = {}
        self._node_element: dict[int, tuple[int, NodeType, str]] = {}
        self._model_element_names: dict[tuple[int, str], tuple[int, str]] = {}
        self._geometry_object_ids: dict[tuple[int, str], int] = {}
        self._next_geometry_object_id = GEOMETRY_OBJECT_BASE
        self._component_entries: dict[tuple[int, str], tuple[_ModelComponentEntry, ...]] = {}
        self._next_component_id: dict[tuple[int, str], int] = {}
        self._geom_nodes: dict[int, int] = {}
        self._site_nodes: dict[int, int] = {}
        self._flex_nodes: dict[int, int] = {}
        self._skin_nodes: dict[int, int] = {}
        self._deformables = []
        self._mesh_updates = {}
        self._lights_dynamic = False
        self._lights_edited = False
        self._area_lights = np.zeros(0, bool)

        if path is not None:
            self.load(path)

    def load(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        try:
            spec = _load_editable_spec(path)
            model = spec.compile()
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        self._path = path
        self._root_path = path
        self._root_spec = spec
        self._root_edited = False
        self._attached_models.clear()
        self._next_model_id = 1
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=True)
        self._install(model)

    def new_scene(self) -> None:
        self._path = None
        self._root_path = None
        self._root_spec = mujoco.MjSpec()
        self._root_edited = False
        self._attached_models.clear()
        self._next_model_id = 1
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=True)
        self._install(self._root_spec.compile())

    def load_model(self, model, data=None) -> None:
        """Install an existing MuJoCo model and optional data object."""
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        if data is not None and data.model is not model:
            raise ValueError("data was created for a different MuJoCo model")
        self._path = None
        self._root_path = None
        self._root_spec = None
        self._root_edited = False
        self._attached_models.clear()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=False)
        self._install(model, data)

    def use_data(self, data) -> None:
        """Bind dynamic state supplied by a programmatic rendering workflow."""
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not self._m:
            raise ValueError("data was created for a different MuJoCo model")
        if data is self._d:
            return
        self._d = data
        self._bind_data_views()

    def apply_scene_option(self, option) -> bool:
        """Apply MuJoCo visual-group visibility to the stable scene source."""
        changed = False
        fields = {
            "geom": "geomgroup",
            "site": "sitegroup",
            "joint": "jointgroup",
            "tendon": "tendongroup",
            "flex": "flexgroup",
            "skin": "skingroup",
        }
        for category, field in fields.items():
            visible = np.asarray(getattr(option, field), bool)
            groups = self._visual_groups[category]
            if not np.array_equal(groups, visible):
                groups[:] = visible
                changed = True
        if not changed:
            return False
        self._ray_geomgroup[:] = self._visual_groups["geom"]
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def refresh_model_visuals(self) -> bool:
        """Refresh cached scene data after direct MjModel visual edits."""
        source_changed = self._refresh_snapshots(self._visual_state)
        lights_changed = self._refresh_snapshots(self._light_state)
        if source_changed:
            self._source = None
            self._structure_revision += 1
        if lights_changed:
            self._lights_edited = True
        return source_changed or lights_changed

    def _refresh_snapshots(self, snapshots: dict[str, np.ndarray]) -> bool:
        changed = False
        for name, previous in snapshots.items():
            current = np.asarray(getattr(self._m, name))
            if not np.array_equal(current, previous):
                np.copyto(previous, current)
                changed = True
        return changed

    def reload(self) -> None:
        if self._root_path is None and self._root_spec is None:
            raise RuntimeError("No asset has been loaded")
        if self._root_path is not None:
            self._root_spec = _load_editable_spec(self._root_path)
        self._install(self._compile_composed_model())

    def scene_models(self) -> tuple[SceneModelInfo, ...]:
        roots = (
            (SceneModelInfo(0, self._root_path.stem, self._root_path, False),)
            if self._root_path is not None
            else ()
        )
        preview = self._model_transform_preview
        attached = []
        for item in self._attached_models:
            position = item.position
            rotation = item.rotation
            if preview is not None and preview.model_id == item.model_id:
                position = preview.position
                rotation = preview.rotation
            attached.append(
                SceneModelInfo(
                    item.model_id,
                    item.name,
                    item.path,
                    True,
                    tuple(float(value) for value in position),
                    tuple(tuple(float(value) for value in row) for row in rotation),
                )
            )
        return (*roots, *attached)

    def capture_edit_state(self) -> object | None:
        if self._root_spec is None:
            return None
        models = tuple(
            _AttachedModel(
                item.model_id,
                item.name,
                item.path,
                item.prefix,
                item.position.copy(),
                item.rotation.copy(),
                item.spec.copy(),
                item.edited,
            )
            for item in self._attached_models
        )
        return _CompositionEditState(
            models,
            self.capture_state(),
            self._root_spec.copy(),
            self._root_edited,
            dict(self._geometry_object_ids),
            self._next_geometry_object_id,
            dict(self._component_entries),
        )

    def restore_edit_state(self, state: object) -> bool:
        if not isinstance(state, _CompositionEditState) or self._root_spec is None:
            return False
        self._attached_models = [
            _AttachedModel(
                item.model_id,
                item.name,
                item.path,
                item.prefix,
                item.position.copy(),
                item.rotation.copy(),
                item.spec.copy(),
                item.edited,
            )
            for item in state.models
        ]
        self._root_spec = state.root_spec.copy()
        self._root_edited = state.root_edited
        self._geometry_object_ids = dict(state.geometry_object_ids)
        self._next_geometry_object_id = state.next_geometry_object_id
        self._component_entries = dict(state.component_entries)
        for key, entries in self._component_entries.items():
            self._next_component_id[key] = max(
                self._next_component_id.get(key, 0),
                max((entry.component_id + 1 for entry in entries), default=0),
            )
        self._next_model_id = max((item.model_id for item in state.models), default=0) + 1
        self._install(self._compile_composed_model())
        return self.restore_state(state.physics)

    def add_scene_model(self, path: Path, position, rotation) -> int:
        if self._root_spec is None:
            return -1
        path = Path(path).expanduser().resolve()
        model_id = self._next_model_id
        spec = _load_editable_spec(path)
        # MjSpec.copy() can omit unresolved declarations originating in an include.
        # Compiling keyed models once resolves their full actuator/state layout before
        # the stored editable spec is copied for composition.
        self._resolve_attached_keyframes(spec)
        item = _AttachedModel(
            model_id=model_id,
            name=path.stem,
            path=path,
            prefix=f"forge_{model_id}_",
            position=np.asarray(position, np.float64).reshape(3).copy(),
            rotation=np.asarray(rotation, np.float64).reshape(3, 3).copy(),
            spec=spec,
        )
        state = self._capture_named_model_state()
        self._attached_models.append(item)
        try:
            model = self._compile_composed_model()
        except Exception as exc:
            self._attached_models.pop()
            raise RuntimeError(f"Failed to add {path}: {exc}") from exc
        self._next_model_id += 1
        self._install(model)
        self._restore_named_model_state(state)
        return model_id

    def remove_scene_model(self, model_id: int) -> bool:
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index < 0:
            return False
        state = self._capture_named_model_state()
        item = self._attached_models.pop(index)
        try:
            model = self._compile_composed_model()
        except Exception as exc:
            self._attached_models.insert(index, item)
            raise RuntimeError(f"Failed to remove {item.name}: {exc}") from exc
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def set_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return False
        next_position = np.asarray(position, np.float64).reshape(3)
        next_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        if np.array_equal(item.position, next_position) and np.array_equal(
            item.rotation, next_rotation
        ):
            self._model_transform_preview = None
            return True
        previous_position = item.position.copy()
        previous_rotation = item.rotation.copy()
        state = self._transform_named_model_state(
            self._capture_named_model_state(),
            item.prefix,
            previous_position,
            previous_rotation,
            next_position,
            next_rotation,
        )
        item.position[:] = next_position
        item.rotation[:] = next_rotation
        try:
            model = self._compile_composed_model()
        except Exception:
            item.position[:] = previous_position
            item.rotation[:] = previous_rotation
            raise
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def preview_scene_model_transform(self, model_id: int, position, rotation) -> bool:
        """Preview placement in frame buffers; physics remains committed until release."""
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return False
        next_position = np.asarray(position, np.float64).reshape(3)
        next_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        if not np.all(np.isfinite(next_position)) or not np.all(np.isfinite(next_rotation)):
            raise ValueError("Model transform preview must contain finite values")
        preview = self._model_transform_preview
        if preview is None or preview.model_id != item.model_id:
            preview = self._make_model_transform_preview(item)
            self._model_transform_preview = preview
        preview.position[:] = next_position
        preview.rotation[:] = next_rotation
        preview.delta_rotation[:] = next_rotation @ item.rotation.T
        return True

    def clear_scene_model_transform_preview(self, model_id: int) -> bool:
        preview = self._model_transform_preview
        if preview is None or preview.model_id != int(model_id):
            return False
        self._model_transform_preview = None
        return True

    def _make_model_transform_preview(self, item: _AttachedModel) -> _ModelTransformPreview:
        model = self._m
        body_owner = np.zeros(model.nbody, np.int32)
        for body in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            owner, _ = self._model_element_name(name, mujoco.mjtObj.mjOBJ_BODY)
            body_owner[body] = owner if owner else body_owner[int(model.body_parentid[body])]

        def owned_indices(object_type, count: int, body_ids=None) -> np.ndarray:
            owned = []
            for index in range(count):
                name = mujoco.mj_id2name(model, object_type, index) or ""
                owner, _ = self._model_element_name(name, object_type)
                if not owner and body_ids is not None:
                    owner = int(body_owner[int(body_ids[index])])
                if owner == item.model_id:
                    owned.append(index)
            return np.asarray(owned, np.intp)

        body_indices = np.flatnonzero(body_owner == item.model_id).astype(np.intp)
        geom_indices = owned_indices(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, model.geom_bodyid)
        site_indices = owned_indices(mujoco.mjtObj.mjOBJ_SITE, model.nsite, model.site_bodyid)
        joint_indices = owned_indices(mujoco.mjtObj.mjOBJ_JOINT, model.njnt, model.jnt_bodyid)
        light_indices = owned_indices(mujoco.mjtObj.mjOBJ_LIGHT, model.nlight, model.light_bodyid)
        camera_indices = owned_indices(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam, model.cam_bodyid)
        light_mask = np.zeros(model.nlight, bool)
        camera_mask = np.zeros(model.ncam, bool)
        light_mask[light_indices] = True
        camera_mask[camera_indices] = True
        capacity = max(
            1,
            len(body_indices),
            len(geom_indices),
            len(site_indices),
            len(joint_indices),
        )
        return _ModelTransformPreview(
            model_id=item.model_id,
            previous_position=item.position.copy(),
            previous_rotation=item.rotation.copy(),
            position=item.position.copy(),
            rotation=item.rotation.copy(),
            delta_rotation=np.eye(3, dtype=np.float64),
            body_indices=body_indices,
            geom_indices=geom_indices,
            site_indices=site_indices,
            joint_indices=joint_indices,
            light_mask=light_mask,
            camera_mask=camera_mask,
            point_input=np.zeros((capacity, 3), np.float32),
            point_output=np.zeros((capacity, 3), np.float32),
            matrix_input=np.zeros((capacity, 3, 3), np.float32),
            matrix_output=np.zeros((capacity, 3, 3), np.float32),
        )

    def add_model_element(self, parent_node_id: int, element_type: str, name: str) -> int:
        if self._model_parent(int(parent_node_id)) is None or not str(name).strip():
            return -1
        results = self.apply_model_edit_batch(
            (AddModelElementEdit(ModelElementRef(node_id=int(parent_node_id)), element_type, name),)
        )
        return results[0] if results else -1

    def remove_model_element(self, node_id: int) -> bool:
        identity = self._node_element.get(int(node_id))
        if identity is None or identity[1] not in {
            NodeType.LINK,
            NodeType.ROBOT,
            NodeType.GEOM,
            NodeType.JOINT,
            NodeType.SITE,
            NodeType.CAMERA,
            NodeType.LIGHT,
        }:
            return False
        return bool(
            self.apply_model_edit_batch(
                (RemoveModelElementEdit(ModelElementRef(node_id=int(node_id))),)
            )
        )

    def rename_model_element(self, node_id: int, name: str) -> bool:
        identity = self._node_element.get(int(node_id))
        value = str(name).strip()
        if identity is None or not value:
            return False
        if identity[1] in (NodeType.WORLD, NodeType.MODEL) or value == identity[2]:
            return value == identity[2]
        return bool(
            self.apply_model_edit_batch(
                (RenameModelElementEdit(ModelElementRef(node_id=int(node_id)), name),)
            )
        )

    def apply_model_edit_batch(self, edits: tuple[ModelEdit, ...]) -> tuple[int, ...]:
        operations = tuple(edits)
        if self._root_spec is None or not operations:
            return ()
        if not all(
            isinstance(
                edit,
                (AddModelElementEdit, RemoveModelElementEdit, RenameModelElementEdit),
            )
            for edit in operations
        ):
            raise ValueError("Unsupported model edit operation")

        # Node IDs describe the currently installed hierarchy. Keep their semantic
        # identities while edits rename elements and before the hierarchy is rebuilt.
        nodes_by_id = {node.node_id: node for node in self.nodes()}
        node_identities: dict[int, tuple[int, NodeType, str]] = {}
        batch_identities: dict[str, tuple[int, NodeType, str]] = {}
        batch_elements: dict[str, Any] = {}
        result_identities: list[tuple[int, NodeType, str] | None] = []

        def node_identity(node_id: int) -> tuple[int, NodeType, str]:
            value = node_identities.get(int(node_id))
            if value is None:
                value = self._node_element.get(int(node_id))
                if value is None:
                    raise ValueError(f"Unknown model node_id={node_id}")
                node_identities[int(node_id)] = value
            return value

        def identity(ref: ModelElementRef) -> tuple[int, NodeType, str]:
            key = str(ref.batch_key).strip()
            has_node = int(ref.node_id) >= 0
            if has_node == bool(key):
                raise ValueError("A model element reference must use one node ID or batch key")
            if key:
                value = batch_identities.get(key)
                if value is None:
                    raise ValueError(f"Unknown model edit batch key {key!r}")
                return value
            return node_identity(int(ref.node_id))

        def parent_identity(ref: ModelElementRef) -> tuple[int, NodeType, str]:
            if str(ref.batch_key).strip():
                return identity(ref)
            node = nodes_by_id.get(int(ref.node_id))
            while node is not None:
                value = node_identities.get(node.node_id)
                if value is None:
                    value = self._node_element.get(node.node_id)
                    if value is not None:
                        node_identities[node.node_id] = value
                if value is not None and value[1] in {
                    NodeType.MODEL,
                    NodeType.WORLD,
                    NodeType.LINK,
                    NodeType.ROBOT,
                }:
                    return value
                node = nodes_by_id.get(node.parent)
            raise ValueError(f"Model node_id={ref.node_id} cannot own topology children")

        previous_root = self._root_spec
        previous_models = self._attached_models
        previous_root_edited = self._root_edited
        previous_object_ids = dict(self._geometry_object_ids)
        previous_next_object_id = self._next_geometry_object_id
        self._root_spec = previous_root.copy()
        self._attached_models = [replace(item, spec=item.spec.copy()) for item in previous_models]
        changed_models: set[int] = set()

        try:
            for edit in operations:
                if isinstance(edit, AddModelElementEdit):
                    model_id, parent_type, parent_name = parent_identity(edit.parent)
                    spec = self._spec_for_model(model_id)
                    if spec is None:
                        raise ValueError(f"Model {model_id} is unavailable")
                    if parent_type in (NodeType.MODEL, NodeType.WORLD):
                        body = spec.worldbody
                    elif parent_type in (NodeType.LINK, NodeType.ROBOT):
                        parent_key = str(edit.parent.batch_key).strip()
                        body = (
                            batch_elements.get(parent_key) if parent_key else spec.body(parent_name)
                        )
                    else:
                        body = None
                    if body is None:
                        raise ValueError(f"Model parent {parent_name!r} is unavailable")

                    value = str(edit.name).strip()
                    if not value:
                        raise ValueError("A model element name cannot be empty")
                    type_name, _, subtype = str(edit.element_type).partition(":")
                    if self._element(model_id, type_name, value) is not None:
                        raise ValueError(f"{type_name} {value!r} already exists")
                    if type_name == "body":
                        element = body.add_body(name=value)
                    elif type_name == "geom":
                        element = body.add_geom(name=value)
                        element.type = {
                            "box": mujoco.mjtGeom.mjGEOM_BOX,
                            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
                            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
                        }.get(subtype, mujoco.mjtGeom.mjGEOM_SPHERE)
                        element.size = [4.0, 4.0, 0.02] if subtype == "plane" else [0.1, 0.1, 0.1]
                    elif type_name == "joint":
                        element = body.add_joint(name=value)
                        element.type = {
                            "slide": mujoco.mjtJoint.mjJNT_SLIDE,
                            "ball": mujoco.mjtJoint.mjJNT_BALL,
                            "free": mujoco.mjtJoint.mjJNT_FREE,
                        }.get(subtype, mujoco.mjtJoint.mjJNT_HINGE)
                    elif type_name == "site":
                        element = body.add_site(name=value)
                        element.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        element.size = [0.03, 0.03, 0.03]
                    elif type_name == "camera":
                        element = body.add_camera(name=value)
                    elif type_name == "light":
                        element = body.add_light(name=value)
                    else:
                        raise ValueError(f"Unsupported model element type {type_name!r}")
                    target_type = NodeType(type_name if type_name != "body" else "link")
                    target = (model_id, target_type, value)
                    key = str(edit.key).strip()
                    if key:
                        if key in batch_identities:
                            raise ValueError(f"Duplicate model edit batch key {key!r}")
                        batch_identities[key] = target
                        batch_elements[key] = element
                    result_identities.append(target)
                    changed_models.add(model_id)
                    continue

                target = identity(edit.target)
                model_id, node_type, current = target
                target_key = str(edit.target.batch_key).strip()
                element = (
                    batch_elements.get(target_key)
                    if target_key
                    else self._element(model_id, node_type.value, current)
                )
                spec = self._spec_for_model(model_id)
                if spec is None or element is None:
                    raise ValueError(f"{node_type.value} {current!r} is unavailable")

                if isinstance(edit, RemoveModelElementEdit):
                    if node_type not in {
                        NodeType.LINK,
                        NodeType.ROBOT,
                        NodeType.GEOM,
                        NodeType.JOINT,
                        NodeType.SITE,
                        NodeType.CAMERA,
                        NodeType.LIGHT,
                    }:
                        raise ValueError(f"{node_type.value} {current!r} cannot be removed")
                    spec.delete(element)
                    result_identities.append(None)
                    changed_models.add(model_id)
                    continue

                value = str(edit.name).strip()
                if not value:
                    raise ValueError("A model element name cannot be empty")
                if node_type in (NodeType.WORLD, NodeType.MODEL):
                    raise ValueError(f"{node_type.value} {current!r} cannot be renamed")
                if value == current:
                    result_identities.append(target)
                    continue
                if self._element(model_id, node_type.value, value) is not None:
                    raise ValueError(f"{node_type.value} {value!r} already exists")
                if node_type is NodeType.GEOM:
                    object_id = self._geometry_object_ids.pop((model_id, current), None)
                    if object_id is not None:
                        self._geometry_object_ids[(model_id, value)] = object_id
                element.name = value
                renamed = (model_id, node_type, value)
                batch_identities.update(
                    (key, renamed if item == target else item)
                    for key, item in tuple(batch_identities.items())
                )
                node_identities.update(
                    (node_id, renamed if item == target else item)
                    for node_id, item in tuple(node_identities.items())
                )
                result_identities = [
                    renamed if item == target else item for item in result_identities
                ]
                result_identities.append(renamed)
                changed_models.add(model_id)

            if not changed_models:
                self._root_spec = previous_root
                self._attached_models = previous_models
            else:
                for model_id in changed_models:
                    self._mark_model_edited(model_id)
                self._recompile_topology()
                self.nodes()
        except Exception:
            self._root_spec = previous_root
            self._attached_models = previous_models
            self._root_edited = previous_root_edited
            self._geometry_object_ids = previous_object_ids
            self._next_geometry_object_id = previous_next_object_id
            raise

        by_identity = {identity: node_id for node_id, identity in self._node_element.items()}
        return tuple(
            by_identity.get(item, -1) if item is not None else -1 for item in result_identities
        )

    def scene_model_xml(self, model_id: int) -> str | None:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        if model_id == 0:
            return spec.to_xml() if self._root_edited else None
        item = next((item for item in self._attached_models if item.model_id == model_id), None)
        return spec.to_xml() if item is not None and item.edited else None

    def scene_model_source(self, model_id: int) -> str | None:
        spec = self._spec_for_model(model_id)
        return spec.to_xml() if spec is not None else None

    def set_scene_model_xml(self, model_id: int, xml: str) -> bool:
        spec = mujoco.MjSpec.from_string(str(xml))
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            spec.modelfiledir = str(path.parent)
        return self._replace_model_spec(model_id, spec)

    def model_components(self, model_id: int, category: str) -> tuple[ModelComponentInfo, ...]:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            self._component_entries.pop((int(model_id), str(category)), None)
            return ()
        entries = self._sync_model_component_ids(model_id, category, section)
        components = []
        for entry, element in zip(entries, section, strict=True):
            attributes = dict(element.attrib)
            path = tuple(
                ModelComponentPathItem(
                    child.tag,
                    _component_path_fields(root, category, element.tag, child),
                )
                for child in element
            )
            components.append(
                ModelComponentInfo(
                    entry.component_id,
                    int(model_id),
                    category,
                    element.tag,
                    attributes.get("name", f"{category}{entry.component_id}"),
                    _component_fields(root, category, element.tag, attributes),
                    path,
                    _component_path_presets(root, category, element.tag),
                )
            )
        return tuple(components)

    def _sync_model_component_ids(
        self,
        model_id: int,
        category: str,
        section: ET.Element,
        forced: dict[int, int] | None = None,
    ) -> tuple[_ModelComponentEntry, ...]:
        """Reconcile stable runtime IDs without ever retargeting a stale ID."""

        key = (int(model_id), str(category))
        previous = self._component_entries.get(key, ())
        descriptors = tuple(
            (
                element.tag,
                str(element.attrib.get("name", "")).strip(),
                ET.tostring(element, encoding="unicode"),
            )
            for element in section
        )
        if len(previous) == len(descriptors) and all(
            (entry.subtype, entry.name, entry.signature) == descriptor
            for entry, descriptor in zip(previous, descriptors, strict=True)
        ):
            return previous

        forced = forced or {}
        used: set[int] = set()
        reconciled = []
        for index, (subtype, name, signature) in enumerate(descriptors):
            component_id = forced.get(index, -1)
            if component_id < 0 and name:
                match = next(
                    (
                        entry
                        for entry in previous
                        if entry.component_id not in used
                        and entry.name == name
                        and (category == "actuator" or entry.subtype == subtype)
                    ),
                    None,
                )
                component_id = match.component_id if match is not None else -1
            if component_id < 0 and not name:
                matches = tuple(
                    entry
                    for entry in previous
                    if entry.component_id not in used
                    and entry.subtype == subtype
                    and not entry.name
                    and entry.signature == signature
                )
                component_id = matches[0].component_id if len(matches) == 1 else -1
            if component_id < 0:
                component_id = self._next_component_id.get(key, 0)
                self._next_component_id[key] = component_id + 1
            used.add(component_id)
            reconciled.append(_ModelComponentEntry(component_id, subtype, name, signature))
        result = tuple(reconciled)
        if result:
            self._component_entries[key] = result
        else:
            self._component_entries.pop(key, None)
        return result

    def _model_component_index(
        self, model_id: int, category: str, section: ET.Element, component_id: int
    ) -> tuple[int, tuple[_ModelComponentEntry, ...]]:
        entries = self._sync_model_component_ids(model_id, category, section)
        index = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry.component_id == int(component_id)
            ),
            -1,
        )
        return index, entries

    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        joint_elements = tuple(root.iter("joint"))
        joints = _named_elements(root, "joint")
        ball_joints = tuple(
            element.attrib["name"]
            for element in joint_elements
            if element.attrib.get("name") and element.attrib.get("type", "hinge") == "ball"
        )
        limited_joints = tuple(
            element.attrib["name"]
            for element in joint_elements
            if element.attrib.get("name")
            and (
                element.attrib.get("limited") in {"true", "1"} or bool(element.attrib.get("range"))
            )
        )
        bodies = _named_elements(root, "body")
        geoms = _named_elements(root, "geom")
        sites = _named_elements(root, "site")
        tendons = _named_elements(root, "fixed") + _named_elements(root, "spatial")
        tendon_section = root.find("tendon")
        limited_tendons = tuple(
            element.attrib["name"]
            for element in tendon_section or ()
            if element.attrib.get("name")
            and (
                element.attrib.get("limited") in {"true", "1"} or bool(element.attrib.get("range"))
            )
        )
        actuators = _named_elements(root, "actuator")
        cameras = _named_elements(root, "camera")
        meshes = _named_elements(root, "mesh")
        flexes = _named_elements(root, "flex")
        if category == "contact":
            return (
                *(("pair",) if len(geoms) >= 2 else ()),
                *(("exclude",) if len(bodies) >= 2 else ()),
            )
        if category == "actuator":
            return (
                *(
                    (
                        "general",
                        "motor",
                        "position",
                        "velocity",
                        "intvelocity",
                        "damper",
                        "cylinder",
                        "muscle",
                        "dcmotor",
                    )
                    if joints
                    else ()
                ),
                *(("orientation",) if ball_joints else ()),
                *(("adhesion",) if bodies else ()),
            )
        if category == "sensor":
            return (
                *(
                    (
                        "touch",
                        "accelerometer",
                        "velocimeter",
                        "gyro",
                        "force",
                        "torque",
                        "magnetometer",
                        "rangefinder",
                    )
                    if sites
                    else ()
                ),
                *(("camprojection",) if sites and cameras else ()),
                *(("jointpos", "jointvel", "jointactuatorfrc") if joints else ()),
                *(("ballquat", "ballangvel") if ball_joints else ()),
                *(("jointlimitpos", "jointlimitvel", "jointlimitfrc") if limited_joints else ()),
                *(("tendonpos", "tendonvel", "tendonactuatorfrc") if tendons else ()),
                *(
                    ("tendonlimitpos", "tendonlimitvel", "tendonlimitfrc")
                    if limited_tendons
                    else ()
                ),
                *(("actuatorpos", "actuatorvel", "actuatorfrc") if actuators else ()),
                *(
                    (
                        "framepos",
                        "framequat",
                        "framexaxis",
                        "frameyaxis",
                        "framezaxis",
                        "framelinvel",
                        "frameangvel",
                        "framelinacc",
                        "frameangacc",
                        "subtreecom",
                        "subtreelinvel",
                        "subtreeangmom",
                    )
                    if bodies
                    else ()
                ),
                *(("insidesite",) if sites and bodies else ()),
                *(("distance", "normal", "fromto", "contact") if len(bodies) >= 2 else ()),
                *(("tactile",) if meshes else ()),
                "e_potential",
                "e_kinetic",
                "clock",
                "user",
            )
        if category == "tendon":
            return (
                *(("fixed",) if joints else ()),
                *(("spatial",) if len(sites) >= 2 else ()),
            )
        if category == "equality":
            return (
                *(("joint",) if joints else ()),
                *(("weld", "connect") if bodies else ()),
                *(("tendon",) if tendons else ()),
                *(("flex", "flexvert", "flexstrain") if flexes else ()),
            )
        if category == "custom":
            return ("numeric", "text", "tuple")
        _component_section(root, category)
        return ()

    def add_model_component(self, model_id: int, category: str, subtype: str, name: str) -> int:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return -1
        presets = self.model_component_presets(model_id, category)
        if subtype not in presets:
            raise ValueError(f"Cannot add {category} subtype {subtype!r} to this model")
        root, _xml = _component_xml(spec)
        section = _component_section(root, category, create=True)
        assert section is not None
        previous_entries = self._sync_model_component_ids(model_id, category, section)
        if any(
            item.attrib.get("name") == value
            and (category not in {"contact", "custom"} or item.tag == subtype)
            for item in section
        ):
            raise ValueError(f"{category} {value!r} already exists")
        element = ET.SubElement(section, subtype, {"name": value})
        joint_elements = tuple(root.iter("joint"))
        joints = _named_elements(root, "joint")
        ball_joints = tuple(
            element.attrib["name"]
            for element in joint_elements
            if element.attrib.get("name") and element.attrib.get("type", "hinge") == "ball"
        )
        limited_joints = tuple(
            element.attrib["name"]
            for element in joint_elements
            if element.attrib.get("name")
            and (
                element.attrib.get("limited") in {"true", "1"} or bool(element.attrib.get("range"))
            )
        )
        bodies = _named_elements(root, "body")
        geoms = _named_elements(root, "geom")
        sites = _named_elements(root, "site")
        tendons = _named_elements(root, "fixed") + _named_elements(root, "spatial")
        tendon_section = root.find("tendon")
        limited_tendons = tuple(
            item.attrib["name"]
            for item in tendon_section or ()
            if item.attrib.get("name")
            and (item.attrib.get("limited") in {"true", "1"} or bool(item.attrib.get("range")))
        )
        actuators = _named_elements(root, "actuator")
        cameras = _named_elements(root, "camera")
        meshes = _named_elements(root, "mesh")
        flexes = _named_elements(root, "flex")
        if category == "contact" and subtype == "pair":
            element.set("geom1", geoms[0])
            element.set("geom2", geoms[1])
        elif category == "contact":
            element.set("body1", bodies[0])
            element.set("body2", bodies[1])
        elif category == "actuator":
            if subtype == "adhesion":
                element.set("body", bodies[0])
                element.set("ctrlrange", "0 1")
            elif subtype == "orientation":
                element.set("joint", ball_joints[0])
            else:
                element.set("joint", joints[0])
                if subtype == "damper":
                    element.set("ctrlrange", "0 1")
                elif subtype == "muscle":
                    element.set("lengthrange", "0 1")
                elif subtype == "dcmotor":
                    element.set("motorconst", "1")
                    element.set("resistance", "1")
                    element.set("inductance", "1")
        elif category == "sensor":
            site_sensors = {
                "touch",
                "accelerometer",
                "velocimeter",
                "gyro",
                "force",
                "torque",
                "magnetometer",
                "rangefinder",
                "camprojection",
            }
            if subtype in site_sensors:
                element.set("site", sites[0])
                if subtype == "camprojection":
                    element.set("camera", cameras[0])
            elif subtype in {"ballquat", "ballangvel"}:
                element.set("joint", ball_joints[0])
            elif subtype.startswith("jointlimit"):
                element.set("joint", limited_joints[0])
            elif subtype.startswith("joint"):
                element.set("joint", joints[0])
            elif subtype.startswith("tendonlimit"):
                element.set("tendon", limited_tendons[0])
            elif subtype.startswith("tendon"):
                element.set("tendon", tendons[0])
            elif subtype.startswith("actuator"):
                element.set("actuator", actuators[0])
            elif subtype in {"subtreecom", "subtreelinvel", "subtreeangmom"}:
                element.set("body", bodies[0])
            elif subtype == "insidesite":
                element.set("site", sites[0])
                element.set("objtype", "body")
                element.set("objname", bodies[0])
            elif subtype in {"distance", "normal", "fromto", "contact"}:
                element.set("body1", bodies[0])
                element.set("body2", bodies[1])
            elif subtype == "tactile":
                element.set("mesh", meshes[0])
            elif subtype == "user":
                element.set("dim", "1")
                element.set("needstage", "pos")
            elif subtype.startswith("frame"):
                element.set("objtype", "body")
                element.set("objname", bodies[0])
            else:
                pass
        elif category == "tendon" and subtype == "fixed":
            ET.SubElement(element, "joint", {"joint": joints[0], "coef": "1"})
        elif category == "tendon":
            ET.SubElement(element, "site", {"site": sites[0]})
            ET.SubElement(element, "site", {"site": sites[1]})
        elif category == "equality" and subtype == "joint":
            element.set("joint1", joints[0])
        elif category == "equality" and subtype == "connect":
            if len(sites) >= 2:
                element.set("site1", sites[0])
                element.set("site2", sites[1])
            else:
                element.set("body1", bodies[0])
                element.set("anchor", "0 0 0")
        elif category == "equality" and subtype == "weld":
            element.set("body1", bodies[0])
        elif category == "equality" and subtype == "tendon":
            element.set("tendon1", tendons[0])
        elif category == "equality":
            element.set("flex", flexes[0])
            if subtype == "flexstrain":
                element.set("cell", "0")
        elif category == "custom" and subtype == "numeric":
            element.set("size", "1")
            element.set("data", "0")
        elif category == "custom" and subtype == "text":
            element.set("data", "text")
        elif category == "custom":
            ET.SubElement(element, "element", {"objtype": "body", "objname": "world"})
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        if not self._replace_model_spec(model_id, new_spec):
            return -1
        entries = self._sync_model_component_ids(
            model_id,
            category,
            section,
            {index: entry.component_id for index, entry in enumerate(previous_entries)},
        )
        return entries[-1].component_id

    def update_model_component(
        self,
        model_id: int,
        category: str,
        component_id: int,
        name: str,
        fields: tuple[tuple[str, str], ...],
        path: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return False
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            return False
        component_index, _entries = self._model_component_index(
            model_id, category, section, component_id
        )
        if component_index < 0:
            return False
        if any(
            index != component_index and item.attrib.get("name") == value
            for index, item in enumerate(section)
            if category not in {"contact", "custom"} or item.tag == section[component_index].tag
        ):
            raise ValueError(f"{category} {value!r} already exists")
        element = section[component_index]
        next_attributes = {
            str(field_name).strip(): str(field_value).strip()
            for field_name, field_value in fields
            if str(field_name).strip()
            and str(field_name).strip() != "name"
            and str(field_value).strip()
        }
        next_attributes = {"name": value, **next_attributes}
        next_path = tuple(
            (
                str(item_kind).strip(),
                tuple(
                    (str(field_name).strip(), str(field_value).strip())
                    for field_name, field_value in item_fields
                    if str(field_name).strip() and str(field_value).strip()
                ),
            )
            for item_kind, item_fields in path
            if str(item_kind).strip()
        )
        current_path = tuple((child.tag, tuple(child.attrib.items())) for child in element)
        path_category = category in {"tendon", "custom"}
        if dict(element.attrib) == next_attributes and (
            not path_category or current_path == next_path
        ):
            return True
        element.attrib.clear()
        element.attrib.update(next_attributes)
        if path_category:
            element[:] = []
            for item_kind, item_fields in next_path:
                ET.SubElement(element, item_kind, dict(item_fields))
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        changed = self._replace_model_spec(model_id, new_spec)
        if changed:
            self._sync_model_component_ids(
                model_id, category, section, {component_index: int(component_id)}
            )
        return changed

    def remove_model_component(self, model_id: int, category: str, component_id: int) -> bool:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return False
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            return False
        component_index, entries = self._model_component_index(
            model_id, category, section, component_id
        )
        if component_index < 0:
            return False
        section.remove(section[component_index])
        if not len(section):
            root.remove(section)
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        changed = self._replace_model_spec(model_id, new_spec)
        if changed:
            remaining = tuple(
                entry for index, entry in enumerate(entries) if index != component_index
            )
            self._sync_model_component_ids(
                model_id,
                category,
                section,
                {index: entry.component_id for index, entry in enumerate(remaining)},
            )
        return changed

    def model_property_groups(self, model_id: int) -> tuple[ModelPropertyGroup, ...]:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        groups: list[ModelPropertyGroup] = []
        for group_id, label, path in _GLOBAL_PROPERTY_GROUPS:
            element = root.find("/".join(path))
            attributes = dict(element.attrib) if element is not None else {}
            schema = _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", *path), ())
            names = tuple(dict.fromkeys((*schema, *attributes)))
            groups.append(
                ModelPropertyGroup(
                    int(model_id),
                    group_id,
                    "global",
                    label,
                    _model_property_fields(root, group_id, names, attributes),
                )
            )

        asset = root.find("asset")
        for asset_type, schema in _ASSET_PROPERTY_FIELDS.items():
            elements = tuple(asset.findall(asset_type)) if asset is not None else ()
            for index, element in enumerate(elements):
                attributes = dict(element.attrib)
                group_id = f"asset:{asset_type}:{index}"
                names = schema
                name = attributes.get("name") or attributes.get("file") or str(index)
                groups.append(
                    ModelPropertyGroup(
                        int(model_id),
                        group_id,
                        "asset",
                        f"{asset_type}: {name}",
                        _model_property_fields(root, group_id, names, attributes),
                    )
                )

        defaults = tuple(root.iter("default"))
        for default_index, default in enumerate(defaults):
            class_name = default.attrib.get("class") or "main"
            class_group = f"default:{default_index}:class"
            groups.append(
                ModelPropertyGroup(
                    int(model_id),
                    class_group,
                    "default",
                    f"Default {class_name} / class",
                    (ModelComponentField("class", default.attrib.get("class", "")),),
                )
            )
            for element_type in _DEFAULT_PROPERTY_TYPES:
                element = default.find(element_type)
                attributes = dict(element.attrib) if element is not None else {}
                group_id = f"default:{default_index}:{element_type}"
                schema = _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "default", element_type), ())
                names = tuple(dict.fromkeys((*schema, *attributes)))
                groups.append(
                    ModelPropertyGroup(
                        int(model_id),
                        group_id,
                        "default",
                        f"Default {class_name} / {element_type}",
                        _model_property_fields(root, group_id, names, attributes),
                    )
                )
        return tuple(groups)

    def model_assets(self, model_id: int) -> tuple[ModelAssetInfo, ...]:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        asset = root.find("asset")
        if asset is None:
            return ()
        material_indices: dict[str, int] = {}
        for material_id in range(self._m.nmat):
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
            owner, local_name = self._model_element_name(
                compiled_name or f"material{material_id}", mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if owner == int(model_id):
                material_indices[local_name] = material_id
        height_field_previews: dict[
            str,
            tuple[
                tuple[int, int],
                tuple[int, int],
                tuple[float, ...],
                tuple[float, float],
            ],
        ] = {}
        for field_id in range(self._m.nhfield):
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_HFIELD, field_id)
            owner, local_name = self._model_element_name(
                compiled_name or f"hfield{field_id}", mujoco.mjtObj.mjOBJ_HFIELD
            )
            if owner != int(model_id):
                continue
            rows = int(self._m.hfield_nrow[field_id])
            columns = int(self._m.hfield_ncol[field_id])
            address = int(self._m.hfield_adr[field_id])
            data = np.asarray(
                self._m.hfield_data[address : address + rows * columns], np.float32
            ).reshape(rows, columns)
            preview_rows = min(rows, 48)
            preview_columns = min(columns, 48)
            row_indices = np.rint(np.linspace(0, rows - 1, preview_rows)).astype(np.intp)
            column_indices = np.rint(np.linspace(0, columns - 1, preview_columns)).astype(np.intp)
            preview = data[np.ix_(row_indices, column_indices)]
            height_field_previews[local_name] = (
                (rows, columns),
                (preview_rows, preview_columns),
                tuple(float(value) for value in preview.reshape(-1)),
                (float(np.min(data)), float(np.max(data))),
            )
        items: list[ModelAssetInfo] = []
        for asset_type in _MODEL_ASSET_TYPES:
            for index, element in enumerate(asset.findall(asset_type)):
                attributes = dict(element.attrib)
                name = str(attributes.get("name", "")).strip()
                if not name:
                    continue
                group_id = f"asset:{asset_type}:{index}"
                field_names = _ASSET_PROPERTY_FIELDS.get(asset_type)
                fields = (
                    _model_property_fields(root, group_id, field_names, attributes)
                    if field_names is not None
                    else tuple(
                        ModelComponentField(field, value)
                        for field, value in attributes.items()
                        if field not in {"name", "file"}
                    )
                )
                if asset_type == "hfield" and "elevation" in attributes:
                    fields = (
                        *fields,
                        ModelComponentField("elevation", attributes["elevation"]),
                    )
                preview = height_field_previews.get(name, ((0, 0), (0, 0), (), (0.0, 0.0)))
                texture_layers: tuple[tuple[str, str], ...] = ()
                if asset_type == "material":
                    layer_map = {
                        str(layer.attrib.get("role", "rgb")): str(layer.attrib.get("texture", ""))
                        for layer in element.findall("layer")
                        if str(layer.attrib.get("texture", ""))
                    }
                    legacy_texture = str(element.attrib.get("texture", ""))
                    if legacy_texture and "rgb" not in layer_map:
                        layer_map["rgb"] = legacy_texture
                    texture_layers = tuple(
                        (role, layer_map[role])
                        for role in MATERIAL_TEXTURE_ROLES
                        if role in layer_map
                    )
                items.append(
                    ModelAssetInfo(
                        model_id=int(model_id),
                        type=asset_type,
                        name=name,
                        index=index,
                        file=str(attributes.get("file", "")),
                        fields=fields,
                        references=_model_asset_references(root, asset_type, name, element),
                        data_shape=preview[0],
                        preview_shape=preview[1],
                        preview_values=preview[2],
                        preview_range=preview[3],
                        runtime_index=(
                            material_indices.get(name, -1) if asset_type == "material" else -1
                        ),
                        texture_layers=texture_layers,
                    )
                )
        return tuple(items)

    def import_model_asset(
        self,
        model_id: int,
        asset_type: str,
        path: Path,
        name: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        spec = self._spec_for_model(model_id)
        source = Path(path).expanduser().resolve()
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        if spec is None or not source.is_file() or kind not in ("mesh", "hfield") or not value:
            return False
        root, _xml = _component_xml(spec)
        asset, existing = _model_asset_element(root, kind, value)
        if existing is not None:
            return False
        if asset is None:
            asset = _ensure_model_asset_section(root)
        allowed = set(_MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", kind), ()))
        attributes = {"name": value, "file": str(source)}
        for raw_name, raw_value in fields:
            field = str(raw_name).strip()
            field_value = str(raw_value).strip()
            if not field or field in {"name", "file"} or field not in allowed:
                raise ValueError(f"Unknown {kind} asset field {field!r}")
            if field_value:
                attributes[field] = field_value
        ET.SubElement(asset, kind, attributes)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def create_height_field(
        self,
        model_id: int,
        name: str,
        rows: int,
        columns: int,
        size: tuple[float, float, float, float],
        elevation: tuple[float, ...],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        try:
            samples = np.asarray(elevation, np.float64).reshape(-1)
            dimensions = np.asarray(size, np.float64).reshape(4)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            spec is None
            or not value
            or int(rows) < 2
            or int(columns) < 2
            or samples.size != int(rows) * int(columns)
            or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions[:3] <= 0.0)
            or dimensions[3] < 0.0
        ):
            return False
        root, _xml = _component_xml(spec)
        _asset, existing = _model_asset_element(root, "hfield", value)
        if existing is not None:
            return False
        asset = _ensure_model_asset_section(root)
        ET.SubElement(
            asset,
            "hfield",
            {
                "name": value,
                "nrow": str(int(rows)),
                "ncol": str(int(columns)),
                "size": _format_mjcf_values(dimensions),
                "elevation": _format_mjcf_values(samples),
            },
        )
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def set_height_field_data(
        self,
        model_id: int,
        name: str,
        rows: int,
        columns: int,
        size: tuple[float, float, float, float],
        elevation: tuple[float, ...],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        try:
            samples = np.asarray(elevation, np.float64).reshape(-1)
            dimensions = np.asarray(size, np.float64).reshape(4)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            spec is None
            or not value
            or int(rows) < 2
            or int(columns) < 2
            or samples.size != int(rows) * int(columns)
            or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions[:3] <= 0.0)
            or dimensions[3] < 0.0
        ):
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, "hfield", value)
        if target is None or str(target.attrib.get("file", "")).strip():
            return False
        target.set("nrow", str(int(rows)))
        target.set("ncol", str(int(columns)))
        target.set("size", _format_mjcf_values(dimensions))
        target.set("elevation", _format_mjcf_values(samples))
        target.attrib.pop("content_type", None)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def rename_model_asset(self, model_id: int, asset_type: str, name: str, new_name: str) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        before_name = str(name).strip()
        after_name = str(new_name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not before_name or not after_name:
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, kind, before_name)
        _asset, collision = _model_asset_element(root, kind, after_name)
        if target is None or collision is not None:
            return False
        target.set("name", after_name)
        for element in root.iter():
            if element is target:
                continue
            for field in _MODEL_ASSET_REFERENCE_FIELDS.get(kind, ()):
                if element.attrib.get(field) == before_name:
                    element.set(field, after_name)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def duplicate_model_asset(
        self, model_id: int, asset_type: str, name: str, new_name: str
    ) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        source_name = str(name).strip()
        duplicate_name = str(new_name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not source_name or not duplicate_name:
            return False
        root, _xml = _component_xml(spec)
        asset, target = _model_asset_element(root, kind, source_name)
        _asset, collision = _model_asset_element(root, kind, duplicate_name)
        if asset is None or target is None or collision is not None:
            return False
        duplicate = deepcopy(target)
        duplicate.set("name", duplicate_name)
        children = tuple(asset)
        asset.insert(children.index(target) + 1, duplicate)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def replace_model_asset_file(
        self, model_id: int, asset_type: str, name: str, path: Path
    ) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        source = Path(path).expanduser().resolve()
        if (
            spec is None
            or kind not in ("mesh", "hfield", "texture")
            or not value
            or not source.is_file()
        ):
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, kind, value)
        if target is None:
            return False
        target.set("file", str(source))
        target.attrib.pop("content_type", None)
        if kind == "hfield":
            target.attrib.pop("nrow", None)
            target.attrib.pop("ncol", None)
            target.attrib.pop("elevation", None)
        elif kind == "mesh":
            for field in ("vertex", "normal", "texcoord", "face", "builtin", "params"):
                target.attrib.pop(field, None)
        else:
            for field in (
                "builtin",
                "width",
                "height",
                "rgb1",
                "rgb2",
                "mark",
                "markrgb",
                "random",
            ):
                target.attrib.pop(field, None)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def remove_model_asset(self, model_id: int, asset_type: str, name: str) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not value:
            return False
        root, _xml = _component_xml(spec)
        asset, target = _model_asset_element(root, kind, value)
        if asset is None or target is None:
            return False
        if _model_asset_references(root, kind, value, target):
            return False
        asset.remove(target)
        if not len(asset):
            root.remove(asset)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def set_model_property_groups(
        self,
        model_id: int,
        updates: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        if spec is None or not updates:
            return False
        root, _xml = _component_xml(spec)
        before = _serialize_component_xml(root)
        global_paths = {group_id: path for group_id, _label, path in _GLOBAL_PROPERTY_GROUPS}

        for raw_group_id, raw_fields in updates:
            group_id = str(raw_group_id)
            fields = tuple((str(name).strip(), str(value).strip()) for name, value in raw_fields)
            if (
                not fields
                or any(not name for name, _value in fields)
                or len({name for name, _value in fields}) != len(fields)
            ):
                raise ValueError(f"Invalid fields for model property group {group_id!r}")

            element = None
            allowed: tuple[str, ...] = ()
            if group_id in global_paths:
                path = global_paths[group_id]
                allowed = _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", *path), ())
                element = root.find("/".join(path))
                if element is None and any(value for _name, value in fields):
                    element = _find_or_create_xml_path(root, path)
            elif group_id.startswith("asset:"):
                _prefix, asset_type, index_text = group_id.split(":", 2)
                asset = root.find("asset")
                elements = tuple(asset.findall(asset_type)) if asset is not None else ()
                index = int(index_text)
                if asset_type not in _ASSET_PROPERTY_FIELDS or not 0 <= index < len(elements):
                    raise ValueError(f"Unknown model asset property group {group_id!r}")
                element = elements[index]
                allowed = _ASSET_PROPERTY_FIELDS[asset_type]
            elif group_id.startswith("default:"):
                _prefix, index_text, element_type = group_id.split(":", 2)
                defaults = tuple(root.iter("default"))
                index = int(index_text)
                if not 0 <= index < len(defaults):
                    raise ValueError(f"Unknown default property group {group_id!r}")
                default = defaults[index]
                if element_type == "class":
                    element = default
                    allowed = ("class",)
                elif element_type in _DEFAULT_PROPERTY_TYPES:
                    allowed = _MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "default", element_type), ())
                    element = default.find(element_type)
                    if element is None and any(value for _name, value in fields):
                        element = ET.SubElement(default, element_type)
                else:
                    raise ValueError(f"Unknown default property group {group_id!r}")
            else:
                raise ValueError(f"Unknown model property group {group_id!r}")

            unknown = {name for name, _value in fields}.difference(allowed)
            if unknown:
                raise ValueError(f"Unknown fields for {group_id}: {', '.join(sorted(unknown))}")
            if element is None:
                continue
            for name, value in fields:
                if value:
                    element.set(name, value)
                else:
                    element.attrib.pop(name, None)
            if group_id.startswith("default:") and not element.attrib and not len(element):
                _prefix, index_text, element_type = group_id.split(":", 2)
                if element_type != "class":
                    tuple(root.iter("default"))[int(index_text)].remove(element)

        after = _serialize_component_xml(root)
        if after == before:
            return True
        new_spec = self._spec_from_component_xml(model_id, after)
        return self._replace_model_spec(model_id, new_spec)

    def add_model_default(self, model_id: int, parent_default_id: int, name: str) -> int:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return -1
        root, _xml = _component_xml(spec)
        if any(item.attrib.get("class") == value for item in root.iter("default")):
            raise ValueError(f"Default class {value!r} already exists")
        defaults = tuple(root.iter("default"))
        if not defaults:
            main = ET.Element("default")
            insertion = next(
                (
                    index
                    for index, child in enumerate(root)
                    if child.tag in {"asset", "contact", "worldbody"}
                ),
                len(root),
            )
            root.insert(insertion, main)
            defaults = (main,)
        parent_index = int(parent_default_id)
        if parent_index < 0:
            parent_index = 0
        if not 0 <= parent_index < len(defaults):
            return -1
        added = ET.SubElement(defaults[parent_index], "default", {"class": value})
        default_id = tuple(root.iter("default")).index(added)
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return default_id if self._replace_model_spec(model_id, new_spec) else -1

    def remove_model_default(self, model_id: int, default_id: int) -> bool:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return False
        root, _xml = _component_xml(spec)
        defaults = tuple(root.iter("default"))
        index = int(default_id)
        if not 0 <= index < len(defaults):
            return False
        target = defaults[index]
        if not target.attrib.get("class"):
            return False
        parent = next((item for item in root.iter() if target in tuple(item)), None)
        if parent is None:
            return False
        parent.remove(target)
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, new_spec)

    def _spec_from_component_xml(self, model_id: int, xml: str):
        spec = mujoco.MjSpec.from_string(xml)
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            spec.modelfiledir = str(path.parent)
        return spec

    def _replace_model_spec(self, model_id: int, spec) -> bool:
        # MjSpec.attach can defer a broken local reference until later serialization.
        # Validate the edited standalone model before it enters the composed document.
        spec.to_xml()
        state = self._capture_named_model_state()
        if int(model_id) == 0:
            if self._root_spec is None:
                return False
            previous_spec = self._root_spec
            previous_edited = self._root_edited
            self._root_spec = spec
            self._root_edited = True
            try:
                model = self._compile_composed_model()
            except Exception:
                self._root_spec = previous_spec
                self._root_edited = previous_edited
                raise
        else:
            index = next(
                (
                    index
                    for index, item in enumerate(self._attached_models)
                    if item.model_id == int(model_id)
                ),
                -1,
            )
            if index < 0:
                return False
            previous = self._attached_models[index]
            self._attached_models[index] = replace(previous, spec=spec, edited=True)
            try:
                model = self._compile_composed_model()
            except Exception:
                self._attached_models[index] = previous
                raise
        self._install(model)
        self._restore_named_model_state(state)
        return True

    def _recompile_topology(self) -> None:
        state = self._capture_named_model_state()
        self._install(self._compile_composed_model())
        self._restore_named_model_state(state)

    def _spec_for_model(self, model_id: int):
        if int(model_id) == 0:
            return self._root_spec
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        return item.spec if item is not None else None

    def _mark_model_edited(self, model_id: int) -> None:
        if int(model_id) == 0:
            self._root_edited = True
            return
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index >= 0:
            self._attached_models[index] = replace(self._attached_models[index], edited=True)

    def _store_model_spec(self, model_id: int, spec) -> None:
        """Replace one stored editable spec without recompiling the composed model."""
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            # MjSpec.from_string() drops modelfiledir.  Material texture edits
            # reparse the spec below, so restore the model's resource root before
            # a later edit, export, or topology rebuild resolves relative meshes.
            spec.modelfiledir = str(path.parent)
        if int(model_id) == 0:
            self._root_spec = spec
            self._root_edited = True
            return
        index = next(
            (
                index
                for index, item in enumerate(self._attached_models)
                if item.model_id == int(model_id)
            ),
            -1,
        )
        if index >= 0:
            self._attached_models[index] = replace(
                self._attached_models[index], spec=spec, edited=True
            )

    def _element(self, model_id: int, element_type: str, name: str):
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        lookup = "body" if element_type in ("link", "robot") else element_type
        finder = getattr(spec, lookup, None)
        element = finder(name) if finder is not None else None
        if element is not None:
            return element
        # MuJoCo's name lookup can lag behind an in-place MjSpec rename until
        # the corresponding collection is materialized. Batch edits must still
        # be able to address that element by its new semantic identity.
        collection = getattr(
            spec,
            {
                "body": "bodies",
                "geom": "geoms",
                "joint": "joints",
                "site": "sites",
                "camera": "cameras",
                "light": "lights",
            }.get(lookup, ""),
            (),
        )
        return next((item for item in collection if item.name == name), None)

    def _reset_geometry_object_ids(self) -> None:
        self._geometry_object_ids.clear()
        self._next_geometry_object_id = GEOMETRY_OBJECT_BASE

    def _geometry_object_id(self, model_id: int, name: str) -> int:
        identity = (int(model_id), str(name))
        object_id = self._geometry_object_ids.get(identity)
        if object_id is None:
            object_id = self._next_geometry_object_id
            self._next_geometry_object_id += 1
            self._geometry_object_ids[identity] = object_id
        return object_id

    def _node_for_id(self, node_id: int) -> SceneNode | None:
        nodes = self.nodes()
        index = int(node_id)
        if 0 <= index < len(nodes) and nodes[index].node_id == index:
            return nodes[index]
        return None

    def _model_parent(self, node_id: int):
        node = self._node_for_id(node_id)
        while node is not None:
            identity = self._node_element.get(node.node_id)
            if identity is not None:
                model_id, node_type, name = identity
                spec = self._spec_for_model(model_id)
                if spec is None:
                    return None
                if node_type in (NodeType.MODEL, NodeType.WORLD):
                    return model_id, spec.worldbody
                if node_type in (NodeType.LINK, NodeType.ROBOT):
                    return model_id, spec.body(name)
            node = self._node_for_id(node.parent)
        return None

    def _composed_spec(self):
        if self._root_spec is None:
            raise RuntimeError("Model composition is unavailable")
        spec = self._root_spec.copy()
        if self._root_path is not None:
            self._resolve_asset_paths(spec, self._root_path.parent)
        for index, item in enumerate(self._attached_models):
            child = item.spec.copy()
            self._resolve_asset_paths(child, item.path.parent)
            # Resolve keyframes inherited from nested model assets before the child is
            # attached again. MuJoCo can then namespace their compiled state correctly.
            child_model = self._resolve_attached_keyframes(child)
            if child_model is not None:
                self._transform_attached_keyframes(
                    child,
                    child_model,
                    item.position,
                    item.rotation,
                )
            if index == 0 and self._root_path is None:
                spec.option = child.option
                spec.visual = child.visual
                spec.stat = child.stat
                spec.compiler = child.compiler
                spec.memory = child.memory
            child.option = spec.option
            frame = spec.worldbody.add_frame(name=f"forge_model_{item.model_id}")
            frame.pos = item.position
            frame.quat = math3d.mat3_to_quat(item.rotation)
            self._copy_world_attached_flexes(
                spec,
                child,
                item.prefix,
                item.position,
                item.rotation,
            )
            spec.attach(child, prefix=item.prefix, frame=frame)
            self._restore_attached_world_targets(spec, item.prefix)
        return spec

    @staticmethod
    def _resolve_attached_keyframes(spec):
        """Compile an attached spec early only when unresolved keyframes require it."""
        if len(spec.keys):
            return spec.compile()
        return None

    @staticmethod
    def _transform_attached_keyframes(spec, model, position, rotation) -> None:
        """Transform model-local free and mocap poses before MjSpec attachment.

        MuJoCo namespaces attached keyframes but leaves their pose arrays unchanged,
        even though those arrays become world-space in the composed model.
        """
        offset = np.asarray(position, np.float64).reshape(3)
        matrix = np.asarray(rotation, np.float64).reshape(3, 3)
        for key_index, key in enumerate(spec.keys):
            qpos = np.asarray(model.key_qpos[key_index], np.float64).copy()
            for joint in range(model.njnt):
                if int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                address = int(model.jnt_qposadr[joint])
                qpos[address : address + 3] = qpos[address : address + 3] @ matrix.T + offset
                qpos[address + 3 : address + 7] = math3d.mat3_to_quat(
                    matrix @ math3d.quat_to_mat3(qpos[address + 3 : address + 7])
                )
            key.qpos = qpos
            if model.nmocap:
                mocap_position = np.asarray(model.key_mpos[key_index], np.float64).reshape(-1, 3)
                mocap_quaternion = np.asarray(model.key_mquat[key_index], np.float64).reshape(-1, 4)
                key.mpos = (mocap_position @ matrix.T + offset).reshape(-1)
                key.mquat = np.asarray(
                    [
                        math3d.mat3_to_quat(matrix @ math3d.quat_to_mat3(quaternion))
                        for quaternion in mocap_quaternion
                    ]
                ).reshape(-1)

    def _compile_composed_model(self):
        spec = self._composed_spec()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attach conflict.*")
            return spec.compile()

    @staticmethod
    def _resolve_asset_paths(spec, directory: Path) -> None:
        def asset_base(asset, compiler_field: str) -> Path:
            compiler = getattr(asset, "compiler", spec.compiler)
            relative = str(getattr(compiler, compiler_field, ""))
            return directory / relative if relative else directory

        for assets, compiler_field in (
            (spec.meshes, "meshdir"),
            (spec.textures, "texturedir"),
            (spec.hfields, ""),
            (spec.skins, "meshdir"),
        ):
            for asset in assets:
                file = str(asset.file)
                if file and not Path(file).is_absolute():
                    base = asset_base(asset, compiler_field) if compiler_field else directory
                    asset.file = str((base / file).resolve())
        spec.compiler.meshdir = ""
        spec.compiler.texturedir = ""

    @staticmethod
    def _restore_attached_world_targets(spec, prefix: str) -> None:
        """Restore camera and light targets that refer to MuJoCo's world body.

        MjSpec attachment namespaces every explicit target body. The world body is
        shared by the composed model and retains the reserved name ``world``.
        """

        namespaced_world = f"{prefix}world"
        for camera in spec.cameras:
            if camera.targetbody == namespaced_world:
                camera.targetbody = "world"
        for light in spec.lights:
            if light.targetbody == namespaced_world:
                light.targetbody = "world"

    @staticmethod
    def _copy_world_attached_flexes(spec, child, prefix: str, position, rotation) -> None:
        """Copy world-referencing flexes that MuJoCo omits during attachment.

        Flex vertices owned by the world body require the model-root transform.
        Body-owned vertices remain in body-local coordinates and move with the
        attached frame.
        """

        def namespaced_body(name: str) -> str:
            if not name or name == "world":
                return name
            return f"{prefix}{name}"

        def transformed_points(values, owners) -> list[float]:
            points = np.asarray(values, np.float64).reshape(-1, 3).copy()
            if not len(points):
                return []
            if owners:
                world_owned = np.asarray(
                    [not name or name == "world" for name in owners],
                    dtype=bool,
                )
            else:
                world_owned = np.ones(len(points), dtype=bool)
            points[world_owned] = points[world_owned] @ np.asarray(rotation).T + position
            return points.reshape(-1).tolist()

        for flex in child.flexes:
            references_world = (
                any(name == "world" for name in flex.nodebody)
                or any(name == "world" for name in flex.vertbody)
                or (len(flex.node) > 0 and not flex.nodebody)
                or (len(flex.vert) > 0 and not flex.vertbody)
            )
            if not references_world:
                continue
            fields = {
                name: getattr(flex, name).tolist()
                if hasattr(getattr(flex, name), "tolist")
                else getattr(flex, name)
                for name in _FLEX_COPY_FIELDS
            }
            fields.update(
                name=f"{prefix}{flex.name}" if flex.name else None,
                material=f"{prefix}{flex.material}" if flex.material else None,
                nodebody=[namespaced_body(name) for name in flex.nodebody],
                vertbody=[namespaced_body(name) for name in flex.vertbody],
                node=transformed_points(flex.node, flex.nodebody),
                vert=transformed_points(flex.vert, flex.vertbody),
            )
            spec.add_flex(**fields)

    def current_pose_modified(self) -> bool:
        if self._m is None or self._d is None:
            return False
        return not np.allclose(self._d.qpos, self._m.qpos0, rtol=1e-6, atol=1e-7)

    def export_mjcf(
        self,
        path: Path,
        source: SceneSource,
        frame: SceneFrame,
        options: SceneSaveOptions | None = None,
    ) -> Path:
        """Write the composed MuJoCo model and Forge-authored entities as MJCF."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._validate_authored_mjcf(source)
        spec = self._composed_spec()
        self._append_authored_scene(spec, target, source, frame)
        key_name = (options or SceneSaveOptions()).current_pose_keyframe
        if key_name:
            previous = spec.key(key_name)
            if previous is not None:
                spec.delete(previous)
            state = self.capture_state()
            spec.add_key(
                name=key_name,
                qpos=state.qpos,
                qvel=np.zeros_like(state.qvel),
                act=state.act,
                mpos=state.mocap_pos.reshape(-1),
                mquat=state.mocap_quat.reshape(-1),
                ctrl=state.ctrl,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attach conflict.*")
            preview = spec.compile()
            _apply_environment(spec, source.lights, float(preview.stat.extent))
            spec.compile()
        assets = self._stage_mjcf_assets(spec, target)
        xml = spec.to_xml()
        for source_file, exported_file in assets:
            xml = xml.replace(
                f'="{escape(source_file, quote=True)}"',
                f'="{escape(exported_file, quote=True)}"',
            )
        target.write_text(xml.rstrip() + "\n", encoding="utf-8")
        mujoco.MjModel.from_xml_path(str(target))
        return target

    @staticmethod
    def _validate_authored_mjcf(source: SceneSource) -> None:
        unsupported: list[str] = []
        for light in source.lights.lights:
            texture = source.textures.get(light.texture or "")
            if light.type is LightType.IMAGE and (
                texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX)
            ):
                unsupported.append("image light without a cube texture")
        if source.skybox is not None:
            texture = source.textures.get(source.skybox)
            if texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX):
                unsupported.append("skybox without a cube texture")
        if unsupported:
            details = ", ".join(dict.fromkeys(unsupported))
            raise RuntimeError(f"MJCF export cannot preserve {details}")

    @staticmethod
    def _stage_mjcf_assets(spec, target: Path) -> list[tuple[str, str]]:
        assets = target.parent / f"{target.stem}_assets"
        exported: list[tuple[str, str]] = []
        groups = (
            ("mesh", spec.meshes),
            ("texture", spec.textures),
            ("hfield", spec.hfields),
            ("skin", spec.skins),
        )
        for asset_type, items in groups:
            for index, asset in enumerate(items):
                files = [str(asset.file or "")]
                files.extend(str(file or "") for file in getattr(asset, "cubefiles", ()))
                for file_index, file in enumerate(files):
                    if not file:
                        continue
                    source = Path(file).expanduser().resolve()
                    suffix = source.suffix or ".bin"
                    name = _mjcf_name(asset_type, str(asset.name), index)
                    face = f"_{file_index - 1}" if file_index else ""
                    destination = (
                        source
                        if source.parent == assets.resolve()
                        else assets / f"{name}{face}{suffix}"
                    )
                    assets.mkdir(parents=True, exist_ok=True)
                    if source != destination.resolve():
                        shutil.copy2(source, destination)
                    exported.append(
                        (str(source), destination.relative_to(target.parent).as_posix())
                    )
        return exported

    def _append_authored_scene(
        self, spec, target: Path, source: SceneSource, frame: SceneFrame
    ) -> None:
        texture_names = self._export_authored_textures(spec, target, source)
        material_names: list[str] = []
        for index, material in enumerate(source.materials):
            name = _mjcf_name("forge_material", material.name, index)
            textures = (
                ["", texture_names[material.texture]] if material.texture in texture_names else []
            )
            spec.add_material(
                name=name,
                textures=textures,
                texuniform=material.tex_uniform,
                texrepeat=material.tex_repeat,
                emission=material.emission,
                specular=material.specular,
                shininess=material.shininess,
                reflectance=material.reflectance,
                metallic=material.metallic,
                roughness=material.roughness,
                rgba=material.rgba,
            )
            material_names.append(name)

        positions = frame.geom_xpos
        rotations = frame.geom_xmat
        if positions is None or rotations is None:
            positions = np.zeros((source.instance_count, 3), np.float32)
            rotations = np.repeat(np.eye(3, dtype=np.float32)[None], source.instance_count, axis=0)
        nodes = {node.node_id: node for node in source.nodes}
        for index, key in enumerate(source.geom_mesh):
            node = nodes.get(int(source.geom_node[index]))
            owner = nodes.get(node.parent) if node is not None else None
            name = _mjcf_name(
                "forge_object",
                owner.name if owner is not None else (node.name if node is not None else "object"),
                index,
            )
            body = spec.worldbody.add_body(
                name=name,
                pos=positions[index],
                quat=math3d.mat3_to_quat(rotations[index]),
            )
            material = (
                material_names[int(source.geom_material[index])]
                if material_names and 0 <= int(source.geom_material[index]) < len(material_names)
                else ""
            )
            geom = {
                "name": f"{name}_geom",
                "rgba": source.geom_rgba[index],
                "material": material,
                "contype": 0,
                "conaffinity": 0,
            }
            shape = key.shape
            size = np.asarray(source.geom_size[index], np.float64)
            if shape is MeshShape.SPHERE:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=size, **geom)
            elif shape is MeshShape.BOX:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, **geom)
            elif shape is MeshShape.PLANE:
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                    size=(float(size[0]), float(size[1]), max(float(size[2]), 1e-3)),
                    **geom,
                )
            elif shape is MeshShape.CYLINDER and np.isclose(size[0], size[1]):
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(float(size[0]), float(size[2])),
                    **geom,
                )
            else:
                mesh = source.meshes.get(key) or _authored_builtin_mesh(shape)
                if mesh is None:
                    continue
                mesh_name = f"{name}_mesh"
                spec.add_mesh(
                    name=mesh_name,
                    scale=size,
                    uservert=np.asarray(mesh.positions, np.float64).reshape(-1),
                    userface=np.asarray(mesh.indices, np.int32).reshape(-1),
                )
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                    **geom,
                )

        camera_names = {
            node.camera_index: node.name
            for node in source.nodes
            if node.type is NodeType.CAMERA and node.camera_index >= 0
        }
        for index, camera in enumerate(source.cameras):
            name = _mjcf_name(
                "forge_camera",
                camera_names.get(index, "camera"),
                index,
            )
            spec.worldbody.add_camera(
                name=name,
                pos=camera.eye,
                quat=_camera_quaternion(camera),
                mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                proj=(
                    mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
                    if camera.orthographic
                    else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
                ),
                fovy=(camera.ortho_height if camera.orthographic else np.degrees(camera.fov_y)),
                focal_length=camera.focal_length if camera.uses_intrinsics() else None,
                sensor_size=camera.sensor_size if camera.uses_intrinsics() else None,
                principal_length=camera.principal_offset if camera.uses_intrinsics() else None,
            )

        light_names = {
            node.light_index: node.name
            for node in source.nodes
            if node.type is NodeType.LIGHT and node.light_index >= 0
        }
        light_types = {
            LightType.DIRECTIONAL: mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            LightType.POINT: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.SPOT: mujoco.mjtLightType.mjLIGHT_SPOT,
            # MuJoCo has no area-light enum.  Forge records the semantic type in
            # custom metadata and uses the native point-light bulb radius as the
            # portable fallback representation.
            LightType.AREA: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.IMAGE: mujoco.mjtLightType.mjLIGHT_IMAGE,
        }
        area_names = set(_spec_text_names(spec, _FORGE_AREA_LIGHTS_TEXT))
        for index, light in enumerate(source.lights.lights):
            name = _mjcf_name(
                "forge_light",
                light_names.get(index, "light"),
                index,
            )
            spec.worldbody.add_light(
                name=name,
                pos=light.position,
                dir=light.direction,
                mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                active=light.active,
                type=light_types[light.type],
                texture=texture_names.get(light.texture or "", ""),
                castshadow=light.cast_shadow,
                bulbradius=light.area_radius,
                intensity=light.intensity,
                range=light.range,
                attenuation=light.attenuation,
                cutoff=light.cutoff,
                exponent=light.exponent,
                ambient=light.ambient,
                diffuse=light.diffuse,
                specular=light.specular,
            )
            if light.type is LightType.AREA:
                area_names.add(name)
        _set_text_names(spec, _FORGE_AREA_LIGHTS_TEXT, area_names)

    @staticmethod
    def _export_authored_textures(spec, target: Path, source: SceneSource) -> dict[str, str]:
        from PIL import Image

        texture_names: dict[str, str] = {}
        assets = target.parent / f"{target.stem}_assets"
        for index, texture in enumerate(source.textures.values()):
            assets.mkdir(parents=True, exist_ok=True)
            name = _mjcf_name("forge_texture", texture.name, index)
            if texture.type is TextureType.TWO_D:
                file = assets / f"{name}.png"
                Image.fromarray(texture.pixels).save(file)
                spec.add_texture(
                    name=name,
                    type=mujoco.mjtTexture.mjTEXTURE_2D,
                    file=str(file.resolve()),
                )
            else:
                files = []
                for face, pixels in enumerate(np.asarray(texture.pixels)):
                    file = assets / f"{name}_{face}.png"
                    Image.fromarray(pixels).save(file)
                    files.append(str(file.resolve()))
                texture_type = (
                    mujoco.mjtTexture.mjTEXTURE_SKYBOX
                    if texture.name == source.skybox
                    else mujoco.mjtTexture.mjTEXTURE_CUBE
                )
                spec.add_texture(name=name, type=texture_type, cubefiles=files)
            texture_names[texture.name] = name
        return texture_names

    @staticmethod
    def _joint_state_key(model, joint: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name:
            return name
        body = int(model.jnt_bodyid[joint])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)
        ordinal = joint - int(model.body_jntadr[body])
        return f"{body_name}:joint:{ordinal}"

    @staticmethod
    def _actuator_state_key(model, actuator: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        return name or f"actuator:{actuator}"

    @staticmethod
    def _span(addresses, index: int, total: int) -> slice:
        start = int(addresses[index])
        stop = int(addresses[index + 1]) if index + 1 < len(addresses) else int(total)
        return slice(start, stop)

    def _capture_named_model_state(self) -> _NamedModelState:
        model, data = self._m, self._d
        joints = {}
        for joint in range(model.njnt):
            qpos = self._span(model.jnt_qposadr, joint, model.nq)
            qvel = self._span(model.jnt_dofadr, joint, model.nv)
            joints[self._joint_state_key(model, joint)] = (
                np.asarray(data.qpos[qpos]).copy(),
                np.asarray(data.qvel[qvel]).copy(),
            )
        actuators = {}
        for actuator in range(model.nactuator):
            ctrl_start = int(model.actuator_ctrladr[actuator])
            ctrl_stop = ctrl_start + int(model.actuator_ctrlnum[actuator])
            act_start = int(model.actuator_actadr[actuator])
            act_stop = act_start + int(model.actuator_actnum[actuator])
            activation = (
                np.asarray(data.act[act_start:act_stop]).copy()
                if act_start >= 0
                else np.zeros(0, np.float64)
            )
            actuators[self._actuator_state_key(model, actuator)] = (
                np.asarray(data.ctrl[ctrl_start:ctrl_stop]).copy(),
                activation,
            )
        mocap = {}
        for body in range(1, model.nbody):
            mocap_id = int(model.body_mocapid[body])
            if mocap_id < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body:{body}"
            mocap[name] = (data.mocap_pos[mocap_id].copy(), data.mocap_quat[mocap_id].copy())
        equality = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
            or f"equality:{index}": bool(data.eq_active[index])
            for index in range(model.neq)
        }
        return _NamedModelState(joints, actuators, mocap, equality, float(data.time))

    def _transform_named_model_state(
        self,
        state: _NamedModelState,
        prefix: str,
        previous_position: np.ndarray,
        previous_rotation: np.ndarray,
        next_position: np.ndarray,
        next_rotation: np.ndarray,
    ) -> _NamedModelState:
        """Move world-space free-joint and mocap state with an attached model root."""
        delta_rotation = next_rotation @ previous_rotation.T
        joints = dict(state.joints)
        for joint in range(self._m.njnt):
            if int(self._m.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            body = int(self._m.jnt_bodyid[joint])
            body_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if not body_name.startswith(prefix):
                continue
            key = self._joint_state_key(self._m, joint)
            values = joints.get(key)
            if values is None or values[0].shape != (7,) or values[1].shape != (6,):
                continue
            qpos, qvel = values[0].copy(), values[1].copy()
            qpos[:3] = next_position + delta_rotation @ (qpos[:3] - previous_position)
            qpos[3:7] = math3d.mat3_to_quat(delta_rotation @ math3d.quat_to_mat3(qpos[3:7]))
            # Free-joint linear velocity is world-space; angular velocity is in
            # the local body frame and therefore remains unchanged.
            qvel[:3] = delta_rotation @ qvel[:3]
            joints[key] = (qpos, qvel)

        mocap = dict(state.mocap)
        for name, values in state.mocap.items():
            if not name.startswith(prefix):
                continue
            position, quaternion = values[0].copy(), values[1].copy()
            position[:] = next_position + delta_rotation @ (position - previous_position)
            quaternion[:] = math3d.mat3_to_quat(delta_rotation @ math3d.quat_to_mat3(quaternion))
            mocap[name] = (position, quaternion)
        return replace(state, joints=joints, mocap=mocap)

    def _restore_named_model_state(self, state: _NamedModelState) -> None:
        model, data = self._m, self._d
        for joint in range(model.njnt):
            values = state.joints.get(self._joint_state_key(model, joint))
            if values is None:
                continue
            qpos = self._span(model.jnt_qposadr, joint, model.nq)
            qvel = self._span(model.jnt_dofadr, joint, model.nv)
            if data.qpos[qpos].shape == values[0].shape:
                data.qpos[qpos] = values[0]
            if data.qvel[qvel].shape == values[1].shape:
                data.qvel[qvel] = values[1]
        for actuator in range(model.nactuator):
            values = state.actuators.get(self._actuator_state_key(model, actuator))
            if values is None:
                continue
            ctrl_start = int(model.actuator_ctrladr[actuator])
            ctrl_stop = ctrl_start + int(model.actuator_ctrlnum[actuator])
            act_start = int(model.actuator_actadr[actuator])
            act_stop = act_start + int(model.actuator_actnum[actuator])
            if data.ctrl[ctrl_start:ctrl_stop].shape == values[0].shape:
                data.ctrl[ctrl_start:ctrl_stop] = values[0]
            if act_start >= 0 and data.act[act_start:act_stop].shape == values[1].shape:
                data.act[act_start:act_stop] = values[1]
        for body in range(1, model.nbody):
            mocap_id = int(model.body_mocapid[body])
            if mocap_id < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body:{body}"
            values = state.mocap.get(name)
            if values is not None:
                data.mocap_pos[mocap_id], data.mocap_quat[mocap_id] = values
        for index in range(model.neq):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index) or f"equality:{index}"
            )
            if name in state.equality:
                data.eq_active[index] = state.equality[name]
        data.time = state.time
        mujoco.mj_forward(model, data)

    def _install(self, model, data=None) -> None:
        # A compiled model is authoritative. Transient placement indices refer to
        # the previous model layout and must never survive an install.
        self._model_transform_preview = None
        self._m = model
        self._d = data if data is not None else mujoco.MjData(model)
        self._notes = []
        self._rebuild_model_element_names()
        mujoco.mj_forward(self._m, self._d)

        g, b = model.ngeom, model.nbody
        self._rangefinder_specs = self._build_rangefinder_specs(model)
        rangefinder_count = sum(spec.ray_count for spec in self._rangefinder_specs)
        self._geom_xpos_buf = np.zeros((g, 3), np.float32)
        self._geom_xmat_buf = np.zeros((g, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((model.nsite, 3), np.float32)
        self._site_xmat_buf = np.zeros((model.nsite, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((b, 3), np.float32)
        self._body_xmat_buf = np.zeros((b, 3, 3), np.float32)
        self._diagnostic_frame = DiagnosticFrame(
            joint_xpos=np.zeros((model.njnt, 3), np.float32),
            joint_xaxis=np.zeros((model.njnt, 3), np.float32),
            subtree_com=np.zeros((b, 3), np.float32),
            body_xipos=np.zeros((b, 3), np.float32),
            body_ximat=np.zeros((b, 3, 3), np.float32),
            rangefinder_starts=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_ends=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_normals=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_lines=np.zeros(rangefinder_count, bool),
            rangefinder_points=np.zeros(rangefinder_count, bool),
            rangefinder_normal_arrows=np.zeros(rangefinder_count, bool),
            constraint_starts=np.zeros((model.neq, 3), np.float32),
            constraint_ends=np.zeros((model.neq, 3), np.float32),
            constraint_visible=np.zeros(model.neq, bool),
        )
        self._qpos_buf = np.zeros(model.nq, np.float32)
        self._qvel_buf = np.zeros(model.nv, np.float32)
        self._ctrl_buf = np.zeros(model.nu, np.float32)
        self._equality_enabled_buf = np.zeros(model.neq, bool)
        self._activation_buf = np.zeros(model.nactuator, np.float32)
        self._actuator_ctrl_address = np.asarray(model.actuator_ctrladr, np.int32).copy()
        self._ctrl_actuator = np.full(model.nu, -1, np.int32)
        for actuator, (address, count) in enumerate(
            zip(model.actuator_ctrladr, model.actuator_ctrlnum, strict=True)
        ):
            self._ctrl_actuator[int(address) : int(address) + int(count)] = actuator
        self._actuator_act_index = np.where(
            np.asarray(model.actuator_dyntype) != 0,
            np.asarray(model.actuator_actadr) + np.asarray(model.actuator_actnum) - 1,
            -1,
        ).astype(np.int32)
        self._sensor_buf = np.zeros(model.nsensordata, np.float32)
        self._flex_vertices_buf = np.zeros((model.nflexvert, 3), np.float32)
        self._contact_buf = np.zeros((max(model.ngeom, 64), 7), np.float32)
        self._contact_view = self._contact_buf[:0]
        self._contact_force_buf = np.zeros((len(self._contact_buf), 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf[:0]
        self._contact_island_rgba_buf = np.zeros((len(self._contact_buf), 4), np.float32)
        self._contact_island_rgba_view = self._contact_island_rgba_buf[:0]
        self._island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_island_rgba_buf = np.zeros((model.ntendon, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((model.nflex, 4), np.float32)
        self._body_island_rgba_buf = np.zeros((model.nbody, 4), np.float32)
        d = self._d
        wrap_capacity = d.wrap_xpos.size // 3
        self._tendon_segments = np.zeros((wrap_capacity, 2, 3), np.float32)
        self._tendon_ids = np.zeros(wrap_capacity, np.int32)
        self._tendon_widths = np.zeros(wrap_capacity, np.float32)
        self._actuator_visual_pose_types = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)
        self._slider_crank_actuators = np.zeros(0, np.int32)
        self._bvh_pose_type = np.zeros(0, np.uint8)
        self._bvh_pose_source = np.zeros(0, np.int32)
        self._bvh_global_index = np.zeros(0, np.int32)
        self._bvh_local_center = np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.zeros((0, 3), np.float32)
        self._bvh_control_body = np.zeros((0, 2), np.int32)
        self._bvh_control_local = np.zeros((0, 2, 3), np.float32)
        self._bvh_source_ready = False
        self._bind_data_views()
        self._fast_pose = self._verify_pose_layout()

        self._frame = SceneFrame()
        self._source = None
        self._nodes = []
        self._node_body = {}
        self._node_model = {}
        self._node_element = {}
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}
        self._deformables = []
        self._mesh_updates = {}
        for groups in self._visual_groups.values():
            groups[:] = [g in DEFAULT_GEOM_GROUPS for g in range(6)]
        self._ray_geomgroup[:] = self._visual_groups["geom"]
        self._lights_dynamic = bool(model.nlight) and bool(
            np.any(model.light_bodyid != 0)
            or np.any(model.light_mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        )
        area_names: set[str] = set()
        for prefix, names in _compiled_text_names(model, _FORGE_AREA_LIGHTS_TEXT):
            area_names.update(f"{prefix}{name}" for name in names)
        self._area_lights = np.asarray(
            [
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or "") in area_names
                for i in range(model.nlight)
            ],
            bool,
        )
        visual_fields = (
            "geom_rgba",
            "site_rgba",
            "flex_rgba",
            "skin_rgba",
            "tendon_rgba",
            "tendon_width",
            "mat_rgba",
            "mat_emission",
            "mat_specular",
            "mat_shininess",
            "mat_reflectance",
            "mat_texrepeat",
            "mat_texuniform",
            "mat_texid",
            "tex_data",
        )
        light_fields = (
            "light_type",
            "light_pos",
            "light_dir",
            "light_diffuse",
            "light_specular",
            "light_ambient",
            "light_attenuation",
            "light_range",
            "light_cutoff",
            "light_exponent",
            "light_intensity",
            "light_castshadow",
            "light_active",
        )
        self._visual_state = {
            name: np.asarray(getattr(model, name)).copy()
            for name in visual_fields
            if hasattr(model, name)
        }
        self._light_state = {
            name: np.asarray(getattr(model, name)).copy()
            for name in light_fields
            if hasattr(model, name)
        }
        self._lights_edited = False
        self._perturb = mujoco.MjvPerturb()
        self._perturb_body = -1
        self._perturb_jac = np.zeros((3, model.nv), np.float64)
        self._perturb_jac_m2 = np.zeros((3, model.nv), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(model.nv, np.float64)
        self._structure_revision += 1

    def _bind_data_views(self) -> None:
        m, d = self._m, self._d
        self._mj_geom_xpos = d.geom_xpos
        self._mj_geom_xmat3 = d.geom_xmat.reshape(m.ngeom, 3, 3)
        self._mj_site_xmat3 = d.site_xmat.reshape(m.nsite, 3, 3)
        # MuJoCo exposes wrap_xpos as packed xyz triples addressed by ten_wrapadr/num.
        self._mj_wrap_points = d.wrap_xpos.reshape(-1, 3)
        self._mj_wrap_objects = d.wrap_obj.reshape(-1)
        self._mj_body_xpos = d.xpos
        self._mj_body_xmat3 = d.xmat.reshape(m.nbody, 3, 3)

    @property
    def structure_revision(self) -> int:
        return self._structure_revision

    def _verify_pose_layout(self) -> bool:
        m, d = self._m, self._d
        g = m.ngeom
        if g == 0:
            return True
        try:
            xpos, xmat3 = self._mj_geom_xpos, self._mj_geom_xmat3
            if xpos.shape != (g, 3) or xmat3.shape != (g, 3, 3):
                return False
            if xpos.dtype != np.float64 or xmat3.dtype != np.float64:
                return False
            if not (xpos.flags["C_CONTIGUOUS"] and xmat3.flags["C_CONTIGUOUS"]):
                return False

            probe_pos = np.arange(g * 3, dtype=np.float64).reshape(g, 3) * 0.5 + 1.0
            probe_mat = np.arange(g * 9, dtype=np.float64).reshape(g, 3, 3) * 0.25 - 3.0
            xpos[:] = probe_pos
            xmat3[:] = probe_mat.reshape(g, 3, 3)
            for i in range(g):
                view = d.geom(i)
                if not np.array_equal(view.xpos, probe_pos[i]):
                    return False
                if not np.array_equal(view.xmat.reshape(3, 3), probe_mat[i]):
                    return False
            return True
        except Exception:
            return False
        finally:
            mujoco.mj_forward(m, d)

    def _fill_poses(self) -> None:
        if self._fast_pose:
            np.copyto(self._geom_xpos_buf, self._mj_geom_xpos, casting="unsafe")
            np.copyto(self._geom_xmat_buf, self._mj_geom_xmat3, casting="unsafe")
            np.copyto(self._body_xpos_buf, self._mj_body_xpos, casting="unsafe")
            np.copyto(self._body_xmat_buf, self._mj_body_xmat3, casting="unsafe")
            return

        d = self._d
        for i in range(len(self._geom_xpos_buf)):
            view = d.geom(i)
            self._geom_xpos_buf[i] = view.xpos
            self._geom_xmat_buf[i] = view.xmat.reshape(3, 3)
        for i in range(len(self._body_xpos_buf)):
            view = d.body(i)
            self._body_xpos_buf[i] = view.xpos
            self._body_xmat_buf[i] = view.xmat.reshape(3, 3)

    def reset(self) -> None:
        mujoco.mj_resetData(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)

    def set_paused(self, paused: bool) -> bool:
        """Pause ownership lives in Session; local MuJoCo needs no additional state."""
        return True

    def step(self, count: int = 1) -> None:
        for _ in range(max(1, int(count))):
            mujoco.mj_step(self._m, self._d)

    def timestep(self) -> float:
        return float(self._m.opt.timestep)

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        if self.prepare_frame(needs):
            self.scene_source()
        d = self._d
        f = self._frame
        f.time = float(d.time)

        if needs.poses:
            self._fill_poses()
            np.copyto(self._site_xpos_buf, d.site_xpos, casting="unsafe")
            np.copyto(self._site_xmat_buf, self._mj_site_xmat3, casting="unsafe")
            self._apply_model_transform_preview_poses()
            f.geom_xpos = self._geom_xpos_buf
            f.geom_xmat = self._geom_xmat_buf
            f.site_xpos = self._site_xpos_buf
            f.site_xmat = self._site_xmat_buf
            f.body_xpos = self._body_xpos_buf
            f.body_xmat = self._body_xmat_buf
        else:
            f.geom_xpos = f.geom_xmat = f.site_xpos = f.site_xmat = None
            f.body_xpos = f.body_xmat = None

        if needs.qpos:
            np.copyto(self._qpos_buf, d.qpos, casting="unsafe")
            f.qpos = self._qpos_buf
        else:
            f.qpos = None

        if needs.qvel:
            np.copyto(self._qvel_buf, d.qvel, casting="unsafe")
            f.qvel = self._qvel_buf
        else:
            f.qvel = None

        if needs.actuator:
            np.copyto(self._ctrl_buf, d.ctrl, casting="unsafe")
            f.ctrl = self._ctrl_buf
            np.take(d.ctrl, self._actuator_ctrl_address, out=self._activation_buf)
            active = self._actuator_act_index >= 0
            self._activation_buf[active] = d.act[self._actuator_act_index[active]]
            f.actuator_activation = self._activation_buf
        else:
            f.ctrl = None
            f.actuator_activation = None

        if needs.sensors:
            np.copyto(self._sensor_buf, d.sensordata, casting="unsafe")
            f.sensors = self._sensor_buf
        else:
            f.sensors = None

        np.copyto(self._equality_enabled_buf, d.eq_active, casting="unsafe")
        f.equality_enabled = self._equality_enabled_buf

        if needs.contacts:
            f.contacts = self._fill_contacts(needs.islands)
            f.contact_forces = self._contact_force_view
            f.contact_island_rgba = self._contact_island_rgba_view if needs.islands else None
        else:
            f.contacts = None
            f.contact_forces = None
            f.contact_island_rgba = None
        if needs.tendons:
            f.tendon_segments, f.tendon_ids, f.tendon_widths = self._fill_tendons()
        else:
            f.tendon_segments = f.tendon_ids = f.tendon_widths = None
        if needs.deformables:
            update_deformables(self._deformables, d)
            f.mesh_updates = self._mesh_updates
            np.copyto(self._flex_vertices_buf, d.flexvert_xpos, casting="unsafe")
            f.flex_vertices = self._flex_vertices_buf
        else:
            f.mesh_updates = None
            f.flex_vertices = None

        if needs.islands:
            f.island_rgba, f.tendon_island_rgba, f.flex_island_rgba = self._fill_island_colors()
        else:
            f.island_rgba = None
            f.tendon_island_rgba = None
            f.flex_island_rgba = None

        if needs.diagnostics:
            diagnostics = self._diagnostic_frame
            np.copyto(diagnostics.joint_xpos, d.xanchor, casting="unsafe")
            np.copyto(diagnostics.joint_xaxis, d.xaxis, casting="unsafe")
            np.copyto(diagnostics.subtree_com, d.subtree_com, casting="unsafe")
            np.copyto(diagnostics.body_xipos, d.xipos, casting="unsafe")
            np.copyto(
                diagnostics.body_ximat,
                d.ximat.reshape(self._m.nbody, 3, 3),
                casting="unsafe",
            )
            self._fill_actuator_visual_poses(diagnostics)
            self._fill_slider_crank_visuals(diagnostics)
            self._fill_autoconnect_visuals(diagnostics)
            self._fill_rangefinder_visuals(diagnostics)
            self._fill_constraint_visuals(diagnostics)
            if needs.bvh:
                self._fill_bvh_visuals(diagnostics)
            self._apply_model_transform_preview_diagnostics(diagnostics)
            f.diagnostics = diagnostics
            f.cameras = tuple(self.camera_view(i) for i in range(self._m.ncam))
        else:
            f.diagnostics = None
            f.cameras = None

        f.lights = (
            self._dynamic_lights()
            if self._lights_dynamic
            or self._lights_edited
            or self._model_transform_preview is not None
            else None
        )
        return f

    def _apply_model_transform_preview_poses(self) -> None:
        preview = self._model_transform_preview
        if preview is None:
            return
        self._transform_preview_group(
            self._body_xpos_buf, self._body_xmat_buf, preview.body_indices
        )
        self._transform_preview_group(
            self._geom_xpos_buf, self._geom_xmat_buf, preview.geom_indices
        )
        self._transform_preview_group(
            self._site_xpos_buf, self._site_xmat_buf, preview.site_indices
        )

    def _apply_model_transform_preview_diagnostics(self, diagnostics: DiagnosticFrame) -> None:
        preview = self._model_transform_preview
        if preview is None:
            return
        self._transform_preview_points(diagnostics.joint_xpos, preview.joint_indices)
        self._transform_preview_directions(diagnostics.joint_xaxis, preview.joint_indices)
        self._transform_preview_points(diagnostics.subtree_com, preview.body_indices)
        self._transform_preview_group(
            diagnostics.body_xipos, diagnostics.body_ximat, preview.body_indices
        )

    def _transform_preview_group(
        self, positions: np.ndarray, matrices: np.ndarray, indices: np.ndarray
    ) -> None:
        self._transform_preview_points(positions, indices)
        self._transform_preview_matrices(matrices, indices)

    def _transform_preview_points(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.point_input[:count]
        target = preview.point_output[:count]
        np.take(values, indices, axis=0, out=source)
        source -= preview.previous_position
        np.einsum(
            "ij,nj->ni",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        target += preview.position
        values[indices] = target

    def _transform_preview_directions(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.point_input[:count]
        target = preview.point_output[:count]
        np.take(values, indices, axis=0, out=source)
        np.einsum(
            "ij,nj->ni",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        values[indices] = target

    def _transform_preview_matrices(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.matrix_input[:count]
        target = preview.matrix_output[:count]
        np.take(values, indices, axis=0, out=source)
        np.einsum(
            "ij,njk->nik",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        values[indices] = target

    def prepare_frame(self, needs: FrameNeeds) -> bool:
        """Build potentially huge BVH diagnostics only after they are requested."""
        if not needs.bvh or self._bvh_source_ready:
            return False
        self._bvh_source_ready = True
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def _fill_tendons(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = 0
        self._tendon_ids.fill(-1)
        for ti in range(self._m.ntendon):
            start = int(self._d.ten_wrapadr[ti])
            points = int(self._d.ten_wrapnum[ti])
            segments = max(points - 1, 0)
            stop = count + segments
            self._tendon_segments[count:stop, 0] = self._mj_wrap_points[start : start + segments]
            self._tendon_segments[count:stop, 1] = self._mj_wrap_points[
                start + 1 : start + 1 + segments
            ]
            self._tendon_ids[count:stop] = ti
            widths = self._tendon_widths[count:stop]
            widths[:] = float(self._m.tendon_width[ti])
            inside = (self._mj_wrap_objects[start : start + segments] >= 0) & (
                self._mj_wrap_objects[start + 1 : start + 1 + segments] >= 0
            )
            widths[inside] *= 0.5
            count = stop
        return (
            self._tendon_segments[:count],
            self._tendon_ids[:count],
            self._tendon_widths[:count],
        )

    def _fill_contacts(self, islands: bool = False) -> np.ndarray:
        d, m = self._d, self._m
        n = int(d.ncon)
        if n > len(self._contact_buf):
            capacity = max(n, 2 * len(self._contact_buf))
            self._contact_buf = np.zeros((capacity, 7), np.float32)
            self._contact_force_buf = np.zeros((capacity, 2, 3), np.float32)
            self._contact_island_rgba_buf = np.zeros((capacity, 4), np.float32)
            self._contact_view = self._contact_buf[:0]
            self._contact_force_view = self._contact_force_buf[:0]
            self._contact_island_rgba_view = self._contact_island_rgba_buf[:0]
        for i in range(n):
            c = d.contact[i]
            self._contact_buf[i, 0:3] = c.pos
            self._contact_buf[i, 3:6] = c.frame[0:3]
            mujoco.mj_contactForce(m, d, i, self._contact_force)
            local = self._contact_force[:3].copy()
            if int(c.dim) < 3:
                local[int(c.dim) :] = 0.0
            rotation = np.asarray(c.frame, np.float64).reshape(3, 3).T
            self._contact_force_buf[i, 0] = rotation[:, 0] * local[0]
            self._contact_force_buf[i, 1] = rotation[:, 1:] @ local[1:]
            first = int(m.geom_bodyid[c.geom[0]]) if c.geom[0] >= 0 else m.nbody + int(c.flex[0])
            second = int(m.geom_bodyid[c.geom[1]]) if c.geom[1] >= 0 else m.nbody + int(c.flex[1])
            if first > second:
                self._contact_force_buf[i] *= -1.0
            self._contact_buf[i, 6] = np.linalg.norm(local)
            if islands:
                address = int(c.efc_address)
                if address >= 0:
                    island = int(d.efc_island[address]) if int(d.nisland) else -1
                    key = int(d.island_dofadr[island]) if island >= 0 else -1
                    self._write_island_color(self._contact_island_rgba_buf[i], key, True)
                else:
                    self._contact_island_rgba_buf[i] = m.vis.rgba.contactgap
        if len(self._contact_view) != n:
            self._contact_view = self._contact_buf[:n]
            self._contact_force_view = self._contact_force_buf[:n]
            self._contact_island_rgba_view = self._contact_island_rgba_buf[:n]
        return self._contact_view

    def _fill_island_colors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.scene_source()
        bodies = source.instance_island_body
        moving = bodies >= 0
        np.copyto(self._island_rgba_buf, source.geom_rgba)
        for body in np.unique(bodies[moving]):
            key, awake = self._body_island_key(int(body))
            self._write_island_color(self._body_island_rgba_buf[int(body)], key, awake)
        if np.any(moving):
            self._island_rgba_buf[moving] = self._body_island_rgba_buf[bodies[moving]]

        model, data = self._m, self._d
        for flex in range(model.nflex):
            body = self._flex_island_body(flex)
            if body >= 0:
                self._flex_island_rgba_buf[flex] = self._body_island_rgba_buf[body]
            else:
                color = np.asarray(model.flex_rgba[flex], np.float32)
                material = int(model.flex_matid[flex])
                if material >= 0 and np.array_equal(color, _GEOM_RGBA_DEFAULT):
                    color = model.mat_rgba[material]
                self._flex_island_rgba_buf[flex] = color

        for tendon in range(model.ntendon):
            key = -1
            address = int(data.tendon_efcadr[tendon])
            if int(data.nisland) and address >= 0:
                island = int(data.efc_island[address])
                key = int(data.island_dofadr[island])
            self._write_island_color(self._tendon_island_rgba_buf[tendon], key, True)
        return (
            self._island_rgba_buf,
            self._tendon_island_rgba_buf,
            self._flex_island_rgba_buf,
        )

    def _body_island_key(self, body: int) -> tuple[int, bool]:
        model, data = self._m, self._d
        weld = int(model.body_weldid[body])
        dof = int(model.body_dofadr[weld])
        island = int(data.dof_island[dof]) if int(data.nisland) else -1
        key = int(data.island_dofadr[island]) if island >= 0 else -1
        awake = bool(data.body_awake[body])
        sleep = int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        if key < 0 and sleep:
            tree = int(model.dof_treeid[dof])
            if not awake:
                tree = self._sleep_cycle(data.tree_asleep, tree)
            if tree >= 0:
                key = int(model.tree_dofadr[tree])
        return key, awake

    def _flex_island_body(self, flex: int) -> int:
        model = self._m
        if bool(model.flex_interp[flex]):
            address = int(model.flex_nodeadr[flex])
            count = int(model.flex_nodenum[flex])
            bodies = model.flex_nodebodyid[address : address + count]
        else:
            address = int(model.flex_vertadr[flex])
            count = int(model.flex_vertnum[flex])
            bodies = model.flex_vertbodyid[address : address + count]
        return next((int(body) for body in bodies if int(model.body_treeid[body]) >= 0), -1)

    def _sleep_cycle(self, tree_asleep: np.ndarray, start: int) -> int:
        if start < 0 or start >= self._m.ntree:
            return -1
        smallest = current = start
        for _ in range(self._m.ntree + 1):
            current = int(tree_asleep[current])
            if current < 0 or current >= self._m.ntree:
                return -1
            smallest = min(smallest, current)
            if current == start:
                return smallest
        return -1

    @staticmethod
    def _write_island_color(out: np.ndarray, key: int, awake: bool) -> None:
        hue, saturation, value = 1.0, 0.0, 0.7
        if key >= 0:
            hue = float(mujoco.mju_Halton(key + 1, 7))
            saturation = 0.5 + 0.5 * float(mujoco.mju_Halton(key + 1, 3))
            value = 0.6 + 0.4 * float(mujoco.mju_Halton(key + 1, 5))
        if not awake:
            saturation *= 0.7
            value *= 0.6
        out[:3] = hsv_to_rgb(hue, saturation, value)
        out[3] = 1.0

    def scene_source(self) -> SceneSource:
        if self._source is None:
            self._source = self._build_source()
        return self._source

    def nodes(self) -> list[SceneNode]:
        if not self._nodes:
            self._nodes = self._build_nodes()
        return self._nodes

    def _build_source(self) -> SceneSource:
        m = self._m
        src = SceneSource()
        src.shading_model = ShadingModel.MUJOCO_CLASSIC
        src.nodes = self.nodes()
        src.diagnostics = self._build_diagnostic_source()
        trn_tendon = np.asarray(m.actuator_trntype) == int(mujoco.mjtTrn.mjTRN_TENDON)
        src.actuator_tendon = np.where(trn_tendon, m.actuator_trnid[:, 0], -1).astype(np.int32)
        src.actuator_visible = self._group_visibility(m.actuator_group, "actuator")
        disabled_groups = int(m.opt.disableactuator)
        groups = np.clip(np.asarray(m.actuator_group, np.int32), 0, 30)
        src.actuator_visible &= (disabled_groups & (1 << groups)) == 0
        src.actuator_ctrl_address = self._actuator_ctrl_address.copy()
        src.actuator_ctrl_limited = np.asarray(m.actuator_ctrllimited, bool).copy()
        src.actuator_ctrl_range = np.asarray(m.actuator_ctrlrange, np.float32).copy()
        src.actuator_act_limited = np.asarray(m.actuator_actlimited, bool).copy()
        src.actuator_act_range = np.asarray(m.actuator_actrange, np.float32).copy()
        src.actuator_dynamic = (np.asarray(m.actuator_dyntype) != 0).astype(bool)
        src.actuator_rgba = np.asarray(
            [m.vis.rgba.actuatornegative, m.vis.rgba.actuator, m.vis.rgba.actuatorpositive],
            np.float32,
        )
        src.actuator_tendon_scale = float(m.vis.map.actuatortendon)

        textures = self._build_textures()
        src.textures = textures
        src.skybox = next((t.name for t in textures.values() if t.type is TextureType.SKYBOX), None)
        materials, mat_of_matid = self._build_materials(textures)
        src.materials = materials

        meshes: dict[MeshKey, MeshData] = {}
        mesh_keys: list[MeshKey] = []
        convex_mesh_keys: list[MeshKey] = []
        mats: list[int] = []
        sizes: list[np.ndarray] = []
        rgbas: list[np.ndarray] = []
        object_ids: list[int] = []
        bodies: list[int] = []
        sources: list[int] = []
        pose_sources: list[int] = []
        visuals: list[int] = []
        statics: list[bool] = []
        island_bodies: list[int] = []
        node_ids: list[int] = []
        locals_: list[np.ndarray] = []
        infinite: list[bool] = []
        geom_groups = set(np.flatnonzero(self._visual_groups["geom"]))
        site_groups = set(np.flatnonzero(self._visual_groups["site"]))
        flex_groups = set(np.flatnonzero(self._visual_groups["flex"]))
        skin_groups = set(np.flatnonzero(self._visual_groups["skin"]))
        skipped: set[int] = set()

        def append_parts(
            parts,
            *,
            mat_index: int,
            rgba: np.ndarray,
            body: int,
            source: int,
            pose_source: InstancePoseSource,
            node_id: int,
            object_id: int,
            is_infinite: bool = False,
            visual: InstanceVisual = InstanceVisual.DEFAULT,
            is_static: bool = False,
            island_body: int = -1,
            convex_mesh: MeshKey | None = None,
        ) -> None:
            for key, scale, cap_offset in parts:
                if key.shape not in (MeshShape.ASSET, MeshShape.CONVEX_HULL) and key not in meshes:
                    meshes[key] = None
                mesh_keys.append(key)
                convex_mesh_keys.append(convex_mesh or key)
                mats.append(mat_index)
                sizes.append(np.asarray(scale, np.float32))
                rgbas.append(rgba)
                object_ids.append(object_id)
                bodies.append(body)
                sources.append(source)
                pose_sources.append(int(pose_source))
                visuals.append(int(visual))
                statics.append(is_static)
                island_bodies.append(island_body)
                node_ids.append(node_id)
                local = np.eye(4, dtype=np.float32)
                if cap_offset is not None:
                    local[2, 3] = cap_offset
                    if cap_offset < 0.0:
                        local[1, 1] = -1.0
                        local[2, 2] = -1.0
                locals_.append(local)
                infinite.append(is_infinite)

        for gi in range(m.ngeom):
            if int(m.geom_group[gi]) not in geom_groups:
                continue
            gtype = int(m.geom_type[gi])
            size = np.asarray(m.geom_size[gi], np.float64)
            body = int(m.geom_bodyid[gi])
            matid = int(m.geom_matid[gi])
            rgba = self._geom_rgba(gi, matid)
            mat_index = mat_of_matid[matid] if matid >= 0 else mat_of_matid[-1]
            is_infinite = False
            hull_key = None

            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                key = MeshKey(MeshShape.PLANE)

                scale = np.array([size[0], size[1], 1.0], np.float64)
                is_infinite = size[0] == 0.0 or size[1] == 0.0
                parts = [(key, scale, None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_HFIELD:
                data_id = int(m.geom_dataid[gi])
                if data_id < 0:
                    skipped.add(gtype)
                    continue
                key = MeshKey(MeshShape.HEIGHTFIELD, data_id)
                if key not in meshes:
                    meshes[key] = self._build_heightfield(data_id)
                hs = np.asarray(m.hfield_size[data_id], np.float64)
                parts = [(key, hs[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                parts = [(MeshKey(MeshShape.SPHERE), np.full(3, size[0]), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                parts = [(MeshKey(MeshShape.SPHERE), size[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                parts = [(MeshKey(MeshShape.BOX), size[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                parts = [(MeshKey(MeshShape.CYLINDER), np.array([size[0], size[0], size[1]]), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, half = float(size[0]), float(size[1])
                parts = [
                    (MeshKey(MeshShape.CAPSULE_SHAFT), np.array([r, r, half]), None),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), +half),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), -half),
                ]
            elif gtype in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
                data_id = int(m.geom_dataid[gi])
                if data_id < 0:
                    skipped.add(gtype)
                    continue
                key = MeshKey(MeshShape.ASSET, data_id)
                if key not in meshes:
                    meshes[key] = self._build_mesh(data_id)
                parts = [(key, np.ones(3), None)]
                if int(m.mesh_graphadr[data_id]) >= 0 and (
                    int(m.geom_contype[gi]) or int(m.geom_conaffinity[gi])
                ):
                    hull_key = MeshKey(MeshShape.CONVEX_HULL, data_id)
                    if hull_key not in meshes:
                        meshes[hull_key] = self._build_convex_hull(data_id)
            else:
                skipped.add(gtype)
                continue

            append_parts(
                parts,
                mat_index=mat_index,
                rgba=rgba,
                body=body,
                source=gi,
                pose_source=InstancePoseSource.GEOM,
                node_id=self._geom_nodes.get(gi, -1),
                object_id=(
                    self.nodes()[self._geom_nodes[gi]].object_id
                    if body == 0 and gi in self._geom_nodes
                    else body
                ),
                is_infinite=is_infinite,
                is_static=int(m.body_weldid[body]) == 0,
                island_body=body if int(m.body_dofnum[int(m.body_weldid[body])]) else -1,
                convex_mesh=hull_key,
            )

        for si in range(m.nsite):
            if int(m.site_group[si]) not in site_groups:
                continue
            stype = int(m.site_type[si])
            size = np.asarray(m.site_size[si], np.float64)
            if stype == mujoco.mjtGeom.mjGEOM_SPHERE:
                parts = [(MeshKey(MeshShape.SPHERE), np.full(3, size[0]), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                parts = [(MeshKey(MeshShape.SPHERE), size[:3].copy(), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_BOX:
                parts = [(MeshKey(MeshShape.BOX), size[:3].copy(), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                parts = [(MeshKey(MeshShape.CYLINDER), np.array([size[0], size[0], size[1]]), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, half = float(size[0]), float(size[1])
                parts = [
                    (MeshKey(MeshShape.CAPSULE_SHAFT), np.array([r, r, half]), None),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), +half),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), -half),
                ]
            else:
                skipped.add(stype)
                continue
            body = int(m.site_bodyid[si])
            matid = int(m.site_matid[si])
            mat_index = mat_of_matid[matid] if matid >= 0 else mat_of_matid[-1]
            rgba = self._site_rgba(si, matid)
            append_parts(
                parts,
                mat_index=mat_index,
                rgba=rgba,
                body=body,
                source=si,
                pose_source=InstancePoseSource.SITE,
                node_id=self._site_nodes.get(si, -1),
                object_id=0,
                is_static=int(m.body_weldid[body]) == 0,
            )

        self._deformables = build_deformables(m, self._d, flex_groups, skin_groups)
        self._mesh_updates = {spec.key: spec.update_data for spec in self._deformables}
        for spec in self._deformables:
            meshes[spec.key] = spec.mesh
            mat_index = mat_of_matid[spec.matid] if spec.matid >= 0 else mat_of_matid[-1]
            rgba = spec.rgba
            if spec.matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
                rgba = np.asarray(m.mat_rgba[spec.matid], np.float32)
            is_flex = spec.key.shape in (MeshShape.FLEX, MeshShape.FLEX_FACE)
            node_id = (
                self._flex_nodes.get(spec.key.index, -1)
                if is_flex
                else self._skin_nodes.get(spec.key.index, -1)
            )
            object_id = m.nbody + spec.key.index if is_flex else m.nbody + m.nflex + spec.key.index
            append_parts(
                [(spec.key, np.ones(3), None)],
                mat_index=mat_index,
                rgba=np.asarray(rgba, np.float32).copy(),
                body=0,
                source=0,
                pose_source=InstancePoseSource.WORLD,
                node_id=node_id,
                object_id=object_id,
                visual=spec.visual,
                island_body=self._flex_island_body(spec.key.index) if is_flex else -1,
            )

        src.meshes = {k: v for k, v in meshes.items() if v is not None}
        src.dynamic_meshes = frozenset(spec.key for spec in self._deformables)
        src.geom_mesh = mesh_keys
        src.geom_convex_mesh = convex_mesh_keys
        src.geom_material = mats
        n = len(mesh_keys)
        src.geom_size = np.stack(sizes) if n else np.zeros((0, 3), np.float32)
        src.geom_rgba = np.stack(rgbas) if n else np.zeros((0, 4), np.float32)
        src.geom_object_id = np.array(object_ids, np.uint32)
        src.geom_body = np.array(bodies, np.int32)
        src.geom_source = np.array(sources, np.int32)
        src.geom_pose_source = np.array(pose_sources, np.uint8)
        src.geom_visual = np.array(visuals, np.uint8)
        src.geom_static = np.array(statics, bool)
        src.instance_island_body = np.array(island_bodies, np.int32)
        src.geom_node = np.array(node_ids, np.int32)
        src.geom_local = np.stack(locals_) if n else np.zeros((0, 4, 4), np.float32)
        src.geom_infinite_plane = np.array(infinite, bool)
        src.body_names = _object_names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody, "body")
        src.joint_names = _object_names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt, "joint")
        src.geom_names = _object_names(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom, "geom")
        src.site_names = _object_names(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite, "site")
        src.camera_names = _object_names(m, mujoco.mjtObj.mjOBJ_CAMERA, m.ncam, "camera")
        src.light_names = _object_names(m, mujoco.mjtObj.mjOBJ_LIGHT, m.nlight, "light")
        src.tendon_names = _object_names(m, mujoco.mjtObj.mjOBJ_TENDON, m.ntendon, "tendon")
        src.actuator_names = _object_names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu, "actuator")
        src.constraint_names = _object_names(m, mujoco.mjtObj.mjOBJ_EQUALITY, m.neq, "constraint")
        src.flex_names = _object_names(m, mujoco.mjtObj.mjOBJ_FLEX, m.nflex, "flex")
        self._build_flex_debug_source(src)
        src.lights = self._build_lights()
        src.cameras = tuple(self.camera_view(i) for i in range(m.ncam))
        src.scene_extent = float(m.stat.extent)

        src.shadow_clip = float(m.vis.map.shadowclip) or 1.0
        src.scene_center = np.asarray(m.stat.center, np.float32)
        src.debug_frame_length = float(m.stat.meansize) * float(m.vis.scale.framelength)

        src.initial_qpos = np.asarray(m.qpos0, np.float32).copy()

        src.initial_ctrl = np.zeros(m.nu, np.float32)
        tendon_matid = np.asarray(m.tendon_matid, np.int32)
        src.tendon_material = np.asarray(
            [mat_of_matid[int(matid)] for matid in tendon_matid], np.int32
        )
        src.tendon_rgba = np.asarray(m.tendon_rgba, np.float32).copy()
        material_color = (tendon_matid >= 0) & np.all(src.tendon_rgba == _GEOM_RGBA_DEFAULT, axis=1)
        src.tendon_rgba[material_color] = m.mat_rgba[tendon_matid[material_color]]
        src.tendon_visible = self._group_visibility(m.tendon_group, "tendon")
        self._island_rgba_buf = src.geom_rgba.copy()
        self._tendon_island_rgba_buf = np.zeros((m.ntendon, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((m.nflex, 4), np.float32)

        if skipped:
            names = ", ".join(sorted(str(mujoco.mjtGeom(t)) for t in skipped))

            note = f"Skipped unsupported geom types: {names}"
            if note not in self._notes:
                self._notes.append(note)
            self.caps = replace(self.caps, notes=tuple(self._notes))
        return src

    def _build_flex_debug_source(self, source: SceneSource) -> None:
        model = self._m
        vertex_indices: list[np.ndarray] = []
        edges: list[np.ndarray] = []
        vertex_colors: list[np.ndarray] = []
        edge_colors: list[np.ndarray] = []
        vertex_owners: list[np.ndarray] = []
        edge_owners: list[np.ndarray] = []
        ranges = np.zeros((model.nflex, 2), np.int32)
        for flex in range(model.nflex):
            if not self._visual_groups["flex"][int(model.flex_group[flex])]:
                continue
            vertex_address = int(model.flex_vertadr[flex])
            vertex_count = int(model.flex_vertnum[flex])
            edge_address = int(model.flex_edgeadr[flex])
            edge_count = int(model.flex_edgenum[flex])
            ranges[flex] = (vertex_address, vertex_count)
            color = np.asarray(model.flex_rgba[flex], np.float32).copy()
            material = int(model.flex_matid[flex])
            if material >= 0 and np.array_equal(color, _GEOM_RGBA_DEFAULT):
                color = np.asarray(model.mat_rgba[material], np.float32).copy()
            vertex_indices.append(
                np.arange(vertex_address, vertex_address + vertex_count, dtype=np.int32)
            )
            edges.append(
                np.asarray(model.flex_edge[edge_address : edge_address + edge_count], np.int32)
            )
            vertex_colors.append(np.repeat(color[None], vertex_count, axis=0))
            edge_colors.append(np.repeat(color[None], edge_count, axis=0))
            vertex_owners.append(np.full(vertex_count, flex, np.int32))
            edge_owners.append(np.full(edge_count, flex, np.int32))
        source.flex_vertex_indices = (
            np.concatenate(vertex_indices) if vertex_indices else np.zeros(0, np.int32)
        )
        source.flex_edges = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), np.int32)
        source.flex_vertex_rgba = (
            np.concatenate(vertex_colors, axis=0) if vertex_colors else np.zeros((0, 4), np.float32)
        )
        source.flex_edge_rgba = (
            np.concatenate(edge_colors, axis=0) if edge_colors else np.zeros((0, 4), np.float32)
        )
        source.flex_vertex_owner = (
            np.concatenate(vertex_owners) if vertex_owners else np.zeros(0, np.int32)
        )
        source.flex_edge_owner = (
            np.concatenate(edge_owners) if edge_owners else np.zeros(0, np.int32)
        )
        source.flex_vertex_ranges = ranges

    def _build_bvh_records(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self._m
        record_types: list[int] = []
        depths: list[int] = []
        leaves: list[bool] = []
        pose_types: list[int] = []
        pose_sources: list[int] = []
        global_indices: list[int] = []
        centers: list[np.ndarray] = []
        sizes: list[np.ndarray] = []

        def append(
            record_type: int,
            depth: int,
            leaf: bool,
            pose_type: int,
            pose_source: int,
            global_index: int,
            center,
            size,
        ) -> None:
            record_types.append(record_type)
            depths.append(depth)
            leaves.append(leaf)
            pose_types.append(pose_type)
            pose_sources.append(pose_source)
            global_indices.append(global_index)
            centers.append(np.asarray(center, np.float32))
            sizes.append(np.asarray(size, np.float32))

        for body in range(m.nbody):
            start = int(m.body_bvhadr[body])
            for index in range(start, start + int(m.body_bvhnum[body])):
                leaf = bool(np.all(np.asarray(m.bvh_child[index]) == -1))
                geom = int(m.bvh_nodeid[index])
                if leaf:
                    center, size = m.geom_aabb[geom, :3], m.geom_aabb[geom, 3:]
                    pose_type, pose_source = _BVH_POSE_GEOM, geom
                else:
                    center, size = m.bvh_aabb[index, :3], m.bvh_aabb[index, 3:]
                    pose_type, pose_source = _BVH_POSE_BODY, body
                append(
                    int(BvhType.BODY),
                    int(m.bvh_depth[index]),
                    leaf,
                    pose_type,
                    pose_source,
                    index,
                    center,
                    size,
                )

        for flex in range(m.nflex):
            if not self._visual_groups["flex"][int(m.flex_group[flex])]:
                continue
            start = int(m.flex_bvhadr[flex])
            for index in range(start, start + int(m.flex_bvhnum[flex])):
                append(
                    int(BvhType.FLEX),
                    int(m.bvh_depth[index]),
                    bool(np.all(np.asarray(m.bvh_child[index]) == -1)),
                    _BVH_POSE_DYNAMIC,
                    index,
                    index,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                )

        mesh_geom = int(mujoco.mjtGeom.mjGEOM_MESH)
        hfield_geom = int(mujoco.mjtGeom.mjGEOM_HFIELD)
        for geom in range(m.ngeom):
            mesh = int(m.geom_dataid[geom])
            if mesh < 0:
                continue
            if int(m.geom_type[geom]) == mesh_geom and int(m.mesh_octadr[mesh]) < 0:
                start = int(m.mesh_bvhadr[mesh])
                for index in range(start, start + int(m.mesh_bvhnum[mesh])):
                    append(
                        int(BvhType.MESH),
                        int(m.bvh_depth[index]),
                        bool(np.all(np.asarray(m.bvh_child[index]) == -1)),
                        _BVH_POSE_GEOM,
                        geom,
                        index,
                        m.bvh_aabb[index, :3],
                        m.bvh_aabb[index, 3:],
                    )
            if int(m.geom_type[geom]) not in (hfield_geom,) and int(m.mesh_octadr[mesh]) >= 0:
                start = int(m.mesh_octadr[mesh])
                for index in range(start, start + int(m.mesh_octnum[mesh])):
                    append(
                        int(BvhType.OCTREE),
                        int(m.oct_depth[index]),
                        False,
                        _BVH_POSE_GEOM,
                        geom,
                        -1,
                        m.oct_aabb[index, :3],
                        m.oct_aabb[index, 3:],
                    )

        self._bvh_pose_type = np.asarray(pose_types, np.uint8)
        self._bvh_pose_source = np.asarray(pose_sources, np.int32)
        self._bvh_global_index = np.asarray(global_indices, np.int32)
        self._bvh_local_center = np.stack(centers) if centers else np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.stack(sizes) if sizes else np.zeros((0, 3), np.float32)
        return (
            np.asarray(record_types, np.uint8),
            np.asarray(depths, np.int32),
            np.asarray(leaves, bool),
        )

    def _build_bvh_control_cages(self) -> int:
        m = self._m
        edges: list[tuple[int, int]] = []
        centered = np.zeros(len(m.flex_nodebodyid), bool)
        for flex in range(m.nflex):
            order = abs(int(m.flex_interp[flex]))
            if not order:
                continue
            start = int(m.flex_nodeadr[flex])
            centered[start : start + int(m.flex_nodenum[flex])] = bool(m.flex_centered[flex])
            nx, ny, nz = np.asarray(m.flex_cellnum[flex], np.int32) * order + 1
            shell = int(m.flex_interp[flex]) < 0

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        current = _grid_node(start, ny, nz, i, j, k)
                        if not m.body_jntnum[m.flex_nodebodyid[current]] or (
                            shell and not _grid_boundary(nx, ny, nz, i, j, k)
                        ):
                            continue
                        for neighbor in (
                            _grid_node(start, ny, nz, i + 1, j, k) if i + 1 < nx else -1,
                            _grid_node(start, ny, nz, i, j + 1, k) if j + 1 < ny else -1,
                            _grid_node(start, ny, nz, i, j, k + 1) if k + 1 < nz else -1,
                        ):
                            if neighbor >= 0 and m.body_jntnum[m.flex_nodebodyid[neighbor]]:
                                local = neighbor - start
                                ni, rem = divmod(local, ny * nz)
                                nj, nk = divmod(rem, nz)
                                if not shell or _grid_boundary(nx, ny, nz, ni, nj, nk):
                                    edges.append((current, neighbor))

        nodes = np.asarray(edges, np.int32).reshape(-1, 2)
        self._bvh_control_body = np.asarray(m.flex_nodebodyid[nodes], np.int32)
        self._bvh_control_local = np.asarray(m.flex_node[nodes], np.float32).copy()
        self._bvh_control_local[centered[nodes]] = 0.0
        return len(nodes)

    def _build_diagnostic_source(self) -> DiagnosticSource:
        m = self._m
        if self._bvh_source_ready:
            bvh_type, bvh_depth, bvh_leaf = self._build_bvh_records()
            bvh_control_count = self._build_bvh_control_cages()
        else:
            bvh_type = np.zeros(0, np.uint8)
            bvh_depth = np.zeros(0, np.int32)
            bvh_leaf = np.zeros(0, bool)
            bvh_control_count = 0
        bvh_rgba = np.asarray(m.vis.rgba.bv, np.float32)
        if bvh_rgba.shape != (4,):
            bvh_rgba = np.array([0.0, 1.0, 0.0, 0.5], np.float32)
        bvh_active_rgba = np.asarray(m.vis.rgba.bvactive, np.float32)
        if bvh_active_rgba.shape != (4,):
            bvh_active_rgba = np.array([1.0, 0.0, 0.0, 0.5], np.float32)
        joint_types = np.empty(m.njnt, np.uint8)
        joint_type_map = {
            int(mujoco.mjtJoint.mjJNT_FREE): JointVisualType.FREE,
            int(mujoco.mjtJoint.mjJNT_BALL): JointVisualType.BALL,
            int(mujoco.mjtJoint.mjJNT_SLIDE): JointVisualType.SLIDE,
            int(mujoco.mjtJoint.mjJNT_HINGE): JointVisualType.HINGE,
        }
        for source_type, visual_type in joint_type_map.items():
            joint_types[np.asarray(m.jnt_type) == source_type] = int(visual_type)

        meansize = float(m.stat.meansize)
        meanmass = float(m.stat.meanmass)
        com_bodies = np.flatnonzero(np.asarray(m.body_parentid[1:]) == 0).astype(np.int32) + 1
        inertia_bodies = np.flatnonzero(
            (np.asarray(m.body_dofnum) > 0) & (np.asarray(m.body_mass) > 0.0)
        ).astype(np.int32)
        inertia = np.asarray(m.body_inertia[inertia_bodies], np.float64)
        mass = np.asarray(m.body_mass[inertia_bodies], np.float64)
        inertia_sizes = np.sqrt(
            np.maximum(
                1.5
                * np.column_stack(
                    (
                        inertia[:, 1] + inertia[:, 2] - inertia[:, 0],
                        inertia[:, 0] + inertia[:, 2] - inertia[:, 1],
                        inertia[:, 0] + inertia[:, 1] - inertia[:, 2],
                    )
                )
                / mass[:, None],
                0.0,
            )
        )
        volume_scale = np.cbrt(mass / (8000.0 * np.prod(inertia_sizes, axis=1)))
        scaled_inertia_sizes = inertia_sizes * volume_scale[:, None]

        visual_types: list[int] = []
        visual_actuators: list[int] = []
        visual_sizes: list[np.ndarray] = []
        pose_types: list[int] = []
        pose_indices: list[int] = []
        slider_crank_actuators: list[int] = []
        primitive_types = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE): ActuatorVisualType.SPHERE,
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ActuatorVisualType.ELLIPSOID,
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): ActuatorVisualType.CAPSULE,
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): ActuatorVisualType.CYLINDER,
            int(mujoco.mjtGeom.mjGEOM_BOX): ActuatorVisualType.BOX,
        }

        def append_primitive(actuator: int, geom_type: int, size, pose_type: int, source: int):
            visual_type = primitive_types.get(int(geom_type))
            if visual_type is None:
                return
            raw = np.asarray(size, np.float32)
            if visual_type is ActuatorVisualType.SPHERE:
                normalized = np.full(3, raw[0], np.float32)
            elif visual_type in (ActuatorVisualType.CAPSULE, ActuatorVisualType.CYLINDER):
                normalized = np.array((raw[0], raw[0], raw[1]), np.float32)
            else:
                normalized = raw[:3].copy()
            visual_types.append(int(visual_type))
            visual_actuators.append(actuator)
            visual_sizes.append(1.05 * normalized)
            pose_types.append(pose_type)
            pose_indices.append(source)

        joint_transmissions = {
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        for actuator, transmission in enumerate(np.asarray(m.actuator_trntype)):
            source = int(m.actuator_trnid[actuator, 0])
            transmission = int(transmission)
            if transmission in joint_transmissions:
                joint_type = int(m.jnt_type[source])
                if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    visual_type = ActuatorVisualType.SLIDE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    visual_type = ActuatorVisualType.HINGE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                    visual_type = ActuatorVisualType.BALL
                else:
                    visual_type = ActuatorVisualType.FREE
                if visual_type in (ActuatorVisualType.SLIDE, ActuatorVisualType.HINGE):
                    size = (
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorlength),
                    )
                    pose_type = _ACTUATOR_POSE_JOINT_AXIS
                else:
                    radius = meansize * float(m.vis.scale.jointlength) * 0.33
                    size = (radius, radius, radius)
                    pose_type = _ACTUATOR_POSE_JOINT_BODY
                visual_types.append(int(visual_type))
                visual_actuators.append(actuator)
                visual_sizes.append(np.asarray(size, np.float32))
                pose_types.append(pose_type)
                pose_indices.append(source)
            elif transmission == int(mujoco.mjtTrn.mjTRN_SITE):
                append_primitive(
                    actuator,
                    int(m.site_type[source]),
                    m.site_size[source],
                    _ACTUATOR_POSE_SITE,
                    source,
                )
            elif transmission == int(mujoco.mjtTrn.mjTRN_BODY):
                start = int(m.body_geomadr[source])
                stop = start + int(m.body_geomnum[source])
                for geom in range(start, stop):
                    append_primitive(
                        actuator,
                        int(m.geom_type[geom]),
                        m.geom_size[geom],
                        _ACTUATOR_POSE_GEOM,
                        geom,
                    )
            elif transmission == int(mujoco.mjtTrn.mjTRN_SLIDERCRANK):
                slider_crank_actuators.append(actuator)

        count = len(visual_types)
        self._actuator_visual_pose_types = np.asarray(pose_types, np.uint8)
        self._actuator_visual_pose_indices = np.asarray(pose_indices, np.int32)
        self._slider_crank_actuators = np.asarray(slider_crank_actuators, np.int32)
        self._diagnostic_frame.actuator_xpos = np.zeros((count, 3), np.float32)
        self._diagnostic_frame.actuator_xmat = np.zeros((count, 3, 3), np.float32)
        slider_crank_count = len(slider_crank_actuators)
        self._diagnostic_frame.slider_crank_points = np.zeros(
            (slider_crank_count, 3, 3), np.float32
        )
        self._diagnostic_frame.slider_crank_broken = np.zeros(slider_crank_count, bool)
        autoconnect_count = sum(
            1 + int(m.body_jntnum[body])
            for body in range(1, m.nbody)
            if int(m.body_parentid[body]) != 0
        )
        self._diagnostic_frame.autoconnect_segments = np.zeros(
            (autoconnect_count, 2, 3), np.float32
        )
        bvh_count = len(bvh_type)
        self._diagnostic_frame.bvh_centers = np.zeros((bvh_count, 3), np.float32)
        self._diagnostic_frame.bvh_matrices = np.zeros((bvh_count, 3, 3), np.float32)
        self._diagnostic_frame.bvh_sizes = self._bvh_local_size.copy()
        self._diagnostic_frame.bvh_active = np.zeros(bvh_count, bool)
        self._diagnostic_frame.bvh_control_segments = np.zeros(
            (bvh_control_count, 2, 3), np.float32
        )

        return DiagnosticSource(
            joint_types=joint_types,
            joint_visible=self._group_visibility(m.jnt_group, "joint"),
            joint_length=meansize * float(m.vis.scale.jointlength),
            joint_width=meansize * float(m.vis.scale.jointwidth),
            joint_rgba=np.asarray(m.vis.rgba.joint, np.float32).copy(),
            com_bodies=com_bodies,
            com_radius=meansize * float(m.vis.scale.com),
            com_rgba=np.asarray(m.vis.rgba.com, np.float32).copy(),
            inertia_bodies=inertia_bodies,
            inertia_sizes=np.asarray(inertia_sizes, np.float32),
            scaled_inertia_sizes=np.asarray(scaled_inertia_sizes, np.float32),
            inertia_rgba=np.asarray(m.vis.rgba.inertia, np.float32).copy(),
            actuator_visual_types=np.asarray(visual_types, np.uint8),
            actuator_visual_actuators=np.asarray(visual_actuators, np.int32),
            actuator_visual_sizes=(
                np.stack(visual_sizes) if count else np.zeros((0, 3), np.float32)
            ),
            slider_crank_actuators=self._slider_crank_actuators.copy(),
            slider_crank_width=meansize * float(m.vis.scale.slidercrank),
            slider_crank_rgba=np.asarray(m.vis.rgba.slidercrank, np.float32).copy(),
            slider_crank_broken_rgba=np.asarray(m.vis.rgba.crankbroken, np.float32).copy(),
            camera_rgba=np.asarray(m.vis.rgba.camera, np.float32).copy(),
            light_rgba=np.asarray(m.vis.rgba.light, np.float32).copy(),
            rangefinder_rgba=np.asarray(m.vis.rgba.rangefinder, np.float32).copy(),
            rangefinder_normal_length=meansize * 0.25,
            constraint_radius=meansize * float(m.vis.scale.constraint),
            constraint_connect_rgba=np.asarray(m.vis.rgba.connect, np.float32).copy(),
            constraint_rgba=np.asarray(m.vis.rgba.constraint, np.float32).copy(),
            contact_point_rgba=np.asarray(m.vis.rgba.contactpoint, np.float32).copy(),
            contact_force_rgba=np.asarray(m.vis.rgba.contactforce, np.float32).copy(),
            contact_friction_rgba=np.asarray(m.vis.rgba.contactfriction, np.float32).copy(),
            contact_force_scale=float(m.vis.map.force) / meanmass if meanmass > 0.0 else 0.0,
            autoconnect_width=meansize * float(m.vis.scale.connect),
            autoconnect_rgba=np.asarray(m.vis.rgba.connect, np.float32).copy(),
            bvh_type=bvh_type,
            bvh_depth=bvh_depth,
            bvh_leaf=bvh_leaf,
            bvh_active_highlight=bool(m.vis.global_.bvactive),
            bvh_rgba=bvh_rgba.copy(),
            bvh_active_rgba=bvh_active_rgba.copy(),
            bvh_control_count=bvh_control_count,
        )

    def _fill_constraint_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        visible = diagnostics.constraint_visible
        visible.fill(False)
        connect = int(mujoco.mjtEq.mjEQ_CONNECT)
        weld = int(mujoco.mjtEq.mjEQ_WELD)
        site = int(mujoco.mjtObj.mjOBJ_SITE)
        for equality in np.flatnonzero(np.asarray(d.eq_active)):
            constraint_type = int(m.eq_type[equality])
            if constraint_type not in (connect, weld):
                continue
            first = int(m.eq_obj1id[equality])
            second = int(m.eq_obj2id[equality])
            if int(m.eq_objtype[equality]) == site:
                diagnostics.constraint_starts[equality] = d.site_xpos[first]
                diagnostics.constraint_ends[equality] = d.site_xpos[second]
            else:
                data = m.eq_data[equality]
                start_offset = 3 if constraint_type == weld else 0
                end_offset = 0 if constraint_type == weld else 3
                diagnostics.constraint_starts[equality] = (
                    d.xpos[first]
                    + d.xmat[first].reshape(3, 3) @ data[start_offset : start_offset + 3]
                )
                diagnostics.constraint_ends[equality] = (
                    d.xpos[second]
                    + d.xmat[second].reshape(3, 3) @ data[end_offset : end_offset + 3]
                )
            visible[equality] = True

    def _fill_bvh_visuals(self, diagnostics: DiagnosticFrame) -> None:
        if not len(self._bvh_pose_type) and not len(self._bvh_control_body):
            return
        m, d = self._m, self._d
        centers = diagnostics.bvh_centers
        matrices = diagnostics.bvh_matrices
        np.copyto(diagnostics.bvh_sizes, self._bvh_local_size)

        body = self._bvh_pose_type == _BVH_POSE_BODY
        if np.any(body):
            source = self._bvh_pose_source[body]
            rotation = d.ximat[source].reshape(-1, 3, 3)
            matrices[body] = rotation
            centers[body] = d.xipos[source] + np.einsum(
                "nij,nj->ni", rotation, self._bvh_local_center[body]
            )

        geom = self._bvh_pose_type == _BVH_POSE_GEOM
        if np.any(geom):
            source = self._bvh_pose_source[geom]
            rotation = d.geom_xmat[source].reshape(-1, 3, 3)
            matrices[geom] = rotation
            centers[geom] = d.geom_xpos[source] + np.einsum(
                "nij,nj->ni", rotation, self._bvh_local_center[geom]
            )

        dynamic = self._bvh_pose_type == _BVH_POSE_DYNAMIC
        if np.any(dynamic):
            source = self._bvh_global_index[dynamic] - int(m.nbvhstatic)
            aabb = d.bvh_aabb_dyn[source]
            centers[dynamic] = aabb[:, :3]
            matrices[dynamic] = np.eye(3, dtype=np.float32)
            diagnostics.bvh_sizes[dynamic] = aabb[:, 3:]

        diagnostics.bvh_active.fill(False)
        indexed = self._bvh_global_index >= 0
        diagnostics.bvh_active[indexed] = d.bvh_active[self._bvh_global_index[indexed]]

        body = self._bvh_control_body
        if len(body):
            rotation = d.xmat[body].reshape(-1, 2, 3, 3)
            diagnostics.bvh_control_segments[:] = d.xpos[body] + np.einsum(
                "...ij,...j->...i", rotation, self._bvh_control_local
            )

    @staticmethod
    def _build_rangefinder_specs(model) -> tuple[_RangefinderSpec, ...]:
        sensor_type = int(mujoco.mjtSensor.mjSENS_RANGEFINDER)
        offset = 0
        specs = []
        for sensor in np.flatnonzero(np.asarray(model.sensor_type) == sensor_type):
            sensor = int(sensor)
            fields = int(model.sensor_intprm[sensor, 0])
            stride = sum(
                width for field, width in enumerate(_RAY_FIELD_WIDTHS) if fields & (1 << field)
            )
            ray_count = int(model.sensor_dim[sensor]) // stride
            specs.append(
                _RangefinderSpec(
                    sensor,
                    fields,
                    ray_count,
                    stride,
                    offset,
                    int(model.sensor_objtype[sensor]),
                    int(model.sensor_objid[sensor]),
                )
            )
            offset += ray_count
        return tuple(specs)

    def _fill_rangefinder_visuals(self, diagnostics: DiagnosticFrame) -> None:
        diagnostics.rangefinder_lines.fill(False)
        diagnostics.rangefinder_points.fill(False)
        diagnostics.rangefinder_normal_arrows.fill(False)
        for spec in self._rangefinder_specs:
            output = slice(spec.frame_offset, spec.frame_offset + spec.ray_count)
            data_start = int(self._m.sensor_adr[spec.sensor])
            values = self._d.sensordata[
                data_start : data_start + spec.ray_count * spec.stride
            ].reshape(spec.ray_count, spec.stride)
            fields = {}
            cursor = 0
            for field, width in enumerate(_RAY_FIELD_WIDTHS):
                bit = 1 << field
                if spec.fields & bit:
                    fields[bit] = values[:, cursor : cursor + width]
                    cursor += width

            origins, directions = self._rangefinder_rays(spec)
            if _RAY_ORIGIN in fields:
                origins = fields[_RAY_ORIGIN]
            if _RAY_DIR in fields:
                directions = fields[_RAY_DIR]
            diagnostics.rangefinder_starts[output] = origins

            distance = fields.get(_RAY_DIST)
            point = fields.get(_RAY_POINT)
            normal = fields.get(_RAY_NORMAL)
            if distance is not None:
                hit = distance[:, 0] >= 0.0
            elif point is not None:
                hit = np.linalg.norm(point, axis=1) > 1e-12
            else:
                hit = np.zeros(spec.ray_count, bool)

            ends = origins.copy()
            if point is not None:
                ends[hit] = point[hit]
            elif distance is not None:
                ends[hit] += directions[hit] * distance[hit]
            diagnostics.rangefinder_ends[output] = ends
            diagnostics.rangefinder_lines[output] = hit & bool(spec.fields & _RAY_DIST)
            diagnostics.rangefinder_points[output] = hit & bool(spec.fields & _RAY_POINT)

            if normal is not None:
                diagnostics.rangefinder_normals[output] = normal
                diagnostics.rangefinder_normal_arrows[output] = hit & (
                    np.linalg.norm(normal, axis=1) > 1e-12
                )

    def _rangefinder_rays(self, spec: _RangefinderSpec) -> tuple[np.ndarray, np.ndarray]:
        if spec.object_type == int(mujoco.mjtObj.mjOBJ_SITE):
            origin = np.asarray(self._d.site_xpos[spec.object_id], np.float32)
            rotation = np.asarray(self._d.site_xmat[spec.object_id], np.float32).reshape(3, 3)
            return (
                np.repeat(origin[None], spec.ray_count, axis=0),
                np.repeat(rotation[:, 2][None], spec.ray_count, axis=0),
            )

        width, height = (int(value) for value in self._m.cam_resolution[spec.object_id])
        view = self.camera_view(spec.object_id).with_aspect(width / height)
        columns = np.tile(np.arange(width), height)
        rows = np.repeat(np.arange(height), width)
        clip = np.column_stack(
            (
                2.0 * (columns + 0.5) / width - 1.0,
                1.0 - 2.0 * (rows + 0.5) / height,
                np.full(spec.ray_count, -1.0),
                np.ones(spec.ray_count),
            )
        )
        local = clip @ np.linalg.inv(view.proj_matrix()).T
        local = local[:, :3] / local[:, 3:4]
        rotation = np.asarray(self._d.cam_xmat[spec.object_id], np.float32).reshape(3, 3)
        if view.orthographic:
            origins = view.eye + local[:, :2] @ rotation[:, :2].T
            directions = np.repeat((-rotation[:, 2])[None], spec.ray_count, axis=0)
        else:
            origins = np.repeat(np.asarray(view.eye)[None], spec.ray_count, axis=0)
            directions = local @ rotation.T
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return origins.astype(np.float32), directions.astype(np.float32)

    def _fill_actuator_visual_poses(self, diagnostics: DiagnosticFrame) -> None:
        d, m = self._d, self._m
        for record, (pose_type, source) in enumerate(
            zip(self._actuator_visual_pose_types, self._actuator_visual_pose_indices, strict=True)
        ):
            source = int(source)
            if pose_type == _ACTUATOR_POSE_JOINT_AXIS:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = self._axis_rotation(d.xaxis[source])
            elif pose_type == _ACTUATOR_POSE_JOINT_BODY:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = d.xmat[int(m.jnt_bodyid[source])].reshape(3, 3)
            elif pose_type == _ACTUATOR_POSE_SITE:
                diagnostics.actuator_xpos[record] = d.site_xpos[source]
                diagnostics.actuator_xmat[record] = d.site_xmat[source].reshape(3, 3)
            else:
                diagnostics.actuator_xpos[record] = d.geom_xpos[source]
                diagnostics.actuator_xmat[record] = d.geom_xmat[source].reshape(3, 3)

    def _fill_slider_crank_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        for record, actuator in enumerate(self._slider_crank_actuators):
            actuator = int(actuator)
            crank_site, slider_site = map(int, m.actuator_trnid[actuator])
            axis = d.site_xmat[slider_site].reshape(3, 3)[:, 2]
            offset = d.site_xpos[crank_site] - d.site_xpos[slider_site]
            axial = float(np.dot(offset, axis))
            rod = float(m.actuator_cranklength[actuator])
            determinant = axial * axial + rod * rod - float(np.dot(offset, offset))
            diagnostics.slider_crank_broken[record] = determinant < 0.0
            slider_length = axial - np.sqrt(max(determinant, 0.0))
            points = diagnostics.slider_crank_points[record]
            points[0] = d.site_xpos[slider_site]
            points[1] = points[0] + axis * slider_length
            points[2] = d.site_xpos[crank_site]

    def _fill_autoconnect_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        record = 0
        for body in range(1, m.nbody):
            parent = int(m.body_parentid[body])
            if parent == 0:
                continue
            current = d.xipos[body]
            start = int(m.body_jntadr[body])
            for joint in range(start + int(m.body_jntnum[body]) - 1, start - 1, -1):
                diagnostics.autoconnect_segments[record] = current, d.xanchor[joint]
                current = d.xanchor[joint]
                record += 1
            diagnostics.autoconnect_segments[record] = current, d.xipos[parent]
            record += 1

    @staticmethod
    def _axis_rotation(axis) -> np.ndarray:
        z = np.asarray(axis, np.float32)
        z = z / np.linalg.norm(z)
        reference = np.array((1.0, 0.0, 0.0), np.float32)
        if abs(float(z[0])) > 0.9:
            reference = np.array((0.0, 1.0, 0.0), np.float32)
        x = np.cross(reference, z)
        x /= np.linalg.norm(x)
        return np.column_stack((x, np.cross(z, x), z)).astype(np.float32)

    def _site_rgba(self, si: int, matid: int) -> np.ndarray:
        rgba = np.asarray(self._m.site_rgba[si], np.float32)
        if matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
            return np.asarray(self._m.mat_rgba[matid], np.float32).copy()
        return rgba.copy()

    def _geom_rgba(self, gi: int, matid: int) -> np.ndarray:
        rgba = np.asarray(self._m.geom_rgba[gi], np.float32)
        if matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
            return np.asarray(self._m.mat_rgba[matid], np.float32).copy()
        return rgba.copy()

    def _build_mesh(self, mesh_id: int) -> MeshData:
        m = self._m
        va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
        fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
        na = int(m.mesh_normaladr[mesh_id])
        ta, tn = int(m.mesh_texcoordadr[mesh_id]), int(m.mesh_texcoordnum[mesh_id])

        verts = np.asarray(m.mesh_vert[va : va + vn], np.float32)
        face = np.asarray(m.mesh_face[fa : fa + fn], np.int32)
        fnorm = np.asarray(m.mesh_facenormal[fa : fa + fn], np.int32)
        normals_all = np.asarray(
            m.mesh_normal[na : na + int(m.mesh_normalnum[mesh_id])], np.float32
        )

        has_uv = ta >= 0 and tn > 0
        if has_uv:
            uvs_all = np.asarray(m.mesh_texcoord[ta : ta + tn], np.float32)
            ftex = np.asarray(m.mesh_facetexcoord[fa : fa + fn], np.int32)
        else:
            uvs_all = np.zeros((0, 2), np.float32)
            ftex = face

        aligned = np.array_equal(fnorm, face) and (not has_uv or np.array_equal(ftex, face))
        aligned = aligned and len(normals_all) == vn and (not has_uv or tn == vn)
        if aligned:
            uvs = uvs_all.copy() if has_uv else np.zeros((vn, 2), np.float32)
            return MeshData(
                positions=verts.copy(),
                normals=normals_all.copy(),
                uvs=uvs,
                indices=face.reshape(-1).astype(np.uint32),
            )

        corner_v = face.reshape(-1)
        corner_n = fnorm.reshape(-1)
        positions = verts[corner_v]
        normals = normals_all[corner_n] if len(normals_all) else np.zeros_like(positions)
        if has_uv:
            uvs = uvs_all[ftex.reshape(-1)]
        else:
            uvs = np.zeros((len(positions), 2), np.float32)
        return MeshData(
            positions=np.ascontiguousarray(positions, np.float32),
            normals=np.ascontiguousarray(normals, np.float32),
            uvs=np.ascontiguousarray(uvs, np.float32),
            indices=np.arange(len(positions), dtype=np.uint32),
        )

    def _build_convex_hull(self, mesh_id: int) -> MeshData:
        m = self._m
        graph_adr = int(m.mesh_graphadr[mesh_id])
        vertex_count = int(m.mesh_graph[graph_adr])
        face_count = int(m.mesh_graph[graph_adr + 1])
        face_adr = graph_adr + 2 + 3 * vertex_count + 3 * face_count
        faces = np.asarray(m.mesh_graph[face_adr : face_adr + 3 * face_count], np.int32).reshape(
            -1, 3
        )

        vertex_adr = int(m.mesh_vertadr[mesh_id])
        vertices = np.asarray(
            m.mesh_vert[vertex_adr : vertex_adr + int(m.mesh_vertnum[mesh_id])], np.float32
        )
        triangles = vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(lengths, 1e-12)

        texcoord_adr = int(m.mesh_texcoordadr[mesh_id])
        texcoord_count = int(m.mesh_texcoordnum[mesh_id])
        if texcoord_adr >= 0 and texcoord_count > int(faces.max(initial=-1)):
            texcoords = np.asarray(
                m.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_count], np.float32
            )[faces]
        else:
            texcoords = np.zeros((face_count, 3, 2), np.float32)

        positions = triangles.reshape(-1, 3)
        normals = np.repeat(face_normals[:, None, :], 3, axis=1).reshape(-1, 3)
        return MeshData(
            positions=np.ascontiguousarray(positions, np.float32),
            normals=np.ascontiguousarray(normals, np.float32),
            uvs=np.ascontiguousarray(texcoords.reshape(-1, 2), np.float32),
            indices=np.arange(len(positions), dtype=np.uint32),
        )

    def _build_heightfield(self, field_id: int) -> MeshData:
        m = self._m
        rows = int(m.hfield_nrow[field_id])
        cols = int(m.hfield_ncol[field_id])
        adr = int(m.hfield_adr[field_id])
        height = float(m.hfield_size[field_id][2])
        base = float(m.hfield_size[field_id][3])
        z0 = -base / max(height, 1e-12)
        data = np.asarray(m.hfield_data[adr : adr + rows * cols], np.float32).reshape(rows, cols)

        positions: list[tuple[float, float, float]] = []
        uvs: list[tuple[float, float]] = []
        indices: list[int] = []

        def vertex(x: float, y: float, z: float, u: float, v: float) -> int:
            positions.append((x, y, z))
            uvs.append((u, v))
            return len(positions) - 1

        top = np.zeros((rows, cols), np.int32)
        for r in range(rows):
            v = r / max(rows - 1, 1)
            y = 2.0 * v - 1.0
            for c in range(cols):
                u = c / max(cols - 1, 1)
                top[r, c] = vertex(2.0 * u - 1.0, y, float(data[r, c]), u, v)
        for r in range(rows - 1):
            for c in range(cols - 1):
                a, b = int(top[r, c]), int(top[r, c + 1])
                d, e = int(top[r + 1, c]), int(top[r + 1, c + 1])
                indices += (a, b, e, a, e, d)

        boundary = [
            [(float(top[0, c]), c / (cols - 1)) for c in range(cols)],
            [(float(top[r, cols - 1]), r / (rows - 1)) for r in range(rows)],
            [(float(top[rows - 1, c]), 1.0 - c / (cols - 1)) for c in range(cols - 1, -1, -1)],
            [(float(top[r, 0]), 1.0 - r / (rows - 1)) for r in range(rows - 1, -1, -1)],
        ]
        pos = positions
        for edge in boundary:
            for (top_a, u0), (top_b, u1) in pairwise(edge):
                pa, pb = pos[int(top_a)], pos[int(top_b)]
                a = vertex(*pa, u0, 1.0)
                b = vertex(*pb, u1, 1.0)
                c = vertex(pb[0], pb[1], z0, u1, 0.0)
                d = vertex(pa[0], pa[1], z0, u0, 0.0)
                indices += (a, b, c, a, c, d)

        bottom = [
            vertex(-1.0, -1.0, z0, 0.0, 0.0),
            vertex(1.0, -1.0, z0, 1.0, 0.0),
            vertex(1.0, 1.0, z0, 1.0, 1.0),
            vertex(-1.0, 1.0, z0, 0.0, 1.0),
        ]
        indices += (bottom[0], bottom[2], bottom[1], bottom[0], bottom[3], bottom[2])

        p = np.asarray(positions, np.float32)
        idx = np.asarray(indices, np.uint32)
        tri = idx.reshape(-1, 3)
        face_n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        normals = np.zeros_like(p)
        for corner in range(3):
            np.add.at(normals, tri[:, corner], face_n)
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(length, 1e-12)
        return MeshData(p, normals, np.asarray(uvs, np.float32), idx)

    def _build_textures(self) -> dict[str, TextureData]:
        """Convert compiled MuJoCo textures to Forge texture arrays."""

        m = self._m
        out: dict[str, TextureData] = {}
        for ti in range(m.ntex):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, ti) or f"tex{ti}"
            w, h = int(m.tex_width[ti]), int(m.tex_height[ti])
            c = int(m.tex_nchannel[ti])
            adr = int(m.tex_adr[ti])
            raw = np.asarray(m.tex_data[adr : adr + w * h * c], np.uint8)
            ttype = int(m.tex_type[ti])
            if ttype == mujoco.mjtTexture.mjTEXTURE_CUBE:
                texture_type = TextureType.CUBE
            elif ttype == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                texture_type = TextureType.SKYBOX
            else:
                texture_type = TextureType.TWO_D
            if texture_type is TextureType.TWO_D:
                pixels = raw.reshape(h, w, c)
            elif h == w:
                # MuJoCo repeats a square cube texture on all six faces.
                face = raw.reshape(h, w, c)
                pixels = np.repeat(face[None, ...], 6, axis=0)
            else:
                # Separate cube faces are stored as a vertical 6-by-1 strip.
                pixels = raw.reshape(6, w, w, c)
            out[name] = TextureData(name=name, type=texture_type, pixels=pixels.copy(), srgb=True)
        return out

    def _build_materials(
        self, textures: dict[str, TextureData]
    ) -> tuple[list[Material], dict[int, int]]:
        m = self._m
        tex_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i) or f"tex{i}" for i in range(m.ntex)
        ]
        out: list[Material] = []
        index: dict[int, int] = {}
        for mi in range(m.nmat):
            texid = -1
            for role in (_TEXROLE_RGB, _TEXROLE_RGBA):
                cand = int(m.mat_texid[mi][role])
                if cand >= 0:
                    texid = cand
                    break
            tex = tex_names[texid] if 0 <= texid < len(tex_names) else None
            if tex is not None and tex not in textures:
                tex = None
            index[mi] = len(out)
            out.append(
                Material(
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mi) or f"mat{mi}",
                    rgba=np.asarray(m.mat_rgba[mi], np.float32).copy(),
                    emission=float(m.mat_emission[mi]),
                    specular=float(m.mat_specular[mi]),
                    shininess=float(m.mat_shininess[mi]),
                    reflectance=float(m.mat_reflectance[mi]),
                    metallic=float(m.mat_metallic[mi]),
                    roughness=float(m.mat_roughness[mi]),
                    texture=tex,
                    tex_repeat=np.asarray(m.mat_texrepeat[mi], np.float32).copy(),
                    tex_uniform=bool(m.mat_texuniform[mi]),
                )
            )

        index[-1] = len(out)
        out.append(Material(name="__geom__"))
        return out, index

    def _light(self, i: int, pos: np.ndarray, direction: np.ndarray) -> Light:
        m = self._m
        ltype = int(m.light_type[i])

        # MuJoCo values: spot=0, directional=1, point=2, image=3.

        if i < len(self._area_lights) and self._area_lights[i]:
            light_type = LightType.AREA
        elif ltype == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
            light_type = LightType.DIRECTIONAL
        elif ltype == mujoco.mjtLightType.mjLIGHT_POINT:
            light_type = LightType.POINT
        elif ltype == mujoco.mjtLightType.mjLIGHT_SPOT:
            light_type = LightType.SPOT
        else:
            light_type = LightType.IMAGE
        texid = int(m.light_texid[i])
        texture = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, texid) if texid >= 0 else None
        return Light(
            type=light_type,
            position=np.asarray(pos, np.float32).copy(),
            direction=np.asarray(direction, np.float32).copy(),
            diffuse=np.asarray(m.light_diffuse[i], np.float32).copy(),
            specular=np.asarray(m.light_specular[i], np.float32).copy(),
            ambient=np.asarray(m.light_ambient[i], np.float32).copy(),
            attenuation=np.asarray(m.light_attenuation[i], np.float32).copy(),
            range=float(m.light_range[i]),
            area_radius=float(m.light_bulbradius[i]),
            cutoff=float(m.light_cutoff[i]),
            exponent=float(m.light_exponent[i]),
            texture=texture,
            intensity=float(m.light_intensity[i]),
            cast_shadow=bool(m.light_castshadow[i]),
            active=bool(m.light_active[i]),
        )

    def _build_lights(self) -> LightSet:
        d = self._d
        return self._light_set(
            tuple(self._light(i, d.light_xpos[i], d.light_xdir[i]) for i in range(self._m.nlight))
        )

    def _dynamic_lights(self) -> LightSet:
        d = self._d
        preview = self._model_transform_preview
        lights = []
        for i in range(self._m.nlight):
            position = d.light_xpos[i]
            direction = d.light_xdir[i]
            if preview is not None and preview.light_mask[i]:
                position = preview.position + preview.delta_rotation @ (
                    np.asarray(position, np.float64) - preview.previous_position
                )
                direction = preview.delta_rotation @ np.asarray(direction, np.float64)
            lights.append(self._light(i, position, direction))
        return self._light_set(tuple(lights))

    def _light_set(self, lights: tuple[Light, ...]) -> LightSet:
        m = self._m
        extent = float(m.stat.extent) or 1.0
        ambient = _numeric_values(m, _FORGE_AMBIENT_NUMERIC)
        haze = _numeric_values(m, _FORGE_HAZE_NUMERIC)
        return LightSet(
            lights=lights,
            headlight=self._headlight(),
            ambient=(
                np.asarray(ambient[:3], np.float32).copy()
                if ambient is not None and len(ambient) >= 3
                else self._global_ambient()
            ),
            fog_color=np.asarray(m.vis.rgba.fog[:3], np.float32).copy(),
            fog_start=float(m.vis.map.fogstart) * extent,
            fog_end=float(m.vis.map.fogend) * extent,
            haze_color=np.asarray(m.vis.rgba.haze[:3], np.float32).copy(),
            haze_density=float(m.vis.map.haze),
            horizon_haze=bool(haze[0]) if haze is not None and len(haze) >= 1 else True,
            horizon_haze_slices=(
                max(3, round(haze[1]))
                if haze is not None and len(haze) >= 2
                else max(3, int(m.vis.quality.numslices))
            ),
        )

    def _headlight(self) -> Light | None:
        hl = self._m.vis.headlight
        if not bool(hl.active):
            return None
        return Light(
            type=DEFAULT_HEADLIGHT.type,
            diffuse=np.asarray(hl.diffuse, np.float32).copy(),
            specular=np.asarray(hl.specular, np.float32).copy(),
            ambient=np.asarray(hl.ambient, np.float32).copy(),
            cast_shadow=DEFAULT_HEADLIGHT.cast_shadow,
        )

    def _global_ambient(self) -> np.ndarray:
        total = np.zeros(3, np.float32)
        hl = self._m.vis.headlight
        if bool(hl.active):
            total += np.asarray(hl.ambient, np.float32)
        if self._m.nlight:
            lights = np.asarray(self._m.light_ambient, np.float32)
            active = np.asarray(self._m.light_active, np.float32).reshape(-1, 1)
            total += (lights * active).sum(axis=0)
        return np.clip(total, 0.0, 1.0)

    def _build_nodes(self) -> list[SceneNode]:
        m = self._m
        nodes: list[SceneNode] = []
        body_node: dict[int, int] = {}
        self._node_body = {}
        self._node_element = {}
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}

        body_parent = np.asarray(m.body_parentid, np.int32)
        has_child = np.zeros(m.nbody, bool)
        if m.nbody > 1:
            has_child[body_parent[1:]] = True
        has_kinematic_dof = np.zeros(m.nbody, bool)
        for body in range(1, m.nbody):
            has_kinematic_dof[body] = (
                bool(m.body_jntnum[body]) or has_kinematic_dof[body_parent[body]]
            )

        def add(name: str, node_type: NodeType, parent: int, body: int, **kw) -> int:
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id=node_id,
                    name=name,
                    type=node_type,
                    parent=parent,
                    body_index=body,
                    **kw,
                )
            )
            if parent >= 0:
                nodes[parent].children.append(node_id)
            self._node_body[node_id] = body
            return node_id

        world_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 0) or "world"
        body_node[0] = add(world_name, NodeType.WORLD, -1, 0, object_id=0)
        self._node_element[body_node[0]] = (0, NodeType.WORLD, world_name)
        model_parents = {
            item.model_id: add(
                item.name,
                NodeType.MODEL,
                body_node[0],
                -1,
                model_id=item.model_id,
                object_id=MODEL_OBJECT_BASE + item.model_id,
                posable=True,
            )
            for item in self._attached_models
        }
        self._node_model = {node_id: model_id for model_id, node_id in model_parents.items()}
        for model_id, node_id in model_parents.items():
            self._node_element[node_id] = (model_id, NodeType.MODEL, "")

        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            parent = int(body_parent[b])
            node_type = NodeType.ROBOT if parent == 0 and has_child[b] else NodeType.LINK
            model_id, raw_name = self._model_element_name(name, mujoco.mjtObj.mjOBJ_BODY)
            parent_node = body_node[parent]
            if parent == 0 and model_id in model_parents:
                parent_node = model_parents[model_id]
            body_node[b] = add(
                name,
                node_type,
                parent_node,
                b,
                object_id=b,
                posable=self._is_posable_body(b) or (model_id >= 0 and not has_kinematic_dof[b]),
            )
            nodes[body_node[b]].model_id = model_id
            self._node_element[body_node[b]] = (model_id, node_type, raw_name)

        for b in range(m.nbody):
            parent = body_node[b]
            adr, num = int(m.body_geomadr[b]), int(m.body_geomnum[b])
            for gi in range(adr, adr + num):
                if not self._visual_groups["geom"][int(m.geom_group[gi])]:
                    continue
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom{gi}"
                model_id, raw_name = self._model_element_name(gname, mujoco.mjtObj.mjOBJ_GEOM)
                is_plane = int(m.geom_type[gi]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                is_infinite_plane = is_plane and (
                    float(m.geom_size[gi, 0]) == 0.0 or float(m.geom_size[gi, 1]) == 0.0
                )
                object_id = (
                    self._geometry_object_id(model_id, raw_name)
                    if b == 0 and is_plane and not is_infinite_plane
                    else 0
                )
                self._geom_nodes[gi] = add(
                    gname,
                    NodeType.GEOM,
                    parent,
                    b,
                    object_id=object_id,
                    geom_index=gi,
                    posable=True,
                )
                nodes[self._geom_nodes[gi]].model_id = model_id
                self._node_element[self._geom_nodes[gi]] = (
                    model_id,
                    NodeType.GEOM,
                    raw_name,
                )
            ja, jn = int(m.body_jntadr[b]), int(m.body_jntnum[b])
            for ji in range(ja, ja + jn):
                if not self._visual_groups["joint"][int(m.jnt_group[ji])]:
                    continue
                jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}"
                node_id = add(jname, NodeType.JOINT, parent, b, joint_index=ji)
                model_id, raw_name = self._model_element_name(jname, mujoco.mjtObj.mjOBJ_JOINT)
                nodes[node_id].model_id = model_id
                self._node_element[node_id] = (model_id, NodeType.JOINT, raw_name)

        for li in range(m.nlight):
            b = int(m.light_bodyid[li])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, li) or f"light{li}"
            node_id = add(
                name,
                NodeType.LIGHT,
                body_node[b],
                b,
                object_id=LIGHT_OBJECT_BASE + li,
                visible=bool(m.light_active[li]),
                light_index=li,
            )
            model_id, raw_name = self._model_element_name(name, mujoco.mjtObj.mjOBJ_LIGHT)
            nodes[node_id].model_id = model_id
            self._node_element[node_id] = (model_id, NodeType.LIGHT, raw_name)
        for ci in range(m.ncam):
            b = int(m.cam_bodyid[ci])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, ci) or f"camera{ci}"
            node_id = add(
                name,
                NodeType.CAMERA,
                body_node[b],
                b,
                object_id=CAMERA_OBJECT_BASE + ci,
                camera_index=ci,
            )
            model_id, raw_name = self._model_element_name(name, mujoco.mjtObj.mjOBJ_CAMERA)
            nodes[node_id].model_id = model_id
            self._node_element[node_id] = (model_id, NodeType.CAMERA, raw_name)
        for si in range(m.nsite):
            if not self._visual_groups["site"][int(m.site_group[si])]:
                continue
            b = int(m.site_bodyid[si])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, si) or f"site{si}"
            model_id, raw_name = self._model_element_name(name, mujoco.mjtObj.mjOBJ_SITE)
            self._site_nodes[si] = add(
                name,
                NodeType.SITE,
                body_node[b],
                b,
                site_index=si,
                posable=model_id >= 0 and not has_kinematic_dof[b],
            )
            nodes[self._site_nodes[si]].model_id = model_id
            self._node_element[self._site_nodes[si]] = (model_id, NodeType.SITE, raw_name)
        for fi in range(m.nflex):
            if not self._visual_groups["flex"][int(m.flex_group[fi])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fi) or f"flex{fi}"
            self._flex_nodes[fi] = add(name, NodeType.FLEX, body_node[0], 0, object_id=m.nbody + fi)
        for si in range(m.nskin):
            if not self._visual_groups["skin"][int(m.skin_group[si])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SKIN, si) or f"skin{si}"
            self._skin_nodes[si] = add(
                name,
                NodeType.SKIN,
                body_node[0],
                0,
                object_id=m.nbody + m.nflex + si,
            )
        return nodes

    def _rebuild_model_element_names(self) -> None:
        """Index compiled model names by owning editable MjSpec in O(E)."""
        collections = (
            (mujoco.mjtObj.mjOBJ_BODY, "bodies"),
            (mujoco.mjtObj.mjOBJ_GEOM, "geoms"),
            (mujoco.mjtObj.mjOBJ_JOINT, "joints"),
            (mujoco.mjtObj.mjOBJ_SITE, "sites"),
            (mujoco.mjtObj.mjOBJ_CAMERA, "cameras"),
            (mujoco.mjtObj.mjOBJ_LIGHT, "lights"),
            (mujoco.mjtObj.mjOBJ_MATERIAL, "materials"),
            (mujoco.mjtObj.mjOBJ_TEXTURE, "textures"),
            (mujoco.mjtObj.mjOBJ_HFIELD, "hfields"),
            (mujoco.mjtObj.mjOBJ_KEY, "keys"),
        )
        index: dict[tuple[int, str], tuple[int, str]] = {}
        specs = [(item.model_id, item.prefix, item.spec) for item in self._attached_models]
        if self._root_spec is not None:
            # Root names win if they merely resemble an attached-model prefix.
            specs.append((0, "", self._root_spec))
        for model_id, prefix, spec in specs:
            for object_type, collection_name in collections:
                for element in getattr(spec, collection_name, ()):
                    raw_name = str(element.name or "")
                    if raw_name:
                        index[(int(object_type), f"{prefix}{raw_name}")] = (
                            int(model_id),
                            raw_name,
                        )
        self._model_element_names = index

    def _model_element_name(self, compiled_name: str, object_type) -> tuple[int, str]:
        name = str(compiled_name)
        return self._model_element_names.get((int(object_type), name), (0, name))

    def _is_free_body(self, body: int) -> bool:
        m = self._m
        adr, num = int(m.body_jntadr[body]), int(m.body_jntnum[body])
        return num == 1 and int(m.jnt_type[adr]) == mujoco.mjtJoint.mjJNT_FREE

    def _is_posable_body(self, body: int) -> bool:
        return int(self._m.body_mocapid[body]) >= 0 or self._is_free_body(body)

    def joints(self) -> list[JointInfo]:
        m = self._m
        joint_types = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        dofs = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}
        out = []
        for ji in range(m.njnt):
            joint_type = joint_types.get(int(m.jnt_type[ji]), "hinge")
            out.append(
                JointInfo(
                    joint_id=ji,
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}",
                    type=joint_type,
                    limited=bool(m.jnt_limited[ji]),
                    range=(float(m.jnt_range[ji][0]), float(m.jnt_range[ji][1])),
                    qpos_adr=int(m.jnt_qposadr[ji]),
                    qvel_adr=int(m.jnt_dofadr[ji]),
                    dof=dofs[joint_type],
                    body=int(m.jnt_bodyid[ji]),
                    axis=tuple(float(value) for value in m.jnt_axis[ji]),
                    damping=float(m.dof_damping[int(m.jnt_dofadr[ji])]),
                    stiffness=float(m.jnt_stiffness[ji]),
                )
            )
        return out

    def set_joint_properties(
        self,
        joint_id: int,
        axis: np.ndarray,
        limited: bool,
        value_range: tuple[float, float],
        damping: float,
        stiffness: float,
    ) -> bool:
        joint = int(joint_id)
        if not 0 <= joint < self._m.njnt:
            return False
        joint_type = int(self._m.jnt_type[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return False
        values = np.asarray(value_range, np.float64).reshape(2)
        axis = np.asarray(axis, np.float64).reshape(3)
        damping = float(damping)
        stiffness = float(stiffness)
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(axis))
            or not np.isfinite((damping, stiffness)).all()
            or damping < 0.0
            or stiffness < 0.0
        ):
            return False
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            if bool(limited) and values[1] <= 0.0:
                return False
            values[0] = 0.0
        elif bool(limited) and values[1] <= values[0]:
            return False
        if joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                return False
            axis /= norm

        node = next((item for item in self.nodes() if item.joint_index == joint), None)
        identity = self._node_element.get(node.node_id) if node is not None else None
        if identity is None:
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        spec = self._spec_for_model(model_id)
        if element is None or spec is None:
            return False

        if joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            element.axis = axis
            self._m.jnt_axis[joint] = axis
        element.limited = bool(limited)
        authored_range = values.copy()
        if bool(spec.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            authored_range = np.degrees(authored_range)
        element.range = authored_range
        authored_damping = np.asarray(element.damping, np.float64).copy()
        authored_stiffness = np.asarray(element.stiffness, np.float64).copy()
        authored_damping[0] = damping
        authored_stiffness[0] = stiffness
        element.damping = authored_damping
        element.stiffness = authored_stiffness

        self._m.jnt_limited[joint] = bool(limited)
        self._m.jnt_range[joint] = values
        dof_address = int(self._m.jnt_dofadr[joint])
        dof_count = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        self._m.dof_damping[dof_address : dof_address + dof_count] = damping
        self._m.jnt_stiffness[joint] = stiffness
        mujoco.mj_setConst(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)
        self._mark_model_edited(model_id)
        self._structure_revision += 1
        return True

    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None:
        joint = int(joint_id)
        if not 0 <= joint < self._m.njnt:
            return None
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_JOINT)
        if not name:
            return None
        spec = self._spec_for_model(model_id)
        element = spec.joint(name) if spec is not None else None
        if spec is None or element is None:
            return None
        reference = float(element.ref)
        spring_reference = float(element.springref)
        joint_type = int(self._m.jnt_type[joint])
        if bool(spec.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            reference = float(np.radians(reference))
            spring_reference = float(np.radians(spring_reference))
        force_limit_modes = {
            int(mujoco.mjtLimited.mjLIMITED_AUTO): "auto",
            int(mujoco.mjtLimited.mjLIMITED_FALSE): "unlimited",
            int(mujoco.mjtLimited.mjLIMITED_TRUE): "limited",
        }
        return JointAdvancedProperties(
            joint_id=joint,
            group=int(element.group),
            armature=float(element.armature),
            friction_loss=float(element.frictionloss),
            reference=reference,
            spring_reference=spring_reference,
            margin=float(element.margin),
            limit_solver_reference=tuple(float(value) for value in element.solref_limit),
            limit_solver_impedance=tuple(float(value) for value in element.solimp_limit),
            friction_solver_reference=tuple(float(value) for value in element.solref_friction),
            friction_solver_impedance=tuple(float(value) for value in element.solimp_friction),
            actuator_force_limit_mode=force_limit_modes[int(element.actfrclimited)],
            actuator_force_range=tuple(float(value) for value in self._m.jnt_actfrcrange[joint]),
            actuator_gravity_compensation=bool(element.actgravcomp),
        )

    def set_joint_advanced_properties(self, properties: JointAdvancedProperties) -> bool:
        joint = int(properties.joint_id)
        if not 0 <= joint < self._m.njnt:
            return False
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_JOINT)
        if not name:
            return False
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.joint(name)
        if element is None:
            return False
        reference = float(properties.reference)
        spring_reference = float(properties.spring_reference)
        joint_type = int(self._m.jnt_type[joint])
        if bool(working.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            reference = float(np.degrees(reference))
            spring_reference = float(np.degrees(spring_reference))
        element.group = int(properties.group)
        element.armature = float(properties.armature)
        element.frictionloss = float(properties.friction_loss)
        element.ref = reference
        element.springref = spring_reference
        element.margin = float(properties.margin)
        element.solref_limit = properties.limit_solver_reference
        element.solimp_limit = properties.limit_solver_impedance
        element.solref_friction = properties.friction_solver_reference
        element.solimp_friction = properties.friction_solver_impedance
        element.actfrclimited = {
            "auto": mujoco.mjtLimited.mjLIMITED_AUTO,
            "unlimited": mujoco.mjtLimited.mjLIMITED_FALSE,
            "limited": mujoco.mjtLimited.mjLIMITED_TRUE,
        }[properties.actuator_force_limit_mode]
        element.actfrcrange = properties.actuator_force_range
        element.actgravcomp = bool(properties.actuator_gravity_compensation)
        return self._replace_model_spec(model_id, working)

    def site_properties(self, node_id: int) -> SiteProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.SITE
            or not 0 <= node.site_index < self._m.nsite
        ):
            return None
        model_id, _node_type, name = identity
        spec = self._spec_for_model(model_id)
        element = spec.site(name) if spec is not None else None
        if element is None:
            return None
        type_names = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        }
        site_type = type_names.get(int(self._m.site_type[node.site_index]))
        if site_type is None:
            return None
        authored_from_to = np.asarray(element.fromto, np.float64).reshape(6)
        use_from_to = bool(np.all(np.isfinite(authored_from_to)))
        if use_from_to:
            from_to = authored_from_to
        else:
            center = np.asarray(element.pos, np.float64).reshape(3)
            axis = math3d.quat_to_mat3(element.quat)[:, 2]
            half_length = (
                float(self._m.site_size[node.site_index, 1])
                if site_type in ("capsule", "cylinder")
                else 0.5
            )
            from_to = np.concatenate((center - axis * half_length, center + axis * half_length))
        return SiteProperties(
            node_id=int(node_id),
            type=site_type,
            group=int(element.group),
            use_from_to=use_from_to,
            from_to=tuple(float(value) for value in from_to),
        )

    def set_site_properties(self, properties: SiteProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.SITE
            or not 0 <= node.site_index < self._m.nsite
        ):
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.site(name)
        if element is None:
            return False
        site_types = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
        }
        site_type = site_types.get(str(properties.type))
        if site_type is None:
            return False
        element.type = site_type
        element.group = int(properties.group)
        if properties.use_from_to:
            if str(properties.type) not in ("capsule", "cylinder"):
                return False
            element.fromto = np.asarray(properties.from_to, np.float64).reshape(6)
        else:
            index = int(node.site_index)
            element.fromto = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            element.pos = np.asarray(self._m.site_pos[index], np.float64)
            element.quat = np.asarray(self._m.site_quat[index], np.float64)
            element.size = np.asarray(self._m.site_size[index], np.float64)
        return self._replace_model_spec(model_id, working)

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        node = self._node_for_id(node_id)
        if node is None or node.type is not NodeType.GEOM or node.geom_index < 0:
            return None
        geom = int(node.geom_index)
        return GeometryProperties(
            node_id=int(node_id),
            friction=tuple(float(value) for value in self._m.geom_friction[geom]),
            collision_type_mask=int(self._m.geom_contype[geom]),
            collision_affinity_mask=int(self._m.geom_conaffinity[geom]),
            contact_dimension=int(self._m.geom_condim[geom]),
            contact_priority=int(self._m.geom_priority[geom]),
            margin=float(self._m.geom_margin[geom]),
            gap=float(self._m.geom_gap[geom]),
            solver_mix=float(self._m.geom_solmix[geom]),
            solver_reference=tuple(float(value) for value in self._m.geom_solref[geom]),
            solver_impedance=tuple(float(value) for value in self._m.geom_solimp[geom]),
            adhesion=float(self._m.geom_adhesion[geom]),
            surface_velocity=tuple(float(value) for value in self._m.geom_surfacevel[geom]),
        )

    def set_geometry_properties(self, properties: GeometryProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return False
        try:
            friction = np.asarray(properties.friction, np.float64).reshape(3)
            masks = (
                int(properties.collision_type_mask),
                int(properties.collision_affinity_mask),
            )
            contact_dimension = int(properties.contact_dimension)
            contact_priority = int(properties.contact_priority)
            values = np.asarray(
                (properties.margin, properties.gap, properties.solver_mix), np.float64
            )
            solver_reference = np.asarray(properties.solver_reference, np.float64).reshape(2)
            solver_impedance = np.asarray(properties.solver_impedance, np.float64).reshape(5)
            adhesion = float(properties.adhesion)
            surface_velocity = np.asarray(properties.surface_velocity, np.float64).reshape(6)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not np.all(np.isfinite(friction))
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(solver_reference))
            or not np.all(np.isfinite(solver_impedance))
            or not np.all(np.isfinite(surface_velocity))
            or not np.isfinite(adhesion)
            or np.any(friction < 0.0)
            or any(value < 0 or value > np.iinfo(np.int32).max for value in masks)
            or contact_dimension not in (1, 3, 4, 6)
            or not 0 <= contact_priority <= np.iinfo(np.int32).max
            or values[0] < 0.0
            or values[1] < 0.0
            or not 0.0 <= values[2] <= 1.0
            or adhesion < 0.0
            or not (np.all(solver_reference > 0.0) or np.all(solver_reference <= 0.0))
            or not 0.0 <= solver_impedance[0] <= solver_impedance[1] <= 1.0
            or solver_impedance[2] <= 0.0
            or not 0.0 <= solver_impedance[3] <= 1.0
            or solver_impedance[4] < 1.0
        ):
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False
        element.friction = friction
        element.contype = masks[0]
        element.conaffinity = masks[1]
        element.condim = contact_dimension
        element.priority = contact_priority
        element.margin = float(values[0])
        element.gap = float(values[1])
        element.solmix = float(values[2])
        element.solref = solver_reference
        element.solimp = solver_impedance
        element.adhesion = adhesion
        element.surfacevel = surface_velocity

        geom = int(node.geom_index)
        self._m.geom_friction[geom] = friction
        self._m.geom_contype[geom] = masks[0]
        self._m.geom_conaffinity[geom] = masks[1]
        self._m.geom_condim[geom] = contact_dimension
        self._m.geom_priority[geom] = contact_priority
        self._m.geom_margin[geom] = values[0]
        self._m.geom_gap[geom] = values[1]
        self._m.geom_solmix[geom] = values[2]
        self._m.geom_solref[geom] = solver_reference
        self._m.geom_solimp[geom] = solver_impedance
        self._m.geom_adhesion[geom] = adhesion
        self._m.geom_surfacevel[geom] = surface_velocity
        self._mark_model_edited(model_id)
        mujoco.mj_forward(self._m, self._d)
        self._structure_revision += 1
        return True

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return None
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return None
        authored_mass = float(element.mass)
        mass_mode = "mass" if np.isfinite(authored_mass) else "density"
        return GeometryAdvancedProperties(
            node_id=int(node_id),
            visual_group=int(element.group),
            mass_mode=mass_mode,
            mass=authored_mass if mass_mode == "mass" else 1.0,
            density=float(element.density),
            inertia_mode=(
                "shell"
                if int(element.typeinertia) == int(mujoco.mjtGeomInertia.mjINERTIA_SHELL)
                else "volume"
            ),
            fluid_ellipsoid=bool(float(element.fluid_ellipsoid) > 0.0),
            fluid_coefficients=tuple(float(value) for value in element.fluid_coefs),
        )

    def set_geometry_advanced_properties(self, properties: GeometryAdvancedProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if node is None or identity is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        edited = source_spec.copy()
        element = edited.geom(name)
        if element is None:
            return False
        mass_mode = str(properties.mass_mode).strip().lower()
        inertia_mode = str(properties.inertia_mode).strip().lower()
        if mass_mode not in ("density", "mass") or inertia_mode not in ("volume", "shell"):
            return False
        try:
            fluid_coefficients = np.asarray(properties.fluid_coefficients, np.float64).reshape(5)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not 0 <= int(properties.visual_group) < 6
            or not np.isfinite((properties.mass, properties.density, *fluid_coefficients)).all()
            or (mass_mode == "mass" and properties.mass <= 0.0)
            or (mass_mode == "density" and properties.density <= 0.0)
            or np.any(fluid_coefficients < 0.0)
        ):
            return False
        element.group = int(properties.visual_group)
        element.density = float(properties.density)
        element.mass = float(properties.mass) if mass_mode == "mass" else np.nan
        element.typeinertia = (
            mujoco.mjtGeomInertia.mjINERTIA_SHELL
            if inertia_mode == "shell"
            else mujoco.mjtGeomInertia.mjINERTIA_VOLUME
        )
        element.fluid_ellipsoid = 1.0 if properties.fluid_ellipsoid else 0.0
        element.fluid_coefs = fluid_coefficients
        return self._replace_model_spec(model_id, edited)

    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return None
        model_id, node_type, name = identity
        spec = self._spec_for_model(model_id)
        element = self._element(model_id, node_type.value, name)
        if spec is None or element is None:
            return None
        geom_types = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }
        geom_type = geom_types.get(int(element.type))
        if geom_type is None:
            return None
        resource_name = (
            str(element.meshname)
            if geom_type == "mesh"
            else str(element.hfieldname)
            if geom_type == "hfield"
            else ""
        )
        return GeometryShapeProperties(
            node_id=int(node_id),
            type=geom_type,
            resource_name=resource_name,
            mesh_names=tuple(str(item.name) for item in spec.meshes if item.name),
            height_field_names=tuple(str(item.name) for item in spec.hfields if item.name),
        )

    def set_geometry_shape(self, node_id: int, geom_type: str, resource_name: str) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        types = {
            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
            "hfield": mujoco.mjtGeom.mjGEOM_HFIELD,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "mesh": mujoco.mjtGeom.mjGEOM_MESH,
        }
        kind = str(geom_type).strip().lower()
        resource = str(resource_name).strip()
        if kind not in types:
            return False
        element = source_spec.geom(name)
        if element is None:
            return False
        if kind == "mesh" and source_spec.mesh(resource) is None:
            return False
        if kind == "hfield" and source_spec.hfield(resource) is None:
            return False
        size = np.asarray(element.size, np.float64).reshape(3).copy()
        defaults = {
            "plane": np.array((1.0, 1.0, 0.1)),
            "sphere": np.array((0.1, 0.1, 0.1)),
            "capsule": np.array((0.1, 0.2, 0.0)),
            "ellipsoid": np.array((0.1, 0.1, 0.1)),
            "cylinder": np.array((0.1, 0.2, 0.0)),
            "box": np.array((0.1, 0.1, 0.1)),
        }
        required = {
            "plane": (0, 1),
            "sphere": (0,),
            "capsule": (0, 1),
            "ellipsoid": (0, 1, 2),
            "cylinder": (0, 1),
            "box": (0, 1, 2),
        }.get(kind, ())
        if required and any(
            not np.isfinite(size[index]) or size[index] <= 0.0 for index in required
        ):
            size = defaults[kind]
        root, _xml = _component_xml(source_spec)
        target = next(
            (item for item in root.iter("geom") if str(item.attrib.get("name", "")) == name),
            None,
        )
        if target is None:
            return False
        target.set("type", kind)
        if kind == "mesh":
            target.set("mesh", resource)
        else:
            target.attrib.pop("mesh", None)
        if kind == "hfield":
            target.set("hfield", resource)
        else:
            target.attrib.pop("hfield", None)
        if required:
            target.set("size", _format_mjcf_values(size))
        else:
            target.attrib.pop("size", None)
        if kind not in ("capsule", "cylinder"):
            target.attrib.pop("fromto", None)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def import_model_geometry_resource(
        self, node_id: int, resource_type: str, path: Path, name: str
    ) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, element_name = identity
        source_spec = self._spec_for_model(model_id)
        source = Path(path).expanduser().resolve()
        kind = str(resource_type).strip().lower()
        value = str(name).strip()
        if (
            source_spec is None
            or not source.is_file()
            or kind not in ("mesh", "hfield")
            or not value
        ):
            return False
        if kind == "mesh" and source_spec.mesh(value) is not None:
            return False
        if kind == "hfield" and source_spec.hfield(value) is not None:
            return False
        working = source_spec.copy()
        if kind == "mesh":
            working.add_mesh(name=value, file=str(source))
        else:
            working.add_hfield(name=value, file=str(source), size=(1.0, 1.0, 1.0, 0.1))
        element = working.geom(element_name)
        if element is None:
            return False
        element.type = (
            mujoco.mjtGeom.mjGEOM_MESH if kind == "mesh" else mujoco.mjtGeom.mjGEOM_HFIELD
        )
        element.meshname = value if kind == "mesh" else ""
        element.hfieldname = value if kind == "hfield" else ""
        element.fromto = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self._replace_model_spec(model_id, working)

    def body_properties(self, node_id: int) -> BodyProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type not in (NodeType.LINK, NodeType.ROBOT)
            or not 0 < node.body_index < self._m.nbody
        ):
            return None
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return None
        body = int(node.body_index)
        authored_full = np.asarray(element.fullinertia, np.float64).reshape(6)
        if not bool(element.explicitinertial):
            inertia_mode = "auto"
        elif np.isfinite(authored_full[0]):
            inertia_mode = "full"
        else:
            inertia_mode = "diagonal"
        diagonal = np.asarray(self._m.body_inertia[body], np.float64)
        full = (
            authored_full
            if inertia_mode == "full"
            else np.array((*diagonal, 0.0, 0.0, 0.0), np.float64)
        )
        sleep_policies = {
            int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER): "never",
            int(mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED): "allowed",
            int(mujoco.mjtSleepPolicy.mjSLEEP_INIT): "init",
        }
        return BodyProperties(
            node_id=int(node_id),
            inertia_mode=inertia_mode,
            mass=float(self._m.body_mass[body]),
            inertial_position=tuple(float(value) for value in self._m.body_ipos[body]),
            inertial_quaternion=tuple(float(value) for value in self._m.body_iquat[body]),
            diagonal_inertia=tuple(float(value) for value in diagonal),
            full_inertia=tuple(float(value) for value in full),
            gravity_compensation=float(element.gravcomp),
            mocap=bool(element.mocap),
            sleep_policy=sleep_policies.get(int(element.sleep), "auto"),
        )

    def set_body_properties(self, properties: BodyProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type not in (NodeType.LINK, NodeType.ROBOT)
            or node.body_index <= 0
        ):
            return False
        model_id, node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        edited = source_spec.copy()
        lookup = "body" if node_type in (NodeType.LINK, NodeType.ROBOT) else node_type.value
        element = getattr(edited, lookup)(name)
        if element is None:
            return False
        mode = str(properties.inertia_mode).strip().lower()
        sleep_policy = str(properties.sleep_policy).strip().lower()
        if mode not in ("auto", "diagonal", "full") or sleep_policy not in (
            "auto",
            "never",
            "allowed",
            "init",
        ):
            return False
        element.gravcomp = float(properties.gravity_compensation)
        element.mocap = bool(properties.mocap)
        element.sleep = {
            "auto": mujoco.mjtSleepPolicy.mjSLEEP_AUTO,
            "never": mujoco.mjtSleepPolicy.mjSLEEP_NEVER,
            "allowed": mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED,
            "init": mujoco.mjtSleepPolicy.mjSLEEP_INIT,
        }[sleep_policy]
        if mode == "auto":
            # MjSpec retains explicit inertial values after toggling the flag. Clear every
            # sentinel-backed field so the next compile really derives inertia from geoms.
            element.explicitinertial = False
            element.mass = 0.0
            element.ipos = [np.nan, 0.0, 0.0]
            element.iquat = [1.0, 0.0, 0.0, 0.0]
            element.inertia = [0.0, 0.0, 0.0]
            element.fullinertia = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            quaternion = np.asarray(properties.inertial_quaternion, np.float64).reshape(4)
            quaternion /= np.linalg.norm(quaternion)
            element.explicitinertial = True
            element.mass = float(properties.mass)
            element.ipos = np.asarray(properties.inertial_position, np.float64).reshape(3)
            element.iquat = quaternion
            if mode == "diagonal":
                element.inertia = np.asarray(properties.diagonal_inertia, np.float64).reshape(3)
                element.fullinertia = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                element.inertia = [0.0, 0.0, 0.0]
                element.fullinertia = np.asarray(properties.full_inertia, np.float64).reshape(6)
        return self._replace_model_spec(model_id, edited)

    def actuators(self) -> list[ActuatorInfo]:
        m = self._m
        out = []
        for ai in range(m.nactuator):
            joint = -1
            if int(m.actuator_trntype[ai]) in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            ):
                joint = int(m.actuator_trnid[ai][0])
            out.append(
                ActuatorInfo(
                    actuator_id=ai,
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or f"act{ai}",
                    ctrl_range=(
                        float(m.actuator_ctrlrange[ai][0]),
                        float(m.actuator_ctrlrange[ai][1]),
                    ),
                    ctrl_limited=bool(m.actuator_ctrllimited[ai]),
                    ctrl_address=int(m.actuator_ctrladr[ai]),
                    ctrl_count=int(m.actuator_ctrlnum[ai]),
                    gain=float(m.actuator_gainprm[ai][0]),
                    joint=joint,
                )
            )
        return out

    def cameras(self) -> list[CameraInfo]:
        m = self._m
        return [
            CameraInfo(
                camera_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera{i}",
                object_id=CAMERA_OBJECT_BASE + i,
            )
            for i in range(m.ncam)
        ]

    def keyframes(self) -> list[KeyframeInfo]:
        m = self._m
        result = []
        for index in range(m.nkey):
            compiled_name = (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, index) or f"Key {index:03d}"
            )
            model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_KEY)
            result.append(
                KeyframeInfo(
                    keyframe_id=index,
                    name=name or compiled_name,
                    time=float(m.key_time[index]),
                    model_id=model_id,
                )
            )
        return result

    def _keyframe_identity(self, keyframe_id: int) -> tuple[int, str] | None:
        index = int(keyframe_id)
        if not 0 <= index < self._m.nkey:
            return None
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_KEY, index) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_KEY)
        return (model_id, name) if name else None

    def keyframe_properties(self, keyframe_id: int) -> KeyframeProperties | None:
        identity = self._keyframe_identity(keyframe_id)
        if identity is None:
            return None
        model_id, name = identity
        spec = self._spec_for_model(model_id)
        if spec is None or spec.key(name) is None:
            return None
        local_model = spec.copy().compile()
        local_key = mujoco.mj_name2id(local_model, mujoco.mjtObj.mjOBJ_KEY, name)
        if local_key < 0:
            return None
        return KeyframeProperties(
            keyframe_id=int(keyframe_id),
            model_id=model_id,
            name=name,
            time=float(local_model.key_time[local_key]),
            qpos=tuple(float(value) for value in local_model.key_qpos[local_key]),
            qvel=tuple(float(value) for value in local_model.key_qvel[local_key]),
            act=tuple(float(value) for value in local_model.key_act[local_key]),
            ctrl=tuple(float(value) for value in local_model.key_ctrl[local_key]),
            mocap_position=tuple(float(value) for value in local_model.key_mpos[local_key]),
            mocap_quaternion=tuple(float(value) for value in local_model.key_mquat[local_key]),
        )

    def _model_prefix(self, model_id: int) -> str:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        return item.prefix if item is not None else ""

    def _model_transform(self, model_id: int) -> tuple[np.ndarray, np.ndarray]:
        item = next(
            (item for item in self._attached_models if item.model_id == int(model_id)), None
        )
        if item is None:
            return np.zeros(3, np.float64), np.eye(3, dtype=np.float64)
        return item.position, item.rotation

    def _model_object_offset(self, model_id: int, local, object_type, count_name: str) -> int:
        """Map local object indices to the composed model without requiring names."""
        prefix = self._model_prefix(model_id)
        count = int(getattr(local, count_name))
        start = 1 if object_type == mujoco.mjtObj.mjOBJ_BODY else 0
        for local_index in range(start, count):
            name = mujoco.mj_id2name(local, object_type, local_index) or ""
            if not name:
                continue
            compiled_index = mujoco.mj_name2id(self._m, object_type, f"{prefix}{name}")
            if compiled_index >= 0:
                return compiled_index - local_index
        if int(model_id) == 0:
            return 0
        if self._root_spec is None:
            return -1
        offset = int(getattr(self._root_spec.copy().compile(), count_name))
        for item in self._attached_models:
            if item.model_id == int(model_id):
                return offset
            offset += int(getattr(item.spec.copy().compile(), count_name))
        return -1

    def _capture_model_keyframe_values(self, model_id: int):
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        local = spec.copy().compile()
        model_position, model_rotation = self._model_transform(model_id)
        joint_offset = self._model_object_offset(model_id, local, mujoco.mjtObj.mjOBJ_JOINT, "njnt")
        actuator_offset = self._model_object_offset(
            model_id, local, mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"
        )
        body_offset = self._model_object_offset(model_id, local, mujoco.mjtObj.mjOBJ_BODY, "nbody")
        qpos = np.asarray(local.qpos0, np.float64).copy()
        qvel = np.zeros(local.nv, np.float64)
        act = np.zeros(local.na, np.float64)
        ctrl = np.zeros(local.nu, np.float64)
        mocap_position = np.zeros((local.nmocap, 3), np.float64)
        mocap_quaternion = np.zeros((local.nmocap, 4), np.float64)
        if local.nmocap:
            mocap_quaternion[:, 0] = 1.0

        for local_joint in range(local.njnt):
            compiled_joint = local_joint + joint_offset
            if not 0 <= compiled_joint < self._m.njnt:
                continue
            local_qpos = self._span(local.jnt_qposadr, local_joint, local.nq)
            compiled_qpos = self._span(self._m.jnt_qposadr, compiled_joint, self._m.nq)
            local_qvel = self._span(local.jnt_dofadr, local_joint, local.nv)
            compiled_qvel = self._span(self._m.jnt_dofadr, compiled_joint, self._m.nv)
            if local_qpos.stop - local_qpos.start == compiled_qpos.stop - compiled_qpos.start:
                qpos[local_qpos] = self._d.qpos[compiled_qpos]
                if int(local.jnt_type[local_joint]) == mujoco.mjtJoint.mjJNT_FREE:
                    values = qpos[local_qpos]
                    values[:3] = (values[:3] - model_position) @ model_rotation
                    values[3:7] = math3d.mat3_to_quat(
                        model_rotation.T @ math3d.quat_to_mat3(values[3:7])
                    )
            if local_qvel.stop - local_qvel.start == compiled_qvel.stop - compiled_qvel.start:
                qvel[local_qvel] = self._d.qvel[compiled_qvel]

        for local_actuator in range(local.nu):
            compiled_actuator = local_actuator + actuator_offset
            if not 0 <= compiled_actuator < self._m.nu:
                continue
            local_ctrl = self._span(local.actuator_ctrladr, local_actuator, local.nu)
            compiled_ctrl = self._span(self._m.actuator_ctrladr, compiled_actuator, self._m.nu)
            if local_ctrl.stop - local_ctrl.start == compiled_ctrl.stop - compiled_ctrl.start:
                ctrl[local_ctrl] = self._d.ctrl[compiled_ctrl]
            local_activation = self._span(local.actuator_actadr, local_actuator, local.na)
            compiled_activation = self._span(self._m.actuator_actadr, compiled_actuator, self._m.na)
            if (
                local_activation.stop - local_activation.start
                == compiled_activation.stop - compiled_activation.start
            ):
                act[local_activation] = self._d.act[compiled_activation]

        for local_body in range(1, local.nbody):
            local_mocap = int(local.body_mocapid[local_body])
            if local_mocap < 0:
                continue
            compiled_body = local_body + body_offset
            compiled_mocap = int(self._m.body_mocapid[compiled_body]) if compiled_body >= 0 else -1
            if compiled_mocap >= 0:
                mocap_position[local_mocap] = (
                    self._d.mocap_pos[compiled_mocap] - model_position
                ) @ model_rotation
                mocap_quaternion[local_mocap] = math3d.mat3_to_quat(
                    model_rotation.T @ math3d.quat_to_mat3(self._d.mocap_quat[compiled_mocap])
                )
        return (
            qpos,
            qvel,
            act,
            ctrl,
            mocap_position.reshape(-1),
            mocap_quaternion.reshape(-1),
        )

    def add_model_keyframe(self, model_id: int, name: str) -> int:
        value = str(name).strip()
        source_spec = self._spec_for_model(model_id)
        values = self._capture_model_keyframe_values(model_id)
        if source_spec is None or values is None or not value or source_spec.key(value) is not None:
            return -1
        working = source_spec.copy()
        working.add_key(
            name=value,
            time=float(self._d.time),
            qpos=values[0],
            qvel=values[1],
            act=values[2],
            ctrl=values[3],
            mpos=values[4],
            mquat=values[5],
        )
        if not self._replace_model_spec(model_id, working):
            return -1
        return mujoco.mj_name2id(
            self._m,
            mujoco.mjtObj.mjOBJ_KEY,
            f"{self._model_prefix(model_id)}{value}",
        )

    def set_keyframe_properties(self, properties: KeyframeProperties) -> bool:
        identity = self._keyframe_identity(properties.keyframe_id)
        if identity is None or identity[0] != int(properties.model_id):
            return False
        model_id, current_name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        local = source_spec.copy().compile()
        expected = (
            local.nq,
            local.nv,
            local.na,
            local.nu,
            local.nmocap * 3,
            local.nmocap * 4,
        )
        arrays = tuple(
            np.asarray(values, np.float64).reshape(-1)
            for values in (
                properties.qpos,
                properties.qvel,
                properties.act,
                properties.ctrl,
                properties.mocap_position,
                properties.mocap_quaternion,
            )
        )
        if tuple(len(values) for values in arrays) != expected:
            return False
        working = source_spec.copy()
        element = working.key(current_name)
        value = str(properties.name).strip()
        duplicate = working.key(value)
        if element is None or not value or (value != current_name and duplicate is not None):
            return False
        element.name = value
        element.time = float(properties.time)
        element.qpos = arrays[0]
        element.qvel = arrays[1]
        element.act = arrays[2]
        element.ctrl = arrays[3]
        element.mpos = arrays[4]
        element.mquat = arrays[5]
        return self._replace_model_spec(model_id, working)

    def remove_model_keyframe(self, keyframe_id: int) -> bool:
        identity = self._keyframe_identity(keyframe_id)
        if identity is None:
            return False
        model_id, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.key(name)
        if element is None:
            return False
        working.delete(element)
        return self._replace_model_spec(model_id, working)

    def sensors(self) -> list[SensorInfo]:
        m = self._m
        return [
            SensorInfo(
                sensor_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor{i}",
                type=str(mujoco.mjtSensor(int(m.sensor_type[i]))).split(".")[-1],
                data_adr=int(m.sensor_adr[i]),
                dim=int(m.sensor_dim[i]),
            )
            for i in range(m.nsensor)
        ]

    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        m = self._m
        return [
            EqualityConstraintInfo(
                constraint_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"equality{i}",
                type=str(mujoco.mjtEq(int(m.eq_type[i]))).split(".")[-1],
                enabled=bool(self._d.eq_active[i]),
            )
            for i in range(m.neq)
        ]

    def load_keyframe(self, keyframe_id: int) -> bool:
        i = int(keyframe_id)
        if not 0 <= i < self._m.nkey:
            return False
        mujoco.mj_resetDataKeyframe(self._m, self._d, i)
        mujoco.mj_forward(self._m, self._d)
        self._perturb_body = -1
        return True

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Resolve a model camera from MuJoCo's current forward-kinematics result."""
        i = int(camera_id)
        m, d = self._m, self._d
        if not 0 <= i < m.ncam:
            return None
        rot = np.asarray(d.cam_xmat[i], np.float32).reshape(3, 3)
        eye = np.asarray(d.cam_xpos[i], np.float32).copy()
        preview = self._model_transform_preview
        if preview is not None and preview.camera_mask[i]:
            eye = (
                preview.position
                + preview.delta_rotation @ (np.asarray(eye, np.float64) - preview.previous_position)
            ).astype(np.float32)
            rot = (preview.delta_rotation @ np.asarray(rot, np.float64)).astype(np.float32)
        distance = max(float(m.stat.extent), 1e-3)
        projection = getattr(m, "cam_projection", None)
        orthographic = bool(
            projection is not None
            and int(projection[i]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        )
        fovy = float(m.cam_fovy[i])
        intrinsics = np.asarray(m.cam_intrinsic[i], np.float32)
        sensor_size = np.asarray(m.cam_sensorsize[i], np.float32)
        return CameraView(
            eye=eye,
            target=(eye - rot[:, 2] * distance).astype(np.float32),
            up=rot[:, 1].copy(),
            fov_y=float(np.deg2rad(fovy if not orthographic else 45.0)),
            near=max(float(m.vis.map.znear) * distance, 1e-4),
            far=max(float(m.vis.map.zfar) * distance, distance),
            orthographic=orthographic,
            ortho_height=fovy if orthographic else 2.0 * distance * np.tan(np.deg2rad(fovy) * 0.5),
            focal_length=intrinsics[:2].copy(),
            sensor_size=sensor_size.copy(),
            principal_offset=intrinsics[2:4].copy(),
        )

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        i = int(camera_id)
        if not 0 <= i < self._m.ncam:
            return False
        identity = self._element_identity(NodeType.CAMERA, i)
        if identity is None:
            return False
        model_id, name = identity
        element = self._element(model_id, "camera", name)
        if element is None:
            return False

        body = int(self._m.cam_bodyid[i])
        body_position = np.asarray(self._d.xpos[body], np.float64)
        body_rotation = np.asarray(self._d.xmat[body], np.float64).reshape(3, 3)
        eye = np.asarray(camera.eye, np.float64).reshape(3)
        forward = math3d.normalize(np.asarray(camera.target, np.float64) - eye)
        up = math3d.normalize(np.asarray(camera.up, np.float64))
        right = math3d.normalize(np.cross(forward, up))
        if not np.any(right):
            right = math3d.normalize(np.cross(forward, np.array((0.0, 0.0, 1.0))))
        up = math3d.normalize(np.cross(right, forward))
        world_rotation = np.column_stack((right, up, -forward))
        local_position = body_rotation.T @ (eye - body_position)
        local_rotation = body_rotation.T @ world_rotation
        quaternion = math3d.mat3_to_quat(local_rotation)

        element.pos = local_position
        element.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        element.quat = quaternion
        element.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        element.proj = (
            mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
            if camera.orthographic
            else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
        )
        element.fovy = (
            float(camera.ortho_height) if camera.orthographic else float(np.degrees(camera.fov_y))
        )
        if camera.uses_intrinsics():
            element.focal_length = np.asarray(camera.focal_length, np.float64)
            element.sensor_size = np.asarray(camera.sensor_size, np.float64)
            element.principal_length = np.asarray(camera.principal_offset, np.float64)
        else:
            zeros = np.zeros(2, np.float64)
            element.focal_length = zeros
            element.focal_pixel = zeros
            element.sensor_size = zeros
            element.principal_length = zeros
            element.principal_pixel = zeros
        self._mark_model_edited(model_id)

        self._m.cam_pos[i] = local_position
        self._m.cam_quat[i] = quaternion
        self._m.cam_fovy[i] = element.fovy
        projection = getattr(self._m, "cam_projection", None)
        if projection is not None:
            projection[i] = int(element.proj)
        self._m.cam_intrinsic[i, :2] = camera.focal_length
        self._m.cam_intrinsic[i, 2:4] = camera.principal_offset
        self._m.cam_sensorsize[i] = camera.sensor_size
        extent = max(float(self._m.stat.extent), 1e-6)
        self._m.vis.map.znear = max(float(camera.near) / extent, 1e-7)
        self._m.vis.map.zfar = max(float(camera.far) / extent, self._m.vis.map.znear)
        mujoco.mj_forward(self._m, self._d)
        return True

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        return tuple(
            VisualGroupInfo(name, tuple(bool(x) for x in self._visual_groups[name]))
            for name in VISUAL_GROUP_CATEGORIES
        )

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        groups = self._visual_groups.get(str(category))
        i = int(group)
        if groups is None or not 0 <= i < len(groups):
            return False
        value = bool(visible)
        if bool(groups[i]) == value:
            return True
        groups[i] = value
        if category == "geom":
            self._ray_geomgroup[i] = int(value)
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def _group_visibility(self, model_groups, category: str) -> np.ndarray:
        groups = self._visual_groups[category]
        return groups[np.asarray(model_groups, np.intp)].copy()

    def set_qpos(self, index: int, value: float) -> bool:
        if not 0 <= int(index) < self._m.nq:
            return False
        self._d.qpos[int(index)] = float(value)
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_qpos_batch(self, indices: np.ndarray, values: np.ndarray) -> bool:
        raw_slots = np.asarray(indices).reshape(-1)
        if not np.issubdtype(raw_slots.dtype, np.integer):
            return False
        slots = raw_slots.astype(np.intp, copy=False)
        coordinates = np.asarray(values, np.float64).reshape(-1)
        if (
            not len(slots)
            or len(slots) != len(coordinates)
            or np.any(slots < 0)
            or np.any(slots >= self._m.nq)
            or len(np.unique(slots)) != len(slots)
            or not np.all(np.isfinite(coordinates))
        ):
            return False
        self._d.qpos[slots] = coordinates
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool:
        i = int(constraint_id)
        if not 0 <= i < self._m.neq:
            return False
        self._d.eq_active[i] = bool(enabled)
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_ctrl(self, index: int, value: float) -> bool:
        m, i = self._m, int(index)
        if not 0 <= i < m.nu:
            return False
        v = float(value)
        actuator = int(self._ctrl_actuator[i])
        if actuator >= 0 and bool(m.actuator_ctrllimited[actuator]):
            lo, hi = m.actuator_ctrlrange[actuator]
            v = float(np.clip(v, lo, hi))
        self._d.ctrl[i] = v
        return True

    def set_light(self, light_index: int, light: Light) -> bool:
        i = int(light_index)
        if not 0 <= i < self._m.nlight:
            return False
        identity = self._element_identity(NodeType.LIGHT, i)
        light_types = {
            LightType.DIRECTIONAL: mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            LightType.POINT: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.SPOT: mujoco.mjtLightType.mjLIGHT_SPOT,
            LightType.IMAGE: mujoco.mjtLightType.mjLIGHT_IMAGE,
            # AREA is a Forge render extension represented by a MuJoCo point
            # light plus custom metadata.
            LightType.AREA: mujoco.mjtLightType.mjLIGHT_POINT,
        }
        if light.type not in light_types:
            return False
        m = self._m
        texture_id = -1
        texture_name = ""
        if light.type is LightType.IMAGE:
            texture_id = (
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, light.texture)
                if light.texture
                else -1
            )
            if texture_id < 0:
                return False
            texture_model, texture_name = self._model_element_name(
                light.texture or "", mujoco.mjtObj.mjOBJ_TEXTURE
            )
            # A stored child MjSpec can only reference assets from that child.
            # The composed model exposes prefixed names, so strip the prefix for
            # write-back and reject references that cannot survive standalone export.
            if identity is not None and texture_model != identity[0]:
                return False
        m.light_type[i] = int(light_types[light.type])
        m.light_pos[i] = light.position
        direction = np.asarray(light.direction, np.float64)
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            m.light_dir[i] = direction / length

        # MuJoCo compiles the reference poses used by track and trackcom lights
        # into light_pos0/light_poscom0/light_dir0. mj_forward() consumes those
        # arrays but does not rebuild them after an interactive model edit, so a
        # paused gizmo write would otherwise snap back to the compiled pose.
        body = int(m.light_bodyid[i])
        body_rotation = np.asarray(self._d.xmat[body], np.float64).reshape(3, 3)
        world_position = np.asarray(self._d.xpos[body], np.float64) + body_rotation @ np.asarray(
            m.light_pos[i], np.float64
        )
        world_direction = body_rotation @ np.asarray(m.light_dir[i], np.float64)
        target = int(m.light_targetbodyid[i])
        reference_body = target if target >= 0 else body
        m.light_pos0[i] = world_position - np.asarray(self._d.xpos[body], np.float64)
        m.light_poscom0[i] = world_position - np.asarray(
            self._d.subtree_com[reference_body], np.float64
        )
        world_direction_length = float(np.linalg.norm(world_direction))
        if world_direction_length > 0.0:
            m.light_dir0[i] = world_direction / world_direction_length
        m.light_diffuse[i] = light.diffuse
        m.light_specular[i] = light.specular
        m.light_ambient[i] = light.ambient
        m.light_attenuation[i] = light.attenuation
        m.light_range[i] = light.range
        m.light_cutoff[i] = light.cutoff
        m.light_exponent[i] = light.exponent
        m.light_bulbradius[i] = light.area_radius
        m.light_texid[i] = texture_id
        m.light_intensity[i] = light.intensity
        m.light_castshadow[i] = light.cast_shadow
        m.light_active[i] = light.active
        if identity is not None:
            model_id, name = identity
            element = self._element(model_id, "light", name)
            if element is not None:
                element.type = light_types[light.type]
                element.pos = light.position
                element.dir = light.direction
                element.diffuse = light.diffuse
                element.specular = light.specular
                element.ambient = light.ambient
                element.attenuation = light.attenuation
                element.range = light.range
                element.cutoff = light.cutoff
                element.exponent = light.exponent
                element.texture = texture_name
                element.intensity = light.intensity
                element.castshadow = light.cast_shadow
                element.active = light.active
                element.bulbradius = light.area_radius
                names = set(
                    _spec_text_names(self._spec_for_model(model_id), _FORGE_AREA_LIGHTS_TEXT)
                )
                if light.type is LightType.AREA:
                    names.add(name)
                else:
                    names.discard(name)
                _set_text_names(self._spec_for_model(model_id), _FORGE_AREA_LIGHTS_TEXT, names)
                self._mark_model_edited(model_id)
        if i < len(self._area_lights):
            self._area_lights[i] = light.type is LightType.AREA
        mujoco.mj_forward(m, self._d)
        self._lights_edited = True
        if self._source is not None:
            self._source.lights = self._build_lights()
        return True

    def set_skybox(self, texture: str | None) -> bool:
        selected = -1
        if texture is not None:
            selected = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, texture)
            if selected < 0 or int(self._m.tex_type[selected]) not in (
                int(mujoco.mjtTexture.mjTEXTURE_CUBE),
                int(mujoco.mjtTexture.mjTEXTURE_SKYBOX),
            ):
                return False

        updates = []
        for index in range(self._m.ntex):
            current = int(self._m.tex_type[index])
            next_type = (
                mujoco.mjtTexture.mjTEXTURE_SKYBOX
                if index == selected
                else (
                    mujoco.mjtTexture.mjTEXTURE_CUBE
                    if current == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
                    else current
                )
            )
            if int(next_type) == current:
                continue
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, index) or ""
            model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_TEXTURE)
            spec = self._spec_for_model(model_id)
            element = spec.texture(name) if spec is not None and name else None
            if element is None:
                return False
            updates.append((index, next_type, model_id, compiled_name, element))

        changed_models: set[int] = set()
        for index, next_type, model_id, _compiled_name, element in updates:
            self._m.tex_type[index] = int(next_type)
            element.type = next_type
            changed_models.add(model_id)
        for model_id in changed_models:
            self._mark_model_edited(model_id)
        if self._source is not None:
            for _index, next_type, _model_id, compiled_name, _element in updates:
                item = self._source.textures.get(compiled_name)
                if item is not None:
                    item_type = (
                        TextureType.SKYBOX
                        if int(next_type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
                        else TextureType.CUBE
                    )
                    self._source.textures[compiled_name] = replace(item, type=item_type)
            self._source.skybox = texture
        return True

    def _element_identity(self, node_type: NodeType, slot: int) -> tuple[int, str] | None:
        self.nodes()
        field = "camera_index" if node_type is NodeType.CAMERA else "light_index"
        node = next(
            (
                node
                for node in self._nodes
                if node.type is node_type and getattr(node, field) == slot
            ),
            None,
        )
        identity = self._node_element.get(node.node_id) if node is not None else None
        return (identity[0], identity[2]) if identity is not None else None

    def set_material(self, material_index: int, material: Material) -> bool:
        i = int(material_index)
        if not 0 <= i < self._m.nmat:
            return False
        m = self._m
        texture_id = (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, material.texture)
            if material.texture
            else -1
        )
        if material.texture and texture_id < 0:
            return False
        compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL)
        spec = self._spec_for_model(model_id)
        element = spec.material(name) if spec is not None and name else None
        texture_name = ""
        if element is not None and material.texture:
            texture_model, texture_name = self._model_element_name(
                material.texture, mujoco.mjtObj.mjOBJ_TEXTURE
            )
            if texture_model != model_id:
                return False

        current_texture_id = next(
            (
                int(m.mat_texid[i, role])
                for role in (_TEXROLE_RGB, _TEXROLE_RGBA)
                if int(m.mat_texid[i, role]) >= 0
            ),
            -1,
        )
        current_texture = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, current_texture_id) or ""
            if current_texture_id >= 0
            else ""
        )
        texture_changed = current_texture != (material.texture or "")

        m.mat_rgba[i] = material.rgba
        m.mat_emission[i] = material.emission
        m.mat_specular[i] = material.specular
        m.mat_shininess[i] = material.shininess
        m.mat_reflectance[i] = material.reflectance
        m.mat_metallic[i] = material.metallic
        m.mat_roughness[i] = material.roughness
        m.mat_texrepeat[i] = material.tex_repeat
        m.mat_texuniform[i] = material.tex_uniform
        if texture_changed:
            m.mat_texid[i, _TEXROLE_RGB] = texture_id
            m.mat_texid[i, _TEXROLE_RGBA] = -1
        if element is not None:
            element.rgba = material.rgba
            element.emission = material.emission
            element.specular = material.specular
            element.shininess = material.shininess
            element.reflectance = material.reflectance
            element.metallic = material.metallic
            element.roughness = material.roughness
            element.texrepeat = material.tex_repeat
            element.texuniform = material.tex_uniform
            if texture_changed:
                textures = list(element.textures)
                textures.extend([""] * (int(mujoco.mjtTextureRole.mjNTEXROLE) - len(textures)))
                textures[_TEXROLE_RGB] = texture_name
                textures[_TEXROLE_RGBA] = ""
                element.textures = textures
            # MuJoCo 3.11 serializes MjsMaterial.textures correctly but retains
            # the old compiled texture reference in that live MjSpec. Reparse the
            # serialized spec so a later topology rebuild or export sees the edit.
            self._store_model_spec(model_id, mujoco.MjSpec.from_string(spec.to_xml()))
        return True

    def model_material_indices(self, model_id: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self._m.nmat)
            if (name := mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, index))
            and self._model_element_name(name, mujoco.mjtObj.mjOBJ_MATERIAL)[0] == int(model_id)
        )

    def model_texture_names(self, model_id: int) -> tuple[str, ...]:
        return tuple(
            name
            for index in range(self._m.ntex)
            if (name := mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, index))
            and self._model_element_name(name, mujoco.mjtObj.mjOBJ_TEXTURE)[0] == int(model_id)
        )

    def create_model_material(self, model_id: int, name: str) -> int:
        model_id = int(model_id)
        value = str(name).strip()
        spec = self._spec_for_model(model_id)
        if spec is None or not value or spec.material(value) is not None:
            return -1
        working = spec.copy()
        working.add_material(name=value)
        if not self._replace_model_spec(model_id, working):
            return -1
        return mujoco.mj_name2id(
            self._m,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            f"{self._model_prefix(model_id)}{value}",
        )

    def set_model_material_layers(
        self, model_id: int, name: str, layers: tuple[tuple[str, str], ...]
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return False
        root, _xml = _component_xml(spec)
        asset, target = _model_asset_element(root, "material", value)
        if asset is None or target is None:
            return False
        texture_names = {
            str(element.attrib.get("name", ""))
            for element in asset.findall("texture")
            if str(element.attrib.get("name", ""))
        }
        normalized = tuple(
            (str(role).strip().lower(), str(texture).strip()) for role, texture in layers
        )
        roles = tuple(role for role, _texture in normalized)
        if (
            len(roles) != len(set(roles))
            or any(role not in MATERIAL_TEXTURE_ROLES for role in roles)
            or any(texture not in texture_names for _role, texture in normalized)
        ):
            return False
        target.attrib.pop("texture", None)
        for child in tuple(target):
            if child.tag == "layer":
                target.remove(child)
        for role, texture in normalized:
            ET.SubElement(target, "layer", {"role": role, "texture": texture})
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def add_model_material(self, node_id: int, name: str, copy_from: int = -1) -> int:
        identity = self._node_element.get(int(node_id))
        value = str(name).strip()
        if identity is None or identity[1] not in (NodeType.GEOM, NodeType.SITE) or not value:
            return -1
        model_id, node_type, element_name = identity
        spec = self._spec_for_model(model_id)
        if spec is None or spec.material(value) is not None:
            return -1
        source_name = ""
        source_index = int(copy_from)
        if source_index >= 0:
            if not 0 <= source_index < self._m.nmat:
                return -1
            compiled_name = (
                mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, source_index) or ""
            )
            source_model, source_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if source_model != model_id:
                return -1

        if source_name:
            root, _xml = _component_xml(spec)
            asset, source_element = _model_asset_element(root, "material", source_name)
            if asset is None or source_element is None:
                return -1
            duplicate = deepcopy(source_element)
            duplicate.set("name", value)
            children = tuple(asset)
            asset.insert(children.index(source_element) + 1, duplicate)
            working = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        else:
            working = spec.copy()
            working.add_material(name=value)
        target = getattr(working, node_type.value)(element_name)
        if target is None:
            return -1
        target.material = value
        if not self._replace_model_spec(model_id, working):
            return -1
        compiled_name = value
        if model_id > 0:
            attached = next(
                (item for item in self._attached_models if item.model_id == model_id), None
            )
            if attached is None:
                return -1
            compiled_name = f"{attached.prefix}{value}"
        return mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, compiled_name)

    def import_model_texture(
        self,
        model_id: int,
        path: Path,
        name: str,
        material_index: int = -1,
        texture_type: str = "2d",
    ) -> bool:
        model_id = int(model_id)
        source = Path(path).expanduser().resolve()
        value = str(name).strip()
        spec = self._spec_for_model(model_id)
        if spec is None or not source.is_file() or not value or spec.texture(value) is not None:
            return False
        texture_types = {
            "2d": mujoco.mjtTexture.mjTEXTURE_2D,
            "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
            "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        }
        kind = str(texture_type).strip().lower()
        if kind not in texture_types or (kind != "2d" and int(material_index) >= 0):
            return False
        working = spec.copy()
        working.add_texture(
            name=value,
            type=texture_types[kind],
            file=str(source),
        )
        material = int(material_index)
        if material >= 0:
            if not 0 <= material < self._m.nmat:
                return False
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material) or ""
            material_model, material_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            target = working.material(material_name) if material_model == model_id else None
            if target is None:
                return False
            textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
            textures[_TEXROLE_RGB] = value
            target.textures = textures
        return self._replace_model_spec(model_id, working)

    def set_geometry_material(self, node_id: int, material_index: int) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or identity[1] not in (NodeType.GEOM, NodeType.SITE):
            return False
        model_id, node_type, name = identity
        material = int(material_index)
        material_name = ""
        if material >= 0:
            if not 0 <= material < self._m.nmat:
                return False
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material) or ""
            material_model, material_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if material_model != model_id:
                return False
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False
        element.material = material_name
        if node_type is NodeType.GEOM:
            self._m.geom_matid[node.geom_index] = material
        else:
            self._m.site_matid[node.site_index] = material
        self._mark_model_edited(model_id)
        self._source = None
        self._structure_revision += 1
        return True

    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool:
        source = self.scene_source()
        instances = np.flatnonzero(source.geom_node == int(node_id))
        if not len(instances):
            return False
        color = np.asarray(rgba, np.float32).reshape(4)
        for instance in instances:
            pose_source = InstancePoseSource(int(source.geom_pose_source[instance]))
            source_id = int(source.geom_source[instance])
            if pose_source is InstancePoseSource.GEOM:
                self._m.geom_rgba[source_id] = color
            elif pose_source is InstancePoseSource.SITE:
                self._m.site_rgba[source_id] = color
            else:
                mesh = source.geom_mesh[instance]
                if mesh.shape in (MeshShape.FLEX, MeshShape.FLEX_FACE):
                    self._m.flex_rgba[mesh.index] = color
                elif mesh.shape is MeshShape.SKIN:
                    self._m.skin_rgba[mesh.index] = color
        source.geom_rgba[instances] = color
        identity = self._node_element.get(int(node_id))
        if identity is not None:
            model_id, node_type, name = identity
            if node_type in (NodeType.GEOM, NodeType.SITE):
                element = self._element(model_id, node_type.value, name)
                if element is not None:
                    element.rgba = color
                    self._mark_model_edited(model_id)
        return True

    def set_geometry_size(self, node_id: int, size: np.ndarray) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type not in (NodeType.GEOM, NodeType.SITE):
            return False
        values = np.asarray(size, np.float64).reshape(3)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            return False

        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False

        if node_type is NodeType.GEOM:
            index = int(node.geom_index)
            if not 0 <= index < self._m.ngeom:
                return False
            primitive_type = int(self._m.geom_type[index])
            compiled_size = np.asarray(self._m.geom_size[index], np.float64).copy()
        else:
            index = int(node.site_index)
            if not 0 <= index < self._m.nsite:
                return False
            primitive_type = int(self._m.site_type[index])
            compiled_size = np.asarray(self._m.site_size[index], np.float64).copy()

        supported = {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }
        if primitive_type not in supported or (
            node_type is NodeType.SITE and primitive_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            return False

        authored_size = np.asarray(element.size, np.float64).copy()
        if primitive_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            authored_size[:2] = values[:2]
            compiled_size[:2] = values[:2]
        elif primitive_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            authored_size[0] = values[0]
            compiled_size[0] = values[0]
        elif primitive_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }:
            authored_size[0] = values[0]
            compiled_size[:2] = (values[0], values[2])
            fromto = np.asarray(element.fromto, np.float64).copy()
            if np.all(np.isfinite(fromto)):
                center = 0.5 * (fromto[:3] + fromto[3:])
                axis = fromto[3:] - fromto[:3]
                length = float(np.linalg.norm(axis))
                if length <= 1e-12:
                    return False
                axis /= length
                fromto[:3] = center - axis * values[2]
                fromto[3:] = center + axis * values[2]
                element.fromto = fromto
            else:
                authored_size[1] = values[2]
        else:
            authored_size[:3] = values[:3]
            compiled_size[:3] = values[:3]
        element.size = authored_size
        if node_type is NodeType.GEOM:
            self._m.geom_size[index] = compiled_size
        else:
            self._m.site_size[index] = compiled_size
        mujoco.mj_setConst(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)

        self._mark_model_edited(model_id)
        self._source = None
        self._structure_revision += 1
        return True

    def set_pose(self, node_id: int, position, rotation) -> bool:
        model_id = self._node_model.get(int(node_id), -1)
        if model_id >= 0:
            return self.set_scene_model_transform(model_id, position, rotation)
        identity = self._node_element.get(int(node_id))
        if identity is not None:
            _model_id, node_type, _name = identity
            body = self._node_body.get(int(node_id), -1)
            if node_type in (NodeType.LINK, NodeType.ROBOT) and self._is_posable_body(body):
                return self._set_dynamic_body_pose(body, position, rotation)
            node = self._node_for_id(node_id)
            if (
                node is not None
                and node.posable
                and node_type
                in (
                    NodeType.LINK,
                    NodeType.ROBOT,
                    NodeType.GEOM,
                    NodeType.SITE,
                )
            ):
                return self._set_model_element_pose(int(node_id), position, rotation)
        body = self._node_body.get(int(node_id), -1)
        if body < 0 or not self._is_posable_body(body):
            return False
        return self._set_dynamic_body_pose(body, position, rotation)

    def _set_dynamic_body_pose(self, body: int, position, rotation) -> bool:
        mocap = int(self._m.body_mocapid[body])
        if mocap >= 0:
            self._d.mocap_pos[mocap] = np.asarray(position, np.float64).reshape(3)
            self._d.mocap_quat[mocap] = math3d.mat3_to_quat(rotation)
            mujoco.mj_forward(self._m, self._d)
            return True
        adr = int(self._m.jnt_qposadr[int(self._m.body_jntadr[body])])
        self._d.qpos[adr : adr + 3] = np.asarray(position, np.float64).reshape(3)
        self._d.qpos[adr + 3 : adr + 7] = math3d.mat3_to_quat(rotation)

        dof = int(self._m.jnt_dofadr[int(self._m.body_jntadr[body])])
        self._d.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self._m, self._d)
        return True

    def _set_model_element_pose(self, node_id: int, position, rotation) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None:
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        spec = self._spec_for_model(model_id)
        if element is None or spec is None:
            return False

        world_position = np.asarray(position, np.float64).reshape(3)
        world_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        parent_body = (
            int(self._m.body_parentid[node.body_index])
            if node_type in (NodeType.LINK, NodeType.ROBOT)
            else int(node.body_index)
        )
        compiled_parent_position = np.asarray(self._d.xpos[parent_body], np.float64)
        compiled_parent_rotation = np.asarray(self._d.xmat[parent_body], np.float64).reshape(3, 3)
        compiled_position, compiled_rotation = _relative_pose(
            world_position,
            world_rotation,
            compiled_parent_position,
            compiled_parent_rotation,
        )

        spec_parent_position = compiled_parent_position
        spec_parent_rotation = compiled_parent_rotation
        if element.parent is spec.worldbody:
            attached = next(
                (item for item in self._attached_models if item.model_id == model_id), None
            )
            if attached is not None:
                spec_parent_position = attached.position
                spec_parent_rotation = attached.rotation
        local_position, local_rotation = _relative_pose(
            world_position,
            world_rotation,
            spec_parent_position,
            spec_parent_rotation,
        )
        fromto = (
            np.asarray(element.fromto, np.float64).copy()
            if node_type in (NodeType.GEOM, NodeType.SITE)
            else np.empty(0, np.float64)
        )
        if fromto.shape == (6,) and np.all(np.isfinite(fromto)):
            half_length = 0.5 * float(np.linalg.norm(fromto[3:] - fromto[:3]))
            axis = local_rotation[:, 2]
            fromto[:3] = local_position - axis * half_length
            fromto[3:] = local_position + axis * half_length
            element.fromto = fromto
        else:
            element.pos = local_position
            element.quat = math3d.mat3_to_quat(local_rotation)

        if node_type in (NodeType.LINK, NodeType.ROBOT):
            self._m.body_pos[node.body_index] = compiled_position
            self._m.body_quat[node.body_index] = math3d.mat3_to_quat(compiled_rotation)
        elif node_type is NodeType.GEOM:
            self._m.geom_pos[node.geom_index] = compiled_position
            self._m.geom_quat[node.geom_index] = math3d.mat3_to_quat(compiled_rotation)
        elif node_type is NodeType.SITE:
            self._m.site_pos[node.site_index] = compiled_position
            self._m.site_quat[node.site_index] = math3d.mat3_to_quat(compiled_rotation)
        else:
            return False
        self._mark_model_edited(model_id)
        mujoco.mj_forward(self._m, self._d)
        return True

    def capture_state(self) -> PhysicsState:
        data = self._d
        return PhysicsState(
            qpos=np.asarray(data.qpos, np.float64).copy(),
            qvel=np.asarray(data.qvel, np.float64).copy(),
            act=np.asarray(data.act, np.float64).copy(),
            ctrl=np.asarray(data.ctrl, np.float64).copy(),
            time=float(data.time),
            mocap_pos=np.asarray(data.mocap_pos, np.float64).copy(),
            mocap_quat=np.asarray(data.mocap_quat, np.float64).copy(),
        )

    def restore_state(self, state: PhysicsState) -> bool:
        data = self._d
        arrays = (
            (data.qpos, state.qpos),
            (data.qvel, state.qvel),
            (data.act, state.act),
            (data.ctrl, state.ctrl),
            (data.mocap_pos, state.mocap_pos),
            (data.mocap_quat, state.mocap_quat),
        )
        if any(np.shape(dst) != np.shape(src) for dst, src in arrays):
            return False
        for dst, src in arrays:
            np.copyto(dst, src)
        data.time = float(state.time)
        mujoco.mj_forward(self._m, data)
        self._perturb_body = -1
        return True

    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool:
        body = self._node_body.get(int(node_id), -1)
        if body <= 0:
            return False
        if int(self._m.body_weldid[body]) == 0:
            return False

        pert = self._perturb
        if self._perturb_body != body:
            point = np.asarray(self._d.xpos[body], np.float64)
            np.sqrt(self._d.qLDiagInv, out=self._perturb_sqrt_inv_d)
            mujoco.mj_jac(self._m, self._d, self._perturb_jac, None, point, body)
            mujoco.mj_solveM2(
                self._m,
                self._d,
                self._perturb_jac_m2,
                self._perturb_jac,
                self._perturb_sqrt_inv_d,
            )
            invmass = float(np.sum(self._perturb_jac_m2 * self._perturb_jac_m2))
            pert.localmass = 3.0 / max(invmass, 1e-15)
            pert.select = body
            pert.localpos[:] = 0.0
            self._perturb_body = body

        pert.active2 = 0
        if mode == "translate":
            pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
            pert.refselpos[:] = np.asarray(target_position, np.float64).reshape(3)
        elif mode == "rotate":
            pert.active = int(mujoco.mjtPertBit.mjPERT_ROTATE)
            body_quat = np.asarray(math3d.mat3_to_quat(target_rotation), np.float64)
            mujoco.mju_mulQuat(self._perturb_quat, body_quat, self._m.body_iquat[body])
            pert.refquat[:] = self._perturb_quat
        else:
            return False
        mujoco.mjv_applyPerturbForce(self._m, self._d, pert)
        return True

    def clear_perturb(self) -> None:
        self._d.xfrc_applied[:] = 0.0
        self._perturb.active = 0
        self._perturb.active2 = 0
        self._perturb_body = -1

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:
        self._ray_pnt[:] = np.asarray(origin, np.float64).reshape(3)
        v = np.asarray(direction, np.float64).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return 0, float("inf")
        self._ray_vec[:] = v / n
        self._ray_geomid[0] = -1
        dist = mujoco.mj_ray(
            self._m,
            self._d,
            self._ray_pnt,
            self._ray_vec,
            self._ray_geomgroup,
            True,
            -1,
            self._ray_geomid,
        )
        gid = int(self._ray_geomid[0])
        if dist < 0.0 or gid < 0:
            return 0, float("inf")
        body = int(self._m.geom_bodyid[gid])
        if body == 0:
            node_id = self._geom_nodes.get(gid, -1)
            node = self._node_for_id(node_id)
            if node is None or not node.object_id:
                return 0, float("inf")
            return int(node.object_id), float(dist)
        return body, float(dist)

    def camera_hint(self) -> CameraView | None:
        m = self._m
        extent = float(m.stat.extent) or 1.0
        center = np.asarray(m.stat.center, np.float32)
        az = np.deg2rad(float(m.vis.global_.azimuth))
        el = np.deg2rad(float(m.vis.global_.elevation))
        forward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32
        )
        distance = 1.5 * extent
        return CameraView(
            eye=(center - forward * distance).astype(np.float32),
            target=center.copy(),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            fov_y=float(np.deg2rad(float(m.vis.global_.fovy))),
            near=float(m.vis.map.znear) * extent,
            # The free editor camera is not constrained to MuJoCo's classic
            # viewport range. Keep distant authored and composed entities visible.
            far=max(float(m.vis.map.zfar), 200.0) * extent,
        )

    def release(self) -> None:
        self._model_transform_preview = None
        self._m = None
        self._d = None
        self._root_spec = None
        self._attached_models.clear()
        self._source = None
        self._nodes.clear()
        self._node_body.clear()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self._next_component_id.clear()
        self._node_model.clear()
        self._node_element.clear()
        self._geom_nodes.clear()
        self._site_nodes.clear()
        self._flex_nodes.clear()
        self._skin_nodes.clear()
        self._deformables.clear()
        self._mesh_updates.clear()
        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_wrap_objects = None
        self._mj_body_xpos = None
        self._mj_body_xmat3 = None
        self._geom_xpos_buf = np.zeros((0, 3), np.float32)
        self._geom_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((0, 3), np.float32)
        self._site_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((0, 3), np.float32)
        self._body_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._qpos_buf = np.zeros(0, np.float32)
        self._qvel_buf = np.zeros(0, np.float32)
        self._ctrl_buf = np.zeros(0, np.float32)
        self._sensor_buf = np.zeros(0, np.float32)
        self._contact_buf = np.zeros((0, 7), np.float32)
        self._contact_view = self._contact_buf
        self._contact_force_buf = np.zeros((0, 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf
        self._tendon_segments = np.zeros((0, 2, 3), np.float32)
        self._tendon_ids = np.zeros(0, np.int32)
        self._tendon_widths = np.zeros(0, np.float32)
        self._flex_vertices_buf = np.zeros((0, 3), np.float32)
        self._perturb_jac = np.zeros((3, 0), np.float64)
        self._perturb_jac_m2 = np.zeros((3, 0), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(0, np.float64)
        self._visual_state = {}
        self._light_state = {}

    @property
    def model(self):
        return self._m

    @property
    def data(self):
        return self._d

    @property
    def fast_pose(self) -> bool:
        return self._fast_pose


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
    _set_numeric(spec, _FORGE_AMBIENT_NUMERIC, lights.ambient)
    _set_numeric(
        spec,
        _FORGE_HAZE_NUMERIC,
        (float(lights.horizon_haze), float(lights.horizon_haze_slices)),
    )


def _authored_builtin_mesh(shape: MeshShape) -> MeshData | None:
    if shape not in (MeshShape.CYLINDER, MeshShape.CONE):
        return None
    segments = 32
    positions: list[tuple[float, float, float]] = []
    indices: list[int] = []
    radius_top = 1.0 if shape is MeshShape.CYLINDER else 0.0
    for segment in range(segments):
        angle = 2.0 * np.pi * segment / segments
        x, y = float(np.cos(angle)), float(np.sin(angle))
        positions.extend(((x, y, -1.0), (radius_top * x, radius_top * y, 1.0)))
    bottom_center = len(positions)
    positions.append((0.0, 0.0, -1.0))
    top_center = len(positions)
    positions.append((0.0, 0.0, 1.0))
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        bottom = 2 * segment
        top = bottom + 1
        next_bottom = 2 * next_segment
        next_top = next_bottom + 1
        indices.extend((bottom, next_bottom, top, top, next_bottom, next_top))
        indices.extend((bottom_center, next_bottom, bottom))
        if radius_top > 0.0:
            indices.extend((top_center, top, next_top))
    vertex_count = len(positions)
    return MeshData(
        positions=np.asarray(positions, np.float32),
        normals=np.zeros((vertex_count, 3), np.float32),
        uvs=np.zeros((vertex_count, 2), np.float32),
        indices=np.asarray(indices, np.uint32),
    )


def _relative_pose(position, rotation, parent_position, parent_rotation):
    parent_rotation = np.asarray(parent_rotation, np.float64).reshape(3, 3)
    local_rotation = parent_rotation.T @ np.asarray(rotation, np.float64).reshape(3, 3)
    local_position = parent_rotation.T @ (
        np.asarray(position, np.float64).reshape(3)
        - np.asarray(parent_position, np.float64).reshape(3)
    )
    return local_position, local_rotation
