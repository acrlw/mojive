"""MJCF field inventories, reference choices and asset schema."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import cached_property

from ..base import (
    ModelComponentField,
    ModelComponentPathItem,
)
from .engine import mujoco

_MODEL_COMPONENT_CATEGORIES = (
    "contact",
    "actuator",
    "sensor",
    "tendon",
    "equality",
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


_MODEL_ASSET_TYPES = ("material", "texture", "mesh", "hfield", "skin", "model")


_MODEL_ASSET_REFERENCE_FIELDS = {
    "material": ("material",),
    "texture": ("texture",),
    "mesh": ("mesh",),
    "hfield": ("hfield",),
    "model": ("model",),
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


class _ComponentReferences:
    """Share immutable reference choices across one component query.

    Do not retain XML across calls: authoring can mutate MjSpec in place. A
    query-local index avoids both stale references and quadratic tree scans.
    """

    def __init__(self, root: ET.Element):
        self.root = root
        self._names: dict[str, tuple[str, ...]] = {}
        self._objects: dict[str, tuple[str, ...]] = {}
        self._choices: dict[tuple[str, str, str, bool], tuple[str, ...]] = {}

    @cached_property
    def ball_joints(self) -> tuple[str, ...]:
        return tuple(
            item.attrib["name"]
            for item in self.root.iter("joint")
            if item.attrib.get("name") and item.attrib.get("type", "hinge") == "ball"
        )

    @staticmethod
    def _limited_names(elements) -> tuple[str, ...]:
        return tuple(
            item.attrib["name"]
            for item in elements
            if item.attrib.get("name")
            and (item.attrib.get("limited") in {"true", "1"} or bool(item.attrib.get("range")))
        )

    @cached_property
    def limited_joints(self) -> tuple[str, ...]:
        return self._limited_names(self.root.iter("joint"))

    @cached_property
    def limited_tendons(self) -> tuple[str, ...]:
        return self._limited_names(self.root.find("tendon") or ())

    @cached_property
    def tendons(self) -> tuple[str, ...]:
        return self.names("fixed") + self.names("spatial")

    def names(self, tag: str) -> tuple[str, ...]:
        if tag not in self._names:
            self._names[tag] = _named_elements(self.root, tag)
        return self._names[tag]

    def objects(self, object_type: str) -> tuple[str, ...]:
        if object_type not in self._objects:
            tags = _OBJECT_REFERENCE_TAGS.get(str(object_type).lower(), ())
            names = tuple(dict.fromkeys(name for tag in tags for name in self.names(tag)))
            self._objects[object_type] = ("world", *names) if object_type == "body" else names
        return self._objects[object_type]

    def choices(
        self, name: str, attributes: dict[str, str], *, optional: bool = False
    ) -> tuple[str, ...]:
        key = (name, attributes.get("objtype", ""), attributes.get("reftype", ""), optional)
        if key not in self._choices:
            if name == "class":
                values = tuple(
                    value
                    for element in self.root.iter("default")
                    if (value := str(element.attrib.get("class", "")).strip())
                )
            elif name in {"objtype", "reftype"}:
                values = tuple(kind for kind in _OBJECT_REFERENCE_TAGS if self.objects(kind))
            elif name == "objname":
                values = self.objects(key[1])
            elif name == "refname":
                values = self.objects(key[2])
            else:
                target = _REFERENCE_ELEMENT.get(name)
                values = self.names(target) if target else ()
            self._choices[key] = ("", *values) if optional and values else values
        return self._choices[key]


def _component_fields(
    references: _ComponentReferences, category: str, subtype: str, attributes: dict[str, str]
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
        options = references.choices(name, values, optional=True)
        if options:
            return options
        if name in tri_state:
            return ("", "false", "true", "auto")
        if name in _BOOLEAN_PROPERTY_FIELDS:
            return ("", "false", "true")
        return ()

    return tuple(ModelComponentField(name, value, choices(name)) for name, value in values.items())


def _component_path_presets(
    references: _ComponentReferences, category: str, subtype: str
) -> tuple[ModelComponentPathItem, ...]:
    if category != "tendon":
        return ()
    if subtype == "fixed":
        joints = references.names("joint")
        return (
            ModelComponentPathItem(
                "joint",
                (
                    ModelComponentField("joint", joints[0] if joints else "", joints),
                    ModelComponentField("coef", "1"),
                ),
            ),
        )
    sites = references.names("site")
    geoms = references.names("geom")
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
    references: _ComponentReferences,
    category: str,
    subtype: str,
    child: ET.Element,
) -> tuple[ModelComponentField, ...]:
    values = dict(child.attrib)
    return tuple(
        ModelComponentField(name, value, references.choices(name, values))
        for name, value in values.items()
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


def _component_presets(references: _ComponentReferences, category: str) -> tuple[str, ...]:
    """Offer only component kinds whose required model references exist."""
    if category == "contact":
        return (
            *(("pair",) if len(references.names("geom")) >= 2 else ()),
            *(("exclude",) if len(references.names("body")) >= 2 else ()),
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
                if references.names("joint")
                else ()
            ),
            *(("orientation",) if references.ball_joints else ()),
            *(("adhesion",) if references.names("body") else ()),
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
                if references.names("site")
                else ()
            ),
            *(
                ("camprojection",)
                if references.names("site") and references.names("camera")
                else ()
            ),
            *(("jointpos", "jointvel", "jointactuatorfrc") if references.names("joint") else ()),
            *(("ballquat", "ballangvel") if references.ball_joints else ()),
            *(
                ("jointlimitpos", "jointlimitvel", "jointlimitfrc")
                if references.limited_joints
                else ()
            ),
            *(("tendonpos", "tendonvel", "tendonactuatorfrc") if references.tendons else ()),
            *(
                ("tendonlimitpos", "tendonlimitvel", "tendonlimitfrc")
                if references.limited_tendons
                else ()
            ),
            *(
                ("actuatorpos", "actuatorvel", "actuatorfrc")
                if references.names("actuator")
                else ()
            ),
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
                if references.names("body")
                else ()
            ),
            *(("insidesite",) if references.names("site") and references.names("body") else ()),
            *(
                ("distance", "normal", "fromto", "contact")
                if len(references.names("body")) >= 2
                else ()
            ),
            *(("tactile",) if references.names("mesh") else ()),
            "e_potential",
            "e_kinetic",
            "clock",
            "user",
        )
    if category == "tendon":
        return (
            *(("fixed",) if references.names("joint") else ()),
            *(("spatial",) if len(references.names("site")) >= 2 else ()),
        )
    if category == "equality":
        return (
            *(("joint",) if references.names("joint") else ()),
            *(("weld", "connect") if references.names("body") else ()),
            *(("tendon",) if references.tendons else ()),
            *(("flex", "flexvert", "flexstrain") if references.names("flex") else ()),
        )
    return ()
