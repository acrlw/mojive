"""JSON schemas and value conversion for public application operations."""

from __future__ import annotations

import math
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, validators

from .adapters.base import AdapterCaps, PhysicsState
from .config import InteractionConfig, SelectionStyle
from .control_errors import ControlError
from .render.backend import DebugView, ShadowQuality
from .types import CameraView, Light, LightType, Material, MeshShape


def json_value(value: Any) -> Any:
    """Convert native contract values to the JSON representation accepted by operations."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _finite_number(_checker, value):
    try:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


# Non-finite numbers are not JSON values. Python's JSON decoder accepts them by
# default, so reject them at the same boundary as all other parameter errors.
Validator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine_many(
        {
            "number": _finite_number,
            "integer": lambda _checker, value: type(value) is int,
        }
    ),
)

NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}
STRING = {"type": "string"}
NAME = {"type": "string", "minLength": 1}
ID = {"type": "integer", "minimum": 0}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
COUNT = {"type": "integer", "minimum": 1}


def array(items: dict, length: int | None = None) -> dict:
    """Describe a homogeneous JSON array with an optional exact length."""
    result = {"type": "array", "items": items}
    if length is not None:
        result.update(minItems=length, maxItems=length)
    return result


def obj(properties: dict[str, dict] | None = None, required=(), **constraints) -> dict:
    """Describe strict named parameters; optional fields are omitted, not coerced."""
    return {
        "type": "object",
        "properties": properties or {},
        "required": list(required),
        "additionalProperties": False,
        **constraints,
    }


VECTOR = array(NUMBER)
VECTOR2 = array(NUMBER, 2)
VECTOR3 = array(NUMBER, 3)
ROTATION = array(VECTOR3, 3)
RGBA = array({"type": "number", "minimum": 0, "maximum": 1}, 4)
DOCUMENT = obj({"id": NAME, "revision": ID}, required=("id", "revision"))
PRECONDITION = obj({"id": NAME, "revision": ID}, required=("id",))
COMMAND_RESULT = {
    "type": "object",
    "required": ["ok", "message", "entity_id"],
    "properties": {"ok": BOOLEAN, "message": STRING, "entity_id": INTEGER, "document": DOCUMENT},
}
NODE = {
    "type": "object",
    "required": ["node_id", "object_id", "name", "type", "visible"],
    "properties": {
        "node_id": ID,
        "object_id": ID,
        "name": STRING,
        "type": STRING,
        "parent": INTEGER,
        "visible": {
            **BOOLEAN,
            "description": "Local node visibility flag; ancestors can hide this node.",
        },
        "posable": BOOLEAN,
        "source_editable": {
            **BOOLEAN,
            "description": (
                "Whether a compiled adapter node maps to an editable model-source element. "
                "Scene authoring has separate capabilities."
            ),
        },
        "body_index": INTEGER,
        "site_index": INTEGER,
    },
}
CAMERA_FIELDS = {
    "eye": VECTOR3,
    "target": VECTOR3,
    "up": VECTOR3,
    "fov_y": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": math.pi},
    "near": POSITIVE,
    "far": POSITIVE,
    "aspect": POSITIVE,
    "orthographic": BOOLEAN,
    "ortho_height": POSITIVE,
    "focal_length": array({"type": "number", "minimum": 0}, 2),
    "sensor_size": array({"type": "number", "minimum": 0}, 2),
    "principal_offset": VECTOR2,
    "orthographic_blend": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
}
CAMERA = obj(CAMERA_FIELDS)
CAMERA_PATCH = obj(
    {
        **CAMERA_FIELDS,
        "camera_id": ID,
        "source": {"type": "integer", "minimum": -1},
        "yaw": NUMBER,
        "pitch": NUMBER,
        "distance": POSITIVE,
    },
    minProperties=1,
)
CAPTURE_RESULT = {
    "type": "object",
    "required": ["path", "scope"],
    "properties": {
        "path": NAME,
        "scope": STRING,
        "shape": array(COUNT),
        "dtype": STRING,
        "mode": STRING,
        "orientation": {"const": "top_left"},
        "structure_generation": ID,
        "step": ID,
        "time": NUMBER,
        "document": DOCUMENT,
    },
}


def value_schema(value) -> dict:
    if hasattr(value, "__dataclass_fields__"):
        return obj({item.name: value_schema(getattr(value, item.name)) for item in fields(value)})
    if isinstance(value, (np.ndarray, tuple, list)):
        return array(value_schema(value[0]) if len(value) else NUMBER, len(value))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return BOOLEAN
    if isinstance(value, int):
        return INTEGER
    if isinstance(value, float):
        return NUMBER
    if value is None:
        return {"type": ["string", "null"]}
    return STRING


MATERIAL = value_schema(Material())
LIGHT = value_schema(Light())
LIGHT["properties"]["type"] = {
    "enum": [*[item.name.lower() for item in LightType], *[int(item) for item in LightType]]
}
LIGHT["properties"]["intensity"] = {"type": "number", "minimum": 0}


def record(properties: dict[str, dict]) -> dict:
    """Describe a result record with required fields and room for future additions."""
    return obj(properties, properties, additionalProperties=True)


CAMERA_RESULT = record(
    {
        **{name: schema for name, schema in CAMERA_FIELDS.items() if name != "orthographic_blend"},
        "format": NAME,
        "version": COUNT,
        "source": INTEGER,
        "yaw": NUMBER,
        "pitch": NUMBER,
        "distance": POSITIVE,
    }
)
GEOMETRY_RESULT = record(
    {
        "instance_index": {
            **ID,
            "description": "Render instance index scoped to structure_generation.",
        },
        "node_id": {**INTEGER, "description": "Geometry edit target; -1 means no hierarchy entry."},
        "object_id": ID,
        "mesh": record({"shape": {"enum": [item.value for item in MeshShape]}, "index": INTEGER}),
        "size": {
            **VECTOR3,
            "description": "Render-size vector for this instance; box values are half extents.",
        },
        "rgba": {
            **array(NUMBER, 4),
            "description": "Current composed instance color, including Session overrides.",
        },
        "material_index": {
            **INTEGER,
            "description": "Current scene material index; -1 selects the default material.",
        },
        "material": MATERIAL,
        "dimensions": {
            "anyOf": [record({"label": STRING, "values": VECTOR}), {"type": "null"}],
            "description": "Conventional primitive dimensions; null for unsupported shape families.",
        },
    }
)
INSPECTED_NODE = record(
    {
        **NODE["properties"],
        "position": VECTOR3,
        "rotation": ROTATION,
        "children": array(NODE),
        "document": DOCUMENT,
        "structure_generation": ID,
        "hierarchy_visible": {
            **BOOLEAN,
            "description": "Visibility inherited through the hierarchy; excludes render flags and occlusion.",
        },
        "geometries": {
            **array(GEOMETRY_RESULT),
            "description": "All render instances in this hierarchy subtree, including hidden geometry.",
        },
    }
)
BOUNDS_RESULT = record({"minimum": VECTOR3, "maximum": VECTOR3})
SCENE_RESULT = record(
    {
        "document": DOCUMENT,
        "asset": STRING,
        "structure_generation": ID,
        "bounds": BOUNDS_RESULT,
        "objects": array(NODE),
        "cameras": array(record({"camera_id": ID, "object_id": ID, "name": STRING})),
    }
)
PHYSICS_RESULT = record(
    {
        item.name: NUMBER if item.name == "time" else record({"shape": array(ID), "values": VECTOR})
        for item in fields(PhysicsState)
    }
)
STATE_RESULT = record(
    {
        "document": DOCUMENT,
        "asset": STRING,
        "paused": BOOLEAN,
        "active_keyframe": INTEGER,
        "selected_object_id": ID,
        "speed": POSITIVE,
        "step": ID,
        "time": NUMBER,
        "physics": {"anyOf": [PHYSICS_RESULT, {"type": "null"}]},
        "camera": CAMERA_RESULT,
    }
)
CAPABILITIES_RESULT = record(
    {
        "protocol_version": COUNT,
        "service": NAME,
        "viewer_attached": BOOLEAN,
        "deadline_clock": {"const": "monotonic"},
        "methods": array(NAME),
        "available_methods": array(NAME),
        "schema_dialect": NAME,
        "document": DOCUMENT,
        "capture_modes": array(NAME),
        "capture_scope": {"const": "session_scene"},
        "adapter": value_schema(AdapterCaps()),
    }
)
OPERATION_DESCRIPTION = record(
    {
        "name": NAME,
        "description": STRING,
        "scope": {"enum": ["scene", "capture", "viewport", "service"]},
        "mutates": BOOLEAN,
        "transactional": BOOLEAN,
        "input_schema": {
            "type": ["object", "boolean"],
            "description": "JSON Schema Draft 2020-12.",
        },
        "output_schema": {
            "type": ["object", "boolean"],
            "description": "JSON Schema Draft 2020-12.",
        },
        "requirements": record({"capabilities": array(NAME), "paused": BOOLEAN}),
        "available": BOOLEAN,
        "unavailable_reason": {"type": ["string", "null"]},
    }
)
OPERATION_DESCRIPTION["properties"]["alias_of"] = NAME
DESCRIPTION_RESULT = record(
    {"schema_dialect": NAME, "document": DOCUMENT, "operations": array(OPERATION_DESCRIPTION)}
)
CAPTURE_SETTINGS_RESULT = record(
    {
        "scope": {"const": "capture"},
        "camera": CAMERA_RESULT,
        "supported_render_flags": array(NAME),
        "supported_visualization_flags": array(NAME),
        "render_flag_overrides": {"type": "object", "additionalProperties": BOOLEAN},
        "debug_view": {"enum": [item.value for item in DebugView]},
        "dynamic_opacity": NUMBER,
        "modes": array(NAME),
    }
)
PANEL_RESULT = record({"id": NAME, "name": STRING, "open": BOOLEAN, "enabled": BOOLEAN})
VIEWER_SETTINGS_RESULT = record(
    {
        "interactions": value_schema(InteractionConfig()),
        "selection_style": value_schema(SelectionStyle()),
        "shadow_quality": {"enum": [item.value for item in ShadowQuality]},
        "layout": record({"persistence": BOOLEAN, "path": {"type": ["string", "null"]}}),
        "panels": array(PANEL_RESULT),
    }
)
COMMAND_RESULT["properties"].update(object_id=ID, camera_id=ID, light_id=ID, count=ID)


def validate(validator, value, method: str) -> None:
    """Raise a structured client error at the first rejected parameter path."""
    error = next(validator.iter_errors(value), None)
    if error is not None:
        path = "/" + "/".join(str(item) for item in error.absolute_path)
        raise ControlError(
            "invalid_params",
            f"{method}{path}: {error.message}",
            details={"method": method, "path": path, "rule": error.validator},
        )


def camera_value(value: dict) -> CameraView:
    """Decode and validate a shared camera, retaining roll and physical intrinsics."""
    view = CameraView(
        **{
            name: np.asarray(item, np.float32)
            if name
            in {
                "eye",
                "target",
                "up",
                "focal_length",
                "sensor_size",
                "principal_offset",
            }
            else item
            for name, item in value.items()
            if name in CAMERA_FIELDS
        }
    )
    forward = np.asarray(view.target) - np.asarray(view.eye)
    if view.far <= view.near:
        raise ControlError("invalid_params", "camera far must be greater than near")
    if np.linalg.norm(forward) < 1e-8 or np.linalg.norm(np.cross(forward, view.up)) < 1e-8:
        raise ControlError("invalid_params", "camera eye, target, and up must define a valid view")
    focal, sensor = np.asarray(view.focal_length), np.asarray(view.sensor_size)
    if (np.any(focal > 0) or np.any(sensor > 0)) and not (np.all(focal > 0) and np.all(sensor > 0)):
        raise ControlError(
            "invalid_params", "physical intrinsics require positive focal_length and sensor_size"
        )
    return view
