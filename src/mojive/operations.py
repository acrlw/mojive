"""One catalog for application command routing, schemas, and capability discovery."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as cmd
from .config import InteractionConfig, SelectionStyle
from .control_errors import ControlError
from .control_schema import (
    BOOLEAN,
    CAMERA,
    CAMERA_PATCH,
    CAMERA_RESULT,
    CAPABILITIES_RESULT,
    CAPTURE_RESULT,
    CAPTURE_SETTINGS_RESULT,
    COMMAND_RESULT,
    COUNT,
    DESCRIPTION_RESULT,
    ID,
    INSPECTED_NODE,
    INTEGER,
    LIGHT,
    MATERIAL,
    NAME,
    NODE,
    NUMBER,
    PANEL_RESULT,
    POSITIVE,
    PRECONDITION,
    RGBA,
    ROTATION,
    SCENE_RESULT,
    STATE_RESULT,
    STRING,
    VECTOR,
    VECTOR3,
    VIEWER_SETTINGS_RESULT,
    Validator,
    array,
    camera_value,
    json_value,
    obj,
    validate,
    value_schema,
)
from .render.backend import RenderFlag, RenderProduct, ShadowQuality
from .types import CameraView, Light, LightType, Material, MeshKey, MeshShape

CAPTURE_PRODUCTS = {
    "rgb": RenderProduct.COLOR,
    "depth": RenderProduct.METRIC_DEPTH,
    "object_id": RenderProduct.OBJECT_ID,
    "segmentation": RenderProduct.SEGMENTATION,
}
CAPTURE_RENDER_FLAGS = frozenset(
    {
        RenderFlag.SHADOW,
        RenderFlag.WIREFRAME,
        RenderFlag.REFLECTION,
        RenderFlag.ADDITIVE,
        RenderFlag.SKYBOX,
        RenderFlag.FOG,
        RenderFlag.HAZE,
        RenderFlag.CULL_FACE,
    }
)
CAPTURE_VISUAL_FLAGS = (
    frozenset(RenderFlag)
    - CAPTURE_RENDER_FLAGS
    - {
        RenderFlag.OUTLINE,
        RenderFlag.TONEMAP,
        RenderFlag.MSAA,
    }
)


COMMAND_ID_FIELDS = {
    cmd.AddSceneObject: "object_id",
    cmd.RemoveSceneObject: "object_id",
    cmd.DuplicateSceneEntity: "object_id",
    cmd.RemoveSceneEntity: "object_id",
    cmd.RenameSceneEntity: "object_id",
    cmd.AddSceneCamera: "camera_id",
    cmd.RemoveSceneCamera: "camera_id",
    cmd.AddSceneLight: "light_id",
    cmd.RemoveSceneLight: "light_id",
}


def _light(value):
    values = dict(value)
    kind = values.get("type", LightType.DIRECTIONAL)
    values["type"] = LightType[kind.upper()] if isinstance(kind, str) else LightType(kind)
    return Light(**values)


def _rotation(value):
    rotation = np.asarray(value, np.float32)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0:
        raise ControlError(
            "invalid_params", "rotation must be an orthonormal 3x3 matrix with positive determinant"
        )
    return rotation


def _command_factory(kind):
    def build(params):
        values = dict(params)
        if "shape" in values and not isinstance(values["shape"], MeshKey):
            shape = values["shape"]
            values["shape"] = (
                MeshKey(MeshShape(shape["shape"]), shape.get("index", -1))
                if isinstance(shape, dict)
                else MeshShape(shape)
            )
        if "rotation" in values:
            values["rotation"] = _rotation(values["rotation"])
        if "position" in values:
            values["position"] = np.asarray(values["position"], np.float32)
        if "camera" in values and not isinstance(values["camera"], CameraView):
            values["camera"] = camera_value(values["camera"])
        if "light" in values and not isinstance(values["light"], Light):
            values["light"] = _light(values["light"])
        if "material" in values and not isinstance(values["material"], Material):
            values["material"] = Material(**values["material"])
        if "path" in values:
            values["path"] = Path(values["path"]).expanduser().resolve()
        return kind(**values)

    return build


@dataclass(frozen=True)
class Operation:
    """Public operation definition shared by validation, dispatch, and discovery."""

    name: str
    description: str
    input_schema: dict
    output_schema: dict
    handler: str = ""
    command: Callable[[dict], cmd.Command] | None = None
    scope: str = "scene"
    mutates: bool = False
    capabilities: tuple[str, ...] = ()
    paused: bool = False
    transactional: bool = False
    alias_of: str | None = None
    validator: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        schema = deepcopy(self.input_schema)
        if self.mutates and self.scope == "scene":
            schema["properties"]["expected_document"] = PRECONDITION
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(self, "validator", Validator(schema))

    def validate(self, params: dict) -> dict:
        """Validate parameters without applying defaults or mutating caller data."""
        values = json_value(params)
        validate(self.validator, values, self.name)
        return values

    def unavailable_reason(self, session, *, viewer_attached: bool = False) -> str | None:
        """Explain why this operation cannot run in the current application state."""
        if self.scope == "viewport" and not viewer_attached:
            return "This operation requires an attached viewer"
        for capability in self.capabilities:
            if not getattr(session.adapter.caps, capability):
                return f"The {session.adapter.caps.name} adapter does not support {capability}"
        if self.paused and not session.paused:
            return "Pause the simulation before this operation"
        if self.name == "undo" and not session.can_undo:
            return "Nothing to undo"
        if self.name == "redo" and not session.can_redo:
            return "Nothing to redo"
        if self.name in {"undo", "redo", "edit_scene"} and session.editing:
            return "Finish the active edit transaction first"
        if self.name == "set_keyframe" and not session.keyframes:
            return "The scene contains no keyframes"
        if self.name == "set_ctrl" and not session.actuators:
            return "The scene contains no actuator controls"
        return None

    def describe(self, session, *, viewer_attached: bool = False) -> dict:
        """Return machine-readable schemas, scope, and current availability."""
        reason = self.unavailable_reason(session, viewer_attached=viewer_attached)
        result = {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "mutates": self.mutates,
            "transactional": self.transactional,
            "input_schema": deepcopy(self.input_schema),
            "output_schema": deepcopy(self.output_schema),
            "requirements": {"capabilities": list(self.capabilities), "paused": self.paused},
            "available": reason is None,
            "unavailable_reason": reason,
        }
        if self.alias_of:
            result["alias_of"] = self.alias_of
        return result


def _op(name, description, params=None, required=(), *, result=COMMAND_RESULT, **options):
    return Operation(name, description, obj(params, required), result, **options)


def _cmd(name, kind, params=None, required=(), **options):
    properties = deepcopy(params or {})
    for item in fields(kind):
        if item.name in properties:
            default = item.default
            if default is MISSING and item.default_factory is not MISSING:
                default = item.default_factory()
            if default is not MISSING:
                properties[item.name]["default"] = json_value(default)
    result = deepcopy(COMMAND_RESULT)
    identity = COMMAND_ID_FIELDS.get(kind)
    if kind in (cmd.Select, cmd.SelectNode):
        result["properties"]["object"] = {"anyOf": [NODE, {"type": "null"}]}
        result["required"].append("object")
    if identity:
        result["properties"][identity] = ID
        result["required"].append(identity)
    return _op(
        name,
        kind.__doc__.split("\n")[0],
        properties,
        required,
        result=result,
        command=_command_factory(kind),
        mutates=True,
        **options,
    )


_EDIT = {"capabilities": ("scene_authoring",), "paused": True, "transactional": True}
_HISTORY = {"capabilities": ("edit_history",)}
_VIEW = {"scope": "viewport"}
_PHYSICS = {"capabilities": ("simulation",)}
_STATE = {"capabilities": ("state_snapshots",), "paused": True}
_CAPTURE_PARAMS = {
    "mode": {"enum": list(CAPTURE_PRODUCTS), "default": "rgb"},
    "width": {**COUNT, "default": 640},
    "height": {**COUNT, "default": 480},
    "output": NAME,
}
_SCALAR_OR_VECTOR = obj(
    {"index": ID, "value": NUMBER, "values": VECTOR},
    oneOf=[
        {"required": ["index", "value"], "not": {"required": ["values"]}},
        {
            "required": ["values"],
            "not": {"anyOf": [{"required": ["index"]}, {"required": ["value"]}]},
        },
    ],
)
_CATALOG = [
    _op(
        "hello",
        "Describe the service and its recognized and available operations.",
        result=CAPABILITIES_RESULT,
        handler="_capabilities",
        scope="service",
    ),
    _op(
        "get_capabilities",
        "Describe the service and adapter capabilities.",
        result=CAPABILITIES_RESULT,
        handler="_capabilities",
        scope="service",
        alias_of="hello",
    ),
    _op(
        "describe_operations",
        "Describe operation schemas and current availability; filter by name or scope.",
        {
            "name": NAME,
            "scope": {"enum": ["scene", "capture", "viewport", "service"]},
            "available_only": BOOLEAN,
        },
        result=DESCRIPTION_RESULT,
        handler="_describe_operations",
        scope="service",
    ),
    _op(
        "get_scene",
        "Read current hierarchy, camera IDs, bounds, and document identity.",
        result=SCENE_RESULT,
        handler="_scene",
    ),
    _op(
        "get_state",
        "Read simulation, selection, capture camera, and document state.",
        result=STATE_RESULT,
        handler="_state",
    ),
    _op(
        "get_bounds",
        "Read the scene's world-space bounds.",
        result=obj({"minimum": VECTOR3, "maximum": VECTOR3}),
        handler="_bounds",
    ),
    _op(
        "list_objects",
        "List current hierarchy nodes and selection identities.",
        result=array(NODE),
        handler="_list_objects",
    ),
    Operation(
        "inspect_object",
        "Inspect hierarchy state, world transform, and composed geometry appearance.",
        obj(
            {"object_id": ID, "node_id": ID},
            oneOf=[{"required": ["object_id"]}, {"required": ["node_id"]}],
        ),
        INSPECTED_NODE,
        handler="_inspect_object",
    ),
    _cmd("select_object", cmd.Select, {"object_id": ID}, ("object_id",), handler="_select_object"),
    _cmd("select_node", cmd.SelectNode, {"node_id": ID}, ("node_id",), handler="_select_node"),
    _cmd(
        "set_visible", cmd.SetVisible, {"node_id": ID, "visible": BOOLEAN}, ("node_id", "visible")
    ),
    _cmd("pause", cmd.Pause, **_PHYSICS),
    _cmd("resume", cmd.Play, handler="_resume", **_PHYSICS),
    _op(
        "step",
        "Apply optional controls, step a paused simulation, and optionally return state.",
        {"count": {**COUNT, "default": 1}, "ctrl": VECTOR, "observe": BOOLEAN},
        result={
            **deepcopy(COMMAND_RESULT),
            "properties": {**COMMAND_RESULT["properties"], "state": STATE_RESULT},
        },
        command=lambda p: cmd.Step(p.get("count", 1)),
        handler="_step",
        mutates=True,
        paused=True,
        **_PHYSICS,
    ),
    _cmd("reset", cmd.Reset),
    _cmd("set_speed", cmd.SetSpeed, {"factor": POSITIVE}, ("factor",), **_PHYSICS),
    _cmd(
        "set_keyframe",
        cmd.LoadKeyframe,
        {"keyframe_id": ID},
        ("keyframe_id",),
        capabilities=("keyframes",),
        paused=True,
    ),
    Operation(
        "set_qpos",
        "Set generalized positions by index or complete vector.",
        _SCALAR_OR_VECTOR,
        COMMAND_RESULT,
        handler="_set_qpos",
        command=lambda p: cmd.SetQpos(p["index"], p["value"]),
        mutates=True,
        capabilities=("write_qpos",),
        paused=True,
    ),
    _op(
        "set_qvel",
        "Replace the generalized velocity vector.",
        {"values": VECTOR},
        ("values",),
        handler="_set_qvel",
        mutates=True,
        **_STATE,
    ),
    Operation(
        "set_ctrl",
        "Set actuator controls by index or complete vector.",
        _SCALAR_OR_VECTOR,
        COMMAND_RESULT,
        handler="_set_ctrl",
        command=lambda p: cmd.SetCtrl(p["index"], p["value"]),
        mutates=True,
        **_PHYSICS,
    ),
    Operation(
        "set_mocap",
        "Replace mocap position and/or quaternion arrays while paused.",
        obj(
            {"position": array(VECTOR3), "quaternion": array(array(NUMBER, 4))},
            anyOf=[{"required": ["position"]}, {"required": ["quaternion"]}],
        ),
        COMMAND_RESULT,
        handler="_set_mocap",
        mutates=True,
        **_STATE,
    ),
    Operation(
        "set_state",
        "Restore supplied physics fields while paused.",
        obj(
            {
                "qpos": VECTOR,
                "qvel": VECTOR,
                "act": VECTOR,
                "ctrl": VECTOR,
                "mocap_pos": array(VECTOR3),
                "mocap_quat": array(array(NUMBER, 4)),
                "time": NUMBER,
                "active_keyframe": INTEGER,
            },
            minProperties=1,
        ),
        COMMAND_RESULT,
        handler="_set_state_fields",
        mutates=True,
        **_STATE,
    ),
    _op(
        "load",
        "Replace the active model or workspace.",
        {"path": NAME},
        ("path",),
        handler="_load",
        mutates=True,
        capabilities=("asset_loading",),
    ),
    _cmd("reload", cmd.Reload, handler="_reload", capabilities=("reload",)),
    _cmd("new_scene", cmd.NewScene, capabilities=("scene_files",)),
    _cmd("open_scene", cmd.OpenScene, {"path": NAME}, ("path",), capabilities=("scene_files",)),
    _cmd(
        "save_scene",
        cmd.SaveScene,
        {"path": NAME, "current_pose_keyframe": {"type": ["string", "null"]}},
        ("path",),
        capabilities=("scene_files",),
    ),
    _cmd(
        "set_pose",
        cmd.SetPose,
        {"node_id": ID, "position": VECTOR3, "rotation": ROTATION},
        ("node_id", "position", "rotation"),
        capabilities=("write_pose",),
        paused=True,
        transactional=True,
    ),
    _cmd(
        "add_scene_object",
        cmd.AddSceneObject,
        {
            "shape": {
                "oneOf": [
                    {
                        "enum": [
                            item.value
                            for item in MeshShape
                            if item
                            not in {
                                MeshShape.HEIGHTFIELD,
                                MeshShape.FLEX,
                                MeshShape.FLEX_FACE,
                                MeshShape.SKIN,
                                MeshShape.ASSET,
                            }
                        ]
                    },
                    obj(
                        {
                            "shape": {"enum": [item.value for item in MeshShape]},
                            "index": {"type": "integer", "minimum": -1},
                        },
                        ("shape",),
                    ),
                ]
            },
            "name": NAME,
            "size": array(POSITIVE, 3),
            "position": VECTOR3,
            "rotation": ROTATION,
            "color": RGBA,
            "material": MATERIAL,
        },
        ("shape",),
        **_EDIT,
    ),
    _cmd("remove_scene_object", cmd.RemoveSceneObject, {"object_id": ID}, ("object_id",), **_EDIT),
    _cmd(
        "duplicate_scene_entity",
        cmd.DuplicateSceneEntity,
        {"object_id": ID},
        ("object_id",),
        **_EDIT,
    ),
    _cmd("remove_scene_entity", cmd.RemoveSceneEntity, {"object_id": ID}, ("object_id",), **_EDIT),
    _cmd(
        "rename_scene_entity",
        cmd.RenameSceneEntity,
        {"object_id": ID, "name": NAME},
        ("object_id", "name"),
        **_EDIT,
    ),
    _cmd(
        "set_geometry_color",
        cmd.SetGeometryColor,
        {"node_id": ID, "rgba": RGBA},
        ("node_id", "rgba"),
        paused=True,
        transactional=True,
    ),
    _cmd(
        "set_geometry_size",
        cmd.SetGeometrySize,
        {"node_id": ID, "size": array(POSITIVE, 3)},
        ("node_id", "size"),
        paused=True,
        transactional=True,
    ),
    _cmd(
        "add_scene_camera",
        cmd.AddSceneCamera,
        {"name": NAME, "camera": CAMERA},
        ("name", "camera"),
        **_EDIT,
    ),
    _cmd(
        "set_scene_camera",
        cmd.SetSceneCamera,
        {"camera_id": ID, "camera": CAMERA},
        ("camera_id", "camera"),
        capabilities=("model_cameras",),
        paused=True,
        transactional=True,
    ),
    _cmd("remove_scene_camera", cmd.RemoveSceneCamera, {"camera_id": ID}, ("camera_id",), **_EDIT),
    _cmd(
        "add_scene_light",
        cmd.AddSceneLight,
        {"name": NAME, "light": LIGHT},
        ("name", "light"),
        **_EDIT,
    ),
    _cmd("remove_scene_light", cmd.RemoveSceneLight, {"light_id": ID}, ("light_id",), **_EDIT),
    _cmd("undo", cmd.Undo, **_HISTORY),
    _cmd("redo", cmd.Redo, **_HISTORY),
    _op(
        "edit_scene",
        "Apply supported edits as one atomic undo record; roll back on any failure.",
        {
            "label": NAME,
            "operations": {
                "type": "array",
                "minItems": 1,
                "items": obj({"method": NAME, "params": {"type": "object"}}, ("method", "params")),
            },
        },
        ("operations",),
        result={
            **deepcopy(COMMAND_RESULT),
            "properties": {**COMMAND_RESULT["properties"], "results": array(COMMAND_RESULT)},
            "required": [*COMMAND_RESULT["required"], "results"],
        },
        handler="_edit_scene",
        mutates=True,
        paused=True,
        **_HISTORY,
    ),
    Operation(
        "set_capture_camera",
        "Set an independent offscreen camera or select a scene camera.",
        CAMERA_PATCH,
        CAMERA_RESULT,
        handler="_set_camera",
        scope="capture",
        mutates=True,
    ),
    Operation(
        "set_camera",
        "Legacy alias for the independent capture camera.",
        CAMERA_PATCH,
        CAMERA_RESULT,
        handler="_set_camera",
        scope="capture",
        mutates=True,
        alias_of="set_capture_camera",
    ),
    _op(
        "get_capture_settings",
        "Read offscreen camera, render flags, and image products.",
        result=CAPTURE_SETTINGS_RESULT,
        handler="_capture_settings",
        scope="capture",
    ),
    _op(
        "load_camera_bookmark",
        "Load a saved bookmark into the capture camera.",
        {"name": NAME, "directory": NAME},
        ("name",),
        result=CAMERA_RESULT,
        handler="_load_camera_bookmark",
        scope="capture",
        mutates=True,
    ),
    _cmd(
        "set_visual_group",
        cmd.SetVisualGroup,
        {"category": NAME, "group": ID, "visible": BOOLEAN},
        ("category", "group", "visible"),
        capabilities=("visual_groups",),
    ),
    _op(
        "set_render_flag",
        "Set a capture rendering feature or legacy RGB debug view.",
        {
            "name": {
                **NAME,
                "examples": [
                    *sorted(flag.value for flag in CAPTURE_RENDER_FLAGS),
                    "depth",
                    "idcolor",
                    "segment",
                ],
            },
            "enabled": BOOLEAN,
        },
        ("name", "enabled"),
        result=obj({"name": STRING, "enabled": BOOLEAN}),
        handler="_set_render_flag",
        scope="capture",
        mutates=True,
    ),
    _op(
        "set_visualization_flag",
        "Set a capture visualization feature.",
        {
            "name": {**NAME, "examples": sorted(flag.value for flag in CAPTURE_VISUAL_FLAGS)},
            "enabled": BOOLEAN,
        },
        ("name", "enabled"),
        result=obj({"name": STRING, "enabled": BOOLEAN}),
        handler="_set_visualization_flag",
        scope="capture",
        mutates=True,
    ),
    _op(
        "capture",
        "Render the current composed scene to RGB, metric depth, selection IDs, or semantic pairs.",
        _CAPTURE_PARAMS,
        result=CAPTURE_RESULT,
        handler="_capture",
        scope="capture",
    ),
    _op(
        "capture_viewport",
        "Save the next fully presented viewport or window; wait for the completed artifact.",
        {"surface": {"enum": ["viewport", "window"], "default": "viewport"}, "output": NAME},
        result=CAPTURE_RESULT,
        handler="_capture_viewport",
        **_VIEW,
    ),
    Operation(
        "set_viewport_camera",
        "Set the visible viewer camera; preserve independent capture settings.",
        CAMERA_PATCH,
        CAMERA_RESULT,
        handler="_set_viewport_camera",
        mutates=True,
        **_VIEW,
    ),
    _op(
        "get_viewport_camera",
        "Read the viewer's current camera from Session state.",
        result=CAMERA_RESULT,
        handler="_viewport_camera",
        **_VIEW,
    ),
    _op(
        "get_viewer_settings",
        "Read viewer interactions, selection style, panels, and shadow quality.",
        result=VIEWER_SETTINGS_RESULT,
        handler="_viewer_settings",
        **_VIEW,
    ),
    _op(
        "set_interactions",
        "Update viewer input ownership; persist only when explicitly requested.",
        {**value_schema(InteractionConfig())["properties"], "persist": BOOLEAN},
        result=value_schema(InteractionConfig()),
        handler="_set_interactions",
        mutates=True,
        **_VIEW,
    ),
    _op(
        "set_selection_style",
        "Update how the viewer displays the current selection.",
        {**value_schema(SelectionStyle())["properties"], "persist": BOOLEAN},
        result=value_schema(SelectionStyle()),
        handler="_set_selection_style",
        mutates=True,
        **_VIEW,
    ),
    _op(
        "set_shadow_quality",
        "Set the viewer's shadow quality preset.",
        {"quality": {"enum": [item.value for item in ShadowQuality]}, "persist": BOOLEAN},
        ("quality",),
        result=obj({"quality": STRING}),
        handler="_set_shadow_quality",
        mutates=True,
        **_VIEW,
    ),
    _op(
        "reset_layout",
        "Rebuild the viewer's default layout.",
        result=obj({"reset": BOOLEAN}),
        handler="_reset_layout",
        mutates=True,
        **_VIEW,
    ),
    _op(
        "get_panels",
        "List viewer panel IDs and current open/enabled state.",
        result=array(PANEL_RESULT),
        handler="_panels",
        **_VIEW,
    ),
    Operation(
        "set_panel",
        "Update one viewer panel's open/enabled state.",
        obj(
            {"id": NAME, "open": BOOLEAN, "enabled": BOOLEAN},
            ("id",),
            anyOf=[{"required": ["open"]}, {"required": ["enabled"]}],
        ),
        PANEL_RESULT,
        handler="_set_panel",
        mutates=True,
        **_VIEW,
    ),
]

OPERATIONS = {operation.name: operation for operation in _CATALOG}


def document_state(session) -> dict:
    """Return the document identity and authored history revision used by preconditions."""
    return {"id": session.document_id, "revision": session.document_revision}


def prepare_operation(
    session, name: str, params: dict, *, viewer_attached=False
) -> tuple[Operation, dict]:
    """Validate shape, applicability, and document identity before constructing a command."""
    operation = OPERATIONS.get(name)
    if operation is None:
        raise ControlError("unknown_method", f"Unknown control method: {name}")
    values = operation.validate(params)
    expected = values.pop("expected_document", None)
    _check_document_precondition(session, expected)
    reason = operation.unavailable_reason(session, viewer_attached=viewer_attached)
    if reason:
        raise ControlError("unsupported", reason, details={"method": name})
    return operation, values


def _check_document_precondition(session, expected) -> None:
    """Check document identity before either transport constructs a native command."""
    if expected is not None:
        actual = document_state(session)
        if expected["id"] != actual["id"] or (
            "revision" in expected and expected["revision"] != actual["revision"]
        ):
            raise ControlError(
                "stale_document",
                "The document changed; inspect the current scene before editing",
                details={"expected": expected, "actual": actual},
            )


def apply_session_operation(session, name: str, params: dict) -> cmd.CommandResult:
    """Apply a cataloged typed command, sharing semantics with native remote publishers."""
    operation = OPERATIONS.get(name)
    if operation is None:
        raise ControlError("unknown_method", f"Unknown control method: {name}")
    values = operation.validate(params)
    _check_document_precondition(session, values.pop("expected_document", None))
    if operation.command is None:
        raise ControlError("unsupported", f"{name} requires an application service")
    # Preserve native CameraView/Material/mesh values after validating their JSON form.
    return session.submit(operation.command({name: params[name] for name in values}))
