"""Inspector: support."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import math3d
from mojive.adapters.base import (
    ModelComponentPathItem,
    NodeType,
    SceneNode,
)
from mojive.ui.panels import (
    PanelContext,
)

GIZMO_REFUSAL_RUNNING = "Physics is running; pause to move things"
GIZMO_REFUSAL_DRIVEN = "This link is joint-driven; use its joint gizmo or the Joints panel"
_MATERIAL_PRESETS = {
    "Matte": (0.0, 0.05, 0.10, 0.0),
    "Plastic": (0.0, 0.40, 0.50, 0.05),
    "Metal": (0.0, 0.90, 0.90, 0.65),
    "Rubber": (0.0, 0.08, 0.20, 0.0),
    "Emissive": (1.0, 0.10, 0.20, 0.0),
}


def _unique_component_name(category: str, existing: set[str]) -> str:
    if category not in existing:
        return category
    index = 2
    while f"{category}{index}" in existing:
        index += 1
    return f"{category}{index}"


_MULTILINE_COMPONENT_FIELDS = {
    "act",
    "body",
    "cellcount",
    "ctrl",
    "data",
    "element",
    "elemtexcoord",
    "face",
    "mpos",
    "mquat",
    "node",
    "point",
    "qpos",
    "qvel",
    "texcoord",
    "vertex",
    "vertid",
    "vertweight",
}


def _component_value_editor(
    ctx: PanelContext,
    label: str,
    value: str,
    choices: tuple[str, ...],
    *,
    multiline: bool = False,
) -> str:
    if choices:
        if imgui.begin_combo(label, value or ctx.tr("select")):
            for index, choice in enumerate(choices):
                selected, _ = imgui.selectable(
                    f"{choice or ctx.tr('<default>')}##choice-{index}",
                    choice == value,
                )
                if selected:
                    value = choice
            imgui.end_combo()
        return value
    if multiline:
        _changed, value = imgui.input_text_multiline(
            label,
            value,
            imgui.ImVec2(-1.0, 68.0),
        )
        return value
    _changed, value = imgui.input_text(label, value)
    return value


_MODEL_COMPONENT_CATEGORIES = (
    "contact",
    "actuator",
    "sensor",
    "tendon",
    "equality",
)


def _path_preset_label(preset: ModelComponentPathItem) -> str:
    object_type = next((field.value for field in preset.fields if field.name == "objtype"), "")
    return f"{preset.type} · {object_type}" if object_type else preset.type


def _matching_path_preset(
    presets: tuple[ModelComponentPathItem, ...],
    element_type: str,
    fields: list[list[str]],
) -> ModelComponentPathItem | None:
    object_type = next((value for name, value in fields if name == "objtype"), "")
    if not object_type:
        return None
    return next(
        (
            preset
            for preset in presets
            if preset.type == element_type
            and any(
                field.name == "objtype" and field.value == object_type for field in preset.fields
            )
        ),
        None,
    )


def gizmo_refusal_reason(
    paused: bool,
    posable: bool,
) -> str | None:
    if not paused:
        return GIZMO_REFUSAL_RUNNING
    if not posable:
        return GIZMO_REFUSAL_DRIVEN
    return None


def _body_pose(xpos, xmat, body_index: int):
    if xpos is None or body_index < 0 or body_index >= len(xpos):
        return None, None
    mat = xmat[body_index] if xmat is not None and body_index < len(xmat) else None
    return xpos[body_index], mat


def _node_pose(frame, node: SceneNode):
    if node.type is NodeType.GEOM:
        return _body_pose(frame.geom_xpos, frame.geom_xmat, node.geom_index)
    if node.type is NodeType.SITE:
        return _body_pose(frame.site_xpos, frame.site_xmat, node.site_index)
    return _body_pose(frame.body_xpos, frame.body_xmat, node.body_index)


def _pose_editable(write_pose: bool, paused: bool, posable: bool) -> bool:
    return bool(write_pose and paused and posable)


def _nearest_euler_degrees(matrix, reference=None) -> np.ndarray:
    base = np.degrees(math3d.mat3_to_euler_xyz(matrix)).astype(np.float64)
    if reference is None:
        return base
    reference = np.asarray(reference, np.float64)
    alternate = np.array((base[0] + 180.0, 180.0 - base[1], base[2] + 180.0))
    candidates = []
    for value in (base, alternate):
        candidates.append(value + 360.0 * np.round((reference - value) / 360.0))
    return min(candidates, key=lambda value: float(np.linalg.norm(value - reference)))


def _free_velocity(qvel, joints, body_index: int):
    if qvel is None:
        return None
    joint = next(
        (j for j in joints if j.body == body_index and j.type == "free" and j.dof >= 6), None
    )
    if joint is None or joint.qvel_adr + 6 > len(qvel):
        return None
    values = np.asarray(qvel[joint.qvel_adr : joint.qvel_adr + 6], np.float64)
    return values[:3], values[3:]


def _has_free_velocity(joints, body_index: int) -> bool:
    return any(j.body == body_index and j.type == "free" and j.dof >= 6 for j in joints)
