"""Capability requirements shared by local commands and remote discovery.

Scene appearance and selection remain Session-owned. Physics write-back and
source editing require explicit adapter contracts, independent of engine names.
"""

from __future__ import annotations

from . import commands as cmd
from .adapters.base import AdapterCaps

_REQUIREMENTS = {
    **dict.fromkeys(
        (cmd.CaptureSceneSnapshot, cmd.RestoreSceneSnapshot, cmd.RemoveSceneSnapshot),
        ("state_snapshots",),
    ),
    **dict.fromkeys(
        (
            cmd.Pause,
            cmd.Play,
            cmd.Step,
            cmd.SetSpeed,
        ),
        ("simulation",),
    ),
    **dict.fromkeys(
        (
            cmd.NewScene,
            cmd.OpenScene,
            cmd.SaveScene,
        ),
        ("scene_files",),
    ),
    **dict.fromkeys((cmd.SetPose,), ("write_pose",)),
    **dict.fromkeys(
        (
            cmd.SetJointProperties,
            cmd.SetJointAdvancedProperties,
            cmd.SetSiteProperties,
            cmd.SetGeometryProperties,
            cmd.SetGeometryAdvancedProperties,
            cmd.SetGeometryShape,
            cmd.SetBodyProperties,
        ),
        ("model_properties",),
    ),
    **dict.fromkeys(
        (
            cmd.ImportModelGeometryResource,
            cmd.ImportModelAsset,
            cmd.SetHeightFieldSize,
            cmd.RenameModelAsset,
            cmd.DuplicateModelAsset,
            cmd.ReplaceModelAssetFile,
            cmd.RemoveModelAsset,
            cmd.CreateModelMaterial,
            cmd.AddModelMaterial,
            cmd.ImportModelTexture,
            cmd.SetGeometryMaterial,
        ),
        ("model_assets",),
    ),
    **dict.fromkeys(
        (
            cmd.AddSceneObject,
            cmd.RemoveSceneObject,
            cmd.AddSceneLight,
            cmd.RemoveSceneLight,
            cmd.AddSceneCamera,
            cmd.RemoveSceneCamera,
            cmd.DuplicateSceneEntity,
            cmd.RemoveSceneEntity,
            cmd.RenameSceneEntity,
        ),
        ("scene_authoring",),
    ),
    **dict.fromkeys((cmd.LoadAsset,), ("asset_loading",)),
    **dict.fromkeys((cmd.Reload,), ("reload",)),
    **dict.fromkeys(
        (
            cmd.AddSceneModel,
            cmd.RemoveSceneModel,
            cmd.SetSceneModelTransform,
            cmd.PreviewSceneModelTransform,
            cmd.ClearSceneModelTransformPreview,
        ),
        ("model_composition",),
    ),
    **dict.fromkeys(
        (
            cmd.AddModelElement,
            cmd.DuplicateModelElement,
            cmd.RemoveModelElement,
            cmd.RenameModelElement,
            cmd.ModelEditBatch,
        ),
        ("topology_editing",),
    ),
    cmd.SetModelSource: ("topology_editing", "mujoco.mjcf"),
    **dict.fromkeys(
        (cmd.AddModelComponent, cmd.UpdateModelComponent, cmd.RemoveModelComponent),
        ("topology_editing", "model.components"),
    ),
    **dict.fromkeys(
        (cmd.AddModelKeyframe, cmd.SetModelKeyframe, cmd.RemoveModelKeyframe),
        ("keyframes", "topology_editing", "model.keyframe_edit"),
    ),
    cmd.LoadKeyframe: ("keyframes",),
    cmd.Perturb: ("perturb",),
    **dict.fromkeys((cmd.SetCtrl, cmd.SetCtrlVector), ("write_ctrl",)),
    **dict.fromkeys((cmd.SetQpos, cmd.SetQposBatch), ("write_qpos",)),
    cmd.SetEqualityEnabled: ("equality_constraints",),
    cmd.SetVisualGroup: ("visual_groups",),
}


def requirements(command_type: type[cmd.Command]) -> tuple[str, ...]:
    """Return required revision-1 adapter contracts for a typed command."""
    return _REQUIREMENTS.get(command_type, ())


def unavailable_reason(caps: AdapterCaps, command: cmd.Command) -> str | None:
    """Reject unsupported operations before history capture or physics fences."""
    for feature in requirements(type(command)):
        if not caps.supports(feature):
            if feature == "simulation":
                return f"{caps.name} has no simulation (revision 1)"
            return f"{caps.name} does not support {feature.replace('_', ' ')} (revision 1)"
    if (
        caps.simulation
        and not caps.clock_control
        and isinstance(command, (cmd.Pause, cmd.Play, cmd.Step, cmd.Reset, cmd.SetSpeed))
    ):
        return "Physics clock control belongs to the external caller"
    if isinstance(command, cmd.SetSpeed) and caps.external_clock:
        return "Simulation speed belongs to the external clock owner"
    if isinstance(command, (cmd.LoadAsset, cmd.AddSceneModel)) and not caps.accepts_model(
        command.path
    ):
        return f"{caps.name} does not support the model format: {command.path}"
    return None
