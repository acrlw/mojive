"""Capability requirements shared by local commands and remote discovery.

Scene appearance and selection remain Session-owned. Physics write-back and
source editing require explicit adapter contracts, independent of engine names.
"""

from __future__ import annotations

from functools import lru_cache

from mojive import commands as cmd
from mojive.adapters.base import AdapterCaps

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
    cmd.NewScene: ("scene_new",),
    cmd.OpenScene: ("scene_open",),
    cmd.SaveScene: ("scene_save",),
    **dict.fromkeys((cmd.SetPose,), ("write_pose",)),
    cmd.SetScale: ("write_scale",),
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
    cmd.SetPhysicsOptions: ("physics.options",),
    cmd.SetWorldSelection: ("world.selection",),
    cmd.SetReplayPlayback: ("replay.control",),
    cmd.SeekReplay: ("replay.control",),
    cmd.SyncRollout: ("replay.sync",),
    **dict.fromkeys((cmd.SetCtrl, cmd.SetCtrlVector), ("write_ctrl",)),
    **dict.fromkeys((cmd.SetQpos, cmd.SetQposBatch), ("write_qpos",)),
    cmd.SetEqualityEnabled: ("equality_constraints",),
    cmd.SetVisualGroup: ("visual_groups",),
}


@lru_cache(maxsize=128)
def resolve_command_type(command_type: type[cmd.Command]) -> type[cmd.Command]:
    """Return the first public command declaration in Python's resolution order.

    Dispatch, capability checks and pending edits share this policy so subclasses
    retain the contracts and edit behavior of the same command they extend.
    """
    return next(
        (
            base
            for base in command_type.__mro__
            if issubclass(base, cmd.Command) and getattr(cmd, base.__name__, None) is base
        ),
        command_type,
    )


def requirements(command_type: type[cmd.Command]) -> tuple[str, ...]:
    """Return required revision-1 adapter contracts for a typed command."""
    return _REQUIREMENTS.get(resolve_command_type(command_type), ())


def unavailable_reason(caps: AdapterCaps, command: cmd.Command) -> str | None:
    """Reject unsupported operations before history capture or physics fences."""
    command_type = resolve_command_type(type(command))
    for feature in requirements(command_type):
        if not caps.supports(feature):
            if feature == "simulation":
                return f"{caps.name} has no simulation (revision 1)"
            return f"{caps.name} does not support {feature.replace('_', ' ')} (revision 1)"
    if (
        caps.simulation
        and not caps.clock_control
        and command_type in (cmd.Pause, cmd.Play, cmd.Step, cmd.Reset, cmd.SetSpeed)
    ):
        return "Physics clock control belongs to the external caller"
    if command_type is cmd.SetSpeed and caps.external_clock:
        return "Simulation speed belongs to the external clock owner"
    if (
        command_type is cmd.Perturb
        and command.local_position is not None
        and not caps.supports("physics.perturb_point")
    ):
        return f"{caps.name} does not support physics.perturb_point (revision 1)"
    if command_type in (cmd.LoadAsset, cmd.AddSceneModel) and not caps.accepts_model(command.path):
        return f"{caps.name} does not support the model format: {command.path}"
    if (
        command_type is cmd.Perturb
        and command.strength != 1.0
        and not caps.supports("physics.perturb_strength")
    ):
        return f"{caps.name} does not support physics.perturb_strength (revision 1)"
    return None
