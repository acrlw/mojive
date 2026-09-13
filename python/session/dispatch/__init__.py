"""Typed command routing; handlers mutate only the owning Session."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from mojive import commands as cmd
from mojive.commands import Command, CommandResult

from . import assets, documents, keyframes, physics, playback, properties, scene, transforms

if TYPE_CHECKING:
    from ..core import Session

_HANDLERS = {
    cmd.CaptureSceneSnapshot: playback.scene_snapshot,
    cmd.RestoreSceneSnapshot: playback.scene_snapshot,
    cmd.RemoveSceneSnapshot: playback.scene_snapshot,
    cmd.StartStateTakeRecording: playback.start_state_take_recording,
    cmd.StopStateTakeRecording: playback.stop_state_take_recording,
    cmd.PlayStateTake: playback.play_state_take,
    cmd.PauseStateTake: playback.pause_state_take,
    cmd.SeekStateTake: playback.seek_state_take,
    cmd.SetStateTakeLoop: playback.set_state_take_loop,
    cmd.ClearStateTake: playback.clear_state_take,
    cmd.Pause: playback.pause,
    cmd.Play: playback.play,
    cmd.Step: playback.step,
    cmd.StepBack: playback.step_back,
    cmd.Reset: playback.reset,
    cmd.Reload: documents.reload,
    cmd.NewScene: documents.new_scene,
    cmd.OpenScene: documents.open_scene,
    cmd.SaveScene: documents.save_scene,
    cmd.LoadAsset: documents.load_asset,
    cmd.AddSceneModel: documents.add_scene_model,
    cmd.RemoveSceneModel: documents.remove_scene_model,
    cmd.SetSceneModelTransform: documents.set_scene_model_transform,
    cmd.PreviewSceneModelTransform: documents.preview_scene_model_transform,
    cmd.ClearSceneModelTransformPreview: documents.clear_scene_model_transform_preview,
    cmd.AddModelElement: documents.add_model_element,
    cmd.DuplicateModelElement: documents.duplicate_model_element,
    cmd.RemoveModelElement: documents.remove_model_element,
    cmd.RenameModelElement: documents.rename_model_element,
    cmd.ModelEditBatch: documents.model_edit_batch,
    cmd.SetModelSource: documents.set_model_source,
    cmd.AddModelComponent: documents.add_model_component,
    cmd.UpdateModelComponent: documents.update_model_component,
    cmd.RemoveModelComponent: documents.remove_model_component,
    cmd.AddResourceRoot: documents.add_resource_root,
    cmd.RemoveResourceRoot: documents.remove_resource_root,
    cmd.LoadKeyframe: keyframes.load_keyframe,
    cmd.AddModelKeyframe: keyframes.add_model_keyframe,
    cmd.SetModelKeyframe: keyframes.set_model_keyframe,
    cmd.RemoveModelKeyframe: keyframes.remove_model_keyframe,
    cmd.Select: transforms.select,
    cmd.SelectNode: transforms.select_node,
    cmd.SetVisible: transforms.set_visible,
    cmd.SetVisualGroup: transforms.set_visual_group,
    cmd.SetPose: transforms.set_pose,
    cmd.SetScale: transforms.set_scale,
    cmd.SetQpos: transforms.set_qpos,
    cmd.SetQposBatch: transforms.set_qpos_batch,
    cmd.SetJointProperties: properties.set_joint_properties,
    cmd.SetJointAdvancedProperties: properties.set_joint_advanced_properties,
    cmd.SetSiteProperties: properties.set_site_properties,
    cmd.SetGeometryProperties: properties.set_geometry_properties,
    cmd.SetGeometryAdvancedProperties: properties.set_geometry_advanced_properties,
    cmd.SetGeometryShape: properties.set_geometry_shape,
    cmd.ImportModelGeometryResource: assets.import_model_geometry_resource,
    cmd.ImportModelAsset: assets.import_model_asset,
    cmd.SetHeightFieldSize: assets.set_height_field_size,
    cmd.RenameModelAsset: assets.rename_model_asset,
    cmd.DuplicateModelAsset: assets.duplicate_model_asset,
    cmd.ReplaceModelAssetFile: assets.replace_model_asset_file,
    cmd.RemoveModelAsset: assets.remove_model_asset,
    cmd.SetBodyProperties: properties.set_body_properties,
    cmd.CreateModelMaterial: assets.create_model_material,
    cmd.AddModelMaterial: assets.add_model_material,
    cmd.ImportModelTexture: assets.import_model_texture,
    cmd.SetGeometryMaterial: assets.set_geometry_material,
    cmd.SetEqualityEnabled: physics.set_equality_enabled,
    cmd.SetCtrlVector: physics.set_ctrl_vector,
    cmd.SetCtrl: physics.set_ctrl,
    cmd.Perturb: physics.perturb,
    cmd.ClearPerturb: physics.clear_perturb,
    cmd.SetLight: scene.set_light,
    cmd.SetEnvironment: scene.set_environment,
    cmd.SetSkybox: scene.set_skybox,
    cmd.SetMaterial: scene.set_material,
    cmd.SetGeometryColor: scene.set_geometry_color,
    cmd.SetGeometrySize: scene.set_geometry_size,
    cmd.SetSceneCamera: scene.set_scene_camera,
    cmd.AddSceneObject: scene.add_scene_object,
    cmd.RemoveSceneObject: scene.remove_scene_object,
    cmd.AddSceneLight: scene.add_scene_light,
    cmd.RemoveSceneLight: scene.remove_scene_light,
    cmd.AddSceneCamera: scene.add_scene_camera,
    cmd.RemoveSceneCamera: scene.remove_scene_camera,
    cmd.DuplicateSceneEntity: scene.duplicate_scene_entity,
    cmd.RemoveSceneEntity: scene.remove_scene_entity,
    cmd.RenameSceneEntity: scene.rename_scene_entity,
    cmd.SetSpeed: playback.set_speed,
    cmd.SetCamera: playback.set_camera,
}


@lru_cache(maxsize=128)
def _handler(command_type: type[Command]) -> Callable[[Session, Command], CommandResult] | None:
    handler = _HANDLERS.get(command_type)
    if handler is not None:
        return handler
    # Preserve isinstance dispatch for callers subclassing a public command.
    return next(
        (handler for base, handler in _HANDLERS.items() if issubclass(command_type, base)), None
    )


def dispatch(session: Session, command: Command) -> CommandResult:
    handler = _handler(type(command))
    if handler is None:
        return CommandResult.bad(f"Unknown command: {type(command).__name__}")
    return handler(session, command)
