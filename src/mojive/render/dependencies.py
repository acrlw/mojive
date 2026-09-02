"""Stable dependency keys for reusable render-pass products."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..types import CameraView, Light, LightSet
from .backend import RenderFlag
from .scene import RenderScene


def _array_key(value) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def camera_key(camera: CameraView) -> tuple:
    """Value key for every camera property consumed by a render pass."""

    return (
        _array_key(camera.eye),
        _array_key(camera.target),
        _array_key(camera.up),
        float(camera.fov_y),
        float(camera.near),
        float(camera.far),
        float(camera.aspect),
        bool(camera.orthographic),
        float(camera.ortho_height),
        None if camera.orthographic_blend is None else float(camera.orthographic_blend),
        _array_key(camera.focal_length),
        _array_key(camera.sensor_size),
        _array_key(camera.principal_offset),
    )


def _light_key(light: Light | None) -> tuple | None:
    if light is None:
        return None
    return (
        int(light.type),
        _array_key(light.position),
        _array_key(light.direction),
        _array_key(light.diffuse),
        _array_key(light.specular),
        _array_key(light.ambient),
        _array_key(light.attenuation),
        float(light.range),
        float(light.area_radius),
        float(light.cutoff),
        float(light.exponent),
        light.texture,
        float(light.intensity),
        bool(light.cast_shadow),
        bool(light.active),
    )


def lights_key(lights: LightSet) -> tuple:
    """Value key resilient to replacement or in-place edits of light arrays."""

    return (
        tuple(_light_key(light) for light in lights.lights),
        _light_key(lights.headlight),
        _array_key(lights.ambient),
        _array_key(lights.fog_color),
        float(lights.fog_start),
        float(lights.fog_end),
        _array_key(lights.haze_color),
        float(lights.haze_density),
        bool(lights.horizon_haze),
        int(lights.horizon_haze_slices),
    )


def lifecycle_key(
    scene: RenderScene,
    frame_serial: int,
    *,
    visual: bool = False,
    identity: bool = False,
) -> tuple:
    """Scene revision key, conservatively unique for legacy revision-zero sources."""

    revisions = [scene.structure_revision, scene.pose_revision]
    if visual:
        revisions.append(scene.visual_revision)
    if identity:
        revisions.append(scene.identity_revision)
    if not all(revisions):
        return ("frame", int(frame_serial))
    return tuple(int(revision) for revision in revisions)


def flags_key(flags: Mapping[RenderFlag, bool]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted((flag.value, bool(value)) for flag, value in flags.items()))
