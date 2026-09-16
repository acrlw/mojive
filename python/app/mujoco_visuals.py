"""Translate MuJoCo presentation settings for both offscreen and interactive products.

This module owns camera/flag conversion only, without a renderer or UI lifecycle.
"""

from __future__ import annotations

import numpy as np

from mojive.render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from mojive.types import CameraView

try:
    import mujoco
except ImportError:  # pragma: no cover - callers validate the optional dependency
    mujoco = None


VIS_FLAGS = {
    "mjVIS_CONVEXHULL": RenderFlag.CONVEXHULL,
    "mjVIS_TEXTURE": RenderFlag.TEXTURE,
    "mjVIS_JOINT": RenderFlag.JOINT,
    "mjVIS_CAMERA": RenderFlag.CAMERA,
    "mjVIS_ACTUATOR": RenderFlag.ACTUATOR,
    "mjVIS_ACTIVATION": RenderFlag.ACTIVATION,
    "mjVIS_LIGHT": RenderFlag.LIGHT,
    "mjVIS_TENDON": RenderFlag.TENDON,
    "mjVIS_RANGEFINDER": RenderFlag.RANGEFINDER,
    "mjVIS_CONSTRAINT": RenderFlag.CONSTRAINT,
    "mjVIS_INERTIA": RenderFlag.INERTIA,
    "mjVIS_SCLINERTIA": RenderFlag.SCLINERTIA,
    "mjVIS_CONTACTPOINT": RenderFlag.CONTACTPOINT,
    "mjVIS_ISLAND": RenderFlag.ISLAND,
    "mjVIS_CONTACTFORCE": RenderFlag.CONTACTFORCE,
    "mjVIS_CONTACTSPLIT": RenderFlag.CONTACTSPLIT,
    "mjVIS_AUTOCONNECT": RenderFlag.AUTOCONNECT,
    "mjVIS_COM": RenderFlag.COM,
    "mjVIS_STATIC": RenderFlag.STATIC,
    "mjVIS_SKIN": RenderFlag.SKIN,
    "mjVIS_FLEXVERT": RenderFlag.FLEXVERT,
    "mjVIS_FLEXEDGE": RenderFlag.FLEXEDGE,
    "mjVIS_FLEXFACE": RenderFlag.FLEXFACE,
    "mjVIS_FLEXSKIN": RenderFlag.FLEXSKIN,
    "mjVIS_BODYBVH": RenderFlag.BODYBVH,
    "mjVIS_MESHBVH": RenderFlag.MESHBVH,
}


RND_FLAGS = {
    "mjRND_SHADOW": RenderFlag.SHADOW,
    "mjRND_WIREFRAME": RenderFlag.WIREFRAME,
    "mjRND_REFLECTION": RenderFlag.REFLECTION,
    "mjRND_ADDITIVE": RenderFlag.ADDITIVE,
    "mjRND_SKYBOX": RenderFlag.SKYBOX,
    "mjRND_FOG": RenderFlag.FOG,
    "mjRND_HAZE": RenderFlag.HAZE,
    "mjRND_CULL_FACE": RenderFlag.CULL_FACE,
}


LABEL_MODES = {
    "mjLABEL_NONE": LabelMode.NONE,
    "mjLABEL_BODY": LabelMode.BODY,
    "mjLABEL_JOINT": LabelMode.JOINT,
    "mjLABEL_GEOM": LabelMode.GEOM,
    "mjLABEL_SITE": LabelMode.SITE,
    "mjLABEL_CAMERA": LabelMode.CAMERA,
    "mjLABEL_LIGHT": LabelMode.LIGHT,
    "mjLABEL_TENDON": LabelMode.TENDON,
    "mjLABEL_ACTUATOR": LabelMode.ACTUATOR,
    "mjLABEL_CONSTRAINT": LabelMode.CONSTRAINT,
    "mjLABEL_FLEX": LabelMode.FLEX,
    "mjLABEL_SELECTION": LabelMode.SELECTION,
    "mjLABEL_CONTACTPOINT": LabelMode.CONTACT_POINT,
    "mjLABEL_CONTACTFORCE": LabelMode.CONTACT_FORCE,
}


FRAME_MODES = {
    "mjFRAME_NONE": FrameMode.NONE,
    "mjFRAME_BODY": FrameMode.BODY,
    "mjFRAME_GEOM": FrameMode.GEOM,
    "mjFRAME_SITE": FrameMode.SITE,
    "mjFRAME_CAMERA": FrameMode.CAMERA,
    "mjFRAME_LIGHT": FrameMode.LIGHT,
    "mjFRAME_CONTACT": FrameMode.CONTACT,
    "mjFRAME_WORLD": FrameMode.WORLD,
}


def camera_view(scene, model, aspect: float) -> CameraView:
    left, right = scene.camera[0], scene.camera[1]
    eye = (np.asarray(left.pos, np.float32) + np.asarray(right.pos, np.float32)) * 0.5
    forward = np.asarray(left.forward, np.float32)
    up = np.asarray(left.up, np.float32)
    near = max(float(left.frustum_near), 1e-6)
    far = max(float(left.frustum_far), near)
    height = max(float(left.frustum_top - left.frustum_bottom), 1e-6)
    orthographic = bool(left.orthographic)
    fov_y = np.deg2rad(45.0) if orthographic else 2.0 * np.arctan2(height * 0.5, near)
    distance = max(float(model.stat.extent), 1e-3)
    return CameraView(
        eye=eye,
        target=(eye + forward * distance).astype(np.float32),
        up=up,
        fov_y=float(fov_y),
        near=near,
        far=far,
        aspect=float(aspect),
        orthographic=orthographic,
        ortho_height=height if orthographic else 2.0 * distance * np.tan(fov_y * 0.5),
    )


def flag_enabled(values, enum_type, name: str) -> bool:
    member = getattr(enum_type, name, None)
    return bool(member is not None and member.value < len(values) and values[member.value])


def mode_value(enum_type, name: str) -> int | None:
    member = getattr(enum_type, name, None)
    return None if member is None else int(member.value)


def apply_render_options(backend, option, scene) -> None:
    for name, flag in VIS_FLAGS.items():
        backend.set_flag(flag, flag_enabled(option.flags, mujoco.mjtVisFlag, name))
    for name, flag in RND_FLAGS.items():
        backend.set_flag(flag, flag_enabled(scene.flags, mujoco.mjtRndFlag, name))

    debug_view = DebugView.SHADED
    if flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_DEPTH"):
        debug_view = DebugView.DEPTH
    elif flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_IDCOLOR"):
        debug_view = DebugView.IDCOLOR
    elif flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_SEGMENT"):
        debug_view = DebugView.SEGMENT
    backend.set_debug_view(debug_view)

    label = next(
        (
            mode
            for name, mode in LABEL_MODES.items()
            if mode_value(mujoco.mjtLabel, name) == int(option.label)
        ),
        LabelMode.NONE,
    )
    frame = next(
        (
            mode
            for name, mode in FRAME_MODES.items()
            if mode_value(mujoco.mjtFrame, name) == int(option.frame)
        ),
        FrameMode.NONE,
    )
    backend.set_label_mode(label)
    backend.set_frame_mode(frame)
    backend.set_bvh_depth(int(option.bvh_depth))
