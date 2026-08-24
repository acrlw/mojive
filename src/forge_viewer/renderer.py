"""MuJoCo-compatible offscreen rendering through the selected backend."""

from __future__ import annotations

import contextlib
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from .adapters.base import FrameNeeds
from .render.backend import DebugView, FrameMode, LabelMode, RenderFlag
from .types import CameraView, InstancePoseSource, InstanceVisual

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - optional dependency
    mujoco = None
    _IMPORT_ERROR: ImportError | None = exc
    DEFAULT_FONT_SCALE = 150
else:
    _IMPORT_ERROR = None
    DEFAULT_FONT_SCALE = mujoco.mjtFontScale.mjFONTSCALE_150


_FRAME_NEEDS = FrameNeeds(
    poses=True,
    contacts=True,
    tendons=True,
    actuator=True,
    deformables=True,
    diagnostics=True,
    islands=True,
    bvh=True,
)

_VIS_FLAGS = {
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

_RND_FLAGS = {
    "mjRND_SHADOW": RenderFlag.SHADOW,
    "mjRND_WIREFRAME": RenderFlag.WIREFRAME,
    "mjRND_REFLECTION": RenderFlag.REFLECTION,
    "mjRND_ADDITIVE": RenderFlag.ADDITIVE,
    "mjRND_SKYBOX": RenderFlag.SKYBOX,
    "mjRND_FOG": RenderFlag.FOG,
    "mjRND_HAZE": RenderFlag.HAZE,
    "mjRND_CULL_FACE": RenderFlag.CULL_FACE,
}

_LABEL_MODES = {
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

_FRAME_MODES = {
    "mjFRAME_NONE": FrameMode.NONE,
    "mjFRAME_BODY": FrameMode.BODY,
    "mjFRAME_GEOM": FrameMode.GEOM,
    "mjFRAME_SITE": FrameMode.SITE,
    "mjFRAME_CAMERA": FrameMode.CAMERA,
    "mjFRAME_LIGHT": FrameMode.LIGHT,
    "mjFRAME_CONTACT": FrameMode.CONTACT,
    "mjFRAME_WORLD": FrameMode.WORLD,
}


class _GLFWContext:
    def __init__(self, width: int, height: int) -> None:
        import glfw

        self._glfw = glfw
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        for hint, value in (
            (glfw.CONTEXT_VERSION_MAJOR, 3),
            (glfw.CONTEXT_VERSION_MINOR, 3),
            (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
            (glfw.OPENGL_FORWARD_COMPAT, True),
            (glfw.VISIBLE, False),
        ):
            glfw.window_hint(hint, value)
        self.window = glfw.create_window(
            max(1, width), max(1, height), "forge renderer", None, None
        )
        if not self.window:
            raise RuntimeError("Failed to create an OpenGL 3.3 core context")
        self.gl_context = None

    @contextmanager
    def current(self):
        glfw = self._glfw
        previous = glfw.get_current_context()
        glfw.make_context_current(self.window)
        try:
            yield
        finally:
            glfw.make_context_current(previous)

    def close(self) -> None:
        if self.window is None:
            return
        current = self._glfw.get_current_context()
        if current == self.window:
            self._glfw.make_context_current(None)
        self._glfw.destroy_window(self.window)
        self.window = None


class _StandaloneContext:
    def __init__(self, backend: str) -> None:
        import moderngl

        self.gl_context = moderngl.create_standalone_context(require=330, backend=backend)

    @contextmanager
    def current(self):
        yield

    def close(self) -> None:
        if self.gl_context is None:
            return
        self.gl_context.release()
        self.gl_context = None


def _create_context(width: int, height: int):
    requested = os.environ.get("FORGE_VIEWER_GL", "").strip().lower()
    if requested == "egl" or (requested in {"", "auto"} and sys.platform.startswith("linux")):
        return _StandaloneContext("egl")
    if requested not in {"", "auto", "glfw", "native"}:
        raise ValueError(f"Unsupported FORGE_VIEWER_GL backend: {requested}")
    return _GLFWContext(width, height)


def _select_backend(width: int, height: int, samples: int):
    """Create the render backend selected by FORGE_VIEWER_BACKEND.

    Returns ``(context, backend)``; ``context`` is ``None`` for backends that
    manage no GL state of their own (the webgpu backend needs no window, EGL,
    or GLFW at all).
    """
    requested = os.environ.get("FORGE_VIEWER_BACKEND", "").strip().lower()
    if requested in {"wgpu", "webgpu"}:
        from .render.webgpu.backend import WgpuBackend

        return None, WgpuBackend(max(1, width), max(1, height), samples, gpu_timing=False)
    if requested not in {"", "forge"}:
        raise ValueError(f"Unsupported FORGE_VIEWER_BACKEND: {requested}")
    from .render.forge.backend import ForgeBackend

    context = _create_context(width, height)
    try:
        with context.current():
            backend = ForgeBackend(context.gl_context, max(1, width), max(1, height), samples)
    except Exception:
        context.close()
        raise
    return context, backend


class Renderer:
    """Render an existing MuJoCo model through the selected backend."""

    def __init__(
        self,
        model,
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
        font_scale: Any = DEFAULT_FONT_SCALE,
    ) -> None:
        if mujoco is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"MuJoCo is not installed: {_IMPORT_ERROR}. Install the [mujoco] optional dependency."
            )
        self._context = None
        self._backend = None
        self._adapter = None
        self._closed = False
        self._depth_rendering = False
        self._segmentation_rendering = False

        self._check_framebuffer(model, int(width), int(height))
        self._model = model
        self._height = int(height)
        self._width = int(width)
        self._font_scale = font_scale
        self._scene = mujoco.MjvScene(model=model, maxgeom=int(max_geom))
        self._scene_option = mujoco.MjvOption()

        from .adapters.mujoco_adapter import MuJoCoAdapter

        adapter = MuJoCoAdapter()
        adapter.load_model(model)
        adapter_source = adapter.scene_source()
        self._transparent_visual = False
        source = _limit_scene_source(adapter_source, int(max_geom), model, False)
        self._segmentation_table = _configure_segmentation(source, model)
        self._source = source
        self._adapter_source = adapter_source
        samples = max(0, int(model.vis.quality.offsamples))
        context, backend = _select_backend(self._width, self._height, samples)
        self._context = context
        self._backend = backend
        try:
            with self._gl_current():
                backend.set_background((0.0, 0.0, 0.0, 1.0))
                backend.set_scene(source)
                self._view = (adapter.camera_hint() or CameraView()).with_aspect(self._aspect)
                backend.set_camera(self._view)
        except Exception:
            self._context = None
            self._backend = None
            if context is not None:
                context.close()
            adapter.release()
            raise
        self._adapter = adapter

    @contextmanager
    def _gl_current(self):
        if self._context is None:
            yield
        else:
            with self._context.current():
                yield

    @property
    def model(self):
        return self._model

    @property
    def scene(self):
        return self._scene

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def _aspect(self) -> float:
        return self._width / max(self._height, 1)

    def update_scene(self, data, camera=-1, scene_option=None) -> None:
        self._require_open("update_scene")
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not self._model:
            raise ValueError("data was created for a different MuJoCo model")
        camera = self._resolve_camera(camera)
        option = scene_option or self._scene_option
        self._adapter.apply_scene_option(option)
        mujoco.mjv_updateScene(
            self._model,
            data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        self._adapter.refresh_model_visuals()
        adapter_source = self._adapter.scene_source()
        transparent_visual = _flag_enabled(option.flags, mujoco.mjtVisFlag, "mjVIS_TRANSPARENT")
        if (
            adapter_source is not self._adapter_source
            or transparent_visual != self._transparent_visual
        ):
            source = _limit_scene_source(
                adapter_source,
                self._scene.maxgeom,
                self._model,
                transparent_visual,
            )
            self._segmentation_table = _configure_segmentation(source, self._model)
            self._source = source
            self._adapter_source = adapter_source
            self._transparent_visual = transparent_visual
            with self._gl_current():
                self._backend.set_scene(source)
        _apply_render_options(self._backend, option, self._scene)
        self._adapter.use_data(data)
        frame = self._adapter.frame(_FRAME_NEEDS)
        view = _camera_view(self._scene, self._model, self._aspect)
        with self._gl_current():
            self._view = view
            self._backend.set_camera(view)
            self._backend.update(frame)

    def render(self, *, out: np.ndarray | None = None) -> np.ndarray:
        self._require_open("render")
        if self._depth_rendering:
            shape = (self._height, self._width)
        elif self._segmentation_rendering:
            shape = (self._height, self._width, 2)
        else:
            shape = (self._height, self._width, 3)
        if out is not None and out.shape != shape:
            raise ValueError(
                f"Expected `out.shape == {shape}`. Got `out.shape={out.shape}` instead."
            )
        with self._gl_current():
            self._backend.render()
            if self._depth_rendering:
                image = _metric_depth(self._backend.target.read_depth(flip=True), self._view)
            elif self._segmentation_rendering:
                image = _segmentation_image(
                    self._backend.target.read_ids(flip=True), self._segmentation_table
                )
            else:
                image = np.ascontiguousarray(self._backend.target.read_color(flip=True)[..., :3])
        if out is None:
            return image
        np.copyto(out, image, casting="unsafe")
        return out

    def enable_depth_rendering(self) -> None:
        self._segmentation_rendering = False
        self._depth_rendering = True
        self._backend.set_transparent_id_rendering(False)

    def disable_depth_rendering(self) -> None:
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = True
        self._depth_rendering = False
        self._backend.set_transparent_id_rendering(True)

    def disable_segmentation_rendering(self) -> None:
        self._segmentation_rendering = False
        self._backend.set_transparent_id_rendering(False)

    def set_render_flag(self, name: str, enabled: bool) -> None:
        """Set one MuJoCo render flag for the next image."""
        self._require_open("set_render_flag")
        member = getattr(mujoco.mjtRndFlag, str(name), None)
        if member is None:
            raise ValueError(f"Unknown MuJoCo render flag: {name}")
        self._scene.flags[int(member)] = bool(enabled)
        backend_flag = _RND_FLAGS.get(member.name)
        if backend_flag is not None:
            self._backend.set_flag(backend_flag, bool(enabled))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._context is not None and self._backend is not None:
            with self._gl_current():
                self._backend.release()
        if self._adapter is not None:
            self._adapter.release()
        if self._context is not None:
            self._context.close()
        self._backend = None
        self._adapter = None
        self._context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _resolve_camera(self, camera):
        if isinstance(camera, mujoco.MjvCamera):
            return camera
        camera_id = camera
        if isinstance(camera_id, str):
            camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            if camera_id == -1:
                raise ValueError(f'The camera "{camera}" does not exist.')
        if camera_id < -1 or camera_id >= self._model.ncam:
            raise ValueError(f"The camera id {camera_id} is out of range [-1, {self._model.ncam}).")
        resolved = mujoco.MjvCamera()
        resolved.fixedcamid = camera_id
        if camera_id == -1:
            resolved.type = mujoco.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self._model, resolved)
        else:
            resolved.type = mujoco.mjtCamera.mjCAMERA_FIXED
        return resolved

    def _require_open(self, operation: str) -> None:
        if self._closed:
            raise RuntimeError(f"{operation} cannot be called after close.")

    @staticmethod
    def _check_framebuffer(model, width: int, height: int) -> None:
        if width > model.vis.global_.offwidth:
            raise ValueError(
                f"Image width {width} > framebuffer width {model.vis.global_.offwidth}."
            )
        if height > model.vis.global_.offheight:
            raise ValueError(
                f"Image height {height} > framebuffer height {model.vis.global_.offheight}."
            )


def _camera_view(scene, model, aspect: float) -> CameraView:
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


def _metric_depth(depth: np.ndarray, camera: CameraView) -> np.ndarray:
    values = np.asarray(depth, np.float64)
    near, far = float(camera.near), float(camera.far)
    if camera.orthographic:
        metric = near + values * (far - near)
    else:
        metric = near * far / np.maximum(far - values * (far - near), 1e-12)
    return np.ascontiguousarray(metric.astype(np.float32))


def _limit_scene_source(source, max_geom: int, model, transparent_visual: bool):
    limit = max(int(max_geom), 0)
    keep = np.zeros(source.instance_count, bool)
    logical: set[tuple[int, int]] = set()
    for index in range(source.instance_count):
        pose = int(source.geom_pose_source[index])
        source_id = int(source.geom_source[index])
        if pose == int(InstancePoseSource.GEOM):
            key = (int(mujoco.mjtObj.mjOBJ_GEOM), source_id)
        elif pose == int(InstancePoseSource.SITE):
            key = (int(mujoco.mjtObj.mjOBJ_SITE), source_id)
        else:
            visual = int(source.geom_visual[index])
            object_id = int(source.geom_object_id[index])
            if visual in {
                int(InstanceVisual.FLEX_EDGE),
                int(InstanceVisual.FLEX_FACE),
                int(InstanceVisual.FLEX_SKIN),
            }:
                key = (int(mujoco.mjtObj.mjOBJ_FLEX), object_id - model.nbody)
            elif visual == int(InstanceVisual.SKIN):
                key = (
                    int(mujoco.mjtObj.mjOBJ_SKIN),
                    object_id - model.nbody - model.nflex,
                )
            else:
                key = (-1, index)
        if key not in logical:
            if len(logical) >= limit:
                continue
            logical.add(key)
        keep[index] = True

    indices = np.flatnonzero(keep)
    rgba = source.geom_rgba[indices].copy()
    if transparent_visual:
        rgba[~source.geom_static[indices], 3] *= 0.3
    return replace(
        source,
        geom_mesh=[source.geom_mesh[i] for i in indices],
        geom_convex_mesh=[source.geom_convex_mesh[i] for i in indices],
        geom_material=[source.geom_material[i] for i in indices],
        geom_size=source.geom_size[indices].copy(),
        geom_rgba=rgba,
        geom_object_id=source.geom_object_id[indices].copy(),
        geom_body=source.geom_body[indices].copy(),
        geom_source=source.geom_source[indices].copy(),
        geom_pose_source=source.geom_pose_source[indices].copy(),
        geom_visual=source.geom_visual[indices].copy(),
        geom_static=source.geom_static[indices].copy(),
        instance_island_body=source.instance_island_body[indices].copy(),
        geom_node=source.geom_node[indices].copy(),
        geom_local=source.geom_local[indices].copy(),
        geom_infinite_plane=source.geom_infinite_plane[indices].copy(),
    )


def _configure_segmentation(source, model) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    encoded = np.zeros(source.instance_count, np.uint32)
    pair_to_id: dict[tuple[int, int], int] = {}
    for index in range(source.instance_count):
        pose = int(source.geom_pose_source[index])
        source_id = int(source.geom_source[index])
        if pose == int(InstancePoseSource.GEOM):
            pair = (source_id, int(mujoco.mjtObj.mjOBJ_GEOM))
        elif pose == int(InstancePoseSource.SITE):
            pair = (source_id, int(mujoco.mjtObj.mjOBJ_SITE))
        else:
            visual = int(source.geom_visual[index])
            object_id = int(source.geom_object_id[index])
            if visual in {
                int(InstanceVisual.FLEX_EDGE),
                int(InstanceVisual.FLEX_FACE),
                int(InstanceVisual.FLEX_SKIN),
            }:
                pair = (object_id - model.nbody, int(mujoco.mjtObj.mjOBJ_FLEX))
            elif visual == int(InstanceVisual.SKIN):
                pair = (
                    object_id - model.nbody - model.nflex,
                    int(mujoco.mjtObj.mjOBJ_SKIN),
                )
            else:
                continue
        segment_id = pair_to_id.get(pair)
        if segment_id is None:
            segment_id = len(pairs) + 1
            pair_to_id[pair] = segment_id
            pairs.append(pair)
        encoded[index] = segment_id
    source.geom_object_id = encoded
    table = np.full((len(pairs) + 1, 2), -1, np.int32)
    if pairs:
        table[1:] = np.asarray(pairs, np.int32)
    return table


def _segmentation_image(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, np.uint32)
    image = np.full((*ids.shape, 2), -1, np.int32)
    valid = ids < len(table)
    image[valid] = table[ids[valid]]
    return np.ascontiguousarray(image)


def _flag_enabled(values, enum_type, name: str) -> bool:
    member = getattr(enum_type, name, None)
    return bool(member is not None and member.value < len(values) and values[member.value])


def _mode_value(enum_type, name: str) -> int | None:
    member = getattr(enum_type, name, None)
    return None if member is None else int(member.value)


def _apply_render_options(backend, option, scene) -> None:
    for name, flag in _VIS_FLAGS.items():
        backend.set_flag(flag, _flag_enabled(option.flags, mujoco.mjtVisFlag, name))
    for name, flag in _RND_FLAGS.items():
        backend.set_flag(flag, _flag_enabled(scene.flags, mujoco.mjtRndFlag, name))

    debug_view = DebugView.SHADED
    if _flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_DEPTH"):
        debug_view = DebugView.DEPTH
    elif _flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_IDCOLOR"):
        debug_view = DebugView.IDCOLOR
    elif _flag_enabled(scene.flags, mujoco.mjtRndFlag, "mjRND_SEGMENT"):
        debug_view = DebugView.SEGMENT
    backend.set_debug_view(debug_view)

    label = next(
        (
            mode
            for name, mode in _LABEL_MODES.items()
            if _mode_value(mujoco.mjtLabel, name) == int(option.label)
        ),
        LabelMode.NONE,
    )
    frame = next(
        (
            mode
            for name, mode in _FRAME_MODES.items()
            if _mode_value(mujoco.mjtFrame, name) == int(option.frame)
        ),
        FrameMode.NONE,
    )
    backend.set_label_mode(label)
    backend.set_frame_mode(frame)
    backend.set_bvh_depth(int(option.bvh_depth))
