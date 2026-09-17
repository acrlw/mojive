"""MuJoCo-compatible offscreen rendering through the selected backend."""

from __future__ import annotations

import contextlib
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mojive.adapters.base import FrameNeeds
from mojive.app.mujoco_visuals import (
    RND_FLAGS,
    apply_render_options,
    camera_view,
    flag_enabled,
    mode_value,
)
from mojive.render.backend import (
    RenderRequest,
    ShadowQuality,
)
from mojive.render.context import _select_backend
from mojive.render.geometry import GeometryView, geometry_view, set_geometry_view
from mojive.types import CameraView, InstancePoseSource, InstanceVisual

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - optional dependency
    mujoco = None
    _IMPORT_ERROR: ImportError | None = exc
    DEFAULT_FONT_SCALE = 150
else:
    _IMPORT_ERROR = None
    DEFAULT_FONT_SCALE = mujoco.mjtFontScale.mjFONTSCALE_150


@dataclass(frozen=True)
class _RenderOutput:
    shape: tuple[int, ...]
    dtype: np.dtype
    request: RenderRequest
    read_name: str
    async_read_name: str


class Renderer:
    """Render an existing MuJoCo model through the selected backend.

    The public control flow mirrors ``mujoco.Renderer``: update from ``MjData``,
    select RGB, depth, or segmentation output, then render into a new or reused
    NumPy array. Use the object as a context manager to release GPU resources.
    """

    def __init__(
        self,
        model,
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
        font_scale: Any = DEFAULT_FONT_SCALE,
        *,
        shadow_quality: ShadowQuality | str = ShadowQuality.BALANCED,
        renderer: str | None = None,
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
        self._canvas_2d = None
        shadow_quality = ShadowQuality(shadow_quality)

        self._check_framebuffer(model, int(width), int(height))
        self._model = model
        self._height = int(height)
        self._width = int(width)
        self._font_scale = font_scale
        self._scene = mujoco.MjvScene(model=model, maxgeom=int(max_geom))
        self._scene_option = mujoco.MjvOption()
        self._option_flags = np.zeros(0, np.uint8)
        self._option_label: int | None = None
        self._option_frame: int | None = None
        self._cached_frame_needs: FrameNeeds | None = None
        self._render_flags = np.zeros(0, np.uint8)
        self._render_bvh_depth: int | None = None

        from mojive.adapters.mujoco import MuJoCoAdapter

        adapter = MuJoCoAdapter()
        adapter.load_model(model)
        adapter_source = adapter.scene_source()
        self._transparent_visual = False
        source = _limit_scene_source(adapter_source, int(max_geom), model, False)
        _configure_segmentation(source)
        self._source = source
        self._adapter_source = adapter_source
        samples = max(0, int(model.vis.quality.offsamples))
        try:
            context, backend = _select_backend(self._width, self._height, samples, renderer)
        except Exception:
            adapter.release()
            raise
        self._context = context
        self._backend = backend
        try:
            with self._gl_current():
                backend.set_background((0.0, 0.0, 0.0, 1.0))
                backend.set_scene(source)
                if not backend.set_shadow_quality(shadow_quality) and (
                    backend.caps.shadows or shadow_quality != ShadowQuality.BALANCED
                ):
                    raise RuntimeError(
                        f"The {backend.caps.name} backend does not support shadow quality presets"
                    )
                self._view = (adapter.camera_hint() or CameraView()).with_aspect(self._aspect)
                backend.set_camera(self._view)
        except Exception:
            with contextlib.suppress(Exception), self._gl_current():
                backend.release()
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
        """Return the MuJoCo model bound at construction."""
        return self._model

    @property
    def scene(self):
        """Return the compatibility ``MjvScene`` updated for each frame."""
        return self._scene

    @property
    def height(self) -> int:
        """Return output image height in pixels."""
        return self._height

    @property
    def width(self) -> int:
        """Return output image width in pixels."""
        return self._width

    @property
    def debug(self):
        """Return retained 3D diagnostics rendered with RGB output, without a Viewer."""
        self._require_open("debug")
        draw = getattr(self._backend, "debug", None)
        if draw is None:
            raise RuntimeError("the active backend does not provide debug drawing")
        return draw

    @property
    def canvas2d(self):
        """Return a retained 2D diagnostic canvas rendered with RGB output."""

        self._require_open("canvas2d")
        if self._canvas_2d is None:
            from mojive.render.canvas import Canvas2D

            self._canvas_2d = Canvas2D(self.debug)
        return self._canvas_2d

    @property
    def _aspect(self) -> float:
        return self._width / max(self._height, 1)

    def update_scene(
        self,
        data: Any,
        camera: Any = -1,
        scene_option: Any | None = None,
    ) -> None:
        """Update dynamic scene state from MuJoCo data.

        Args:
            data: ``mujoco.MjData`` created for :attr:`model`.
            camera: Free camera ``-1``, fixed camera ID or name, or ``MjvCamera``.
            scene_option: Optional ``MjvOption`` controlling visual categories.
        """
        self._require_open("update_scene")
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not self._model:
            raise ValueError("data was created for a different MuJoCo model")
        camera = self._resolve_camera(camera)
        option = scene_option or self._scene_option
        option_changed = self._sync_option_state(option)
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
        self._adapter.use_data(data)
        if option_changed or self._cached_frame_needs is None:
            self._cached_frame_needs = _frame_needs(option, self._model)
        frame = self._adapter.frame(self._cached_frame_needs)
        adapter_source = self._adapter.scene_source()
        transparent_visual = flag_enabled(option.flags, mujoco.mjtVisFlag, "mjVIS_TRANSPARENT")
        if (
            adapter_source is not self._adapter_source
            or transparent_visual != self._transparent_visual
        ):
            source = _limit_scene_source(
                adapter_source,
                self._scene.maxgeom,
                self._model,
                transparent_visual,
                geometry_view(self._backend),
            )
            _configure_segmentation(source)
            self._source = source
            self._adapter_source = adapter_source
            self._transparent_visual = transparent_visual
            with self._gl_current():
                self._backend.set_scene(source)
        if self._sync_render_option_state(option, option_changed):
            apply_render_options(self._backend, option, self._scene)
        view = camera_view(self._scene, self._model, self._aspect)
        with self._gl_current():
            self._view = view
            self._backend.set_camera(view)
            self._backend.update(frame)

    def _sync_option_state(self, option) -> bool:
        """Track in-place MjvOption edits without recomputing stable translations."""

        flags = np.asarray(option.flags)
        label = int(option.label)
        frame = int(option.frame)
        changed = (
            self._option_flags.shape != flags.shape
            or not np.array_equal(self._option_flags, flags)
            or self._option_label != label
            or self._option_frame != frame
        )
        if changed:
            self._option_flags = flags.astype(np.uint8, copy=True)
            self._option_label = label
            self._option_frame = frame
        return changed

    def _sync_render_option_state(self, option, option_changed: bool) -> bool:
        """Track flags callers may mutate through the compatibility scene."""

        flags = np.asarray(self._scene.flags)
        bvh_depth = int(option.bvh_depth)
        changed = (
            option_changed
            or self._render_flags.shape != flags.shape
            or not np.array_equal(self._render_flags, flags)
            or self._render_bvh_depth != bvh_depth
        )
        if changed:
            self._render_flags = flags.astype(np.uint8, copy=True)
            self._render_bvh_depth = bvh_depth
        return changed

    def render(self, *, out: np.ndarray | None = None) -> np.ndarray:
        """Render the selected output mode.

        Args:
            out: Optional correctly shaped destination array.

        Returns:
            RGB ``uint8``, metric depth ``float32``, or MuJoCo segmentation IDs.
        """
        self._require_open("render")
        spec = self._render_output()
        if out is not None and out.shape != spec.shape:
            raise ValueError(
                f"Expected `out.shape == {spec.shape}`. Got `out.shape={out.shape}` instead."
            )
        direct_out = out if out is not None and out.flags.c_contiguous else None
        with self._gl_current():
            self._backend.render(request=spec.request)
            native_out = (
                direct_out if direct_out is not None and direct_out.dtype == spec.dtype else None
            )
            image = getattr(self._backend.target, spec.read_name)(flip=True, out=native_out)
        if out is None:
            return image
        if image is out:
            return out
        np.copyto(out, image, casting="unsafe")
        return out

    def render_async(self, *, out: np.ndarray | None = None) -> Future[np.ndarray]:
        """Submit a render and return a future for its CPU image.

        Native bgfx uses bounded asynchronous readback. OpenGL preserves the
        same API with an already-completed future.
        When ``out`` is supplied, the caller owns it but must not read or mutate
        it until the returned future completes.
        """

        self._require_open("render_async")
        spec = self._render_output()
        if out is not None and out.shape != spec.shape:
            raise ValueError(
                f"Expected `out.shape == {spec.shape}`. Got `out.shape={out.shape}` instead."
            )
        reader = getattr(self._backend.target, spec.async_read_name, None)
        if not callable(reader):
            future: Future[np.ndarray] = Future()
            try:
                future.set_result(self.render(out=out))
            except Exception as exc:
                future.set_exception(exc)
            return future
        with self._gl_current():
            self._backend.render(request=spec.request)
            return reader(flip=True, out=out)

    def _render_output(self) -> _RenderOutput:
        if self._depth_rendering:
            return _RenderOutput(
                (self._height, self._width),
                np.dtype(np.float32),
                RenderRequest.metric_depth(),
                "read_metric_depth",
                "read_metric_depth_async",
            )
        if self._segmentation_rendering:
            return _RenderOutput(
                (self._height, self._width, 2),
                np.dtype(np.int32),
                RenderRequest.segmentation(),
                "read_segmentation",
                "read_segmentation_async",
            )
        return _RenderOutput(
            (self._height, self._width, 3),
            np.dtype(np.uint8),
            RenderRequest.color(),
            "read_rgb",
            "read_rgb_async",
        )

    def enable_depth_rendering(self) -> None:
        """Select metric depth output for subsequent :meth:`render` calls."""
        self._require_open("enable_depth_rendering")
        self._segmentation_rendering = False
        self._depth_rendering = True
        self._backend.set_transparent_id_rendering(False)

    def disable_depth_rendering(self) -> None:
        """Return from depth output to RGB output."""
        self._require_open("disable_depth_rendering")
        self._depth_rendering = False

    def enable_segmentation_rendering(self) -> None:
        """Select MuJoCo object and object-type IDs for subsequent renders."""
        self._require_open("enable_segmentation_rendering")
        self._segmentation_rendering = True
        self._depth_rendering = False
        self._backend.set_transparent_id_rendering(True)

    def disable_segmentation_rendering(self) -> None:
        """Return from segmentation output to RGB output."""
        self._require_open("disable_segmentation_rendering")
        self._segmentation_rendering = False
        self._backend.set_transparent_id_rendering(False)

    def set_render_flag(self, name: str, enabled: bool) -> None:
        """Set one MuJoCo render flag for the next image."""
        self._require_open("set_render_flag")
        member = getattr(mujoco.mjtRndFlag, str(name), None)
        if member is None:
            raise ValueError(f"Unknown MuJoCo render flag: {name}")
        self._scene.flags[int(member)] = bool(enabled)
        backend_flag = RND_FLAGS.get(member.name)
        if backend_flag is not None:
            self._backend.set_flag(backend_flag, bool(enabled))

    def set_geometry_view(self, view: GeometryView | str) -> None:
        """Choose default, visual, collision, or both; publish with update_scene()."""
        self._require_open("set_geometry_view")
        view = GeometryView(view)
        if view == geometry_view(self._backend):
            return
        with self._gl_current():
            if not set_geometry_view(self._backend, view):
                raise NotImplementedError("The backend does not support geometry views")
        self._adapter_source = None

    @property
    def shadow_quality(self) -> ShadowQuality:
        """Return the active shadow quality preset."""
        self._require_open("shadow_quality")
        return self._backend.get_shadow_quality()

    def set_shadow_quality(self, quality: ShadowQuality | str) -> None:
        """Set shadow filtering and near-cascade density for subsequent renders."""
        self._require_open("set_shadow_quality")
        quality = ShadowQuality(quality)
        if not self._backend.set_shadow_quality(quality):
            raise RuntimeError(
                f"The {self._backend.caps.name} backend does not support shadow quality presets"
            )

    def close(self) -> None:
        """Release renderer, adapter, and graphics-context resources."""
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
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


def _limit_scene_source(
    source,
    max_geom: int,
    model,
    transparent_visual: bool,
    view: GeometryView = GeometryView.DEFAULT,
):
    limit = max(int(max_geom), 0)
    keep = np.zeros(source.instance_count, bool)
    logical: set[tuple[int, int]] = set()
    role_mask = {
        GeometryView.DEFAULT: 0,
        GeometryView.VISUAL: 1,
        GeometryView.COLLISION: 2,
        GeometryView.BOTH: 3,
    }[view]
    for index in range(source.instance_count):
        if view is GeometryView.DEFAULT:
            if len(source.geom_group_visible) and not source.geom_group_visible[index]:
                continue
        elif len(source.geom_role) and not int(source.geom_role[index]) & role_mask:
            continue
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
        geom_segmentation=(
            source.geom_segmentation[indices].copy()
            if len(source.geom_segmentation) == source.instance_count
            else np.full((len(indices), 2), -1, np.int32)
        ),
        geom_body=source.geom_body[indices].copy(),
        geom_source=source.geom_source[indices].copy(),
        geom_pose_source=source.geom_pose_source[indices].copy(),
        geom_visual=source.geom_visual[indices].copy(),
        geom_static=source.geom_static[indices].copy(),
        geom_role=source.geom_role[indices].copy() if len(source.geom_role) else source.geom_role,
        geom_group_visible=source.geom_group_visible[indices].copy()
        if len(source.geom_group_visible)
        else source.geom_group_visible,
        geom_collision_mesh=[source.geom_collision_mesh[i] for i in indices]
        if len(source.geom_collision_mesh)
        else [],
        instance_island_body=source.instance_island_body[indices].copy(),
        geom_node=source.geom_node[indices].copy(),
        geom_local=source.geom_local[indices].copy(),
        geom_infinite_plane=source.geom_infinite_plane[indices].copy(),
    )


def _configure_segmentation(source) -> None:
    pairs: list[tuple[int, int]] = []
    encoded = np.zeros(source.instance_count, np.uint32)
    pair_to_id: dict[tuple[int, int], int] = {}
    for index, values in enumerate(source.geom_segmentation):
        pair = tuple(int(value) for value in values)
        if pair == (-1, -1):
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
    source.geom_segmentation = table[encoded]


def _frame_needs(option, model=None) -> FrameNeeds:
    """Translate visible MuJoCo features into adapter-side dynamic data needs."""

    def visible(*names: str) -> bool:
        return any(flag_enabled(option.flags, mujoco.mjtVisFlag, name) for name in names)

    def present(field: str) -> bool:
        return model is None or int(getattr(model, field, 0)) > 0

    bvh = visible("mjVIS_BODYBVH", "mjVIS_MESHBVH") and present("ngeom")
    contacts = visible(
        "mjVIS_CONTACTPOINT", "mjVIS_CONTACTFORCE", "mjVIS_CONTACTSPLIT"
    ) and present("ngeom")
    actuator = visible("mjVIS_ACTUATOR", "mjVIS_ACTIVATION") and present("nactuator")
    tendons = (visible("mjVIS_TENDON") and present("ntendon")) or actuator
    deformables = visible("mjVIS_SKIN", "mjVIS_FLEXFACE", "mjVIS_FLEXSKIN", "mjVIS_FLEXVERT") and (
        model is None or present("nskin") or present("nflex")
    )
    islands = visible("mjVIS_ISLAND")
    if islands:
        contacts = tendons = deformables = True
    rangefinder = visible("mjVIS_RANGEFINDER") and (
        # MuJoCo enables the flag by default even when a model has no such sensor.
        model is None
        or bool(np.any(np.asarray(model.sensor_type) == int(mujoco.mjtSensor.mjSENS_RANGEFINDER)))
    )
    diagnostics = (
        bvh
        or rangefinder
        or any(
            (
                visible("mjVIS_JOINT") and present("njnt"),
                visible("mjVIS_ACTUATOR", "mjVIS_ACTIVATION") and present("nactuator"),
                visible("mjVIS_CAMERA") and present("ncam"),
                visible("mjVIS_LIGHT") and present("nlight"),
                visible("mjVIS_CONSTRAINT") and present("neq"),
                visible("mjVIS_AUTOCONNECT") and present("nbody"),
                visible("mjVIS_COM", "mjVIS_INERTIA", "mjVIS_SCLINERTIA") and present("nbody"),
            )
        )
    )
    label_none = mode_value(mujoco.mjtLabel, "mjLABEL_NONE")
    frame_none = mode_value(mujoco.mjtFrame, "mjFRAME_NONE")
    diagnostics = diagnostics or int(option.label) != label_none or int(option.frame) != frame_none
    return FrameNeeds(
        poses=True,
        contacts=contacts,
        tendons=tendons,
        actuator=actuator,
        deformables=deformables,
        diagnostics=diagnostics,
        islands=islands,
        bvh=bvh,
    )
