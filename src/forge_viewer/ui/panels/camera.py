from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...types import CameraView
from . import Panel, PanelContext, begin_kv_table, labeled, value_slider

#


PRESETS: tuple[tuple[str, float, float], ...] = (
    ("front", -90.0, 0.0),
    ("back", 90.0, 0.0),
    ("left", 180.0, 0.0),
    ("right", 0.0, 0.0),
    ("top", -90.0, 89.9),
    ("bottom", -90.0, -89.9),
    ("iso", -135.0, 30.0),
)


PARAM_SLIDERS: tuple[tuple[str, float, float, str, float | None], ...] = (
    ("yaw", -180.0, 180.0, "%.1f deg", -135.0),
    ("pitch", -89.9, 89.9, "%.1f deg", 30.0),
    ("distance", 0.05, 200.0, "%.3f m", None),
    ("fov_y_deg", 10.0, 120.0, "%.1f deg", 45.0),
)


@runtime_checkable
class CameraLike(Protocol):
    yaw: float
    pitch: float
    distance: float
    fov_y_deg: float

    def view(self) -> CameraView: ...


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


class CameraPanel(Panel):
    name = "Camera"
    default_open = True
    shortcut = "F6"

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, qpos=True)

    def draw(self, ctx: PanelContext) -> None:
        camera = ctx.camera
        if camera is None:
            imgui.text_disabled("no camera attached to this context")
            return

        self._source(ctx)
        if ctx.model_camera_id >= 0 and ctx.model_camera_view is not None:
            imgui.text_disabled("model camera follows scene kinematics; choose free to orbit")
            imgui.separator()
            self._readout_view(ctx, ctx.model_camera_view)
            return

        self._presets(ctx, camera)
        imgui.separator()
        self._params(ctx, camera)
        imgui.separator()
        self._readout(ctx, camera)

    def _source(self, ctx: PanelContext) -> None:
        cameras = ctx.session.cameras
        if not cameras:
            return
        by_id = {c.camera_id: c.name for c in cameras}
        current = "free" if ctx.model_camera_id < 0 else by_id.get(ctx.model_camera_id, "missing")
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##camera_source", f"source: {current}"):
            return
        selected, _ = imgui.selectable("free", ctx.model_camera_id < 0)
        if selected and ctx.select_model_camera is not None:
            ctx.select_model_camera(-1)
        for info in cameras:
            selected, _ = imgui.selectable(info.name, ctx.model_camera_id == info.camera_id)
            if selected and ctx.select_model_camera is not None:
                ctx.select_model_camera(info.camera_id)
        imgui.end_combo()
        imgui.separator()

    def _presets(self, ctx: PanelContext, camera: Any) -> None:
        imgui.text_disabled("presets")
        has_setter = hasattr(camera, "set_preset") or (
            hasattr(camera, "yaw") and hasattr(camera, "pitch")
        )
        imgui.begin_disabled(not has_setter)
        for i, (label, yaw, pitch) in enumerate(PRESETS):
            if i % 4:
                imgui.same_line()
            if imgui.button(label, imgui.ImVec2(64, 0)):
                if hasattr(camera, "set_preset"):
                    camera.set_preset(label)
                else:
                    camera.yaw = yaw
                    camera.pitch = pitch
        imgui.end_disabled()
        if not has_setter:
            imgui.set_item_tooltip("this camera exposes neither set_preset() nor yaw/pitch")

        can_frame = hasattr(camera, "frame_all")
        imgui.begin_disabled(not can_frame)
        if imgui.button("frame all", imgui.ImVec2(84, 0)):
            lo, hi = ctx.session.bounds()
            camera.frame_all(lo, hi)
        imgui.end_disabled()
        if not can_frame:
            imgui.set_item_tooltip("this camera has no frame_all()")

    def _params(self, ctx: PanelContext, camera: Any) -> None:
        for attr, lo, hi, fmt, initial in PARAM_SLIDERS:
            current = _get(camera, attr)
            if current is None:
                continue
            edit = value_slider(
                f"{attr}##cam", float(current), lo, hi, initial=initial, fmt=fmt, more_hint="none"
            )
            if edit.changed:
                setattr(camera, attr, edit.value)

        ortho = _get(camera, "orthographic")
        if ortho is not None:
            supported = ctx.backend.caps.orthographic
            imgui.begin_disabled(not supported)
            changed, value = imgui.checkbox("orthographic", bool(ortho))
            imgui.end_disabled()
            if not supported:
                imgui.set_item_tooltip(f"{ctx.backend.caps.name} has no orthographic projection")
            elif changed:
                camera.orthographic = value

    def _readout(self, ctx: PanelContext, camera: Any) -> None:
        view = camera.view() if hasattr(camera, "view") else None
        if view is None:
            imgui.text_disabled("this camera exposes no view()")
            return
        self._readout_view(ctx, view, camera)

    def _readout_view(self, ctx: PanelContext, view: CameraView, camera: Any = None) -> None:
        snapshot = camera_snapshot(view, ctx.viewport_rect, camera)
        if imgui.button("copy camera snapshot"):
            imgui.set_clipboard_text(snapshot)
        imgui.set_item_tooltip("copy exact camera and viewport values for bug reproduction")
        qpos = ctx.session.frame.qpos
        imgui.begin_disabled(qpos is None)
        if imgui.button("copy qpos") and qpos is not None:
            imgui.set_clipboard_text(qpos_snapshot(qpos))
        imgui.set_item_tooltip("copy the complete generalized position vector")
        if imgui.button("copy reproduction state") and qpos is not None:
            imgui.set_clipboard_text(reproduction_snapshot(ctx, view, camera))
        imgui.set_item_tooltip("copy the model, qpos, simulation time, camera, and viewport")
        imgui.end_disabled()
        if begin_kv_table("cam_kv"):
            if camera is not None:
                labeled(
                    "orbit",
                    f"yaw {_get(camera, 'yaw', 0.0):+.4f}  "
                    f"pitch {_get(camera, 'pitch', 0.0):+.4f}  "
                    f"distance {_get(camera, 'distance', 0.0):.6g}",
                )
            labeled("eye", "  ".join(f"{v:+.6f}" for v in view.eye))
            labeled("target", "  ".join(f"{v:+.6f}" for v in view.target))
            labeled("up", "  ".join(f"{v:+.6f}" for v in view.up))
            labeled("fov y", f"{view.fov_y:.8f} rad / {math.degrees(view.fov_y):.5f} deg")
            labeled("near / far", f"{view.near:.8g} / {view.far:.8g}")
            labeled("aspect", f"{view.aspect:.8f}")
            labeled("viewport", "  ".join(f"{v:.1f}" for v in ctx.viewport_rect))
            labeled("projection", "orthographic" if view.orthographic else "perspective")
            imgui.end_table()


def camera_snapshot(
    view: CameraView,
    viewport_rect: tuple[float, float, float, float],
    camera: Any = None,
) -> str:

    def vec(values) -> str:
        return "(" + ", ".join(f"{float(v):+.9g}" for v in values) + ")"

    lines = ["forge-viewer camera"]
    if camera is not None:
        lines.append(
            "orbit="
            f"(yaw={float(_get(camera, 'yaw', 0.0)):+.9g}, "
            f"pitch={float(_get(camera, 'pitch', 0.0)):+.9g}, "
            f"distance={float(_get(camera, 'distance', 0.0)):.9g})"
        )
    lines.extend(
        (
            f"eye={vec(view.eye)}",
            f"target={vec(view.target)}",
            f"up={vec(view.up)}",
            f"fov_y_rad={float(view.fov_y):.12g}",
            f"near={float(view.near):.12g}",
            f"far={float(view.far):.12g}",
            f"aspect={float(view.aspect):.12g}",
            f"orthographic={bool(view.orthographic)}",
            "viewport=" + vec(viewport_rect),
        )
    )
    return "\n".join(lines)


def qpos_snapshot(qpos) -> str:
    values = ", ".join(f"{float(value):+.9g}" for value in qpos)
    return f"qpos=[{values}]"


def reproduction_snapshot(ctx: PanelContext, view: CameraView, camera: Any = None) -> str:
    session = ctx.session
    frame = session.frame
    asset = str(session.asset_path) if session.asset_path is not None else "<runtime scene>"
    lines = [
        "forge-viewer reproduction",
        f"asset={asset}",
        f"physics_backend={session.adapter.caps.name}",
        f"render_backend={ctx.backend.caps.name}",
        f"time={float(frame.time):.12g}",
        f"step={int(frame.step)}",
        qpos_snapshot(frame.qpos),
        "",
        camera_snapshot(view, ctx.viewport_rect, camera),
    ]
    return "\n".join(lines)
