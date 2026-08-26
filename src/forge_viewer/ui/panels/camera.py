"""Camera controls and reproducible scene snapshots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...scene_state import (
    apply_camera_bookmark,
    camera_bookmark,
    capture_scene,
    delete_named_snapshot,
    list_named_snapshots,
    load_named_snapshot,
    restore_scene,
    save_named_snapshot,
)
from ...types import CameraView
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_row_layout,
    button_width,
    labeled,
    value_slider,
)

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
    ("far", 1.0, 100000.0, "%.1f m", 200.0),
)


@runtime_checkable
class CameraLike(Protocol):
    yaw: float
    pitch: float
    distance: float
    fov_y_deg: float
    far: float

    def view(self) -> CameraView: ...


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


class CameraPanel(Panel):
    name = "Camera"
    default_open = True
    shortcut = "F6"

    def __init__(self) -> None:
        super().__init__()
        self._bookmark_name = "view-1"
        self._snapshot_name = "scene-1"
        self._bookmark_index = 0
        self._snapshot_index = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, qpos=True)

    def draw(self, ctx: PanelContext) -> None:
        camera = ctx.camera
        if camera is None:
            imgui.text_disabled("no camera attached to this context")
            return

        self._source(ctx)
        if ctx.model_camera_id >= 0 and ctx.model_camera_view is not None:
            if ctx.select_model_camera is not None and imgui.button(
                ctx.tr("Return to Editor Camera")
            ):
                ctx.select_model_camera(-1)
            imgui.text_disabled(ctx.tr("model camera follows scene kinematics"))
            imgui.separator()
            self._readout_view(ctx, ctx.model_camera_view)
            return

        self._presets(ctx, camera)
        imgui.separator()
        self._params(ctx, camera)
        imgui.separator()
        self._stored_states(ctx, camera)
        imgui.separator()
        self._readout(ctx, camera)

    def _stored_states(self, ctx: PanelContext, camera: Any) -> None:
        if not imgui.collapsing_header(f"{ctx.tr('bookmarks and snapshots')}###camera_states"):
            return
        camera_dir = Path("output/snapshots/cameras")
        scene_dir = Path("output/snapshots/scenes")
        view = ctx.model_camera_view if ctx.model_camera_id >= 0 else camera.view()

        imgui.text_disabled(ctx.tr("camera bookmark"))
        _changed, self._bookmark_name = imgui.input_text("##bookmark_name", self._bookmark_name)
        if imgui.button("save##camera_bookmark"):
            save_named_snapshot(
                self._bookmark_name,
                camera_bookmark(camera, view, ctx.model_camera_id),
                camera_dir,
            )
        imgui.same_line()
        bookmarks = list_named_snapshots(camera_dir)
        self._bookmark_index = min(self._bookmark_index, max(len(bookmarks) - 1, 0))
        if bookmarks:
            imgui.set_next_item_width(140.0 * ctx.style_scale)
            _changed, self._bookmark_index = imgui.combo(
                "##camera_bookmarks", self._bookmark_index, bookmarks
            )
            name = bookmarks[self._bookmark_index]
            if imgui.button("load##camera_bookmark"):
                apply_camera_bookmark(
                    load_named_snapshot(name, camera_dir), camera, ctx.select_model_camera
                )
            imgui.same_line()
            if imgui.button("copy##camera_bookmark"):
                imgui.set_clipboard_text(
                    json.dumps(load_named_snapshot(name, camera_dir), indent=2)
                )
            imgui.same_line()
            if imgui.button("delete##camera_bookmark"):
                delete_named_snapshot(name, camera_dir)

        imgui.text_disabled(ctx.tr("scene snapshot"))
        _changed, self._snapshot_name = imgui.input_text(
            "##scene_snapshot_name", self._snapshot_name
        )
        state_available = ctx.session.adapter.caps.state_snapshots
        imgui.begin_disabled(not state_available)
        if imgui.button("save##scene_snapshot"):
            save_named_snapshot(
                self._snapshot_name,
                capture_scene(
                    ctx.session,
                    ctx.backend,
                    camera,
                    camera_source=ctx.model_camera_id,
                    camera_view=view,
                ),
                scene_dir,
            )
        imgui.end_disabled()
        imgui.same_line()
        snapshots = list_named_snapshots(scene_dir)
        self._snapshot_index = min(self._snapshot_index, max(len(snapshots) - 1, 0))
        if snapshots:
            imgui.set_next_item_width(140.0 * ctx.style_scale)
            _changed, self._snapshot_index = imgui.combo(
                "##scene_snapshots", self._snapshot_index, snapshots
            )
            name = snapshots[self._snapshot_index]
            imgui.begin_disabled(not ctx.session.paused)
            if imgui.button("load##scene_snapshot"):
                try:
                    restore_scene(
                        load_named_snapshot(name, scene_dir),
                        ctx.session,
                        ctx.backend,
                        camera,
                        select_source=ctx.select_model_camera,
                    )
                except ValueError as error:
                    ctx.status = str(error)
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("copy##scene_snapshot"):
                imgui.set_clipboard_text(json.dumps(load_named_snapshot(name, scene_dir), indent=2))
            imgui.same_line()
            if imgui.button("delete##scene_snapshot"):
                delete_named_snapshot(name, scene_dir)

    def _source(self, ctx: PanelContext) -> None:
        cameras = ctx.session.cameras
        if not cameras:
            return
        by_id = {c.camera_id: c.name for c in cameras}
        current = (
            ctx.tr("free")
            if ctx.model_camera_id < 0
            else by_id.get(ctx.model_camera_id, ctx.tr("missing"))
        )
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##camera_source", f"{ctx.tr('source')}: {current}"):
            return
        selected, _ = imgui.selectable(ctx.tr("free"), ctx.model_camera_id < 0)
        if selected and ctx.select_model_camera is not None:
            ctx.select_model_camera(-1)
        for info in cameras:
            selected, _ = imgui.selectable(info.name, ctx.model_camera_id == info.camera_id)
            if selected and ctx.select_model_camera is not None:
                ctx.select_model_camera(info.camera_id)
        imgui.end_combo()
        imgui.separator()

    def _presets(self, ctx: PanelContext, camera: Any) -> None:
        imgui.text_disabled(ctx.tr("presets"))
        has_setter = hasattr(camera, "set_preset") or (
            hasattr(camera, "yaw") and hasattr(camera, "pitch")
        )
        widths = tuple(
            button_width(ctx.tr(label), 64.0 * ctx.style_scale) for label, _yaw, _pitch in PRESETS
        )
        row = button_row_layout(
            widths,
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        imgui.begin_disabled(not has_setter)
        for i, (label, yaw, pitch) in enumerate(PRESETS):
            if row[i]:
                imgui.same_line()
            if imgui.button(ctx.tr(label), imgui.ImVec2(widths[i], 0)):
                if hasattr(camera, "set_preset"):
                    camera.set_preset(label)
                else:
                    camera.yaw = yaw
                    camera.pitch = pitch
        imgui.end_disabled()
        if not has_setter:
            imgui.set_item_tooltip("this camera exposes neither set_preset() nor yaw/pitch")

        can_frame = hasattr(camera, "frame_all")
        frame_width = button_width(ctx.tr("frame all"), 84.0 * ctx.style_scale)
        imgui.begin_disabled(not can_frame)
        if imgui.button(ctx.tr("frame all"), imgui.ImVec2(frame_width, 0)):
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
            changed, value = imgui.checkbox(
                f"{ctx.tr('orthographic')}##editor_camera_orthographic", bool(ortho)
            )
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
        if imgui.button(ctx.tr("copy camera snapshot")):
            imgui.set_clipboard_text(snapshot)
        imgui.set_item_tooltip("copy exact camera and viewport values for bug reproduction")
        qpos = ctx.session.frame.qpos
        imgui.begin_disabled(qpos is None)
        if imgui.button(ctx.tr("copy qpos")) and qpos is not None:
            imgui.set_clipboard_text(qpos_snapshot(qpos))
        imgui.set_item_tooltip("copy the complete generalized position vector")
        if imgui.button(ctx.tr("copy reproduction state")) and qpos is not None:
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
            if view.uses_intrinsics():
                labeled("focal length", "  ".join(f"{v:.8g}" for v in view.focal_length))
                labeled("sensor size", "  ".join(f"{v:.8g}" for v in view.sensor_size))
                labeled(
                    "principal offset",
                    "  ".join(f"{v:+.8g}" for v in view.principal_offset),
                )
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
    if view.uses_intrinsics():
        lines.extend(
            (
                f"focal_length={vec(view.focal_length)}",
                f"sensor_size={vec(view.sensor_size)}",
                f"principal_offset={vec(view.principal_offset)}",
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
