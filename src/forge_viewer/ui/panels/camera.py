"""Camera controls and named camera bookmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...scene_state import (
    apply_camera_bookmark,
    camera_bookmark,
    delete_named_snapshot,
    list_named_snapshots,
    load_named_snapshot,
    next_available_snapshot_name,
    save_named_snapshot,
)
from ...types import CameraView
from ..camera import DEFAULT_PITCH, DEFAULT_YAW
from . import (
    Panel,
    PanelContext,
    segmented_control,
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
    ("yaw", -180.0, 180.0, "%.1f deg", DEFAULT_YAW),
    ("pitch", -89.9, 89.9, "%.1f deg", DEFAULT_PITCH),
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
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._bookmark_name = "view-1"
        self._bookmark_index = 0
        self._bookmark_error = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

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
            return

        self._presets(ctx, camera)
        imgui.separator()
        self._params(ctx, camera)
        imgui.separator()
        self._stored_states(ctx, camera)

    def _stored_states(self, ctx: PanelContext, camera: Any) -> None:
        if not imgui.collapsing_header(f"{ctx.tr('camera bookmarks')}###camera_states"):
            return
        camera_dir = Path("output/snapshots/cameras")
        view = ctx.model_camera_view if ctx.model_camera_id >= 0 else camera.view()

        imgui.text_disabled(ctx.tr("camera bookmark"))
        imgui.text_disabled(f"{ctx.tr('Stored in')}: {camera_dir}")
        imgui.set_item_tooltip(str(camera_dir.resolve()))
        _changed, self._bookmark_name = imgui.input_text("##bookmark_name", self._bookmark_name)
        if imgui.button("save##camera_bookmark"):
            try:
                name = next_available_snapshot_name(self._bookmark_name, camera_dir)
                path = save_named_snapshot(
                    name,
                    camera_bookmark(camera, view, ctx.model_camera_id),
                    camera_dir,
                )
                bookmarks = list_named_snapshots(camera_dir)
                self._bookmark_index = bookmarks.index(path.stem)
                self._bookmark_name = next_available_snapshot_name(path.stem, camera_dir)
                self._bookmark_error = ""
                ctx.report(
                    f"Saved camera bookmark to {path.resolve()}",
                    level="success",
                )
            except (OSError, TypeError, ValueError) as error:
                self._report_storage_error(ctx, error)
        imgui.same_line()
        bookmarks = list_named_snapshots(camera_dir)
        self._bookmark_index = min(self._bookmark_index, max(len(bookmarks) - 1, 0))
        if bookmarks:
            imgui.set_next_item_width(140.0 * ctx.style_scale)
            changed, self._bookmark_index = imgui.combo(
                "##camera_bookmarks", self._bookmark_index, bookmarks
            )
            name = bookmarks[self._bookmark_index]
            if changed:
                try:
                    apply_camera_bookmark(
                        load_named_snapshot(name, camera_dir), camera, ctx.select_model_camera
                    )
                    self._bookmark_error = ""
                    ctx.report(f"Loaded camera bookmark '{name}'", level="success")
                except (OSError, KeyError, TypeError, ValueError) as error:
                    self._report_storage_error(ctx, error)
            imgui.same_line()
            if imgui.button("copy##camera_bookmark"):
                try:
                    imgui.set_clipboard_text(
                        json.dumps(load_named_snapshot(name, camera_dir), indent=2)
                    )
                except (OSError, TypeError, ValueError) as error:
                    self._report_storage_error(ctx, error)
            imgui.same_line()
            if imgui.button("delete##camera_bookmark"):
                try:
                    delete_named_snapshot(name, camera_dir)
                    ctx.report(f"Deleted camera bookmark '{name}'", level="success")
                except OSError as error:
                    self._report_storage_error(ctx, error)

        if self._bookmark_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._bookmark_error)
            if imgui.small_button("Copy error##camera_bookmark"):
                imgui.set_clipboard_text(self._bookmark_error)

    def _report_storage_error(self, ctx: PanelContext, error: Exception) -> None:
        self._bookmark_error = str(error)
        ctx.report(self._bookmark_error, level="error", duration=10.0)

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
        can_frame = hasattr(camera, "frame_all")
        entries = (*PRESETS, ("frame all", 0.0, 0.0))
        flags = imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("camera_presets", 4, flags):
            return
        for index, (label, yaw, pitch) in enumerate(entries):
            if index % 4 == 0:
                imgui.table_next_row()
            imgui.table_next_column()
            enabled = can_frame if label == "frame all" else has_setter
            imgui.begin_disabled(not enabled)
            if imgui.button(ctx.tr(label), imgui.ImVec2(-1.0, 0.0)):
                if label == "frame all":
                    lo, hi = ctx.session.bounds()
                    camera.frame_all(lo, hi)
                elif hasattr(camera, "set_preset"):
                    camera.set_preset(label)
                else:
                    camera.yaw = yaw
                    camera.pitch = pitch
            imgui.end_disabled()
        imgui.end_table()

    def _params(self, ctx: PanelContext, camera: Any) -> None:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("camera_properties", 2, flags):
            return
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.38)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch, 0.62)
        for attr, lo, hi, fmt, initial in PARAM_SLIDERS:
            current = _get(camera, attr)
            if current is None:
                continue
            self._property_label(attr)
            imgui.set_next_item_width(-1.0)
            edit = value_slider(
                f"##camera-{attr}",
                float(current),
                lo,
                hi,
                initial=initial,
                fmt=fmt,
                more_hint="none",
            )
            if edit.changed:
                setattr(camera, attr, edit.value)

        ortho = _get(camera, "orthographic")
        if ortho is not None:
            self._property_label(ctx.tr("projection"))
            supported = ctx.backend.caps.orthographic
            imgui.begin_disabled(not supported)
            selected = segmented_control(
                "camera-projection",
                (ctx.tr("persp"), ctx.tr("ortho")),
                1 if bool(ortho) else 0,
                theme=ctx.theme,
            )
            imgui.end_disabled()
            if not supported:
                imgui.set_item_tooltip(f"{ctx.backend.caps.name} has no orthographic projection")
            else:
                camera.orthographic = selected == 1
        imgui.end_table()

    @staticmethod
    def _property_label(label: str) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        available = imgui.get_content_region_avail().x
        width = imgui.calc_text_size(label).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, available - width))
        imgui.text_disabled(label)
        imgui.table_next_column()
