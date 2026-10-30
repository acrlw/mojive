"""Camera controls and named camera bookmarks."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds, NodeType
from mojive.scene.state import (
    apply_camera_bookmark,
    camera_bookmark,
    delete_named_snapshot,
    list_named_snapshots,
    load_named_snapshot,
    next_available_snapshot_name,
    save_named_snapshot,
)
from mojive.types import CameraView
from mojive.ui.camera import DEFAULT_PITCH, DEFAULT_YAW
from mojive.ui.camera_tracking import can_track_node
from mojive.ui.controls import begin_property_table, clearable_combo, property_row
from mojive.ui.localization import camera_caption
from mojive.ui.panels import (
    Panel,
    PanelContext,
    ValueEdit,
    button_width,
    search_input,
    segmented_control,
)
from mojive.ui.panels.value_cards import value_rail

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


def camera_preset_column_count(available_width: float, minimum_cell_width: float) -> int:
    """Keep the eight camera presets in a balanced grid without clipping labels."""

    available = max(0.0, float(available_width))
    required = max(1.0, float(minimum_cell_width))
    if available >= 4.0 * required:
        return 4
    if available >= 2.0 * required:
        return 2
    return 1


@runtime_checkable
class CameraLike(Protocol):
    yaw: float
    pitch: float
    distance: float
    fov_y_deg: float
    far: float

    def view(self) -> CameraView: ...


class CameraPresentation(Protocol):
    """Optional camera control presentation; CameraPanel retains actions and state."""

    def section(self, label: str, flags: int = 0) -> bool: ...
    def begin_table(self, table_id: str) -> bool: ...
    def end_table(self) -> None: ...
    def row(self, label: str, *, rail: bool = False, control_part="input_number") -> None: ...
    def value_rail(self, item_id: str, value: float, bounds, **options) -> ValueEdit: ...
    def segmented(self, item_id: str, labels, selected: int, *, icons=None) -> int: ...
    def combo(self, key: str, index: int, labels, flags: int = 0, *, preview=None): ...
    def clearable_combo(self, key: str, preview: str, *, choices=(), **options): ...
    def choice(self, label: str, selected: bool = False, *, enabled: bool = True): ...
    def choice_filter(self, key: str, value: str, *, hint: str, clear_tooltip: str): ...
    def input_text(self, key: str, value: str, flags: int = 0): ...
    def button(self, label: str, size=None) -> bool: ...


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


class CameraPanel(Panel):
    id = "camera"
    name = "Camera"
    default_open = True
    shortcut = "F6"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self.bookmark_directory = Path("output/snapshots/cameras")
        self._bookmark_name = ""
        self._bookmark_index = 0
        self._bookmark_error = ""
        self._tracking_filter = ""
        self._angular_degrees = True
        self._initial_distance = None
        self.presentation: CameraPresentation | None = None

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        camera = ctx.camera
        if camera is None:
            imgui.text_disabled(ctx.tr("no camera attached to this context"))
            return

        self._source(ctx)
        if ctx.model_camera_id >= 0 and ctx.model_camera_view is not None:
            if ctx.select_model_camera is not None and imgui.button(
                ctx.tr("Return to Editor Camera")
            ):
                ctx.select_model_camera(-1)
            imgui.text_disabled(ctx.tr("model camera follows scene kinematics"))
            imgui.text_wrapped(
                ctx.tr("Return to Editor Camera to orbit, pan, zoom, or frame objects.")
            )
            return

        self._params(ctx, camera)
        if self._section(ctx.tr("Presets"), imgui.TreeNodeFlags_.default_open):
            self._presets(ctx, camera)
        self._stored_states(ctx, camera)
        self._tracking(ctx)

    def _tracking(self, ctx: PanelContext) -> None:
        if ctx.tracking is None or ctx.track_node is None:
            return
        if not self._section(
            f"{ctx.tr('Tracking')}###camera_tracking", imgui.TreeNodeFlags_.default_open
        ):
            return
        node = ctx.session.node(ctx.tracking_node_id) if ctx.tracking_node_id is not None else None
        if self._begin_properties(ctx, "tracking_properties"):
            self._property_label(ctx.tr("Target"), control_part="select")
            current = node.name if node is not None else ctx.tr("Choose target")
            imgui.set_next_item_width(-1)
            selected_node = ctx.session.selected_node
            selected_label = ctx.tr("Use selection")
            if selected_node is not None:
                selected_label += f": {selected_node.name}"
            candidates = tuple(
                candidate
                for candidate in ctx.session.nodes
                if candidate.type in (NodeType.ROBOT, NodeType.LINK)
                or (candidate.type is NodeType.GEOM and candidate.body_index < 0)
                or candidate.node_id == ctx.tracking_node_id
            )
            combo = self.presentation.clearable_combo if self.presentation else clearable_combo
            choice = self.presentation.choice if self.presentation else imgui.selectable
            options = {}
            if self.presentation:
                options["choices"] = (selected_label, *(candidate.name for candidate in candidates))
            else:
                imgui.set_next_window_size_constraints((240 * ctx.style_scale, 0), (10000, 10000))
            with combo(
                "##tracking-target",
                current,
                has_value=node is not None,
                clear_tooltip=ctx.tr("Stop tracking"),
                tooltip=node.name
                if node is not None
                else ctx.tr("Choose an object to start tracking."),
                **options,
            ) as (opened, cleared):
                if cleared:
                    ctx.track_node(None)
                if opened:
                    imgui.begin_disabled(not can_track_node(selected_node))
                    if choice(selected_label + "##track-selected", False)[0]:
                        ctx.track_node(selected_node.node_id)
                    imgui.end_disabled()
                    if self.presentation:
                        _, self._tracking_filter = self.presentation.choice_filter(
                            "##tracking-filter",
                            self._tracking_filter,
                            hint=ctx.tr("Filter targets"),
                            clear_tooltip=ctx.tr("Clear search"),
                        )
                    else:
                        imgui.separator()
                        imgui.set_next_item_width(-1)
                        _, self._tracking_filter = search_input(
                            "##tracking-filter",
                            self._tracking_filter,
                            hint=ctx.tr("Filter targets"),
                            search_tooltip=ctx.tr("Filter targets"),
                            clear_tooltip=ctx.tr("Clear search"),
                        )
                    query = self._tracking_filter.casefold()
                    for candidate in candidates:
                        if query not in candidate.name.casefold():
                            continue
                        if choice(
                            f"{candidate.name}##tracking-{candidate.node_id}",
                            candidate.node_id == ctx.tracking_node_id,
                        )[0]:
                            ctx.track_node(candidate.node_id)
            self._property_label(ctx.tr("Axes"))
            axes = self._segmented(
                "tracking-axes",
                ("X-Y", "X-Y-Z"),
                int(ctx.tracking.axes == "xyz"),
                theme=ctx.theme,
            )
            self._tooltip(ctx.tr("X-Y keeps camera height. X-Y-Z also follows height."))
            if axes != int(ctx.tracking.axes == "xyz"):
                ctx.set_camera_tracking(replace(ctx.tracking, axes="xyz" if axes else "xy"))
            self._property_label(ctx.tr("Smoothing"), rail=True)
            imgui.set_next_item_width(-1)
            edit = self._rail(
                ctx,
                "##tracking-smoothing",
                ctx.tracking.smoothing,
                (0.0, 2.0),
                initial=0.25,
                fmt="%.2f",
                show_reset=False,
                unit="s",
            )
            self._tooltip(
                ctx.tr("Time to halve the position error. Higher is smoother; 0 follows directly.")
            )
            if edit.changed:
                ctx.set_camera_tracking(replace(ctx.tracking, smoothing=max(0.0, edit.value)))
            self._end_properties()
        if self.presentation is None:
            imgui.separator()

    def save_bookmark(self, ctx: PanelContext, camera: Any, *, replace_name=None) -> bool:
        """Persist a new name or explicitly update the selected bookmark."""
        try:
            view = ctx.model_camera_view if ctx.model_camera_id >= 0 else camera.view()
            name = replace_name or next_available_snapshot_name(
                self._bookmark_name.strip() or "view-1", self.bookmark_directory
            )
            path = save_named_snapshot(
                name, camera_bookmark(camera, view, ctx.model_camera_id), self.bookmark_directory
            )
            names = list_named_snapshots(self.bookmark_directory)
            self._bookmark_index = names.index(path.stem)
            if replace_name is None:
                self._bookmark_name = ""
            self._bookmark_error = ""
            ctx.report(f"{ctx.tr('Saved camera bookmark to')} {path.resolve()}", level="success")
            return True
        except (OSError, TypeError, ValueError) as error:
            self._report_storage_error(ctx, error)
            return False

    def load_bookmark(self, ctx: PanelContext, camera: Any, name: str) -> bool:
        try:
            apply_camera_bookmark(
                load_named_snapshot(name, self.bookmark_directory), camera, ctx.select_model_camera
            )
            self._bookmark_index = list_named_snapshots(self.bookmark_directory).index(name)
            self._bookmark_error = ""
            ctx.report(f"{ctx.tr('Loaded camera bookmark')} '{name}'", level="success")
            return True
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._report_storage_error(ctx, error)
            return False

    def copy_bookmark(self, ctx: PanelContext, name: str) -> bool:
        try:
            imgui.set_clipboard_text(
                json.dumps(load_named_snapshot(name, self.bookmark_directory), indent=2)
            )
            self._bookmark_error = ""
            return True
        except (OSError, TypeError, ValueError) as error:
            self._report_storage_error(ctx, error)
            return False

    def delete_bookmark(self, ctx: PanelContext, name: str) -> bool:
        try:
            delete_named_snapshot(name, self.bookmark_directory)
            self._bookmark_error = ""
            ctx.report(f"{ctx.tr('Deleted camera bookmark')} '{name}'", level="success")
            return True
        except OSError as error:
            self._report_storage_error(ctx, error)
            return False

    def _stored_states(self, ctx: PanelContext, camera: Any) -> None:
        opened = self._section(f"{ctx.tr('camera bookmarks')}###camera_states")
        self._tooltip(f"{ctx.tr('Stored in')}: {self.bookmark_directory.resolve()}")
        if not opened:
            return
        imgui.set_next_item_width(-1.0)
        input_text = (
            self.presentation.input_text if self.presentation is not None else imgui.input_text
        )
        _changed, self._bookmark_name = input_text("##bookmark_name", self._bookmark_name)
        if self._bookmark_button(ctx, "Save"):
            self.save_bookmark(ctx, camera)
        bookmarks = list_named_snapshots(self.bookmark_directory)
        self._bookmark_index = min(self._bookmark_index, max(len(bookmarks) - 1, 0))
        if bookmarks:
            imgui.set_next_item_width(-1.0)
            combo = self.presentation.combo if self.presentation is not None else imgui.combo
            changed, self._bookmark_index = combo(
                "##camera_bookmarks", self._bookmark_index, bookmarks
            )
            name = bookmarks[self._bookmark_index]
            if changed:
                self.load_bookmark(ctx, camera, name)
            if self._bookmark_button(ctx, "Copy"):
                self.copy_bookmark(ctx, name)
            if self.presentation is None:
                imgui.same_line()
            if self._bookmark_button(ctx, "Delete"):
                self.delete_bookmark(ctx, name)
        if self._bookmark_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), ctx.tr(self._bookmark_error))
            if self._bookmark_button(ctx, "Copy error", small=True):
                imgui.set_clipboard_text(self._bookmark_error)

    def _bookmark_button(self, ctx, label, *, small=False):
        if self.presentation is not None:
            return self.presentation.button(ctx.tr(label))
        button = imgui.small_button if small else imgui.button
        return button(f"{ctx.tr(label)}##camera_bookmark")

    def _report_storage_error(self, ctx: PanelContext, error: Exception) -> None:
        self._bookmark_error = str(error)
        ctx.report(self._bookmark_error, level="error", duration=10.0)

    def _source(self, ctx: PanelContext) -> None:
        cameras = ctx.session.cameras
        if not cameras:
            return
        by_id = {c.camera_id: c.name for c in cameras}
        current = (
            ctx.tr("Editor Camera")
            if ctx.model_camera_id < 0
            else by_id.get(ctx.model_camera_id, ctx.tr("missing"))
        )
        if self.presentation is not None:
            ids = (-1, *(info.camera_id for info in cameras))
            labels = (ctx.tr("Editor Camera"), *(info.name for info in cameras))
            index = ids.index(ctx.model_camera_id) if ctx.model_camera_id in ids else -1
            imgui.set_next_item_width(-1)
            changed, selected = self.presentation.combo(
                "##camera_source", index, labels, preview=f"{ctx.tr('source')}: {current}"
            )
            if changed and ctx.select_model_camera is not None:
                ctx.select_model_camera(ids[selected])
            return
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##camera_source", f"{ctx.tr('source')}: {current}"):
            return
        selected, _ = imgui.selectable(ctx.tr("Editor Camera"), ctx.model_camera_id < 0)
        if selected and ctx.select_model_camera is not None:
            ctx.select_model_camera(-1)
        for info in cameras:
            selected, _ = imgui.selectable(info.name, ctx.model_camera_id == info.camera_id)
            if selected and ctx.select_model_camera is not None:
                ctx.select_model_camera(info.camera_id)
        imgui.end_combo()
        if self.presentation is None:
            imgui.separator()

    def _presets(self, ctx: PanelContext, camera: Any) -> None:
        has_setter = hasattr(camera, "set_preset") or (
            hasattr(camera, "yaw") and hasattr(camera, "pitch")
        )
        can_frame = hasattr(camera, "frame_all")
        entries = (*PRESETS, ("frame all", 0.0, 0.0))
        labels = tuple(ctx.tr(label) for label, _yaw, _pitch in entries)
        style = imgui.get_style()
        minimum_cell_width = max(button_width(label) for label in labels) + 2.0 * float(
            style.cell_padding.x
        )
        columns = camera_preset_column_count(
            imgui.get_content_region_avail().x,
            minimum_cell_width,
        )
        flags = imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("camera_presets", columns, flags):
            return
        for index, ((label, yaw, pitch), display_label) in enumerate(
            zip(entries, labels, strict=True)
        ):
            if index % columns == 0:
                imgui.table_next_row()
            imgui.table_next_column()
            enabled = can_frame if label == "frame all" else has_setter
            imgui.begin_disabled(not enabled)
            if imgui.button(display_label, imgui.ImVec2(-1.0, 0.0)):
                if label == "frame all":
                    if ctx.track_node is not None:
                        ctx.track_node(None)
                    lo, hi = ctx.session.bounds()
                    camera.frame_all(lo, hi)
                elif hasattr(camera, "set_preset"):
                    camera.set_preset(label)
                else:
                    camera.yaw = yaw
                    camera.pitch = pitch
            imgui.end_disabled()
        imgui.end_table()

    def _toggle_angle_unit(self):
        self._angular_degrees = not self._angular_degrees

    def _params(self, ctx: PanelContext, camera: Any) -> None:
        if self._initial_distance is None:
            self._initial_distance = _get(camera, "distance")
        for title, parameters in (("View", PARAM_SLIDERS[:3]), ("Lens", PARAM_SLIDERS[3:])):
            if not self._section(ctx.tr(title), imgui.TreeNodeFlags_.default_open):
                continue
            if not self._begin_properties(ctx, f"camera_{title.lower()}"):
                continue
            for attr, lo, hi, fmt, initial in parameters:
                current = _get(camera, attr)
                if current is None:
                    continue
                # Orbit transitions retain unwrapped yaw for the shortest animation path.
                if attr == "yaw":
                    current = (float(current) + 180.0) % 360.0 - 180.0
                angular = attr in ("yaw", "pitch", "fov_y_deg")
                factor = math.pi / 180.0 if angular else 1.0
                initial = self._initial_distance if initial is None else initial
                editor_distance = attr == "distance" and camera is ctx.camera
                if editor_distance:
                    lo, hi = camera.navigation.min_distance, camera.navigation.max_distance
                caption = (
                    "field of view"
                    if attr == "fov_y_deg" and self.presentation is not None
                    else attr
                )
                self._property_label(
                    camera_caption(caption, ctx.language, ctx.tr),
                    rail=True,
                    tooltip=ctx.tr("Vertical field of view") if attr == "fov_y_deg" else "",
                )
                edit = self._rail(
                    ctx,
                    f"##camera-{attr}",
                    float(current) * factor,
                    (lo * factor, hi * factor),
                    initial=None if initial is None else initial * factor,
                    fmt="%.3f" if angular and not self._angular_degrees else fmt.split()[0],
                    show_reset=False,
                    unit="rad" if angular else "m",
                    angular_degrees=self._angular_degrees,
                    toggle_unit=self._toggle_angle_unit,
                    logarithmic=editor_distance,
                )
                if edit.changed:
                    setattr(camera, attr, edit.value / factor)
            if title == "Lens":
                self._projection(ctx, camera)
            self._end_properties()

    def _projection(self, ctx: PanelContext, camera: Any) -> None:
        ortho = _get(camera, "orthographic")
        if ortho is not None:
            self._property_label(ctx.tr("projection"))
            imgui.set_next_item_width(-1.0)
            supported = ctx.backend.caps.orthographic
            imgui.begin_disabled(not supported)
            selected = self._segmented(
                "camera-projection",
                (ctx.tr("persp"), ctx.tr("ortho")),
                1 if bool(ortho) else 0,
                theme=ctx.theme,
                icons=("persp", "ortho"),
            )
            imgui.end_disabled()
            if not supported:
                self._tooltip(
                    f"{ctx.backend.caps.name}: {ctx.tr('orthographic projection unavailable')}"
                )
            else:
                target = selected == 1
                if target != bool(ortho):
                    setter = getattr(camera, "set_orthographic", None)
                    if setter is not None:
                        setter(target, animate=True)
                    else:
                        camera.orthographic = target

    def _section(self, label, flags=0):
        return (
            self.presentation.section(label, flags)
            if self.presentation
            else imgui.collapsing_header(label, flags)
        )

    def _rail(self, ctx, item_id, value, bounds, **options):
        return (
            self.presentation.value_rail(item_id, value, bounds, **options)
            if self.presentation
            else value_rail(ctx, item_id, value, bounds, **options)
        )

    def _segmented(self, item_id, labels, selected, *, theme, icons=None):
        return (
            self.presentation.segmented(item_id, labels, selected, icons=icons)
            if self.presentation
            else segmented_control(item_id, labels, selected, theme=theme, icons=icons)
        )

    def _end_properties(self):
        if self.presentation:
            self.presentation.end_table()
        else:
            imgui.end_table()

    def _begin_properties(self, ctx, str_id):
        if self.presentation:
            return self.presentation.begin_table(str_id)
        labels = tuple(
            ctx.tr(name)
            for name in (
                "yaw",
                "pitch",
                "distance",
                "fov_y_deg",
                "far",
                "projection",
                "Target",
                "Axes",
                "Smoothing",
            )
        )
        return begin_property_table(str_id, labels=labels)

    def _property_label(
        self, label: str, *, rail=False, control_part="input_number", tooltip=""
    ) -> None:
        if self.presentation:
            self.presentation.row(label, rail=rail, control_part=control_part)
        else:
            property_row(label, wrap=True)
        if tooltip:
            self._tooltip(tooltip)
