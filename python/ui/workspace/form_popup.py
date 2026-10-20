"""Measured compact forms with independent native popup geometry."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, floor

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, independent_corner_scope, part_scope
from mojive.ui.workspace.corner_geometry import nested_corners
from mojive.ui.workspace.design import BODY_SIZE
from mojive.ui.workspace.menu_geometry import native_edges, native_unit
from mojive.ui.workspace.panel_geometry import corner_inset
from mojive.ui.workspace.window_chrome import surface_ink


@dataclass(frozen=True)
class FormPopupLayout:
    width: float
    height: float
    padding: tuple[float, float, float, float]
    row_height: float
    row_gap: float
    group_gap: float
    label_width: float
    column_gap: float
    radius: float
    inner_radius: float
    row_tops: tuple[float, ...]

    def row(self, origin, index):
        return Rect(
            origin.x + self.padding[0],
            origin.y + self.row_tops[index],
            self.width - self.padding[0] - self.padding[2],
            self.row_height,
        )

    def field(self, origin, index):
        row = self.row(origin, index)
        start = row.x + self.label_width + self.column_gap
        return Rect(start, row.y, max(0, row.right - start), row.height)


@independent_corner_scope()
def form_corners(ui, part="form_popup", *, control_part="input_number", inner_limit=None):
    """Balance the shell, native inset and contained control corners."""
    with part_scope(part, inherit=False):
        scale, style = ui.s, ui.component_style(part)
        padding_style = style if part == "dialog" else ui.component_style("form_popup")
        default_padding = 12 if part == "dialog" else 8
        padding = max(native_edges(padding_style, "padding", scale, (default_padding,) * 4))
        control = ui.component_style(control_part)
        inner = max(0, control.get("radius", 4) if control.explicit("radius") else 4) * scale
        if inner_limit is not None:
            inner = min(inner, inner_limit)
        corners = nested_corners(style, scale, (padding,) * 4, inner)
        if corners.authored_radius:
            # Smaller shells reduce both the inset and the child corner. Round
            # the inset first so their sum remains exact at fractional UI scales.
            inset_ratio = default_padding / (default_padding + 4)
            padding = min(padding, native_unit(corners.radius * inset_ratio), floor(corners.radius))
            if inner_limit is not None:
                padding = corner_inset(corners.radius, padding, inner_limit)
        return nested_corners(style, scale, (padding,) * 4, inner)


@part_scope("form_popup", inherit=False)
@independent_corner_scope()
def measure_form_popup(ui, labels, *, field_width, row_height, group_starts=()):
    """Resolve all layout extents on the same grid as ImGui's native windows."""
    scale, style = ui.s, ui.component_style("form_popup")
    gap = native_unit(style.get("columnGap", 12) * scale)
    row_gap = native_unit(style.get("rowGap", 6) * scale)
    group_gap = native_unit(style.get("groupGap", 12) * scale)
    label_style = ui.component_style("form_label")
    label_font = label_style.get("fontSize", style.get("fontSize", BODY_SIZE))
    label_width = ceil(
        max(
            (
                ui.measure(label + " ", label_font, translate=False, style=label_style)
                for label in labels
            ),
            default=0,
        )
    )
    row_height = ceil(max(row_height, label_style.line_height(label_font * 1.4) * scale))
    limit = max(0, min(field_width, row_height) / 2 - 1)
    corners = form_corners(ui, inner_limit=limit)
    padding = corners.padding
    width = ceil(label_width + gap + field_width + padding[0] + padding[2])
    width = min(width, native_unit(imgui.get_main_viewport().work_size.x - 16 * scale))
    label_width = min(label_width, max(0, width - padding[0] - padding[2] - gap - field_width))
    tops, bottom = [], padding[1]
    for index in range(len(labels)):
        if index:
            bottom += group_gap if index in group_starts else row_gap
        tops.append(bottom)
        bottom += row_height
    return FormPopupLayout(
        width,
        bottom + padding[3],
        padding,
        row_height,
        row_gap,
        group_gap,
        label_width,
        gap,
        corners.radius,
        corners.inner_radius,
        tuple(tops),
    )


@contextmanager
def form_popup(ui, key, layout):
    """Own the complete form extent, without trailing native row spacing."""
    with part_scope("form_popup", inherit=False):
        style = ui.component_style("form_popup")
        imgui.set_next_window_size((layout.width, layout.height))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.popup_rounding, layout.radius)
        imgui.push_style_color(
            imgui.Col_.popup_bg, tuple(v / 255 for v in surface_ink(ui, "form_popup", "popup"))
        )
        imgui.push_font(ui.resolve_font(style), style.get("fontSize", BODY_SIZE) * ui.s)
        opened = imgui.begin_popup(key)
        try:
            with ui.corner_scope(layout.inner_radius, ("input_number", "axis_input")):
                yield opened
        finally:
            if opened:
                imgui.end_popup()
            imgui.pop_font()
            imgui.pop_style_color()
            imgui.pop_style_var(4)
