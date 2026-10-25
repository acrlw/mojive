"""Adopted workspace dimensions in logical points, before platform DPI scaling."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from mojive.ui.control_style import RailStyle


@dataclass(frozen=True, slots=True)
class WorkspaceDesign:
    """Shared geometry for native drawing, hit testing and responsive layout."""

    body_size: float = 14
    title_size: float = 16
    mono_size: float = 14
    control_height: float = 28
    control_radius: float = 6
    checkbox_size: float = 18
    switch_track_width: float = 44
    switch_track_height: float = 24
    switch_thumb_size: float = 20
    container_radius: float = 8
    grab_radius: float = 4
    child_radius: float = 6
    scrollbar_padding: float = 8
    scrollbar_grab_width: float = 3
    panel_padding_x: float = 12
    panel_padding_y: float = 10
    frame_padding_x: float = 8
    frame_padding_y: float = 8
    item_spacing_x: float = 6
    row_gap: float = 8
    numeric_gap: float = 4
    label_width: float = 120
    label_minimum: float = 48
    label_gap: float = 12
    application_height: float = 36
    sidebar_width: float = 48
    scene_width: float = 252
    inspector_width: float = 320
    bottom_height: float = 276
    timeline_column_gap: float = 8
    timeline_row_gap: float = 8
    timeline_lane_height: float = 36
    timeline_ruler_size: float = 12
    timeline_lane_size: float = 12
    timeline_label_gap: float = 14
    timeline_sheet_gap: float = 2
    output_row_gap: float = 4
    menu_item_height: float = 30
    menu_row_gap: float = 2
    menu_padding_x: float = 6
    navigation_size: float = 90
    navigation_margin: float = 12
    transport_padding_y: float = 6
    range_thumb_size: float = 12
    range_track_size: float = 4
    data_row_gap: float = 10
    camera_right: float = 0
    camera_bottom: float = 0
    file_receipt_padding: float = 4
    rail_entry_width: float = 84
    rail_unit_width: float = 24
    rail_minimum_track_width: float = 64

    def control_padding_y(self, font_size: float) -> float:
        return max(0, (self.control_height - font_size) / 2)


DEFAULT_DESIGN = WorkspaceDesign()
_DESIGN = ContextVar("workspace_design", default=DEFAULT_DESIGN)


class _DesignView:
    def __getattr__(self, name):
        return getattr(_DESIGN.get(), name)


DESIGN = _DesignView()


def current_design():
    return _DESIGN.get()


@contextmanager
def use_design(resolve):
    token = _DESIGN.set(resolve(DEFAULT_DESIGN))
    try:
        yield
    finally:
        _DESIGN.reset(token)


def rail_style():
    return RailStyle(
        thumb_size=DESIGN.range_thumb_size,
        track_size=DESIGN.range_track_size,
        entry_width=DESIGN.rail_entry_width,
        unit_width=DESIGN.rail_unit_width,
        minimum_track_width=DESIGN.rail_minimum_track_width,
        unit_accent=False,
        degree_unit="°",
    )


RAIL_STYLE = rail_style()
BODY_SIZE = DESIGN.body_size
TITLE_SIZE = DESIGN.title_size
MONO_SIZE = DESIGN.mono_size
CONTROL_HEIGHT = DESIGN.control_height
CONTROL_RADIUS = DESIGN.control_radius
