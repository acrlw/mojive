"""Measure section/body spacing once around each owned native property table."""

import math
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.design.groups import scoped_parts
from mojive.design.native.geometry import Rect, budget_margins
from mojive.ui.workspace.design import DESIGN


@dataclass(slots=True)
class _TableGroup:
    key: str
    table_id: int
    origin: tuple[float, float]
    width: float
    first: bool = True


class PropertyGroups:
    """Own table extents without changing Inspector or settings transactions."""

    def __init__(self, ui):
        self.ui = ui
        self.tables = []

    def begin(self, key):
        table = imgui.internal.get_current_table()
        position = imgui.get_cursor_screen_pos()
        self.tables.append(
            _TableGroup(
                key, table.id_, (position.x, position.y), imgui.get_content_region_avail().x
            )
        )

    def row(self, x, y, width):
        table = imgui.internal.get_current_table()
        if not self.tables or table is None or self.tables[-1].table_id != table.id_:
            return x, y, width
        owner = self.tables[-1]
        ui, scale = self.ui, self.ui.s
        section = budget_margins(
            ui.component_style("section").edges("margin", scale), width, math.inf
        )
        body = ui.component_style("section_body")
        margin, padding = body.edges("margin", scale), body.edges("padding", scale)
        if owner.first:
            owner.origin, owner.width = (x, y), width
            y += section[1] + margin[1] + padding[1]
            owner.first = False
        section_frame = Rect(x, y, width, 0).inset(section[0], 0, section[2], 0)
        margin = budget_margins(margin, section_frame.width, math.inf)
        body_frame = section_frame.inset(margin[0], 0, margin[2], 0)
        content = body_frame.inset(padding[0], 0, padding[2], 0)
        return content.x, y, content.width

    def end(self):
        table = imgui.internal.get_current_table()
        if not self.tables or table is None or self.tables[-1].table_id != table.id_:
            imgui.end_table()
            return
        owner = self.tables.pop()
        imgui.end_table()
        ui, scale = self.ui, self.ui.s
        position = imgui.get_cursor_screen_pos()
        section = budget_margins(
            ui.component_style("section").edges("margin", scale), owner.width, math.inf
        )
        body = ui.component_style("section_body")
        margin, padding = body.edges("margin", scale), body.edges("padding", scale)
        bottom = position.y + section[3] + margin[3] + padding[3]
        observe_section = ui.design is not None and ui.design.runtime.observes(
            scoped_parts("section")
        )
        observe_body = ui.design is not None and ui.design.runtime.observes(
            scoped_parts("section_body")
        )
        if observe_section or observe_body:
            outer = Rect(*owner.origin, owner.width, max(scale, bottom - owner.origin[1]))
            section_frame = outer.inset(*section)
            if observe_section:
                ui.design.record(
                    "section/" + owner.key,
                    ("section",),
                    section_frame.tuple(),
                    "section",
                    outer=outer.tuple(),
                    margin=list(section),
                )
        if observe_body:
            margin = budget_margins(margin, section_frame.width, section_frame.height)
            body_frame = section_frame.inset(*margin)
            ui.design.record(
                "section-body/" + owner.key,
                ("section_body",),
                body_frame.tuple(),
                "section_body",
                outer=section_frame.tuple(),
                content=body_frame.inset(*padding).tuple(),
                margin=list(margin),
                padding=list(padding),
                rowGap=body.get("rowGap", DESIGN.row_gap),
                fields=[
                    record["hit"]
                    for record in ui.design.runtime.observed
                    if record["key"].startswith(owner.key + "/row/")
                ],
            )
        imgui.set_cursor_screen_pos((owner.origin[0], position.y))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
        imgui.dummy((0, max(0, bottom - position.y)))
        imgui.pop_style_var()
