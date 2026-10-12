"""Native spacing for the shared live renderer statistics panel."""

from imgui_bundle import imgui

from mojive.ui.panels.stats import StatsPanel

from .design import DESIGN


class WorkspaceStatistics(StatsPanel):
    name = "Statistics"

    def draw(self, ctx):
        style = imgui.get_style()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            (style.item_spacing.x, DESIGN.data_row_gap * ctx.style_scale),
        )
        imgui.push_style_var(
            imgui.StyleVar_.cell_padding,
            (style.cell_padding.x, DESIGN.data_row_gap * ctx.style_scale / 2),
        )
        try:
            super().draw(ctx)
        finally:
            imgui.pop_style_var(2)
