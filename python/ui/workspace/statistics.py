"""Native statistics presentation; StatsPanel owns sampling and the live history."""

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.localization import render_note_text
from mojive.ui.panels.stats import StatsPanel
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.localization import statistics_caption
from mojive.ui.workspace.widgets import DIM, MUTED, TEXT


class NativeStatisticsPresentation:
    """Render real counters, notes and pass timings with measured native rows."""

    def __init__(self, workspace):
        self.workspace = workspace

    @property
    def ui(self):
        return self.workspace.ui

    def _advance(self, x, y, width):
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((width, 0))
        imgui.pop_style_var()
        self._cursor = imgui.ImVec2(x, y)

    def _caption(self, part, rect, text, color, **options):
        self.ui.component_text(part, rect, text, color, **options)
        style = self.ui.component_style(part)
        size = options.get("size", style.get("fontSize", BODY_SIZE))
        if (
            self.ui.measure(str(text), size, translate=options.get("translate", True), style=style)
            > rect.width
        ):
            imgui.internal.item_add(
                imgui.internal.ImRect((rect.x, rect.y), (rect.right, rect.bottom)), 0
            )
            if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                self.ui.tooltip(str(text), translate=options.get("translate", True))

    @part_scope("statistics_list")
    def draw(self, ctx, history, scale_ms, stats, caps):
        ui, scale = self.ui, self.workspace.s
        list_gap = (
            ui.metric(
                "statistics_list",
                "rowGap",
                ui.metric("statistics_list", "gap", DESIGN.data_row_gap),
            )
            * scale
        )
        origin = imgui.get_cursor_screen_pos()
        self._cursor = origin
        width = imgui.get_content_region_avail().x
        font = ui.metric("data_list", "fontSize", BODY_SIZE)
        imgui.push_font(ui.resolve_font(ui.component_style("data_list")), font * scale)
        try:
            recent = ctx.dt * 1000
            imgui.plot_lines(
                "##frame",
                history.values,
                values_offset=history.offset,
                overlay_text=f"{recent:6.2f} ms   {1000 / recent if recent > 0.01 else 0:5.1f} FPS",
                scale_min=0,
                scale_max=scale_ms,
                graph_size=imgui.ImVec2(width, 60 * scale),
            )
            y = imgui.get_item_rect_max().y + list_gap
            line = font * 1.4 * scale
            self._caption(
                "data_list",
                Rect(origin.x, y, width, line),
                f"{ctx.tr('scale')} 0 .. {scale_ms:.1f} ms   (60 FPS = 16.7 ms)",
                DIM,
                translate=False,
            )
            self._advance(origin.x, y + line + list_gap, width)
            self._pairs(
                "stats-counts",
                (
                    (ctx.tr("draw calls"), str(stats.draw_calls)),
                    (ctx.tr("instances"), str(stats.instances)),
                    (ctx.tr("triangles"), f"{stats.triangles:,}"),
                    (ctx.tr("buckets"), str(stats.buckets)),
                    (ctx.tr("frame cpu"), f"{stats.frame_cpu_ms:.3f} ms"),
                ),
            )
            p = self._cursor
            ui.line(p.x, p.y + list_gap / 2, width)
            self._advance(p.x, p.y + list_gap, width)
            if stats.cpu_ms or stats.gpu_ms:
                names = tuple(stats.cpu_ms) + tuple(
                    k for k in stats.gpu_ms if k not in stats.cpu_ms
                )
                rows = tuple(
                    (
                        name,
                        f"{stats.cpu_ms[name]:.3f}" if name in stats.cpu_ms else "—",
                        f"{stats.gpu_ms[name]:.3f}" if name in stats.gpu_ms else "—",
                    )
                    for name in names
                )
                self._table(
                    "stats-passes", (ctx.tr("pass"), ctx.tr("cpu ms"), ctx.tr("gpu ms")), rows
                )
                if not caps.gpu_timing:
                    p = self._cursor
                    self._caption(
                        "data_list",
                        Rect(p.x, p.y, width, line),
                        f"{caps.name}: {ctx.tr('no GPU timer queries on this machine; cpu column only')}",
                        DIM,
                        translate=False,
                    )
                    self._advance(p.x, p.y + line + list_gap, width)
            else:
                p = self._cursor
                self._caption(
                    "data_list",
                    Rect(p.x, p.y, width, line),
                    ctx.tr("no per-pass timing from this backend"),
                    DIM,
                    translate=False,
                )
                self._advance(p.x, p.y + line + list_gap, width)
            if stats.notes:
                self._pairs(
                    "stats-notes",
                    tuple(
                        (
                            statistics_caption(self.workspace.app.localizer, key),
                            render_note_text(value, ctx.tr),
                        )
                        for key, value in stats.notes.items()
                    ),
                )
        finally:
            if ui.design is not None:
                ui.design.record(
                    "statistics-list",
                    ("statistics_list",),
                    (origin.x, origin.y, width, max(0, self._cursor.y - origin.y)),
                    "group",
                )
            imgui.pop_font()

    def _row_height(self, style, font, value_font):
        scale = self.workspace.s
        padding, margin = style.edges("padding", scale, (8, 4, 8, 4)), style.edges("margin", scale)
        line = max(style.line_height(font * 1.4), value_font * 1.4) * scale
        height = (
            max(style.get("height", 28) * scale, line + padding[1] + padding[3])
            + margin[1]
            + margin[3]
        )
        return height, padding

    def _pairs(self, key, rows):
        ui, scale = self.ui, self.workspace.s
        position = self._cursor
        width = imgui.get_content_region_avail().x
        group = ui.component_style("data_list")
        # Counts use a compact shared value column; long diagnostic notes retain
        # a wider reading measure. Authored component widths remain authoritative.
        if not group.explicit("width"):
            width = min(width, (560 if key == "stats-notes" else 360) * scale)
        with ui.component_scope("data_list"):
            row = ui.component_style("stat_row")
        font = row.get("fontSize", group.get("fontSize", BODY_SIZE))
        value_font = group.get("valueSize", font)
        gap = (
            row.get("columnGap", row.get("gap", 12)) * scale
            + group.get("columnGap", group.get("gap", 8)) * scale
        )
        row_gap = group.get("rowGap", group.get("gap", 8)) * scale
        height, row_padding = self._row_height(row, font, value_font)
        padding, margin = group.edges("padding", scale), group.edges("margin", scale)
        natural_height = (
            len(rows) * height
            + max(0, len(rows) - 1) * row_gap
            + padding[1]
            + padding[3]
            + margin[1]
            + margin[3]
        )
        box = ui.component_box(
            "data_list",
            position.x,
            position.y,
            width,
            natural_height,
            key=key,
            padding=padding,
            observe=False,
        )
        list_columns, list_rows, child_rects = [], [], []
        for index, (label, value) in enumerate(rows):
            item = ui.component_box(
                "stat_row",
                box.content.x,
                box.content.y + index * (height + row_gap),
                box.content.width,
                height,
                key=f"{key}-{index}",
                padding=row_padding,
                observe=False,
            )
            label_width = min(item.content.width, group.get("labelWidth", 90) * scale)
            value_width = max(0, item.content.width - label_width - gap)
            value_text_width = min(
                value_width,
                ui.measure(value, value_font, translate=False, mono=True, font_resolved=True),
            )
            align = row.get("align", "space-between")
            if align == "space-between":
                label_rect = Rect(item.content.x, item.content.y, label_width, item.content.height)
                value_rect = Rect(
                    item.content.right - value_width,
                    item.content.y,
                    value_width,
                    item.content.height,
                )
            else:
                placed = row.align_rect(
                    item.content,
                    label_width + gap + value_text_width,
                    item.content.height,
                    horizontal=align,
                )
                label_rect = Rect(placed.x, placed.y, label_width, placed.height)
                value_rect = Rect(
                    placed.x + label_width + gap, placed.y, value_text_width, placed.height
                )
            child_rects.append(list(item.outer.tuple()))
            list_columns.append(
                [
                    label_rect.right + row.get("columnGap", row.get("gap", 12)) * scale,
                    label_rect.y,
                    group.get("columnGap", group.get("gap", 8)) * scale,
                    label_rect.height,
                ]
            )
            if index < len(rows) - 1:
                list_rows.append([item.outer.x, item.outer.bottom, item.outer.width, row_gap])
            row_gap_width = row.get("columnGap", row.get("gap", 12)) * scale
            ui.record_component_box(
                "stat_row",
                item,
                key=f"{key}-{index}",
                fields=[list(label_rect.tuple()), list(value_rect.tuple())],
                fieldRegions={
                    name: [label_rect.right, label_rect.y, row_gap_width, label_rect.height]
                    for name in ("gap", "columnGap")
                },
            )
            with ui.component_scope("data_list"):
                self._caption(
                    "stat_row", label_rect, label, MUTED, size=font, align="start", translate=False
                )
            self._caption(
                "data_list",
                value_rect,
                value,
                TEXT,
                size=value_font,
                font_field="valueSize",
                align="end",
                mono=True,
                translate=False,
            )
        ui.record_component_box(
            "data_list",
            box,
            key=key,
            fields=child_rects,
            fieldRegions={
                "columnGap": list_columns,
                "rowGap": list_rows,
                "gap": [*list_columns, *list_rows],
            },
        )
        self._advance(
            position.x,
            box.outer.bottom
            + ui.metric(
                "statistics_list",
                "rowGap",
                ui.metric("statistics_list", "gap", DESIGN.data_row_gap),
            )
            * scale,
            width,
        )

    def _table(self, key, headers, rows):
        ui, scale = self.ui, self.workspace.s
        position = self._cursor
        width = imgui.get_content_region_avail().x
        table = ui.component_style("table")
        if not table.explicit("width"):
            width = min(width, 560 * scale)
        with ui.component_scope("table"):
            row = ui.component_style("stat_row")
        font = row.get("fontSize", table.get("fontSize", BODY_SIZE))
        height, row_padding = self._row_height(row, font, font)
        padding, margin = table.edges("padding", scale, (7, 5, 7, 5)), table.edges("margin", scale)
        total = (len(rows) + 1) * height + padding[1] + padding[3] + margin[1] + margin[3]
        box = ui.component_box(
            "table", position.x, position.y, width, total, key=key, padding=padding
        )
        gap = row.get("columnGap", row.get("gap", 12)) * scale
        for index, values in enumerate((headers, *rows)):
            item = ui.component_box(
                "stat_row",
                box.content.x,
                box.content.y + index * height,
                box.content.width,
                height,
                key=f"{key}-{index}",
                padding=row_padding,
            )
            if index % 2:
                ui.rect(*item.frame.tuple(), (255, 255, 255, 6), 0)
            first = max(0, item.content.width * 0.5 - gap)
            rest = max(0, (item.content.width - first - gap * 2) / 2)
            widths = (first, rest, rest)
            x = item.content.x
            for column, (value, cell_width) in enumerate(zip(values, widths, strict=True)):
                with ui.component_scope("stat_row"):
                    self._caption(
                        "table",
                        Rect(x, item.content.y, cell_width, item.content.height),
                        value,
                        MUTED if index == 0 or column == 0 else TEXT,
                        size=font,
                        align=("start" if column == 0 else table.get("textAlign", "right"))
                        if row.get("align", "space-between") == "space-between"
                        else row.get("align", "start"),
                        mono=index > 0 and column > 0,
                        translate=False,
                    )
                x += cell_width + gap
        self._advance(
            position.x,
            box.outer.bottom
            + ui.metric(
                "statistics_list",
                "rowGap",
                ui.metric("statistics_list", "gap", DESIGN.data_row_gap),
            )
            * scale,
            width,
        )


class WorkspaceStatistics(StatsPanel):
    """Retain StatsPanel history and consume shared presentation parameters."""

    name = "Statistics"

    def __init__(self, workspace):
        super().__init__(presentation=NativeStatisticsPresentation(workspace))
