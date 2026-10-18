"""Compact timeline selection details using the shared editor's edit boundary."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.keyframe_editor.properties import draw_error, draw_selected
from mojive.ui.workspace.design import CONTROL_RADIUS
from mojive.ui.workspace.widgets import ACCENT, MUTED, TEXT


class NativeTimelineDetails:
    """Keep selection identity, fields and actions in one responsive detail bar."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.ctx = None
        self._origin = (0.0, 0.0)
        self._width = 0.0
        self._action_pos = (0.0, 0.0)

    def _style(self):
        return self.workspace.ui.component_style("timeline_details")

    def _row_height(self):
        style = self._style()
        height = (
            max(style.get("height", 28), style.line_height(style.get("fontSize", 14) * 1.4))
            * self.workspace.s
        )
        ui = self.workspace.ui
        if ui.design is not None:
            for part in ("input_text", "input_number"):
                margin = ui.component_style(part).edges("margin", self.workspace.s)
                height = max(
                    height,
                    ui.design.primitives.field_height(part) * self.workspace.s
                    + margin[1]
                    + margin[3],
                )
        return height

    def _row_gap(self):
        return self._style().get("rowGap", self._style().get("gap", 6)) * self.workspace.s

    def _gap(self, width):
        style = self._style()
        return style.get("columnGap", style.get("gap", 6)) * self.workspace.s

    def _identity_width(self, width):
        extent = self._style().get("iconSize", 20) * self.workspace.s
        return extent + self._gap(width)

    def _action_widths(self, labels, width):
        ui, s = self.workspace.ui, self.workspace.s
        return tuple(
            28 * s
            if (len(labels) > 1 and index == len(labels) - 1)
            or (len(labels) == 3 and index == 0 and width < 420 * s)
            else max(
                44 * s,
                ui.measure(label, self._style().get("fontSize", 14))
                + (20 if len(labels) == 3 and index == 0 else 12) * s,
            )
            for index, label in enumerate(labels)
        )

    def _layout(self, width, labels, *, fields):
        s = self.workspace.s
        gap = self._gap(width)
        widths = self._action_widths(labels, width)
        action_width = sum(widths) + gap * (len(widths) - 1)
        available = max(s, width - self._identity_width(width))
        inline = width >= 260 * s
        time_width = (70 if inline and width < 420 * s else 88) * s
        fields_inline = inline or available >= 72 * s + time_width + gap
        row_height, row_gap = self._row_height(), self._row_gap()
        content_height = row_height if fields_inline or not fields else 2 * row_height + row_gap
        actions_y = 0 if inline else content_height + row_gap
        if inline:
            available -= action_width + gap
        time_width = min(time_width, available)
        if not fields:
            time_width = min(time_width, available * 0.45)
        name_width = (
            max(s, available - time_width - gap) if fields_inline or not fields else available
        )
        if fields:
            name_width = min(280 * s, name_width)
        action_rows, row_width = 1, 0.0
        for button_width in widths:
            if row_width and row_width + button_width > width:
                action_rows += 1
                row_width = 0.0
            row_width += button_width + gap
        height = max(
            content_height, actions_y + row_height * action_rows + row_gap * (action_rows - 1)
        )
        return widths, inline, fields_inline, name_width, time_width, actions_y, height

    def height(self, editor, ctx, width):
        """Return physical height, including the empty row and current error feedback."""
        style = self._style()
        margin, padding = (
            style.edges("margin", self.workspace.s),
            style.edges("padding", self.workspace.s),
        )
        width = max(self.workspace.s, width - margin[0] - margin[2] - padding[0] - padding[2])
        selected = editor.selected_snapshot >= 0 or (
            editor.edit_lane == "model"
            and editor.selected_id >= 0
            and len(editor.selected_keyframes) <= 1
        )
        labels = (
            (ctx.tr("Load"), ctx.tr("Delete"))
            if editor.selected_snapshot >= 0
            else (ctx.tr("Apply"), ctx.tr("Load"), ctx.tr("Delete"))
        )
        height = (
            self._layout(width, labels, fields=editor.selected_snapshot < 0)[-1]
            if selected
            else self._row_height()
        )
        return (
            height
            + (self._row_height() + self._row_gap() if editor.error else 0)
            + margin[1]
            + margin[3]
            + padding[1]
            + padding[3]
        )

    @part_scope("timeline_details")
    def draw(self, editor, ctx, editable):
        self.ctx = ctx
        p = imgui.get_cursor_screen_pos()
        ui, scale = self.workspace.ui, self.workspace.s
        width = imgui.get_content_region_avail().x
        box = ui.component_box(
            "timeline_details",
            p.x,
            p.y,
            width,
            self.height(editor, ctx, width),
            key="timeline-detail-container",
        )
        self._origin, self._width = (box.content.x, box.content.y), box.content.width
        font_size = self._style().get("fontSize", 14)
        imgui.set_cursor_screen_pos(self._origin)
        imgui.push_font(ui.resolve_font(self._style()), font_size * scale)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (8 * scale, max(0, (self._row_height() - font_size * scale) / 2)),
        )
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, CONTROL_RADIUS * self.workspace.s)
        imgui.begin_group()
        try:
            if editor.selected_snapshot >= 0:
                self._snapshot(editor)
                draw_error(editor, ctx, presentation=self)
            else:
                draw_selected(editor, ctx, editable, presentation=self)
            self._reserve_empty_row()
            self._advance(
                max(
                    box.outer.bottom,
                    imgui.get_cursor_screen_pos().y + box.padding[3] + box.margin[3],
                )
            )
        finally:
            imgui.end_group()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.workspace.ui.hits["timeline-detail-bar"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            imgui.pop_style_var(2)
            imgui.pop_font()

    def _reserve_empty_row(self):
        _, y = self._origin
        if imgui.get_cursor_screen_pos().y < y + self._row_height():
            self._advance(y + self._row_height())

    def _advance(self, y):
        imgui.set_cursor_screen_pos((self._origin[0], y))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((self._width, 0))
        imgui.pop_style_var()

    def _identity(self, icon):
        x, y = self._origin
        style = self._style()
        size = style.get("iconSize", 20) * self.workspace.s
        rect = style.align_rect(
            Rect(x, y, self._identity_width(self._width), self._row_height()),
            size,
            size,
            horizontal="start",
        )
        self.workspace.ui.icon(icon, rect.x, rect.y, size / self.workspace.s, MUTED)

    def _record(self, key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.workspace.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def fields(self, name, time):
        ui = self.workspace.ui
        x, y = self._origin
        labels = (self.ctx.tr("Apply"), self.ctx.tr("Load"), self.ctx.tr("Delete"))
        _, _, inline, name_width, time_width, actions_y, _ = self._layout(
            self._width, labels, fields=True
        )
        identity_width = self._identity_width(self._width)
        gap = self._gap(self._width)
        self._identity("diamond")
        _, name = ui.input_text("##keyframe-name", x + identity_width, y, name_width, name)
        self._record("timeline-detail-name")
        if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            ui.tooltip(
                f"{self.ctx.tr('name')}: {name}" if name else self.ctx.tr("name"), translate=False
            )
        time_x = x + identity_width + name_width + gap if inline else x + identity_width
        time_y = y if inline else y + self._row_height() + self._row_gap()
        _, time = ui.input("##keyframe-time", time_x, time_y, time_width, time, format="%.9g s")
        self._record("timeline-detail-time")
        if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            ui.tooltip(self.ctx.tr("time"), translate=False)
        self._action_pos = (x, y + actions_y)
        return name, time

    def _buttons(self, labels, keys, enabled, y):
        ui, s = self.workspace.ui, self.workspace.s
        gap = self._gap(self._width)
        widths = self._action_widths(labels, self._width)
        total = sum(widths) + gap * (len(widths) - 1)
        x = self._origin[0] + max(0, self._width - total)
        row_x = x
        clicked = []
        for label, key, width, active in zip(labels, keys, widths, enabled, strict=True):
            if x > row_x and x + width > self._origin[0] + self._width:
                x, y = self._origin[0], y + self._row_height() + self._row_gap()
            icon = None
            if key == "timeline-detail-delete":
                icon = "delete"
            elif key == "timeline-detail-apply" and self._width < 420 * s:
                icon = "check"
            clicked.append(
                ui.button(
                    key,
                    x,
                    self._style()
                    .align_rect(Rect(x, y, width, self._row_height()), width, 28 * s)
                    .y,
                    min(width, self._width) / s,
                    label="" if icon else label,
                    icon=icon,
                    icon_color=ACCENT if key == "timeline-detail-apply" and active else None,
                    enabled=active,
                    filled=key == "timeline-detail-apply" and active,
                    align="center",
                    tooltip=label,
                )
            )
            x += width + gap
        self._advance(y + self._row_height())
        return tuple(clicked)

    def actions(self, *, applicable, loadable, deletable):
        labels = (self.ctx.tr("Apply"), self.ctx.tr("Load"), self.ctx.tr("Delete"))
        return self._buttons(
            labels,
            ("timeline-detail-apply", "timeline-detail-load", "timeline-detail-delete"),
            (applicable, loadable, deletable),
            self._action_pos[1],
        )

    def _marker(self, name, time, *, loadable, deletable):
        ui = self.workspace.ui
        x, y = self._origin
        labels = (
            (self.ctx.tr("Load"), self.ctx.tr("Delete")) if deletable else (self.ctx.tr("Load"),)
        )
        _, _, _, name_width, time_width, actions_y, _ = self._layout(
            self._width, labels, fields=False
        )
        identity_width = self._identity_width(self._width)
        self._identity("snapshot" if deletable else "diamond")
        ui.component_text(
            "timeline_details",
            Rect(x + identity_width, y, name_width, self._row_height()),
            name,
            TEXT,
        )
        ui.hits["timeline-detail-name"] = (x + identity_width, y, name_width, self._row_height())
        if ui.measure(name, self._style().get("fontSize", 14)) > name_width:
            imgui.internal.item_add(
                imgui.internal.ImRect(
                    (x + identity_width, y),
                    (x + identity_width + name_width, y + self._row_height()),
                ),
                0,
            )
            if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
                ui.tooltip(name, translate=False)
        time_x = x + identity_width + name_width + self._gap(self._width)
        ui.component_text(
            "timeline_details",
            Rect(time_x, y, time_width, self._row_height()),
            f"{time:g} {self.ctx.tr('s')}",
            MUTED,
            mono=True,
        )
        ui.hits["timeline-detail-time"] = (time_x, y, time_width, self._row_height())
        return self._buttons(
            labels,
            ("timeline-detail-load", "timeline-detail-delete")
            if deletable
            else ("timeline-detail-load",),
            (loadable, True) if deletable else (loadable,),
            y + actions_y,
        )

    def marker(self, name, time, *, loadable):
        return self._marker(name, time, loadable=loadable, deletable=False)[0]

    def _snapshot(self, editor):
        ctx = self.ctx
        snapshot = next(
            (
                item
                for item in ctx.session.scene_snapshots
                if item.snapshot_id == editor.selected_snapshot
            ),
            None,
        )
        if snapshot is None:
            editor.selected_snapshot = -1
            return
        loadable = (
            ctx.session.paused
            and not ctx.session.state_take_playing
            and not ctx.session.state_take_recording
            and not ctx.take_video_active
        )
        load, delete = self._marker(snapshot.name, snapshot.time, loadable=loadable, deletable=True)
        if load:
            result = ctx.submit(cmd.RestoreSceneSnapshot(snapshot.snapshot_id))
            editor.error = "" if result.ok else result.message
            if result.ok:
                editor.playhead = snapshot.time
        if delete:
            result = ctx.submit(cmd.RemoveSceneSnapshot(snapshot.snapshot_id))
            editor.error = "" if result.ok else result.message
            if result.ok:
                editor.selected_snapshot = -1

    def error(self, message):
        ui, s = self.workspace.ui, self.workspace.s
        x = self._origin[0]
        y = (
            max(imgui.get_cursor_screen_pos().y, self._origin[1] + self._row_height())
            + self._row_gap()
        )
        danger = tuple(round(value * 255) for value in self.ctx.theme.danger)
        ui.component_text(
            "timeline_details",
            Rect(x, y, max(s, self._width - 34 * s), self._row_height()),
            message,
            danger,
        )
        ui.hits["timeline-detail-error"] = (x, y, max(s, self._width - 34 * s), self._row_height())
        copy = ui.button(
            "timeline-detail-copy-error",
            x + max(0, self._width - 28 * s),
            y,
            icon="copy",
            tooltip=self.ctx.tr("Copy error"),
        )
        imgui.internal.item_add(
            imgui.internal.ImRect(
                (x, y), (x + max(s, self._width - 34 * s), y + self._row_height())
            ),
            0,
        )
        if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
            ui.tooltip(message, translate=False)
        self._advance(y + self._row_height())
        return copy
