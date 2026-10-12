"""Compact timeline selection details using the shared editor's edit boundary."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.keyframe_editor.properties import draw_error, draw_selected

from .widgets import MUTED, SAGE, TEXT


class NativeTimelineDetails:
    """Keep selection identity, fields and actions in one responsive detail bar."""

    def __init__(self, study):
        self.study = study
        self.ctx = None
        self._origin = (0.0, 0.0)
        self._width = 0.0
        self._action_pos = (0.0, 0.0)

    def _gap(self, width):
        return (4 if width < 420 * self.study.s else 6) * self.study.s

    def _identity_width(self, width):
        return (20 if 260 * self.study.s <= width < 420 * self.study.s else 24) * self.study.s

    def _action_widths(self, labels, width):
        ui, s = self.study.ui, self.study.s
        return tuple(
            28 * s
            if (len(labels) > 1 and index == len(labels) - 1)
            or (len(labels) == 3 and index == 0 and width < 420 * s)
            else max(
                44 * s, ui.measure(label, 14) + (20 if len(labels) == 3 and index == 0 else 12) * s
            )
            for index, label in enumerate(labels)
        )

    def _layout(self, width, labels, *, fields):
        s = self.study.s
        gap = self._gap(width)
        widths = self._action_widths(labels, width)
        action_width = sum(widths) + gap * (len(widths) - 1)
        available = max(s, width - self._identity_width(width))
        inline = width >= 260 * s
        time_width = (70 if inline and width < 420 * s else 88) * s
        fields_inline = inline or available >= 72 * s + time_width + gap
        content_height = (28 if fields_inline or not fields else 62) * s
        actions_y = 0 if inline else content_height + 6 * s
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
        height = max(content_height, actions_y + (28 + 34 * (action_rows - 1)) * s)
        return widths, inline, fields_inline, name_width, time_width, actions_y, height

    def height(self, editor, ctx, width):
        """Return physical height, including the empty row and current error feedback."""
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
            else 28 * self.study.s
        )
        return height + (34 * self.study.s if editor.error else 0)

    def draw(self, editor, ctx, editable):
        self.ctx = ctx
        p = imgui.get_cursor_screen_pos()
        self._origin, self._width = (p.x, p.y), imgui.get_content_region_avail().x
        imgui.push_font(self.study.ui.body, 14 * self.study.s)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (8 * self.study.s, 7 * self.study.s))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.8 * self.study.s)
        imgui.begin_group()
        try:
            if editor.selected_snapshot >= 0:
                self._snapshot(editor)
                draw_error(editor, ctx, presentation=self)
            else:
                draw_selected(editor, ctx, editable, presentation=self)
            self._reserve_empty_row()
        finally:
            imgui.end_group()
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            self.study.ui.hits["timeline-detail-bar"] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            imgui.pop_style_var(2)
            imgui.pop_font()

    def _reserve_empty_row(self):
        _, y = self._origin
        if imgui.get_cursor_screen_pos().y < y + 28 * self.study.s:
            self._advance(y + 28 * self.study.s)

    def _advance(self, y):
        imgui.set_cursor_screen_pos((self._origin[0], y))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((self._width, 0))
        imgui.pop_style_var()

    def _identity(self, icon):
        x, y = self._origin
        self.study.ui.role_icon(
            icon, x, y, 20 * self.study.s, 28 * self.study.s, role="inline", c=MUTED
        )

    def _record(self, key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.study.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def fields(self, name, time):
        ui, s = self.study.ui, self.study.s
        x, y = self._origin
        labels = (self.ctx.tr("Apply"), self.ctx.tr("Load"), self.ctx.tr("Delete"))
        _, _, inline, name_width, time_width, actions_y, _ = self._layout(
            self._width, labels, fields=True
        )
        identity_width = self._identity_width(self._width)
        gap = self._gap(self._width)
        self._identity("diamond")
        imgui.set_cursor_screen_pos((x + identity_width, y))
        imgui.set_next_item_width(name_width)
        _, name = imgui.input_text("##keyframe-name", name)
        self._record("timeline-detail-name")
        if imgui.is_item_hovered():
            ui.tooltip(f"{self.ctx.tr('name')}: {name}" if name else self.ctx.tr("name"))
        time_x = x + identity_width + name_width + gap if inline else x + identity_width
        time_y = y if inline else y + 34 * s
        imgui.set_cursor_screen_pos((time_x, time_y))
        imgui.set_next_item_width(time_width)
        _, time = imgui.input_double("##keyframe-time", time, 0.0, 0.0, "%.9g s")
        self._record("timeline-detail-time")
        if imgui.is_item_hovered():
            ui.tooltip(self.ctx.tr("time"))
        self._action_pos = (x, y + actions_y)
        return name, time

    def _buttons(self, labels, keys, enabled, y):
        ui, s = self.study.ui, self.study.s
        gap = self._gap(self._width)
        widths = self._action_widths(labels, self._width)
        total = sum(widths) + gap * (len(widths) - 1)
        x = self._origin[0] + max(0, self._width - total)
        row_x = x
        clicked = []
        for label, key, width, active in zip(labels, keys, widths, enabled, strict=True):
            if x > row_x and x + width > self._origin[0] + self._width:
                x, y = self._origin[0], y + 34 * s
            icon = None
            if key == "timeline-detail-delete":
                icon = "delete"
            elif key == "timeline-detail-apply" and self._width < 420 * s:
                icon = "check"
            clicked.append(
                ui.button(
                    key,
                    x,
                    y,
                    min(width, self._width) / s,
                    label="" if icon else label,
                    icon=icon,
                    icon_color=SAGE if key == "timeline-detail-apply" and active else None,
                    enabled=active,
                    filled=key == "timeline-detail-apply" and active,
                    align="center",
                    tooltip=label,
                )
            )
            x += width + gap
        self._advance(y + 28 * s)
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
        ui, s = self.study.ui, self.study.s
        x, y = self._origin
        labels = (
            (self.ctx.tr("Load"), self.ctx.tr("Delete")) if deletable else (self.ctx.tr("Load"),)
        )
        _, _, _, name_width, time_width, actions_y, _ = self._layout(
            self._width, labels, fields=False
        )
        identity_width = self._identity_width(self._width)
        self._identity("link" if deletable else "diamond")
        ui.text(x + identity_width, y + 7 * s, name, TEXT, 14, max_width=name_width)
        ui.hits["timeline-detail-name"] = (x + identity_width, y, name_width, 28 * s)
        if ui.measure(name, 14) > name_width and imgui.is_mouse_hovering_rect(
            (x + identity_width, y), (x + identity_width + name_width, y + 28 * s)
        ):
            ui.tooltip(name)
        time_x = x + identity_width + name_width + self._gap(self._width)
        ui.text(
            time_x,
            y + 7 * s,
            f"{time:g} {self.ctx.tr('s')}",
            MUTED,
            14,
            mono=True,
            max_width=time_width,
        )
        ui.hits["timeline-detail-time"] = (time_x, y, time_width, 28 * s)
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
        ui, s = self.study.ui, self.study.s
        x = self._origin[0]
        y = max(imgui.get_cursor_screen_pos().y, self._origin[1] + 28 * s) + 6 * s
        danger = tuple(round(value * 255) for value in self.ctx.theme.danger)
        ui.text(x, y + 7 * s, message, danger, 14, max_width=max(s, self._width - 34 * s))
        ui.hits["timeline-detail-error"] = (x, y, max(s, self._width - 34 * s), 28 * s)
        copy = ui.button(
            "timeline-detail-copy-error",
            x + max(0, self._width - 28 * s),
            y,
            icon="copy",
            tooltip=self.ctx.tr("Copy error"),
        )
        if imgui.is_mouse_hovering_rect((x, y), (x + max(s, self._width - 34 * s), y + 28 * s)):
            ui.tooltip(message)
        self._advance(y + 28 * s)
        return copy
