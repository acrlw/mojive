"""Interaction reference rows reuse the status bar's input hints and keycaps."""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import InputAction
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.text_layout import text_line_y
from mojive.ui.viewport_widgets.hints import (
    _tool_hint_width,
    draw_mouse_hint_glyph,
    draw_tool_hints,
    pointer_tool_hint,
)
from mojive.ui.viewport_widgets.model import ToolHint
from mojive.ui.workspace.design import BODY_SIZE
from mojive.ui.workspace.status_components import StatusBadge
from mojive.ui.workspace.widgets import MUTED, TEXT


@dataclass(frozen=True)
class GuideRow:
    title: str
    hints: tuple[ToolHint, ...]
    note: str = ""


@lru_cache(maxsize=32)
def interaction_rows(bindings, mac=False):
    """Resolve the same primary mouse chords and current keys as status hints."""

    def key(action):
        return (
            ToolHint("key", bindings.label(action))
            if bindings.binding(action).key is not None
            else ToolHint("text", label="Unbound")
        )

    def mouse(action):
        hint = pointer_tool_hint(action, bindings, "", hint_id=action.value)
        return hint if hint is not None else ToolHint("text", label="Unbound")

    modifier = "Cmd" if mac else "Ctrl"
    return (
        GuideRow("Orbit", (mouse(PointerAction.ORBIT),), "Drag an empty area"),
        GuideRow("Pan", (mouse(PointerAction.PAN),), "Drag"),
        GuideRow("Zoom", (mouse(PointerAction.DOLLY),)),
        GuideRow(
            "Fly camera",
            tuple(
                key(action)
                for action in (
                    InputAction.FLY_FORWARD,
                    InputAction.FLY_LEFT,
                    InputAction.FLY_BACK,
                    InputAction.FLY_RIGHT,
                    InputAction.FLY_UP,
                    InputAction.FLY_DOWN,
                )
            ),
        ),
        GuideRow(
            "Transform tools",
            (
                ToolHint("key", "V"),
                key(InputAction.GIZMO_TRANSLATE),
                key(InputAction.GIZMO_ROTATE),
            ),
        ),
        GuideRow("World / body frame", (key(InputAction.GIZMO_SPACE),)),
        GuideRow("Snap", (key(InputAction.SNAP),)),
        GuideRow("Frame all / focus", (key(InputAction.FRAME_SCENE), mouse(PointerAction.FOCUS))),
        GuideRow(
            "Settings",
            (ToolHint("key", "F9"), ToolHint("text", label="/"), ToolHint("key", modifier + "+,")),
        ),
        GuideRow(
            "Precise gizmo input", (mouse(PointerAction.GIZMO_VALUE),), "Double-click a handle"
        ),
        GuideRow(
            "Undo / redo",
            (
                ToolHint("key", modifier + "+Z"),
                ToolHint("text", label="/"),
                ToolHint("key", modifier + "+Shift+Z"),
            ),
        ),
        GuideRow("Simulation playback", (key(InputAction.TOGGLE_PAUSE),)),
        GuideRow("Timeline", (ToolHint("key", "T"),)),
        GuideRow("Dock panels", (ToolHint("mouse", "left"),), "Drag a panel tab"),
    )


def draw_interaction_rows(ui, bindings, width):
    """Wrap complete input indicators without shrinking or dropping guide content."""
    # The guide owns row spacing independently of its native host.
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (imgui.get_style().item_spacing.x, 0))
    try:
        return _draw_interaction_rows(ui, bindings, width)
    finally:
        imgui.pop_style_var()


def _draw_interaction_rows(ui, bindings, width):
    scale = ui.s
    width = min(width, imgui.get_content_region_avail().x)
    style = ui.component_style("dialog_body")
    font = style.get("fontSize", BODY_SIZE)
    line = max(font, style.line_height(font * 1.4)) * scale
    gap = style.get("rowGap", 8) * scale
    input_gap = 8 * scale
    draw = ui.text_draw(ImguiDraw2D(imgui.get_window_draw_list()), "dialog_body")
    badge = StatusBadge(ui, keycap=True, font_size=font)
    slot = ceil(badge.size("Esc")[1])
    badge.minimum_size = (slot / scale, slot / scale)
    theme = ui.control_theme()
    rows = interaction_rows(bindings, imgui.get_io().config_mac_osx_behaviors)

    def hint_width(hint):
        if hint.kind != "mouse":
            return _tool_hint_width(draw, scale, hint, keycap=badge)
        prefix = (
            _tool_hint_width(draw, scale, ToolHint("key", hint.modifier, "+"), keycap=badge)
            + input_gap
            if hint.modifier
            else 0
        )
        return prefix + slot + (5 * scale + draw.text_size(hint.suffix)[0] if hint.suffix else 0)

    def paint_hint(hint, x, center_y):
        if hint.kind != "mouse":
            return draw_tool_hints(draw, (x, center_y), theme, scale, (hint,), keycap=badge)
        cursor = x
        if hint.modifier:
            cursor += (
                draw_tool_hints(
                    draw,
                    (cursor, center_y),
                    theme,
                    scale,
                    (ToolHint("key", hint.modifier, "+"),),
                    keycap=badge,
                )
                + input_gap
            )
        cursor += draw_mouse_hint_glyph(
            draw,
            cursor,
            center_y,
            hint.control,
            hint.suffix,
            theme,
            scale,
            size=(slot / scale, slot / scale),
        )
        return cursor - x

    note_inset = (
        max((sum(hint_width(hint) for hint in row.hints) for row in rows if row.note), default=0)
        + input_gap
    )
    fields = []
    for index, row in enumerate(rows):
        origin = imgui.get_cursor_screen_pos()
        widths = [hint_width(hint) for hint in row.hints]
        height = ceil(max(line, slot))
        column = width * 0.4
        stacked = (
            ui.measure(row.title, font, style=style) + input_gap > column
            or max(widths, default=0) > width - column
        )
        if stacked:
            ui.component_text(
                "dialog_body", Rect(origin.x, origin.y, width, line), row.title, MUTED
            )
            left, y, available = origin.x, origin.y + line + 4 * scale, width
        else:
            ui.component_text(
                "dialog_body",
                Rect(origin.x, origin.y + (height - line) / 2, column, line),
                row.title,
                MUTED,
            )
            left, y, available = origin.x + column, origin.y, width - column
        cursor = left
        input_fields = []
        for hint, measured in zip(row.hints, widths, strict=True):
            if cursor > left and cursor + measured > left + available:
                cursor, y = left, y + height + 4 * scale
            if hint.kind == "text":
                hint = replace(hint, label=ui.translate(hint.label))
            painted = paint_hint(hint, cursor, y + height / 2)
            input_fields.append((cursor, y + (height - slot) / 2, painted, slot))
            cursor += measured + input_gap
        bottom = y + height
        if row.note:
            note = ui.translate(row.note)
            cursor = max(cursor, left + note_inset)
            remaining = left + available - cursor
            if remaining > 0 and draw.text_size(note)[0] <= remaining:
                draw.text(
                    (cursor, text_line_y(draw, y + height / 2)), theme.text, note, pixel_snap=False
                )
            else:
                imgui.set_cursor_screen_pos((left, bottom + 4 * scale))
                imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + available)
                tint = ui.style_color(style, "textColor", ui.ink(TEXT, style))
                imgui.push_style_color(imgui.Col_.text, tuple(value / 255 for value in tint))
                with draw.font_scope():
                    imgui.text_wrapped(note)
                imgui.pop_style_color()
                imgui.pop_text_wrap_pos()
                bottom = imgui.get_item_rect_max().y
        bottom = ceil(bottom)
        bounds = [origin.x, origin.y, width, bottom - origin.y]
        fields.append(bounds)
        if ui.design is not None:
            ui.design.record(
                f"guide-row-{index}",
                ("dialog_body",),
                bounds,
                "interaction_hint",
                label=ui.translate(row.title),
                fields=input_fields,
                stacked=stacked,
                inputSlot=slot,
                noteColumn=left + note_inset,
                centerY=origin.y + height / 2,
            )
        imgui.set_cursor_screen_pos((origin.x, origin.y))
        imgui.dummy((width, bottom - origin.y + gap))
    return fields
