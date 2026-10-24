"""Paint custom surfaces from the native item's current interaction state."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.geometry2d.curves import capsule_end_padding
from mojive.ui.design_palette import rgba8

_REFERENCE_PAINT = ContextVar("reference_item_paint", default=None)


@contextmanager
def reference_paint(state):
    """Render a state plate through the production painter without synthesizing input."""
    token = _REFERENCE_PAINT.set(state)
    try:
        yield
    finally:
        _REFERENCE_PAINT.reset(token)


@dataclass(frozen=True, slots=True)
class ItemPaint:
    hovered: bool = False
    active: bool = False
    disabled: bool = False

    @classmethod
    def capture(cls, imgui, *, opened=False):
        reference = _REFERENCE_PAINT.get()
        if reference is not None:
            return reference
        disabled = bool(
            imgui.get_current_context().last_item_data.item_flags & imgui.ItemFlags_.disabled.value
        )
        hovered = not disabled and imgui.is_item_hovered()
        active = not disabled and (opened or imgui.is_item_active())
        return _ITEM_PAINTS[int(hovered) | (int(active) << 1) | (int(disabled) << 2)]

    def color(self, idle="field", hover="field_hovered", active="field_active"):
        return rgba8(active if self.active else hover if self.hovered else idle)

    def combine(self, other):
        return ItemPaint(
            self.hovered or other.hovered,
            self.active or other.active,
            self.disabled and other.disabled,
        )


# Interaction feedback has eight immutable combinations. Keep no window, ID,
# draw list or gesture state in these shared values.
_ITEM_PAINTS = tuple(ItemPaint(bool(i & 1), bool(i & 2), bool(i & 4)) for i in range(8))


class FieldLayer:
    """Keep native text/caret above a background chosen after item submission.

    A separate splitter composes with table channels and nested compound fields.
    Retain the parent draw list when BeginCombo enters its popup window.
    """

    def __init__(self, imgui, draw, scratch=None):
        self.draw = draw
        self.scratch = scratch
        self.splitter = scratch.pop() if scratch else imgui.ImDrawListSplitter()

    def __enter__(self):
        self.splitter.split(self.draw, 2)
        self.splitter.set_current_channel(self.draw, 1)
        return self

    def background(self):
        self.splitter.set_current_channel(self.draw, 0)

    def __exit__(self, *exc):
        try:
            self.splitter.merge(self.draw)
            # Each active layer borrows its own splitter, including nested fields
            # and combo popups. Keep native buffer capacity, never a frame's draw list.
            if self.scratch is not None and len(self.scratch) < 8:
                self.scratch.append(self.splitter)
        finally:
            self.draw = self.splitter = None


def _mix_ink(first, second, amount):
    return (
        *(round(a + (b - a) * amount) for a, b in zip(first[:3], second[:3], strict=True)),
        first[3],
    )


def draw_switch(ui, x, y, width, height, thumb, value, paint):
    """Paint one switch track and thumb for production and reference controls."""
    if width <= 0 or height <= 0:
        return
    thumb = min(thumb, width, height)
    field, accent, text, secondary, disabled = (
        ui.ink(rgba8(role)) for role in ("field", "primary", "text", "secondary", "disabled")
    )
    held = paint.active and not paint.disabled
    hovered = paint.hovered and not paint.disabled
    if paint.disabled:
        track = _mix_ink(accent, field, 0.76) if value else _mix_ink(field, text, 0.03)
        tint = disabled
    elif value:
        track = _mix_ink(accent, field, 0.42)
        track = (
            _mix_ink(track, field, 0.22)
            if held
            else _mix_ink(track, text, 0.12)
            if hovered
            else track
        )
        tint = _mix_ink(text, (255, 255, 255, 255), 0.6) if hovered and not held else text
    else:
        track = _mix_ink(field, text, 0.3 if held else 0.18 if hovered else 0.06)
        tint = text if held or hovered else secondary
    ui.rect(
        x,
        y,
        width,
        height,
        track,
        r=height / ui.s / 2,
        color_resolved=True,
    )
    inset = max(0.0, (height - thumb) / 2)
    cx = x + width - inset - thumb / 2 if value else x + inset + thumb / 2
    if getattr(ui, "design", None) is not None:
        native = ui.design.runtime.spec["native"]
        if native["g3"] and native["g3_capsule_optical"]:
            padding = capsule_end_padding(width, height, native["g3_capsule_smoothing"])
            cx = x + width - padding if value else x + padding
    cx = min(max(cx, x + thumb / 2), x + width - thumb / 2)
    # Resolve palette roles before deriving feedback, and submit them once.
    # A neutral thumb retains its edge against an accent-colored track.
    ui.draw.add_circle_filled(
        (cx, y + height / 2),
        thumb * (0.46 if held else 0.5),
        imgui.get_color_u32(imgui.ImVec4(*(channel / 255 for channel in tint))),
        num_segments=32,
    )
