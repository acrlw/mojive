"""Paint custom surfaces from the native item's current interaction state."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from imgui_bundle import imgui

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
            imgui.get_current_context().last_item_data.item_flags & imgui.ItemFlags_.disabled
        )
        return cls(
            not disabled and imgui.is_item_hovered(),
            not disabled and (opened or imgui.is_item_active()),
            disabled,
        )

    def color(self, idle="field", hover="field_hovered", active="field_active"):
        return rgba8(active if self.active else hover if self.hovered else idle)

    def combine(self, other):
        return ItemPaint(
            self.hovered or other.hovered,
            self.active or other.active,
            self.disabled and other.disabled,
        )


class FieldLayer:
    """Keep native text/caret above a background chosen after item submission.

    A separate splitter composes with table channels and nested compound fields.
    Retain the parent draw list when BeginCombo enters its popup window.
    """

    def __init__(self, imgui, draw):
        self.draw = draw
        self.splitter = imgui.ImDrawListSplitter()

    def __enter__(self):
        self.splitter.split(self.draw, 2)
        self.splitter.set_current_channel(self.draw, 1)
        return self

    def background(self):
        self.splitter.set_current_channel(self.draw, 0)

    def __exit__(self, *exc):
        self.splitter.merge(self.draw)


def draw_switch(ui, x, y, width, height, thumb, value, paint):
    """Paint one switch track and thumb for production and reference controls."""
    if width <= 0 or height <= 0:
        return
    thumb = min(thumb, width, height)
    ui.rect(
        x,
        y,
        width,
        height,
        paint.color("primary_dim", "primary", "primary_bright")
        if value
        else paint.color("field_hovered", "field_active", "border"),
        r=height / ui.s / 2,
    )
    inset = max(0.0, (height - thumb) / 2)
    cx = x + width - inset - thumb / 2 if value else x + inset + thumb / 2
    if getattr(ui, "design", None) is not None:
        native = ui.design.runtime.spec["native"]
        if native["g3"] and native["g3_capsule_optical"]:
            from mojive.ui.capsule_geometry import capsule_end_padding

            padding = capsule_end_padding(width, height, native["g3_capsule_smoothing"])
            cx = x + width - padding if value else x + padding
    cx = min(max(cx, x + thumb / 2), x + width - thumb / 2)
    tint = (
        paint.color("primary", "primary_bright", "text")
        if value
        else paint.color("secondary", "text", "primary_bright")
    )
    ui.draw.add_circle_filled(
        (cx, y + height / 2),
        thumb / 2,
        imgui.get_color_u32(imgui.ImVec4(*(channel / 255 for channel in tint))),
        num_segments=32,
    )
