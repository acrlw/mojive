"""Apply catalogue styles at explicit production control boundaries."""

from contextlib import contextmanager

from imgui_bundle import imgui

from mojive.design.groups import scoped_parts
from mojive.design.native.controls import NativePrimitives
from mojive.design.native.geometry import (
    CURRENT_NATIVE_PARTS,
    Rect,
    classify_button,
)
from mojive.ui.workspace.design import DESIGN

PANEL_FAMILIES = {
    "Inspector": "inspector",
    "Timeline": "timeline",
    "Settings": "settings",
    "Output": "output",
    "Pose": "pose",
    "Scene": "navigation",
    "Cameras": "navigation",
    "Viewport": "viewport",
    "Application": "chrome",
    "Navigation": "chrome",
    "application_status": "chrome",
}


class DesignPresentation:
    def __init__(self, ui, runtime):
        self.ui, self.runtime = ui, runtime
        self.primitives = NativePrimitives(ui, runtime)
        self._family_override = None
        self._window_families = {}
        ui.design_primitives = self.primitives

    def begin_frame(self):
        # Native parentage is stable during one frame. Never retain a native
        # window or infer its family again for every child control's style lookup.
        self._window_families.clear()
        self.primitives.scale = self.ui.s
        self.primitives.observed.clear()

    def style(self, parts):
        self.primitives.family = self.family()
        return self.primitives.style(parts, authored_defaults=True)

    def family(self):
        if self._family_override is not None:
            return self._family_override
        context = imgui.get_current_context()
        window = (
            context.current_window if context is not None and context.within_frame_scope else None
        )
        if window is None:
            return None
        window_id = window.id_
        if window_id in self._window_families:
            return self._window_families[window_id]
        while window is not None:
            name = window.name.split("###")[-1]
            if name in PANEL_FAMILIES:
                family = self._window_families[window_id] = PANEL_FAMILIES[name]
                return family
            window = window.parent_window
        self._window_families[window_id] = None
        return None

    @contextmanager
    def family_scope(self, family):
        """Retain a panel's style family when embedding it in another window."""
        previous = self._family_override
        self._family_override = family
        try:
            yield
        finally:
            self._family_override = previous

    def text_size(self, fallback):
        for part in reversed(CURRENT_NATIVE_PARTS.get()):
            if self.runtime.edited(part, "fontSize"):
                return self.runtime.resolved_part(part, "fontSize", self.family())
        return fallback

    def button_parts(self, key, label):
        chosen = classify_button(key, label)
        if str(key).startswith("menu-"):
            chosen = ("menu_trigger",)
        if str(key).startswith("filter-"):
            chosen = ("filter",)
        return (("button", *chosen)) if chosen != ("button",) and label else chosen

    def box(self, part, rect):
        return self.style(part).box(*rect, scale=self.ui.s)

    def record(self, key, parts, rect, kind, *, draw=None, **geometry):
        parts = scoped_parts(parts)
        if not self.runtime.observes(parts):
            return
        context = imgui.get_current_context()
        window = (
            context.current_window if context is not None and context.within_frame_scope else None
        )
        if window is None:
            return
        # Reading ImGui's implicit draw list outside a window activates its
        # fallback Debug window. Observation must not create a UI surface.
        current = window.draw_list
        target = current if draw is None else draw
        foreground = (
            draw is not None
            and target.vtx_buffer.data_address() != current.vtx_buffer.data_address()
        )
        clip_min, clip_max = target.get_clip_rect_min(), target.get_clip_rect_max()
        self.runtime.observed.append(
            {
                "key": key,
                "parts": tuple(parts),
                "hit": tuple(rect),
                "kind": kind,
                "styleFamily": self.family(),
                "clip": (clip_min.x, clip_min.y, clip_max.x, clip_max.y),
                "windowId": window.id_,
                "drawListSource": "foreground" if foreground else "window",
                **({"viewportId": imgui.get_main_viewport().id_} if foreground else {}),
                "submittedFrame": imgui.get_frame_count(),
                **geometry,
            }
        )

    def button(self, key, x, y, w, h, **options):
        self.primitives.family = self.family()
        parts = self.button_parts(key, options.get("label", ""))
        allowed = {
            key: value
            for key, value in options.items()
            if key
            in (
                "label",
                "icon",
                "active",
                "filled",
                "enabled",
                "tooltip",
                "text_color",
                "icon_color",
                "solid",
                "background",
                "translate",
                "icon_size",
                "icon_role",
                "align",
                "width_limit",
                "radius",
            )
        }
        engine = self.runtime.paint_context
        inset = (
            0
            if not options.get("label")
            else 8
            if options.get("icon") or "menu_trigger" in parts
            else 6
        )
        engine.paint_depth += 1
        try:
            return self.primitives.button(
                key,
                Rect(x, y, w * self.ui.s, h * self.ui.s),
                parts=parts,
                authored_defaults=True,
                padding=(
                    inset * self.ui.s,
                    0,
                    inset * self.ui.s,
                    0,
                ),
                **allowed,
            ).changed
        finally:
            engine.paint_depth -= 1

    def number(self, key, x, y, width, value, **options):
        self.primitives.family = self.family()
        parts = ("input_number", "axis_input") if options.get("axis") else ("input_number",)
        style = self.style(parts)
        margin = style.edges("margin", self.ui.s)
        height = self.primitives._field_height(style) * self.ui.s + margin[1] + margin[3]
        return self.primitives.input_number(
            key, Rect(x, y, width, height), value, parts=parts, **options
        )

    def switch(self, key, x, y, value, *, width=None):
        style = self.style("switch")
        natural_width = (style.get("switchWidth", DESIGN.switch_track_width) + 4) * self.ui.s
        padding = style.edges("padding", self.ui.s)
        natural_width += padding[0] + padding[2]
        margin = style.edges("margin", self.ui.s)
        outer_width = natural_width + margin[0] + margin[2]
        if width is not None:
            outer_width = min(outer_width, max(0, width))
        height = self.primitives._field_height(style) * self.ui.s + margin[1] + margin[3]
        box = self.primitives.field_box(style, Rect(x, y, outer_width, height), (0, 0, 0, 0))
        return self.primitives.switch(
            key,
            box.frame,
            value,
            layout_box=box,
            control_x=box.content.right
            - style.get("switchWidth", DESIGN.switch_track_width) * self.ui.s,
        ).value
