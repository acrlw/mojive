"""Send native workbench gestures through ImGui's input queue."""

from contextlib import contextmanager

from imgui_bundle import imgui

from mojive.design.native.geometry import NativePartStyle


class WorkbenchInput:
    def __init__(self, viewer, workbench):
        self.viewer = viewer
        self.workbench = workbench
        self._tracking_reads = workbench.runtime.track_field_reads
        workbench.runtime.track_field_reads = True
        self.events = []
        self.reads = set()
        self.before_editor = self.after_editor = None
        self.default_mismatches = {}
        self.style_get = NativePartStyle.get
        NativePartStyle.get = lambda style, key, fallback=None: self.observe_style_get(
            style, key, fallback
        )
        self.hover_part = None
        self.owner = viewer.window._input
        self.original = self.owner.process_inputs
        self.owner.process_inputs = self.process
        self.panels = viewer.app.panels
        self.panel_draw = self.panels.draw
        self.panels.draw = self.draw_panels
        self.scrollbars = getattr(viewer.app, "scrollbar_visibility", None)
        self.scrollbar_clock = self.scrollbars.clock if self.scrollbars is not None else None
        if self.scrollbars is not None:
            # Hold scrollbar paint constant during parameter comparisons.
            # Actual scroll-only visibility has separate native GPU tests.
            self.scrollbars.clock = lambda: 0.0
        panel = self.panels.get("stats")
        if panel is not None and hasattr(panel, "_frame_ms"):
            ring = panel._frame_ms
            ring.data[:] = 1000 / 60
            ring.filled = len(ring.data)

    def close(self):
        self.workbench.runtime.track_field_reads = self._tracking_reads
        NativePartStyle.get = self.style_get
        self.owner.process_inputs = self.original
        self.panels.draw = self.panel_draw
        if self.scrollbars is not None:
            self.scrollbars.clock = self.scrollbar_clock

    def observe_style_get(self, style, key, fallback=None):
        """Diagnose authored/default differences without changing returned paint values."""
        value = self.style_get(style, key, fallback)
        canonical = style.values.get(key)
        rule = style.catalog.get(style.origins.get(key), {}).get("fields", {}).get(key, {})
        condition = rule.get("condition")
        if (
            not style.authored_defaults
            or style.explicit(key)
            or canonical is None
            or value is None
            or (condition and style.values.get(condition["field"]) != condition["value"])
            or (key in ("rowGap", "columnGap") and style.explicit("gap"))
        ):
            return value
        if isinstance(canonical, (float, int)):
            try:
                observed = float(value)
            except (TypeError, ValueError):
                return value
        elif isinstance(canonical, str) and isinstance(value, str):
            observed = value
        else:
            return value
        if observed != canonical:
            identity = style.part_ids, key, repr(observed), repr(canonical)
            self.default_mismatches[identity] = {
                "parts": list(style.part_ids),
                "field": key,
                "resolved": canonical,
                "actualGetReturn": observed,
            }
        return value

    def draw_panels(self, context):
        """Keep telemetry payloads fixed while comparing native layout changes."""
        stats = context.backend.stats
        stats.frame_cpu_ms = 1.0
        for values in (stats.cpu_ms, stats.gpu_ms):
            for name in values:
                values[name] = 1.0
        self.panel_draw(context)

    def process(self):
        self.original()
        if self.scrollbars is not None:
            for owner in self.scrollbars.states:
                self.scrollbars.notify_scroll(owner, now=0.0)
        self.viewer.app._dt = 1 / 60
        self.viewer.app._frame_rate.value = 60.0
        io = imgui.get_io()
        for method, args in self.events:
            getattr(io, method)(*args)
        self.events.clear()
        io.delta_time = 1 / 60

    def frames(self, count=1):
        for _ in range(count):
            self.viewer.sync()
            for part, fields in getattr(self.workbench.runtime, "used_fields", {}).items():
                self.reads.update((part, field) for field in fields)

    def event(self, method, *args):
        self.events.append((method, args))
        self.frames()

    def batch(self, *events):
        self.events.extend(events)
        self.frames()

    def key(self, key):
        self.event("add_key_event", key, True)
        self.event("add_key_event", key, False)

    def locate(self, key):
        """Scroll the parameter child until the authored item is reachable."""
        for _ in range(4):
            rect = self.workbench.hits.get(key)
            if rect is None:
                self.frames(2)
                continue
            context = imgui.get_current_context()
            windows = [
                window
                for window in context.windows
                if (window.active or window.was_active) and "参数逐项列表" in window.name
            ]
            if not windows:
                return rect
            window = windows[-1]
            clip = window.clip_rect
            _x, y, _w, h = rect
            if clip.min.y <= y and y + h <= clip.max.y:
                return rect
            target = window.scroll.y + y - clip.min.y - 8
            imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, target)))
            self.frames(2)
        raise RuntimeError(f"Native editor item is not reachable: {key}")

    def click(self, key):
        x, y, w, h = self.locate(key)
        self.event("add_mouse_pos_event", x + w / 2, y + h / 2)
        # Hover can settle a scrolling child after a different field was reset.
        # Resolve the current item again before pressing its native hit target.
        x, y, w, h = self.locate(key)
        self.event("add_mouse_pos_event", x + w / 2, y + h / 2)
        self.event("add_mouse_button_event", 0, True)
        self.event("add_mouse_button_event", 0, False)

    def show_field(self, field):
        key = f"value/parts/{self.workbench.part}/{field}"
        if self.workbench.field_search == field and key in self.workbench.hits:
            return
        self.workbench.field_group = "全部参数"
        self.workbench.field_search = field
        self.frames(2)

    def set_field(self, part, field, rule, value):
        with self.editor_access():
            self._set_field(part, field, rule, value)

    @contextmanager
    def editor_access(self):
        if self.before_editor is not None:
            self.before_editor()
        try:
            yield
        finally:
            if self.after_editor is not None:
                self.after_editor()

    def _set_field(self, part, field, rule, value):
        self.show_field(field)
        key = f"value/parts/{part}/{field}"
        self.click(key)
        if rule["type"] in ("number", "color"):
            modifier = (
                imgui.Key.mod_super
                if imgui.get_io().config_mac_osx_behaviors
                else imgui.Key.mod_ctrl
            )
            self.batch(
                ("add_key_event", (modifier, True)),
                ("add_key_event", (imgui.Key.a, True)),
            )
            self.batch(
                ("add_key_event", (imgui.Key.a, False)),
                ("add_key_event", (modifier, False)),
            )
            self.event("add_input_characters_utf8", str(value))
            self.key(imgui.Key.enter)
        else:
            # Combo windows finish positioning on the frame after opening.
            self.frames()
            choices = rule["choices"]
            stack = imgui.get_current_context().open_popup_stack
            popup = stack[len(stack) - 1].window if stack else None
            if popup is None or not self.workbench._editor_window(popup):
                raise RuntimeError(f"Native editor combo did not open: {part}.{field}")
            start, end = popup.dc.cursor_start_pos, popup.dc.cursor_pos
            pitch = (end.y - start.y) / len(choices)
            if pitch <= 0:
                raise RuntimeError(f"Native editor combo has no submitted choices: {part}.{field}")
            self.event(
                "add_mouse_pos_event",
                (popup.work_rect.min.x + popup.work_rect.max.x) / 2,
                start.y + (choices.index(value) + 0.4) * pitch,
            )
            self.event("add_mouse_button_event", 0, True)
            self.event("add_mouse_button_event", 0, False)
        self.frames(2 if rule["type"] == "select" else 1)
        actual = self.workbench.runtime.resolved_part(part, field)
        if actual != value:
            raise RuntimeError(f"Native input did not commit {part}.{field}: {actual} != {value}")

    def restore(self, part, field):
        with self.editor_access():
            self.show_field(field)
            self.click(f"reset/parts/{part}/{field}")
            self.frames()

    def neutral(self):
        context = imgui.get_current_context()
        window = context.active_id_window
        if context.active_id and window is not None:
            while (
                window.flags & imgui.WindowFlags_.child_window and window.parent_window is not None
            ):
                window = window.parent_window
            # A native empty-header click ends text focus and caret animation;
            # it keeps the popup open and leaves the edited text unchanged.
            self.event("add_mouse_pos_event", window.pos.x + 4, window.pos.y + 4)
            self.event("add_mouse_button_event", 0, True)
            self.event("add_mouse_button_event", 0, False)
        rect = self.workbench.hits.get("part")
        if self.hover_part is not None:
            record = next(
                (
                    item
                    for item in self.workbench.native_instances
                    if self.hover_part in item["parts"]
                ),
                None,
            )
            if record is not None:
                rect = record["visibleHit"]
        if rect is not None:
            self.event("add_mouse_pos_event", rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        self.frames()
