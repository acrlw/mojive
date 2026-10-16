"""Reveal scrollbars for scrolling or track hover without changing layout or input."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

from imgui_bundle import imgui

DEFAULT_HIDE_DELAY = 1.0
MIN_HIDE_DELAY = 0.1
MAX_HIDE_DELAY = 10.0


def normalize_hide_delay(value: object) -> float:
    """Resolve saved or edited seconds to the supported finite interval."""
    try:
        delay = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_HIDE_DELAY
    if isinstance(value, bool) or not math.isfinite(delay):
        return DEFAULT_HIDE_DELAY
    return min(MAX_HIDE_DELAY, max(MIN_HIDE_DELAY, delay))


@dataclass
class ScrollbarVisibility:
    idle_seconds: float = DEFAULT_HIDE_DELAY
    last_scroll: float | None = None
    last_hover: float | None = None
    visible: bool = False

    def opacity(self, now: float, *, scrolling: bool = False, hovered: bool = False) -> float:
        if scrolling:
            self.last_scroll = now
        if hovered:
            self.last_hover = now
        if scrolling or hovered:
            self.visible = True
        else:
            last_input = self.last_scroll
            if self.last_hover is not None:
                last_input = (
                    self.last_hover if last_input is None else max(last_input, self.last_hover)
                )
            if last_input is None or now - last_input >= self.idle_seconds:
                self.visible = False
        return float(self.visible)


_COLORS = (
    imgui.Col_.scrollbar_bg,
    imgui.Col_.scrollbar_grab,
    imgui.Col_.scrollbar_grab_hovered,
    imgui.Col_.scrollbar_grab_active,
)
_AXES = ((imgui.internal.Axis.x, "scrollbar_x"), (imgui.internal.Axis.y, "scrollbar_y"))


class NativeScrollbarVisibility:
    """Each registered panel owns the visibility of its children and popups."""

    def __init__(self, *, idle_seconds=DEFAULT_HIDE_DELAY, clock=None):
        self.idle_seconds = normalize_hide_delay(idle_seconds)
        self.clock = clock or time.monotonic
        self.states: dict[str, ScrollbarVisibility] = {}
        self.alphas: dict[str, float] = {}
        self._window_owners: dict[int, str] = {}
        self._panel_windows: set[int] = set()
        self._resolved: dict[int, str | None] = {}
        self._scrolling_owners: set[str] = set()
        self._query_seen: set[int] = set()
        self._query_windows = []
        self._owner_stack: list[str] = []
        self._visible: dict[str, bool] = {}
        self._now = self.clock()
        self._base = None

    def register(self, owner: str):
        """New owners remain hidden until scrolling or hovering their track."""
        state = self.states.get(owner)
        if state is None:
            state = self.states[owner] = ScrollbarVisibility(self.idle_seconds)
            self.alphas[owner] = state.opacity(self._now)
        return state

    def set_idle_seconds(self, value: float) -> None:
        """Apply one preference without resetting owners' scroll/hover timestamps."""
        self.idle_seconds = normalize_hide_delay(value)
        for owner, state in self.states.items():
            state.idle_seconds = self.idle_seconds
            self.alphas[owner] = state.opacity(self._now)

    def bind_window(self, owner, window, *, panel=True):
        """Bind the actual panel window, never the shared dock-space host."""
        self.register(owner)
        self._window_owners[window.id_] = owner
        if panel:
            self._panel_windows.add(window.id_)
        self._resolved.pop(window.id_, None)

    def notify_scroll(self, owner, *, now=None):
        """Record a scrolling gesture for its actual target panel."""
        state = self.register(owner)
        self.alphas[owner] = state.opacity(self.clock() if now is None else now, scrolling=True)

    def set_visible(self, owner, visible):
        if not visible and self._visible.get(owner, False):
            state = self.states.get(owner)
            if state is not None:
                state.last_scroll = None
                state.last_hover = None
                state.visible = False
                self.alphas[owner] = 0.0
        self._visible[owner] = visible

    def sync_visible(self, panels):
        for owner, visible in panels.items():
            self.set_visible(owner, visible)

    def bind_current_window(self):
        if self._owner_stack:
            window = imgui.get_current_context().current_window
            if window is not None:
                self.bind_window(self._owner_stack[-1], window)

    def owner_for_window(self, window):
        """Follow Begin ownership; dock roots can contain several independent panels."""
        if window is None:
            return None
        if window.id_ in self._resolved:
            return self._resolved[window.id_]
        seen, pending = self._query_seen, self._query_windows
        seen.clear()
        pending.clear()
        pending.append(window)
        owner, fallback = None, None
        while pending:
            candidate = pending.pop()
            if candidate.id_ in seen:
                continue
            seen.add(candidate.id_)
            registered = self._window_owners.get(candidate.id_)
            if registered is not None and candidate.id_ in self._panel_windows:
                owner = registered
                break
            if fallback is None:
                fallback = registered
            for field in (
                "parent_window",
                "parent_window_in_begin_stack",
                "parent_window_for_focus_route",
            ):
                parent = getattr(candidate, field, None)
                if parent is not None:
                    pending.append(parent)
        owner = owner or fallback
        self._resolved[window.id_] = owner
        return owner

    def owner_alpha(self, owner):
        return self.alphas.get(owner, 0.0)

    def update(self, *, now=None):
        """Wheel input, native scrollbar capture and track hover reveal only their owner."""
        self._now = self.clock() if now is None else now
        self._resolved.clear()
        io = imgui.get_io()
        context = imgui.get_current_context()
        active = self._scrolling_owners
        active.clear()
        window = context.active_id_window
        active_button = context.active_id_mouse_button
        scrollbar_drag = (
            window is not None
            and context.active_id != 0
            and active_button >= 0
            and imgui.is_mouse_down(active_button)
            and not imgui.is_mouse_clicked(active_button)
            and (
                (
                    window.scrollbar_x
                    and context.active_id
                    == imgui.internal.get_window_scrollbar_id(window, imgui.internal.Axis.x)
                )
                or (
                    window.scrollbar_y
                    and context.active_id
                    == imgui.internal.get_window_scrollbar_id(window, imgui.internal.Axis.y)
                )
            )
        )
        if scrollbar_drag:
            owner = self.owner_for_window(window)
            if owner is not None:
                active.add(owner)
        if io.mouse_wheel != 0 or io.mouse_wheel_h != 0:
            owner = self.owner_for_window(context.wheeling_window or context.hovered_window)
            if owner is not None:
                active.add(owner)
        hovered_owner = None
        hovered_window = context.hovered_window
        if hovered_window is not None and not imgui.is_any_mouse_down():
            for axis, flag in _AXES:
                if not getattr(hovered_window, flag):
                    continue
                if (
                    hovered_window.scroll_max.x
                    if axis == imgui.internal.Axis.x
                    else hovered_window.scroll_max.y
                ) <= 0:
                    continue
                track = imgui.internal.get_window_scrollbar_rect(hovered_window, axis)
                if (
                    track.min.x <= io.mouse_pos.x < track.max.x
                    and track.min.y <= io.mouse_pos.y < track.max.y
                ):
                    hovered_owner = self.owner_for_window(hovered_window)
                    break
        for owner, state in self.states.items():
            self.alphas[owner] = state.opacity(
                self._now, scrolling=owner in active, hovered=owner == hovered_owner
            )

    @contextmanager
    def panel_scope(self, owner, *, window_name=None):
        """Push colors before Begin so native background/grab paint uses this owner."""
        self.register(owner)
        if window_name is not None:
            window = imgui.internal.find_window_by_name(window_name)
            if window is not None:
                self.bind_window(owner, window)
        base = self._base or tuple(tuple(imgui.get_style().color_(name)) for name in _COLORS)
        context = imgui.get_current_context()
        windows_before = len(context.windows)
        for name, color in zip(_COLORS, base, strict=True):
            imgui.push_style_color(name, (*color[:3], color[3] * self.owner_alpha(owner)))
        self._owner_stack.append(owner)
        try:
            yield
        finally:
            # Independent popup/overlay roots can have no parent_window. Their
            # creation still occurs inside this explicit panel Begin scope.
            for index in range(windows_before, len(context.windows)):
                window = context.windows[index]
                if window.id_ not in self._window_owners:
                    self.bind_window(owner, window, panel=False)
            self._owner_stack.pop()
            imgui.pop_style_color(len(_COLORS))

    @contextmanager
    def scope(self):
        """Capture current theme colors, then restore them after this frame."""
        style = imgui.get_style()
        previous = self._base
        self._base = tuple(tuple(style.color_(name)) for name in _COLORS)
        try:
            yield
        finally:
            for name, color in zip(_COLORS, self._base, strict=True):
                style.set_color_(name, color)
            self._base = previous
