"""Declarative passive input and bounded, caller-acknowledged action requests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from queue import Full
from typing import TYPE_CHECKING

from mojive.interaction.input import InputClaim, InputContext, is_modifier_key, normalize_key

if TYPE_CHECKING:
    from mojive.ui.viewport_widgets import ToolHint


@dataclass(frozen=True)
class PassiveAction:
    """One focused key or optional transport button handled by the physics caller.

    ``key`` uses InputContext key names, such as ``8`` or ``space`` (no chords).
    ``control`` can route ``toggle``, ``reset``, ``step`` or ``previous`` from the
    playback toolbar. No clock capability is granted to the display adapter.
    Labels are supplied by the application in its desired language.
    """

    name: str
    key: str = ""
    label: str = ""
    control: str = ""

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("action name must not be empty")
        if self.control not in ("", "toggle", "reset", "step", "previous"):
            raise ValueError("unsupported passive playback control")
        if not self.key and not self.control:
            raise ValueError("an action needs a key or a playback control")
        if self.key:
            object.__setattr__(self, "key", normalize_key(self.key))
            if is_modifier_key(self.key):
                raise ValueError("a passive action needs a non-modifier key")


@dataclass(frozen=True)
class PassiveEvent:
    """An action request, not a notification that the simulation already changed."""

    sequence: int
    action: str


class PassiveInput:
    """Display-process input owner; callbacks only enqueue requests, never run policy."""

    def __init__(self, viewer, events):
        self.viewer = viewer
        self.events = events
        self.actions: tuple[PassiveAction, ...] = ()
        self.pending: dict[int, PassiveEvent] = {}
        self.sequence = 0
        self.keys = frozenset()
        self.claim = None
        self.hint_surface = "status"

    def configure(
        self,
        actions: Sequence[PassiveAction],
        *,
        hints: Sequence[ToolHint] | None = None,
        hint_surface: str = "status",
    ) -> None:
        from mojive.interaction.input import _imgui_keys
        from mojive.ui.viewport_widgets import (
            ToolHint,
            ViewportControl,
            _play_icon,
            _previous_icon,
            _reset_icon,
            _step_icon,
        )
        from mojive.ui.viewport_widgets.model import PLAYBACK_CONTROLS

        actions = tuple(actions)
        if self.pending:
            raise RuntimeError("Acknowledge pending actions before replacing bindings")
        if len(actions) > 64 or any(not isinstance(action, PassiveAction) for action in actions):
            raise ValueError("Expected at most 64 PassiveAction values")
        for attribute in ("name", "key", "control"):
            values = [
                getattr(action, attribute) for action in actions if getattr(action, attribute)
            ]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate passive action {attribute}")
        for action in actions:
            if action.key:
                _imgui_keys(action.key)
        if hint_surface not in ("status", "scene"):
            raise ValueError("hint_surface must be 'status' or 'scene'")
        if hints is None:
            hints = tuple(
                ToolHint(
                    "key",
                    action.key.removeprefix("digit_").title(),
                    action.label or action.name,
                    hint_id=f"passive.{action.name}",
                )
                for action in actions
                if action.key
            )
        else:
            hints = tuple(hints)
            if len(hints) > 64 or any(not isinstance(hint, ToolHint) for hint in hints):
                raise ValueError("Expected at most 64 ToolHint values")
        app = self.viewer.app
        registry = app.viewport_chrome
        self.configure_hints(hints, surface=hint_surface)
        for action in self.actions:
            if action.control:
                registry.remove("playback", action.control)
        # These four controls are owned by this bridge for the passive window.
        registry.playback_controls[:] = list(PLAYBACK_CONTROLS)
        self.actions = actions
        self.keys = frozenset(action.key for action in actions if action.key)
        self.claim = InputClaim(keys=self.keys) if self.keys else None
        glyphs = {
            "toggle": _play_icon,
            "previous": _previous_icon,
            "reset": _reset_icon,
            "step": _step_icon,
        }
        for action in actions:
            if action.control:
                index = next(
                    i
                    for i, item in enumerate(registry.playback_controls)
                    if item.name == action.control
                )
                registry.add_playback(
                    ViewportControl(
                        action.control,
                        icon=glyphs[action.control],
                        tooltip=action.label or action.name,
                    ),
                    self._handler(action.name),
                    index=index,
                )
        self._refresh_controls()

    def configure_hints(self, hints: Sequence[ToolHint], *, surface: str = "status") -> None:
        """Update presentation without changing actions, pending requests or controls."""
        self.viewer.app.viewport_chrome.tool_hints.configure(hints, surface=surface)
        self.hint_surface = surface

    def _handler(self, name: str) -> Callable[[], None]:
        return lambda: self.request(name)

    def _refresh_controls(self):
        pending = {event.action for event in self.pending.values()}
        app = self.viewer.app
        app.external_playback_actions = frozenset(
            action.control
            for action in self.actions
            if action.control and action.name not in pending
        )
        labels = {
            action.control: (
                f"{action.label or action.name}: {app.localizer.text('Waiting for the external application')}"
                if action.name in pending
                else action.label or action.name
            )
            for action in self.actions
            if action.control
        }
        app.viewport_chrome.playback_controls[:] = [
            replace(control, tooltip=labels[control.name]) if control.name in labels else control
            for control in app.viewport_chrome.playback_controls
        ]

    def request(self, name: str) -> None:
        if any(event.action == name for event in self.pending.values()):
            return
        event = PassiveEvent(self.sequence + 1, name)
        try:
            self.events.put_nowait(event)
        except Full:
            self.viewer.session.report_message(
                self.viewer.app.localizer.text("External action queue is full"), level="warning"
            )
            return
        self.sequence = event.sequence
        self.pending[event.sequence] = event
        self._refresh_controls()

    def acknowledge(self, event: PassiveEvent, *, error: str | None = None) -> None:
        if self.pending.get(event.sequence) != event:
            raise ValueError("Unknown or already acknowledged passive event")
        del self.pending[event.sequence]
        self._refresh_controls()
        if error:
            self.viewer.session.report_message(error, level="error")

    def __call__(self, context: InputContext) -> InputClaim | None:
        from imgui_bundle import imgui

        if not self.keys:
            return None
        if context.blocked or not context.viewport_focused or imgui.is_any_item_active():
            return None
        if any(context.key_down(key) for key in ("ctrl", "super", "alt", "shift")):
            return None
        for action in self.actions:
            if action.key and context.key_pressed(action.key):
                self.request(action.name)
        return self.claim
