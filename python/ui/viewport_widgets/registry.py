"""Viewport widgets: registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .model import (
    PLAYBACK_CONTROLS,
    TOOL_GROUPS,
    ToolHint,
    ViewportControl,
)


class ToolHintRegistry:
    """User-extensible tool hints for the status bar and viewport scene.

    Defaults are supplied by the viewer each frame because they depend on the
    current interaction mode.  Callers can add custom hints or suppress a
    default by its stable ``hint_id`` without coupling to either renderer.
    """

    _SURFACES = frozenset(("status", "scene"))

    def __init__(self) -> None:
        self._custom: dict[str, dict[str, ToolHint]] = {surface: {} for surface in self._SURFACES}
        self._hidden_defaults: dict[str, set[str]] = {surface: set() for surface in self._SURFACES}

    def add(self, hint_id: str, hint: ToolHint, *, surface: str = "status") -> None:
        """Add or replace one custom hint on ``surface``."""

        target = self._surface(surface)
        key = str(hint_id).strip()
        if not key:
            raise ValueError("tool hint id must not be empty")
        self._custom[target][key] = ToolHint(
            hint.kind,
            hint.control,
            hint.label,
            hint.suffix,
            hint.hint_id or key,
            hint.modifier,
        )
        self._hidden_defaults[target].discard(key)

    def remove(self, hint_id: str, *, surface: str = "status") -> None:
        """Remove a custom hint and suppress a default with the same id."""

        target = self._surface(surface)
        key = str(hint_id).strip()
        self._custom[target].pop(key, None)
        if key:
            self._hidden_defaults[target].add(key)

    def restore(self, hint_id: str, *, surface: str = "status") -> None:
        """Allow a previously suppressed default hint to render again."""

        self._hidden_defaults[self._surface(surface)].discard(str(hint_id).strip())

    def resolve(
        self,
        defaults: Sequence[ToolHint] = (),
        *,
        surface: str = "status",
    ) -> tuple[ToolHint, ...]:
        """Compose visible defaults followed by caller-defined hints."""

        target = self._surface(surface)
        hidden = self._hidden_defaults[target]
        visible = tuple(hint for hint in defaults if not hint.hint_id or hint.hint_id not in hidden)
        return visible + tuple(self._custom[target].values())

    @classmethod
    def _surface(cls, value: str) -> str:
        surface = str(value).strip().lower()
        if surface not in cls._SURFACES:
            raise ValueError(f"unknown tool hint surface: {value!r}")
        return surface


class ViewportChromeRegistry:
    """Mutable extension seam for viewport actions and reusable tool hints."""

    def __init__(self) -> None:
        self.playback_controls: list[ViewportControl] = list(PLAYBACK_CONTROLS)
        self.tool_groups: list[list[ViewportControl]] = [list(group) for group in TOOL_GROUPS]
        self.tool_hints = ToolHintRegistry()
        self._handlers: dict[tuple[str, str], Callable[[], None]] = {}

    def add_playback(
        self,
        control: ViewportControl,
        handler: Callable[[], None],
        *,
        index: int | None = None,
    ) -> None:
        """Register a playback action without changing the viewer draw loop."""

        self._require_custom_icon(control)
        self.playback_controls[:] = [
            item for item in self.playback_controls if item.name != control.name
        ]
        target = len(self.playback_controls) if index is None else int(index)
        self.playback_controls.insert(max(0, min(target, len(self.playback_controls))), control)
        self._handlers[("playback", control.name)] = handler

    def add_tool(
        self,
        control: ViewportControl,
        handler: Callable[[], None],
        *,
        group: int | None = None,
    ) -> None:
        """Register a Tool Column action in an existing or new group."""

        self._require_custom_icon(control)
        self.tool_groups[:] = [
            [item for item in items if item.name != control.name] for items in self.tool_groups
        ]
        self.tool_groups[:] = [items for items in self.tool_groups if items]
        if group is None or group >= len(self.tool_groups):
            self.tool_groups.append([control])
        else:
            self.tool_groups[max(0, int(group))].append(control)
        self._handlers[("tool", control.name)] = handler

    def remove(self, surface: str, name: str) -> None:
        """Remove a registered action while preserving built-in dispatch code."""

        area = str(surface).strip().lower()
        key = str(name)
        if area == "playback":
            self.playback_controls[:] = [
                item for item in self.playback_controls if item.name != key
            ]
        elif area == "tool":
            self.tool_groups[:] = [
                [item for item in group if item.name != key] for group in self.tool_groups
            ]
            self.tool_groups[:] = [group for group in self.tool_groups if group]
        else:
            raise ValueError(f"unknown viewport chrome surface: {surface!r}")
        self._handlers.pop((area, key), None)

    def dispatch(self, surface: str, name: str) -> bool:
        handler = self._handlers.get((str(surface).strip().lower(), str(name)))
        if handler is None:
            return False
        handler()
        return True

    @staticmethod
    def _require_custom_icon(control: ViewportControl) -> None:
        if not control.name:
            raise ValueError("viewport control name must not be empty")
        if control.icon is None:
            raise ValueError("custom viewport controls require an icon callback")
