"""Public input hook for applications that embed the interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache


@lru_cache(maxsize=1)
def _macos_modifier_keys():
    from imgui_bundle import imgui

    key = imgui.Key
    pairs = (
        (key.left_ctrl, key.left_super),
        (key.right_ctrl, key.right_super),
        (key.mod_ctrl, key.mod_super),
    )
    return {a: b for pair in pairs for a, b in (pair, pair[::-1])}


def imgui_key_for_physical_key(key):
    """Map a physical key to ImGui's current platform-specific key representation."""
    swapped = _macos_modifier_keys().get(key)
    if swapped is not None:
        from imgui_bundle import imgui

        if imgui.get_io().config_mac_osx_behaviors:
            return swapped
    return key


def physical_ctrl_super(io) -> tuple[bool, bool]:
    """Return physical Control and Super states while retaining native text-editing behavior."""
    if io.config_mac_osx_behaviors:
        return bool(io.key_super), bool(io.key_ctrl)
    return bool(io.key_ctrl), bool(io.key_super)


def add_physical_mouse_button_event(io, button: int, down: bool) -> None:
    """Queue a physical button without ImGui's macOS Control-click alias.

    Viewer gestures bind physical Control and mouse buttons independently.
    Keep native macOS key and text editing behavior outside this event call.
    """
    macos = io.config_mac_osx_behaviors
    if button != 0 or not macos:
        io.add_mouse_button_event(button, down)
        return
    try:
        io.config_mac_osx_behaviors = False
        io.add_mouse_button_event(button, down)
    finally:
        io.config_mac_osx_behaviors = macos


def normalize_key(value: str) -> str:
    """Return the stable identifier used by input claims and key bindings."""

    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "control": "ctrl",
        "escape": "escape",
        "esc": "escape",
        "return": "enter",
        "option": "alt",
        "cmd": "super",
        "command": "super",
    }
    if len(key) == 1 and key.isdigit():
        return f"digit_{key}"
    return aliases.get(key, key)


def _imgui_keys(identifier: str) -> tuple[object, ...]:
    from imgui_bundle import imgui

    key_id = normalize_key(identifier)
    modifiers = {
        "ctrl": (imgui.Key.left_ctrl, imgui.Key.right_ctrl),
        "shift": (imgui.Key.left_shift, imgui.Key.right_shift),
        "alt": (imgui.Key.left_alt, imgui.Key.right_alt),
        "super": (imgui.Key.left_super, imgui.Key.right_super),
    }
    if key_id in modifiers:
        return tuple(imgui_key_for_physical_key(key) for key in modifiers[key_id])
    attribute = f"_{key_id[6:]}" if key_id.startswith("digit_") else key_id
    key = getattr(imgui.Key, attribute, None)
    if key is None:
        raise ValueError(f"Unsupported input key: {identifier}")
    return (key,)


@dataclass(frozen=True)
class InputClaim:
    """Input reserved by an embedding application for the current frame.

    Claimed input is still observable by the application callback, but Mojive's
    built-in camera, tools, selection, and editor or panel shortcuts do not
    consume it. Explicit menu clicks remain available.
    """

    keys: frozenset[str] = field(default_factory=frozenset)
    mouse_buttons: frozenset[int] = field(default_factory=frozenset)
    keyboard: bool = False
    pointer: bool = False
    wheel: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", frozenset(normalize_key(key) for key in self.keys))
        object.__setattr__(
            self,
            "mouse_buttons",
            frozenset(int(button) for button in self.mouse_buttons),
        )

    def claims_key(self, key: str | None) -> bool:
        return bool(self.keyboard or (key is not None and normalize_key(key) in self.keys))

    def claims_button(self, button: int) -> bool:
        return bool(self.pointer or int(button) in self.mouse_buttons)


@dataclass(frozen=True)
class InputContext:
    """Read-only view of the current UI input frame passed to an input handler."""

    viewport_hovered: bool
    viewport_focused: bool
    blocked: bool
    cursor: tuple[float, float]
    delta: tuple[float, float]
    wheel: float

    def key_down(self, key: str) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and any(imgui.is_key_down(value) for value in _imgui_keys(key))

    def key_pressed(self, key: str, *, repeat: bool = False) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and any(
            imgui.is_key_pressed(value, repeat) for value in _imgui_keys(key)
        )

    def key_released(self, key: str) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and any(imgui.is_key_released(value) for value in _imgui_keys(key))

    def mouse_down(self, button: int) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and imgui.is_mouse_down(int(button))

    def mouse_clicked(self, button: int) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and imgui.is_mouse_clicked(int(button))

    def mouse_released(self, button: int) -> bool:
        from imgui_bundle import imgui

        return not self.blocked and imgui.is_mouse_released(int(button))


__all__ = ["InputClaim", "InputContext", "normalize_key"]
