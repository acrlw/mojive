"""Central viewport key bindings and user-facing shortcut labels.

Viewport code asks this map for actions instead of reading ImGui keys directly.
Settings replaces the immutable map atomically, so drawn hints, tooltips, and
interaction polling cannot drift apart during an input frame.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from imgui_bundle import imgui


class InputAction(enum.StrEnum):
    TOGGLE_PAUSE = "toggle_pause"
    STEP_BACK = "step_back"
    FRAME_SCENE = "frame_scene"
    GIZMO_TRANSLATE = "gizmo_translate"
    GIZMO_ROTATE = "gizmo_rotate"
    GIZMO_DIMENSIONS = "gizmo_dimensions"
    GIZMO_SPACE = "gizmo_space"
    SNAP = "snap"
    PERTURB = "perturb"
    AXIS_X = "axis_x"
    AXIS_Y = "axis_y"
    AXIS_Z = "axis_z"
    FLY_FORWARD = "fly_forward"
    FLY_BACK = "fly_back"
    FLY_RIGHT = "fly_right"
    FLY_LEFT = "fly_left"
    FLY_UP = "fly_up"
    FLY_DOWN = "fly_down"


@dataclass(frozen=True)
class KeyBinding:
    key: object | None
    label: str
    identifier: str | None = None


@dataclass(frozen=True)
class KeyChoice:
    identifier: str | None
    key: object | None
    label: str


@dataclass(frozen=True)
class InputBindings:
    """One replaceable binding set shared by polling, hints, and tooltips."""

    entries: tuple[tuple[InputAction, KeyBinding], ...]

    def binding(self, action: InputAction) -> KeyBinding:
        return next(binding for candidate, binding in self.entries if candidate is action)

    def label(self, action: InputAction) -> str:
        return self.binding(action).label

    def key_id(self, action: InputAction) -> str | None:
        binding = self.binding(action)
        if binding.key is None:
            return None
        return binding.identifier or _choice_for_key(binding.key).identifier

    def down(self, action: InputAction) -> bool:
        key = self.binding(action).key
        if key is None:
            return False
        keys = key if isinstance(key, tuple) else (key,)
        return any(imgui.is_key_down(candidate) for candidate in keys)

    def pressed(self, action: InputAction) -> bool:
        key = self.binding(action).key
        if key is None:
            return False
        keys = key if isinstance(key, tuple) else (key,)
        return any(imgui.is_key_pressed(candidate, False) for candidate in keys)

    def press_count(self, action: InputAction, *, delay: float, rate: float) -> int:
        """Return immediate and held-repeat presses at an action-specific cadence."""

        key = self.binding(action).key
        if key is None:
            return 0
        keys = key if isinstance(key, tuple) else (key,)
        return max(
            (imgui.get_key_pressed_amount(candidate, delay, rate) for candidate in keys),
            default=0,
        )

    def with_binding(self, action: InputAction, key: object | None, label: str) -> InputBindings:
        """Return a changed map without mutating an in-flight input frame."""

        choice = _choice_for_key(key)
        replacement = KeyBinding(key, str(label), choice.identifier)
        return replace(
            self,
            entries=tuple(
                (candidate, replacement if candidate is action else binding)
                for candidate, binding in self.entries
            ),
        )

    def remap(self, action: InputAction, key_id: str | None) -> InputBindings:
        """Bind one action and swap the displaced action to the previous key.

        A viewport key always belongs to exactly one action. Swapping instead
        of silently accepting a duplicate prevents one press from activating
        two tools and keeps remapping reversible.
        """

        choice = key_choice(key_id)
        previous = self.binding(action)
        displaced = next(
            (
                candidate
                for candidate, binding in self.entries
                if choice.identifier is not None
                and candidate is not action
                and self.key_id(candidate) == choice.identifier
            ),
            None,
        )
        entries = []
        for candidate, binding in self.entries:
            if candidate is action:
                entries.append((candidate, KeyBinding(choice.key, choice.label, choice.identifier)))
            elif candidate is displaced:
                entries.append((candidate, previous))
            else:
                entries.append((candidate, binding))
        return replace(self, entries=tuple(entries))

    def preferences(self) -> dict[str, str | None]:
        """Return the stable JSON representation used by editor settings."""

        return {action.value: self.key_id(action) for action, _binding in self.entries}

    @classmethod
    def from_preferences(cls, value: object) -> InputBindings:
        """Restore valid saved bindings while ignoring malformed entries."""

        if not isinstance(value, dict):
            return DEFAULT_INPUT_BINDINGS
        bindings = DEFAULT_INPUT_BINDINGS
        for action in InputAction:
            if action.value not in value:
                continue
            key_id = value.get(action.value)
            if key_id is not None and not isinstance(key_id, str):
                continue
            if key_id not in _KEY_CHOICES_BY_ID:
                continue
            bindings = bindings.remap(action, key_id)
        return bindings


def input_action_name(action: InputAction) -> str:
    return _ACTION_NAMES[action]


def key_choices() -> tuple[KeyChoice, ...]:
    return _KEY_CHOICES


def key_choice(identifier: str | None) -> KeyChoice:
    try:
        return _KEY_CHOICES_BY_ID[identifier]
    except KeyError as error:
        raise ValueError(f"Unsupported viewport key: {identifier}") from error


def _choice_for_key(key: object | None) -> KeyChoice:
    for choice in _KEY_CHOICES:
        if choice.key == key:
            return choice
    raise ValueError(f"Unsupported viewport key object: {key!r}")


_ACTION_NAMES = {
    InputAction.TOGGLE_PAUSE: "Play / Pause",
    InputAction.STEP_BACK: "Previous frame",
    InputAction.FRAME_SCENE: "Frame selection",
    InputAction.GIZMO_TRANSLATE: "Move tool",
    InputAction.GIZMO_ROTATE: "Rotate tool",
    InputAction.GIZMO_DIMENSIONS: "Dimensions tool",
    InputAction.GIZMO_SPACE: "World / Body",
    InputAction.SNAP: "Snap",
    InputAction.PERTURB: "Perturb",
    InputAction.AXIS_X: "Constrain X",
    InputAction.AXIS_Y: "Constrain Y",
    InputAction.AXIS_Z: "Constrain Z",
    InputAction.FLY_FORWARD: "Fly forward",
    InputAction.FLY_BACK: "Fly backward",
    InputAction.FLY_RIGHT: "Fly right",
    InputAction.FLY_LEFT: "Fly left",
    InputAction.FLY_UP: "Fly up",
    InputAction.FLY_DOWN: "Fly down",
}


_KEY_CHOICES = (
    KeyChoice(None, None, "Unbound"),
    KeyChoice("space", imgui.Key.space, "Space"),
    KeyChoice("backspace", imgui.Key.backspace, "Backspace"),
    KeyChoice(
        "shift",
        (imgui.Key.left_shift, imgui.Key.right_shift),
        "Shift",
    ),
    KeyChoice(
        "ctrl",
        (imgui.Key.left_ctrl, imgui.Key.right_ctrl),
        "Ctrl",
    ),
    *tuple(
        KeyChoice(letter.casefold(), getattr(imgui.Key, letter.casefold()), letter)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ),
    *tuple(
        KeyChoice(f"digit_{digit}", getattr(imgui.Key, f"_{digit}"), digit)
        for digit in "0123456789"
    ),
)
_KEY_CHOICES_BY_ID = {choice.identifier: choice for choice in _KEY_CHOICES}


DEFAULT_INPUT_BINDINGS = InputBindings(
    (
        (InputAction.TOGGLE_PAUSE, KeyBinding(imgui.Key.space, "Space", "space")),
        (InputAction.STEP_BACK, KeyBinding(imgui.Key.backspace, "Backspace", "backspace")),
        (InputAction.FRAME_SCENE, KeyBinding(imgui.Key.f, "F", "f")),
        (InputAction.GIZMO_TRANSLATE, KeyBinding(imgui.Key.g, "G", "g")),
        (InputAction.GIZMO_ROTATE, KeyBinding(imgui.Key.r, "R", "r")),
        (InputAction.GIZMO_DIMENSIONS, KeyBinding(None, "Unbound", None)),
        (InputAction.GIZMO_SPACE, KeyBinding(imgui.Key.t, "T", "t")),
        (
            InputAction.SNAP,
            KeyBinding((imgui.Key.left_shift, imgui.Key.right_shift), "Shift", "shift"),
        ),
        (
            InputAction.PERTURB,
            KeyBinding((imgui.Key.left_ctrl, imgui.Key.right_ctrl), "Ctrl", "ctrl"),
        ),
        (InputAction.AXIS_X, KeyBinding(imgui.Key.x, "X", "x")),
        (InputAction.AXIS_Y, KeyBinding(imgui.Key.y, "Y", "y")),
        (InputAction.AXIS_Z, KeyBinding(imgui.Key.z, "Z", "z")),
        (InputAction.FLY_FORWARD, KeyBinding(imgui.Key.w, "W", "w")),
        (InputAction.FLY_BACK, KeyBinding(imgui.Key.s, "S", "s")),
        (InputAction.FLY_RIGHT, KeyBinding(imgui.Key.d, "D", "d")),
        (InputAction.FLY_LEFT, KeyBinding(imgui.Key.a, "A", "a")),
        (InputAction.FLY_UP, KeyBinding(imgui.Key.q, "Q", "q")),
        (InputAction.FLY_DOWN, KeyBinding(imgui.Key.e, "E", "e")),
    )
)
