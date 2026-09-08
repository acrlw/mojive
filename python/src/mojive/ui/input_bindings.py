"""Central viewport key bindings and user-facing shortcut labels.

Viewport code asks this map for actions instead of reading ImGui keys directly.
Settings replaces the immutable map atomically, so drawn hints, tooltips, and
interaction polling cannot drift apart during an input frame.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

from imgui_bundle import imgui

from ..input import InputClaim, _imgui_keys, imgui_key_for_physical_key, physical_ctrl_super
from .pointer_bindings import (
    DEFAULT_POINTER_BINDINGS,
    NAVIGATION_PRESETS,
    PointerAction,
    PointerChord,
    PointerFrame,
    exclusive_group,
)


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
    pointers: tuple[tuple[PointerAction, tuple[PointerChord, ...]], ...] = DEFAULT_POINTER_BINDINGS

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
        return any(imgui.is_key_down(imgui_key_for_physical_key(candidate)) for candidate in keys)

    def pressed(self, action: InputAction) -> bool:
        key = self.binding(action).key
        if key is None:
            return False
        keys = key if isinstance(key, tuple) else (key,)
        return any(
            imgui.is_key_pressed(imgui_key_for_physical_key(candidate), False) for candidate in keys
        )

    def press_count(self, action: InputAction, *, delay: float, rate: float) -> int:
        """Return immediate and held-repeat presses at an action-specific cadence."""

        key = self.binding(action).key
        if key is None:
            return 0
        keys = key if isinstance(key, tuple) else (key,)
        return max(
            (
                imgui.get_key_pressed_amount(imgui_key_for_physical_key(candidate), delay, rate)
                for candidate in keys
            ),
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
        candidate = replace(self, entries=tuple(entries))
        candidate._validate_pointers()
        return candidate

    def preferences(self) -> dict[str, object]:
        """Return the stable JSON representation used by editor settings."""

        return {
            **{action.value: self.key_id(action) for action, _binding in self.entries},
            "pointer": {
                action.value: [chord.identifier() for chord in chords]
                for action, chords in self.pointers
            },
        }

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
            try:
                bindings = bindings.remap(action, key_id)
            except ValueError:
                continue
        pointer_values = value.get("pointer")
        if isinstance(pointer_values, dict):
            entries = dict(bindings.pointers)
            for action in PointerAction:
                if action.value not in pointer_values:
                    continue
                chords = pointer_values[action.value]
                if not isinstance(chords, list) or not all(
                    isinstance(chord, str) for chord in chords
                ):
                    continue
                try:
                    entries[action] = tuple(PointerChord.parse(chord) for chord in chords)
                except ValueError:
                    continue
            candidate = replace(bindings, pointers=tuple(entries.items()))
            try:
                candidate._validate_pointers()
            except ValueError:
                pass
            else:
                bindings = candidate
        return bindings

    def pointer_chords(self, action: PointerAction) -> tuple[PointerChord, ...]:
        return next((chords for candidate, chords in self.pointers if candidate == action), ())

    def resolved_chord(self, chord: PointerChord) -> PointerChord | None:
        keys = []
        for key in chord.modifiers:
            if key in ("snap", "perturb"):
                key = self.key_id(InputAction(key))
                if key is None:
                    return None
            keys.append(key)
        return replace(chord, modifiers=tuple(sorted(set(keys))))

    def pointer_label(self, action: PointerAction) -> str:
        labels = []
        for chord in self.pointer_chords(action):
            resolved = self.resolved_chord(chord)
            if resolved is not None:
                labels.append(resolved.identifier())
        return "; ".join(labels) or "Unbound"

    def remap_pointer(self, action: PointerAction, identifiers: tuple[str, ...]) -> InputBindings:
        """Replace all alternatives of an action, rejecting conflicts atomically."""
        entries = dict(self.pointers)
        entries[PointerAction(action)] = tuple(PointerChord.parse(value) for value in identifiers)
        candidate = replace(self, pointers=tuple(entries.items()))
        candidate._validate_pointers()
        return candidate

    def navigation_preset(self, name: str) -> InputBindings:
        """Replace camera navigation while retaining editing and application bindings."""
        entries = dict(self.pointers)
        if name not in NAVIGATION_PRESETS:
            raise ValueError(f"Unknown navigation preset: {name}")
        for action, values in NAVIGATION_PRESETS[name].items():
            entries[action] = tuple(PointerChord.parse(value) for value in values)
        entries[PointerAction.DOLLY] = (PointerChord.parse("wheel"),)
        candidate = replace(self, pointers=tuple(entries.items()))
        candidate._validate_pointers()
        return candidate

    def _validate_pointers(self) -> None:
        claimed = {}
        for action, chords in self.pointers:
            for chord in chords:
                resolved = self.resolved_chord(chord)
                if resolved is None:
                    continue
                if resolved.wheel != (action in (PointerAction.DOLLY, PointerAction.TIMELINE_ZOOM)):
                    raise ValueError(
                        f"{action} requires {'wheel' if not resolved.wheel else 'buttons'}"
                    )
                key = (exclusive_group(action), resolved)
                if key in claimed:
                    raise ValueError(f"{resolved.identifier()} is already used by {claimed[key]}")
                claimed[key] = action

    @cached_property
    def pointer_keys(self) -> frozenset[str]:
        keys = {"ctrl", "shift", "alt", "super"}
        for _action, chords in self.pointers:
            for chord in chords:
                resolved = self.resolved_chord(chord)
                if resolved is not None:
                    keys.update(resolved.modifiers)
        return frozenset(keys)

    def pointer_frame(self, claim: InputClaim = InputClaim()) -> PointerFrame:
        """Sample physical buttons, modifiers and registered held keys once per UI frame."""
        required = self.pointer_keys
        io = imgui.get_io()
        ctrl, super_key = physical_ctrl_super(io)
        modifiers = {
            "ctrl": ctrl,
            "super": super_key,
            "shift": io.key_shift
            or imgui.is_key_down(imgui.Key.left_shift)
            or imgui.is_key_down(imgui.Key.right_shift),
            "alt": io.key_alt
            or imgui.is_key_down(imgui.Key.left_alt)
            or imgui.is_key_down(imgui.Key.right_alt),
        }
        keys = frozenset(
            key
            for key in required
            if not claim.claims_key(key)
            and (
                modifiers[key]
                if key in modifiers
                else any(imgui.is_key_down(value) for value in _imgui_keys(key))
            )
        )
        available = tuple(button for button in range(3) if not claim.claims_button(button))
        return PointerFrame(
            buttons=frozenset(button for button in available if imgui.is_mouse_down(button)),
            clicked=frozenset(button for button in available if imgui.is_mouse_clicked(button)),
            doubled=frozenset(
                button for button in available if imgui.is_mouse_double_clicked(button)
            ),
            released=frozenset(button for button in available if imgui.is_mouse_released(button)),
            keys=keys,
            wheel=0.0 if claim.wheel or claim.pointer else float(imgui.get_io().mouse_wheel),
        )

    def pointer_match(
        self, action: PointerAction, frame: PointerFrame, *, press: bool = False
    ) -> PointerChord | None:
        """Resolve an action, preferring the most specific matching chord in its context."""
        return next(
            (
                chord
                for candidate, chord in self.pointer_matches(frame, press=press)
                if candidate == action
            ),
            None,
        )

    @cached_property
    def resolved_pointers(self) -> tuple[tuple[PointerAction, PointerChord], ...]:
        return tuple(
            (action, resolved)
            for action, chords in self.pointers
            for chord in chords
            if (resolved := self.resolved_chord(chord)) is not None
        )

    def pointer_matches(
        self, frame: PointerFrame, *, press: bool = False
    ) -> tuple[tuple[PointerAction, PointerChord], ...]:
        return _pointer_matches(self.resolved_pointers, frame, press)


@lru_cache(maxsize=32)
def _pointer_matches(
    entries: tuple[tuple[PointerAction, PointerChord], ...], frame: PointerFrame, press: bool
) -> tuple[tuple[PointerAction, PointerChord], ...]:
    candidates = []
    best = {}
    for action, chord in entries:
        group = exclusive_group(action)
        if frame.matches(chord, press=press, extra_keys=group in ("gizmo", "view_cube")):
            rank = (len(chord.modifiers), chord.clicks)
            candidates.append((action, chord, group, rank))
            best[group] = max(best.get(group, (0, 0)), rank)
    return tuple(
        (action, chord) for action, chord, group, rank in candidates if rank == best[group]
    )


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
    InputAction.FRAME_SCENE: "Frame All",
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
