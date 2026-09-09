"""Serializable mouse chords and semantic editor gestures, without an input backend."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class PointerAction(enum.StrEnum):
    ORBIT = "camera.orbit"
    PAN = "camera.pan"
    DOLLY = "camera.dolly"
    DOLLY_DRAG = "camera.dolly_drag"
    SELECT = "selection.pick"
    FOCUS = "selection.focus"
    VIEW_CUBE = "view_cube.activate"
    GIZMO = "gizmo.drag"
    GIZMO_VALUE = "gizmo.type_value"
    PERTURB_TRANSLATE = "perturb.translate"
    PERTURB_ROTATE = "perturb.rotate"
    TIMELINE_SCRUB = "timeline.scrub"
    TIMELINE_PAN = "timeline.pan"
    TIMELINE_ZOOM = "timeline.zoom"
    TIMELINE_RANGE = "timeline.range"
    TIMELINE_LOAD = "timeline.load"
    PANEL_FOCUS = "panel.focus"
    NAME_EDIT = "panel.edit_name"
    COPY_NAME = "panel.copy_name"
    VALUE_RESET = "value.reset"
    VALUE_COPY = "value.copy"
    VALUE_EXPAND = "value.expand"
    ADD_SELECTION = "hierarchy.add_selection"
    OVERLAY_DRAG = "overlay.drag"
    STATUS_COPY = "status.copy"
    PROPERTY_COPY = "property.copy"
    SORT_ALTERNATE = "sort.alternate"
    OUTPUT_SELECT = "output.select"
    OUTPUT_ADD = "output.add_selection"
    OUTPUT_RANGE = "output.range_selection"
    OUTPUT_CONTEXT = "output.context"


@dataclass(frozen=True)
class PointerChord:
    """A physical button chord, optional held keys, and click count.

    ``snap`` and ``perturb`` refer to the corresponding configurable keyboard
    actions. Other key identifiers denote physical keys on every platform.
    """

    buttons: tuple[int, ...] = ()
    modifiers: tuple[str, ...] = ()
    wheel: bool = False
    clicks: int = 1

    def identifier(self) -> str:
        tokens = (*self.modifiers, *(("left", "right", "middle")[i] for i in self.buttons))
        value = "+".join((*tokens, "wheel") if self.wheel else tokens)
        return value + (":double" if self.clicks == 2 else "")

    @classmethod
    def parse(cls, value: str) -> PointerChord:
        """Parse one stable chord; malformed or ambiguous inputs fail atomically."""
        suffix = value.strip().casefold().split(":")
        if len(suffix) > 2 or (len(suffix) == 2 and suffix[1] != "double"):
            raise ValueError(f"Invalid mouse chord: {value}")
        tokens = suffix[0].split("+")
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"Repeated input in mouse chord: {value}")
        buttons = tuple(i for i, name in enumerate(("left", "right", "middle")) if name in tokens)
        keys = tuple(token for token in tokens if token not in {"left", "right", "middle", "wheel"})
        valid_keys = {
            "ctrl",
            "shift",
            "alt",
            "super",
            "snap",
            "perturb",
            *"abcdefghijklmnopqrstuvwxyz",
        }
        if any(key not in valid_keys for key in keys):
            raise ValueError(f"Unknown held key in mouse chord: {value}")
        wheel = "wheel" in tokens
        if (not buttons and not wheel) or (buttons and wheel) or (wheel and len(suffix) == 2):
            raise ValueError(f"A chord requires buttons or a wheel: {value}")
        return cls(buttons, tuple(sorted(keys)), wheel, len(suffix))


@dataclass(frozen=True)
class PointerFrame:
    """One immutable physical sample shared by viewport and panel routing."""

    buttons: frozenset[int] = frozenset()
    clicked: frozenset[int] = frozenset()
    doubled: frozenset[int] = frozenset()
    released: frozenset[int] = frozenset()
    keys: frozenset[str] = frozenset()
    wheel: float = 0.0

    def held(self, chord: PointerChord) -> bool:
        """Retain an acquired drag until its buttons release, even if modifiers change."""
        return bool(chord.buttons) and set(chord.buttons) <= self.buttons

    def matches(
        self, chord: PointerChord, *, press: bool = False, extra_keys: bool = False
    ) -> bool:
        if not set(chord.modifiers) <= self.keys:
            return False
        # Physical modifiers reserve combinations even when an action is disabled.
        if not extra_keys and (self.keys & {"ctrl", "shift", "alt", "super"}) - set(
            chord.modifiers
        ):
            return False
        if chord.wheel:
            return bool(self.wheel)
        if frozenset(chord.buttons) != self.buttons:
            return False
        if chord.clicks == 2:
            return bool(self.doubled.intersection(chord.buttons))
        return not press or bool(self.clicked.intersection(chord.buttons))


POINTER_ACTION_NAMES = {
    PointerAction.ORBIT: "Camera orbit",
    PointerAction.PAN: "Camera pan",
    PointerAction.DOLLY: "Camera wheel zoom",
    PointerAction.DOLLY_DRAG: "Camera drag zoom",
    PointerAction.SELECT: "Scene picking",
    PointerAction.FOCUS: "Focus on double-click",
    PointerAction.VIEW_CUBE: "View cube",
    PointerAction.GIZMO: "Gizmo drag",
    PointerAction.GIZMO_VALUE: "Gizmo numeric input",
    PointerAction.PERTURB_TRANSLATE: "Translation perturbation",
    PointerAction.PERTURB_ROTATE: "Rotation perturbation",
    PointerAction.TIMELINE_SCRUB: "Move playhead",
    PointerAction.TIMELINE_PAN: "Timeline pan",
    PointerAction.TIMELINE_ZOOM: "Timeline zoom",
    PointerAction.TIMELINE_RANGE: "Select loop range",
    PointerAction.TIMELINE_LOAD: "Load snapshot",
    PointerAction.PANEL_FOCUS: "Focus hierarchy or joint row",
    PointerAction.NAME_EDIT: "Edit entity name",
    PointerAction.COPY_NAME: "Copy row name",
    PointerAction.VALUE_RESET: "Reset slider value",
    PointerAction.VALUE_COPY: "Copy slider value",
    PointerAction.VALUE_EXPAND: "Expand slider options",
    PointerAction.ADD_SELECTION: "Add hierarchy selection",
    PointerAction.OVERLAY_DRAG: "Move viewport toolbar",
    PointerAction.STATUS_COPY: "Copy status text",
    PointerAction.PROPERTY_COPY: "Copy property value",
    PointerAction.SORT_ALTERNATE: "Alternative sort toggle",
    PointerAction.OUTPUT_SELECT: "Select output entry",
    PointerAction.OUTPUT_ADD: "Add output selection",
    PointerAction.OUTPUT_RANGE: "Select output range",
    PointerAction.OUTPUT_CONTEXT: "Output context menu",
}


def _bindings(**values) -> tuple[tuple[PointerAction, tuple[PointerChord, ...]], ...]:
    return tuple(
        (action, tuple(PointerChord.parse(chord) for chord in values.get(action.name, ())))
        for action in PointerAction
    )


DEFAULT_POINTER_BINDINGS = _bindings(
    ORBIT=("left",),
    PAN=("right", "middle", "snap+left", "snap+right", "snap+middle"),
    DOLLY=("wheel",),
    SELECT=("left",),
    FOCUS=("left:double",),
    VIEW_CUBE=("left",),
    GIZMO=("left",),
    GIZMO_VALUE=("left:double",),
    PERTURB_TRANSLATE=("perturb+left",),
    PERTURB_ROTATE=("perturb+right",),
    TIMELINE_SCRUB=("left",),
    TIMELINE_PAN=("right",),
    TIMELINE_ZOOM=("wheel",),
    TIMELINE_RANGE=("shift+right",),
    TIMELINE_LOAD=("left:double",),
    PANEL_FOCUS=("left:double",),
    NAME_EDIT=("left:double",),
    COPY_NAME=("right",),
    VALUE_RESET=("right",),
    VALUE_COPY=("left:double",),
    VALUE_EXPAND=("shift+right",),
    ADD_SELECTION=("ctrl+left", "super+left"),
    OVERLAY_DRAG=("left",),
    STATUS_COPY=("right",),
    PROPERTY_COPY=("right",),
    SORT_ALTERNATE=("right",),
    OUTPUT_SELECT=("left",),
    OUTPUT_ADD=("ctrl+left", "super+left"),
    OUTPUT_RANGE=("shift+left",),
    OUTPUT_CONTEXT=("right",),
)

# Presets affect navigation only; editing, perturbation and timeline bindings remain explicit.
NAVIGATION_PRESETS = {
    "Mojive": {
        PointerAction.ORBIT: ("left",),
        PointerAction.PAN: ("right", "middle", "snap+left", "snap+right", "snap+middle"),
        PointerAction.DOLLY_DRAG: (),
    },
    "Blender": {
        PointerAction.ORBIT: ("middle",),
        PointerAction.PAN: ("shift+middle",),
        PointerAction.DOLLY_DRAG: ("ctrl+middle",),
    },
    "Unity": {
        PointerAction.ORBIT: ("alt+left",),
        PointerAction.PAN: ("middle",),
        PointerAction.DOLLY_DRAG: ("alt+right",),
    },
    "Unreal": {
        PointerAction.ORBIT: ("alt+left",),
        PointerAction.PAN: ("alt+middle",),
        PointerAction.DOLLY_DRAG: ("alt+right",),
    },
    "MuJoCo": {
        PointerAction.ORBIT: ("left", "shift+left"),
        PointerAction.PAN: ("right", "shift+right"),
        PointerAction.DOLLY_DRAG: ("middle",),
    },
}


def exclusive_group(action: PointerAction) -> str:
    if action is PointerAction.NAME_EDIT:
        return "entity_name"
    if action in (PointerAction.SELECT, PointerAction.FOCUS):
        return "selection"
    if action in (PointerAction.GIZMO, PointerAction.GIZMO_VALUE):
        return "gizmo"
    if action is PointerAction.VIEW_CUBE:
        return "view_cube"
    if action in (
        PointerAction.ADD_SELECTION,
        PointerAction.OVERLAY_DRAG,
        PointerAction.STATUS_COPY,
    ):
        return action.value.split(".")[0]
    if action.value.startswith(("property.", "sort.", "output.")):
        return action.value.split(".")[0]
    if action.value.startswith("panel."):
        return "panel"
    if action.value.startswith("value."):
        return "value"
    if action.value.startswith("timeline."):
        return "timeline"
    return "viewport"
