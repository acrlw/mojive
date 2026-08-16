from __future__ import annotations

import enum
from dataclasses import dataclass


class Claim(enum.StrEnum):
    NONE = "none"
    UI = "ui"
    CAMERA = "camera"
    VIEW_CUBE = "view_cube"
    OBJECT_GIZMO = "object_gizmo"
    PERTURB = "perturb"


@dataclass(frozen=True)
class InputState:
    left: bool = False
    right: bool = False
    middle: bool = False
    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    wheel: float = 0.0
    cursor: tuple[float, float] = (0.0, 0.0)

    delta: tuple[float, float] = (0.0, 0.0)

    over_viewport: bool = False
    over_view_cube: bool = False

    gizmo_available: bool = False

    gizmo_hovered: bool = False
    has_selection: bool = False

    perturbing: bool = False

    ui_wants_mouse: bool = False

    @property
    def any_button(self) -> bool:
        return self.left or self.right or self.middle


def viewport_input_allowed(inside: bool, hovered_window: str | None) -> bool:

    return bool(inside and hovered_window == "Viewport")


def gizmo_yields(state: InputState) -> bool:

    return bool(state.ctrl or state.perturbing)


def claim_for(state: InputState) -> Claim:

    if state.ui_wants_mouse:
        return Claim.UI

    if state.over_view_cube and state.left:
        return Claim.VIEW_CUBE

    if gizmo_yields(state):
        if state.has_selection and (state.left or state.right):
            return Claim.PERTURB
        return Claim.NONE

    if state.gizmo_available and state.gizmo_hovered and state.left:
        return Claim.OBJECT_GIZMO

    if state.over_viewport and (state.any_button or state.wheel):
        return Claim.CAMERA

    return Claim.NONE


class CameraGesture(enum.StrEnum):
    NONE = "none"
    ORBIT = "orbit"
    PAN = "pan"
    DOLLY = "dolly"


def camera_gesture(state: InputState) -> CameraGesture:

    if state.right or state.middle or (state.left and state.shift):
        return CameraGesture.PAN
    if state.left:
        return CameraGesture.ORBIT
    if state.wheel:
        return CameraGesture.DOLLY
    return CameraGesture.NONE


def perturb_mode(state: InputState) -> str:

    return "rotate" if state.right else "translate"


class GestureRouter:
    def __init__(self) -> None:
        self._claim = Claim.NONE
        self._held = False
        self._released = False
        self._mode = "translate"
        self._press_cursor = (0.0, 0.0)
        self._travel = 0.0

    @property
    def claim(self) -> Claim:
        return self._claim

    @property
    def held(self) -> bool:

        return self._held

    @property
    def released(self) -> bool:

        return self._released

    @property
    def mode(self) -> str:

        return self._mode

    @property
    def travel(self) -> float:

        return self._travel

    @property
    def press_cursor(self) -> tuple[float, float]:
        return self._press_cursor

    def update(self, state: InputState) -> Claim:
        if self._held:
            if state.any_button:
                self._travel += abs(state.delta[0]) + abs(state.delta[1])
                self._released = False
                return self._claim

            self._held = False
            self._released = True
            return self._claim
        self._released = False

        claim = claim_for(state)
        if state.any_button and claim is not Claim.NONE:
            self._held = True
            self._mode = perturb_mode(state)
            self._press_cursor = state.cursor
            self._travel = 0.0
        self._claim = claim
        return claim

    def abort(self) -> None:

        self._held = False
        self._released = False
        self._claim = Claim.NONE

    def wants_camera(self) -> bool:
        return self._claim is Claim.CAMERA

    def wants_perturb(self) -> bool:
        return self._claim is Claim.PERTURB

    def wants_view_cube(self) -> bool:
        return self._claim is Claim.VIEW_CUBE

    def wants_gizmo(self) -> bool:
        return self._claim is Claim.OBJECT_GIZMO
