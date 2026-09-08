"""Input ownership and viewport gesture classification."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .pointer_bindings import PointerAction


class Claim(enum.StrEnum):
    NONE = "none"
    UI = "ui"
    CAMERA = "camera"
    VIEW_CUBE = "view_cube"
    OBJECT_GIZMO = "object_gizmo"
    PERTURB = "perturb"


@dataclass(frozen=True)
class InputState:
    # Blocking prompts own the entire interaction frame, including modifiers
    # that would otherwise alter viewport hints without moving the scene.
    blocked: bool = False
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
    actions: frozenset[PointerAction] | None = None

    def action(self, action: PointerAction, legacy: bool = False) -> bool:
        return legacy if self.actions is None else action in self.actions

    @property
    def buttons(self) -> frozenset[int]:
        return frozenset(i for i, down in enumerate((self.left, self.right, self.middle)) if down)

    @property
    def any_button(self) -> bool:
        return self.left or self.right or self.middle


def viewport_input_allowed(inside: bool, hovered_window: str | None) -> bool:
    if not inside or hovered_window is None:
        return False
    window_id = hovered_window.rsplit("###", 1)[-1]
    return window_id == "Viewport"


def gizmo_yields(state: InputState) -> bool:
    return bool(state.ctrl or state.perturbing)


def claim_for(state: InputState) -> Claim:
    if state.blocked:
        return Claim.UI
    if state.ui_wants_mouse:
        return Claim.UI

    if state.over_view_cube and state.action(PointerAction.VIEW_CUBE, state.left):
        return Claim.VIEW_CUBE

    if gizmo_yields(state):
        if state.has_selection and (
            state.action(PointerAction.PERTURB_TRANSLATE, state.left)
            or state.action(PointerAction.PERTURB_ROTATE, state.right)
        ):
            return Claim.PERTURB
        return Claim.NONE

    if (
        state.gizmo_available
        and state.gizmo_hovered
        and state.action(PointerAction.GIZMO, state.left)
    ):
        return Claim.OBJECT_GIZMO

    navigation = state.actions is None or bool(
        state.actions
        & {
            PointerAction.ORBIT,
            PointerAction.PAN,
            PointerAction.DOLLY,
            PointerAction.DOLLY_DRAG,
            PointerAction.SELECT,
            PointerAction.FOCUS,
        }
    )
    if state.over_viewport and (state.any_button or state.wheel) and navigation:
        return Claim.CAMERA

    return Claim.NONE


class CameraGesture(enum.StrEnum):
    NONE = "none"
    ORBIT = "orbit"
    PAN = "pan"
    DOLLY = "dolly"
    DOLLY_DRAG = "dolly_drag"


def camera_gesture(state: InputState) -> CameraGesture:
    if state.actions is not None:
        for action, gesture in (
            (PointerAction.ORBIT, CameraGesture.ORBIT),
            (PointerAction.PAN, CameraGesture.PAN),
            (PointerAction.DOLLY, CameraGesture.DOLLY),
            (PointerAction.DOLLY_DRAG, CameraGesture.DOLLY_DRAG),
        ):
            if action in state.actions:
                return gesture
        return CameraGesture.NONE
    if state.right or state.middle or (state.left and state.shift):
        return CameraGesture.PAN
    if state.left:
        return CameraGesture.ORBIT
    if state.wheel:
        return CameraGesture.DOLLY
    return CameraGesture.NONE


def perturb_mode(state: InputState) -> str:
    return "rotate" if state.action(PointerAction.PERTURB_ROTATE, state.right) else "translate"


class GestureRouter:
    def __init__(self) -> None:
        self._claim = Claim.NONE
        self._held = False
        self._released = False
        self._mode = "translate"
        self._press_cursor = (0.0, 0.0)
        self._travel = 0.0
        self._started_with_left = False
        self._started_with_focus = False
        self._camera_gesture = CameraGesture.NONE
        self._buttons: frozenset[int] = frozenset()
        self._drain_buttons = False

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

    @property
    def started_with_left(self) -> bool:
        """Whether the current/released claim began with only the left button."""

        return self._started_with_left

    @property
    def started_with_focus(self) -> bool:
        return self._started_with_focus

    def camera_gesture(self, state: InputState) -> CameraGesture:
        return self._camera_gesture if self._held or self._released else camera_gesture(state)

    def update(self, state: InputState) -> Claim:
        if state.blocked:
            self.abort()
            self._claim = Claim.UI
            # Dismissing a popup must not transfer its held press to the scene
            # when the popup disappears on the following frame.
            self._held = state.any_button
            return self._claim
        if self._drain_buttons:
            self._drain_buttons = state.any_button
            self._released = False
            self._claim = Claim.NONE
            return self._claim
        if self._held:
            buttons = state.buttons
            # Assemble multi-button navigation before motion starts. Once moving,
            # ownership and the gesture remain stable until the acquired buttons release.
            if (
                state.actions is not None
                and self._claim is Claim.CAMERA
                and self._travel == 0.0
                and self._buttons < buttons
                and claim_for(state) is Claim.CAMERA
                and camera_gesture(state) is not CameraGesture.NONE
            ):
                self._buttons = buttons
                self._camera_gesture = camera_gesture(state)
                self._started_with_left = state.action(PointerAction.SELECT)
                self._started_with_focus = state.action(PointerAction.FOCUS)
            remains_held = state.any_button if self._claim is Claim.UI else self._buttons <= buttons
            if remains_held:
                self._travel += abs(state.delta[0]) + abs(state.delta[1])
                self._released = False
                return self._claim

            self._held = False
            self._released = True
            self._drain_buttons = state.any_button
            return self._claim
        self._released = False

        claim = claim_for(state)
        if state.any_button and claim is not Claim.NONE:
            self._held = True
            self._buttons = state.buttons
            self._mode = perturb_mode(state)
            self._press_cursor = state.cursor
            self._travel = 0.0
            self._started_with_left = state.action(
                PointerAction.SELECT, bool(state.left and not state.right and not state.middle)
            )
            self._started_with_focus = state.action(PointerAction.FOCUS)
            self._camera_gesture = camera_gesture(state)
        else:
            self._started_with_left = False
            self._started_with_focus = False
        self._claim = claim
        return claim

    def abort(self) -> None:
        self._held = False
        self._released = False
        self._started_with_left = False
        self._started_with_focus = False
        self._camera_gesture = CameraGesture.NONE
        self._buttons = frozenset()
        self._drain_buttons = False
        self._claim = Claim.NONE

    def wants_camera(self) -> bool:
        return self._claim is Claim.CAMERA

    def wants_perturb(self) -> bool:
        return self._claim is Claim.PERTURB

    def wants_view_cube(self) -> bool:
        return self._claim is Claim.VIEW_CUBE

    def wants_gizmo(self) -> bool:
        return self._claim is Claim.OBJECT_GIZMO
