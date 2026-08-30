from __future__ import annotations

import itertools
import re

from forge_viewer.ui.gestures import (
    CameraGesture,
    Claim,
    GestureRouter,
    InputState,
    camera_gesture,
    claim_for,
    gizmo_yields,
    viewport_input_allowed,
)


def press(**kw) -> InputState:

    return InputState(**{"over_viewport": True, "has_selection": True, "left": True, **kw})


def test_blocking_prompt_aborts_an_existing_viewport_claim():
    router = GestureRouter()
    assert router.update(press(gizmo_available=True, gizmo_hovered=True)) is Claim.OBJECT_GIZMO
    assert router.held

    assert router.update(InputState(blocked=True, left=True)) is Claim.UI
    assert not router.held
    assert not router.wants_gizmo()


def test_camera_and_perturb_are_never_both_active():

    flags = (False, True)
    for left, right, middle, ctrl, shift, sel, perturbing, cube, giz in itertools.product(
        flags, flags, flags, flags, flags, flags, flags, flags, flags
    ):
        state = InputState(
            left=left,
            right=right,
            middle=middle,
            ctrl=ctrl,
            shift=shift,
            has_selection=sel,
            perturbing=perturbing,
            over_view_cube=cube,
            gizmo_available=giz,
            gizmo_hovered=giz,
            over_viewport=True,
        )
        router = GestureRouter()
        router.update(state)
        active = [
            router.wants_camera(),
            router.wants_perturb(),
            router.wants_view_cube(),
            router.wants_gizmo(),
        ]
        assert sum(active) <= 1, state


def test_claim_is_latched_at_the_press_frame_and_never_re_decided():

    router = GestureRouter()
    assert router.update(press(ctrl=True)) is Claim.PERTURB

    mid_drag = InputState(
        over_viewport=True, has_selection=True, left=True, ctrl=False, perturbing=True
    )
    assert router.update(mid_drag) is Claim.PERTURB

    reset_mid_drag = InputState(over_viewport=True, has_selection=True, left=True, ctrl=False)
    assert router.update(reset_mid_drag) is Claim.PERTURB
    assert router.wants_camera() is False

    release = InputState(over_viewport=True, has_selection=True, perturbing=True)
    assert router.update(release) is Claim.PERTURB
    assert router.released is True
    assert router.update(InputState(over_viewport=True)) is Claim.NONE


def test_a_started_orbit_is_not_stolen_by_ctrl_mid_drag():

    router = GestureRouter()
    assert router.update(press()) is Claim.CAMERA
    assert router.update(press(ctrl=True)) is Claim.CAMERA


def test_ctrl_makes_the_gizmo_yield_before_any_drag_exists():

    assert gizmo_yields(InputState(ctrl=True, perturbing=False)) is True

    assert gizmo_yields(InputState(ctrl=False, perturbing=True)) is True

    assert gizmo_yields(InputState()) is False


def test_gizmo_gets_nothing_while_ctrl_is_held():

    state = press(ctrl=True, gizmo_available=True, gizmo_hovered=True)
    assert claim_for(state) is Claim.PERTURB
    router = GestureRouter()
    router.update(state)
    assert router.wants_gizmo() is False


def test_view_cube_is_not_affected_by_ctrl():

    for ctrl in (False, True):
        state = press(ctrl=ctrl, over_view_cube=True)
        assert claim_for(state) is Claim.VIEW_CUBE


def test_ctrl_drag_without_a_selection_does_nothing():

    state = press(ctrl=True, has_selection=False)
    assert claim_for(state) is Claim.NONE


def test_mouse_gesture_table_matches_the_spec():

    assert camera_gesture(InputState(left=True)) is CameraGesture.ORBIT
    assert camera_gesture(InputState(right=True)) is CameraGesture.PAN
    assert camera_gesture(InputState(middle=True)) is CameraGesture.PAN
    assert camera_gesture(InputState(left=True, shift=True)) is CameraGesture.PAN
    assert camera_gesture(InputState(wheel=1.0)) is CameraGesture.DOLLY
    assert camera_gesture(InputState()) is CameraGesture.NONE


def test_perturb_mode_follows_the_button():

    router = GestureRouter()
    router.update(press(ctrl=True))
    assert router.mode == "translate"
    router = GestureRouter()
    router.update(press(ctrl=True, left=False, right=True))
    assert router.mode == "rotate"


def test_panels_take_the_mouse_outside_the_viewport():

    state = InputState(left=True, ui_wants_mouse=True, over_viewport=False)
    assert claim_for(state) is Claim.UI


def test_ui_drag_keeps_ownership_after_the_cursor_enters_the_viewport():

    router = GestureRouter()
    assert router.update(InputState(left=True, ui_wants_mouse=True)) is Claim.UI
    assert router.update(InputState(left=True, over_viewport=True, delta=(120.0, 0.0))) is Claim.UI
    assert not router.wants_camera()


def test_floating_panels_block_input_even_when_they_overlap_the_viewport_rect():
    assert viewport_input_allowed(inside=True, hovered_window="Viewport")
    assert viewport_input_allowed(inside=True, hovered_window="视口###Viewport")
    assert not viewport_input_allowed(inside=True, hovered_window="视口###Viewport/child")
    assert not viewport_input_allowed(inside=True, hovered_window="Inspector")
    assert not viewport_input_allowed(inside=True, hovered_window=None)


def test_help_panel_lists_every_key_the_main_loop_polls():
    from forge_viewer.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
    from forge_viewer.ui.panels.help import KEYS

    polled_actions = (
        InputAction.TOGGLE_PAUSE,
        InputAction.FRAME_SCENE,
        InputAction.GIZMO_TRANSLATE,
        InputAction.GIZMO_ROTATE,
        InputAction.GIZMO_SPACE,
        InputAction.AXIS_X,
        InputAction.AXIS_Y,
        InputAction.AXIS_Z,
        InputAction.FLY_FORWARD,
        InputAction.FLY_BACK,
        InputAction.FLY_RIGHT,
        InputAction.FLY_LEFT,
        InputAction.FLY_UP,
        InputAction.FLY_DOWN,
    )
    polled = {DEFAULT_INPUT_BINDINGS.label(action).casefold() for action in polled_actions}
    documented: set[str] = set()
    for keys, _text in KEYS:
        documented.update(t.lower() for t in re.split(r"[^A-Za-z0-9]+", keys) if t)
    assert polled
    assert polled <= documented


def test_travel_separates_a_click_from_a_drag():

    router = GestureRouter()
    router.update(press())
    router.update(press(delta=(3.0, 4.0)))
    assert router.travel == 7.0
