"""Dispatch normalized gizmo gestures independently of windows and ImGui."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mojive.ui.gestures import InputState

if TYPE_CHECKING:
    from mojive.session import Session
    from mojive.types import CameraView

    from .core import ObjectGizmo
    from .state import PreciseGizmoInput


@dataclass(frozen=True, slots=True)
class GizmoInput:
    """Input ownership and editor policy resolved by the application for one frame."""

    state: InputState
    enabled: bool = True
    viewing_selected_camera: bool = False
    axis: int = -1
    precise_requested: bool = False
    claimed: bool = False
    released: bool = False
    snap: bool = False
    style_scale: float = 1.0


def update_gizmo_input(
    gizmo: ObjectGizmo,
    session: Session,
    camera: CameraView,
    rect: tuple[float, float, float, float],
    event: GizmoInput,
) -> PreciseGizmoInput | None:
    """Update one owned gesture, returning a precise-edit request for the host UI."""
    state = event.state
    if not event.enabled or state.blocked:
        gizmo.cancel()
        return None
    if event.viewing_selected_camera:
        gizmo.keyboard_interact(
            session, camera, rect, state.cursor, -1, style_scale=event.style_scale
        )
        gizmo.interact(
            session,
            camera,
            rect,
            state.cursor,
            claimed=False,
            left_down=state.any_button and event.claimed,
            released=event.released,
            style_scale=event.style_scale,
        )
        return None
    keyboard_was_active = gizmo.keyboard_using
    axis = event.axis
    if not keyboard_was_active and (not state.over_viewport or state.any_button):
        axis = -1
    if keyboard_was_active or axis >= 0:
        gizmo.keyboard_interact(
            session,
            camera,
            rect,
            state.cursor,
            axis,
            snap=state.shift or event.snap,
            style_scale=event.style_scale,
        )
        return None
    if state.gizmo_hovered and event.precise_requested:
        edit = gizmo.precise_input(session)
        if edit is not None:
            gizmo.cancel()
            return edit
    gizmo.interact(
        session,
        camera,
        rect,
        state.cursor,
        claimed=event.claimed,
        left_down=state.any_button and event.claimed,
        released=event.released,
        snap=state.shift or event.snap,
        style_scale=event.style_scale,
    )
    return None
