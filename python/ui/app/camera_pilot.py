"""Scene-camera navigation with an independent editor view and one edit per gesture."""

from dataclasses import dataclass, replace

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.ui import gestures as gs
from mojive.ui.camera import OrbitCamera

from .support import CLICK_SLOP_PT


@dataclass
class CameraPilot:
    camera_id: int
    document_id: str
    orbit: OrbitCamera
    editing: bool = False
    canceled: bool = False


def start_pilot(app, camera_id):
    node = next(
        (
            node
            for node in app.session.nodes
            if node.type is NodeType.CAMERA
            and app.session.cameras[node.camera_index].camera_id == camera_id
        ),
        None,
    )
    view = app.session.camera_view(camera_id)
    if node is None or view is None:
        return False
    app.select_model_camera(camera_id)
    if app._model_camera_id != camera_id or getattr(app, "_camera_pilot", None) is not None:
        return False
    orbit = OrbitCamera(navigation=app.camera.navigation)
    orbit.adopt(view, exact=True)
    app._camera_pilot = CameraPilot(camera_id, app.session.document_id, orbit)
    return True


def finish_pilot(app, *, cancel=False):
    pilot = getattr(app, "_camera_pilot", None)
    if pilot is not None and pilot.editing and pilot.document_id == app.session.document_id:
        result = app.session.submit(
            cmd.CancelEditTransaction() if cancel else cmd.EndEditTransaction()
        )
        if not result.ok:
            return False
    app._camera_pilot = None
    return True


def poll_pilot(app, state, keys, dt):
    pilot = getattr(app, "_camera_pilot", None)
    if pilot is None:
        return
    session = app.session
    source = session.camera_view(pilot.camera_id)
    if pilot.document_id != session.document_id or source is None:
        finish_pilot(app, cancel=True)
        return
    gesture = app.router.camera_gesture(state) if app.router.wants_camera() else None
    allowed = {
        gs.CameraGesture.ORBIT: app.interactions.camera.orbit,
        gs.CameraGesture.PAN: app.interactions.camera.pan,
        gs.CameraGesture.DOLLY_DRAG: app.interactions.camera.dolly,
        gs.CameraGesture.DOLLY: app.interactions.camera.dolly,
    }.get(gesture, False)
    motion = (
        bool(state.wheel)
        if gesture is gs.CameraGesture.DOLLY
        else (app.router.travel >= CLICK_SLOP_PT and state.delta != (0, 0))
    )
    moving = bool(any(keys.fly) or (allowed and motion))
    held = bool(
        any(keys.fly) or (app.router.wants_camera() and not app.router.released and not state.wheel)
    )
    if keys.clear_selection and pilot.editing:
        result = session.submit(cmd.CancelEditTransaction())
        if result.ok:
            pilot.editing, pilot.canceled = False, True
            pilot.orbit.adopt(session.camera_view(pilot.camera_id), exact=True)
        return
    if pilot.canceled:
        pilot.canceled = held
        return
    if moving:
        if not pilot.editing:
            if session.editing or not session.submit(cmd.BeginEditTransaction("Pilot camera")).ok:
                return
            pilot.editing = True
            pilot.orbit.adopt(source, exact=True)
        orbit = pilot.orbit
        if any(keys.fly):
            orbit.fly(dt, *keys.fly)
        if gesture is gs.CameraGesture.ORBIT and app.interactions.camera.orbit:
            orbit.orbit(*state.delta)
        elif gesture is gs.CameraGesture.PAN and app.interactions.camera.pan:
            orbit.pan(*state.delta, app._viewport_rect[3])
        elif gesture is gs.CameraGesture.DOLLY_DRAG and app.interactions.camera.dolly:
            orbit.dolly(-state.delta[1] * 0.05)
        elif gesture is gs.CameraGesture.DOLLY and app.interactions.camera.dolly:
            orbit.dolly(state.wheel)
        view = orbit.view()
        result = session.submit(
            cmd.SetSceneCamera(
                pilot.camera_id, replace(source, eye=view.eye, target=view.target, up=view.up)
            )
        )
        if not result.ok:
            finish_pilot(app, cancel=True)
            return
    if pilot.editing and not held and session.submit(cmd.EndEditTransaction()).ok:
        pilot.editing = False
