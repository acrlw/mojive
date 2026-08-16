from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui

from .. import commands as cmd
from ..adapters.base import FrameNeeds, NodeKind
from ..render.backend import RenderFlag
from ..types import ViewportImage
from . import gestures as gs
from .camera import CameraOut, OrbitCamera, ndc_from_viewport, unproject
from .gizmo import ObjectGizmo
from .panels import PanelContext, PanelSet
from .perturb import PerturbController, draw_fallback
from .theme import THEME, Theme
from .viewcube import ViewCube
from .window import Window, WindowConfig

if TYPE_CHECKING:
    from ..session import Session

CLICK_SLOP_PT = 4.0


PICK_SCREEN_RADIUS_PT = 40.0


@dataclass
class Keys:
    fly: tuple[float, float, float] = (0.0, 0.0, 0.0)
    toggle_pause: bool = False
    frame_scene: bool = False
    gizmo_translate: bool = False
    gizmo_rotate: bool = False
    gizmo_space: bool = False
    gizmo_axis: int = -1


class ViewerApp:
    def __init__(
        self,
        session: Session,
        backend: Any,
        window: Window | None = None,
        *,
        title: str = "forge-viewer",
        theme: Theme | None = None,
        debug_bridge: Any | None = None,
    ) -> None:
        self.session = session
        self.backend = backend
        self.window = window
        self.title = title
        self.theme = theme or THEME
        self.debug_bridge = debug_bridge
        self.camera = OrbitCamera()

        self.camera_out = CameraOut(backend=backend, session=session)
        self.camera.attach(self.camera_out)
        self.gizmo = ObjectGizmo()
        self.view_cube = ViewCube()
        self.perturb = PerturbController()
        self.router = gs.GestureRouter()
        self.panels = PanelSet()
        self._started = False
        self._frame_index = 0
        self._last_time = time.perf_counter()
        self._viewport_rect = (0.0, 0.0, 640.0, 480.0)
        self._viewport_image: ViewportImage | None = None
        self._dt = 0.0
        self._structure_generation = -1
        self._state = gs.InputState()
        self._model_camera_id = -1
        self._model_camera_view = None
        self._fixed_render_size: tuple[int, int] | None = None

    def set_fixed_render_size(self, width: int, height: int) -> None:

        self._fixed_render_size = (max(1, int(width)), max(1, int(height)))
        self.backend.resize(*self._fixed_render_size)
        self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])

    def _startup(self) -> None:

        if self._started:
            return
        if self.window is None:
            self.window = Window(WindowConfig(title=self.title))
        self._sync_structure()
        self._frame_scene(animate=False)
        self.window.show()
        self._started = True
        self._last_time = time.perf_counter()

    def run(self, max_frames: int | None = None) -> None:

        self._startup()
        while not self._should_close():
            if max_frames is not None and self._frame_index >= max_frames:
                break
            self.frame()
        self.release()

    def sync(self) -> None:

        self._startup()
        self.frame()

    def _should_close(self) -> bool:
        return bool(self.window.should_close())

    def release(self) -> None:
        if self.debug_bridge is not None:
            self.debug_bridge.close()
        self.backend.release()
        self.session.release()

    def frame(self) -> None:

        window = self.window
        now = time.perf_counter()
        dt = self._dt = min(0.1, now - self._last_time)
        self._last_time = now

        window.begin_frame()
        keys = self._poll_keys()
        self.apply_keys(keys)

        state = self._state = self._input_state()

        self._claim_gesture(state)

        self._poll_gizmo(state, keys)
        self._poll_camera(state, keys, dt)
        self._poll_perturb(state)
        self._poll_pick(state)
        self._advance_camera(dt)

        frame = self.session.tick(self.frame_needs(), wall_dt=dt)
        self._sync_structure()
        self._sync_model_camera()
        self.backend.update(frame)

        self.backend.highlight(self.session.selected)
        self._sync_viewport_size()

        if self.debug_bridge is not None:
            self.debug_bridge.pump()
            if frame.debug_commands:
                self.debug_bridge.apply_batch(frame.debug_commands)

        self._publish_perturb_marks()
        self._publish_gizmo()

        self._viewport_image = self.backend.render(frame)

        ctx = self._panel_context()
        self._draw_viewport(ctx)
        self.panels.draw(ctx)
        window.end_frame()
        self._frame_index += 1

    def _poll_keys(self) -> Keys:

        k = imgui.Key
        io = imgui.get_io()
        self.panels.poll_shortcuts()

        if io.want_text_input:
            return Keys()

        def down(key) -> float:
            return 1.0 if imgui.is_key_down(key) else 0.0

        axis = next((i for i, key in enumerate((k.x, k.y, k.z)) if imgui.is_key_down(key)), -1)

        return Keys(
            fly=(
                down(k.w) - down(k.s),
                down(k.d) - down(k.a),
                down(k.q) - down(k.e),
            ),
            toggle_pause=imgui.is_key_pressed(k.space, False),
            frame_scene=imgui.is_key_pressed(k.f, False),
            gizmo_translate=imgui.is_key_pressed(k.g, False),
            gizmo_rotate=imgui.is_key_pressed(k.r, False),
            gizmo_space=imgui.is_key_pressed(k.t, False),
            gizmo_axis=axis,
        )

    def _input_state(self) -> gs.InputState:
        io = imgui.get_io()
        cursor = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        rect = self._viewport_rect
        inside = (
            rect[0] <= cursor[0] <= rect[0] + rect[2] and rect[1] <= cursor[1] <= rect[1] + rect[3]
        )
        hovered_window = imgui.get_current_context().hovered_window
        hovered_name = None if hovered_window is None else str(hovered_window.name)
        over_viewport = gs.viewport_input_allowed(inside, hovered_name)
        view = self._camera_view()
        hovered_ball = self.view_cube.update(view, rect, cursor, self.window.style_scale)
        self.gizmo.update_hover(self.session, view, rect, cursor)
        node = self.session.selected_node
        return gs.InputState(
            left=imgui.is_mouse_down(0),
            right=imgui.is_mouse_down(1),
            middle=imgui.is_mouse_down(2),
            ctrl=io.key_ctrl,
            shift=io.key_shift,
            alt=io.key_alt,
            wheel=float(io.mouse_wheel),
            cursor=cursor,
            delta=(float(io.mouse_delta.x), float(io.mouse_delta.y)),
            over_viewport=over_viewport,
            over_view_cube=over_viewport and hovered_ball is not None,
            gizmo_available=(self.gizmo.style == "2d" or self.backend.caps.gizmo)
            and self.gizmo.last_verdict.ok,
            gizmo_hovered=over_viewport and self.gizmo.hovered,
            has_selection=node is not None,
            perturbing=self.session.perturb.active,
            ui_wants_mouse=io.want_capture_mouse and not over_viewport,
        )

    def _claim_gesture(self, state: gs.InputState) -> gs.Claim:

        return self.router.update(state)

    def _poll_gizmo(self, state: gs.InputState, keys: Keys) -> None:

        keyboard_was_active = self.gizmo.keyboard_using
        axis = keys.gizmo_axis
        if not keyboard_was_active and (not state.over_viewport or state.any_button):
            axis = -1
        if keyboard_was_active or axis >= 0:
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                axis,
                snap=state.shift,
            )
            return
        self.gizmo.interact(
            self.session,
            self._camera_view(),
            self._viewport_rect,
            state.cursor,
            claimed=self.router.wants_gizmo(),
            left_down=state.left,
            released=self.router.released,
            snap=state.shift,
        )

    def _publish_gizmo(self) -> None:

        self.gizmo.publish(
            self.backend,
            self.session,
            self._camera_view(),
            self._viewport_rect,
            pixel_scale=self.window.pixel_scale,
            yielding=gs.gizmo_yields(self._state),
            interactive=self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO),
        )

    def _poll_camera(self, state: gs.InputState, keys: Keys, dt: float) -> None:
        fwd, right, up = keys.fly
        if fwd or right or up:
            self._leave_model_camera()
            self.camera.fly(dt, forward=fwd, right=right, up=up)
        if keys.frame_scene:
            self._leave_model_camera()
            self._frame_scene(animate=True)

        if self.router.wants_view_cube():
            ball = self.view_cube.hovered

            if self.router.travel >= CLICK_SLOP_PT and state.delta != (0.0, 0.0):
                self._leave_model_camera()
                self.view_cube.drag(self.camera, *state.delta)
            elif ball is not None and self.router.released and self.router.travel < CLICK_SLOP_PT:
                self._leave_model_camera()
                self.view_cube.click(self.camera, ball, self.camera_out)
            return

        if not self.router.wants_camera():
            return
        gesture = gs.camera_gesture(state)

        settled = self.router.travel >= CLICK_SLOP_PT
        if gesture is gs.CameraGesture.ORBIT and settled:
            self._leave_model_camera()
            self.camera.orbit(*state.delta)
        elif gesture is gs.CameraGesture.PAN and settled:
            self._leave_model_camera()
            self.camera.pan(state.delta[0], state.delta[1], self._viewport_rect[3])
        elif gesture is gs.CameraGesture.DOLLY:
            self._leave_model_camera()
            self.camera.dolly(state.wheel)

    def _advance_camera(self, dt: float) -> None:

        if self._model_camera_id >= 0:
            return
        self.camera.advance(dt, self.camera_out)

    def _camera_view(self):
        return self._model_camera_view or self.camera.view()

    def select_model_camera(self, camera_id: int) -> None:
        i = int(camera_id)
        if i >= 0 and not any(c.camera_id == i for c in self.session.cameras):
            return
        if i < 0:
            self._leave_model_camera(publish=True)
            return
        self._model_camera_id = i

    def _sync_model_camera(self) -> None:
        if self._model_camera_id < 0:
            return
        view = self.session.camera_view(self._model_camera_id)
        if view is None:
            self._leave_model_camera(publish=True)
            return
        aspect = max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0)
        view = view.with_aspect(aspect)
        self._model_camera_view = view
        self.backend.set_camera(view)
        self.session.submit(cmd.SetCamera(view))

    def _leave_model_camera(self, *, publish: bool = False) -> None:
        if self._model_camera_id < 0:
            return
        if self._model_camera_view is not None:
            self.camera.adopt(self._model_camera_view)
        self._model_camera_id = -1
        self._model_camera_view = None
        if publish:
            self.camera.publish(self.camera_out)

    def _frame_scene(self, *, animate: bool = True) -> None:

        self.camera.set_aspect(max(self._viewport_rect[2], 1.0) / max(self._viewport_rect[3], 1.0))
        self.camera.frame_scene(self.session.bounds(), self.camera_out, animate=animate)

    def _poll_perturb(self, state: gs.InputState) -> None:

        st = self.session.perturb
        if not self.router.wants_perturb():
            if st.active:
                self.perturb.end(self.session)
            return

        node = self.session.selected_node
        if node is None:
            return
        cam = self._camera_view()
        if not st.active:
            pos, _ = self._node_pose(node)
            self.perturb.begin(
                self.session, cam, node, pos, self.router.mode, body_radius=self._body_radius(node)
            )
        if st.mode == "translate":
            origin, direction = self._cursor_ray(state.cursor)
            self.perturb.drag_translate(self.session, cam, origin, direction)
        else:
            self.perturb.drag_rotate(self.session, cam, state.delta[0], state.delta[1])
        self.perturb.apply(self.session)

    def _publish_perturb_marks(self) -> None:
        self.perturb.publish_marks(
            self.backend,
            self.session,
            self._camera_view(),
            rect=self._viewport_rect,
            ui_scale=self.window.ui_scale,
        )

    def _poll_pick(self, state: gs.InputState) -> None:

        if not self.router.wants_camera():
            return
        if not self.router.released or self.router.travel > CLICK_SLOP_PT:
            return
        if not state.over_viewport:
            return
        object_id = self._pick_at(state.cursor)
        self.session.submit(cmd.Select(object_id))

    def _pick_at(self, cursor: tuple[float, float]) -> int:

        rect = self._viewport_rect

        img = self._viewport_image
        if self.backend.caps.gpu_pick and img is not None:
            hit = img.pixel_from_viewport_point(cursor, rect)
            if hit is not None:
                object_id = int(self.backend.pick(*hit))
                if self._selectable(object_id):
                    return object_id

        if self.session.adapter.caps.raycast:
            origin, direction = self._cursor_ray(cursor)
            object_id, _dist = self.session.query(cmd.Pick(origin=origin, direction=direction))
            if self._selectable(int(object_id)):
                return int(object_id)

        return self._nearest_link(cursor)

    def _selectable(self, object_id: int) -> bool:

        if object_id <= 0:
            return False
        node = self.session.node_by_object_id(object_id)
        if node is None:
            return False
        return node.kind is not NodeKind.WORLD and node.parent >= 0

    def _nearest_link(self, cursor: tuple[float, float]) -> int:
        frame = self.session.frame
        if frame.body_xpos is None or len(frame.body_xpos) == 0:
            return 0
        cam = self._camera_view()
        mvp = cam.proj_matrix() @ cam.view_matrix()
        pts = np.asarray(frame.body_xpos, np.float64)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ mvp.T
        w = np.where(np.abs(h[:, 3]) < 1e-9, 1e-9, h[:, 3])
        rect = self._viewport_rect
        sx = rect[0] + (h[:, 0] / w * 0.5 + 0.5) * rect[2]
        sy = rect[1] + (0.5 - h[:, 1] / w * 0.5) * rect[3]
        d2 = (sx - cursor[0]) ** 2 + (sy - cursor[1]) ** 2
        d2[w <= 0.0] = np.inf
        best_body = int(np.argmin(d2))
        limit = (PICK_SCREEN_RADIUS_PT * self.window.style_scale) ** 2
        if not np.isfinite(d2[best_body]) or d2[best_body] > limit:
            return 0
        for node in self.session.nodes:
            if node.body_index == best_body and self._selectable(node.object_id):
                return int(node.object_id)
        return 0

    def _draw_viewport(self, ctx: PanelContext) -> None:

        imgui.begin("Viewport", None, imgui.WindowFlags_.no_scrollbar.value)
        pos = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        self._viewport_rect = (
            float(pos.x),
            float(pos.y),
            max(float(size.x), 1.0),
            max(float(size.y), 1.0),
        )
        image = self._viewport_image
        if image is None:
            imgui.text_disabled("No viewport image is available")
        else:
            uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
            uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
            imgui.image(imgui.ImTextureRef(image.texture_id), size, uv0, uv1)
        x, y, w, h = self._viewport_rect
        imgui.push_clip_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), True)
        try:
            st = self.session.perturb
            if st.active and not self.backend.caps.debug_draw:
                node = self.session.node(st.node_id)
                center = self._node_pose(node)[0] if node is not None else st.target_pos
                draw_fallback(
                    self._camera_view(),
                    st,
                    self._viewport_rect,
                    (imgui.get_io().mouse_pos.x, imgui.get_io().mouse_pos.y),
                    center,
                    self.window.style_scale,
                )
            self.gizmo.draw_overlay(
                self._camera_view(),
                self._viewport_rect,
                style_scale=self.window.style_scale,
            )
            self.view_cube.draw(self.window.style_scale)
        finally:
            imgui.pop_clip_rect()
        imgui.end()

    def frame_needs(self) -> FrameNeeds:

        needs = FrameNeeds(poses=True).merge(self.panels.frame_needs())
        needs.contacts = self.backend.get_flag(RenderFlag.CONTACTPOINT) or self.backend.get_flag(
            RenderFlag.CONTACTFORCE
        )
        needs.tendons = self.backend.get_flag(RenderFlag.TENDON) or self.backend.get_flag(
            RenderFlag.ACTUATOR
        )
        needs.actuator = self.backend.get_flag(RenderFlag.ACTUATOR)
        needs.deformables = bool(self.session.source and self.session.source.dynamic_meshes)
        return needs

    def _sync_structure(self) -> None:

        gen = self.session.structure_generation
        if gen != self._structure_generation:
            self._structure_generation = gen
            self.backend.set_scene(self.session.source)

    def _sync_viewport_size(self) -> None:

        if self._fixed_render_size is not None:
            self.backend.resize(*self._fixed_render_size)
            self.camera.set_aspect(self._fixed_render_size[0] / self._fixed_render_size[1])
            return
        settled = self.window.poll_render_size(self._viewport_rect[2:])
        if settled is None:
            return
        sw, sh = settled
        self.backend.resize(sw, sh)
        self.camera.set_aspect(max(sw, 1) / max(sh, 1))

    def _panel_context(self) -> PanelContext:

        return PanelContext(
            session=self.session,
            backend=self.backend,
            camera=self.camera,
            model_camera_id=self._model_camera_id,
            model_camera_view=self._model_camera_view,
            select_model_camera=self.select_model_camera,
            gizmo=self.gizmo,
            perturb=self.perturb,
            theme=self.theme,
            style_scale=self.window.style_scale,
            viewport_rect=self._viewport_rect,
            dt=self._dt,
            status=self.session.last_message,
        )

    def _cursor_ray(self, cursor: tuple[float, float]):
        ndc = ndc_from_viewport(cursor[0], cursor[1], self._viewport_rect)
        return unproject(self._camera_view(), *ndc)

    def _node_pose(self, node) -> tuple[np.ndarray, np.ndarray]:
        from .perturb import current_pose

        return current_pose(self.session, node)

    def _body_radius(self, node) -> float:
        src = self.session.source
        if src is None or len(src.geom_size) == 0:
            return 0.1
        sizes = src.geom_size[np.asarray(src.geom_body) == node.body_index]
        return float(np.max(sizes)) if len(sizes) else 0.1

    def apply_keys(self, keys: Keys) -> None:

        if keys.toggle_pause:
            self.session.submit(cmd.Play() if self.session.paused else cmd.Pause())
        if keys.gizmo_translate:
            self.gizmo.set_mode("translate")
        if keys.gizmo_rotate:
            self.gizmo.set_mode("rotate")
        if keys.gizmo_space:
            self.gizmo.toggle_space()
