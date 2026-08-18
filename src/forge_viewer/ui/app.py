"""Main viewer UI loop and panel coordination."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs

from .. import commands as cmd
from ..adapters.base import FrameNeeds, NodeKind
from ..render.backend import FrameMode, LabelMode, RenderFlag
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
    from ..commands import CommandResult
    from ..session import Session

CLICK_SLOP_PT = 4.0


PICK_SCREEN_RADIUS_PT = 40.0

MODEL_EXTENSIONS = frozenset((".xml", ".mjcf", ".urdf"))
MODEL_FILTERS = ["Model files", "*.xml *.mjcf *.urdf", "All files", "*"]


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
        self._model_dialog: Any | None = None
        self._model_load_error = ""
        self._show_model_load_error = False
        self._model_drop_notice = ""
        self._model_drop_notice_until = 0.0

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
        if self._model_dialog is not None:
            self._model_dialog.kill()
            self._model_dialog = None
        if self.debug_bridge is not None:
            self.debug_bridge.close()
        self.backend.release()
        self.session.release()

    def load_model(self, path: str | Path) -> CommandResult:
        result = self.session.submit(cmd.LoadAsset(Path(path)))
        if result.ok:
            self._after_model_change()
            self._set_model_drop_notice(f"Loaded {self.session.asset_path.name}")
        else:
            self._report_model_error(result.message)
        return result

    def _after_model_change(self) -> None:
        self.router.abort()
        self.gizmo.cancel()
        self._model_camera_id = -1
        self._model_camera_view = None
        self._structure_generation = -1
        self._sync_structure()
        self._frame_scene(animate=False)

    def _open_model_dialog(self) -> None:
        if self._model_dialog is not None:
            return
        current = self.session.asset_path
        default_path = str(current.parent if current is not None else Path.cwd())
        self._model_dialog = portable_file_dialogs.open_file(
            "Open model", default_path, MODEL_FILTERS
        )
        self._set_model_drop_notice("Choose an MJCF or URDF model")

    def _poll_model_dialog(self) -> None:
        dialog = self._model_dialog
        if dialog is None or not dialog.ready(0):
            return
        self._model_dialog = None
        try:
            selected = dialog.result()
        except Exception as exc:
            self._report_model_error(str(exc))
            return
        if selected:
            self.load_model(selected[0])

    def _poll_model_drop(self) -> None:
        paths = self.window.consume_file_drops()
        if not paths:
            return
        if len(paths) != 1:
            self._report_model_error("Drop one model at a time")
            return
        path = paths[0]
        if path.suffix.lower() not in MODEL_EXTENSIONS:
            self._report_model_error(f"Unsupported model file: {path.name}")
            return
        self.load_model(path)

    def _set_model_drop_notice(self, message: str) -> None:
        self._model_drop_notice = message
        self._model_drop_notice_until = time.monotonic() + 1.8

    def _draw_main_menu(self) -> None:
        can_load = bool(self.session.adapter.caps.asset_loading)
        shortcut = "Cmd" if sys.platform == "darwin" else "Ctrl"
        open_model = False
        reload_model = False
        quit_viewer = False
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                open_model, _ = imgui.menu_item(
                    "Open Model...", f"{shortcut}+O", False, can_load and self._model_dialog is None
                )
                reload_model, _ = imgui.menu_item(
                    "Reload Model",
                    f"{shortcut}+Shift+O",
                    False,
                    can_load and self.session.asset_path is not None,
                )
                imgui.separator()
                quit_viewer, _ = imgui.menu_item("Quit", f"{shortcut}+Q", False, True)
                imgui.end_menu()
            path = self.session.asset_path
            if path is not None:
                imgui.text_disabled(path.name)
            imgui.end_main_menu_bar()

        io = imgui.get_io()
        modifier = bool(io.key_ctrl or io.key_super)
        if can_load and modifier and not io.want_text_input:
            open_model |= imgui.is_key_pressed(imgui.Key.o, False) and not io.key_shift
            reload_model |= imgui.is_key_pressed(imgui.Key.o, False) and bool(io.key_shift)
        quit_viewer |= modifier and imgui.is_key_pressed(imgui.Key.q, False)

        if open_model:
            self._open_model_dialog()
        if reload_model:
            result = self.session.submit(cmd.Reload())
            if result.ok:
                self._after_model_change()
                self._set_model_drop_notice(f"Reloaded {self.session.asset_path.name}")
            else:
                self._report_model_error(result.message)
        if quit_viewer:
            self.window.request_close()

    def _report_model_error(self, message: str) -> None:
        self._model_load_error = message
        self._show_model_load_error = True

    def _draw_model_load_error(self) -> None:
        if self._show_model_load_error:
            imgui.open_popup("Model load failed")
            self._show_model_load_error = False
        visible, _ = imgui.begin_popup_modal(
            "Model load failed", None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not visible:
            return
        imgui.text_wrapped(self._model_load_error)
        imgui.spacing()
        if imgui.button("OK", imgui.ImVec2(100.0, 0.0)):
            imgui.close_current_popup()
        imgui.end_popup()

    def frame(self) -> None:
        window = self.window
        now = time.perf_counter()
        dt = self._dt = min(0.1, now - self._last_time)
        self._last_time = now

        window.begin_frame()
        self._poll_model_dialog()
        self._poll_model_drop()
        self._draw_main_menu()
        window.begin_dockspace()
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
        self._draw_model_load_error()
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
            self._draw_model_drop_overlay()
        finally:
            imgui.pop_clip_rect()
        imgui.end()

    def _draw_model_drop_overlay(self) -> None:
        source = self.session.source
        empty = source is not None and source.instance_count == 0
        notice = self._model_drop_notice if time.monotonic() < self._model_drop_notice_until else ""
        dragging = self.window.file_drag_active and self.session.adapter.caps.asset_loading
        if not empty and not notice and not dragging:
            return
        message = (
            "Release to load this model"
            if dragging
            else notice or "Drop an MJCF or URDF model here\nFile > Open Model..."
        )
        lines = message.splitlines()
        sizes = [imgui.calc_text_size(line) for line in lines]
        scale = self.window.style_scale
        pad_x, pad_y = 18.0 * scale, 12.0 * scale
        width = max(float(size.x) for size in sizes) + 2.0 * pad_x
        height = sum(float(size.y) for size in sizes) + 2.0 * pad_y + (len(lines) - 1) * 3.0 * scale
        x, y, w, h = self._viewport_rect
        left = x + (w - width) * 0.5
        top = y + (h - height) * 0.5
        dl = imgui.get_window_draw_list()
        color = imgui.color_convert_float4_to_u32
        if dragging:
            dl.add_rect(
                imgui.ImVec2(x + 3.0 * scale, y + 3.0 * scale),
                imgui.ImVec2(x + w - 3.0 * scale, y + h - 3.0 * scale),
                color(imgui.ImVec4(0.95, 0.68, 0.24, 0.95)),
                8.0 * scale,
                2.0 * scale,
                0,
            )
        dl.add_rect_filled(
            imgui.ImVec2(left, top),
            imgui.ImVec2(left + width, top + height),
            color(imgui.ImVec4(0.08, 0.09, 0.11, 0.88)),
            7.0 * scale,
        )
        cursor_y = top + pad_y
        for line, size in zip(lines, sizes, strict=True):
            dl.add_text(
                imgui.ImVec2(left + (width - float(size.x)) * 0.5, cursor_y),
                color(imgui.ImVec4(0.93, 0.94, 0.95, 1.0)),
                line,
            )
            cursor_y += float(size.y) + 3.0 * scale

    def frame_needs(self) -> FrameNeeds:
        needs = FrameNeeds(poses=True).merge(self.panels.frame_needs())
        label_mode = self.backend.get_label_mode()
        frame_mode = self.backend.get_frame_mode()
        needs.contacts = (
            self.backend.get_flag(RenderFlag.CONTACTPOINT)
            or self.backend.get_flag(RenderFlag.CONTACTFORCE)
            or label_mode in (LabelMode.CONTACT_POINT, LabelMode.CONTACT_FORCE)
            or frame_mode is FrameMode.CONTACT
        )
        needs.tendons = (
            self.backend.get_flag(RenderFlag.TENDON)
            or self.backend.get_flag(RenderFlag.ACTUATOR)
            or label_mode is LabelMode.TENDON
        )
        needs.actuator = (
            self.backend.get_flag(RenderFlag.ACTUATOR) or label_mode is LabelMode.ACTUATOR
        )
        needs.deformables = bool(
            (self.session.source and self.session.source.dynamic_meshes)
            or self.backend.get_flag(RenderFlag.FLEXVERT)
            or self.backend.get_flag(RenderFlag.FLEXEDGE)
            or label_mode is LabelMode.FLEX
        )
        needs.islands = self.backend.get_flag(RenderFlag.ISLAND)
        needs.bvh = self.backend.get_flag(RenderFlag.BODYBVH) or self.backend.get_flag(
            RenderFlag.MESHBVH
        )
        needs.diagnostics = (
            needs.bvh
            or any(
                self.backend.get_flag(flag)
                for flag in (
                    RenderFlag.ACTUATOR,
                    RenderFlag.JOINT,
                    RenderFlag.COM,
                    RenderFlag.INERTIA,
                    RenderFlag.CAMERA,
                    RenderFlag.LIGHT,
                    RenderFlag.RANGEFINDER,
                    RenderFlag.CONSTRAINT,
                    RenderFlag.AUTOCONNECT,
                )
            )
            or label_mode
            in (
                LabelMode.JOINT,
                LabelMode.ACTUATOR,
                LabelMode.CONSTRAINT,
                LabelMode.CAMERA,
                LabelMode.LIGHT,
            )
            or frame_mode in (FrameMode.CAMERA, FrameMode.LIGHT)
        )
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
