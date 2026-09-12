"""App: viewport."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.capture import RecordingPhase
from mojive.interaction.gizmo import GizmoMode
from mojive.render.debugdraw import Occlusion
from mojive.ui import gestures as gs
from mojive.ui.controls import action_menu_popup
from mojive.ui.draw2d import ImguiDraw2D
from mojive.ui.input_bindings import InputAction
from mojive.ui.panels import (
    PanelContext,
)
from mojive.ui.perturb import draw_axes as draw_perturb_axes
from mojive.ui.perturb import (
    draw_fallback,
    draw_translation_link,
)
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import (
    HINT_CHROME_SCALE,
    OVERLAY_CLIP_PADDING,
    OVERLAY_GEOMETRY,
    PLAYBACK_CHROME_SCALE,
    TOOL_CHROME_SCALE,
    ToolHint,
    default_tool_hints,
    draw_playback,
    draw_scene_tool_hints,
    draw_tool_column,
    fitting_tool_hints,
    normalized_overlay_position,
    overlay_border_hit,
    playback_size,
    positioned_overlay_rect,
    tool_column_size,
    tool_hints_size,
    viewport_chrome_scale,
)

from .support import (
    _DEFAULT_INTERACTIONS,
    _NO_INPUT_CLAIM,
    _SELECTION_BOX_EDGES,
    _SELECTION_BOX_SIGNS,
    CLICK_SLOP_PT,
    PICK_SCREEN_RADIUS_PT,
    VIEWPORT_DOUBLE_CLICK_RADIUS_PT,
    VIEWPORT_DOUBLE_CLICK_SECONDS,
    _clipped_overlay_draw,
    _clipped_overlay_host_rect,
    _fit_image_rect,
    _rectangles_overlap,
    precise_input_status_hints,
)


class _Viewport:
    """Private viewport methods of ViewerApp; state belongs to its owner."""

    def _publish_perturb_marks(self) -> None:
        self.perturb.publish_marks(
            self.backend,
            self.session,
            self._camera_view(),
            rect=self._viewport_rect,
            ui_scale=self.window.ui_scale,
            style_scale=self.window.style_scale,
            overlay_axes=True,
        )

    def _publish_selection_style(self) -> None:
        """Publish selected-object helpers without changing logical selection."""

        debug = getattr(self.backend, "debug", None)
        if debug is None:
            return
        layer = debug.layer("ui.selection", Occlusion.GHOST)
        node = self.session.selected_node
        style = self.selection_style
        if node is None or not (style.frame or style.label or style.bounds):
            layer.clear()
            return

        position, rotation = self._node_pose(node)
        if style.frame:
            transform = self._selection_transform
            transform.fill(0.0)
            transform[3, 3] = 1.0
            transform[:3, :3] = rotation
            transform[:3, 3] = position
            length = max(float(self.session.source.debug_frame_length), 1e-4)
            layer.frame("frame", transform, length)
        else:
            layer.erase("frame")

        if style.label:
            layer.text("label", position, node.name, offset_px=(8.0, -8.0))
        else:
            layer.erase("label")

        bounds = self.session.node_world_bounds(node.node_id) if style.bounds else None
        if bounds is None:
            layer.erase("bounds")
            return
        center, half = bounds
        corners = self._selection_bounds_corners
        np.multiply(_SELECTION_BOX_SIGNS, np.asarray(half, np.float32), out=corners)
        corners += np.asarray(center, np.float32)
        np.take(corners, _SELECTION_BOX_EDGES[:, 0], axis=0, out=self._selection_bounds_starts)
        np.take(corners, _SELECTION_BOX_EDGES[:, 1], axis=0, out=self._selection_bounds_ends)
        layer.lines(
            "bounds",
            self._selection_bounds_starts,
            self._selection_bounds_ends,
            self.theme.warning,
            1.5 * self.window.ui_scale,
        )

    def _poll_pick(self, state: gs.InputState) -> None:
        selection = getattr(self, "interactions", _DEFAULT_INTERACTIONS).selection
        input_claim = getattr(self, "_input_claim", _NO_INPUT_CLAIM)
        if not selection.pick or input_claim.pointer:
            return
        # A plain viewport click is classified as a camera-region gesture by
        # the router even when every camera motion switch is disabled. Keep
        # gizmo, view-cube, and perturb releases out of scene selection.
        if not self.router.wants_camera():
            return
        if not self.router.released:
            return
        if self.router.travel > CLICK_SLOP_PT:
            self._last_viewport_click = None
            return
        if not (self.router.started_with_left or self.router.started_with_focus):
            return
        if not state.over_viewport:
            return
        if not selection.pick_on_focus and not getattr(
            self, "_selection_press_started_focused", True
        ):
            return
        object_id = self._pick_at(state.cursor)
        now = time.monotonic()
        previous = self._last_viewport_click
        radius = VIEWPORT_DOUBLE_CLICK_RADIUS_PT * self.window.style_scale
        double_clicked = bool(
            previous is not None
            and object_id > 0
            and previous[2] == object_id
            and now - previous[0] <= VIEWPORT_DOUBLE_CLICK_SECONDS
            and (state.cursor[0] - previous[1][0]) ** 2 + (state.cursor[1] - previous[1][1]) ** 2
            <= radius * radius
        )
        if state.actions is not None:
            double_clicked = self.router.started_with_focus
        self._last_viewport_click = None if double_clicked else (now, state.cursor, object_id)
        if double_clicked and selection.focus_on_double_click:
            node = self.session.node_by_object_id(object_id)
            if node is not None:
                if self._request_node_joint_focus(node):
                    return
                self.request_node_focus(node.node_id)
        if object_id > 0 or selection.clear_on_empty:
            self.session.submit(cmd.Select(object_id))

    def _pick_at(self, cursor: tuple[float, float]) -> int:
        rect = self._viewport_rect

        if self.viewport_layers.helpers:
            helper = self.scene_entities.pick(
                self.session,
                self._camera_view(),
                rect,
                cursor,
                self.window.style_scale,
                self._model_camera_id >= 0,
            )
            if self._selectable(helper):
                return helper

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
        return node.type is not NodeType.WORLD and node.parent >= 0

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

    def _begin_viewport_panel(self) -> None:
        """Resolve the current dock layout before sizing or rendering the scene."""

        title = self.localizer.text("Viewport")
        if title != "Viewport":
            title += "###Viewport"
        # Docked scenes fill their panel; floating scenes retain the native
        # resize border. Neither acquires the ordinary panel content padding.
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.begin(title, None, imgui.WindowFlags_.no_scrollbar.value)
        imgui.pop_style_var()
        pos = imgui.get_cursor_screen_pos()
        size = imgui.get_content_region_avail()
        if not imgui.is_window_docked():
            # Derive the inset from the border, not the drawing clip: moving
            # partly off screen must not resize the scene or change its aspect.
            inset = float(math.ceil(imgui.get_style().window_border_size * 0.5))
            pos = imgui.ImVec2(pos.x + inset, pos.y)
            size = imgui.ImVec2(size.x - 2.0 * inset, size.y - inset)
        panel_position = (float(pos.x), float(pos.y))
        self._viewport_panel_position = panel_position
        self._viewport_panel_size = (max(float(size.x), 1.0), max(float(size.y), 1.0))
        self._viewport_rect = _fit_image_rect(
            panel_position,
            self._viewport_panel_size,
            self._current_viewport_render_size(),
        )

    def _draw_viewport_contents(
        self,
        preview_name: str = "",
        *,
        session_busy: bool = False,
    ) -> None:
        image = self._viewport_image
        if image is None:
            imgui.text_disabled(self.localizer.text("No viewport image is available"))
        else:
            self._viewport_rect = _fit_image_rect(
                self._viewport_panel_position,
                self._viewport_panel_size,
                (image.width, image.height),
            )
            uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
            uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
            x, y, width, height = self._viewport_rect
            imgui.set_cursor_screen_pos(imgui.ImVec2(x, y))
            # Only the docked scene covers the native one-pixel inner border.
            # Floating image bounds stop before resize grips and hover strokes.
            imgui.push_clip_rect((x, y), (x + width, y + height), False)
            imgui.image(
                self.window.viewport_texture_ref(image),
                imgui.ImVec2(width, height),
                uv0,
                uv1,
            )
            imgui.pop_clip_rect()
        x, y, w, h = self._viewport_rect
        imgui.push_clip_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), True)
        try:
            overlay = ImguiDraw2D()
            if not session_busy:
                st = self.session.perturb
                if (
                    st.active
                    and self.viewport_layers.perturbation
                    and not self.backend.caps.debug_draw
                ):
                    node = self.session.node(st.node_id)
                    center = self._node_pose(node)[0] if node is not None else st.target_pos
                    if st.mode == "translate":
                        pose = (
                            self._node_pose(node)
                            if node is not None
                            else (st.target_pos, st.target_mat)
                        )
                        draw_translation_link(
                            self._camera_view(),
                            st,
                            self._viewport_rect,
                            pose,
                            overlay,
                            self.window.style_scale,
                        )
                    else:
                        draw_fallback(
                            self._camera_view(),
                            st,
                            self._viewport_rect,
                            (imgui.get_io().mouse_pos.x, imgui.get_io().mouse_pos.y),
                            center,
                            overlay,
                            self.window.style_scale,
                        )
                if st.active and self.viewport_layers.perturbation and st.mode == "rotate":
                    node = self.session.node(st.node_id)
                    center = self._node_pose(node)[0] if node is not None else st.target_pos
                    draw_perturb_axes(
                        overlay,
                        self._camera_view(),
                        self._viewport_rect,
                        center,
                        st.target_mat,
                        self.window.style_scale,
                    )
                self.gizmo.draw_overlay(
                    self._camera_view(),
                    self._viewport_rect,
                    overlay,
                    style_scale=self.window.style_scale,
                )
                if self.viewport_layers.viewport_ui:
                    self.view_cube.draw(overlay, self.window.style_scale)
                self._draw_model_drop_overlay(overlay)
                if self.viewport_layers.viewport_ui:
                    self._draw_viewport_status(overlay)
        finally:
            imgui.pop_clip_rect()
        if not session_busy and self.viewport_layers.viewport_ui:
            self.camera_preview.draw(
                self.window,
                self._viewport_rect,
                preview_name,
                self.localizer.text,
            )
        imgui.end()
        if not session_busy and self.viewport_layers.gizmos:
            self._draw_joint_limit_controls()
            self._draw_joint_gizmo_picker()

    def _viewport_overlay_rect(
        self,
        name: str,
        size: tuple[float, float],
        default_center: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        position = (
            self.viewport_overlays.playback_position
            if name == "playback"
            else self.viewport_overlays.tool_position
        )
        if self._overlay_drag_kind == name:
            io = imgui.get_io()
            if self._overlay_drag_chord is not None and self.input_bindings.pointer_frame().held(
                self._overlay_drag_chord
            ):
                center = (
                    float(io.mouse_pos.x) - self._overlay_drag_offset[0],
                    float(io.mouse_pos.y) - self._overlay_drag_offset[1],
                )
                moving = (
                    center[0] - size[0] * 0.5,
                    center[1] - size[1] * 0.5,
                    center[0] + size[0] * 0.5,
                    center[1] + size[1] * 0.5,
                )
                position = normalized_overlay_position(self._viewport_rect, moving)
                field = "playback_position" if name == "playback" else "tool_position"
                self.set_viewport_overlays(
                    replace(self.viewport_overlays, **{field: position}), persist=False
                )
            else:
                self._overlay_drag_kind = ""
                self.set_viewport_overlays(self.viewport_overlays, persist=True)
        return positioned_overlay_rect(
            self._viewport_rect,
            size,
            position,
            default_center,
            margin=4.0 * self.window.style_scale,
        )

    def _offer_viewport_overlay_drag(
        self,
        name: str,
        rect: tuple[float, float, float, float],
    ) -> None:
        if not self.viewport_overlays.movable:
            return
        io = imgui.get_io()
        point = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        if self._overlay_drag_kind != name and not overlay_border_hit(
            point, rect, 6.0 * self.window.style_scale
        ):
            return
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_all)
        chord = self.input_bindings.pointer_match(
            PointerAction.OVERLAY_DRAG, self.input_bindings.pointer_frame(), press=True
        )
        if not self._overlay_drag_kind and chord is not None:
            self._overlay_drag_chord = chord
            center = ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5)
            self._overlay_drag_kind = name
            self._overlay_drag_offset = (point[0] - center[0], point[1] - center[1])
            self._consume_scene_pointer_until_release = True

    def _draw_playback_widget(self) -> None:
        """Draw the final playback capsule at the viewport's top center."""

        caps = self.session.adapter.caps
        if (
            not self.viewport_layers.viewport_ui
            or not caps.simulation
            or not self._has_scene_content()
        ):
            self._playback_widget_rect = None
            return
        x, y, width, _height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale * self.viewport_overlays.playback_scale,
            PLAYBACK_CHROME_SCALE,
        )
        widget_width, widget_height = playback_size(scale, self.viewport_chrome.playback_controls)
        if widget_width <= 0.0 or widget_height <= 0.0:
            self._playback_widget_rect = None
            return
        widget_rect = self._viewport_overlay_rect(
            "playback",
            (widget_width, widget_height),
            (x + width * 0.5, y + 12.0 * style_scale + widget_height * 0.5),
        )
        self._playback_widget_rect = widget_rect
        preview_rect = self.camera_preview.bounds
        if preview_rect is not None and _rectangles_overlap(widget_rect, preview_rect):
            # A tiny HiDPI viewport cannot expose two large overlays at once.
            # Keep the camera preview header reachable so it can be moved or
            # disabled instead of placing playback above its drag target.
            self._playback_widget_rect = None
            return
        clip_pad = OVERLAY_CLIP_PADDING * scale
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            self._playback_widget_rect = None
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always.value,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
        )
        imgui.push_style_var(
            imgui.StyleVar_.window_padding,
            imgui.ImVec2(0.0, 0.0),
        )
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(0.0, 0.0),
        )
        visible, _ = imgui.begin(
            f"{self.localizer.text('Playback')}###viewport_playback", None, flags
        )
        if visible:
            take_playing = self.session.state_take_playing
            paused = self.session.paused and not take_playing
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                action = draw_playback(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    playing=not paused,
                    step_enabled=paused and not take_playing,
                    previous_enabled=self.session.can_step_back,
                    recording=self.session.state_take_recording or self.recording.active,
                    record_action=(
                        "pause"
                        if self.recording.phase is RecordingPhase.RECORDING
                        else "resume"
                        if self.recording.phase is RecordingPhase.PAUSED
                        else "stop"
                    ),
                    record_tooltip=self.localizer.text(
                        "Pause Recording"
                        if self.recording.phase is RecordingPhase.RECORDING
                        else "Resume Recording"
                        if self.recording.phase is RecordingPhase.PAUSED
                        else "Cancel Recording"
                        if self.recording.phase is RecordingPhase.COUNTDOWN
                        else "Stop Recording"
                        if self.session.state_take_recording
                        else "Record Take"
                        if self._viewport_recording_mode == "take"
                        else "Record Video"
                    ),
                    record_enabled=self.recording.active
                    or (
                        self.session.adapter.caps.simulation
                        and self.session.adapter.caps.state_snapshots
                    ),
                    enabled=not self._scene_input_blocked()
                    and self.session.adapter.caps.clock_control,
                    bindings=self.input_bindings,
                    labels=self._viewport_labels,
                    control_specs=self.viewport_chrome.playback_controls,
                )
            if action and self.viewport_chrome.dispatch("playback", action):
                pass
            elif action == "toggle":
                self._toggle_playback()
            elif action == "step":
                self.session.submit(cmd.Step(1))
            elif action == "previous":
                self.session.submit(cmd.StepBack())
            elif action in ("reset", "stop"):
                self._reset_playback()
            elif action == "record":
                if self.recording.active:
                    if self.recording.phase is RecordingPhase.PAUSED:
                        self.resume_recording()
                    elif self.recording.phase is RecordingPhase.COUNTDOWN:
                        self.stop_recording()
                    else:
                        self.pause_recording()
                elif self.session.state_take_recording:
                    self.session.submit(cmd.StopStateTakeRecording())
                elif self._viewport_recording_mode == "take":
                    self.session.submit(cmd.StartStateTakeRecording())
                else:
                    self._toggle_viewport_recording()
            elif action == "recording-options":
                imgui.open_popup("viewport-recording-options")
            if imgui.is_popup_open("viewport-recording-options"):
                t = self.localizer.text
                caps = self.session.adapter.caps
                items = []
                if caps.simulation and caps.state_snapshots:
                    items.append(
                        (
                            "record-take",
                            t(
                                "Stop Recording"
                                if self.session.state_take_recording
                                else "Record Take"
                            ),
                            True,
                        )
                    )
                items.extend(
                    (
                        (
                            "record-video",
                            t("Stop Recording" if self.recording.active else "Record Video"),
                            True,
                        ),
                        None,
                        ("recording-settings", t("Recording Settings..."), True),
                    )
                )
                action = action_menu_popup("viewport-recording-options", items)
                if action == "record-take":
                    self._viewport_recording_mode = "take"
                    self.session.submit(
                        cmd.StopStateTakeRecording()
                        if self.session.state_take_recording
                        else cmd.StartStateTakeRecording()
                    )
                elif action == "record-video":
                    self._viewport_recording_mode = "video"
                    self._toggle_viewport_recording()
                elif action == "recording-settings":
                    self.panels.open_panel("Settings")
                    self.panels.get("Settings").show_category("Recording")
            self._offer_viewport_overlay_drag("playback", widget_rect)
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_tool_column_widget(self) -> None:
        """Draw the viewport tool capsule without construction geometry."""

        if not self.viewport_layers.viewport_ui or not self._has_scene_content():
            self._tool_widget_rect = None
            return
        node = self.session.selected_node
        transform = self.gizmo.evaluate_mode(self.session, node, GizmoMode.TRANSLATE)
        dimensions = self.gizmo.evaluate_mode(self.session, node, GizmoMode.DIMENSIONS)
        controls_enabled = bool(
            self.interactions.gizmo
            and self.selection_style.gizmo
            and self.viewport_layers.gizmos
            and (self.gizmo.style == "2d" or self.backend.caps.gizmo)
        )
        enabled_controls: set[str] = set()
        caps = self.session.adapter.caps
        can_arm = node is None and (caps.write_pose or caps.write_qpos or caps.scene_authoring)
        if controls_enabled and (transform.ok or can_arm):
            enabled_controls.update(("move", "rotate", "frame"))
        if controls_enabled and (dimensions.ok or (can_arm and caps.scene_authoring)):
            enabled_controls.add("dimensions")
        if enabled_controls:
            enabled_controls.add("snap")
        # A column made entirely of disabled tools is visual noise and can
        # obscure joint selection. It returns as soon as one tool is usable.
        if not enabled_controls:
            self._tool_widget_rect = None
            return
        x, y, _width, height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale * self.viewport_overlays.tool_scale,
            TOOL_CHROME_SCALE,
        )
        widget_width, widget_height = tool_column_size(scale, self.viewport_chrome.tool_groups)
        if widget_width <= 0.0 or widget_height <= 0.0:
            self._tool_widget_rect = None
            return
        clip_pad = OVERLAY_CLIP_PADDING * scale
        if height < widget_height + 120.0 * style_scale:
            self._tool_widget_rect = None
            return
        widget_rect = self._viewport_overlay_rect(
            "tools",
            (widget_width, widget_height),
            (x + 12.0 * style_scale + widget_width * 0.5, y + height * 0.5),
        )
        self._tool_widget_rect = widget_rect
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            self._tool_widget_rect = None
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        visible, _ = imgui.begin(f"{self.localizer.text('Tools')}###viewport_tools", None, flags)
        if visible:
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                action = draw_tool_column(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    mode=self.gizmo.mode if self.gizmo.enabled else "",
                    space=self.gizmo.space,
                    snap=self._snap_latched or self.gizmo.snapping,
                    enabled=not self._scene_input_blocked(),
                    enabled_controls=enabled_controls,
                    disabled_reasons={
                        "move": transform.reason,
                        "rotate": transform.reason,
                        "frame": transform.reason,
                        "dimensions": dimensions.reason,
                    },
                    bindings=self.input_bindings,
                    labels=self._viewport_labels,
                    groups=self.viewport_chrome.tool_groups,
                )
            if action and self.viewport_chrome.dispatch("tool", action):
                pass
            elif action == "move":
                self.gizmo.toggle_mode("translate")
            elif action == "rotate":
                self.gizmo.toggle_mode("rotate")
            elif action == "dimensions":
                self.gizmo.toggle_mode("dimensions")
            elif action == "frame":
                self.gizmo.toggle_space()
            elif action == "snap":
                self._snap_latched = not self._snap_latched
            self._offer_viewport_overlay_drag("tools", widget_rect)
        imgui.end()
        imgui.pop_style_var()

    def _draw_context_hint_widget(self) -> None:
        """Draw caller-defined scene hints; defaults live in the status bar."""

        if self._scene_input_blocked():
            return
        if self.model_edits.active:
            self._draw_pending_model_edits()
            return
        if (
            not self.viewport_layers.viewport_ui
            or self._scene_input_blocked()
            or not self._has_scene_content()
        ):
            return
        hints = self.tool_hints.resolve(surface="scene")
        if not hints:
            return
        x, y, width, height = self._viewport_rect
        style_scale = self.window.style_scale
        scale = viewport_chrome_scale(
            style_scale,
            self._viewport_overlay_scale,
            HINT_CHROME_SCALE,
        )
        hint_font_scale = scale / max(style_scale, 1e-6)
        imgui.push_font(None, imgui.get_font_size() * hint_font_scale)
        measure = ImguiDraw2D(imgui.get_foreground_draw_list())
        widget_width, widget_height = tool_hints_size(
            measure,
            scale,
            hints,
            labels=self._viewport_labels,
            padding=True,
        )
        clip_pad = OVERLAY_CLIP_PADDING * scale
        if widget_width > width - 24.0 * style_scale:
            content_width = max(
                0.0,
                width - 24.0 * style_scale - 2.0 * OVERLAY_GEOMETRY.hint_padding_x * scale,
            )
            hints = fitting_tool_hints(
                measure,
                scale,
                hints,
                content_width,
                labels=self._viewport_labels,
            )
            if not hints:
                imgui.pop_font()
                return
            widget_width, widget_height = tool_hints_size(
                measure,
                scale,
                hints,
                labels=self._viewport_labels,
                padding=True,
            )
        widget_rect = (
            x + (width - widget_width) * 0.5,
            y + height - widget_height - 16.0 * style_scale,
            x + (width + widget_width) * 0.5,
            y + height - 16.0 * style_scale,
        )
        host_rect = _clipped_overlay_host_rect(self._viewport_rect, widget_rect, clip_pad)
        if host_rect is None:
            imgui.pop_font()
            return
        imgui.set_next_window_pos(
            imgui.ImVec2(host_rect[0], host_rect[1]),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            imgui.ImVec2(host_rect[2], host_rect[3]),
            imgui.Cond_.always,
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_background.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_inputs.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        visible, _ = imgui.begin(f"{self.localizer.text('Hints')}###viewport_hints", None, flags)
        if visible:
            with _clipped_overlay_draw(self._viewport_rect) as draw:
                draw_scene_tool_hints(
                    draw,
                    (widget_rect[0], widget_rect[1]),
                    self.theme,
                    scale,
                    hints,
                    labels=self._viewport_labels,
                    size=(widget_width, widget_height),
                    pixel_size=self.window.pixels_to_points(1.0),
                )
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_font()

    def _context_tool_hint_variant(self) -> str:
        """Return the active input grammar independently of its surface."""

        state = self._state
        if state.ctrl or self.session.perturb.active:
            return "perturb"
        if self.gizmo.using:
            return "dragging"
        if not state.has_selection or not state.gizmo_available:
            return "camera"
        return "ready"

    def _update_status_context(self, ctx: PanelContext) -> None:
        """Latch status ownership on clicks, never on pointer travel or scrolling."""

        if not self._consume_scene_pointer_until_release and any(
            imgui.is_mouse_clicked(button) for button in (0, 1, 2)
        ):
            context = imgui.get_current_context()
            hovered = context.hovered_window
            if hovered is not None:
                owner = hovered.root_window
                # Dock tabs belong to the dock host, not the panel window.
                # Use ImGui's hit-tested tab ID (including clipping/scrolling),
                # rather than its navigation focus, which updates a frame later.
                if context.hovered_id:
                    owner = next(
                        (
                            window
                            for window in context.windows
                            if window.tab_id == context.hovered_id
                            and window.dock_node is not None
                            and window.dock_node.host_window is not None
                            and window.dock_node.host_window.id_ == hovered.id_
                        ),
                        owner,
                    )
                name = str(owner.name).rsplit("###", 1)[-1]
                if name == "Viewport":
                    self._status_panel = "Viewport"
                elif name in ctx.status_hints_by_panel:
                    self._status_panel = name
        if self._status_panel != "Viewport" and self._status_panel not in ctx.status_hints_by_panel:
            self._status_panel = "Viewport"
        self._panel_status_hints = ctx.status_hints_by_panel.get(self._status_panel, ())

    def _status_tool_hints(self, *, loading: bool) -> tuple[ToolHint, ...]:
        if loading:
            return ()
        edit = self._precise_gizmo_edit
        if edit is not None:
            defaults = precise_input_status_hints(edit, self.localizer.text)
            return self.tool_hints.resolve(defaults, surface="status")
        if self._scene_input_blocked():
            return self.tool_hints.resolve((), surface="status")
        if self._gizmo_hint_hover.visible:
            defaults = default_tool_hints(
                "ready_minimal",
                self.input_bindings,
                self._viewport_labels,
            )
        elif self._status_panel != "Viewport":
            defaults = self._panel_status_hints
        elif self._has_scene_content():
            defaults = tuple(
                hint
                for hint in default_tool_hints(
                    self._context_tool_hint_variant(),
                    self.input_bindings,
                    self._viewport_labels,
                )
                if hint.hint_id != "gizmo.type_value"
            )
        else:
            defaults = ()
        # Selection is application state, not part of a panel's hover/gesture
        # grammar. Compose its action before the clicked panel's own hints.
        if self._selection_clear_enabled() and not any(
            hint.kind == "key" and hint.control == "Esc" for hint in defaults
        ):
            defaults = (
                ToolHint(
                    "key",
                    "Esc",
                    self._viewport_labels.clear_selection,
                    hint_id="selection.clear",
                ),
                *defaults,
            )
        if self._status_panel == "Viewport" and self.session.can_step_back:
            defaults = (
                ToolHint(
                    "key",
                    self.input_bindings.label(InputAction.STEP_BACK),
                    self._viewport_labels.rewind,
                    hint_id="playback.previous",
                ),
                *defaults,
            )
        if not (self.interactions.perturb and self.session.adapter.caps.perturb):
            defaults = tuple(hint for hint in defaults if not hint.hint_id.startswith("perturb"))
        return self.tool_hints.resolve(defaults, surface="status")

    def _selection_clear_enabled(self) -> bool:
        """Allow navigation alongside selection; protect edits that own the target."""

        if imgui.is_any_item_active():
            context = imgui.get_current_context()
            window = context.active_id_window
            # ImGui assigns a window's MoveId to an empty-space left press,
            # even when the docked window is not moving. That focus capture
            # is not a widget edit and must not suppress orbit/Shift-pan Esc.
            if (
                window is None
                or context.active_id != window.move_id
                or context.moving_window is not None
            ):
                return False
        return bool(
            self.session.paused
            and not self.session.state_take_playing
            and self.session.selected_node is not None
            and not (
                self.router.held and self.router.claim in (gs.Claim.OBJECT_GIZMO, gs.Claim.PERTURB)
            )
            and not self.session.perturb.active
            and not self.gizmo.using
            and not self.gizmo.keyboard_using
        )
