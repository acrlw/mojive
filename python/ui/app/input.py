"""App: input."""

from __future__ import annotations

import time

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.capture import CaptureSurface
from mojive.interaction.input import InputClaim, InputContext, physical_ctrl_super
from mojive.ui import gestures as gs
from mojive.ui.input_bindings import InputAction
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.viewport_widgets import (
    overlay_border_hit,
)

from .support import (
    _NO_INPUT_CLAIM,
    STEP_BACK_REPEAT_DELAY_SECONDS,
    STEP_BACK_REPEAT_RATE_SECONDS,
    Keys,
    _model_filters,
)


class _Input:
    """Private input methods of ViewerApp; state belongs to its owner."""

    def _scene_input_blocked(self) -> bool:
        """Return whether UI owns the whole interaction frame.

        This check runs before gesture classification so a modal cannot leave
        an old camera, gizmo, or perturb claim alive underneath it.
        """

        pending_prompt = bool(
            self._pending_document_action is not None
            or self._pending_pose_save is not None
            or self._show_model_load_error
            or self._open_resource_repair_popup
            or self._open_rename_popup
            or self._precise_gizmo_edit is not None
        )
        native_dialog = any(
            dialog is not None
            for dialog in (
                self._model_dialog,
                self._scene_dialog,
                self._resource_dialog,
                self._texture_dialog,
                self._geometry_resource_dialog,
                self._model_asset_dialog,
                self._resource_repair_dialog,
            )
        )
        popup_flags = imgui.PopupFlags_.any_popup_id.value | imgui.PopupFlags_.any_popup_level.value
        any_popup = self._popup_owned_frame or bool(imgui.is_popup_open("", popup_flags))
        context = imgui.get_current_context()
        native_activation = context is not None and context.nav_activate_id != 0
        io = imgui.get_io()
        mouse_pos = getattr(io, "mouse_pos", None)
        cursor = (
            (float(mouse_pos.x), float(mouse_pos.y))
            if mouse_pos is not None
            else (float("inf"), float("inf"))
        )
        status_bounds = getattr(self, "_status_path_bounds", None)
        status_reveal = bool(
            (getattr(io, "key_ctrl", False) or getattr(io, "key_super", False))
            and status_bounds is not None
            and status_bounds[0] <= cursor[0] <= status_bounds[2]
            and status_bounds[1] <= cursor[1] <= status_bounds[3]
        )
        overlay_config = getattr(self, "viewport_overlays", None)
        # Read the frame snapshot already returned by get_io(). This avoids a
        # second global-context query and keeps headless embedding/tests safe
        # while an ImGui context is being created or torn down.
        pointer_engaged = any(bool(value) for value in getattr(io, "mouse_down", ())[:3])
        overlay_border = bool(
            overlay_config is not None
            and overlay_config.movable
            # An already-owned scene gesture keeps capture until release;
            # merely crossing a capsule border must not interrupt a drag.
            and not getattr(self.gizmo, "using", False)
            and pointer_engaged
            and any(
                rect is not None and overlay_border_hit(cursor, rect, 6.0 * self.window.style_scale)
                for rect in (
                    getattr(self, "_playback_widget_rect", None),
                    getattr(self, "_tool_widget_rect", None),
                )
            )
        )
        return bool(
            pending_prompt
            or native_dialog
            or any_popup
            or native_activation
            or io.want_text_input
            or self._consume_scene_pointer_until_release
            or getattr(self, "_overlay_drag_kind", "")
            or overlay_border
            or status_reveal
        )

    def _poll_input_handler(self) -> None:
        """Offer input to the embedding application before built-in tools poll it."""

        io = imgui.get_io()
        rect = self._viewport_rect
        cursor = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        inside = bool(
            rect[0] <= cursor[0] <= rect[0] + rect[2] and rect[1] <= cursor[1] <= rect[1] + rect[3]
        )
        focused_before_frame = self._viewport_focused
        self._viewport_focused = bool(imgui.is_window_focused())
        if imgui.is_mouse_clicked(0) and inside:
            self._selection_press_started_focused = focused_before_frame
        self._input_claim = _NO_INPUT_CLAIM
        if self._input_handler is None:
            return
        blocked = self._scene_input_blocked() or self._native_window_input_owned()
        context = InputContext(
            viewport_hovered=inside and not blocked,
            viewport_focused=self._viewport_focused,
            blocked=blocked,
            cursor=cursor,
            delta=(float(io.mouse_delta.x), float(io.mouse_delta.y)),
            wheel=float(io.mouse_wheel),
        )
        claim = self._input_handler(context)
        if claim is None:
            return
        if not isinstance(claim, InputClaim):
            raise TypeError("input handler must return InputClaim or None")
        self._input_claim = claim

    def _poll_application_shortcuts(self) -> None:
        """Dispatch editor shortcuts after the application has claimed this frame."""
        io = imgui.get_io()
        if self._scene_input_blocked() or imgui.is_any_item_active():
            return
        claim = self._input_claim
        if claim.keyboard:
            return
        ctrl, super_key = physical_ctrl_super(io)
        modifier = ctrl or super_key
        if (ctrl and claim.claims_key("ctrl")) or (super_key and claim.claims_key("super")):
            return
        if io.key_shift and claim.claims_key("shift"):
            return

        def pressed(key: str) -> bool:
            return not claim.claims_key(key) and imgui.is_key_pressed(
                getattr(imgui.Key, key), False
            )

        caps = self.session.adapter.caps
        if modifier:
            if caps.edit_history and pressed("z"):
                self.session.submit(cmd.Redo() if io.key_shift else cmd.Undo())
            if caps.scene_files:
                if pressed("n"):
                    self._request_document_action("new_scene")
                if pressed("o") and not io.key_shift:
                    self._open_scene_dialog("open")
                if pressed("s"):
                    if io.key_shift or self.session.asset_path is None:
                        self._open_scene_dialog("save")
                    else:
                        self._request_scene_save(self.session.asset_path)
            elif _model_filters(caps) and pressed("o") and not io.key_shift:
                self._open_model_dialog()
            if caps.reload and io.key_shift and pressed("o"):
                self._queue_model_load("reload", self.session.asset_path)
            if pressed("comma"):
                self.panels.open_panel("Settings")
            if io.key_shift and pressed("p"):
                self.request_capture(surface=CaptureSurface.SCENE)
            if io.key_shift and pressed("r"):
                self._toggle_viewport_recording()
            if pressed("q"):
                self._request_document_action("quit")
        editable_selected = bool(
            self._selected_entity() or self._selected_model_element() is not None
        )
        if caps.scene_authoring and editable_selected:
            if modifier and pressed("d"):
                self._duplicate_selected()
            if not modifier and pressed("delete"):
                self._remove_selected()
            if not modifier and pressed("f2"):
                self._request_selected_rename()

    def _poll_keys(self) -> Keys:
        io = imgui.get_io()
        if self._scene_input_blocked():
            return Keys()
        if self.interactions.panel_shortcuts:
            self.panels.poll_shortcuts(
                claimed_keys=self._input_claim.keys,
                keyboard_claimed=self._input_claim.keyboard,
            )

        clear_selection = bool(
            self.interactions.selection.clear_with_escape
            and not self._input_claim.claims_key("escape")
            and self._selection_clear_enabled()
            and imgui.is_key_pressed(imgui.Key.escape, False)
        )
        bindings = self.input_bindings
        ctrl, super_key = physical_ctrl_super(io)

        def available(action: InputAction) -> bool:
            key_id = bindings.key_id(action)
            # Editor chords reserve their letter keys. A modifier explicitly
            # assigned to a viewport action still works as a standalone key.
            if super_key or (ctrl and key_id != "ctrl"):
                return False
            return not self._input_claim.claims_key(key_id)

        def down(action: InputAction) -> float:
            return 1.0 if available(action) and bindings.down(action) else 0.0

        axis = next(
            (
                index
                for index, action in enumerate(
                    (InputAction.AXIS_X, InputAction.AXIS_Y, InputAction.AXIS_Z)
                )
                if self.interactions.gizmo and available(action) and bindings.down(action)
            ),
            -1,
        )
        return Keys(
            fly=(
                down(InputAction.FLY_FORWARD) - down(InputAction.FLY_BACK),
                down(InputAction.FLY_RIGHT) - down(InputAction.FLY_LEFT),
                down(InputAction.FLY_UP) - down(InputAction.FLY_DOWN),
            )
            if self.interactions.camera.fly
            else (0.0, 0.0, 0.0),
            toggle_pause=bool(
                self.interactions.playback_shortcuts
                and available(InputAction.TOGGLE_PAUSE)
                and bindings.pressed(InputAction.TOGGLE_PAUSE)
            ),
            step_back_count=(
                bindings.press_count(
                    InputAction.STEP_BACK,
                    delay=STEP_BACK_REPEAT_DELAY_SECONDS,
                    rate=STEP_BACK_REPEAT_RATE_SECONDS,
                )
                if self.interactions.playback_shortcuts
                and available(InputAction.STEP_BACK)
                and self.session.can_step_back
                else 0
            ),
            clear_selection=clear_selection,
            frame_scene=available(InputAction.FRAME_SCENE)
            and bindings.pressed(InputAction.FRAME_SCENE),
            gizmo_translate=self.interactions.gizmo
            and available(InputAction.GIZMO_TRANSLATE)
            and bindings.pressed(InputAction.GIZMO_TRANSLATE),
            gizmo_rotate=self.interactions.gizmo
            and available(InputAction.GIZMO_ROTATE)
            and bindings.pressed(InputAction.GIZMO_ROTATE),
            gizmo_dimensions=self.interactions.gizmo
            and available(InputAction.GIZMO_DIMENSIONS)
            and bindings.pressed(InputAction.GIZMO_DIMENSIONS),
            gizmo_space=self.interactions.gizmo
            and available(InputAction.GIZMO_SPACE)
            and bindings.pressed(InputAction.GIZMO_SPACE),
            gizmo_axis=axis,
        )

    def _input_state(self) -> gs.InputState:
        io = imgui.get_io()
        blocked = self._scene_input_blocked()
        cursor = (float(io.mouse_pos.x), float(io.mouse_pos.y))
        rect = self._viewport_rect
        inside = (
            rect[0] <= cursor[0] <= rect[0] + rect[2] and rect[1] <= cursor[1] <= rect[1] + rect[3]
        )
        hovered_window = imgui.get_current_context().hovered_window
        hovered_name = None if hovered_window is None else str(hovered_window.name)
        viewport_window_busy = self._native_window_input_owned()
        over_viewport = (
            gs.viewport_input_allowed(inside, hovered_name)
            and not viewport_window_busy
            and not blocked
        )
        view = self._camera_view()
        hovered_ball = self.view_cube.update(
            view,
            rect,
            cursor,
            self.window.style_scale,
            enabled=over_viewport
            and self.interactions.camera.view_cube
            and self.viewport_layers.viewport_ui
            and not self._input_claim.pointer,
        )
        self.gizmo.update_hover(
            self.session,
            view,
            rect,
            cursor,
            enabled=over_viewport
            and self.interactions.gizmo
            and self.selection_style.gizmo
            and self.viewport_layers.gizmos
            and not self._input_claim.pointer
            and not self._viewing_selected_camera(),
            style_scale=self.window.style_scale,
        )
        self._gizmo_hint_hover.update(
            over_viewport
            and self.gizmo.precise_input_hovered
            and not any(imgui.is_mouse_down(button) for button in range(3)),
            time.monotonic(),
        )
        node = self.session.selected_node
        pointer = self.input_bindings.pointer_frame(self._input_claim)
        actions = frozenset(
            action for action, _chord in self.input_bindings.pointer_matches(pointer)
        )
        if actions & {PointerAction.PERTURB_TRANSLATE, PointerAction.PERTURB_ROTATE}:
            actions -= {PointerAction.GIZMO, PointerAction.GIZMO_VALUE}
            if not (self.interactions.perturb and self.session.adapter.caps.perturb):
                actions -= {PointerAction.PERTURB_TRANSLATE, PointerAction.PERTURB_ROTATE}

        return gs.InputState(
            actions=actions,
            blocked=blocked,
            left=not self._input_claim.claims_button(0) and imgui.is_mouse_down(0),
            right=not self._input_claim.claims_button(1) and imgui.is_mouse_down(1),
            middle=not self._input_claim.claims_button(2) and imgui.is_mouse_down(2),
            ctrl=self.interactions.perturb
            and self.session.adapter.caps.perturb
            and (
                PointerAction.PERTURB_TRANSLATE in actions
                or PointerAction.PERTURB_ROTATE in actions
            ),
            shift=self.interactions.gizmo
            and not self._input_claim.claims_key(self.input_bindings.key_id(InputAction.SNAP))
            and self.input_bindings.down(InputAction.SNAP),
            alt=not self._input_claim.claims_key("alt") and io.key_alt,
            wheel=0.0
            if self._input_claim.wheel or self._input_claim.pointer
            else float(io.mouse_wheel),
            cursor=cursor,
            delta=(float(io.mouse_delta.x), float(io.mouse_delta.y)),
            over_viewport=over_viewport,
            over_view_cube=over_viewport and hovered_ball is not None,
            gizmo_available=self.interactions.gizmo
            and self.selection_style.gizmo
            and self.viewport_layers.gizmos
            and (self.gizmo.style == "2d" or self.backend.caps.gizmo)
            and self.gizmo.last_verdict.ok,
            gizmo_hovered=over_viewport and self.gizmo.hovered,
            has_selection=node is not None,
            perturbing=self.session.perturb.active,
            ui_wants_mouse=blocked
            or viewport_window_busy
            or (io.want_capture_mouse and not over_viewport),
        )

    @staticmethod
    def _native_window_input_owned() -> bool:
        """Honor native controls and dock splitters before claiming a scene drag."""

        context = imgui.get_current_context()
        window = context.current_window
        if window is None:
            return False
        owner = context.active_id_window
        if context.active_id and owner is not None and owner.id_ != window.id_:
            return True
        if context.moving_window is not None:
            return True
        return bool(
            int(window.resize_border_held) >= 0
            or (int(window.resize_border_hovered) >= 0 and imgui.is_mouse_down(0))
        )

    def _claim_gesture(self, state: gs.InputState) -> gs.Claim:
        return self.router.update(state)

    def apply_keys(self, keys: Keys) -> None:
        if keys.toggle_pause:
            self._toggle_playback()
        for _ in range(keys.step_back_count):
            if not self.session.submit(cmd.StepBack()):
                break
        if keys.clear_selection:
            self.session.submit(cmd.Select(0))
            hierarchy = self.panels.get("Hierarchy")
            if hierarchy is not None:
                clear_selection = getattr(hierarchy, "clear_selection", None)
                if clear_selection is not None:
                    clear_selection()
            self.gizmo.cancel()
            self.router.abort()
            # Esc may be pressed during orbit/pan, including before it crosses
            # click slop. Consume the rest of that press so its release cannot
            # restart a gesture or pick the just-cleared object again.
            self._consume_scene_pointer_until_release = any(
                imgui.is_mouse_down(button) for button in range(3)
            )
        if keys.gizmo_translate:
            self.gizmo.toggle_mode("translate")
        if keys.gizmo_rotate:
            self.gizmo.toggle_mode("rotate")
        if keys.gizmo_dimensions:
            self.gizmo.toggle_mode("dimensions")
        if keys.gizmo_space:
            self.gizmo.toggle_space()
