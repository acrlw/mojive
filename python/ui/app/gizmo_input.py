"""App: gizmo input."""

from __future__ import annotations

import time

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import NodeType
from mojive.interaction.gizmo import axis_active_color, axis_hover_color
from mojive.ui import gestures as gs
from mojive.ui.compound_fields import borderless_numeric_input, draw_joined_field_frame
from mojive.ui.gizmo import JointLimitHit, PreciseGizmoInput
from mojive.ui.panels import (
    padded_selectable,
    segmented_control,
)
from mojive.ui.pointer_bindings import PointerAction

from .support import (
    PRECISE_GIZMO_WIDTH_PT,
    Keys,
    _clipped_foreground_overlay_draw,
    _middle_elide_text,
    _toggle_angle_input,
)


class _GizmoInput:
    """Private gizmo input methods of ViewerApp; state belongs to its owner."""

    def _poll_gizmo(self, state: gs.InputState, keys: Keys) -> None:
        if (
            not self.interactions.gizmo
            or not self.selection_style.gizmo
            or not self.viewport_layers.gizmos
        ):
            self.gizmo.cancel()
            return
        if state.blocked:
            self.gizmo.cancel()
            return
        if self._precise_gizmo_edit is not None:
            self.gizmo.cancel()
            return
        if self._viewing_selected_camera():
            self.gizmo.keyboard_interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                -1,
                style_scale=self.window.style_scale,
            )
            self.gizmo.interact(
                self.session,
                self._camera_view(),
                self._viewport_rect,
                state.cursor,
                claimed=False,
                left_down=state.any_button and self.router.wants_gizmo(),
                released=self.router.released,
                style_scale=self.window.style_scale,
            )
            return
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
                snap=state.shift or self._snap_latched,
                style_scale=self.window.style_scale,
            )
            return
        if state.gizmo_hovered and state.action(
            PointerAction.GIZMO_VALUE, imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
        ):
            edit = self.gizmo.precise_input(self.session)
            if edit is not None:
                self.gizmo.cancel()
                self.router.abort()
                self._begin_precise_gizmo_input(edit)
                return
        self.gizmo.interact(
            self.session,
            self._camera_view(),
            self._viewport_rect,
            state.cursor,
            claimed=self.router.wants_gizmo(),
            left_down=state.any_button and self.router.wants_gizmo(),
            released=self.router.released,
            snap=state.shift or self._snap_latched,
            style_scale=self.window.style_scale,
        )

    def _begin_precise_gizmo_input(self, edit: PreciseGizmoInput) -> None:
        self._precise_gizmo_edit = edit
        if not self.gizmo.remember_precise_input_choices:
            self._precise_gizmo_absolute = False
            self._precise_gizmo_angle_unit = "degrees"
        else:
            self._precise_gizmo_absolute = bool(
                self._precise_gizmo_preferred_absolute and edit.absolute_value is not None
            )
        self._precise_gizmo_value = (
            self._precise_gizmo_reference(edit) if self._precise_gizmo_absolute else 0.0
        )
        self._precise_gizmo_error = ""
        self._open_precise_gizmo_popup = True

    def _draw_precise_gizmo_popup(self) -> None:
        edit = self._precise_gizmo_edit
        if edit is None:
            return
        scale = self.window.style_scale
        just_opened = self._open_precise_gizmo_popup
        popup_name = f"{self.localizer.text('Type value')}###precise_gizmo_input"
        if just_opened:
            x, y, width, height = self._viewport_rect
            mouse = imgui.get_io().mouse_pos
            window_width = PRECISE_GIZMO_WIDTH_PT * scale
            estimated_height = 118.0 * scale
            min_x = x + 8.0 * scale
            min_y = y + 8.0 * scale
            max_x = max(min_x, x + width - window_width - 8.0 * scale)
            max_y = max(min_y, y + height - estimated_height - 8.0 * scale)
            imgui.set_next_window_pos(
                imgui.ImVec2(
                    min(max(min_x, mouse.x + 12.0 * scale), max_x),
                    min(max(min_y, mouse.y + 12.0 * scale), max_y),
                ),
                imgui.Cond_.always,
            )
            imgui.set_next_window_focus()
            imgui.open_popup(popup_name)
            self._open_precise_gizmo_popup = False
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(PRECISE_GIZMO_WIDTH_PT * scale, 0.0),
            imgui.ImVec2(
                PRECISE_GIZMO_WIDTH_PT * scale,
                float(np.finfo(np.float32).max),
            ),
        )
        visible = imgui.begin_popup(
            popup_name,
            imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_scroll_with_mouse.value
            | imgui.WindowFlags_.no_saved_settings.value,
        )
        if not visible:
            if not just_opened:
                self._consume_scene_pointer_until_release = bool(
                    self._consume_scene_pointer_until_release
                    or imgui.is_mouse_down(imgui.MouseButton_.left)
                    or imgui.is_mouse_down(imgui.MouseButton_.right)
                    or imgui.is_mouse_down(imgui.MouseButton_.middle)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.left)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.right)
                    or imgui.is_mouse_clicked(imgui.MouseButton_.middle)
                )
                self.router.abort()
                self.gizmo.cancel()
                self._precise_gizmo_edit = None
                self._precise_gizmo_error = ""
            return
        imgui.set_scroll_x(0.0)
        appearing = imgui.is_window_appearing()

        angular = edit.unit == "°"
        unit_shortcut = angular and imgui.is_key_pressed(imgui.Key.u, False)
        if unit_shortcut:
            self._toggle_precise_gizmo_angle_unit()
            # Rebuild InputScalar's private edit buffer from the converted
            # value. Its active text state otherwise keeps the pre-conversion
            # string even though the numeric model has changed.
            imgui.internal.clear_active_id()
        unit = "rad" if angular and self._precise_gizmo_angle_unit == "radians" else edit.unit
        title = f"{self.localizer.text(edit.action)} {self.localizer.text(edit.label)}"
        # Some popup placements preserve a negative horizontal cursor offset
        # from the activating item. Clamp the first line to the popup's own
        # content padding so the beginning of "Rotate" cannot be clipped.
        imgui.set_cursor_pos_x(
            max(
                float(imgui.get_cursor_pos_x()),
                float(imgui.get_style().window_padding.x),
            )
        )
        title_width = max(1.0, float(imgui.get_content_region_avail().x))
        shown_title = _middle_elide_text(
            title,
            title_width,
            lambda value: float(imgui.calc_text_size(value).x),
        )
        imgui.text(shown_title)
        if shown_title != title:
            imgui.set_item_tooltip(title)
        imgui.separator()
        modes = (
            (
                self.localizer.text("Relative"),
                self.localizer.text("Absolute"),
            )
            if edit.absolute_value is not None
            else (self.localizer.text("Relative"),)
        )
        mode_index = 1 if self._precise_gizmo_absolute and len(modes) > 1 else 0
        if len(modes) > 1:
            imgui.text_disabled(self.localizer.text("Mode"))
            next_mode = segmented_control(
                "precise-gizmo-mode",
                modes,
                mode_index,
                theme=self.theme,
            )
        else:
            next_mode = 0
        if next_mode != mode_index:
            mode_index = next_mode
            self._set_precise_gizmo_absolute(edit, bool(mode_index))
        imgui.spacing()
        unit_width = 82.0 * scale if angular else float(imgui.calc_text_size(unit).x)
        if not angular:
            unit_width += 2.0 * float(imgui.get_style().frame_padding.x)
        input_width = max(
            72.0 * scale,
            float(imgui.get_content_region_avail().x)
            - (float(imgui.get_style().item_spacing.x) if angular else 0.0)
            - unit_width,
        )
        if appearing or unit_shortcut:
            imgui.set_keyboard_focus_here()
        draw_list = imgui.get_window_draw_list()
        splitter = imgui.ImDrawListSplitter()
        if not angular:
            splitter.split(draw_list, 2)
            splitter.set_current_channel(draw_list, 1)
            for slot in (
                imgui.Col_.frame_bg,
                imgui.Col_.frame_bg_hovered,
                imgui.Col_.frame_bg_active,
            ):
                imgui.push_style_color(slot, (0, 0, 0, 0))
        imgui.set_next_item_width(input_width)
        with borderless_numeric_input():
            submitted, self._precise_gizmo_value = imgui.input_double(
                "##precise_gizmo_value",
                self._precise_gizmo_value,
                0.0,
                0.0,
                "%.6f" if edit.unit == "m" or unit == "rad" else "%.3f",
                imgui.InputTextFlags_.enter_returns_true.value
                | imgui.InputTextFlags_.auto_select_all.value
                | imgui.InputTextFlags_.chars_scientific.value,
            )
        field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        if not angular:
            imgui.pop_style_color(3)
        imgui.same_line(0, imgui.get_style().item_spacing.x if angular else 0)
        if angular:
            next_unit = segmented_control(
                "precise-gizmo-angle-unit",
                ("deg", "rad"),
                1 if self._precise_gizmo_angle_unit == "radians" else 0,
                width=unit_width,
                theme=self.theme,
            )
            if next_unit != (1 if self._precise_gizmo_angle_unit == "radians" else 0):
                self._toggle_precise_gizmo_angle_unit()
        else:
            badge_lo = imgui.get_cursor_screen_pos()
            imgui.dummy((unit_width, imgui.get_frame_height()))
            badge_hi = imgui.get_item_rect_max()
            size = imgui.calc_text_size(unit)
            draw_list.add_text(
                (
                    badge_lo.x + (unit_width - size.x) * 0.5,
                    badge_lo.y + (imgui.get_frame_height() - size.y) * 0.5,
                ),
                imgui.color_convert_float4_to_u32(imgui.ImVec4(*self.theme.text_disabled)),
                unit,
            )
            splitter.set_current_channel(draw_list, 0)
            draw_joined_field_frame(
                draw_list,
                badge_lo,
                badge_hi,
                field_lo,
                field_hi,
                badge_color=self.theme.bg_child,
                field_color=self.theme.bg_frame,
                rounding=imgui.get_style().frame_rounding,
            )
            splitter.merge(draw_list)
        if self._precise_gizmo_error:
            imgui.spacing()
            imgui.text_wrapped(self._precise_gizmo_error)
            if imgui.small_button(f"{self.localizer.text('Copy error')}##precise-gizmo"):
                imgui.set_clipboard_text(self._precise_gizmo_error)
        submit_requested = submitted or imgui.is_key_pressed(imgui.Key.enter, False)
        cancel = imgui.is_key_pressed(imgui.Key.escape, False)
        if submit_requested:
            value = self._precise_gizmo_value
            if angular and self._precise_gizmo_angle_unit == "radians":
                value = float(np.degrees(value))
            result = self.gizmo.apply_precise_value(
                self.session,
                self._camera_view(),
                edit,
                value,
                absolute=self._precise_gizmo_absolute,
            )
            if result.ok:
                imgui.close_current_popup()
                self._precise_gizmo_edit = None
                self._precise_gizmo_error = ""
            else:
                self._precise_gizmo_error = result.message
        elif cancel:
            imgui.close_current_popup()
            self._precise_gizmo_edit = None
            self._precise_gizmo_error = ""
        imgui.end_popup()

    def _finish_consumed_scene_pointer(self) -> None:
        """Release a consumed pointer gesture after one clean release frame."""

        if not self._consume_scene_pointer_until_release:
            return
        if any(imgui.is_mouse_down(button) for button in range(3)):
            return
        self._consume_scene_pointer_until_release = False

    def _precise_gizmo_reference(self, edit: PreciseGizmoInput) -> float:
        value = float(edit.absolute_value or 0.0)
        if edit.unit == "°" and self._precise_gizmo_angle_unit == "radians":
            return float(np.radians(value))
        return value

    def _set_precise_gizmo_absolute(self, edit: PreciseGizmoInput, absolute: bool) -> None:
        absolute = bool(absolute and edit.absolute_value is not None)
        if absolute == self._precise_gizmo_absolute:
            return
        reference = self._precise_gizmo_reference(edit)
        self._precise_gizmo_value += reference if absolute else -reference
        self._precise_gizmo_absolute = absolute
        self._precise_gizmo_preferred_absolute = absolute
        self._persist_precise_gizmo_choices()

    def _toggle_precise_gizmo_angle_unit(self) -> None:
        self._precise_gizmo_value, self._precise_gizmo_angle_unit = _toggle_angle_input(
            self._precise_gizmo_value,
            self._precise_gizmo_angle_unit,
        )
        self._persist_precise_gizmo_choices()

    def _persist_precise_gizmo_choices(self) -> None:
        if not self.gizmo.remember_precise_input_choices:
            return
        self.localizer.set_preferences(
            {
                "precise_gizmo_absolute": self._precise_gizmo_preferred_absolute,
                "precise_gizmo_angle_unit": self._precise_gizmo_angle_unit,
            }
        )

    def _draw_joint_limit_controls(self) -> None:
        """Make MIN/MAX ticks direct controls and reveal read-only values on dwell."""

        hits = self.gizmo.joint_limit_hits
        now = time.monotonic()
        if not self.session.paused or not hits:
            self._joint_limit_hover.reset()
            return
        hovered_hit = self.gizmo.hovered_joint_limit
        active_hit = self.gizmo.active_joint_limit
        feedback_hit = active_hit or hovered_hit
        if feedback_hit is not None:
            self._draw_joint_limit_feedback(
                feedback_hit,
                hovered=hovered_hit is not None,
                active=active_hit is not None,
            )

        keys = tuple(self._joint_limit_key(hit) for hit in hits)
        visible_key = self._joint_limit_hover.update(
            self._joint_limit_key(hovered_hit) if hovered_hit is not None else None,
            keys,
            now,
        )
        visible_hit = next(
            (hit for hit in hits if self._joint_limit_key(hit) == visible_key),
            None,
        )
        if visible_hit is not None:
            self.gizmo.reveal_joint_precision(visible_hit, now=now)
            with _clipped_foreground_overlay_draw(self._viewport_rect) as draw:
                self.gizmo.draw_joint_limit_label(draw, visible_hit, self.window.style_scale)

    @staticmethod
    def _joint_limit_key(hit: JointLimitHit) -> tuple[int, int, str]:
        return hit.joint_id, hit.qpos_adr, hit.label[:3]

    def _draw_joint_limit_feedback(
        self,
        hit: JointLimitHit,
        *,
        hovered: bool,
        active: bool,
    ) -> None:
        """Repaint one endpoint tick with its pointer interaction state."""

        color = (
            axis_active_color(hit.semantic_color)
            if active
            else axis_hover_color(hit.semantic_color)
            if hovered
            else hit.semantic_color
        )
        with _clipped_foreground_overlay_draw(self._viewport_rect) as draw:
            draw.line(
                hit.tick_start,
                hit.tick_end,
                color,
                hit.tick_width,
                cap=hit.tick_cap,
            )

    def _draw_joint_gizmo_picker(self) -> None:
        """Draw a movable chooser near the click that selected a multi-joint link."""

        if not self.session.paused:
            return
        if not self.gizmo.enabled:
            return
        joints = self.gizmo.joint_choices(self.session)
        node = self.session.selected_node
        if not joints or node is None:
            self._joint_picker_node_id = -1
            return
        x, y, width, height = self._viewport_rect
        scale = self.window.style_scale
        style = imgui.get_style()
        labels = [f"{joint.name or f'joint {joint.joint_id}'}  ({joint.type})" for joint in joints]
        content_width = max(imgui.calc_text_size(label).x for label in labels)
        picker_width = (
            max(
                imgui.calc_text_size(node.name).x,
                content_width + 2.0 * style.frame_padding.x,
            )
            + 2.0 * style.window_padding.x
        )
        picker_height = imgui.get_frame_height() * (len(joints) + 1) + 2.0 * style.window_padding.y
        if node.node_id != self._joint_picker_node_id:
            mouse = imgui.get_io().mouse_pos
            inside = x <= mouse.x <= x + width and y <= mouse.y <= y + height
            desired_x = mouse.x + 14.0 * scale if inside else x + 18.0 * scale
            desired_y = mouse.y + 14.0 * scale if inside else y + 18.0 * scale
            min_x = x + 8.0 * scale
            min_y = y + 8.0 * scale
            max_x = max(min_x, x + width - picker_width - 8.0 * scale)
            max_y = max(min_y, y + height - picker_height - 8.0 * scale)
            imgui.set_next_window_pos(
                imgui.ImVec2(
                    min(max(min_x, desired_x), max_x),
                    min(max(min_y, desired_y), max_y),
                ),
                imgui.Cond_.always,
            )
            self._joint_picker_node_id = node.node_id
        imgui.set_next_window_size(imgui.ImVec2(picker_width, 0.0))
        imgui.set_next_window_bg_alpha(0.92)
        flags = (
            imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_collapse.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        visible, _ = imgui.begin(
            f"{node.name}###viewport_joint_gizmo",
            None,
            flags,
        )
        if visible:
            selected = self.gizmo.selected_joint_id(node.body_index)
            for joint, label in zip(joints, labels, strict=True):
                supported = joint.type in ("hinge", "slide", "ball")
                imgui.begin_disabled(not supported)
                clicked, _selected = padded_selectable(
                    f"{label}##viewport-joint-{joint.joint_id}",
                    selected == joint.joint_id,
                )
                if clicked:
                    self.gizmo.select_joint(node.body_index, joint.joint_id)
                if (
                    supported
                    and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
                    and self.input_bindings.pointer_match(
                        PointerAction.PANEL_FOCUS, self.input_bindings.pointer_frame(), press=True
                    )
                ):
                    self.gizmo.select_joint(node.body_index, joint.joint_id)
                    self._request_node_joint_focus(
                        next(
                            (
                                candidate
                                for candidate in self.session.nodes
                                if candidate.type is NodeType.JOINT
                                and candidate.joint_index == joint.joint_id
                            ),
                            node,
                        )
                    )
                imgui.end_disabled()
        imgui.end()
