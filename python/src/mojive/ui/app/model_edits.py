"""App: model edits."""

from __future__ import annotations

from pathlib import Path

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.session.model_edits import MODEL_REBUILD_COMMANDS
from mojive.ui.draw2d import ImguiDraw2D, fit_text
from mojive.ui.theme import Theme

from .support import _ApplyModelEdits, _ModelLoadJob


class _ModelEdits:
    """Private model edits methods of ViewerApp; state belongs to its owner."""

    def _pending_edits_blocked(self):
        self.session._record_result(
            cmd.CommandResult.bad(self.localizer.text("Apply or discard pending model edits first"))
        )

    def _intercept_model_edit(self, command):
        draft = self.model_edits
        if draft.applying:
            return None
        # Scale is a dimension-baking operation. Keep its factors editable until
        # Apply, even when other model property writes use the live-update mode.
        if isinstance(command, cmd.SetScale):
            return draft.stage(command)
        if (
            isinstance(command, cmd.SetGeometrySize)
            and draft.pending_scale(command.node_id) is not None
        ):
            return draft.stage_scaled_size(command)
        if isinstance(command, cmd.BeginEditTransaction) and not self.live_model_updates:
            if draft._checkpoint is not None or self.session.editing:
                return cmd.CommandResult.bad("An edit transaction is already active")
            draft.begin_transaction(command.label)
            return cmd.CommandResult.good()
        if (
            isinstance(command, (cmd.EndEditTransaction, cmd.CancelEditTransaction))
            and draft._checkpoint is not None
        ):
            editing = self.session.editing
            result = self.session.submit(command) if editing else cmd.CommandResult.good()
            cancelled = isinstance(command, cmd.CancelEditTransaction) or not result.ok
            draft.end_transaction(cancelled)
            if cancelled and editing and not self.session.editing and draft.active:
                # The direct writes rolled back; only the draft's earlier edits survive.
                draft.rebase_after_failure()
            return result
        if draft.active and isinstance(
            command,
            (
                cmd.Play,
                cmd.Step,
                cmd.Reset,
                cmd.Undo,
                cmd.Redo,
                cmd.SaveScene,
                cmd.NewScene,
                cmd.OpenScene,
                cmd.Reload,
                cmd.LoadAsset,
                cmd.AddSceneModel,
            ),
        ):
            return cmd.CommandResult.bad(
                self.localizer.text("Apply or discard pending model edits first")
            )
        if not self.live_model_updates and isinstance(command, cmd.PreviewSceneModelTransform):
            return draft.stage(
                cmd.SetSceneModelTransform(command.model_id, command.position, command.rotation)
            )
        if isinstance(command, cmd.ClearSceneModelTransformPreview) and any(
            isinstance(item, cmd.SetSceneModelTransform) and item.model_id == command.model_id
            for item in draft.commands
        ):
            # Leaving an individual placement gesture retains the document's pending preview.
            return cmd.CommandResult.good()
        if not self.live_model_updates and isinstance(command, cmd.SetPose):
            node = self.session.node(command.node_id)
            if node is not None and node.type is NodeType.MODEL:
                return draft.stage(
                    cmd.SetSceneModelTransform(node.model_id, command.position, command.rotation)
                )
            if (
                node is not None
                and node.model_id >= 0
                and node.type in (NodeType.GEOM, NodeType.SITE)
            ):
                return draft.stage(command)
        if not self.live_model_updates and isinstance(command, MODEL_REBUILD_COMMANDS):
            node = self.session.node(getattr(command, "node_id", -1))
            if not isinstance(command, cmd.SetGeometrySize) or (
                node is not None and node.model_id >= 0
            ):
                result = draft.stage(command)
                if result is not None:
                    return result
        if isinstance(command, cmd.SelectNode) and command.node_key:
            # Select an element that this batch has not applied yet; the draft owns it.
            return draft.stage(command)
        if draft._checkpoint is not None and not self.session.editing:
            # Only direct document writes need a physics undo snapshot during a gesture.
            from mojive.session import _SCENE_EDIT_COMMANDS

            if self.session.adapter.caps.edit_history and isinstance(command, _SCENE_EDIT_COMMANDS):
                result = self.session.submit(cmd.BeginEditTransaction(draft._transaction_label))
                if not result.ok:
                    return result
        return None

    def set_live_model_updates(self, value: bool) -> None:
        self.live_model_updates = bool(value)
        self.localizer.set_preferences({"live_model_updates": self.live_model_updates})
        if value and self.model_edits.active:
            self._apply_model_edits_requested = True

    def set_theme(self, theme: Theme) -> None:
        """Replace this viewer's colors without changing scene or session state."""

        self.theme = theme
        if self.window is not None:
            self.window.apply_theme(theme)

    def _start_pending_model_edits(self) -> None:
        if not self._apply_model_edits_requested:
            return
        draft = self.model_edits
        if self.session.editing or draft._checkpoint is not None:
            # A gesture owns the document; apply again once it commits its transaction.
            self._apply_model_edits_requested = False
            return
        self._apply_model_edits_requested = False
        if not draft.active:
            return
        if not draft.compatible():
            draft.error = "The scene changed; discard the pending model edits"
            self.session._record_result(cmd.CommandResult.bad(draft.error))
            return
        draft.applying = True
        self.gizmo._reset_model_placement()
        self._model_load_queue.append(
            _ModelLoadJob(
                "edit",
                self.session.asset_path or Path("Untitled"),
                _ApplyModelEdits(draft.resolve_commands(self.session)),
                self._finish_pending_model_edits,
            )
        )

    def _finish_pending_model_edits(self, result) -> None:
        if result.ok:
            self.model_edits.clear()
        else:
            self.model_edits.applying = False
            self.model_edits.error = result.message

    def _draw_pending_model_edits(self) -> None:
        x, y, width, height = self._viewport_rect
        scale = self.window.style_scale
        draw = ImguiDraw2D(imgui.get_foreground_draw_list())
        draw.rect(
            (x + scale, y + scale),
            (x + width - scale, y + height - scale),
            self.theme.warning,
            width=2.0 * scale,
        )
        t = self.localizer.text
        padding = 10.0 * scale
        available = max(1.0, width - 24.0 * scale)
        natural = imgui.calc_text_size(t("Pending model edits")).x + 190.0 * scale
        hint_width = min(available, natural)
        stacked = natural > available
        hint_height = 2 * padding + imgui.get_frame_height() * (2 if stacked else 1)
        if stacked:
            hint_height += imgui.get_style().item_spacing.y
        imgui.set_next_window_pos(
            imgui.ImVec2(x + width * 0.5, y + height - 16 * scale),
            imgui.Cond_.always,
            imgui.ImVec2(0.5, 1),
        )
        imgui.set_next_window_size(imgui.ImVec2(hint_width, hint_height))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(padding, padding))
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_docking.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
        )
        visible, _ = imgui.begin(f"{t('Pending model edits')}###pending_model_edits", None, flags)
        if visible:
            imgui.align_text_to_frame_padding()
            label = t("Pending model edits")
            imgui.text(fit_text(ImguiDraw2D(), label, max(1.0, hint_width - 2 * padding)))
            if self.model_edits.error:
                imgui.set_item_tooltip(self.model_edits.error)
            if not stacked:
                imgui.same_line()
            remaining = imgui.get_content_region_avail().x
            button_width = max(1.0, (remaining - imgui.get_style().item_spacing.x) * 0.5)
            if imgui.button(t("Apply") + "##apply_model_edits", imgui.ImVec2(button_width, 0)):
                self._apply_model_edits_requested = True
            imgui.same_line()
            if imgui.button(t("Discard") + "##discard_model_edits", imgui.ImVec2(button_width, 0)):
                self.model_edits.clear()
                self.gizmo._reset_model_placement()
                self._apply_model_edits_requested = False
        imgui.end()
        imgui.pop_style_var()
