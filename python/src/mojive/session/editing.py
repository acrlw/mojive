"""Session: editing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Session

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from uuid import uuid4

from mojive import commands as cmd
from mojive.commands import Command, CommandResult
from mojive.session.history import EditRecord
from mojive.session.model_edits import _ModelEditPlan, intercept_model_edit

from .state import (
    _SCENE_EDIT_COMMANDS,
    PerturbState,
    _DocumentState,
)


class _Editing:
    """Private editing methods of Session; state belongs to its owner."""

    @contextmanager
    def edit(self, label: str = "Edit") -> Iterator[Session]:
        """Group submitted edits, rolling back on exceptions or failed commands.

        Direct adapter or Scene mutations bypass command tracking. Use submit()
        inside this context so failure, dirty state, and history remain consistent.
        """
        result = self.submit(cmd.BeginEditTransaction(label))
        if not result.ok:
            raise RuntimeError(result.message)
        try:
            yield self
        except BaseException as error:
            result = self.submit(cmd.CancelEditTransaction())
            if not result.ok:
                raise RuntimeError(result.message) from error
            raise
        else:
            result = self.submit(cmd.EndEditTransaction())
            if not result.ok:
                raise RuntimeError(result.message)

    @property
    def editing(self) -> bool:
        """Return whether a continuous edit transaction is active."""
        return self._edit_before is not None

    def apply_model_edits(self, commands) -> CommandResult:
        """Apply pending model commands atomically as one undoable rebuild group.

        Accepts a pending edit plan when the batch also styles or selects the elements it
        creates, because those commands only become addressable after the creation phase.
        """

        plan = commands if isinstance(commands, _ModelEditPlan) else _ModelEditPlan(tuple(commands))
        if not self.paused or self.editing:
            return CommandResult.bad("Pause simulation and finish the active gesture before Apply")
        history = self._adapter.caps.edit_history
        before = None if history else self._capture_document_state()
        if history:
            result = self.submit(cmd.BeginEditTransaction("Apply model edits"))
            if not result.ok:
                return result
        self._applying_model_edits = True
        try:
            # Creation must install nodes before dependent commands can address them.
            # Record actual identities immediately: later creations can shift node IDs.
            created = []
            for command in plan.creations:
                if plan.rebind is not None:
                    command = plan.rebind(self, (command,))[0]
                result = self.submit(command)
                if not result.ok:
                    raise ValueError(result.message)
                self._refresh_structure(installed=True)
                node = self.node(result.entity_id)
                if node is None:
                    raise ValueError("Creation did not return an installed entity")
                created.append((node.model_id, node.type, node.name))
            direct = plan.rebind(self, plan.direct) if plan.rebind is not None else plan.direct
            follow_ups = plan.bind(self, tuple(created)) if plan.bind is not None else ()
            with self._adapter.model_edit_batch():
                # All dependent writes share one rebuild. Renames retain the installed
                # identities until the other commands have finished using them.
                ordered = sorted(
                    (*direct, *follow_ups),
                    key=lambda item: isinstance(item, cmd.RenameModelElement),
                )
                for command in ordered:
                    result = self.submit(command)
                    if not result.ok:
                        raise ValueError(result.message)
                state = self._adapter.capture_state()
            changes_layout = any(
                isinstance(
                    item,
                    (
                        cmd.SetModelSource,
                        cmd.AddModelElement,
                        cmd.DuplicateModelElement,
                        cmd.RemoveModelElement,
                        cmd.RemoveSceneModel,
                        cmd.ModelEditBatch,
                        cmd.AddModelComponent,
                        cmd.RemoveModelComponent,
                        cmd.UpdateModelComponent,
                    ),
                )
                for item in (*plan.creations, *plan.direct)
            )
            # Raw arrays preserve renamed DOFs only when the layout stays stable.
            # Topology edits rely on the adapter's identity-based state migration.
            if state is not None and not changes_layout:
                self._adapter.restore_state(state)
            self._applying_model_edits = False
            self._refresh_structure()
            return (
                self.submit(cmd.EndEditTransaction())
                if history
                else CommandResult.good("Applied model edits")
            )
        except Exception as error:
            self._applying_model_edits = False
            rollback = (
                self.submit(cmd.CancelEditTransaction()).ok
                if history
                else self._restore_document_state(before)
            )
            draft = self._model_edit_preview
            if rollback and draft is not None and draft.applying:
                draft.rebase_after_failure()
            return CommandResult.bad(str(error) if rollback else f"{error}; rollback failed")

    def submit(self, command: Command) -> CommandResult:
        """Apply one typed command and update edit history and status text."""
        from mojive.session.command_support import unavailable_reason

        reason = unavailable_reason(self._adapter.caps, command)
        if reason is not None:
            if self.editing:
                self._edit_error = self._edit_error or reason
            return self._record_result(CommandResult.bad(reason))
        intercepted = intercept_model_edit(self, command)
        if intercepted is not None:
            return self._record_result(intercepted)
        driver = self._simulation_driver
        # Camera and selection only change Session-owned state. All other
        # commands fence physics before reading history or mutating an adapter;
        # new command types inherit the safe default.
        if driver is None or isinstance(command, (cmd.SetCamera, cmd.Select, cmd.SelectNode)):
            return self._submit(command)
        was_running = not self._paused and not self._state_take_playing
        self._step_counter += driver.suspend()
        try:
            return self._submit(command)
        finally:
            self._resume_physics(reset_clock=not was_running)

    def _submit(self, command: Command) -> CommandResult:
        if isinstance(command, cmd.BeginEditTransaction):
            result = self._begin_edit(command.label)
            return self._record_result(result)
        if isinstance(command, cmd.EndEditTransaction):
            result = self._end_edit()
            return self._record_result(result)
        if isinstance(command, cmd.CancelEditTransaction):
            return self._record_result(self._cancel_edit())
        if isinstance(command, cmd.Undo):
            result = self._undo()
            return self._record_result(result)
        if isinstance(command, cmd.Redo):
            result = self._redo()
            return self._record_result(result)

        if self.editing and isinstance(
            command, (cmd.Reload, cmd.NewScene, cmd.OpenScene, cmd.LoadAsset, cmd.SaveScene)
        ):
            message = "Finish or cancel the active edit before document operations"
            self._edit_error = self._edit_error or message
            return self._record_result(CommandResult.bad(message))

        scene_edit = isinstance(command, _SCENE_EDIT_COMMANDS)
        before = (
            self._capture_document_state()
            if scene_edit and self._adapter.caps.edit_history and not self.editing
            else None
        )
        result = self._dispatch(command)
        if not result.ok and scene_edit and self.editing:
            self._edit_error = self._edit_error or result.message or "Scene edit failed"
        if result.ok and scene_edit and self._adapter.caps.scene_files:
            if self.editing:
                self._edit_changed = True
            elif before is not None:
                if not self._commit_edit(type(command).__name__, before):
                    result = CommandResult.good(
                        "Edit applied; undo history exceeded its memory budget", result.entity_id
                    )
            else:
                self._advance_document_revision()
        preview = getattr(self, "_model_edit_preview", None)
        if (
            result.ok
            and preview is not None
            and preview.active
            and not preview.applying
            and isinstance(
                command,
                (
                    cmd.SetPose,
                    cmd.SetScale,
                    cmd.SetMaterial,
                    cmd.SetGeometryColor,
                    cmd.SetGeometrySize,
                    cmd.AddSceneObject,
                    cmd.RemoveSceneObject,
                    cmd.DuplicateSceneEntity,
                    cmd.RemoveSceneEntity,
                    cmd.RenameSceneEntity,
                    cmd.AddSceneLight,
                    cmd.AddSceneCamera,
                    cmd.SetLight,
                    cmd.SetEnvironment,
                    cmd.SetSceneCamera,
                    cmd.SetGeometryProperties,
                    cmd.SetJointProperties,
                ),
            )
        ):
            preview._adapter_revision = self._adapter.structure_revision
            preview.refresh_preview()
        return self._record_result(result)

    def _capture_document_state(self) -> _DocumentState:
        state = self._adapter.capture_edit_state()
        if state is None:
            raise RuntimeError(f"{self._adapter.caps.name} did not provide an edit state")
        return _DocumentState(state, self._selected, deepcopy(self._authored))

    def _restore_document_state(self, state: _DocumentState) -> bool:
        if not self._adapter.restore_edit_state(state.adapter_state):
            return False
        self._authored = deepcopy(state.authored)
        self._selected = int(state.selected)
        self._selected_node_id = -1
        self._refresh_structure()
        if self._selected not in self._by_object_id:
            self._selected = 0
            self._selected_node_id = -1
        return True

    def _begin_edit(self, label: str) -> CommandResult:
        if not self._adapter.caps.edit_history:
            return CommandResult.bad(f"{self._adapter.caps.name} does not support edit history")
        if self.editing:
            return CommandResult.bad("An edit transaction is already active")
        self._edit_before = self._capture_document_state()
        self._edit_before_revision = self._document_revision
        self._edit_label = str(label) or "Edit"
        self._edit_changed = False
        self._edit_error = ""
        return CommandResult.good()

    def _cancel_edit(self) -> CommandResult:
        if not self.editing:
            return CommandResult.bad("No edit transaction is active")
        if not self._restore_document_state(self._edit_before):
            return CommandResult.bad("Edit state is incompatible with this scene")
        self._document_revision = self._edit_before_revision
        self._edit_before = None
        self._edit_label = ""
        self._edit_changed = False
        self._edit_error = ""
        return CommandResult.good("Edit cancelled")

    def _end_edit(self) -> CommandResult:
        if not self.editing:
            return CommandResult.bad("No edit transaction is active")
        if self._edit_error:
            error = self._edit_error
            result = self._cancel_edit()
            return (
                CommandResult.bad(f"Edit transaction rolled back: {error}") if result.ok else result
            )
        before = self._edit_before
        label = self._edit_label
        changed = self._edit_changed
        before_revision = self._edit_before_revision
        self._edit_before = None
        self._edit_label = ""
        self._edit_changed = False
        if changed and before is not None:
            if not self._commit_edit(label, before, before_revision):
                return CommandResult.good("Edit applied; undo history exceeded its memory budget")
            return CommandResult.good(label)
        return CommandResult.good()

    def _commit_edit(
        self,
        label: str,
        before: _DocumentState,
        before_revision: int | None = None,
    ) -> bool:
        revision = self._document_revision if before_revision is None else before_revision
        after_revision = self._next_document_revision
        self._next_document_revision += 1
        retained = self._history.append(
            EditRecord(
                label,
                before,
                self._capture_document_state(),
                revision,
                after_revision,
            )
        )
        self._document_revision = after_revision
        return retained

    def _advance_document_revision(self) -> None:
        self._document_revision = self._next_document_revision
        self._next_document_revision += 1
        self._history.clear_redo()

    def _undo(self) -> CommandResult:
        if self.editing:
            return CommandResult.bad("Finish the active edit before undo")
        if not self._undo_stack:
            return CommandResult.bad("Nothing to undo")
        record = self._undo_stack.pop()
        if not self._restore_document_state(record.before):
            self._undo_stack.append(record)
            return CommandResult.bad("Undo state is incompatible with this scene")
        self._redo_stack.append(record)
        self._document_revision = record.before_revision
        return CommandResult.good(f"Undo {record.label}")

    def _redo(self) -> CommandResult:
        if self.editing:
            return CommandResult.bad("Finish the active edit before redo")
        if not self._redo_stack:
            return CommandResult.bad("Nothing to redo")
        record = self._redo_stack.pop()
        if not self._restore_document_state(record.after):
            self._redo_stack.append(record)
            return CommandResult.bad("Redo state is incompatible with this scene")
        self._undo_stack.append(record)
        self._document_revision = record.after_revision
        return CommandResult.good(f"Redo {record.label}")

    def _reset_edit_history(self) -> None:
        self._document_id = uuid4().hex
        self._history.clear()
        self._edit_before = None
        self._edit_label = ""
        self._edit_changed = False
        self._edit_error = ""
        self._document_revision = 0
        self._saved_revision = 0
        self._next_document_revision = 1

    def _pause_loaded_scene(self) -> None:
        """Request an editable pause and reset state associated with the old scene."""

        self._paused = not self._adapter.caps.simulation or self._adapter.set_paused(True)
        self._clear_state_take()
        self._clear_scene_snapshots()
        self._clear_frame_history()
        self._step_counter = 0
        self._pending_steps = 0
        self._sim_time_credit = 0.0
        self._perturb = PerturbState()
        self._active_keyframe = -1

    def _pause_before_model_change(self) -> bool:
        """Stop simulation before an adapter replaces model-owned state."""

        if not self._adapter.caps.simulation or self._paused:
            return True
        if not self._adapter.set_paused(True):
            return False
        self._paused = True
        self._sim_time_credit = 0.0
        self._state_take_recording = False
        self._state_take_playing = False
        return True

    @staticmethod
    def _command_node_id(command, node_key: str) -> int:
        """Return the node a command addresses.

        A pending batch binds its creation keys to real node IDs before applying, so an
        unresolved key names an element that never reached the scene and stays invalid.
        """

        return -1 if node_key else int(command.node_id)

    @staticmethod
    def _command_node_error(command, node_key: str, kind: str) -> str:
        """Describe an unresolved command target for the status output."""

        if node_key:
            return f"pending {kind} {node_key} is unavailable"
        return f"Unknown node_id={command.node_id}"
