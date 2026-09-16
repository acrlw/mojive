"""Atomic execution of prepared edits through the owning Session and adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mojive import commands as cmd

from .edit_plan import EditBatchResult, EditVersion, ModelEditPlan
from .state import _SCENE_EDIT_COMMANDS

if TYPE_CHECKING:
    from .core import Session


_LAYOUT_EDITS = (
    cmd.SetModelSource,
    cmd.AddModelElement,
    cmd.DuplicateModelElement,
    cmd.RemoveModelElement,
    cmd.RemoveSceneModel,
    cmd.ModelEditBatch,
    cmd.AddModelComponent,
    cmd.RemoveModelComponent,
    cmd.UpdateModelComponent,
)
_BATCH_COMMANDS = (*_SCENE_EDIT_COMMANDS, cmd.Select, cmd.SelectNode)


class _CommandFailure(Exception):
    def __init__(self, result, index):
        super().__init__(result.message)
        self.index = index


def execute_edit_plan(session: Session, plan: ModelEditPlan, label: str) -> EditBatchResult:
    """Execute one plan with a shared rebuild group, history and failure recovery."""
    version = EditVersion.capture(session)
    if plan.version is not None and plan.version != version:
        return EditBatchResult(
            cmd.CommandResult.bad(
                "The scene changed; prepare the edits against the current document"
            )
        )
    if not session.paused or session.editing:
        return EditBatchResult(
            cmd.CommandResult.bad("Pause simulation and finish the active gesture before Apply")
        )
    for index, command in enumerate((*plan.creations, *plan.direct)):
        if not isinstance(command, _BATCH_COMMANDS):
            return EditBatchResult(
                cmd.CommandResult.bad(f"{type(command).__name__} cannot be part of an edit batch"),
                failed_index=index,
            )
    history = session.adapter.caps.edit_history
    before = None
    try:
        if history:
            result = session.submit(cmd.BeginEditTransaction(label))
            if not result.ok:
                return EditBatchResult(result)
        else:
            before = session._capture_document_state()
    except Exception as error:
        return EditBatchResult(cmd.CommandResult.bad(str(error)))

    results = []

    def submit(command):
        index = len(results)
        if not isinstance(command, _BATCH_COMMANDS):
            raise _CommandFailure(
                cmd.CommandResult.bad(f"{type(command).__name__} cannot be part of an edit batch"),
                index,
            )
        try:
            result = session.submit(command)
        except Exception as error:
            raise _CommandFailure(cmd.CommandResult.bad(str(error)), index) from error
        if not result.ok:
            raise _CommandFailure(result, index)
        results.append(result)
        return result

    session._applying_model_edits = True
    try:
        # Install each creation before recording its identity: later creations can
        # shift node indices, so neither the draft nor executor may predict IDs.
        created = []
        for command in plan.creations:
            if plan.rebind is not None:
                command = plan.rebind(session, (command,))[0]
            result = submit(command)
            session._refresh_structure(installed=True)
            node = session.node(result.entity_id)
            if node is None:
                raise ValueError("Creation did not return an installed entity")
            created.append((node.model_id, node.type, node.name))
        direct = plan.rebind(session, plan.direct) if plan.rebind is not None else plan.direct
        follow_ups = plan.bind(session, tuple(created)) if plan.bind is not None else ()
        ordered = (*direct, *follow_ups)
        if plan.renames_last:
            ordered = sorted(ordered, key=lambda item: isinstance(item, cmd.RenameModelElement))
        with session.adapter.model_edit_batch():
            for command in ordered:
                submit(command)
            state = session.adapter.capture_state()
        # Reordered topology requires the adapter's named migration. Raw vectors
        # preserve renamed degrees of freedom only while their layout is stable.
        changes_layout = any(
            isinstance(item, _LAYOUT_EDITS) for item in (*plan.creations, *plan.direct)
        )
        if state is not None and not changes_layout and not session.adapter.restore_state(state):
            raise ValueError("Edited model rejected the preserved physics state")
        session._applying_model_edits = False
        session._refresh_structure()
        if not history and any(
            isinstance(command, _SCENE_EDIT_COMMANDS) for command in (*plan.creations, *plan.direct)
        ):
            session._advance_document_revision()
        result = (
            session.submit(cmd.EndEditTransaction()) if history else cmd.CommandResult.good(label)
        )
        if not result.ok:
            raise ValueError(result.message)
        return EditBatchResult(result, tuple(results))
    except BaseException as error:
        session._applying_model_edits = False
        rollback_error = ""
        try:
            if history:
                # EndEditTransaction can already have rolled back a recorded failure.
                if session.editing:
                    rollback = session.submit(cmd.CancelEditTransaction())
                elif (
                    session.document_id == version.document_id
                    and session.document_revision == version.document_revision
                ):
                    rollback = cmd.CommandResult.good()
                else:
                    rollback = cmd.CommandResult.bad(
                        "The transaction checkpoint is no longer available"
                    )
                if not rollback.ok:
                    rollback_error = rollback.message
            elif not session._restore_document_state(before):
                rollback_error = "Edit state is incompatible with this scene"
        except Exception as recovery:
            rollback_error = str(recovery)
        if rollback_error and not history:
            # A failed recovery still owns its checkpoint even without Undo support.
            session._edit_before = before
            session._edit_before_revision = version.document_revision
            session._edit_label = label
            session._edit_error = str(error)
            session._edit_changed = True
        draft = session._model_edit_preview
        if not rollback_error and draft is not None and draft.applying:
            draft.rebase_after_failure()
        if not isinstance(error, Exception):
            if rollback_error:
                error.add_note(f"Edit rollback failed: {rollback_error}")
            raise
        message = str(error)
        if rollback_error:
            message += f"; rollback failed: {rollback_error}"
        return EditBatchResult(
            cmd.CommandResult.bad(message),
            tuple(results),
            error.index if isinstance(error, _CommandFailure) else None,
            rollback_error,
        )
    finally:
        session._applying_model_edits = False
