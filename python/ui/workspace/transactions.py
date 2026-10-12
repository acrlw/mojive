"""Native workspace transactions; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd


def begin_numeric_edit(workspace, key, *, deferred=False, label="Inspector edit"):
    state = workspace.ui.numeric_edits[key]
    if state.activated and workspace._numeric_edit is None and not workspace.app.session.editing:
        draft = workspace.app.model_edits
        if draft._checkpoint is None:
            if deferred:
                draft.begin_transaction(label)
                workspace._numeric_edit = key
            elif workspace._panel_context.submit(cmd.BeginEditTransaction(label)).ok:
                workspace._numeric_edit = key
    return state


def finish_numeric_edit(workspace, key, state):
    if workspace._numeric_edit != key:
        return
    if state.canceled or (state.deactivated and not state.active):
        command = cmd.CancelEditTransaction() if state.canceled else cmd.EndEditTransaction()
        workspace._panel_context.submit(command)
        workspace._numeric_edit = None
        if state.canceled:
            workspace.app.refresh_document()


def finish_numeric_frame(workspace, ctx):
    if workspace._numeric_edit is not None and not imgui.is_any_item_active():
        ctx.submit(cmd.EndEditTransaction())
        workspace._numeric_edit = None
