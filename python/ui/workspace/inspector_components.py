"""Use the editor's component fields inside the native Inspector tabs."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType

from .inspector_style import NativeInspectorPresentation
from .property_style import property_controls


def _empty(ctx):
    imgui.text_disabled(ctx.tr("No properties for this selection."))


def _physics(panel, ctx, node):
    session = ctx.session
    if node.type in (NodeType.LINK, NodeType.ROBOT):
        if session.body_properties(node.node_id) is not None:
            imgui.set_next_item_open(True, imgui.Cond_.once)
            panel._body_properties(ctx, node)
            return
    elif node.type is NodeType.GEOM:
        drawn = False
        if session.geometry_properties(node.node_id) is not None:
            imgui.set_next_item_open(True, imgui.Cond_.once)
            panel._geometry_contact_properties(ctx, node.node_id)
            drawn = True
        if session.geometry_advanced_properties(node.node_id) is not None:
            imgui.set_next_item_open(True, imgui.Cond_.once)
            panel._geometry_advanced_properties(ctx, node.node_id)
            drawn = True
        if drawn:
            return
    elif node.type is NodeType.JOINT:
        panel._joint(ctx, node)
        return
    elif node.type is NodeType.SITE:
        if session.site_properties(node.node_id) is not None:
            imgui.set_next_item_open(True, imgui.Cond_.once)
            panel._site_properties(ctx, node)
            return
    elif node.type is NodeType.ENVIRONMENT and (
        session.adapter.caps.simulation or session.adapter.caps.supports("physics.options")
    ):
        panel._physics_options(ctx)
        return
    _empty(ctx)


def draw_component(workspace, ctx, entity, tab, *, details=None):
    """Apply native presentation to the registered Inspector's real controls."""
    presentation = getattr(workspace, "inspector_presentation", None)
    if presentation is None:
        presentation = workspace.inspector_presentation = NativeInspectorPresentation(workspace.ui)
    previous = ctx.inspector_presentation
    ctx.inspector_presentation = presentation
    presentation.ctx = ctx
    presentation.canceled = False
    presentation.color_edit_started = False
    presentation.color_edit_action = None
    presentation.node_id = entity["node_id"] if entity is not None else -1
    presentation._row = None
    with property_controls(workspace.ui):
        try:
            panel = workspace.app.panels.get("inspector")
            if details is not None:
                panel._geometry_components(ctx, ctx.session.node(entity["node_id"]), fields=details)
                handled = True
            else:
                handled = _draw_component(workspace, ctx, entity, tab)
            if presentation.color_edit_started and not ctx.session.editing:
                result = ctx.submit(cmd.BeginEditTransaction("Inspector color"))
                panel._edit_transaction = result.ok
            if presentation.color_edit_owner is not None and (
                presentation.color_edit_owner[0] != presentation.node_id
                or tab not in ("Properties", "Physics", "Material")
            ):
                presentation.color_edit_action = "cancel"
                presentation.color_edit_open = False
            if presentation.color_edit_action is not None:
                if panel._edit_transaction:
                    command = (
                        cmd.CancelEditTransaction()
                        if presentation.color_edit_action == "cancel"
                        else cmd.EndEditTransaction()
                    )
                    if ctx.submit(command).ok:
                        panel._edit_transaction = False
                presentation.color_edit_owner = None
                presentation.color_edit_open = False
            if presentation.canceled and panel._edit_transaction:
                result = ctx.submit(cmd.CancelEditTransaction())
                if result.ok:
                    panel._edit_transaction = False
                    imgui.internal.clear_active_id()
            return handled
        finally:
            ctx.inspector_presentation = previous


def _draw_component(workspace, ctx, entity, tab):
    """Return whether component fields replaced the native tab's generic contents."""
    panel = workspace.app.panels.get("inspector")
    # Keep the registered panel's drafts, popups and transaction owner. Its
    # finish_frame runs once in WorkspacePanels, after the native window draws.
    panel._draw_model_source(ctx)
    panel._draw_component_editor(ctx)
    node = ctx.session.node(entity["node_id"]) if entity is not None else None
    if node is None:
        return False
    if tab == "Properties":
        if node.type is NodeType.CAMERA:
            panel._camera(ctx, node)
        elif node.type is NodeType.LIGHT:
            panel._light(ctx, node)
        elif node.type is NodeType.MODEL:
            gizmo = ctx.gizmo
            previous_model = gizmo.model_placement_model_id if gizmo is not None else -1
            was_active = gizmo is not None and gizmo.model_placement_active(ctx.session)
            panel._model(ctx, node)
            if (
                gizmo is not None
                and (previous_model != node.model_id or not was_active)
                and gizmo.model_placement_active(ctx.session, node.model_id)
            ):
                # A pinned Inspector can start placement while another node is selected.
                ctx.submit(cmd.SelectNode(node.node_id))
        elif node.type is NodeType.ENVIRONMENT:
            panel._environment_rendering(ctx)
        elif node.type is NodeType.JOINT:
            panel._joint(ctx, node)
        elif node.type is NodeType.SITE:
            panel._transform(ctx, node)
            if ctx.session.site_properties(node.node_id) is not None:
                imgui.set_next_item_open(True, imgui.Cond_.once)
                panel._site_properties(ctx, node)
            panel._geometry_components(ctx, node, fields="geometry")
        elif node.type is NodeType.WORLD:
            _empty(ctx)
        else:
            return False
        return True
    if tab == "Physics":
        _physics(panel, ctx, node)
        return True
    if tab == "Material":
        if entity["colorable"]:
            return False
        _empty(ctx)
        return True
    if tab == "Joint":
        position = imgui.get_cursor_screen_pos()
        workspace.pose_panel(
            position.x, position.y, imgui.get_content_region_avail().x, node_id=node.node_id
        )
        if node.type is NodeType.JOINT:
            imgui.spacing()
            panel._joint(ctx, node)
        return True
    return False
