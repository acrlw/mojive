"""Native presentation of the registered camera panel's shared operations."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.design.native.geometry import part_scope
from mojive.ui.localization import camera_caption
from mojive.ui.workspace.component_layout import component_flow, control_frame_size, control_size

from .camera_bookmarks import camera_bookmarks
from .design import DESIGN
from .property_style import NativePropertyPresentation, advance, panel_empty, property_controls
from .widgets import MUTED


@part_scope("button_row")
def _view_directions(workspace, x, y, width):
    ui, scale = workspace.ui, workspace.s
    style = ui.component_style("button_row")
    padding, margin = style.edges("padding", scale), style.edges("margin", scale)
    available = max(scale, width - padding[0] - padding[2] - margin[0] - margin[2])
    gap = style.get("columnGap", DESIGN.item_spacing_x) * scale
    names = ("Front", "Back", "Left", "Right", "Top", "Bottom", "ISO", "Frame all")
    captions = tuple(
        camera_caption(name, workspace._panel_context.language, ui.translate) for name in names
    )
    icons = (*("view-" + name.lower() for name in names[:-1]), "frame")
    buttons = ui.component_style("button")
    insets = buttons.edges("padding", scale, (8, 0, 8, 0))
    minimum = max(
        ui.measure(label, buttons.get("fontSize", DESIGN.body_size), style=buttons)
        for label in captions
    )
    minimum += (buttons.get("iconSize", DESIGN.body_size) + buttons.get("gap", 5)) * scale
    minimum += insets[0] + insets[2]
    columns = 2 if available >= 2 * minimum + gap else 1
    sizes = [
        control_size(
            ui,
            "camera-" + name,
            max(scale, (available - gap * (columns - 1)) / columns) / scale,
            DESIGN.control_height,
            label=captions[index],
            icon=icons[index],
        )
        for index, name in enumerate(names)
    ]
    box, flow = component_flow(
        ui,
        "button_row",
        x,
        y,
        width,
        sizes,
        key="camera-view-directions",
        column_gap=DESIGN.item_spacing_x,
        row_gap=DESIGN.row_gap,
    )
    for index, (name, rect) in enumerate(zip(names, flow.rectangles, strict=True)):
        key = "camera-frame" if name == "Frame all" else "camera-" + name
        icon = icons[index]
        caption = captions[index]
        frame = control_frame_size(ui, key, rect.width, rect.height, label=caption, icon=icon)
        if ui.button(
            key,
            rect.x,
            rect.y,
            *frame,
            label=caption,
            icon=icon,
            align="left",
            tooltip=ui.translate(name),
            translate=False,
            width_limit=frame[0] * scale,
            filled=False,
        ):
            workspace.app._leave_model_camera()
            if name == "Frame all":
                workspace.app._frame_scene(animate=True)
            else:
                workspace.app.camera.set_preset(name.lower())
    return box.outer.bottom


def camera_action_active(workspace, key):
    return (key == "search" and workspace.camera_search_visible) or (
        key == "sort" and workspace.camera_sorted
    )


def camera_action(workspace, key):
    if key == "search":
        workspace.camera_search_visible = not workspace.camera_search_visible
        if not workspace.camera_search_visible:
            workspace.camera_query = ""
    elif key == "sort":
        workspace.camera_sorted = not workspace.camera_sorted
    elif key == "add":
        workspace.app._add_scene_camera()
        workspace.inspector_camera = None
        workspace.open["Inspector"] = True


def _select_camera(workspace, node):
    workspace._panel_context.submit(cmd.SelectNode(node.node_id))
    workspace.inspector_camera = workspace.pinned = None
    workspace.open["Inspector"] = True
    workspace.inspector_tab = "Properties"


def camera_operation(workspace, node, action):
    app, session = workspace.app, workspace.app.session
    camera_id = session.cameras[node.camera_index].camera_id
    pilot = getattr(app, "_camera_pilot", None)
    active = app._model_camera_id == camera_id
    if action == "view":
        app.select_model_camera(-1 if active and pilot is None else camera_id, animate=True)
    elif action == "pilot":
        from mojive.ui.app.camera_pilot import start_pilot

        if active and pilot is not None:
            app.select_model_camera(-1)
        else:
            start_pilot(app, camera_id)
    elif action == "align":
        session.submit(cmd.SetSceneCamera(camera_id, app._camera_view()))
    elif action == "focus":
        app.select_model_camera(-1)
        app.request_node_focus(node.node_id)
    elif action == "capture":
        app.select_model_camera(camera_id)
        app.request_capture()
    elif action == "delete":
        if active:
            app.select_model_camera(-1)
        session.submit(cmd.RemoveSceneCamera(camera_id))


@part_scope("entity_card")
def camera_panel(workspace, x, y, width):
    """Browse cameras, choose a viewport source, or pilot a scene camera."""
    from mojive.design.native.geometry import Rect

    from .entity_cards import card_surface
    from .property_style import property_section

    ui, scale, ctx = workspace.ui, workspace.s, workspace._panel_context
    if workspace.camera_search_visible:
        _, workspace.camera_query = ui.search(
            "##camera-search", x, y, width, workspace.camera_query, hint=ctx.tr("Search cameras")
        )
        y = imgui.get_item_rect_max().y + 12 * scale
    if ui.button(
        "camera-scene",
        x,
        y,
        width / scale,
        28,
        label="Editor Camera",
        icon="perspective",
        active=ctx.model_camera_id < 0,
    ):
        select_scene_camera(workspace)
    y += 40 * scale
    advance(ui, x, y)
    nodes = [
        node
        for node in ctx.session.nodes
        if node.type is NodeType.CAMERA
        and workspace.camera_query.casefold() in node.name.casefold()
    ]
    if workspace.camera_sorted:
        nodes.sort(key=lambda node: node.name.casefold())
    if property_section(ui, ctx.tr("Scene cameras"), imgui.TreeNodeFlags_.default_open):
        y = imgui.get_cursor_screen_pos().y
        for node in nodes:
            info = ctx.session.cameras[node.camera_index]
            view = ctx.session.camera_view(info.camera_id)
            row = 28 * scale
            box = ui.component_box(
                "entity_card", x, y, width, 76 * scale, key=f"camera-card-{info.camera_id}"
            )
            card_surface(ui, box.frame, selected=ctx.session.selected_node is node)
            bx, by, bw, _ = box.content.tuple()
            key = f"camera-select-{info.camera_id}"
            if ui.button(
                key,
                bx,
                by,
                (bw - row - 4 * scale) / scale,
                28,
                label=node.name,
                icon="camera",
                align="left",
                translate=False,
            ):
                _select_camera(workspace, node)
            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                workspace.app.select_model_camera(info.camera_id, animate=True)
            popup = f"Camera actions##{info.camera_id}"
            if ui.button(
                f"camera-more-{info.camera_id}",
                bx + bw - row,
                by,
                28,
                28,
                icon="more",
                tooltip="More camera actions",
            ):
                imgui.open_popup(popup)
            with workspace.menus.local_popup(popup) as opened:
                if opened:
                    for action, title, icon in (
                        ("focus", "Focus camera", "fit"),
                        ("align", "Align to view", "camera"),
                        ("capture", "Capture image", "video"),
                        ("delete", "Delete camera", "delete"),
                    ):
                        if action == "delete" and node.model_id >= 0:
                            continue
                        if workspace.menus.local_item(
                            title, key=f"camera-{action}-{info.camera_id}", icon=icon
                        )[0]:
                            camera_operation(workspace, node, action)
            by += row + 4 * scale
            if view is not None:
                label = ctx.tr("Ortho" if view.orthographic else "Persp")
                ui.component_text(
                    "control_metadata",
                    Rect(bx + 8 * scale, by, max(0, bw - 72 * scale), row),
                    label,
                    MUTED,
                    size=14,
                    translate=False,
                )
            pilot = getattr(workspace.app, "_camera_pilot", None)
            for index, (action, icon, title) in enumerate(
                (("view", "eye", "View Camera"), ("pilot", "move", "Pilot camera"))
            ):
                active = ctx.model_camera_id == info.camera_id and (
                    (pilot is not None) == (action == "pilot")
                )
                if ui.button(
                    f"camera-{action}-{info.camera_id}",
                    bx + bw - (60 - 32 * index) * scale,
                    by,
                    28,
                    28,
                    icon=icon,
                    active=active,
                    tooltip=title,
                    enabled=view is not None,
                ):
                    camera_operation(workspace, node, action)
            y = box.outer.bottom + 8 * scale
        if not nodes:
            y = panel_empty(
                ui,
                "camera-empty",
                "No matching cameras" if workspace.camera_query else "No scene cameras",
                x,
                y,
                width,
            )
        advance(ui, x, y)
    if property_section(ui, ctx.tr("View presets"), imgui.TreeNodeFlags_.default_open):
        y = imgui.get_cursor_screen_pos().y
        advance(ui, x, _view_directions(workspace, x, y, width))


def camera_entity_actions(workspace, node):
    """Share camera operations between the browser and the inspected entity."""
    from .property_style import property_section

    ui, ctx, scale = workspace.ui, workspace._panel_context, workspace.s
    if not property_section(ui, ctx.tr("Camera view"), imgui.TreeNodeFlags_.default_open):
        return
    pos, width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
    gap, row = 8 * scale, 28 * scale
    cell = (width - gap) / 2
    for index, (action, icon, label) in enumerate(
        (
            ("view", "eye", "View Camera"),
            ("pilot", "move", "Pilot camera"),
            ("align", "camera", "Align to view"),
            ("capture", "video", "Capture image"),
        )
    ):
        if ui.button(
            "inspector-camera-" + action,
            pos.x + (index % 2) * (cell + gap),
            pos.y + (index // 2) * (row + gap),
            cell / scale,
            28,
            icon=icon,
            label=label,
            align="left",
        ):
            camera_operation(workspace, node, action)
    advance(ui, pos.x, pos.y + 2 * (row + gap))


def select_scene_camera(workspace):
    session = workspace.app.session
    session.submit(cmd.Select(0))
    workspace.app.select_model_camera(-1, animate=True)
    workspace.inspector_camera = (session.document_id, session.selection_revision)
    workspace.pinned = None
    workspace.open["Inspector"] = True
    workspace.inspector_tab = "Properties"


def scene_camera_selected(workspace):
    if getattr(workspace, "inspector_camera", None) is None:
        return False
    session = workspace.app.session
    return getattr(workspace, "inspector_camera", None) == (
        session.document_id,
        session.selection_revision,
    )


def scene_camera_entity(workspace):
    """The non-deletable viewport camera is UI state, never a fabricated scene node."""
    return {
        "name": workspace.ui.translate("Editor Camera"),
        "icon": "perspective",
        "icon_color": MUTED,
        "identity": None,
        "node_id": -1,
        "colorable": False,
        "scene_camera": True,
    }


def camera_properties(workspace, x, y, width):
    ui, ctx = workspace.ui, workspace._panel_context
    panel = workspace.app.panels.get("camera")
    if panel.presentation is None:
        panel.presentation = NativePropertyPresentation(
            ui, prefix="camera", fit_labels=True, show_tracks=False
        )
    with property_controls(ui):
        imgui.set_cursor_screen_pos((x, y))
        panel._params(ctx, ctx.camera)
        panel._tracking(ctx)
        camera_bookmarks(workspace, panel, ctx)
