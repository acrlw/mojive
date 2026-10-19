"""Native workspace docking; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.workspace.constants import BOTTOM_PANELS, LEFT_PANELS
from mojive.ui.workspace.design import DESIGN


def toggle_activity(workspace, name):
    if workspace.open[name]:
        if name in BOTTOM_PANELS:
            workspace.collapse_bottom()
        else:
            workspace.open[name] = False
    else:
        workspace.activate(name)


def activate(workspace, name):
    panel = workspace.app.panels.load(name)
    if panel is None or not panel.enabled:
        return
    name = panel.name
    if name not in workspace.open:
        panel.open = True
        return
    if name in LEFT_PANELS:
        for other in LEFT_PANELS:
            workspace.open[other] = other == name
        workspace.left_active = workspace.active = name
    elif name in BOTTOM_PANELS:
        for other in BOTTOM_PANELS:
            other_panel = workspace.app.panels.get(other)
            workspace.open[other] = other_panel is not None and other_panel.enabled
        workspace.bottom_active = name
        workspace.bottom_focus = name
        workspace.timeline_layout_pending = True
    if not workspace.open[name] or imgui.internal.find_window_by_name(name) is None:
        workspace.open[name] = True
        workspace.new_panel = name
    workspace.focus_next = name
    workspace.app.panels.get(name).open = True


def build(workspace):
    ii = imgui.internal
    ii.dock_builder_remove_node(workspace.window.dockspace_id)
    ii.dock_builder_add_node(workspace.window.dockspace_id, ii.DockNodeFlagsPrivate_.dock_space)
    work = imgui.get_current_context().viewports[0].get_build_work_rect()
    size = work.get_size()
    ii.dock_builder_set_node_size(workspace.window.dockspace_id, size)
    rest_root = workspace.window.dockspace_id
    if workspace.tool_panels:
        width = min(390 * workspace.s, size.x * 0.27)
        _, tool, rest_root = ii.dock_builder_split_node_py(
            rest_root, imgui.Dir.right, width / size.x
        )
        for name in workspace.tool_panels:
            ii.dock_builder_dock_window(name, tool)
        size = imgui.ImVec2(size.x - width, size.y)
    left_width = min(DESIGN.scene_width * workspace.s, size.x * 0.26)
    right_width = min(DESIGN.inspector_width * workspace.s, size.x * 0.3)
    _, workspace.left, rest = ii.dock_builder_split_node_py(
        rest_root, imgui.Dir.left, left_width / size.x
    )
    _, workspace.right, workspace.center = ii.dock_builder_split_node_py(
        rest, imgui.Dir.right, right_width / (size.x - left_width)
    )
    if workspace.bottom_open:
        _, workspace.bottom, workspace.center = ii.dock_builder_split_node_py(
            workspace.center, imgui.Dir.down, min(0.68, DESIGN.bottom_height * workspace.s / size.y)
        )
        for name in BOTTOM_PANELS:
            ii.dock_builder_dock_window(name, workspace.bottom)
    for name in LEFT_PANELS:
        ii.dock_builder_dock_window(name, workspace.left)
    ii.dock_builder_dock_window("Inspector", workspace.right)
    ii.dock_builder_dock_window("Viewport", workspace.center)
    ii.dock_builder_finish(workspace.window.dockspace_id)
    workspace.needs_layout = False
    workspace.timeline_layout_pending = False


def toggle_timeline(workspace):
    if workspace.bottom_open and workspace.bottom_active == "Timeline":
        workspace.collapse_bottom()
    else:
        workspace.activate("Timeline")


def collapse_bottom(workspace):
    for name in BOTTOM_PANELS:
        workspace.open[name] = False
    workspace.bottom_focus = None
    workspace.timeline_layout_pending = True


def apply_timeline_layout(workspace):
    if not workspace.timeline_layout_pending:
        return
    workspace.timeline_layout_pending = False
    if workspace.bottom_open:
        for name in BOTTOM_PANELS:
            window = imgui.internal.find_window_by_name(name)
            if window is not None and imgui.internal.dock_builder_get_node(window.dock_id):
                workspace.bottom = window.dock_id
                break
        else:
            workspace.bottom = None
        window = imgui.internal.find_window_by_name("Viewport")
        node = window.dock_node if window is not None else None
        if workspace.bottom is None and node is not None:
            _, workspace.bottom, workspace.center = imgui.internal.dock_builder_split_node_py(
                node.id_,
                imgui.Dir.down,
                min(0.68, DESIGN.bottom_height * workspace.s / node.size.y),
            )
        if workspace.bottom is not None:
            for name in BOTTOM_PANELS:
                imgui.internal.dock_builder_dock_window(name, workspace.bottom)
            imgui.internal.dock_builder_finish(workspace.window.dockspace_id)


def reset_visibility(workspace):
    """Reset local panel state; native docking is rebuilt by the next frame."""
    for name in workspace.open:
        panel = workspace.app.panels.get(name)
        workspace.open[name] = name == "Scene" and panel is not None and panel.enabled
    inspector = workspace.app.panels.get("Inspector")
    workspace.open.update(Inspector=inspector is not None and inspector.enabled, Timeline=False)
    workspace.left_active = workspace.active = "Scene"
    workspace.float_next = False
    workspace.focus_next = "Scene"
    workspace.bottom_focus = None
    workspace.restore_next = False
    workspace.pending = None
    workspace.needs_layout = True
    workspace.timeline_layout_pending = False


def command(workspace, action):
    ii = imgui.internal
    if action == "reset":
        workspace.app.reset_layout()
    elif action == "float":
        workspace.open["Inspector"] = True
        workspace.float_next = True
        workspace.focus_next = "Inspector"
    elif action in ("dock", "group"):
        if action == "group":
            target = ii.find_window_by_name("Scene").dock_id
        else:
            root = ii.dock_builder_get_node(workspace.window.dockspace_id)
            current = ii.find_window_by_name("Inspector").dock_node
            if (
                current
                and abs(current.pos.x + current.size.x - root.pos.x - root.size.x) < 2
                and abs(current.size.y - root.size.y) < 2
            ):
                target = current.id_
            else:
                _, target, _ = ii.dock_builder_split_node_py(
                    workspace.window.dockspace_id,
                    imgui.Dir.right,
                    DESIGN.inspector_width * workspace.s / root.size.x,
                )
        ii.dock_builder_dock_window("Inspector", target)
        ii.dock_builder_finish(workspace.window.dockspace_id)
        workspace.focus_next = "Inspector"
    elif action == "save":
        workspace.saved = imgui.save_ini_settings_to_memory()
    elif action == "restore" and workspace.saved:
        workspace.restore_next = True
    else:
        raise ValueError(action)


def inspector_grouped_with(workspace, name):
    window = imgui.internal.find_window_by_name(name)
    inspector = imgui.internal.find_window_by_name("Inspector")
    return bool(
        workspace.open["Inspector"]
        and window is not None
        and inspector is not None
        and window.dock_id
        and window.dock_id == inspector.dock_id
    )
