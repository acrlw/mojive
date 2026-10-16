"""Measured viewport camera and transform controls sharing catalogue geometry."""

from imgui_bundle import imgui

from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace.component_layout import (
    component_flow,
    control_frame_size,
    control_size,
    flow_layout,
)
from mojive.ui.workspace.constants import VIEWPORT_TOOLS
from mojive.ui.workspace.widgets import DIM

BACKGROUND = (22, 25, 28, 230)


def camera_controls(workspace, rect, overlay):
    ui, scale = workspace.ui, workspace.s
    x, y, width, _ = rect
    style = ui.component_style("viewport_toolbar")
    compact = width < 240 * scale
    gap = style.get("columnGap", style.get("gap", 4)) * scale
    padding = style.edges("padding", scale, (3, 4, 3, 4))
    label = "" if compact else "Ortho" if workspace.app.camera.orthographic else "Persp"
    sizes = [
        control_size(
            ui, "projection-button", 28 if compact else 98, 28, label=label, icon="camera"
        ),
        control_size(ui, "shading", 28, 28, icon="shading"),
    ]
    if width >= 300 * scale:
        sizes.append(control_size(ui, "viewport-layers", 28, 28, icon="layers"))
    natural = sum(size[0] for size in sizes) + (len(sizes) - 1) * gap
    margins = style.edges("margin", scale)
    budget = min(width - 20 * scale, natural + padding[0] + padding[2] + margins[0] + margins[2])
    box, layout = component_flow(
        ui,
        "viewport_toolbar",
        x + style.get("left", 10) * scale,
        y + style.get("top", 10) * scale,
        budget,
        sizes,
        key="viewport-toolbar",
        padding=padding,
        column_gap=gap / scale,
        row_gap=4,
        observe=False,
    )
    ui.record_component_box(
        "viewport_toolbar",
        box,
        key="viewport-toolbar",
        fields=[list(r.tuple()) for r in layout.rectangles],
        anchorRect=list(rect),
    )
    ui.rect(*box.frame.tuple(), BACKGROUND, 8)
    overlay.append(box.outer.tuple())
    p = layout.rectangles[0]
    frame = control_frame_size(
        ui, "projection-button", p.width, p.height, icon="camera", label=label
    )
    camera_clicked = ui.button(
        "projection-button",
        p.x,
        p.y,
        *frame,
        icon="camera",
        label="" if compact else "Ortho" if workspace.app.camera.orthographic else "Persp",
        tooltip="Camera" if compact else None,
        active=imgui.is_popup_open("Camera"),
        align="left",
        width_limit=frame[0] * scale,
    )
    if not compact:
        ui.role_icon("down", p.right - 19 * scale, p.y, 12 * scale, p.height, role="inline", c=DIM)
    workspace.menus.popup("Camera", ui.hits["projection-button"], camera_clicked)
    p = layout.rectangles[1]
    frame = control_frame_size(ui, "shading", p.width, p.height, icon="shading")
    clicked = ui.button(
        "shading",
        p.x,
        p.y,
        *frame,
        icon="shading",
        tooltip="Shading",
        active=imgui.is_popup_open("Shading"),
        width_limit=frame[0] * scale,
    )
    workspace.menus.popup("Shading", ui.hits["shading"], clicked)
    if len(layout.rectangles) > 2:
        p = layout.rectangles[2]
        frame = control_frame_size(ui, "viewport-layers", p.width, p.height, icon="layers")
        if ui.button(
            "viewport-layers",
            p.x,
            p.y,
            *frame,
            icon="layers",
            tooltip="Viewport layers",
            width_limit=frame[0] * scale,
        ):
            workspace.activate("Layers")


def toolbox(workspace, rect, top, horizontal, wrapped, overlay):
    ui, scale, app = workspace.ui, workspace.s, workspace.app
    x, _, width, _ = rect
    style = ui.component_style("toolbox")
    slot = style.get("slotSize", 36)
    gap = style.get("rowGap", style.get("gap", 4)) * scale
    padding = style.edges("padding", scale, (4, 4, 4, 4))
    columns = 3 if wrapped else 6 if horizontal else 1
    rows = 2 if wrapped else 1 if horizontal else 6
    child_sizes = tuple(
        control_size(ui, "tool-" + name, slot, slot, icon=name, icon_size=style.get("iconSize", 20))
        for name in ("select", "move", "rotate", "scale", "frame", "snap")
    )
    child_width = max(v[0] for v in child_sizes)
    child_height = max(v[1] for v in child_sizes)
    content_width = columns * child_width + (columns - 1) * gap
    content_height = rows * child_height + (rows - 1) * gap
    box = ui.component_box(
        "toolbox",
        x + style.get("left", 10) * scale,
        top,
        min(width - 20 * scale, content_width + padding[0] + padding[2]),
        content_height + padding[1] + padding[3],
        key="viewport-toolbox",
        padding=padding,
        observe=False,
    )
    ui.rect(*box.frame.tuple(), BACKGROUND, style.get("radius", 8))
    overlay.append(box.outer.tuple())
    if horizontal:
        positions = flow_layout(box.content, child_sizes, column_gap=gap, row_gap=gap).rectangles
    else:
        from mojive.design.native.geometry import Rect

        positions = tuple(
            Rect(
                box.content.x,
                box.content.y + n * (child_height + gap),
                min(box.content.width, child_width),
                child_height,
            )
            for n in range(6)
        )
    ui.record_component_box(
        "toolbox",
        box,
        key="viewport-toolbox",
        fields=[list(r.tuple()) for r in positions],
        anchorRect=list(rect),
    )
    for n, (name, action) in enumerate(VIEWPORT_TOOLS):
        p = positions[n]
        frame = control_frame_size(ui, "tool-" + name, p.width, p.height, icon=name)
        shortcut = "V" if action is None else app.input_bindings.label(action)
        tip = "Dimensions: select a geometry" if name == "scale" else name.title()
        if shortcut != "Unbound":
            tip += " · " + shortcut
        if ui.button(
            "tool-" + name,
            p.x,
            p.y,
            *frame,
            icon=name,
            icon_size=style.get("iconSize", 20),
            active=app.tool == name,
            solid=True,
            tooltip=tip,
            width_limit=frame[0] * scale,
        ):
            app.tool = name
    for index, key, icon, tooltip in (
        (
            4,
            "frame",
            "body" if workspace.space == "Body" else "world",
            "World / body frame · " + app.input_bindings.label(InputAction.GIZMO_SPACE),
        ),
        (5, "snap", "snap", "Snap · " + app.input_bindings.label(InputAction.SNAP)),
    ):
        p = positions[index]
        frame = control_frame_size(ui, key, p.width, p.height, icon=icon)
        if ui.button(
            key,
            p.x,
            p.y,
            *frame,
            icon=icon,
            icon_size=style.get("iconSize", 20),
            active=workspace.snap if key == "snap" else False,
            tooltip=tooltip,
            width_limit=frame[0] * scale,
        ):
            if key == "snap":
                workspace.snap = not workspace.snap
            else:
                workspace.space = "Body" if workspace.space == "World" else "World"
