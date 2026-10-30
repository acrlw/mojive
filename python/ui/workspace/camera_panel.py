"""Native presentation of the registered camera panel's shared operations."""

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.ui.localization import camera_caption
from mojive.ui.workspace.component_layout import component_flow, control_frame_size, control_size

from .camera_bookmarks import camera_bookmarks
from .design import DESIGN
from .property_style import NativePropertyPresentation, advance, property_controls


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
            tooltip=ui.translate(name),
            translate=False,
            width_limit=frame[0] * scale,
            filled=True,
        ):
            if name == "Frame all":
                workspace.app._frame_scene(animate=True)
            else:
                workspace.app.camera.set_preset(name.lower())
    return box.outer.bottom


def camera_panel(workspace, x, y, width):
    ui, scale, ctx = workspace.ui, workspace.s, workspace._panel_context
    panel = workspace.app.panels.get("camera")
    if panel.presentation is None:
        panel.presentation = NativePropertyPresentation(ui, prefix="camera", fit_labels=True)
    with property_controls(ui):
        imgui.set_cursor_screen_pos((x, y))
        panel._source(ctx)
        model_camera = ctx.model_camera_id >= 0 and ctx.model_camera_view is not None
        if panel._section(ctx.tr("VIEW DIRECTION"), imgui.TreeNodeFlags_.default_open):
            pos = imgui.get_cursor_screen_pos()
            imgui.begin_disabled(model_camera)
            bottom = _view_directions(workspace, x, pos.y, width)
            imgui.end_disabled()
            advance(ui, x, bottom + DESIGN.row_gap * scale)
        if model_camera:
            pos = imgui.get_cursor_screen_pos()
            if ui.button(
                "camera-editor",
                pos.x,
                pos.y,
                width / scale,
                DESIGN.control_height,
                label="Return to Editor Camera",
                filled=True,
            ):
                ctx.select_model_camera(-1)
        else:
            panel._params(ctx, ctx.camera)
            panel._tracking(ctx)
        camera_bookmarks(workspace, panel, ctx)
