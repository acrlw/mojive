"""Native presentation of the registered camera panel's shared operations."""

from imgui_bundle import imgui

from .design import DESIGN
from .property_style import NativePropertyPresentation, advance, property_controls


def camera_panel(workspace, x, y, width):
    ui, scale, ctx = workspace.ui, workspace.s, workspace._panel_context
    panel = workspace.app.panels.get("camera")
    if panel.presentation is None:
        panel.presentation = NativePropertyPresentation(ui, prefix="camera")
    with property_controls(ui):
        imgui.set_cursor_screen_pos((x, y))
        panel._source(ctx)
        model_camera = ctx.model_camera_id >= 0 and ctx.model_camera_view is not None
        if panel._section(ctx.tr("VIEW DIRECTION"), imgui.TreeNodeFlags_.default_open):
            pos = imgui.get_cursor_screen_pos()
            gap = DESIGN.item_spacing_x * scale
            cell_width = (width - gap) / 2
            imgui.begin_disabled(model_camera)
            for index, name in enumerate(("Front", "Back", "Left", "Right", "Top", "Bottom")):
                if ui.button(
                    "camera-" + name,
                    x + (index % 2) * (cell_width + gap),
                    pos.y + (index // 2) * (DESIGN.control_height + DESIGN.row_gap) * scale,
                    cell_width / scale,
                    DESIGN.control_height,
                    icon="camera",
                    label=name,
                    align="center",
                    filled=True,
                ):
                    workspace.app.camera.set_preset(name.lower())
            frame_y = pos.y + 3 * (DESIGN.control_height + DESIGN.row_gap) * scale
            if ui.button(
                "camera-frame",
                x,
                frame_y,
                width / scale,
                DESIGN.control_height,
                icon="frame",
                label="Frame all",
                filled=True,
            ):
                workspace.app._frame_scene(animate=True)
            imgui.end_disabled()
            advance(ui, x, frame_y + (DESIGN.control_height + DESIGN.row_gap) * scale)
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
        panel._stored_states(ctx, ctx.camera)
