"""Named camera views with explicit restore and storage actions."""

from imgui_bundle import imgui

from mojive.design.native.geometry import part_scope
from mojive.scene.state import list_named_snapshots

from .component_layout import component_flow, control_frame_size, control_size
from .design import DESIGN
from .property_style import advance
from .widgets import MUTED


def camera_bookmarks(workspace, panel, ctx):
    if not panel._section(f"{ctx.tr('camera bookmarks')}###camera_states"):
        return
    _bookmark_content(workspace, panel, ctx)


@part_scope("button_row")
def _bookmark_content(workspace, panel, ctx):
    ui, scale = workspace.ui, workspace.s
    pos = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    style = ui.component_style("button_row")
    padding, margin = style.edges("padding", scale), style.edges("margin", scale)
    available = max(scale, width - padding[0] - padding[2] - margin[0] - margin[2])
    gap = style.get("columnGap", style.get("gap", DESIGN.item_spacing_x)) * scale
    action_size = control_size(ui, "camera-bookmark-save", 28, 28, icon="save")
    input_style = ui.component_style("input_text")
    input_margin = input_style.edges("margin", scale)
    input_height = (
        ui.design.primitives.field_height("input_text", DESIGN.control_height)
        if ui.design is not None
        else input_style.dimension("height", DESIGN.control_height)
    ) * scale
    input_height += input_margin[1] + input_margin[3]
    names = list_named_snapshots(panel.bookmark_directory)
    field_width = max(scale, available - action_size[0] - gap)
    # A narrow column moves the save action to its own row rather than crushing the name.
    if field_width < 96 * scale:
        field_width = available
    sizes = [(field_width, input_height), action_size]
    for name in names:
        sizes.append(
            control_size(
                ui,
                "camera-bookmark-open/" + name,
                max(scale, available - action_size[0] - gap) / scale,
                32,
                label=name,
                icon="camera",
            )
        )
        sizes.append(action_size)
    box, flow = component_flow(
        ui,
        "button_row",
        pos.x,
        pos.y,
        width,
        sizes,
        key="camera-bookmarks",
        column_gap=DESIGN.item_spacing_x,
        row_gap=DESIGN.row_gap,
    )
    field, save = flow.rectangles[:2]
    submitted, panel._bookmark_name = ui.input_text(
        "camera-bookmark-name",
        field.x,
        field.y,
        field.width,
        panel._bookmark_name,
        hint=ctx.tr("Bookmark name"),
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    fw, fh = control_frame_size(ui, "camera-bookmark-save", save.width, save.height, icon="save")
    clicked = ui.button(
        "camera-bookmark-save",
        save.x,
        save.y,
        fw,
        fh,
        icon="save",
        filled=True,
        tooltip="Save camera bookmark",
        width_limit=fw * scale,
    )
    if submitted or clicked:
        panel.save_bookmark(ctx, ctx.camera)
    for index, name in enumerate(names):
        row, more = flow.rectangles[2 + index * 2 : 4 + index * 2]
        key = "camera-bookmark-open/" + name
        fw, fh = control_frame_size(ui, key, row.width, row.height, label=name, icon="camera")
        if ui.button(
            key,
            row.x,
            row.y,
            fw,
            fh,
            label=name,
            icon="camera",
            align="start",
            filled=True,
            translate=False,
            width_limit=fw * scale,
        ):
            panel.load_bookmark(ctx, ctx.camera, name)
        popup = "camera-bookmark-actions/" + name
        fw, fh = control_frame_size(ui, popup, more.width, more.height, icon="more")
        if ui.button(
            popup,
            more.x,
            more.y,
            fw,
            fh,
            icon="more",
            tooltip="Bookmark actions",
            width_limit=fw * scale,
        ):
            imgui.open_popup(popup)
        with workspace.menus.local_popup(popup) as opened:
            if opened:
                for action, label, icon in (
                    ("restore", "Restore view", "camera"),
                    ("update", "Update from current view", "save"),
                    ("copy", "Copy", "copy"),
                    ("delete", "Delete", "delete"),
                ):
                    clicked, _ = workspace.menus.local_item(
                        label,
                        key=popup + "/" + action,
                        icon=icon,
                    )
                    if clicked:
                        if action == "restore":
                            panel.load_bookmark(ctx, ctx.camera, name)
                        elif action == "update":
                            panel.save_bookmark(ctx, ctx.camera, replace_name=name)
                        elif action == "copy":
                            panel.copy_bookmark(ctx, name)
                        else:
                            panel.delete_bookmark(ctx, name)
    bottom = box.outer.bottom
    if not names:
        bottom += DESIGN.row_gap * scale
        ui.text(box.content.x, bottom, "No camera bookmarks", MUTED)
        bottom += 20 * scale
    advance(ui, pos.x, bottom + DESIGN.row_gap * scale)
    if panel._bookmark_error:
        imgui.text_wrapped(panel._bookmark_error)
        if panel._bookmark_button(ctx, "Copy error", small=True):
            imgui.set_clipboard_text(panel._bookmark_error)
