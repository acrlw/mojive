"""Read-only inventory of the resources referenced by the current scene."""

from imgui_bundle import imgui

from .design import DESIGN
from .property_style import advance, panel_empty, property_section
from .widgets import MUTED


def resource_panel(workspace, x, y, width):
    ui, scale = workspace.ui, workspace.s
    session = workspace.app.session
    panel = workspace.app.panels.get("assets")
    generation = (session.document_id, session.structure_generation)
    if getattr(workspace, "_resource_generation", None) != generation:
        workspace._resource_generation = generation
        workspace._resource_inventory = tuple(
            (model, tuple(item for item in session.model_assets(model.model_id) if item.references))
            for model in session.scene_models
        )
    _, panel._filter = ui.search(
        "##asset-filter", x, y, width, panel._filter, hint=ui.translate("Search resources")
    )
    y = imgui.get_item_rect_max().y + DESIGN.row_gap * scale
    needle = panel._filter.strip().casefold()
    visible = 0
    for model, assets in workspace._resource_inventory:
        model_match = needle in model.name.casefold() or needle in str(model.path).casefold()
        matches = tuple(
            item
            for item in assets
            if model_match or needle in f"{item.name} {item.type} {item.file}".casefold()
        )
        if needle and not model_match and not matches:
            continue
        imgui.set_cursor_screen_pos((x, y))
        if not property_section(
            ui,
            model.name,
            imgui.TreeNodeFlags_.default_open,
            item_id=f"resources-model-{model.model_id}",
            divider=visible > 0,
        ):
            y = imgui.get_cursor_screen_pos().y
            visible += 1
            continue
        y = imgui.get_cursor_screen_pos().y
        key = f"resource-source-{model.model_id}"
        source = model.path.name
        if ui.button(
            key,
            x,
            y,
            width / scale,
            28,
            label=source,
            icon="file",
            align="left",
            translate=False,
            tooltip=str(model.path),
        ):
            imgui.set_clipboard_text(str(model.path))
        y = ui.hits[key][1] + ui.hits[key][3] + 4 * scale
        for item in matches:
            key = f"resource-{model.model_id}-{item.type}-{item.index}"
            selected = panel._selected == (model.model_id, item.type, item.name)
            icon = (
                "shading"
                if item.type == "material"
                else "capture-image"
                if item.type == "texture"
                else "file"
            )
            if ui.button(
                key,
                x + 12 * scale,
                y,
                max(1, width / scale - 12),
                28,
                label=item.name,
                icon=icon,
                align="left",
                active=selected,
                translate=False,
                tooltip=item.file or item.name,
            ):
                panel._selected = (model.model_id, item.type, item.name)
            y = ui.hits[key][1] + ui.hits[key][3] + 2 * scale
            if selected:
                text = item.file or ui.translate(item.type)
                ui.component_text(
                    "control_metadata",
                    (x + 24 * scale, y, width - 24 * scale, 24 * scale),
                    text,
                    MUTED,
                    translate=False,
                )
                y += 24 * scale
        visible += 1
        y += DESIGN.row_gap * scale
    if not visible:
        y = panel_empty(
            ui,
            "resources-empty",
            "No matching resources" if needle else "No file resources in this scene",
            x,
            y,
            width,
        )
    advance(ui, x, y)
