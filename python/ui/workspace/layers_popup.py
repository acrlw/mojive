"""Compact viewport visibility groups using shared native switches."""

from dataclasses import replace
from math import ceil

from imgui_bundle import imgui

from mojive.config import ViewportLayers
from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.ui.layers import debug_layer_group

from .floating import floating_style
from .form_popup import form_corners
from .menu_geometry import native_unit
from .property_style import advance, popup_row_box, switch_row_box
from .widgets import MUTED, TEXT
from .window_chrome import draw_floating_shadow

POPUP_ID = "Viewport layers###viewport-layers-popup"

_GROUPS = (
    ("Scene display", (("geometry", "Scene geometry"),)),
    (
        "Editing aids",
        (
            ("viewport_ui", "Viewport controls"),
            ("gizmos", "Gizmos"),
            ("helpers", "Camera/light helpers"),
            ("influence", "Selected frustum / light range"),
            ("view_cube", "View cube"),
            ("perturbation", "Perturbation guides"),
        ),
    ),
    ("Selection feedback", (("selection", "Selection highlight"),)),
)


def layers_popup_width(ui):
    """Fit localized captions and the actual switch, including both inset levels."""
    scale, style = ui.s, ui.design.primitives.style("switch")
    font = style.get("fontSize", 14)
    label = max(
        ui.measure(label, font, style=style, font_resolved=True)
        for _title, controls in _GROUPS
        for _name, label in controls
    )
    content = popup_row_box(ui, Rect(0, 0, 1000, 28 * scale)).padding
    padding = form_corners(ui, "layers_popup", control_part="input_text").padding
    width = (
        label
        + style.get("columnGap", style.get("gap", 8)) * scale
        + style.get("switchWidth", 36) * scale
        + content[0]
        + content[2]
        + padding[0]
        + padding[2]
    )
    heading = ui.measure("Viewport layers", 14, bold=True, font_resolved=True)
    width = max(width, heading + (28 + 10) * scale + sum(content[::2]) + sum(padding[::2]))
    return native_unit(ceil(width / (2 * scale)) * 2 * scale)


def draw_layers_popup(workspace):
    ui, scale, app = workspace.ui, workspace.s, workspace.app
    key = POPUP_ID
    if getattr(workspace, "layers_popup_requested", False):
        anchor = ui.hits.get("viewport-layers")
        if anchor is not None:
            imgui.set_next_window_pos((anchor[0], anchor[1] + anchor[3] + 4 * scale))
        imgui.open_popup(key)
        workspace.layers_popup_requested = False
    with group_scope("Layers"), floating_style(ui, part="layers_popup"):
        viewport = imgui.get_main_viewport()
        width = min(layers_popup_width(ui), viewport.work_size.x - 24 * scale)
        imgui.set_next_window_size_constraints(
            (width, 0), (width, viewport.work_size.y - 24 * scale)
        )
        if not imgui.begin_popup(key):
            return
        draw_floating_shadow(ui)
        pos, width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
        x, y = pos.x, pos.y
        inset = popup_row_box(ui, Rect(x, y, width, 28 * scale)).content
        ui.component_text(
            "layers_popup",
            Rect(inset.x, y, width - (inset.x - x) - 38 * scale, 32 * scale),
            "Viewport layers",
            TEXT,
            size=14,
            bold=True,
        )
        window = imgui.internal.get_current_window()
        radius = max(0, window.window_rounding - max(window.window_padding) - 2 * scale)
        with ui.corner_scope(radius, ("button", "icon_button")):
            reset = ui.button(
                "layers-reset",
                x + width - 30 * scale,
                y + 2 * scale,
                28,
                28,
                icon="reset",
                tooltip="Reset viewport layers",
                icon_size=20,
            )
        if reset:
            workspace._panel_context.set_viewport_layers(ViewportLayers())
            app.scene_entities.show_influence = True
        y += 40 * scale
        groups = list(_GROUPS)
        # Publisher diagnostics belong here only when the scene actually has them.
        draw = app.backend.debug
        if draw is not None and any(
            layer.primitives and debug_layer_group(layer.name) in ("debug_3d", "debug_2d")
            for layer in draw.layers()
        ):
            groups.append(
                ("Debug display", (("debug_3d", "3D drawings"), ("debug_2d", "2D drawings")))
            )
        for index, (title, controls) in enumerate(groups):
            if index:
                y += 8 * scale
                ui.line(x, y, width)
                y += 8 * scale
            heading = switch_row_box(ui, Rect(x, y, width, 26 * scale)).content
            ui.text(heading.x, y + 3 * scale, title, MUTED, 12)
            y += 26 * scale
            for name, label in controls:
                layers = app.viewport_layers
                value = (
                    app.scene_entities.show_influence
                    if name == "influence"
                    else getattr(layers, name)
                )
                row = Rect(x, y, width, 28 * scale)
                result = ui.design.primitives.switch(
                    "layer-" + name,
                    row,
                    value,
                    label=label,
                    enabled=name != "influence" or layers.helpers,
                    layout_box=switch_row_box(ui, row),
                )
                if result.value != value:
                    if name == "influence":
                        app.scene_entities.show_influence = result.value
                    else:
                        workspace._panel_context.set_viewport_layers(
                            replace(layers, **{name: result.value})
                        )
                y += 28 * scale
        advance(ui, x, y)
        window = imgui.internal.get_current_window()
        ui.hits["viewport-layers-popup"] = (*window.pos, *window.size)
        imgui.end_popup()
