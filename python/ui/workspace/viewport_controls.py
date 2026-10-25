"""Measured viewport camera and transform controls sharing catalogue geometry."""

from functools import lru_cache

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.ui.design_palette import rgba8
from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace.component_layout import (
    FlowLayout,
    component_flow,
    control_frame_size,
    control_size,
)
from mojive.ui.workspace.constants import VIEWPORT_TOOLS
from mojive.ui.workspace.group_corners import action_corner_scope, action_corners, inset_action_box
from mojive.ui.workspace.tooltips import ShortcutHint
from mojive.ui.workspace.widgets import DIM, color

BACKGROUND = rgba8("chrome", 230)


def toolbar_group_gap(style, scale, gap):
    """Replace one ordinary item gap with the divider and its two clearances."""
    thickness = style.get("separatorThickness", 1) * scale
    return thickness + 2 * style.get("separatorGap", 6) * scale if thickness > 0 else gap


def toolbar_separator(ui, style, before, after, *, horizontal):
    """Paint the divider inside the same space reserved by toolbar measurement."""
    thickness = style.get("separatorThickness", 1) * ui.s
    if thickness <= 0:
        return None
    gap, inset = style.get("separatorGap", 6) * ui.s, style.get("separatorInset", 6) * ui.s
    if horizontal:
        top, bottom = max(before.y, after.y), min(before.bottom, after.bottom)
        rect = Rect(before.right + gap, top + inset, thickness, max(0, bottom - top - 2 * inset))
    else:
        left, right = max(before.x, after.x), min(before.right, after.right)
        rect = Rect(left + inset, before.bottom + gap, max(0, right - left - 2 * inset), thickness)
    if rect.width > 0 and rect.height > 0:
        ink = ui.style_color(style, "separatorColor", ui.palette("border"), inherit_alpha=False)
        ui.draw.add_rect_filled((rect.x, rect.y), (rect.right, rect.bottom), color(ink))
    return rect


@lru_cache(maxsize=256)
def _toolbox_layout(width, height, gap, group_gap, horizontal, wrapped):
    columns = 3 if wrapped else 6 if horizontal else 1
    extra = group_gap - gap
    rectangles = tuple(
        Rect(
            (n % columns) * (width + gap) + (extra if horizontal and n >= 4 else 0),
            (n // columns) * (height + gap) + (extra if not horizontal and n >= 4 else 0),
            width,
            height,
        )
        for n in range(6)
    )
    return FlowLayout(
        rectangles, max(r.right for r in rectangles), max(r.bottom for r in rectangles)
    )


@part_scope("toolbox")
def toolbox_layout(ui, *, horizontal=False, wrapped=False, extent=None):
    """Measure uniform tool slots once, then reuse their extent for each orientation."""
    style, scale = ui.component_style("toolbox"), ui.s
    if extent is None:
        slot = style.get("slotSize", 36)
        sizes = tuple(
            control_size(
                ui, "tool-" + name, slot, slot, icon=name, icon_size=style.get("iconSize", 20)
            )
            for name in ("select", "move", "rotate", "scale", "frame", "snap")
        )
        extent = max(v[0] for v in sizes), max(v[1] for v in sizes)
    gap = style.get("rowGap", style.get("gap", 4)) * scale
    return _toolbox_layout(*extent, gap, toolbar_group_gap(style, scale, gap), horizontal, wrapped)


@part_scope("viewport_toolbar")
@action_corner_scope("viewport_toolbar", (4,) * 4)
def camera_controls(workspace, rect, overlay):
    ui, scale = workspace.ui, workspace.s
    x, y, width, _ = rect
    style = ui.component_style("viewport_toolbar")
    compact = width < 240 * scale
    gap = style.get("columnGap", style.get("gap", 4)) * scale
    corners = action_corners(ui, "viewport_toolbar", (4,) * 4)
    padding = corners.padding
    label = "" if compact else "Ortho" if workspace.app.camera.orthographic else "Persp"
    projection_icon = "orthographic" if workspace.app.camera.orthographic else "perspective"
    sizes = [
        control_size(
            ui,
            "projection-button",
            28 if compact else 98,
            28,
            label=label,
            icon=projection_icon,
            icon_role="toolbar",
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
        resolved_padding=padding,
    )
    ui.record_component_box(
        "viewport_toolbar",
        box,
        key="viewport-toolbar",
        fields=[list(r.tuple()) for r in layout.rectangles],
        anchorRect=list(rect),
        radius=corners.radius,
        childRadius=corners.inner_radius,
    )
    ui.rect(*box.frame.tuple(), BACKGROUND, corners.radius / scale)
    overlay.append(box.outer.tuple())
    p = layout.rectangles[0]
    frame = control_frame_size(
        ui, "projection-button", p.width, p.height, icon=projection_icon, label=label
    )
    camera_clicked = ui.button(
        "projection-button",
        p.x,
        p.y,
        *frame,
        icon=projection_icon,
        icon_role="toolbar",
        label="" if compact else "Ortho" if workspace.app.camera.orthographic else "Persp",
        tooltip="Projection" if compact else None,
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


@part_scope("toolbox")
@action_corner_scope("toolbox", (4,) * 4)
def toolbox(workspace, rect, top, horizontal, wrapped, overlay):
    ui, scale, app = workspace.ui, workspace.s, workspace.app
    x, _, width, _ = rect
    style = ui.component_style("toolbox")
    corners = action_corners(ui, "toolbox", (4,) * 4)
    padding = corners.padding
    margins = style.edges("margin", scale)
    layout = toolbox_layout(ui, horizontal=horizontal, wrapped=wrapped)
    box = ui.component_box(
        "toolbox",
        x + style.get("left", 10) * scale,
        top,
        min(width - 20 * scale, layout.width + padding[0] + padding[2] + margins[0] + margins[2]),
        layout.height + padding[1] + padding[3] + margins[1] + margins[3],
        key="viewport-toolbox",
        padding=padding,
        observe=False,
    )
    box = inset_action_box(box, corners)
    ui.rect(*box.frame.tuple(), BACKGROUND, corners.radius / scale)
    overlay.append(box.outer.tuple())
    positions = tuple(
        Rect(box.content.x + r.x, box.content.y + r.y, r.width, r.height) for r in layout.rectangles
    )
    divider = toolbar_separator(ui, style, positions[3], positions[4], horizontal=horizontal)
    if ui.design is not None and ui.design.runtime.observes(style.part_ids):
        ui.record_component_box(
            "toolbox",
            box,
            key="viewport-toolbox",
            fields=[list(r.tuple()) for r in positions],
            anchorRect=list(rect),
            radius=corners.radius,
            childRadius=corners.inner_radius,
            separators=[list(divider.tuple())] if divider is not None else [],
            fieldRegions={
                field: list(divider.tuple())
                for field in (
                    "separatorThickness",
                    "separatorInset",
                    "separatorGap",
                    "separatorColor",
                )
            }
            if divider is not None
            else {},
        )
    for n, (name, action) in enumerate(VIEWPORT_TOOLS):
        p = positions[n]
        frame = control_frame_size(ui, "tool-" + name, p.width, p.height, icon=name)
        shortcut = "V" if action is None else app.input_bindings.label(action)
        tip = "Dimensions" if name == "scale" else name.title()
        if ui.button(
            "tool-" + name,
            p.x,
            p.y,
            *frame,
            icon=name,
            icon_size=style.get("iconSize", 20),
            active=app.tool == name,
            solid=True,
            tooltip=ShortcutHint(tip, shortcut),
            width_limit=frame[0] * scale,
        ):
            app.tool = name
    for index, key, icon, tooltip in (
        (
            4,
            "frame",
            "tool-body" if workspace.space == "Body" else "tool-world",
            ShortcutHint("World / body frame", app.input_bindings.label(InputAction.GIZMO_SPACE)),
        ),
        (5, "snap", "snap", ShortcutHint("Snap", app.input_bindings.label(InputAction.SNAP))),
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
