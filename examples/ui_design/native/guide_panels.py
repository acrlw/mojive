"""Feature-sized panel proposals composed from production UI primitives."""

import math
from dataclasses import fields
from types import SimpleNamespace

from imgui_bundle import imgui as g

from mojive.design.groups import group_scope
from mojive.design.native.geometry import part_scope
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.viewport_widgets.hints import default_tool_hints
from mojive.ui.viewport_widgets.model import ViewportLabels
from mojive.ui.viewport_widgets.status import draw_status
from mojive.ui.workspace.component_layout import flow_layout
from mojive.ui.workspace.filter_chip import draw_filter_chip, filter_chip_size
from mojive.ui.workspace.material_library import draw_material_library
from mojive.ui.workspace.status_components import (
    StatusBadge,
    StatusGroups,
    resolve_status_bar_layout,
)

from .guide_system import GROUP_CONTAINERS, GROUPS

SAMPLES = {
    "Settings navigation": (
        ("General", "general"),
        ("Camera", "camera"),
        ("Interaction", "mouse"),
        ("Rendering", "shading"),
        ("Recording", "video"),
        ("Content visualization", "mujoco-visuals"),
    ),
    "Scene tree": (
        ("world", "world"),
        ("free_body", "body"),
        ("free_box", "geom"),
        ("hinge_body", "body"),
    ),
    "Scene filters": (
        ("52", "scene"),
        ("11", "body"),
        ("20", "geom"),
        ("11", "joint"),
        ("1", "camera"),
        ("6", "robot"),
    ),
    "Inspector tabs": (
        ("Properties", None),
        ("Physics", None),
        ("Material", None),
        ("Joint", None),
    ),
    "Timeline actions": (
        ("Play", "play"),
        ("Record simulation state", "record"),
        ("Snapshot", "snapshot"),
    ),
}


def draw_group_demo(v, group, x, y, width):
    family = {
        "Settings navigation": "settings",
        "Scene tree": "navigation",
        "Scene filters": "navigation",
        "Inspector tabs": "inspector",
        "Timeline actions": "timeline",
        "Viewport tools": "viewport",
        "Render metric": "chrome",
    }.get(group)
    with v.ui.design.family_scope(family):
        return _draw_group_demo(v, group, x, y, width)


def _draw_group_demo(v, group, x, y, width):
    if group == "Settings navigation":
        from mojive.ui.workspace.settings_layout import draw_navigation
        from mojive.ui.workspace.settings_panel import CATEGORIES

        workspace = v.sample.app.workspace
        workspace.settings._category = CATEGORIES[v.selected.get(group, 0)]
        rect = v.rect(x, y, width, 240)
        with group_scope("Settings"):
            _, height = draw_navigation(
                workspace.settings, v.sample.context, rect, False, fit_content=True
            )
        v.selected[group] = CATEGORIES.index(workspace.settings._category)
        for index, category in enumerate(CATEGORIES):
            v.ui.hits[f"sample/{group}/{index}"] = v.ui.hits["settings-" + category]
        return y + height / v.s
    if group == "Scene tree":
        from mojive.ui.workspace.navigation import Navigation

        workspace = v.sample.app.workspace
        if not hasattr(workspace, "navigation"):
            workspace.navigation = Navigation(workspace)
        navigation = workspace.navigation
        position = v.rect(x, y, width, 0)
        g.set_cursor_screen_pos((position.x, position.y))
        with group_scope("Scene"):
            rows = navigation.scene_rows(v.sample.context)[:4]
            for index, row in enumerate(rows):
                navigation.draw_scene_row(v.sample.context, *row, width * v.s)
                v.ui.hits[f"sample/{group}/{index}"] = v.ui.hits["entity-" + str(row[0].node_id)]
        return (g.get_cursor_screen_pos().y - v.origin[1]) / v.s
    if group == "Inspector tabs":
        from mojive.ui.workspace.inspector_heading import draw_tabs

        labels = tuple(label for label, _icon in SAMPLES[group])
        host = SimpleNamespace(ui=v.ui, s=v.s, inspector_tab=labels[v.selected.get(group, 0)])
        rect = v.rect(x, y, width, 0)
        with group_scope("Inspector"):
            bottom = draw_tabs(host, rect.x, rect.y, rect.width)
        v.selected[group] = labels.index(host.inspector_tab)
        return (bottom - v.origin[1]) / v.s
    if group == "Material cards":
        if not hasattr(v, "app"):
            v.material = 0
            v.app = SimpleNamespace(
                entity={"colorable": True, "color": "#8DAECC"},
                apply_document_edits=lambda: setattr(
                    v, "message", "Material applied to the local sample"
                ),
            )
        with group_scope("Assets"):
            g.set_cursor_screen_pos((v.origin[0] + x * v.s, v.origin[1] + y * v.s))
            g.begin_child("Material library sample", (width * v.s, 440 * v.s))
            position = g.get_cursor_screen_pos()
            draw_material_library(v, position.x, position.y, g.get_content_region_avail().x)
            g.end_child()
        return y + 440
    if group == "Viewport tools":
        return tools(v, x, y, width)
    if group == "Render metric":
        return telemetry(v, x, y, width)
    part, samples = GROUPS[group], SAMPLES[group]
    style = v.ui.component_style(part)
    vertical = group in ("Settings navigation", "Scene tree")
    natural_width = min(
        width,
        192
        if group == "Settings navigation"
        else width
        if vertical
        else 112
        if group == "Inspector tabs"
        else 128
        if group == "Timeline actions"
        else 80,
    )
    button_width = min(width, style.dimension("width", natural_width))
    height = v.ui.design.primitives.field_height(part, 32)
    container = v.ui.component_style(GROUP_CONTAINERS[group])
    content = v.rect(x, y, button_width if vertical else width, 2000)
    sizes = [(button_width * v.s, height * v.s)] * len(samples)
    if group == "Scene filters":
        with group_scope("Scene"):
            sizes = [
                filter_chip_size(v.ui, label, maximum_width=width * v.s) for label, _ in samples
            ]
    layout = flow_layout(
        content,
        sizes,
        column_gap=container.get("columnGap", 8) * v.s,
        row_gap=container.get("rowGap", 8) * v.s,
    )
    for index, ((label, icon), rect) in enumerate(zip(samples, layout.rectangles, strict=True)):
        key = f"sample/{group}/{index}"
        if group == "Scene filters":
            with group_scope("Scene"):
                changed = draw_filter_chip(
                    v.ui,
                    key,
                    rect,
                    label,
                    icon,
                    v.tr(
                        ("All entities", "Bodies", "Geometries", "Joints", "Cameras", "Robots")[
                            index
                        ]
                    ),
                    v.ink("action" if v.selected.get(group, 0) == index else "muted"),
                    v.selected.get(group, 0) == index,
                )
            if changed:
                v.selected[group] = index
            continue
        changed = v.ui.design.primitives.button(
            key,
            rect,
            v.tr(label),
            icon,
            parts=part,
            active=v.selected.get(group, 0) == index,
            filled=True,
            translate=False,
        ).changed
        if changed:
            v.selected[group] = index
    return y + layout.height / v.s


def switches(v, labels, x, y, width, *, prefix="switch-"):
    for index, label in enumerate(labels):
        key = prefix + str(index)
        result = v.ui.design.primitives.switch(
            key,
            v.rect(x, y + index * 48, width, 36),
            v.values.get(key, True),
            label=label,
            control_x=v.rect(x + min(224, width - 64), 0, 0, 0).x,
        )
        v.values[key] = result.value
    return y + len(labels) * 48


def tools(v, x, y, width):
    from mojive.ui.workspace.viewport_controls import toolbox

    workspace = v.sample.app.workspace
    overlays = []
    rect = v.rect(x, y, width, 320)
    toolbox(workspace, rect.tuple(), rect.y, width >= 480, False, overlays)
    return y + max((r[3] for r in overlays), default=0) / v.s


@part_scope("statusbar")
def telemetry(v, x, y, width):
    with v.ui.design.family_scope("chrome"):
        return _telemetry(v, x, y, width)


def _telemetry(v, x, y, width):
    from mojive.ui.workspace.status_tooltip import status_tooltip

    presentation = StatusGroups(v.ui)
    badge = StatusBadge(v.ui, keycap=True)
    layout = resolve_status_bar_layout(v.ui, badge=badge, groups=presentation)
    inset = 8
    canvas_height = layout.height / v.s + 2 * inset
    v.box(x, y, width, canvas_height, "canvas", inset)
    p = v.rect(x + inset, y + inset, width - 2 * inset, layout.height / v.s)
    box = layout.box(p.x, p.y, p.width)
    p = box.content
    labels = ViewportLabels(**{item.name: v.tr(item.default) for item in fields(ViewportLabels)})
    v.status_layout = draw_status(
        ImguiDraw2D(),
        (p.x, p.y),
        p.width,
        p.height,
        v.theme,
        v.s,
        selected="",
        state="running" if v.values.get("playback-running", False) else "paused",
        sim_time=v.values.get("playback-time", 12.125),
        step=12500,
        metric_mode=v.values.get("metric-mode", "steps"),
        backend="OpenGL",
        dt=0.002,
        fps=55 + math.sin(v.clock * 2) * 4.9,
        physics_hz=480,
        show_physics=True,
        presentation=presentation,
        surface_rect=box.frame.tuple(),
        labels=labels,
        tool_hints=default_tool_hints(
            v.values.get("status-hint-context", "camera"), DEFAULT_INPUT_BINDINGS, labels
        ),
        keycap=badge,
        hints_gap=presentation.gap("status_hints"),
        group_gap=v.ui.component_style("statusbar").get("columnGap", 24) * v.s,
    )
    v.ui.record_component_box(
        "statusbar",
        box,
        key="guide-statusbar",
        reservedHeight=layout.height,
        minimumContentHeight=layout.minimum_content_height,
    )
    v.ui.hits["guide-statusbar"] = box.frame.tuple()
    if v.ui.design is not None:
        v.ui.design.record(
            "guide-status-canvas",
            (),
            v.rect(x, y, width, canvas_height).tuple(),
            "status_preview",
            inset=inset * v.s,
            radius=inset,
            innerRadius=0,
            statusBarRect=list(box.frame.tuple()),
        )
    for metric in v.status_layout.metrics:
        left, top, right, bottom = metric.rect
        g.set_cursor_screen_pos((left, top))
        g.invisible_button("guide-metric/" + metric.key, (right - left, bottom - top))
        v.record("guide-metric/" + metric.key)
        if g.is_item_hovered(g.HoveredFlags_.delay_normal):
            status_tooltip(v.ui, metric.exact, v.tr("Right-click to copy"))
        if g.is_item_hovered() and g.is_mouse_clicked(1):
            g.set_clipboard_text(metric.exact)
            v.message = v.tr("Copied {value}", value=metric.exact)
    if v.status_layout.metric_rect is not None:
        left, top, right, bottom = v.status_layout.metric_rect
        g.set_cursor_screen_pos((left, top))
        g.invisible_button("guide-metric/metric", (right - left, bottom - top))
        v.record("guide-metric/metric")
        if g.is_item_hovered(g.HoveredFlags_.delay_normal):
            status_tooltip(
                v.ui,
                v.status_layout.metric_exact,
                labels.show_time
                if v.values.get("metric-mode", "steps") == "steps"
                else labels.show_steps,
                labels.copy_exact,
            )
        if g.is_item_hovered() and g.is_mouse_clicked(1):
            g.set_clipboard_text(v.status_layout.metric_exact)
            v.message = v.tr("Copied {value}", value=v.status_layout.metric_exact)
        if g.is_item_clicked():
            v.values["metric-mode"] = (
                "time" if v.values.get("metric-mode", "steps") == "steps" else "steps"
            )
            if v.values.get("status-readout") in ("time", "steps"):
                v.values["status-readout"] = v.values["metric-mode"]
    return y + canvas_height


def draw_panel(v):
    """Present the registered native panel with the guide's isolated session."""
    v.title(v.page, "Native panel · Local example session")
    if v.page == "Status":
        from .guide_slices import status

        return status(v)
    if v.page == "Menus":
        from .guide_slices import menus

        return menus(v)
    if v.page == "Viewport":
        v.sample.viewport(0, 96, v.width, 464)
        return telemetry(v, 0, 584, v.width) + 24
    if v.page == "Output":
        if v.action("restore-output", v.width - 28, 96, "reset", "Restore sample"):
            v.sample.restore_output()
        return v.sample.panel("Output", 0, 144, v.width, 424) + 24
    width = min(v.width, 384)
    y, height = 96, 624
    if v.page in ("Inspector", "Pose", "Cameras", "Sensors", "Curves"):
        key = {
            "Inspector": "inspector-size",
            "Pose": "pose-size",
            "Cameras": "camera-size",
            "Sensors": "sensor-size",
            "Curves": "curve-size",
        }[v.page]
        _, size = v.viewer_select(
            key,
            0,
            96,
            240,
            v.values.get(key, "Wide"),
            ("Narrow", "Medium", "Wide"),
        )
        v.values[key] = size
        wide = 640 if v.page == "Inspector" else v.width if v.page == "Curves" else 384
        medium = 560 if v.page == "Curves" else 280
        narrow = 280 if v.page == "Curves" else 220
        width = min(v.width, {"Narrow": narrow, "Medium": medium, "Wide": wide}[size])
        y, height = 152, 640
    if v.page in ("Settings", "Timeline", "Statistics"):
        width = v.width
    if v.page == "Timeline":
        height = 440
    return v.sample.panel("Plot" if v.page == "Curves" else v.page, 0, y, width, height) + 24
