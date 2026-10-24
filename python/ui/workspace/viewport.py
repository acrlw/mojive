"""Native workspace viewport; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import part_scope
from mojive.ui.input_bindings import InputAction
from mojive.ui.viewcube import BACKDROP_RADIUS_PT
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.constants import VIEWPORT_TOOLS
from mojive.ui.workspace.design import BODY_SIZE, DESIGN, MONO_SIZE
from mojive.ui.workspace.group_corners import (
    action_corner_scope,
    action_corners,
    centered_action_corners,
    inset_action_box,
)
from mojive.ui.workspace.viewport_controls import (
    camera_controls,
    toolbar_group_gap,
    toolbar_separator,
    toolbox,
    toolbox_layout,
)
from mojive.ui.workspace.widgets import (
    AMBER,
    PANEL,
    TEXT,
)


def _bottom_controls(workspace):
    ui, scale = workspace.ui, workspace.s
    return (
        ("expand-timeline", "right", "", 26, "Expand Timeline"),
        ("timeline-tab", "key", "Timeline", max(95, ui.measure("Timeline") / scale + 39), ""),
        ("output-tab", "terminal", "Output", max(85, ui.measure("Output") / scale + 39), ""),
    )


def collapsed_bottom_height(workspace):
    ui = workspace.ui
    return (
        max(
            control_size(ui, key, width, 28, label=label, icon=icon)[1]
            for key, icon, label, width, _ in _bottom_controls(workspace)
        )
        + 10 * workspace.s
    )


@part_scope("bottom_header")
def _collapsed_bottom(workspace, x, y, width):
    ui, scale = workspace.ui, workspace.s
    height = collapsed_bottom_height(workspace)
    box = ui.component_box(
        "bottom_header",
        x,
        y,
        width,
        height,
        key="collapsed-bottom-header",
        padding=(8 * scale, 5 * scale, 8 * scale, 5 * scale),
    )
    ui.rect(*box.frame.tuple(), PANEL, 0)
    ui.line(x, y, width)
    gap = ui.metric("bottom_header", "columnGap", ui.metric("bottom_header", "gap", 4)) * scale
    cursor = box.content.x
    for key, icon, label, requested, tooltip in _bottom_controls(workspace):
        ow, oh = control_size(ui, key, requested, 28, label=label, icon=icon)
        if cursor + ow > box.content.right:
            break
        fw, fh = control_frame_size(ui, key, ow, oh, label=label, icon=icon)
        if ui.button(
            key,
            cursor,
            box.content.y + (box.content.height - oh) / 2,
            fw,
            fh,
            icon=icon,
            label=label,
            tooltip=tooltip,
        ):
            if key == "expand-timeline":
                workspace.toggle_timeline()
            else:
                workspace.activate(label)
        cursor += ow + gap


def viewport(workspace, preview_name=""):
    u = workspace.ui
    s = workspace.s
    app = workspace.app
    image = app._viewport_image
    app._viewport_rect = app.viewport_surface.draw_image(
        image, workspace.window.viewport_texture_ref
    )
    x, y, w, h = app._viewport_rect
    workspace.rects["viewport_content"] = [x, y, w, h]
    if u.design is not None:
        node = imgui.internal.dock_builder_get_node(workspace.window.dockspace_id)
        if node is not None:
            u.design.record(
                "workspace-layout",
                ("workspace",),
                (node.pos.x, node.pos.y, node.size.x, node.size.y),
                "workspace",
            )
    toolbox_style = u.component_style("toolbox")
    tool_padding = toolbox_style.edges("padding", s, (4, 4, 4, 4))
    tool_margin = toolbox_style.edges("margin", s)
    vertical = toolbox_layout(u)
    extent = vertical.rectangles[0].width, vertical.rectangles[0].height
    vertical_height = (
        vertical.height + tool_padding[1] + tool_padding[3] + tool_margin[1] + tool_margin[3]
    )
    horizontal = h < max(440 * s, vertical_height + 150 * s)
    tools = toolbox_layout(u, horizontal=True, extent=extent) if horizontal else vertical
    wrapped = (
        horizontal
        and w
        < tools.width + tool_padding[0] + tool_padding[2] + tool_margin[0] + tool_margin[2] + 20 * s
    )
    if wrapped:
        tools = toolbox_layout(u, horizontal=True, wrapped=True, extent=extent)
    tools_height = (
        tools.height + tool_padding[1] + tool_padding[3] + tool_margin[1] + tool_margin[3]
    )
    footer_space = (
        10
        if app.model_edits.active and h < 300 * s
        else 42
        if h >= 210 * s and not workspace.open["Inspector"]
        else 32
    )
    tools_y = (
        y + h - tools_height - footer_space * s
        if horizontal
        else y + max(148 * s, (h - vertical_height) / 2)
    )
    stacked_camera = w < 160 * s
    camera_height = 68 if stacked_camera else 36
    transport_visible = h >= (248 if w < 260 * s else 220) * s
    transport_width = 168 if w < 340 * s else 282
    rail_width = tools.width + tool_padding[0] + tool_padding[2] + tool_margin[0] + tool_margin[2]
    collapsed_controls = (
        w < rail_width + 20 * s
        or (horizontal and tools_y < y + (camera_height + 18) * s)
        or (transport_visible and w < (transport_width + 20) * s)
        or (app.camera_preview.enabled and bool(preview_name) and h < 300 * s)
    )
    if app.model_edits.active and horizontal:
        pending_height = 70 if w < 412 * s and h >= 300 * s else 40
        collapsed_controls |= tools_y - (pending_height + 8) * s < y + (camera_height + 18) * s
    workspace.cube.design_ui = u
    cube_scale = workspace.cube.scale_for((x, y, w, h), s)
    cube_rect = None
    if cube_scale > 0:
        center = workspace.cube.widget_center((x, y, w, h), cube_scale)
        radius = BACKDROP_RADIUS_PT * cube_scale
        cube_rect = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        cx, cy, cw, ch = cube_rect
        if horizontal:
            collapsed_controls |= (
                x + 10 * s < cx + cw
                and cx < x + 10 * s + rail_width
                and tools_y < cy + ch
                and cy < tools_y + tools_height
            )
    imgui.push_clip_rect((x, y), (x + w, y + h), True)
    app._draw_scene_overlays(workspace.window.painter())
    overlay = []
    if app.model_edits.active:
        workspace.pending_model_edits(
            x,
            y,
            w,
            h,
            overlay,
            bottom_limit=tools_y if horizontal and not collapsed_controls else None,
        )
    app._playback_widget_rect = None
    if not app.viewport_layers.viewport_ui:
        workspace.rects["overlays"] = overlay
        workspace.input_overlays = list(overlay)
        imgui.pop_clip_rect()
        app._draw_workspace_camera_preview(preview_name)
        imgui.end()
        return

    def shell(rx, ry, rw, rh, *, radius=8):
        u.rect(rx, ry, rw * s, rh * s, (22, 25, 28, 230), radius)
        overlay.append((rx, ry, rw * s, rh * s))

    if collapsed_controls:
        shell(x + 10 * s, y + 10 * s, 36, 36)
        popup = "Viewport controls###viewport-more"
        if u.button(
            "viewport-more",
            x + 14 * s,
            y + 14 * s,
            28,
            28,
            icon="more",
            tooltip="Viewport controls",
            active=imgui.is_popup_open(popup),
        ):
            imgui.open_popup(popup)
        if (
            workspace.menus.requested_path
            and workspace.menus.requested_path[0]
            in (
                "Camera",
                "Shading",
                "Playback",
            )
            and not imgui.is_popup_open(popup)
        ):
            imgui.open_popup(popup)
        with workspace.menus.local_popup(popup) as opened:
            if opened:

                def menu_action(key, label, shortcut="", checked=None, enabled=True):
                    clicked, _ = workspace.menus.local_item(
                        label, shortcut, checked, enabled, key=key
                    )
                    return clicked

                for name in ("Camera", "Shading"):
                    workspace.menus.reveal((name,))
                    workspace.menus.constrain(name)
                    with workspace.menus.local_submenu(
                        name, key=name, content=False
                    ) as submenu_open:
                        if submenu_open:
                            workspace.menus.content(name, (name,))
                if cube_scale <= 0 and workspace.cube.preview_active:
                    with workspace.menus.local_submenu(
                        "View direction", key="viewport-view-direction"
                    ) as direction_open:
                        if direction_open:
                            for name in ("Front", "Back", "Left", "Right", "Top", "Bottom"):
                                if menu_action("camera-preview-view-" + name, name):
                                    app._leave_model_camera()
                                    app.camera.set_preset(name.lower())
                workspace.menus.local_separator()
                for name, action in VIEWPORT_TOOLS:
                    shortcut = "V" if action is None else app.input_bindings.label(action)
                    if shortcut == "Unbound":
                        shortcut = ""
                    if menu_action("tool-" + name, name.title(), shortcut, app.tool == name):
                        app.tool = name
                if menu_action(
                    "frame",
                    "Body frame",
                    app.input_bindings.label(InputAction.GIZMO_SPACE),
                    workspace.space == "Body",
                ):
                    workspace.space = "World" if workspace.space == "Body" else "Body"
                if menu_action(
                    "snap", "Snap", app.input_bindings.label(InputAction.SNAP), workspace.snap
                ):
                    workspace.snap = not workspace.snap
                if menu_action("viewport-layers", "Viewport layers"):
                    workspace.activate("Layers")
                workspace.menus.local_separator()
                workspace.menus.reveal(("Playback",))
                workspace.menus.constrain("Playback")
                with workspace.menus.local_submenu(
                    "Playback options", key="Playback"
                ) as playback_open:
                    if playback_open:
                        session = app.session
                        if menu_action(
                            "play", "Pause simulation" if workspace.playing else "Play simulation"
                        ):
                            app._toggle_playback()
                        if menu_action("go-start", "Go to start"):
                            app._reset_playback()
                        if menu_action(
                            "record",
                            "Record simulation",
                            checked=session.state_take_recording,
                            enabled=session.adapter.caps.simulation,
                        ):
                            app._toggle_state_take_recording()
                        workspace.menus.local_separator()
                        workspace.menus.local_rows(workspace.menus.items("Playback"))
    else:
        camera_controls(workspace, (x, y, w, h), overlay)
        if transport_visible:
            workspace.transport(x, y, w, shell)
        toolbox(workspace, (x, y, w, h), tools_y, horizontal, wrapped, overlay)
    workspace.input_overlays = list(overlay)
    if cube_rect is not None:
        overlay.append(cube_rect)
    workspace.rects["overlays"] = overlay
    imgui.pop_clip_rect()
    if not workspace.bottom_open:
        _collapsed_bottom(workspace, x, y + h, w)
    app._draw_workspace_camera_preview(preview_name)
    imgui.end()


@part_scope("pending_edits")
@action_corner_scope("pending_edits", (8,) * 4)
def pending_model_edits(workspace, x, y, w, h, overlay, *, bottom_limit=None):
    from mojive.design.native.geometry import Rect

    ui, scale, app = workspace.ui, workspace.s, workspace.app
    style = ui.component_style("pending_edits")
    margins = style.edges("margin", scale)
    corners = action_corners(ui, "pending_edits", (8,) * 4)
    padding = corners.padding
    gap = style.get("columnGap", style.get("gap", 6)) * scale
    width = min(w - 20 * scale - margins[0] - margins[2], 392 * scale)
    compact = width < 392 * scale and h < 300 * scale
    stacked = width < 392 * scale and not compact
    row_height = max(
        control_size(
            ui,
            key,
            width,
            28,
            label="" if compact else label,
            icon=icon if compact else None,
        )[1]
        for key, width, label, icon in (
            ("apply-model-edits", 76, "Apply", "check"),
            ("discard-model-edits", 80, "Discard", "close"),
        )
    )
    if not compact:
        row_height = max(
            row_height, style.line_height(style.get("fontSize", BODY_SIZE) * 1.4) * scale
        )
    height = (2 * row_height + gap if stacked else row_height) + padding[1] + padding[3]
    px = x + (w - width - margins[0] - margins[2]) / 2
    bottom = style.get("bottom", 10) * scale
    py = (
        bottom_limit - height - bottom
        if bottom_limit is not None
        else y + max(10 * scale, h - height - bottom)
    )
    box = ui.component_box(
        "pending_edits",
        px,
        py,
        width,
        height,
        key="pending-model-edits",
        padding=padding,
        observe=False,
    )
    box = inset_action_box(box, corners)
    ui.record_component_box(
        "pending_edits",
        box,
        key="pending-model-edits",
        radius=corners.radius,
        childRadius=corners.inner_radius,
    )
    ui.rect(*box.frame.tuple(), (22, 25, 28, 245), corners.radius / scale)
    overlay.append(box.outer.tuple())
    label = "Applying…" if app.model_edits.applying else "Pending model edits"
    apply_width = (
        min(76 * scale, (box.content.width - 28 * scale - 2 * gap) / 2) if compact else 76 * scale
    )
    discard_width = (
        min(80 * scale, (box.content.width - 28 * scale - 2 * gap) / 2) if compact else 80 * scale
    )
    apply_size, discard_size = (
        control_size(
            ui,
            key,
            max(0, width) / scale,
            28,
            label="" if compact else label,
            icon=icon if compact else None,
        )
        for key, width, label, icon in (
            ("apply-model-edits", apply_width, "Apply", "check"),
            ("discard-model-edits", discard_width, "Discard", "close"),
        )
    )
    apply_width, discard_width = apply_size[0], discard_size[0]
    actions_width = apply_width + discard_width + gap
    label_width = box.content.width if stacked else max(0, box.content.width - actions_width - gap)
    # The caption owns the same text inset as the adjacent buttons; the shell's
    # inset alone only measures the clearance around their painted surfaces.
    caption_inset = (
        0 if compact else ui.component_style("button").edges("padding", scale, (8, 0, 8, 0))[0]
    )
    caption_width = max(0, label_width - caption_inset)
    label_rect = style.align_rect(
        Rect(box.content.x + caption_inset, box.content.y, caption_width, row_height),
        caption_width,
        min(row_height, style.line_height(style.get("fontSize", BODY_SIZE) * 1.4) * scale),
        horizontal="start",
    )
    if compact:
        status_size = control_size(
            ui, "model-edit-status", min(28, label_width / scale), 28, icon="info"
        )
        status_rect = style.align_rect(
            Rect(box.content.x, box.content.y, label_width, row_height), *status_size
        )
        ui.button(
            "model-edit-status",
            status_rect.x,
            status_rect.y,
            *control_frame_size(ui, "model-edit-status", *status_size, icon="info"),
            icon="info",
            tooltip=label,
        )
    else:
        ui.component_text("pending_edits", label_rect, label, AMBER)
    if app.model_edits.error and imgui.is_mouse_hovering_rect(
        (box.frame.x, box.frame.y), (box.frame.right, box.frame.bottom)
    ):
        ui.tooltip(app.model_edits.error)
    bx = box.content.x if stacked else box.content.right - actions_width
    by = (
        box.content.y + row_height + gap
        if stacked
        else style.align_rect(box.content, actions_width, row_height, horizontal="end").y
    )
    enabled = not app.model_edits.applying and not app.gizmo.using
    if ui.button(
        "apply-model-edits",
        bx,
        by,
        *control_frame_size(
            ui,
            "apply-model-edits",
            *apply_size,
            label="" if compact else "Apply",
            icon="check" if compact else None,
        ),
        label="" if compact else "Apply",
        icon="check" if compact else None,
        tooltip="Apply" if compact else None,
        active=True,
        solid=True,
        enabled=enabled,
    ):
        app._apply_model_edits_requested = True
    if ui.button(
        "discard-model-edits",
        bx + apply_width + gap,
        by,
        *control_frame_size(
            ui,
            "discard-model-edits",
            *discard_size,
            label="" if compact else "Discard",
            icon="close" if compact else None,
        ),
        label="" if compact else "Discard",
        icon="close" if compact else None,
        tooltip="Discard" if compact else None,
        filled=True,
        enabled=enabled,
    ):
        app.model_edits.clear()
        app.gizmo._reset_model_placement()
        app._apply_model_edits_requested = False


@part_scope("transport")
@action_corner_scope("transport", (6,) * 4)
def transport(workspace, x, y, w, shell):
    from mojive.design.native.geometry import Rect
    from mojive.ui.workspace.component_layout import control_frame_size

    u, scale = workspace.ui, workspace.s
    session, replay = workspace.app.session, workspace.app.session.replay_info
    compact = w < 340 * scale
    style = u.component_style("transport")
    gap = style.get("columnGap", style.get("gap", 3)) * scale
    time_gap = u.metric("time_field", "columnGap", u.metric("time_field", "gap", 4)) * scale
    caption = f"{workspace.time:.3f}"
    time_font = style.get("fontSize", MONO_SIZE)
    time_width = max(
        80 * scale,
        u.measure(caption, time_font, translate=False, mono=True, font_resolved=True)
        + u.measure("s", time_font, translate=False, mono=True, font_resolved=True)
        + time_gap
        + 16 * scale,
    )
    corners = action_corners(u, "transport", (6,) * 4)
    pad = corners.padding
    names = (
        ("previous", "play", "next", "go-start", "playback-options")
        if compact
        else ("previous", "play", "next", "time", "go-start", "record", "playback-options")
    )
    widths = tuple(80 if name == "time" else 32 if name == "play" else 26 for name in names)
    sizes = tuple(
        (time_width, DESIGN.control_height * scale)
        if name == "time"
        else control_size(u, name, width, 28, icon=name)
        for name, width in zip(names, widths, strict=True)
    )
    # The plain time readout can be taller than authored icon buttons. Match
    # their horizontal offset to their centered vertical offset at both ends.
    corners = centered_action_corners(corners, sizes)
    pad = corners.padding
    split = names.index("go-start")
    group_gap = toolbar_group_gap(style, scale, gap)
    width = sum(size[0] for size in sizes) + (len(sizes) - 2) * gap + group_gap + pad[0] + pad[2]
    margins = style.edges("margin", scale)
    if width + margins[0] + margins[2] > w - 20 * scale:
        return
    tx = x + (w - width - margins[0] - margins[2]) / 2
    default_top = 104 if w < 500 * scale else 52 if w < 700 * scale else 10
    ty = y + max(style.get("top", 10), default_top) * scale
    box = u.component_box(
        "transport",
        tx,
        ty,
        width + margins[0] + margins[2],
        max(v[1] for v in sizes) + pad[1] + pad[3] + margins[1] + margins[3],
        key="viewport-transport",
        observe=False,
        padding=pad,
    )
    box = inset_action_box(box, corners)
    workspace.app._playback_widget_rect = (
        box.frame.x,
        box.frame.y,
        box.frame.right,
        box.frame.bottom,
    )
    shell(
        box.frame.x,
        box.frame.y,
        box.frame.width / scale,
        box.frame.height / scale,
        radius=corners.radius / scale,
    )
    positions = []
    cursor = box.content.x
    for index, (item_width, item_height) in enumerate(sizes):
        if index:
            cursor += group_gap if index == split else gap
        positions.append(
            style.align_rect(
                Rect(cursor, box.content.y, item_width, box.content.height), item_width, item_height
            )
        )
        cursor += item_width
    divider = toolbar_separator(u, style, positions[split - 1], positions[split], horizontal=True)
    if u.design is not None and u.design.runtime.observes(style.part_ids):
        u.record_component_box(
            "transport",
            box,
            key="viewport-transport",
            fields=[list(r.tuple()) for r in positions],
            anchorRect=[x, y, w, 0],
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
    for name, p in zip(names, positions, strict=True):
        if name == "time":
            number_width = u.measure(
                caption, time_font, translate=False, mono=True, font_resolved=True
            )
            unit_width = u.measure("s", time_font, translate=False, mono=True, font_resolved=True)
            start = p.x + max(0, (p.width - number_width - unit_width - time_gap) / 2)
            u.component_text(
                "transport",
                Rect(start, p.y, number_width, p.height),
                caption,
                TEXT,
                size=time_font,
                mono=True,
                translate=False,
            )
            if u.design is not None:
                u.design.record(
                    "viewport-time",
                    ("time_field",),
                    p.tuple(),
                    "number-unit",
                    gap=time_gap,
                    fields=[
                        [start, p.y, number_width, p.height],
                        [start + number_width + time_gap, p.y, unit_width, p.height],
                    ],
                    content=[start, p.y, number_width + time_gap + unit_width, p.height],
                )
            u.component_text(
                "transport",
                Rect(start + number_width + time_gap, p.y, unit_width, p.height),
                "s",
                TEXT,
                size=time_font,
                mono=True,
                translate=False,
            )
            continue
        enabled = True
        if name == "previous":
            enabled = replay.frame_index > 0 if replay else session.can_step_back
        elif name == "next":
            enabled = not workspace.playing and (
                replay.frame_index < replay.frame_count - 1
                if replay
                else session.adapter.caps.simulation
            )
        elif name == "record":
            enabled = session.adapter.caps.simulation
        icons = {
            "previous": "playback-previous",
            "play": "playback-pause" if workspace.playing else "playback-play",
            "next": "playback-next",
            "go-start": "playback-reset",
            "record": "playback-record",
            "playback-options": "playback-more",
        }
        tips = {
            "previous": "Previous frame",
            "play": "Play / pause",
            "next": "Next frame",
            "go-start": "Go to start",
            "record": "Record state take",
            "playback-options": "Playback options",
        }
        with u.corner_scope(corners.inner_radius, ("button", "icon_button")):
            clicked = u.button(
                name,
                p.x,
                p.y,
                *control_frame_size(u, name, p.width, p.height, icon=icons[name]),
                icon=icons[name],
                icon_role="subtle",
                tooltip=tips[name],
                enabled=enabled,
                active=name == "play"
                or (name == "record" and session.state_take_recording)
                or (name == "playback-options" and imgui.is_popup_open("Playback")),
            )
        if name == "playback-options":
            workspace.menus.popup("Playback", u.hits[name], clicked)
        elif clicked:
            if name == "play":
                workspace.app._toggle_playback()
            elif name == "go-start":
                workspace.app._reset_playback()
            elif name == "record":
                workspace.app._toggle_state_take_recording()
            elif name == "previous":
                session.submit(
                    cmd.SeekReplay(max(0, replay.frame_index - 1)) if replay else cmd.StepBack()
                )
            elif name == "next":
                session.submit(
                    cmd.SeekReplay(min(replay.frame_count - 1, replay.frame_index + 1))
                    if replay
                    else cmd.Step()
                )
