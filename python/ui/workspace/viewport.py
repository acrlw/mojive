"""Native workspace viewport; shared scene edits use Session commands."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.ui.input_bindings import InputAction
from mojive.ui.viewcube import BACKDROP_RADIUS_PT, widget_center

from .constants import VIEWPORT_TOOLS
from .design import BODY_SIZE, DESIGN, MONO_SIZE
from .widgets import (
    AMBER,
    DIM,
    PANEL,
    TEXT,
    color,
)


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
    horizontal = h < 440 * s
    step = 32 if w < 280 * s else 40
    wrapped = horizontal and w < 260 * s
    tools_height = 76 if wrapped else 40
    footer_space = (
        10
        if app.model_edits.active and h < 300 * s
        else 42
        if h >= 210 * s and not workspace.open["Inspector"]
        else 32
    )
    tools_y = (
        y + h - (tools_height + footer_space) * s
        if horizontal
        else y + max(148, (h / s - 248) / 2) * s
    )
    compact_camera = w < 240 * s
    stacked_camera = w < 160 * s
    camera_height = 68 if stacked_camera else 36
    transport_visible = h >= (248 if w < 260 * s else 220) * s
    transport_width = 168 if w < 340 * s else 282
    rail_width = step * (3 if wrapped else 6) + 8 if horizontal else 44
    collapsed_controls = (
        w < (rail_width + 20) * s
        or (horizontal and tools_y < y + (camera_height + 18) * s)
        or (transport_visible and w < (transport_width + 20) * s)
        or (app.camera_preview.enabled and bool(preview_name) and h < 300 * s)
    )
    if app.model_edits.active and horizontal:
        pending_height = 70 if w < 412 * s and h >= 300 * s else 40
        collapsed_controls |= tools_y - (pending_height + 8) * s < y + (camera_height + 18) * s
    cube_scale = workspace.cube.scale_for((x, y, w, h), s)
    cube_rect = None
    if cube_scale > 0:
        center = widget_center((x, y, w, h), cube_scale)
        radius = BACKDROP_RADIUS_PT * cube_scale
        cube_rect = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        cx, cy, cw, ch = cube_rect
        if horizontal:
            collapsed_controls |= (
                x + 10 * s < cx + cw
                and cx < x + (10 + rail_width) * s
                and tools_y < cy + ch
                and cy < tools_y + tools_height * s
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
    if not app.viewport_layers.viewport_ui:
        workspace.rects["overlays"] = overlay
        workspace.input_overlays = list(overlay)
        imgui.pop_clip_rect()
        app._draw_workspace_camera_preview(preview_name)
        imgui.end()
        return

    def shell(rx, ry, rw, rh):
        u.rect(rx, ry, rw * s, rh * s, (22, 25, 28, 230), 8)
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
        with workspace.menus.style():
            if imgui.begin_popup(popup):

                def menu_action(key, label, shortcut="", checked=False, enabled=True):
                    clicked, _ = imgui.menu_item(u.translate(label), shortcut, checked, enabled)
                    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    u.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
                    return clicked

                for name in ("Camera", "Shading"):
                    workspace.menus.reveal((name,))
                    workspace.menus.constrain(name)
                    if imgui.begin_menu(u.translate(name) + "###" + name):
                        workspace.menus.content(name, (name,))
                        imgui.end_menu()
                if (
                    cube_scale <= 0
                    and workspace.cube.preview_active
                    and imgui.begin_menu(u.translate("View direction"))
                ):
                    for name in ("Front", "Back", "Left", "Right", "Top", "Bottom"):
                        if menu_action("camera-preview-view-" + name, name):
                            app._leave_model_camera()
                            app.camera.set_preset(name.lower())
                    imgui.end_menu()
                imgui.separator()
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
                imgui.separator()
                workspace.menus.reveal(("Playback",))
                workspace.menus.constrain("Playback")
                if imgui.begin_menu(u.translate("Playback options") + "###Playback"):
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
                    imgui.separator()
                    workspace.menus.content("Playback", ("Playback",))
                    imgui.end_menu()
                imgui.end_popup()
    else:
        camera_width = (
            44 if stacked_camera else 72 if compact_camera else 144 if w < 300 * s else 176
        )
        shell(x + 10 * s, y + 10 * s, camera_width, 68 if stacked_camera else 36)
        camera_clicked = u.button(
            "projection-button",
            x + 13 * s,
            y + 14 * s,
            28 if compact_camera else 98,
            28,
            icon="camera",
            label=""
            if compact_camera
            else "Ortho"
            if workspace.app.camera.orthographic
            else "Persp",
            tooltip="Camera" if compact_camera else None,
            active=imgui.is_popup_open("Camera"),
            align="left",
        )
        if not compact_camera:
            u.role_icon("down", x + 92 * s, y + 14 * s, 12 * s, 28 * s, role="inline", c=DIM)
        workspace.menus.popup("Camera", u.hits["projection-button"], camera_clicked)
        shading_clicked = u.button(
            "shading",
            x + (13 if stacked_camera else 47 if compact_camera else 116) * s,
            y + (46 if stacked_camera else 14) * s,
            28,
            28,
            icon="shading",
            tooltip="Shading",
            active=imgui.is_popup_open("Shading"),
        )
        workspace.menus.popup("Shading", u.hits["shading"], shading_clicked)
        if camera_width == 176 and u.button(
            "viewport-layers",
            x + 148 * s,
            y + 14 * s,
            28,
            28,
            icon="layers",
            tooltip="Viewport layers",
        ):
            workspace.activate("Layers")
        if transport_visible:
            workspace.transport(x, y, w, shell)
        shell(
            x + 10 * s,
            tools_y,
            step * (3 if wrapped else 6) + 8 if horizontal else 44,
            tools_height if horizontal else 248,
        )

        def tool_pos(n):
            if wrapped:
                return x + (14 + n % 3 * step) * s, tools_y + (4 + n // 3 * 36) * s
            return (
                x + (14 + n * step if horizontal else 14) * s,
                tools_y + (4 if horizontal else (176, 211)[n - 4] if n >= 4 else 4 + n * 40) * s,
            )

        for n, (name, action) in enumerate(VIEWPORT_TOOLS):
            xx, yy = tool_pos(n)
            button_size = step - 4 if horizontal else 36
            shortcut = "V" if action is None else app.input_bindings.label(action)
            tip = "Dimensions: select a geometry" if name == "scale" else name.title()
            if shortcut != "Unbound":
                tip += " · " + shortcut
            if u.button(
                "tool-" + name,
                xx,
                yy,
                button_size,
                32 if horizontal else 36,
                icon=name,
                active=workspace.app.tool == name,
                solid=True,
                tooltip=tip,
            ):
                workspace.app.tool = name
        if not horizontal:
            u.line(x + 18 * s, tools_y + 168 * s, 28 * s)
        if u.button(
            "frame",
            *tool_pos(4),
            step - 4 if horizontal else 36,
            32,
            icon="body" if workspace.space == "Body" else "world",
            tooltip="World / body frame · " + app.input_bindings.label(InputAction.GIZMO_SPACE),
        ):
            workspace.space = "Body" if workspace.space == "World" else "World"
        if u.button(
            "snap",
            *tool_pos(5),
            step - 4 if horizontal else 36,
            32,
            icon="snap",
            active=workspace.snap,
            tooltip="Snap · " + app.input_bindings.label(InputAction.SNAP),
        ):
            workspace.snap = not workspace.snap
    workspace.input_overlays = list(overlay)
    if cube_rect is not None:
        overlay.append(cube_rect)
    workspace.rects["overlays"] = overlay
    imgui.pop_clip_rect()
    if not workspace.bottom_open:
        yy = y + h
        u.rect(x, yy, w, 38 * s, PANEL, 0)
        u.line(x, yy, w)
        if u.button(
            "expand-timeline",
            x + 8 * s,
            yy + 5 * s,
            26,
            28,
            icon="right",
            tooltip="Expand Timeline",
        ):
            workspace.toggle_timeline()
        timeline_width = max(95, u.measure(u.translate("Timeline")) / s + 39)
        output_width = max(85, u.measure(u.translate("Output")) / s + 39)
        if u.button(
            "timeline-tab",
            x + 38 * s,
            yy + 5 * s,
            timeline_width,
            28,
            icon="key",
            label="Timeline",
        ):
            workspace.activate("Timeline")
        if w > (timeline_width + output_width + 53) * s and u.button(
            "output-tab",
            x + (45 + timeline_width) * s,
            yy + 5 * s,
            output_width,
            28,
            icon="terminal",
            label="Output",
        ):
            workspace.activate("Output")
    app._draw_workspace_camera_preview(preview_name)
    imgui.end()


def pending_model_edits(workspace, x, y, w, h, overlay, *, bottom_limit=None):
    u, s, app = workspace.ui, workspace.s, workspace.app
    width = min(w / s - 20, 392)
    compact = width < 392 and h < 300 * s
    stacked = width < 392 and not compact
    height = 70 if stacked else 40
    px = x + (w - width * s) / 2
    py = (
        bottom_limit - (height + 8) * s
        if bottom_limit is not None
        else y + max(10, h / s - height - 10) * s
    )
    u.rect(px, py, width * s, height * s, (22, 25, 28, 245), 8)
    overlay.append((px, py, width * s, height * s))
    label = "Applying…" if app.model_edits.applying else "Pending model edits"
    if compact:
        u.button("model-edit-status", px + 8 * s, py + 6 * s, 28, 28, icon="info", tooltip=label)
    else:
        u.text(px + 12 * s, py + 13 * s, label, AMBER, BODY_SIZE)
    if app.model_edits.error and imgui.is_mouse_hovering_rect(
        (px, py), (px + width * s, py + height * s)
    ):
        u.tooltip(app.model_edits.error)
    apply_width = min(76, (width - 52) / 2) if compact else 76
    discard_width = min(80, (width - 52) / 2) if compact else 80
    bx = px + (width - apply_width - discard_width - 14 if not stacked else 8) * s
    by = py + (36 if stacked else 6) * s
    enabled = not app.model_edits.applying and not app.gizmo.using
    if u.button(
        "apply-model-edits",
        bx,
        by,
        apply_width,
        28,
        label="" if compact else "Apply",
        icon="check" if compact else None,
        tooltip="Apply" if compact else None,
        active=True,
        solid=True,
        enabled=enabled,
    ):
        app._apply_model_edits_requested = True
    if u.button(
        "discard-model-edits",
        bx + (apply_width + 6) * s,
        by,
        discard_width,
        28,
        label="" if compact else "Discard",
        icon="close" if compact else None,
        tooltip="Discard" if compact else None,
        filled=True,
        enabled=enabled,
    ):
        app.model_edits.clear()
        app.gizmo._reset_model_placement()
        app._apply_model_edits_requested = False


def transport(workspace, x, y, w, shell):
    u, s = workspace.ui, workspace.s
    session = workspace.app.session
    replay = session.replay_info
    compact = w < 340 * s
    width = 168 if compact else 282
    if w < (width + 20) * s:
        return
    tx = x + (w - width * s) / 2
    ty = y + (104 if w < 500 * s else 52 if w < 700 * s else 10) * s
    padding = DESIGN.transport_padding_y
    height = DESIGN.control_height + 2 * padding
    shell(tx, ty, width, height)
    if u.button(
        "previous",
        tx + 4 * s,
        ty + padding * s,
        26,
        28,
        icon="previous",
        icon_role="subtle",
        tooltip="Previous frame",
        enabled=replay.frame_index > 0 if replay is not None else session.can_step_back,
    ):
        session.submit(
            cmd.SeekReplay(max(0, replay.frame_index - 1)) if replay is not None else cmd.StepBack()
        )
    if u.button(
        "play",
        tx + 33 * s,
        ty + padding * s,
        32,
        28,
        icon="pause" if workspace.playing else "play",
        icon_role="subtle",
        active=True,
        tooltip="Play / pause",
    ):
        workspace.app._toggle_playback()
    if u.button(
        "next",
        tx + 69 * s,
        ty + padding * s,
        26,
        28,
        icon="next",
        icon_role="subtle",
        tooltip="Next frame",
        enabled=not workspace.playing
        and (
            replay.frame_index < replay.frame_count - 1
            if replay is not None
            else session.adapter.caps.simulation
        ),
    ):
        session.submit(
            cmd.SeekReplay(min(replay.frame_count - 1, replay.frame_index + 1))
            if replay is not None
            else cmd.Step()
        )
    if not compact:
        caption = f"{workspace.time:.3f} s"
        imgui.push_font(u.mono, MONO_SIZE * s)
        caption_width = imgui.calc_text_size(caption).x
        imgui.pop_font()
        u.text(
            tx + 142 * s - caption_width / 2,
            ty + (height - MONO_SIZE) * s / 2,
            caption,
            TEXT,
            MONO_SIZE,
            mono=True,
        )
    for offset in (100,) if compact else (102, 182):
        u.draw.add_line(
            (tx + offset * s, ty + (padding + 5) * s),
            (tx + offset * s, ty + (height - padding - 5) * s),
            color((255, 255, 255, 32)),
            s,
        )
    if u.button(
        "go-start",
        tx + (107 if compact else 189) * s,
        ty + padding * s,
        27,
        28,
        icon="reset",
        icon_role="subtle",
        tooltip="Go to start",
    ):
        workspace.app._reset_playback()
    if not compact and u.button(
        "record",
        tx + 220 * s,
        ty + padding * s,
        26,
        28,
        icon="record",
        icon_role="subtle",
        active=workspace.app.session.state_take_recording,
        tooltip="Record state take",
        enabled=session.adapter.caps.simulation,
    ):
        workspace.app._toggle_state_take_recording()
    playback_clicked = u.button(
        "playback-options",
        tx + (138 if compact else 250) * s,
        ty + padding * s,
        26,
        28,
        icon="down",
        icon_role="subtle",
        tooltip="Playback options",
        active=imgui.is_popup_open("Playback"),
    )
    workspace.menus.popup("Playback", u.hits["playback-options"], playback_clicked)
