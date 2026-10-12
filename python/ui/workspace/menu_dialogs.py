"""Native interaction guide dialog; document dialogs remain owned by ViewerApp."""

from imgui_bundle import imgui

from mojive.ui.input_bindings import InputAction

from .design import BODY_SIZE
from .widgets import DIM, MUTED, TEXT


def dialogs(menu):
    if menu.dialog_pending:
        menu.dialog = menu.dialog_pending
        menu.dialog_pending = None
        imgui.open_popup("##workspace-dialog")
    with menu.style(controls=True):
        vp = imgui.get_main_viewport()
        width, height = (
            min(500 * menu.s, vp.size.x - 32 * menu.s),
            min(430 * menu.s, vp.size.y - 32 * menu.s),
        )
        imgui.set_next_window_size((width, height))
        imgui.set_next_window_pos(
            (vp.pos.x + vp.size.x / 2, vp.pos.y + vp.size.y / 2),
            imgui.Cond_.appearing,
            (0.5, 0.5),
        )
        opened, _ = imgui.begin_popup_modal(
            "##workspace-dialog",
            None,
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_saved_settings,
        )
        if not opened:
            return
        menu.shadow_and_record("Dialog", ("Dialog",))
        u, s = menu.ui, menu.s
        p = imgui.get_cursor_screen_pos()
        w = imgui.get_content_region_avail().x
        u.text(p.x, p.y + 3 * s, menu.dialog, TEXT, 17, bold=True)
        if u.button(
            "close-guide", p.x + w - 26 * s, p.y, 26, 26, icon="close", tooltip="Close dialog"
        ) or imgui.is_key_pressed(imgui.Key.escape):
            imgui.close_current_popup()
        u.text(p.x, p.y + 32 * s, "MOJIVE", DIM, BODY_SIZE)
        imgui.set_cursor_screen_pos((p.x, p.y + 60 * s))
        imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, 0))
        imgui.begin_child("Guide content", (w, max(24 * s, height - 138 * s)))
        if menu.dialog == "Interaction guide":
            bindings = menu.workspace.app.input_bindings
            for title, value in [
                ("Orbit", "Drag an empty area"),
                ("Pan", "Middle mouse drag"),
                ("Zoom", "Mouse wheel"),
                (
                    "Fly camera",
                    " / ".join(
                        " ".join(bindings.label(action) for action in group)
                        for group in (
                            (
                                InputAction.FLY_FORWARD,
                                InputAction.FLY_LEFT,
                                InputAction.FLY_BACK,
                                InputAction.FLY_RIGHT,
                            ),
                            (InputAction.FLY_UP, InputAction.FLY_DOWN),
                        )
                    ),
                ),
                (
                    "Transform tools",
                    "V / "
                    + " / ".join(
                        bindings.label(action)
                        for action in (InputAction.GIZMO_TRANSLATE, InputAction.GIZMO_ROTATE)
                    ),
                ),
                ("World / body frame", bindings.label(InputAction.GIZMO_SPACE)),
                ("Snap", bindings.label(InputAction.SNAP)),
                (
                    "Frame all / focus",
                    bindings.label(InputAction.FRAME_SCENE) + " / double-click",
                ),
                (
                    "Settings",
                    "F9 / " + ("Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl") + ",",
                ),
                ("Precise gizmo input", "Double-click a handle"),
                (
                    "Undo / redo",
                    ("Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl") + "+Z / Shift+Z",
                ),
                ("Simulation playback", "Space"),
                ("Timeline", "T"),
                ("Dock panels", "Drag a panel tab"),
            ]:
                q = imgui.get_cursor_screen_pos()
                column = w * 0.47
                if (
                    u.measure(title, BODY_SIZE) + 12 * s > column
                    or u.measure(value, BODY_SIZE) > w - column
                ):
                    imgui.push_font(u.body, BODY_SIZE * s)
                    imgui.push_text_wrap_pos(0)
                    imgui.text_colored(tuple(v / 255 for v in MUTED), u.translate(title))
                    imgui.text_wrapped(u.translate(value))
                    imgui.pop_text_wrap_pos()
                    imgui.pop_font()
                    imgui.dummy((0, 8 * s))
                else:
                    u.text(q.x, q.y + 6 * s, title, MUTED, BODY_SIZE)
                    u.text(q.x + column, q.y + 6 * s, value, TEXT, BODY_SIZE)
                    imgui.dummy((w, 28 * s))
        else:
            imgui.text_wrapped(menu.workspace.app.localizer.text("About"))
            imgui.spacing()
            imgui.text_disabled(
                menu.workspace.app.localizer.text("Dear ImGui") + " " + imgui.get_version()
            )
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()
        bottom = imgui.get_window_pos().y + height - 44 * s
        u.line(p.x, bottom - 10 * s, w)
        if u.button("done-guide", p.x + w - 72 * s, bottom, 72, 28, label="Done", active=True):
            imgui.close_current_popup()
        imgui.end_popup()
