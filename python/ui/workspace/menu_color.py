"""Native material color popup and its Session edit transaction."""

from imgui_bundle import imgui

from mojive import commands as cmd

from .design import BODY_SIZE, MONO_SIZE
from .widgets import DIM, MUTED, TEXT


def color_control(menu, entity, x, y, width):
    from .entities import rgba

    u, s = menu.ui, menu.s
    u.text(x, y, "Base color", MUTED, BODY_SIZE)
    clicked = u.button("material-color", x, y + 22 * s, width / s, 30, filled=True)
    u.rect(
        x + 7 * s,
        y + 29 * s,
        16 * s,
        16 * s,
        tuple(int(v * 255) for v in rgba(entity["color"])),
        3,
    )
    u.text(x + 34 * s, y + 31 * s, entity["color"].upper(), TEXT, MONO_SIZE, mono=True)
    u.role_icon("down", x + width - 23 * s, y + 23 * s, 12 * s, 28 * s, role="inline", c=DIM)
    with menu.style(controls=True):
        if (clicked or menu.requested_path == ("Color",)) and not imgui.is_popup_open("Color"):
            menu.color_before = True
            menu.workspace.app.session.submit(cmd.BeginEditTransaction("Edit color"))
            menu.color_entity = entity
            menu.color_editor.begin(entity["rgba"])
            imgui.open_popup("Color")
        if not imgui.is_popup_open("Color"):
            return
        vp = imgui.get_main_viewport()
        content_w = min(300 * s, max(136 * s, vp.work_size.x - 48 * s))
        popup_w = content_w + 32 * s
        popup_h = min(vp.work_size.y - 16 * s, content_w + 248 * s)
        px = min(x, vp.pos.x + vp.size.x - popup_w - 8 * s)
        py = y + 56 * s
        if py + popup_h > vp.pos.y + vp.size.y - 8 * s:
            py = max(vp.pos.y + 8 * s, y - popup_h)
        imgui.set_next_window_pos((px, py))
        imgui.set_next_window_size_constraints((popup_w, 0), (popup_w, vp.size.y - 16 * s))
        if not imgui.begin_popup("Color"):
            return
        menu.color_visible = True
        record = menu.shadow_and_record("Color", ("Color",))
        record["controls"] = {}
        imgui.text_unformatted(u.translate("Base color"))
        changed, values = menu.color_editor.draw(u, "color-editor", entity["rgba"], content_w)
        record["controls"]["picker"] = list(u.hits["color-editor-picker"])
        if "color-editor-hex" in u.hits:
            record["controls"]["hex"] = list(u.hits["color-editor-hex"])
        if changed:
            entity["rgba"] = values
            entity["color"] = "#" + "".join(f"{round(value * 255):02x}" for value in values[:3])
        p = imgui.get_cursor_screen_pos()
        u.text(p.x, p.y + 8 * s, "Esc to cancel", DIM, BODY_SIZE)
        if u.button(
            "done-color",
            p.x + content_w - 64 * s,
            p.y,
            64,
            28,
            label="Done",
            active=True,
            enabled=not menu.color_editor.error,
        ):
            imgui.close_current_popup()
        record["controls"]["done"] = list(u.hits["done-color"])
        imgui.end_popup()


def finish_color_edit(menu):
    if menu.color_before is not None and not menu.color_visible:
        menu.workspace.app.session.submit(cmd.EndEditTransaction())
        menu.color_before = None
        menu.color_entity = None
