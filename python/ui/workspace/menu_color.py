"""Native material color popup and its Session edit transaction."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Rect
from mojive.ui.workspace.design import BODY_SIZE, MONO_SIZE
from mojive.ui.workspace.widgets import DIM, MUTED, TEXT


def color_control(menu, entity, x, y, width):
    from mojive.ui.workspace.entities import rgba

    u, s = menu.ui, menu.s
    u.text(x, y, "Base color", MUTED, BODY_SIZE)
    clicked = u.button(
        "material-color", x, y + 22 * s, width / s, 30, filled=True, width_limit=width
    )
    frame = Rect(*u.hits["material-color"])
    style = u.component_style("color_input")
    content = frame.inset(*style.edges("padding", s, (7, 6, 7, 6)))
    arrow_width = min(12 * s, content.width)
    size = min(16 * s, content.height, max(0, content.width - arrow_width))
    gap = min(10 * s, max(0, content.width - arrow_width - size))
    text_x = content.x + size + gap
    arrow_x = content.right - arrow_width
    text_width = max(0, arrow_x - text_x - 4 * s)
    u.draw.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
    u.rect(
        content.x,
        content.y + (content.height - size) / 2,
        size,
        size,
        tuple(int(v * 255) for v in rgba(entity["color"])),
        3,
    )
    u.text(
        text_x,
        content.y + max(0, (content.height - MONO_SIZE * s) / 2),
        entity["color"].upper(),
        TEXT,
        MONO_SIZE,
        mono=True,
        max_width=text_width,
    )
    u.role_icon(
        "down",
        arrow_x,
        content.y,
        arrow_width,
        content.height,
        role="inline",
        c=DIM,
    )
    u.draw.pop_clip_rect()
    if (
        imgui.is_item_hovered()
        and u.measure(entity["color"], MONO_SIZE, mono=True, translate=False) > text_width
    ):
        u.tooltip(entity["color"].upper(), translate=False)
    if u.design is not None:
        u.design.record(
            "material-color-content",
            ("color_input",),
            frame.tuple(),
            "color_input_content",
            owner="material-color",
            content=content.tuple(),
            swatchRect=[content.x, content.y + (content.height - size) / 2, size, size],
            arrowRect=[arrow_x, content.y, arrow_width, content.height],
            textRect=[text_x, content.y, text_width, content.height],
            textPresent=text_width > 0,
            minimumTextWidth=u.measure(entity["color"], MONO_SIZE, mono=True, translate=False),
        )
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
        content_w = imgui.get_content_region_avail().x
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
