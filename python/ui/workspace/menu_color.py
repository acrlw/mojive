"""Native material color popup and its Session edit transaction."""

from dataclasses import dataclass

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.geometry import Rect
from mojive.ui.workspace.color_editor import color_popup, color_popup_width, place_color_popup
from mojive.ui.workspace.component_layout import control_size
from mojive.ui.workspace.design import BODY_SIZE, MONO_SIZE
from mojive.ui.workspace.widgets import DIM, MUTED, TEXT


@dataclass
class _ColorEdit:
    document_id: str
    entity_identity: tuple
    structure_generation: int
    session_checkpoint: object
    draft_checkpoint: object
    history: bool
    colors: tuple

    def owns(self, app):
        return (
            app.session.document_id == self.document_id
            and app.session._edit_before is self.session_checkpoint
            and app.model_edits._checkpoint is self.draft_checkpoint
        )


def _begin_color_edit(menu, entity):
    from mojive.scene.queries import node_geometry_indices
    from mojive.ui.workspace.entities import EntityProjection

    app, session = menu.workspace.app, menu.workspace.app.session
    history = session.adapter.caps.edit_history
    if session.editing or app.model_edits._checkpoint is not None:
        session._record_result(cmd.CommandResult.bad("An edit transaction is already active"))
        return False
    if history and not session.submit(cmd.BeginEditTransaction("Edit color")).ok:
        return False
    node = session.node(entity["node_id"])
    indices = node_geometry_indices(session, node)
    colors = {}
    for index in indices:
        node_id = int(session.source.geom_node[index])
        target = session.node(node_id)
        colors.setdefault(
            node_id,
            (node_id, EntityProjection._identity(target), session.source.geom_rgba[index].copy()),
        )
    menu.color_edit = _ColorEdit(
        session.document_id,
        entity["identity"],
        session.structure_generation,
        session._edit_before,
        app.model_edits._checkpoint,
        history,
        tuple(colors.values()),
    )
    return True


def _restore_colors(menu, edit):
    from mojive.ui.workspace.entities import EntityProjection

    session = menu.workspace.app.session
    for node_id, identity, _ in edit.colors:
        node = session.node(node_id)
        if node is None or EntityProjection._identity(node) != identity:
            result = cmd.CommandResult.bad("The scene changed; the color edit cannot be restored")
            session._record_result(result)
            return result
    for node_id, _, rgba in edit.colors:
        instances = np.flatnonzero(session.source.geom_node == node_id)
        if np.all(session.source.geom_rgba[instances] == rgba):
            continue
        result = session.submit(cmd.SetGeometryColor(node_id, rgba))
        if not result.ok:
            return result
    return cmd.CommandResult.good()


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
        edit = menu.color_edit
        if edit is not None and (
            not edit.owns(menu.workspace.app)
            or edit.structure_generation != menu.workspace.app.session.structure_generation
            or edit.entity_identity != entity["identity"]
        ):
            # Changing the inspected entity ends the old gesture before its
            # picker can write through the new entity's projected fields.
            finish_color_edit(menu)
            with color_popup(u, "Color") as opened:
                if opened:
                    imgui.close_current_popup()
            return
        if (clicked or menu.requested_path == ("Color",)) and not imgui.is_popup_open("Color"):
            if not _begin_color_edit(menu, entity):
                return
            menu.color_editor.begin(entity["rgba"])
            imgui.open_popup("Color")
        if not imgui.is_popup_open("Color"):
            return
        vp = imgui.get_main_viewport()
        popup_w = color_popup_width(u, len(entity["rgba"]))
        px = min(x, vp.work_pos.x + vp.work_size.x - popup_w - 8 * s)
        py = vp.work_pos.y + 8 * s
        imgui.set_next_window_pos((px, py))
        imgui.set_next_window_size_constraints((popup_w, 0), (popup_w, vp.work_size.y - 16 * s))
        place_color_popup(u, "Color", position=(px, py))
        with color_popup(u, "Color") as opened:
            if not opened:
                return
            if menu.color_edit is None:
                imgui.close_current_popup()
                return
            menu.color_visible = True
            record = menu.record_window("Color", ("Color",))
            record["controls"] = {}
            content_w = imgui.get_content_region_avail().x
            changed, values = menu.color_editor.draw(
                u,
                "color-editor",
                entity["rgba"],
                content_w,
                footer_height=control_size(u, "done-color", 64, 28, label="Done")[1],
            )
            record["controls"]["picker"] = list(u.hits["color-editor-picker"])
            if "color-editor-hex" in u.hits:
                record["controls"]["hex"] = list(u.hits["color-editor-hex"])
            if changed:
                entity["rgba"] = values
                entity["color"] = "#" + "".join(f"{round(value * 255):02x}" for value in values[:3])
                menu.workspace.app.apply_document_edits()
                menu.color_edit.session_checkpoint = menu.workspace.app.session._edit_before
                menu.color_edit.structure_generation = (
                    menu.workspace.app.session.structure_generation
                )
            content_w = menu.color_editor.content_width
            p = imgui.get_cursor_screen_pos()
            inset = u.component_style("input_text").edges("padding", s, (8, 0, 8, 0))[0]
            hint = Rect(p.x + inset, p.y, max(0, content_w - 64 * s - inset - 8 * s), 28 * s)
            u.component_text("menu_item", hint, "Esc to cancel", DIM, size=BODY_SIZE)
            if u.design is not None:
                u.design.record(
                    "color-cancel-hint",
                    ("color_editor",),
                    hint.tuple(),
                    "caption",
                    text="Esc to cancel",
                )
            if u.button(
                "done-color",
                p.x + content_w - 64 * s,
                p.y,
                64,
                28,
                label="Done",
                active=True,
                solid=True,
                enabled=not menu.color_editor.error,
            ):
                imgui.close_current_popup()
            record["controls"]["done"] = list(u.hits["done-color"])


def finish_color_edit(menu, *, cancel=False):
    edit = menu.color_edit
    if edit is None or (menu.color_visible and not cancel):
        return
    app = menu.workspace.app
    if not edit.owns(app):
        menu.color_edit = None
        return
    compatible = edit.structure_generation == app.session.structure_generation
    if edit.history:
        result = app.session.submit(
            cmd.CancelEditTransaction() if cancel and compatible else cmd.EndEditTransaction()
        )
    else:
        result = _restore_colors(menu, edit) if cancel and compatible else cmd.CommandResult.good()
    if not compatible and result.ok:
        result = app.session._record_result(
            cmd.CommandResult.bad("The scene changed; the color editor was closed")
        )
    if result.ok or not edit.owns(app) or not edit.history:
        menu.color_edit = None
    if cancel and result.ok:
        app.refresh_document()
    return result
