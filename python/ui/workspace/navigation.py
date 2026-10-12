"""Scene and joint navigation backed by the shared hierarchy and joint panels."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.render.geometry import GeometryView, geometry_view, set_geometry_view
from mojive.ui.compound_fields import draw_focus_frame
from mojive.ui.controls import segmented_control_width
from mojive.ui.panels import (
    activate_edit_gizmo,
    copy_state_vector,
    pointer_pressed,
    publish_focus_item_hint,
    search_input,
)
from mojive.ui.panels.filters import node_filter_color
from mojive.ui.panels.hierarchy import _TYPE_FILTERS
from mojive.ui.panels.joints import filter_joints, sort_joints
from mojive.ui.pointer_bindings import PointerAction

from .design import BODY_SIZE, DESIGN
from .pose_panel import NativePoseEditor, joint_icon
from .widgets import DIM, INPUT, MUTED

GEOMETRY_VIEWS = (GeometryView.VISUAL, GeometryView.COLLISION, GeometryView.BOTH)
GEOMETRY_ICONS = ("visual", "collision", "both")

NODE_ICONS = {
    "world": "world",
    "model": "folder",
    "robot": "robot",
    "link": "link",
    "geom": "geom",
    "joint": "joint",
    "light": "light",
    "camera": "camera",
    "environment": "env",
    "site": "target",
    "flex": "geom",
    "skin": "body",
}


def node_icon(node, session):
    if node.type is NodeType.JOINT:
        joint = next(
            (joint for joint in session.joints if joint.joint_id == node.joint_index), None
        )
        if joint is not None:
            return joint_icon(joint.type)
    return NODE_ICONS[node.type.value]


def node_accent(theme, kind):
    """Preserve semantic hues while keeping dark node badges readable."""
    color = node_filter_color(theme, kind)
    amount = 0.4 if max(color[:3]) < 0.4 else 0.0
    return (*(round((value + (1.0 - value) * amount) * 255) for value in color[:3]), 255)


def node_text_color(theme, kind, visible=True):
    """Use a restrained entity hue for names while keeping hidden items muted."""
    if not visible:
        return DIM
    accent = node_accent(theme, kind)
    return (
        *(
            round(base * 0.55 + tint * 0.45)
            for base, tint in zip(MUTED[:3], accent[:3], strict=True)
        ),
        255,
    )


class Navigation:
    def __init__(self, workspace):
        self.workspace = workspace
        self.hierarchy = workspace.app.panels.get("hierarchy")
        self.joints = workspace.app.panels.get("joints")
        self._selection_token = None
        self._scene_filter_token = None
        self._scroll_scene_node = None
        self._range_anchor = None
        self.pose_search_visible = False
        self._pose_copy_pending = False
        self.pose_editor = NativePoseEditor(workspace)

    def scene_filters(self, ctx):
        self.hierarchy._refresh(ctx)
        return tuple(
            (kind, self.hierarchy._type_counts.get(kind, 0))
            for kind in _TYPE_FILTERS
            if kind == "all"
            or self.hierarchy._type_counts.get(kind, 0)
            or kind == self.workspace.filter.casefold()
        )

    def scene_rows(self, ctx):
        panel, session = self.hierarchy, ctx.session
        panel._refresh(ctx)
        panel._filter = self.workspace.query
        panel._type_filter = self.workspace.filter.casefold()
        panel._follow_selection(session)
        selected = session.selected_node
        token = self._scene_selection_token(session)
        if token != self._selection_token:
            self._selection_token = token
            self._range_anchor = selected.node_id if selected else None
            self._scroll_scene_node = selected.node_id if selected else None
            parent = panel._by_id.get(selected.parent) if selected else None
            while parent is not None:
                panel._open_state[parent.node_id] = True
                parent = panel._by_id.get(parent.parent)
        return panel._visible_rows()

    def select_scene(self, ctx, node, *, focused=False, additive=False, range_select=False):
        rows = [row[0].node_id for row in self.scene_rows(ctx)]
        if range_select and self._range_anchor in rows and node.node_id in rows:
            first, last = sorted((rows.index(self._range_anchor), rows.index(node.node_id)))
            selected = set(rows[first : last + 1])
            if additive:
                selected.update(self.hierarchy._batch_selected)
            result = ctx.submit(cmd.SelectNode(node.node_id))
            if result.ok:
                self.hierarchy._batch_selected = selected
                self.hierarchy._follow_selection(ctx.session, keep_batch=True)
        else:
            self.hierarchy._select_row(ctx, node.node_id, additive=additive)
            self._range_anchor = node.node_id
        self._selection_token = self._scene_selection_token(ctx.session)
        self._scroll_scene_node = None
        if focused and ctx.focus_node is not None:
            ctx.focus_node(node.node_id)

    @staticmethod
    def _scene_selection_token(session):
        selected = session.selected_node
        return (
            session.document_id,
            session.structure_generation,
            session.selection_revision,
            selected.node_id if selected else -1,
        )

    def select_joint(self, ctx, joint, node, *, focused=False):
        if node is None:
            return
        ctx.submit(cmd.SelectNode(node.node_id))
        if focused:
            activate_edit_gizmo(ctx, node, joint)
            if ctx.focus_joint is not None:
                ctx.focus_joint(joint.joint_id)

    def _pill(self, key, x, y, width, count, icon, tooltip, accent, active):
        u, s = self.workspace.ui, self.workspace.s
        imgui.set_cursor_screen_pos((x, y))
        clicked = imgui.invisible_button(key, (width * s, 28 * s), imgui.ButtonFlags_.enable_nav)
        hovered = imgui.is_item_hovered()
        u.hits[key] = (x, y, width * s, 28 * s)
        background = (*accent[:3], 40 if active else 24) if active or hovered else INPUT
        u.rect(x, y, width * s, 28 * s, background, 14)
        u.role_icon(icon, x + 6 * s, y + 6 * s, 16 * s, 16 * s, accent, role="inline")
        u.text(
            x + 27 * s,
            y + DESIGN.control_padding_y(BODY_SIZE) * s,
            str(count),
            accent if active else MUTED,
            BODY_SIZE,
            translate=False,
        )
        draw_focus_frame(
            imgui.ImVec2(x, y),
            imgui.ImVec2(x + width * s, y + 28 * s),
            rounding=14 * s,
            item_id=imgui.get_item_id(),
        )
        if hovered:
            u.tooltip(tooltip, translate=False)
        return clicked

    def _scene_controls(self, ctx, x, y, w):
        u, s = self.workspace.ui, self.workspace.s
        if self.workspace.search_visible:
            imgui.set_cursor_screen_pos((x, y))
            imgui.set_next_item_width(w)
            _, self.workspace.query = search_input(
                "##search", self.workspace.query, hint=ctx.tr("Search scene")
            )
            y += 38 * s
        labels = tuple(ctx.tr(view.value.title()) for view in GEOMETRY_VIEWS)
        current = geometry_view(ctx.backend)
        imgui.push_font(u.body, BODY_SIZE * s)
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding, (6 * s, DESIGN.control_padding_y(BODY_SIZE) * s)
        )
        show_labels = segmented_control_width(labels, icons=GEOMETRY_ICONS, layout="content") <= w
        imgui.pop_style_var()
        imgui.pop_font()
        chosen = u.segmented(
            "scene-geometry",
            x,
            y,
            w / s,
            labels,
            GEOMETRY_VIEWS.index(current) if current in GEOMETRY_VIEWS else -1,
            keys=tuple("scene-geometry-" + view.value for view in GEOMETRY_VIEWS),
            icons=GEOMETRY_ICONS,
            show_labels=show_labels,
            layout="content",
            tooltips=(
                "Show visual geometry",
                "Show collision geometry",
                "Show visual and collision geometry",
            ),
        )
        if chosen >= 0 and GEOMETRY_VIEWS[chosen] != current:
            set_geometry_view(ctx.backend, GEOMETRY_VIEWS[chosen])
        y += u.hits["scene-geometry"][3] + 8 * s
        xx = x
        for kind, count in self.scene_filters(ctx):
            width = min(w / s, u.measure(str(count), BODY_SIZE, translate=False) / s + 35)
            if xx + width * s > x + w:
                xx, y = x, y + 32 * s
            if self._pill(
                "scene-filter-" + kind,
                xx,
                y,
                width,
                count,
                "tree" if kind == "all" else NODE_ICONS[kind],
                f"{ctx.tr(kind.title())}: {count}",
                node_accent(ctx.theme, kind),
                self.workspace.filter.casefold() == kind,
            ):
                self.workspace.filter = kind.title()
            xx += (width + 4) * s
        return y + 36 * s

    def draw_scene(self, ctx, x, y, w, h):
        s = self.workspace.s
        top = self._scene_controls(ctx, x, y, w)
        rows = self.scene_rows(ctx)
        imgui.set_cursor_screen_pos((x, top))
        if rows:
            # Clipped final rows must retain the same scroll extent as submitted rows.
            imgui.set_next_window_content_size((0, len(rows) * 30 * s))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Scene entities", (w, max(s, h - (top - y))))
        imgui.pop_style_var()
        try:
            token = (self.workspace.query.casefold(), self.workspace.filter.casefold())
            if token != self._scene_filter_token:
                self._scene_filter_token = token
                imgui.set_scroll_y(0)
            if self._scroll_scene_node is not None:
                index = next(
                    (
                        i
                        for i, (node, _, _) in enumerate(rows)
                        if node.node_id == self._scroll_scene_node
                    ),
                    None,
                )
                if index is not None:
                    imgui.set_scroll_y(max(0, index * 30 * s - imgui.get_window_height() * 0.5))
                self._scroll_scene_node = None
            width = imgui.get_content_region_avail().x
            clipper = imgui.ListClipper()
            clipper.begin(len(rows), 30 * s)
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    self._scene_row(ctx, *rows[index], width)
            clipper.end()
            if rows:
                publish_focus_item_hint(ctx)
            else:
                imgui.text_disabled(ctx.tr("no match"))
        finally:
            imgui.end_child()

    def _scene_row(self, ctx, node, depth, leaf, width):
        u, s = self.workspace.ui, self.workspace.s
        pos = imgui.get_cursor_screen_pos()
        selected = (
            node.node_id in self.hierarchy._batch_selected or ctx.session.selected_node is node
        )
        indent = min(depth * 14 * s, max(0, width - 110 * s))
        hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            pos, (pos.x + width, pos.y + 28 * s)
        )
        if selected or hovered:
            u.rect(
                pos.x,
                pos.y,
                width,
                28 * s,
                (156, 191, 141, 30) if selected else (255, 255, 255, 12),
            )
        if not leaf:
            opened = self.hierarchy._open_state.get(
                node.node_id, depth < self.hierarchy._default_open_depth
            )
            if u.button(
                "expand-" + str(node.node_id),
                pos.x + indent,
                pos.y,
                18,
                icon="down" if opened else "right",
                icon_role="inline",
                background=False,
            ):
                self.hierarchy._open_state[node.node_id] = not opened
        pressed = u.button(
            "entity-" + str(node.node_id),
            pos.x + indent + 18 * s,
            pos.y,
            max(1, (width - indent) / s - 46),
            icon=node_icon(node, ctx.session),
            icon_role="inline",
            label=node.name or "?",
            text_color=node_text_color(ctx.theme, node.type.value, node.visible),
            icon_color=node_accent(ctx.theme, node.type.value),
            background=False,
            translate=False,
            align="left",
        )
        focused = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
        clicked = imgui.is_item_clicked() or (pressed and not imgui.is_mouse_released(0))
        if clicked or focused:
            self.select_scene(
                ctx,
                node,
                focused=focused,
                additive=pointer_pressed(ctx, PointerAction.ADD_SELECTION),
                range_select=imgui.get_io().key_shift,
            )
        self._scene_context(ctx, node)
        if node.type not in (NodeType.MODEL, NodeType.ENVIRONMENT) and u.button(
            "visible-" + str(node.node_id),
            pos.x + width - 26 * s,
            pos.y + s,
            24,
            26,
            icon="eye" if node.visible else "eyeoff",
            icon_role="inline",
            tooltip=ctx.tr("hide" if node.visible else "show"),
            background=False,
        ):
            if node.type is NodeType.LIGHT and node.light_index >= 0:
                light = ctx.session.source.lights.lights[node.light_index]
                ctx.submit(cmd.SetLight(node.light_index, replace(light, active=not light.active)))
            else:
                ctx.submit(cmd.SetVisible(node.node_id, not node.visible))
        imgui.set_cursor_screen_pos((pos.x, pos.y))
        imgui.dummy((width, 30 * s - imgui.get_style().item_spacing.y))

    def _scene_context(self, ctx, node):
        if not imgui.begin_popup_context_item("entity-context-" + str(node.node_id)):
            return
        try:
            clicked, _ = imgui.menu_item(ctx.tr("Focus item"), "", False, ctx.model_camera_id < 0)
            if clicked and ctx.focus_node is not None:
                ctx.focus_node(node.node_id)
            clicked, _ = imgui.menu_item(ctx.tr("Copy name"), "", False)
            if clicked:
                imgui.set_clipboard_text(node.name)
            if (node.type is NodeType.MODEL and node.model_id >= 0) or node.source_editable:
                imgui.separator()
                if node.type in (NodeType.MODEL, NodeType.ROBOT, NodeType.LINK):
                    self.hierarchy._model_create_menu(ctx, node)
                if node.type is NodeType.MODEL:
                    clicked, _ = imgui.menu_item(
                        ctx.tr("Remove Model"), "", False, ctx.session.paused
                    )
                    if clicked:
                        ctx.submit(cmd.RemoveSceneModel(node.model_id))
                else:
                    self._entity_actions(ctx, node, model=True)
            elif (
                node.object_id
                and node.model_id < 0
                and ctx.session.adapter.caps.scene_authoring
                and node.type in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
            ):
                imgui.separator()
                self._entity_actions(ctx, node, model=False)
        finally:
            imgui.end_popup()

    @staticmethod
    def _entity_actions(ctx, node, *, model):
        enabled = not ctx.session.adapter.caps.simulation or ctx.session.paused
        duplicate, _ = imgui.menu_item(ctx.tr("Duplicate"), "Cmd/Ctrl+D", False, enabled)
        rename, _ = imgui.menu_item(ctx.tr("Rename"), "F2", False, enabled)
        remove, _ = imgui.menu_item(
            ctx.tr("Delete from Model" if model else "Delete"), "Delete", False, enabled
        )
        if duplicate:
            result = ctx.submit(
                cmd.DuplicateModelElement(node.node_id)
                if model
                else cmd.DuplicateSceneEntity(node.object_id)
            )
            if result.ok:
                ctx.submit(
                    cmd.SelectNode(result.entity_id) if model else cmd.Select(result.entity_id)
                )
        if rename:
            request = ctx.request_model_rename if model else ctx.request_rename
            if request is not None:
                request(node.node_id if model else node.object_id)
        if remove:
            ctx.submit(
                cmd.RemoveModelElement(node.node_id)
                if model
                else cmd.RemoveSceneEntity(node.object_id)
            )

    def pose_joints(self, ctx, node_id=None):
        self.joints._snapshot(ctx)
        joints = ctx.session.joints
        if node_id is not None:
            node = ctx.session.node(node_id)
            if node is None:
                return ()
            joints = (
                tuple(joint for joint in joints if joint.joint_id == node.joint_index)
                if node.type is NodeType.JOINT
                else ctx.session.joints_for_body(node.body_index)
            )
        return sort_joints(
            filter_joints(joints, "" if node_id is not None else self.joints._search),
            by_name=self.joints._sort_by_name,
        )

    def draw_pose(self, ctx, x, y, w, *, node_id=None):
        if node_id is None:
            y = self._pose_controls(ctx, x, y, w)
        joints = self.pose_joints(ctx, node_id)
        return self.pose_editor.draw(
            ctx,
            x,
            y,
            w,
            joints=joints,
            select_joint=self.select_joint,
            show_names=node_id is None or len(joints) > 1,
            joint_color=node_accent(ctx.theme, "joint"),
        )

    def pose_action(self, key):
        if key == "search":
            self.pose_search_visible = not self.pose_search_visible
            if not self.pose_search_visible:
                self.joints._search = ""
        elif key == "sort":
            self.joints._sort_by_name = not self.joints._sort_by_name
        elif key == "copy":
            self._pose_copy_pending = True

    def pose_action_active(self, key):
        return (key == "search" and self.pose_search_visible) or (
            key == "sort" and self.joints._sort_by_name
        )

    def _pose_controls(self, ctx, x, y, w):
        if self._pose_copy_pending:
            imgui.open_popup("Joint state")
            self._pose_copy_pending = False
        if imgui.begin_popup("Joint state"):
            copy_qpos, _ = imgui.menu_item(
                ctx.tr("Copy qpos"), "", False, ctx.session.frame.qpos is not None
            )
            copy_qvel, _ = imgui.menu_item(
                ctx.tr("Copy qvel"),
                "",
                False,
                ctx.session.frame.qvel is not None or ctx.session.adapter.caps.state_snapshots,
            )
            if copy_qpos:
                copy_state_vector(ctx.session.frame.qpos)
            if copy_qvel:
                state = (
                    ctx.session.adapter.capture_state() if ctx.session.frame.qvel is None else None
                )
                copy_state_vector(ctx.session.frame.qvel if state is None else state.qvel)
            imgui.end_popup()
        if self.pose_search_visible:
            imgui.set_cursor_screen_pos((x, y))
            imgui.set_next_item_width(w)
            _, self.joints._search = search_input(
                "##pose-search", self.joints._search, hint=ctx.tr("Search joints")
            )
            y += 38 * self.workspace.s
        return y
