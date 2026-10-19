"""Scene and joint navigation backed by the shared hierarchy and joint panels."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.design.groups import scoped_parts
from mojive.design.native.geometry import CURRENT_NATIVE_PARTS, Box, Rect, part_scope
from mojive.ui.design_palette import rgba8
from mojive.ui.panels import (
    activate_edit_gizmo,
    copy_state_vector,
    pointer_pressed,
    publish_focus_item_hint,
)
from mojive.ui.panels.filters import node_filter_color
from mojive.ui.panels.hierarchy import _TYPE_FILTERS
from mojive.ui.panels.joints import filter_joints, sort_joints
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.workspace.component_layout import flow_layout
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.filter_chip import draw_filter_chip, filter_chip_size
from mojive.ui.workspace.geometry_view import geometry_selector
from mojive.ui.workspace.pose_panel import NativePoseEditor, joint_icon
from mojive.ui.workspace.widgets import DIM, MUTED
from mojive.ui.workspace.window_chrome import fit_native_content_width

NODE_ICONS = {
    "world": "world",
    "model": "folder",
    "robot": "robot",
    "link": "body",
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
        joint = session.joint(node.joint_index)
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
        self._pose_browse_key = None
        self._pose_browse_source = None
        self._pose_browse_joints = ()
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

    @part_scope("panel_tools", "scene_tools")
    def _scene_controls(self, ctx, x, y, w):
        u, s = self.workspace.ui, self.workspace.s
        top = y
        if ctx.session.world_selection is not None:
            imgui.set_cursor_screen_pos((x, y))
            self.hierarchy._world_selection.draw(ctx)
            y = imgui.get_cursor_screen_pos().y
        tools = u.component_style(("panel_tools", "scene_tools"))
        tools_gap = tools.get("rowGap", tools.get("gap", 8)) * s
        if self.workspace.search_visible:
            _, self.workspace.query = u.search(
                "##search", x, y, w, self.workspace.query, hint=ctx.tr("Search scene")
            )
            y = imgui.get_item_rect_max().y + tools_gap
        y += geometry_selector(u, ctx, x, y, w, key="scene-geometry") + tools_gap
        with u.component_scope("type_filters"):
            filters = tuple(self.scene_filters(ctx))
            group = u.component_style("type_filters")
            sizes = [filter_chip_size(u, count, maximum_width=w) for _, count in filters]
            layout = flow_layout(
                Rect(x, y, w, 0),
                sizes,
                column_gap=(
                    tools.get("columnGap", 4)
                    if tools.explicit("columnGap") or tools.explicit("gap")
                    else group.get("columnGap", group.get("gap", 4))
                )
                * s,
                row_gap=group.get("rowGap", group.get("gap", 4)) * s,
            )
            for (kind, count), rect in zip(filters, layout.rectangles, strict=True):
                if draw_filter_chip(
                    u,
                    "scene-filter-" + kind,
                    rect,
                    count,
                    "tree" if kind == "all" else NODE_ICONS[kind],
                    f"{ctx.tr(kind.title())}: {count}",
                    node_accent(ctx.theme, kind),
                    self.workspace.filter.casefold() == kind,
                ):
                    self.workspace.filter = kind.title()
            u.hits["scene-type-filters"] = (x, y, w, layout.height)
            from mojive.ui.workspace.component_layout import record_group

            record_group(
                u,
                "type_filters",
                "scene-type-filters",
                (u.hits["scene-filter-" + kind] for kind, _ in filters),
            )
            if u.design is not None:
                u.design.record(
                    "scene-panel-tools",
                    ("panel_tools", "scene_tools"),
                    (x, top, w, y + layout.height - top),
                    "scene_tools",
                    fields=[list(r.tuple()) for r in layout.rectangles],
                    fieldRegions={"rowGap": [x, y - tools_gap, w, tools_gap]},
                )
        return y + layout.height + tools_gap

    def draw_scene(self, ctx, x, y, w, h):
        u, s = self.workspace.ui, self.workspace.s
        geometry = self._scene_row_geometry()
        stride = (
            geometry[0] + geometry[2][1] + geometry[2][3] + u.metric("tree_row", "rowGap", 2) * s
        )
        top = self._scene_controls(ctx, x, y, w)
        rows = self.scene_rows(ctx)
        imgui.set_cursor_screen_pos((x, top))
        if rows:
            # Clipped final rows must retain the same scroll extent as submitted rows.
            imgui.set_next_window_content_size((0, len(rows) * stride))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Scene entities", (w, max(s, h - (top - y))))
        imgui.pop_style_var()
        fit_native_content_width(imgui.internal.get_current_window())
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
                    imgui.set_scroll_y(max(0, index * stride - imgui.get_window_height() * 0.5))
                self._scroll_scene_node = None
            width = imgui.get_content_region_avail().x
            clipper = imgui.ListClipper()
            clipper.begin(len(rows), stride)
            try:
                with u.component_scope("tree_row"), u.component_font("tree_row"):
                    while clipper.step():
                        for index in range(clipper.display_start, clipper.display_end):
                            self.draw_scene_row(ctx, *rows[index], width)
            finally:
                # Release native clipper state while its owning context is alive,
                # even when a row fails and its traceback outlives the window.
                clipper.end()
            if rows:
                publish_focus_item_hint(ctx)
            else:
                imgui.text_disabled(ctx.tr("no match"))
        finally:
            imgui.end_child()

    def _scene_row_geometry(self):
        from mojive.ui.workspace.component_layout import control_size

        ui, scale = self.workspace.ui, self.workspace.s
        parents = CURRENT_NATIVE_PARTS.get()
        namespace = tuple(dict.fromkeys((*parents, *scoped_parts("tree_row"))))
        key = (
            imgui.get_frame_count(),
            scale,
            namespace,
            ui.design.runtime.revision if ui.design is not None else None,
        )
        if getattr(self, "_row_geometry_key", None) == key:
            return self._row_geometry
        with ui.component_scope("tree_row"):
            style = ui.component_style("tree_row")
            padding, margin = style.edges("padding", scale), style.edges("margin", scale)
            expander = control_size(ui, "expand-measure", 18, 28, icon="right")
            label = control_size(ui, "entity-measure", 100, 28, label="?", icon="tree")
            visibility = control_size(ui, "visible-measure", 24, 22, icon="eye")
        height = max(
            style.get("height", 28) * scale,
            max(
                expander[1],
                label[1],
                visibility[1],
                style.line_height(style.get("fontSize", 14) * 1.4) * scale,
            )
            + padding[1]
            + padding[3],
        )
        self._row_geometry_key = key
        self._row_geometry = height, padding, margin, expander, label, visibility
        return self._row_geometry

    def draw_scene_row(self, ctx, node, depth, leaf, width):
        with self.workspace.ui.component_scope("tree_row"):
            return self._draw_scene_row(ctx, node, depth, leaf, width)

    def _draw_scene_row(self, ctx, node, depth, leaf, width):
        from mojive.ui.workspace.component_layout import control_frame_size

        u, s = self.workspace.ui, self.workspace.s
        pos = imgui.get_cursor_screen_pos()
        selected = (
            node.node_id in self.hierarchy._batch_selected or ctx.session.selected_node is node
        )
        height, padding, margin, expander_size, label_size, visibility_size = (
            self._scene_row_geometry()
        )
        row_gap = u.metric("tree_row", "rowGap", 2)
        gap = u.metric("tree_row", "columnGap", u.metric("tree_row", "gap", 0)) * s
        origin_x, origin_y = pos.x, pos.y
        outer = Rect(pos.x, pos.y, width, height + margin[1] + margin[3])
        frame = Rect(
            pos.x + margin[0], pos.y + margin[1], max(0, width - margin[0] - margin[2]), height
        )
        box = Box(outer, frame, frame.inset(*padding), padding, margin)
        content = box.content
        indent = min(
            depth * u.metric("tree_row", "indent", 14) * s, max(0, content.width - 110 * s)
        )
        hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
            (frame.x, frame.y), (frame.right, frame.bottom)
        )
        if selected or hovered:
            u.rect(
                *frame.tuple(),
                rgba8("primary", 30) if selected else (255, 255, 255, 12),
                r=u.metric("tree_row", "radius", DESIGN.control_radius),
            )
        if not leaf:
            opened = self.hierarchy._open_state.get(
                node.node_id, depth < self.hierarchy._default_open_depth
            )
            ew, eh = control_frame_size(
                u, "expand-" + str(node.node_id), *expander_size, icon="right"
            )
            if u.button(
                "expand-" + str(node.node_id),
                content.x + indent,
                content.y + (content.height - expander_size[1]) / 2,
                ew,
                eh,
                icon="down" if opened else "right",
                icon_role="inline",
                background=False,
            ):
                self.hierarchy._open_state[node.node_id] = not opened
        label_x = content.x + indent + expander_size[0] + gap
        visibility_right = content.right - min(4 * s, max(0, content.width - visibility_size[0]))
        label_width = max(0, visibility_right - visibility_size[0] - gap - label_x)
        lw, lh = control_frame_size(
            u,
            "entity-" + str(node.node_id),
            label_width,
            label_size[1],
            label=node.name or "?",
            icon=node_icon(node, ctx.session),
        )
        pressed = u.button(
            "entity-" + str(node.node_id),
            label_x,
            content.y + (content.height - label_size[1]) / 2,
            lw,
            lh,
            icon=node_icon(node, ctx.session),
            icon_role="inline",
            label=node.name or "?",
            text_color=node_text_color(ctx.theme, node.type.value, node.visible),
            icon_color=node_accent(ctx.theme, node.type.value),
            background=False,
            translate=False,
            align="left",
            width_limit=lw * s,
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
        vw, vh = control_frame_size(u, "visible-" + str(node.node_id), *visibility_size, icon="eye")
        visibility_rect = Rect(
            visibility_right - visibility_size[0],
            content.y + (content.height - visibility_size[1]) / 2,
            *visibility_size,
        )
        # Only the independently painted affordance needs an inset; the tree's
        # expander and name keep their existing hierarchy and column origins.
        visibility_inset = min(
            frame.right - visibility_rect.right,
            visibility_rect.y - frame.y,
            frame.bottom - visibility_rect.bottom,
        )
        visibility_radius = max(
            0, u.metric("tree_row", "radius", DESIGN.control_radius) - visibility_inset / s
        )
        visibility_style = u.component_style("icon_button")
        if visibility_style.explicit("radius"):
            visibility_radius = visibility_style.get("radius", visibility_radius)
        visibility_id = imgui.get_id("visible-" + str(node.node_id))
        context = imgui.get_current_context()
        show_visibility = (
            hovered
            or not node.visible
            or visibility_id == context.active_id
            or (imgui.get_io().nav_visible and visibility_id == context.nav_id)
        )
        if (
            show_visibility
            and node.type not in (NodeType.MODEL, NodeType.ENVIRONMENT)
            and u.button(
                "visible-" + str(node.node_id),
                visibility_rect.x,
                visibility_rect.y,
                vw,
                vh,
                icon="eye" if node.visible else "eyeoff",
                icon_role="inline",
                tooltip=ctx.tr("hide" if node.visible else "show"),
                background=True,
                radius=visibility_radius,
            )
        ):
            if node.type is NodeType.LIGHT and node.light_index >= 0:
                light = ctx.session.source.lights.lights[node.light_index]
                ctx.submit(cmd.SetLight(node.light_index, replace(light, active=not light.active)))
            else:
                ctx.submit(cmd.SetVisible(node.node_id, not node.visible))
        imgui.set_cursor_screen_pos((origin_x, origin_y))
        imgui.dummy(
            (outer.width, max(0, outer.height + row_gap * s - imgui.get_style().item_spacing.y))
        )
        if u.design is not None:
            next_y = imgui.get_cursor_screen_pos().y
            u.design.record(
                "tree-row-" + str(node.node_id),
                ("tree_row",),
                frame.tuple(),
                "tree_row",
                outer=list(outer.tuple()),
                content=list(box.content.tuple()),
                padding=list(padding),
                margin=list(margin),
                background=selected or hovered,
                selected=selected,
                hovered=hovered,
                depth=depth,
                visibilityRect=list(visibility_rect.tuple()),
                visibilityRadius=visibility_radius * s,
                visibilityInset=visibility_inset,
                layoutRect=[origin_x, origin_y, outer.width, next_y - origin_y],
                spacingRect=[outer.x, outer.bottom, outer.width, max(0, next_y - outer.bottom)],
                fieldRegions={
                    name: [
                        [content.x + indent + expander_size[0], content.y, gap, content.height],
                        [
                            visibility_right - visibility_size[0] - gap,
                            content.y,
                            gap,
                            content.height,
                        ],
                    ]
                    for name in ("gap", "columnGap")
                },
            )

    def _scene_context(self, ctx, node):
        with self.workspace.menus.local_popup(
            "entity-context-" + str(node.node_id), context_item=True
        ) as opened:
            if not opened:
                return
            clicked, _ = self.workspace.menus.local_item(
                "Focus item", enabled=ctx.model_camera_id < 0, key="scene-menu-focus", icon="fit"
            )
            if clicked and ctx.focus_node is not None:
                ctx.focus_node(node.node_id)
            clicked, _ = self.workspace.menus.local_item(
                "Copy name", key="scene-menu-copy", icon="copy"
            )
            if clicked:
                imgui.set_clipboard_text(node.name)
            if (node.type is NodeType.MODEL and node.model_id >= 0) or node.source_editable:
                self.workspace.menus.local_separator()
                self._batch_actions(ctx, node)
                if node.type in (NodeType.MODEL, NodeType.ROBOT, NodeType.LINK):
                    self.hierarchy._model_create_menu(ctx, node, presentation=self.workspace.menus)
                if node.type is NodeType.MODEL:
                    clicked, _ = self.workspace.menus.local_item(
                        "Remove Model",
                        enabled=ctx.session.paused,
                        key="scene-menu-remove-model",
                        icon="delete",
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
                self.workspace.menus.local_separator()
                self._entity_actions(ctx, node, model=False)

    def _batch_actions(self, ctx, node):
        selected = self.hierarchy._batch_selected
        if len(selected) < 2 or node.node_id not in selected:
            return
        roots = self.hierarchy._batch_removable_roots()
        if not roots:
            return
        clicked, _ = self.workspace.menus.local_item(
            f"{ctx.tr('Delete')} {len(roots)} {ctx.tr('model elements')}",
            enabled=ctx.session.paused and ctx.session.adapter.caps.topology_editing,
            key="scene-menu-delete-selected",
            icon="delete",
        )
        if clicked:
            result = ctx.submit(
                cmd.ModelEditBatch(
                    tuple(
                        cmd.RemoveModelElementEdit(cmd.ModelElementRef(node_id=node_id))
                        for node_id in roots
                    )
                )
            )
            if result.ok:
                selected.clear()

    def _entity_actions(self, ctx, node, *, model):
        enabled = not ctx.session.adapter.caps.simulation or ctx.session.paused
        duplicate, _ = self.workspace.menus.local_item(
            "Duplicate", "Cmd/Ctrl+D", enabled=enabled, key="scene-menu-duplicate", icon="duplicate"
        )
        rename, _ = self.workspace.menus.local_item(
            "Rename", "F2", enabled=enabled, key="scene-menu-rename", icon="keyboard"
        )
        remove, _ = self.workspace.menus.local_item(
            "Delete from Model" if model else "Delete",
            "Delete",
            enabled=enabled,
            key="scene-menu-delete",
            icon="delete",
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
        session = ctx.session
        if node_id is not None:
            node = session.node(node_id)
            if node is None:
                return ()
            if node.type is NodeType.JOINT:
                joint = session.joint(node.joint_index)
                return (joint,) if joint is not None else ()
            return sort_joints(
                session.joints_for_body(node.body_index), by_name=self.joints._sort_by_name
            )
        key = (
            session.document_id,
            session.structure_generation,
            self.joints._search.strip().casefold(),
            self.joints._sort_by_name,
        )
        if key != self._pose_browse_key or self._pose_browse_source is not session.joints:
            self._pose_browse_key = key
            self._pose_browse_source = session.joints
            self._pose_browse_joints = sort_joints(
                filter_joints(session.joints, self.joints._search),
                by_name=self.joints._sort_by_name,
            )
        return self._pose_browse_joints

    def draw_pose(self, ctx, x, y, w, *, node_id=None):
        if node_id is None:
            y = self._pose_controls(ctx, x, y, w)
        joints = self.pose_joints(ctx, node_id)
        inspected = ctx.session.node(node_id) if node_id is not None else None
        show_identity = inspected is None or inspected.type is not NodeType.JOINT
        return self.pose_editor.draw(
            ctx,
            x,
            y,
            w,
            joints=joints,
            select_joint=self.select_joint,
            initial_qpos=self.joints._initial_qpos,
            show_names=show_identity or len(joints) > 1,
            show_types=inspected is not None and inspected.type is not NodeType.JOINT,
            show_context=node_id is None,
            joint_color=node_accent(ctx.theme, "joint"),
            collapsible_poses=node_id is None,
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

    @part_scope("panel_tools", "pose_tools")
    def _pose_controls(self, ctx, x, y, w):
        if self._pose_copy_pending:
            imgui.open_popup("Joint state")
            self._pose_copy_pending = False
        with self.workspace.menus.local_popup("Joint state") as opened:
            if opened:
                copy_qpos, _ = self.workspace.menus.local_item(
                    "Copy qpos",
                    enabled=ctx.session.frame.qpos is not None,
                    key="pose-menu-copy-qpos",
                    icon="copy",
                )
                copy_qvel, _ = self.workspace.menus.local_item(
                    "Copy qvel",
                    enabled=ctx.session.frame.qvel is not None
                    or ctx.session.adapter.caps.state_snapshots,
                    key="pose-menu-copy-qvel",
                    icon="copy",
                )
                if copy_qpos:
                    copy_state_vector(ctx.session.frame.qpos)
                if copy_qvel:
                    state = (
                        ctx.session.adapter.capture_state()
                        if ctx.session.frame.qvel is None
                        else None
                    )
                    copy_state_vector(ctx.session.frame.qvel if state is None else state.qvel)
        if self.pose_search_visible:
            ui, scale = self.workspace.ui, self.workspace.s
            tools = ui.component_style(("panel_tools", "pose_tools"))
            top = y
            _, self.joints._search = ui.search(
                "##pose-search", x, y, w, self.joints._search, hint=ctx.tr("Search joints")
            )
            gap = tools.get("rowGap", tools.get("gap", 10)) * scale
            y = imgui.get_item_rect_max().y + gap
            if ui.design is not None:
                ui.design.record(
                    "pose-panel-tools",
                    ("panel_tools", "pose_tools"),
                    (x, top, w, y - top),
                    "pose_tools",
                    fields=[list(ui.hits["##pose-search"])],
                    fieldRegions={"rowGap": [x, y - gap, w, gap]},
                )
        return y
