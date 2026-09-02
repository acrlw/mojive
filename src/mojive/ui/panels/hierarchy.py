"""Scene hierarchy browsing, filtering, and visibility."""

from __future__ import annotations

import math
from dataclasses import replace

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, NodeType, SceneNode
from ..draw2d import ImguiDraw2D
from . import Panel, PanelContext, horizontal_wheel_scroll, search_input

_LARGE_SCENE_NODES = 2_000
_VISIBLE_ROW_BUDGET = 512
_TYPE_FILTERS = ("all", "link", "geom", "joint", "site", "camera", "light", "robot", "flex")
_TYPE_COLUMN_WIDTH_PT = 72.0
_VISIBILITY_COLUMN_WIDTH_PT = 24.0
_MIN_NODE_COLUMN_WIDTH_PT = 144.0


def hierarchy_shows_type_column(available_width: float, style_scale: float) -> bool:
    """Keep the node cell wide enough for disclosure and indented names."""

    fixed_width = _TYPE_COLUMN_WIDTH_PT + _VISIBILITY_COLUMN_WIDTH_PT
    return available_width >= (fixed_width + _MIN_NODE_COLUMN_WIDTH_PT) * style_scale


def disclosure_triangle(
    center: tuple[float, float],
    radius: float,
    *,
    opened: bool,
) -> tuple[tuple[float, float], ...]:
    """Return one canonical right triangle, rigidly rotated when expanded."""

    cx, cy = center
    canonical = (
        (-0.55 * radius, -radius),
        (-0.55 * radius, radius),
        (0.65 * radius, 0.0),
    )
    if not opened:
        return tuple((cx + x, cy + y) for x, y in canonical)
    # Screen Y grows downward, so a positive quarter turn maps right to down.
    return tuple((cx - y, cy + x) for x, y in canonical)


class HierarchyPanel(Panel):
    name = "Hierarchy"
    default_open = True
    shortcut = "F3"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._filter = ""
        self._type_filter = "all"
        self._cache_generation = -1
        self._roots: list[SceneNode] = []
        self._by_id: dict[int, SceneNode] = {}
        self._search_names: tuple[str, ...] = ()
        self._default_open_depth = 2
        self._row_budget = _VISIBLE_ROW_BUDGET
        self._rows_drawn = 0
        self._rows_truncated = False
        self._batch_selected: set[int] = set()
        self._open_state: dict[int, bool] = {}
        self._show_type_column = True

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def clear_selection(self) -> None:
        """Clear the hierarchy's optional multi-selection state."""

        self._batch_selected.clear()

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        self._refresh(ctx)

        imgui.set_next_item_width(-1)
        _changed, self._filter = search_input(
            "##filter",
            self._filter,
            hint=ctx.tr("Search hierarchy"),
            search_tooltip=ctx.tr("Search hierarchy"),
            clear_tooltip=ctx.tr("Clear search"),
        )

        self._draw_type_filters(ctx)

        removable = self._batch_removable_roots()
        if len(self._batch_selected) > 1:
            imgui.text_disabled(
                f"{len(self._batch_selected)} {ctx.tr('selected')} (Ctrl/Cmd+click)"
            )
            if removable:
                imgui.same_line()
                if not ctx.session.paused:
                    imgui.begin_disabled()
                if imgui.small_button(
                    f"{ctx.tr('Delete')} {len(removable)} {ctx.tr('model elements')}"
                ):
                    result = ctx.submit(
                        cmd.ModelEditBatch(
                            tuple(
                                cmd.RemoveModelElementEdit(cmd.ModelElementRef(node_id=node_id))
                                for node_id in removable
                            )
                        )
                    )
                    if result.ok:
                        self._batch_selected.clear()
                if not ctx.session.paused:
                    imgui.end_disabled()
                    imgui.set_item_tooltip(
                        ctx.tr("Pause the simulation before editing model topology")
                    )

        imgui.separator()
        if not imgui.begin_child("tree"):
            imgui.end_child()
            return

        table_flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        available_width = float(imgui.get_content_region_avail().x)
        self._show_type_column = hierarchy_shows_type_column(
            available_width,
            ctx.style_scale,
        )
        column_count = 3 if self._show_type_column else 2
        if not imgui.begin_table("hierarchy_rows", column_count, table_flags):
            imgui.end_child()
            return
        imgui.table_setup_column("node", imgui.TableColumnFlags_.width_stretch, 1.0)
        if self._show_type_column:
            imgui.table_setup_column(
                "type",
                imgui.TableColumnFlags_.width_fixed,
                _TYPE_COLUMN_WIDTH_PT * ctx.style_scale,
            )
        imgui.table_setup_column(
            "visible",
            imgui.TableColumnFlags_.width_fixed,
            _VISIBILITY_COLUMN_WIDTH_PT * ctx.style_scale,
        )

        self._rows_drawn = 0
        self._rows_truncated = False
        if self._filter or self._type_filter != "all":
            needle = self._filter.casefold()
            hits = []
            for node, name in zip(s.nodes, self._search_names, strict=True):
                type_matches = self._type_filter == "all" or node.type.value == self._type_filter
                if type_matches and needle in name:
                    if len(hits) >= self._row_budget:
                        self._rows_truncated = True
                        break
                    hits.append(node)
            clipper = imgui.ListClipper()
            clipper.begin(len(hits))
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    self._row(ctx, hits[index], leaf=True, depth=0)
            clipper.end()
            if not hits:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(ctx.tr("no match"))
        else:
            for root in self._roots:
                if self._rows_drawn >= self._row_budget:
                    self._rows_truncated = True
                    break
                self._subtree(ctx, root, depth=0)

        if self._rows_truncated:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(
                f"{ctx.tr('showing the first')} {self._row_budget} "
                f"{ctx.tr('visible nodes; use the filter to narrow')}"
            )

        imgui.end_table()
        imgui.end_child()

    def _draw_type_filters(self, ctx: PanelContext) -> None:
        style = imgui.get_style()
        spacing = float(style.item_spacing.x)
        padding = float(style.frame_padding.x)
        display_labels = _TYPE_FILTERS
        total_width = sum(imgui.calc_text_size(label).x + padding * 2.0 for label in display_labels)
        total_width += spacing * (len(_TYPE_FILTERS) - 1)
        height = imgui.get_frame_height() + imgui.get_style().scrollbar_size + 3.0 * ctx.style_scale
        imgui.set_next_window_content_size(imgui.ImVec2(total_width, 0.0))
        child_flags = imgui.ChildFlags_.none.value
        window_flags = imgui.WindowFlags_.horizontal_scrollbar.value
        if not imgui.begin_child(
            "hierarchy_type_filters",
            imgui.ImVec2(0.0, height),
            child_flags,
            window_flags,
        ):
            imgui.end_child()
            return
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 3.0 * ctx.style_scale)
        for index, (label, display_label) in enumerate(
            zip(_TYPE_FILTERS, display_labels, strict=True)
        ):
            selected = label == self._type_filter
            imgui.push_style_color(
                imgui.Col_.button,
                imgui.ImVec4(*(ctx.theme.bg_frame_active if selected else ctx.theme.bg_frame)),
            )
            imgui.push_style_color(
                imgui.Col_.button_hovered,
                imgui.ImVec4(*ctx.theme.bg_frame_hovered),
            )
            imgui.push_style_color(
                imgui.Col_.button_active,
                imgui.ImVec4(*ctx.theme.bg_frame_active),
            )
            imgui.push_style_color(
                imgui.Col_.text,
                imgui.ImVec4(*(ctx.theme.primary_bright if selected else ctx.theme.text_disabled)),
            )
            if imgui.button(f"{display_label}##hierarchy-type-{label}"):
                self._type_filter = label
            imgui.pop_style_color(4)
            if index + 1 < len(_TYPE_FILTERS):
                imgui.same_line()
        imgui.pop_style_var()
        horizontal_wheel_scroll(step=56.0 * ctx.style_scale)
        imgui.end_child()

    def _refresh(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        if gen == self._cache_generation:
            return
        self._cache_generation = gen
        nodes = ctx.session.nodes
        self._by_id = {n.node_id: n for n in nodes}
        self._batch_selected.intersection_update(self._by_id)
        self._roots = [n for n in nodes if n.parent < 0 or n.parent not in self._by_id]
        self._search_names = tuple(node.name.casefold() for node in nodes)
        self._default_open_depth = hierarchy_open_depth(len(nodes))
        self._row_budget = (
            _VISIBLE_ROW_BUDGET if len(nodes) >= _LARGE_SCENE_NODES else len(nodes) + 1
        )
        self._open_state = {
            node_id: value for node_id, value in self._open_state.items() if node_id in self._by_id
        }

    def _subtree(self, ctx: PanelContext, node: SceneNode, depth: int) -> None:
        if self._rows_drawn >= self._row_budget:
            self._rows_truncated = True
            return
        children = [self._by_id[c] for c in node.children if c in self._by_id]
        opened = self._row(
            ctx,
            node,
            leaf=not children,
            depth=depth,
            default_open=depth < self._default_open_depth,
        )
        if children and opened:
            for child in children:
                if self._rows_drawn >= self._row_budget:
                    self._rows_truncated = True
                    break
                self._subtree(ctx, child, depth + 1)

    def _row(
        self,
        ctx: PanelContext,
        node: SceneNode,
        leaf: bool,
        depth: int,
        default_open: bool = False,
    ) -> bool:
        self._rows_drawn += 1
        row_height = max(imgui.get_frame_height(), 26.0 * ctx.style_scale)
        imgui.table_next_row(0, row_height)
        imgui.table_next_column()
        selected = ctx.session.selected_node
        is_selected = node.node_id in self._batch_selected or (
            selected is not None and node.node_id == selected.node_id
        )
        row_start = imgui.get_cursor_screen_pos()
        width = imgui.get_content_region_avail().x
        imgui.invisible_button(
            f"##hierarchy-node-{node.node_id}",
            imgui.ImVec2(max(1.0, width), row_height),
        )
        hovered = imgui.is_item_hovered()
        draw = ImguiDraw2D()
        if is_selected or hovered:
            color = ctx.theme.bg_header if is_selected else ctx.theme.bg_frame
            imgui.table_set_bg_color(
                imgui.TableBgTarget_.row_bg0,
                imgui.color_convert_float4_to_u32(imgui.ImVec4(*color)),
            )
        indent = depth * 18.0 * ctx.style_scale
        arrow_center = (
            row_start.x + indent + 7.0 * ctx.style_scale,
            row_start.y + row_height * 0.5,
        )
        opened = self._open_state.get(node.node_id, default_open)
        if not leaf:
            radius = 5.0 * ctx.style_scale
            draw.fringed_concave_fill(
                disclosure_triangle(arrow_center, radius, opened=opened),
                ctx.theme.text,
            )
        name = node.name or "?"
        text_height = imgui.calc_text_size(name).y
        text_color = ctx.theme.text if node.visible else ctx.theme.text_disabled
        draw.text(
            (
                row_start.x + indent + 20.0 * ctx.style_scale,
                row_start.y + (row_height - text_height) * 0.5,
            ),
            text_color,
            name,
        )
        if imgui.is_item_clicked():
            mouse_x = imgui.get_io().mouse_pos.x
            if not leaf and mouse_x <= row_start.x + indent + 16.0 * ctx.style_scale:
                opened = not opened
                self._open_state[node.node_id] = opened
            else:
                io = imgui.get_io()
                if io.key_ctrl or io.key_super:
                    if node.node_id in self._batch_selected:
                        self._batch_selected.remove(node.node_id)
                        if self._batch_selected:
                            ctx.submit(cmd.SelectNode(next(reversed(tuple(self._batch_selected)))))
                        else:
                            ctx.submit(cmd.Select(0))
                    else:
                        self._batch_selected.add(node.node_id)
                        ctx.submit(cmd.SelectNode(node.node_id))
                else:
                    self._batch_selected = {node.node_id}
                    ctx.submit(cmd.SelectNode(node.node_id))

        editable = bool(
            ctx.session.adapter.caps.scene_authoring
            and node.object_id
            and node.model_id < 0
            and node.type in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
        )
        if imgui.begin_popup_context_item(f"##entity_context_{node.node_id}"):
            if node.type is NodeType.MODEL and node.model_id >= 0:
                self._model_create_menu(ctx, node)
                remove, _ = imgui.menu_item(ctx.tr("Remove Model"), "", False)
                if remove:
                    ctx.submit(cmd.RemoveSceneModel(node.model_id))
            elif node.model_id >= 0:
                if node.type in (NodeType.ROBOT, NodeType.LINK):
                    self._model_create_menu(ctx, node)
                    imgui.separator()
                removable = node.type in (
                    NodeType.ROBOT,
                    NodeType.LINK,
                    NodeType.GEOM,
                    NodeType.JOINT,
                    NodeType.SITE,
                    NodeType.CAMERA,
                    NodeType.LIGHT,
                )
                duplicate, _ = imgui.menu_item(ctx.tr("Duplicate"), "Cmd/Ctrl+D", False, removable)
                rename, _ = imgui.menu_item(ctx.tr("Rename"), "F2", False, removable)
                remove, _ = imgui.menu_item(ctx.tr("Delete from Model"), "", False, removable)
                if duplicate:
                    result = ctx.submit(cmd.DuplicateModelElement(node.node_id))
                    if result.ok:
                        ctx.submit(cmd.SelectNode(result.entity_id))
                if rename and ctx.request_model_rename is not None:
                    ctx.request_model_rename(node.node_id)
                if remove:
                    ctx.submit(cmd.RemoveModelElement(node.node_id))
            elif editable:
                ctx.submit(cmd.SelectNode(node.node_id))
                duplicate, _ = imgui.menu_item(ctx.tr("Duplicate"), "Cmd/Ctrl+D", False)
                rename, _ = imgui.menu_item(ctx.tr("Rename"), "F2", False)
                remove, _ = imgui.menu_item(ctx.tr("Delete"), "Delete", False)
                if duplicate:
                    ctx.submit(cmd.DuplicateSceneEntity(node.object_id))
                if rename and ctx.request_rename is not None:
                    ctx.request_rename(node.object_id)
                if remove:
                    ctx.submit(cmd.RemoveSceneEntity(node.object_id))
            else:
                imgui.text_disabled(ctx.tr("Read-only entity"))
            imgui.end_popup()

        if self._show_type_column:
            imgui.table_next_column()
            type_pos = imgui.get_cursor_screen_pos()
            type_label = str(node.type)
            type_height = imgui.calc_text_size(type_label).y
            imgui.set_cursor_screen_pos(
                imgui.ImVec2(type_pos.x, row_start.y + (row_height - type_height) * 0.5)
            )
            imgui.text_disabled(type_label)

        imgui.table_next_column()
        self._visibility_toggle(ctx, node, row_start.y, row_height)
        return opened

    def _batch_removable_roots(self) -> tuple[int, ...]:
        removable_types = {
            NodeType.ROBOT,
            NodeType.LINK,
            NodeType.GEOM,
            NodeType.JOINT,
            NodeType.SITE,
            NodeType.CAMERA,
            NodeType.LIGHT,
        }
        eligible = {
            node_id
            for node_id in self._batch_selected
            if (node := self._by_id.get(node_id)) is not None
            and node.model_id >= 0
            and node.type in removable_types
        }
        roots = []
        for node_id in sorted(eligible):
            parent = self._by_id[node_id].parent
            while parent in self._by_id and parent not in eligible:
                parent = self._by_id[parent].parent
            if parent not in eligible:
                roots.append(node_id)
        return tuple(roots)

    @staticmethod
    def _model_create_menu(ctx: PanelContext, node: SceneNode) -> None:
        if not imgui.begin_menu(ctx.tr("Add Child"), ctx.session.adapter.caps.topology_editing):
            return
        entries = [
            ("Body", "body"),
            ("Box Geometry", "geom:box"),
            ("Sphere Geometry", "geom:sphere"),
            ("Capsule Geometry", "geom:capsule"),
            ("Cylinder Geometry", "geom:cylinder"),
            ("Plane Geometry", "geom:plane"),
            ("Hinge Joint", "joint:hinge"),
            ("Slide Joint", "joint:slide"),
            ("Ball Joint", "joint:ball"),
            ("Free Joint", "joint:free"),
            ("Site", "site"),
            ("Camera", "camera"),
            ("Light", "light"),
        ]
        if node.type is NodeType.MODEL:
            entries = [entry for entry in entries if not entry[1].startswith("joint:")]
        names = {item.name for item in ctx.session.nodes}
        for label, element_type in entries:
            clicked, _ = imgui.menu_item(ctx.tr(label), "", False)
            if clicked:
                base = element_type.split(":", 1)[0]
                index = 1
                name = base
                while name in names:
                    index += 1
                    name = f"{base}{index}"
                ctx.submit(cmd.AddModelElement(node.node_id, element_type, name))
        imgui.end_menu()

    def _visibility_toggle(
        self,
        ctx: PanelContext,
        node: SceneNode,
        row_y: float,
        row_height: float,
    ) -> None:
        if node.type in (NodeType.ENVIRONMENT, NodeType.MODEL):
            return
        size = imgui.get_frame_height()
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, (avail - size) * 0.5))
        pressed = imgui.invisible_button(f"##vis{node.node_id}", imgui.ImVec2(size, size))
        hovered = imgui.is_item_hovered()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        center = ((lo.x + hi.x) * 0.5, row_y + row_height * 0.5)
        radius_x = 6.5 * ctx.style_scale
        radius_y = 3.6 * ctx.style_scale
        base = (
            ctx.theme.primary_bright
            if hovered
            else ctx.theme.primary
            if node.visible
            else ctx.theme.text_disabled
        )
        alpha = 1.0 if hovered else 0.58
        color = (*base[:3], alpha)
        draw = ImguiDraw2D()
        top = tuple(
            (
                center[0] - radius_x + radius_x * 2.0 * index / 8.0,
                center[1] - math.sin(math.pi * index / 8.0) * radius_y,
            )
            for index in range(9)
        )
        bottom = tuple(
            (
                center[0] + radius_x - radius_x * 2.0 * index / 8.0,
                center[1] + math.sin(math.pi * index / 8.0) * radius_y,
            )
            for index in range(9)
        )
        if node.visible:
            # Do not submit the shared left/right endpoints twice: overlapping
            # antialiased strokes made the old almond look as if its arcs crossed.
            outline = (*top, *bottom[1:-1])
            draw.polyline(outline, color, 1.35 * ctx.style_scale, closed=True)
            draw.circle(center, 1.8 * ctx.style_scale, color, 1.2 * ctx.style_scale, segments=16)
        else:
            lid = tuple(
                (
                    center[0] - radius_x + radius_x * 2.0 * index / 8.0,
                    center[1] + math.sin(math.pi * index / 8.0) * radius_y * 0.72,
                )
                for index in range(9)
            )
            draw.polyline(lid, color, 1.45 * ctx.style_scale)
            for offset in (-0.52, 0.0, 0.52):
                lash_x = center[0] + radius_x * offset
                lash_y = center[1] + radius_y * 0.72 * math.sqrt(max(0.0, 1.0 - offset**2))
                draw.line(
                    (lash_x, lash_y),
                    (
                        lash_x + offset * 1.6 * ctx.style_scale,
                        lash_y + 2.2 * ctx.style_scale,
                    ),
                    color,
                    1.15 * ctx.style_scale,
                )
        if pressed:
            if node.type is NodeType.LIGHT and node.light_index >= 0:
                source = ctx.session.source
                if source is not None and node.light_index < len(source.lights.lights):
                    light = source.lights.lights[node.light_index]
                    ctx.submit(
                        cmd.SetLight(node.light_index, replace(light, active=not light.active))
                    )
            else:
                ctx.submit(cmd.SetVisible(node.node_id, not node.visible))
        imgui.set_item_tooltip(ctx.tr("hide" if node.visible else "show"))


def hierarchy_open_depth(node_count: int) -> int:
    """Keep the first frame bounded while preserving small-scene expansion."""
    if node_count >= _LARGE_SCENE_NODES:
        return 0
    if node_count >= 1_000:
        return 1
    return 2
