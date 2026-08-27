"""Scene hierarchy browsing, filtering, and visibility."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, NodeType, SceneNode
from ..draw2d import ImguiDraw2D, ink_box
from . import Panel, PanelContext

_LARGE_SCENE_NODES = 2_000
_VISIBLE_ROW_BUDGET = 512


class HierarchyPanel(Panel):
    name = "Hierarchy"
    default_open = True
    shortcut = "F3"

    def __init__(self) -> None:
        super().__init__()
        self._filter = ""
        self._cache_generation = -1
        self._roots: list[SceneNode] = []
        self._by_id: dict[int, SceneNode] = {}
        self._search_names: tuple[str, ...] = ()
        self._default_open_depth = 2
        self._row_budget = _VISIBLE_ROW_BUDGET
        self._rows_drawn = 0
        self._rows_truncated = False
        self._ink_center_cache: dict[tuple[float, str], float] = {}
        self._batch_selected: set[int] = set()

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        self._refresh(ctx)

        imgui.set_next_item_width(-1)
        _changed, self._filter = imgui.input_text("##filter", self._filter)
        if not self._filter:
            imgui.set_item_tooltip("filter by name")

        removable = self._batch_removable_roots()
        if len(self._batch_selected) > 1:
            imgui.text_disabled(f"{len(self._batch_selected)} selected (Ctrl/Cmd+click)")
            if removable:
                imgui.same_line()
                if not ctx.session.paused:
                    imgui.begin_disabled()
                if imgui.small_button(f"Delete {len(removable)} model element(s)"):
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
                    imgui.set_item_tooltip("Pause the simulation before editing model topology")

        imgui.separator()
        if not imgui.begin_child("tree"):
            imgui.end_child()
            return

        table_flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("hierarchy_rows", 2, table_flags):
            imgui.end_child()
            return
        imgui.table_setup_column("node", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column(
            "visible", imgui.TableColumnFlags_.width_fixed, 24.0 * ctx.style_scale
        )

        self._rows_drawn = 0
        self._rows_truncated = False
        if self._filter:
            needle = self._filter.casefold()
            hits = []
            for node, name in zip(s.nodes, self._search_names, strict=True):
                if needle in name:
                    if len(hits) >= self._row_budget:
                        self._rows_truncated = True
                        break
                    hits.append(node)
            for node in hits:
                self._row(ctx, node, leaf=True)
            if not hits:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled("no match")
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
                f"showing the first {self._row_budget} visible nodes; use the filter to narrow"
            )

        imgui.end_table()
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
        self._ink_center_cache.clear()

    def _subtree(self, ctx: PanelContext, node: SceneNode, depth: int) -> None:
        if self._rows_drawn >= self._row_budget:
            self._rows_truncated = True
            return
        children = [self._by_id[c] for c in node.children if c in self._by_id]
        opened = self._row(
            ctx, node, leaf=not children, default_open=depth < self._default_open_depth
        )
        if children and opened:
            for child in children:
                if self._rows_drawn >= self._row_budget:
                    self._rows_truncated = True
                    break
                self._subtree(ctx, child, depth + 1)
            imgui.tree_pop()

    def _row(
        self, ctx: PanelContext, node: SceneNode, leaf: bool, default_open: bool = False
    ) -> bool:
        self._rows_drawn += 1
        imgui.table_next_row()
        imgui.table_next_column()
        flags = imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.span_avail_width
        if leaf:
            flags |= imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open
        if default_open:
            flags |= imgui.TreeNodeFlags_.default_open
        selected = ctx.session.selected_node
        if node.node_id in self._batch_selected or (
            selected is not None and node.node_id == selected.node_id
        ):
            flags |= imgui.TreeNodeFlags_.selected

        color = ctx.theme.node_color(node.type)
        if not node.visible:
            color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1.0)

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
        opened = imgui.tree_node_ex(f"{node.name or '?'}##n{node.node_id}", flags)
        imgui.pop_style_color()

        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
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
                remove, _ = imgui.menu_item("Remove Model", "", False)
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
                remove, _ = imgui.menu_item("Delete from Model", "", False, removable)
                if remove:
                    ctx.submit(cmd.RemoveModelElement(node.node_id))
            elif editable:
                ctx.submit(cmd.SelectNode(node.node_id))
                duplicate, _ = imgui.menu_item("Duplicate", "Cmd/Ctrl+D", False)
                rename, _ = imgui.menu_item("Rename", "F2", False)
                remove, _ = imgui.menu_item("Delete", "Delete", False)
                if duplicate:
                    ctx.submit(cmd.DuplicateSceneEntity(node.object_id))
                if rename and ctx.request_rename is not None:
                    ctx.request_rename(node.object_id)
                if remove:
                    ctx.submit(cmd.RemoveSceneEntity(node.object_id))
            else:
                imgui.text_disabled("Read-only entity")
            imgui.end_popup()

        imgui.same_line()
        imgui.text_disabled(str(node.type))

        imgui.table_next_column()
        self._visibility_toggle(ctx, node)
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
        if not imgui.begin_menu("Add Child", ctx.session.adapter.caps.topology_editing):
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
            clicked, _ = imgui.menu_item(label, "", False)
            if clicked:
                base = element_type.split(":", 1)[0]
                index = 1
                name = base
                while name in names:
                    index += 1
                    name = f"{base}{index}"
                ctx.submit(cmd.AddModelElement(node.node_id, element_type, name))
        imgui.end_menu()

    def _visibility_toggle(self, ctx: PanelContext, node: SceneNode) -> None:
        if node.type in (NodeType.ENVIRONMENT, NodeType.MODEL):
            return
        size = imgui.get_frame_height()
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, (avail - size) * 0.5))
        pressed = imgui.invisible_button(f"##vis{node.node_id}", imgui.ImVec2(size, size))
        hovered = imgui.is_item_hovered()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        # The frame is taller than the label's ink box, so centering on the
        # frame leaves the mark a few pixels low; center on the row's ink.
        label = f"{node.name or '?'} {node.type}"
        font = imgui.get_font()
        font_size = imgui.get_font_size()
        cache_key = (round(font_size, 3), label)
        center_offset = self._ink_center_cache.get(cache_key)
        if center_offset is None:
            box = ink_box(font, font_size, label)
            center_offset = (box[1] + box[3]) * 0.5 if box else (hi.y - lo.y) * 0.5
            self._ink_center_cache[cache_key] = center_offset
        center_y = lo.y + center_offset
        center = ((lo.x + hi.x) * 0.5, center_y)
        radius = 4.0 * ctx.style_scale
        base = ctx.theme.primary_bright if hovered else ctx.theme.primary
        alpha = 1.0 if node.visible else 0.45
        color = (*base[:3], alpha)
        draw = ImguiDraw2D()
        draw.circle(center, radius, color, 1.4 * ctx.style_scale, segments=16)
        if node.visible:
            draw.circle_filled(center, radius * 0.42, color, segments=12)
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
        imgui.set_item_tooltip("hide" if node.visible else "show")


def hierarchy_open_depth(node_count: int) -> int:
    """Keep the first frame bounded while preserving small-scene expansion."""
    if node_count >= _LARGE_SCENE_NODES:
        return 0
    if node_count >= 1_000:
        return 1
    return 2
