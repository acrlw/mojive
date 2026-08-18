"""Scene hierarchy browsing, filtering, and visibility."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, NodeKind, SceneNode
from . import Panel, PanelContext


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

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        self._refresh(ctx)

        imgui.set_next_item_width(-1)
        _changed, self._filter = imgui.input_text("##filter", self._filter)
        if not self._filter:
            imgui.set_item_tooltip("filter by name")

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

        if self._filter:
            needle = self._filter.lower()
            hits = [n for n in s.nodes if needle in n.name.lower()]
            for node in hits:
                self._row(ctx, node, leaf=True)
            if not hits:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled("no match")
        else:
            for root in self._roots:
                self._subtree(ctx, root, depth=0)

        imgui.end_table()
        imgui.end_child()

    def _refresh(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        if gen == self._cache_generation:
            return
        self._cache_generation = gen
        nodes = ctx.session.nodes
        self._by_id = {n.node_id: n for n in nodes}
        self._roots = [n for n in nodes if n.parent < 0 or n.parent not in self._by_id]

    def _subtree(self, ctx: PanelContext, node: SceneNode, depth: int) -> None:
        children = [self._by_id[c] for c in node.children if c in self._by_id]
        opened = self._row(ctx, node, leaf=not children, default_open=depth < 2)
        if children and opened:
            for child in children:
                self._subtree(ctx, child, depth + 1)
            imgui.tree_pop()

    def _row(
        self, ctx: PanelContext, node: SceneNode, leaf: bool, default_open: bool = False
    ) -> bool:
        imgui.table_next_row()
        imgui.table_next_column()
        flags = imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.span_avail_width
        if leaf:
            flags |= imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open
        if default_open:
            flags |= imgui.TreeNodeFlags_.default_open
        selected = ctx.session.selected_node
        if selected is not None and node.node_id == selected.node_id:
            flags |= imgui.TreeNodeFlags_.selected

        color = ctx.theme.node_color(node.kind)
        if not node.visible:
            color = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1.0)

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
        opened = imgui.tree_node_ex(f"{node.name or '?'}##n{node.node_id}", flags)
        imgui.pop_style_color()

        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            ctx.submit(cmd.SelectNode(node.node_id))

        imgui.same_line()
        imgui.text_disabled(str(node.kind))

        imgui.table_next_column()
        self._visibility_toggle(ctx, node)
        return opened

    @staticmethod
    def _visibility_toggle(ctx: PanelContext, node: SceneNode) -> None:
        if node.kind is NodeKind.ENVIRONMENT:
            return
        size = imgui.get_frame_height()
        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, (avail - size) * 0.5))
        pressed = imgui.invisible_button(f"##vis{node.node_id}", imgui.ImVec2(size, size))
        hovered = imgui.is_item_hovered()
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        center = imgui.ImVec2((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5)
        radius = 4.0 * ctx.style_scale
        base = ctx.theme.primary_bright if hovered else ctx.theme.primary
        alpha = 1.0 if node.visible else 0.45
        color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*base[:3], alpha))
        dl = imgui.get_window_draw_list()
        dl.add_circle(center, radius, color, 16, 1.4 * ctx.style_scale)
        if node.visible:
            dl.add_circle_filled(center, radius * 0.42, color, 12)
        if pressed:
            if node.kind is NodeKind.LIGHT and node.light_index >= 0:
                source = ctx.session.source
                if source is not None and node.light_index < len(source.lights.lights):
                    light = source.lights.lights[node.light_index]
                    ctx.submit(
                        cmd.SetLight(node.light_index, replace(light, active=not light.active))
                    )
            else:
                ctx.submit(cmd.SetVisible(node.node_id, not node.visible))
        imgui.set_item_tooltip("hide" if node.visible else "show")
