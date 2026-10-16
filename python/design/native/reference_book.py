"""Browse design contracts and edit shared specifications in a native window."""

from imgui_bundle import imgui as g

from mojive.design.consumer_support import UNAVAILABLE_PARTS
from mojive.design.native.geometry import Rect
from mojive.design.native.reference_boards import (
    foundation_board,
    layout_board,
    title,
)
from mojive.design.native.workbench import NativeWorkbench
from mojive.design.reference_contracts import (
    BEHAVIORS,
    contract_index,
    reference_manifest,
    sample_variants,
)
from mojive.ui.design_palette import rgba

PAGES = ("视觉基础", "组件目录", "交互状态", "布局规则", "功能原型", "覆盖清单")
PROTOTYPES = (
    ("场景与层级", "tree_row"),
    ("姿态与关节", "joint"),
    ("相机", "range_number"),
    ("检查器与物理", "section"),
    ("材质与颜色", "color_input"),
    ("时间轴", "timeline_model_group"),
    ("输出", "output_toolbar"),
    ("统计", "statistics_list"),
    ("设置", "settings_page"),
    ("菜单", "menu_item"),
    ("相机预览", "camera_preview"),
    ("视口与 Gizmo", "toolbox"),
    ("模型源码", "textarea"),
    ("控制", "panel:Control"),
    ("资产与材质库", "panel:Assets"),
    ("传感器", "panel:Sensors"),
    ("图层", "panel:Layers"),
)


class ReferenceBook(NativeWorkbench):
    def __init__(self, editor, runtime, *, width=1440):
        super().__init__(editor, runtime, runtime, width=width)
        self.reference_page = PAGES[0]
        self.part = "switch"
        self.contracts = contract_index()
        self.manifest = reference_manifest(runtime.spec)
        categories = list(dict.fromkeys(item["category"] for item in self.editor.catalog.values()))
        self.catalogue_order = sorted(
            self.editor.catalog,
            key=lambda part: categories.index(self.editor.catalog[part]["category"]),
        )
        self.specimen_width = 420.0
        self.disabled = False
        self.show_bounds = False
        self.detail_tab = "示例与规则"
        self.prototype_request = None
        self.export_request = False
        self.export_message = ""
        self.page_rects = {}
        self.layout_page = "尺寸与层级"
        self.state_part = "switch"
        self.behavior_index = 0
        self.prototype_part = "tree_row"
        self.coverage_category = "全部"
        self.variants = {}
        self._last_reference_page = None
        self._last_part = None
        self.page_scroll_request = None
        self.page_scroll_max = 0
        self.page_height = 0
        self.runtime.observe_components = True

    def draw_contents(self, surface):
        ui, s = surface.ui, surface.s
        ui.hits.clear()
        self.runtime.begin_frame()
        ui.design.begin_frame()
        self._sample_records = []
        title(ui, "Mojive", "原生设计规范与交互原型")
        g.separator()
        height = max(1, g.get_content_region_avail().y)
        nav_width = min(198 * s, g.get_content_region_avail().x * 0.25)
        g.begin_child("Reference navigation", (nav_width, height))
        try:
            for page in PAGES:
                clicked, _ = g.selectable(page, self.reference_page == page, size=(0, 32 * s))
                self.record_item("page/" + page)
                if clicked:
                    self.reference_page = page
            g.spacing()
            g.separator()
            counts = self.manifest["counts"]
            g.text_unformatted(f"{counts['parts']} 类组件")
            g.text_unformatted(f"{counts['fields']} 项规格")
            g.text_unformatted(f"{len(BEHAVIORS)} 项跨组件规则")
            g.spacing()
            if self.button("保存规范", "save"):
                self.editor_action(self.editor.save)
            if self.button("撤销", "undo", enabled=bool(self.editor.undo_stack)):
                self.editor_action(self.editor.undo)
            self.next_button_on_row("重做")
            if self.button("重做", "redo", enabled=bool(self.editor.redo_stack)):
                self.editor_action(self.editor.redo)
            if self.button("导出设计规范", "export-reference"):
                self.export_request = True
            g.text_wrapped(self.export_message)
            if self.editor.error or self.runtime.error:
                g.text_colored(rgba("danger"), self.editor.error or self.runtime.error)
            else:
                g.text_colored(
                    rgba("primary"),
                    "已保存" if self.editor.spec == self.editor.saved_spec else "已修改",
                )
        finally:
            g.end_child()
        g.same_line()
        g.begin_child("Reference page", (0, height))
        try:
            if self._last_reference_page != self.reference_page:
                g.set_scroll_y(0)
                self._last_reference_page = self.reference_page
            if self.page_scroll_request is not None:
                g.set_scroll_y(self.page_scroll_request)
                self.page_scroll_request = None
            self.page_rects[self.reference_page] = (*g.get_window_pos(), *g.get_window_size())
            if self.reference_page == "视觉基础":
                foundation_board(ui, self.runtime)
            elif self.reference_page == "组件目录":
                self.components(surface)
            elif self.reference_page == "交互状态":
                from mojive.design.native.reference_states import STATE_PARTS, state_matrix

                title(ui, "交互状态")
                _, self.state_part = self.combo(
                    "##state-part",
                    self.state_part,
                    STATE_PARTS,
                    {part: entry["label"] for part, entry in self.editor.catalog.items()},
                )
                state_matrix(self, surface)
                self.sample(surface, self.state_part)
                self.behaviors(surface)
            elif self.reference_page == "布局规则":
                self.layout_page = self.choose(
                    surface,
                    "layout-page",
                    self.layout_page,
                    ("尺寸与层级", "共享规范", "模块规范", "整体布局", "原生圆角"),
                )
                if self.layout_page == "尺寸与层级":
                    layout_board(ui, self.runtime)
                    g.spacing()
                    self.responsive_example(surface)
                else:
                    self.mode = self.layout_page
                    if self.mode == "模块规范":
                        from mojive.design.editor import FAMILY_LABELS

                        _, self.family = self.combo(
                            "##reference-family", self.family, tuple(FAMILY_LABELS), FAMILY_LABELS
                        )
                    self.draw_parameter_list(surface)
            elif self.reference_page == "功能原型":
                self.prototypes(surface)
            else:
                self.coverage(surface)
            self.page_scroll_max = g.get_scroll_max_y()
            self.page_height = g.get_window_height()
        finally:
            g.end_child()

        self.draw_overlay(surface, extra_records=self._sample_records)

    def components(self, surface):
        self.mode = "部件参数"
        s = surface.s
        width = min(205 * s, g.get_content_region_avail().x * 0.28)
        g.begin_child("Reference catalogue", (width, 0))
        try:
            g.set_next_item_width(-1)
            _, self.search = g.input_text_with_hint(
                "##reference-search", "搜索名称或标识", self.search
            )
            self.record_item("part-search")
            previous = None
            for part in self.catalogue_order:
                definition = self.editor.catalog[part]
                if self.search.lower() not in (part + definition["label"]).lower():
                    continue
                if previous != definition["category"]:
                    previous = definition["category"]
                    g.spacing()
                    g.text_colored(rgba("primary"), previous)
                clicked, _ = g.selectable(definition["label"] + "##" + part, self.part == part)
                self.record_item("catalogue/" + part)
                if clicked:
                    self.part = part
                    self.selected_field = None
                    self.field_search = ""
        finally:
            g.end_child()
        g.same_line()
        g.begin_child("Reference detail", (0, 0))
        try:
            if self._last_part != self.part:
                g.set_scroll_y(0)
                self._last_part = self.part
            definition = self.editor.catalog[self.part]
            title(surface.ui, definition["label"], self.part)
            self.detail_tab = self.choose(
                surface,
                "reference-detail",
                self.detail_tab,
                ("示例与规则", "尺寸与参数", "适用条件"),
            )
            if self.detail_tab == "尺寸与参数":
                self.sample(surface, self.part)
                g.begin_child("Reference parameters", (0, 0))
                try:
                    self.draw_parameter_list(surface)
                finally:
                    g.end_child()
            elif self.detail_tab == "适用条件":
                self.applicability()
            else:
                self.sample(surface, self.part)
                g.spacing()
                self.quick_parameters(surface, self.part)
        finally:
            g.end_child()

    def sample(self, surface, part):
        s, ui = surface.s, surface.ui
        g.push_id("sample/" + part)
        variants = sample_variants(part)
        if variants:
            choices = ("当前值", *variants)
            changed, selected = self.combo(
                "##content-variant",
                self.variants.get(part, "当前值"),
                choices,
                key="sample-variant",
            )
            self.variants[part] = selected
            if changed and selected in variants:
                self.specimens.values[part] = variants[selected]
        if part in ("select_popup", "select_option"):
            _, with_icon = g.checkbox("显示图标", self.specimens.values.get(part + "/icon", False))
            self.specimens.values[part + "/icon"] = with_icon
        g.set_next_item_width(min(260 * s, g.get_content_region_avail().x))
        _, self.specimen_width = g.slider_float(
            "可用宽度", self.specimen_width, 140, 900, "%.0f px"
        )
        self.record_item("sample-width")
        _, self.disabled = g.checkbox("禁用", self.disabled)
        self.record_item("sample-disabled")
        g.same_line()
        _, self.show_bounds = g.checkbox("显示边界", self.show_bounds)
        if part in (
            "button",
            "tab",
            "filter",
            "menu_trigger",
            "menu_item",
            "link_button",
            "tree_label",
        ):
            g.same_line()
            _, self.specimens.long_text = g.checkbox("长名称", self.specimens.long_text)
        g.same_line()
        if self.button("恢复样例", "sample-reset"):
            self.specimens.values.clear()
            self.specimens.color_editors.clear()
            self.variants.clear()
        available = g.get_content_region_avail().x
        width = min(self.specimen_width * s, max(1, available - 48 * s))
        w, h = self.specimens.measure(ui, part, width)
        height = max(220 * s, h + 64 * s)
        g.begin_child(
            "Reference specimen",
            (0, height),
            g.ChildFlags_.borders,
            g.WindowFlags_.horizontal_scrollbar,
        )
        try:
            x, y = g.get_cursor_screen_pos()
            region = g.get_content_region_avail()
            rect = Rect(x + max(0, (region.x - w) / 2), y + max(0, (region.y - h) / 2), w, h)
            g.push_id(part)
            g.begin_disabled(self.disabled)
            try:
                record = self.specimens.paint(ui, part, rect)
                self._sample_records.extend(record["controls"])
            finally:
                g.end_disabled()
                g.pop_id()
            self.sample_stage = (x, y, region.x, region.y)
            self.sample_rect = rect.tuple()
            if self.show_bounds:
                from mojive.ui.design_palette import rgba8
                from mojive.ui.workspace.widgets import color

                for name, role in (("frame", "border"), ("content", "primary")):
                    a, b, c, d = record[name]
                    ui.draw.add_rect((a, b), (a + c, b + d), color(rgba8(role)))
            g.set_cursor_screen_pos((x, y))
            g.dummy((max(region.x, w), max(region.y, h)))
        finally:
            g.end_child()
            g.pop_id()

    def quick_parameters(self, surface, part):
        from mojive.design.specification import resolve_part_value

        fields = self.editor.catalog[part]["fields"]
        selected = [
            k for k in ("columnGap", "rowGap", "paddingX", "paddingY", "height") if k in fields
        ]
        if not selected:
            return
        g.text_unformatted("间距与尺寸")
        for key in selected:
            rule = fields[key]
            value = resolve_part_value(self.runtime.spec, part, key)
            g.set_next_item_width(160 * surface.s)
            changed, next_value = g.drag_float(
                rule["label"] + "##quick/" + part + "/" + key,
                float(value),
                0.5,
                rule["min"],
                rule["max"],
                "%.0f px",
            )
            self.record_item("quick/" + part + "/" + key)
            if g.is_item_activated():
                self.selected_field = ("parts", part, key)
            if changed:
                self.change(
                    ("parts", part, key), round(next_value) if rule.get("step") == 1 else next_value
                )

    def applicability(self):
        record = self.manifest["parts"][self.part]
        if self.part in UNAVAILABLE_PARTS:
            g.text_colored(rgba("warning"), UNAVAILABLE_PARTS[self.part])
        for field, definition in record["fields"].items():
            g.separator()
            g.text_unformatted(definition["label"] + " · " + field)
            g.text_colored(
                rgba("primary" if definition["status"] == "applied" else "warning"),
                definition["nativeLabel"],
            )
            g.text_wrapped(definition["description"])
            if definition["condition"]:
                g.text_wrapped(definition["condition"])
            for source in definition["consumers"]:
                g.text_wrapped(source["source"])

    def responsive_example(self, surface):
        g.text_unformatted("向量布局 · 往返调整同一宽度可检查排列一致性")
        self.sample(surface, "vector_row")

    def behaviors(self, surface):
        from mojive.design.native.reference_states import scroll_panels

        labels = tuple(item[0] for item in BEHAVIORS)
        current = labels[self.behavior_index]
        changed, current = self.combo("##behavior", current, labels)
        if changed:
            self.behavior_index = labels.index(current)
        label, flow, _rule, part = BEHAVIORS[self.behavior_index]
        g.text_unformatted(flow)
        if label == "滚动条":
            scroll_panels(self, surface)
        else:
            self.sample(surface, part)

    def prototypes(self, surface):
        title(surface.ui, "功能原型")
        entries = dict(PROTOTYPES)
        labels = tuple(entries)
        selected = next(
            (label for label, part in entries.items() if part == self.prototype_part), labels[0]
        )
        changed, selected = self.combo("##prototype-scene", selected, labels)
        if changed:
            self.prototype_part = entries[selected]
        production = getattr(self, "production_prototype", None)
        if production is not None:
            s = surface.s
            x, y = g.get_cursor_screen_pos()
            region = g.get_content_region_avail()
            width = min(
                region.x - 48 * s,
                (
                    860
                    if self.prototype_part
                    in (
                        "timeline_model_group",
                        "settings_page",
                        "toolbox",
                        "camera_preview",
                        "textarea",
                    )
                    else 560
                )
                * s,
            )
            height = min(
                max(240 * s, region.y - 32 * s),
                {
                    "output_toolbar": 280,
                    "panel:Control": 280,
                    "panel:Layers": 320,
                    "panel:Assets": 340,
                    "section": 440,
                    "color_input": 460,
                    "timeline_model_group": 440,
                    "menu_item": 300,
                }.get(self.prototype_part, 800)
                * s,
            )
            rect = Rect(x + (region.x - width) / 2, y + (region.y - height) / 2, width, height)
            self.prototype_rect = rect.tuple()
            production(surface.ui, self.prototype_part, rect)
        else:
            self.sample(
                surface,
                self.prototype_part if not self.prototype_part.startswith("panel:") else "panel",
            )
            self.quick_parameters(
                surface,
                self.prototype_part if not self.prototype_part.startswith("panel:") else "panel",
            )

    def coverage(self, surface):
        title(surface.ui, "组件总览")
        categories = ("全部", *dict.fromkeys(v["category"] for v in self.editor.catalog.values()))
        _, self.coverage_category = self.combo(
            "##coverage-category", self.coverage_category, categories
        )
        ui, s = surface.ui, surface.s
        parts = [
            p
            for p in self.catalogue_order
            if self.coverage_category in ("全部", self.editor.catalog[p]["category"])
        ]
        columns = max(1, min(3, int(g.get_content_region_avail().x / (300 * s))))
        cell_width = (g.get_content_region_avail().x - (columns - 1) * 12 * s) / columns
        for index, part in enumerate(parts):
            if index % columns:
                g.same_line()
            g.push_id("atlas/" + part)
            visible = g.begin_child(
                "card", (cell_width, 244 * s), g.ChildFlags_.borders, g.WindowFlags_.no_scrollbar
            )
            try:
                if visible:
                    if g.selectable(self.editor.catalog[part]["label"], False)[0]:
                        self.part = part
                        self.reference_page = "组件目录"
                    x, y = g.get_cursor_screen_pos()
                    available = g.get_content_region_avail()
                    w, h = self.specimens.measure(ui, part, max(1, available.x - 24 * s))
                    # Each tile shows the real specimen at native scale; tall panels scroll independently.
                    g.begin_child("card-content", (0, 0))
                    try:
                        x, y = g.get_cursor_screen_pos()
                        self.specimens.paint(
                            ui,
                            part,
                            Rect(
                                x + max(0, (available.x - w) / 2),
                                y + max(0, (available.y - h) / 2),
                                w,
                                h,
                            ),
                        )
                        g.set_cursor_screen_pos((x, y))
                        g.dummy((w, h))
                    finally:
                        g.end_child()
            finally:
                g.end_child()
                g.pop_id()
