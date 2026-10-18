"""Native catalogue, specification and file-management pages."""

from mojive.design.editor import FAMILY_LABELS, MISSING
from mojive.design.help import condition_help, get_field_help, get_part_help
from mojive.design.specification import SCHEMA
from mojive.ui.design_palette import rgba

_PARAMETER_GROUPS = ("尺寸与形状", "排版与间距", "文字", "图标")
_EDGE_FIELDS = frozenset(
    prefix + edge for prefix in ("padding", "margin") for edge in ("Left", "Right", "Top", "Bottom")
)


def _parameter_group(field, rule):
    if group := rule.get("group"):
        return {"边距与间距": "排版与间距", "尺寸": "尺寸与形状"}.get(group, group)
    if field in (
        "fontSize",
        "fontFamily",
        "textColor",
        "lineHeightMode",
        "lineHeight",
        "textAlign",
        "detailSize",
        "valueSize",
        "unitSize",
    ):
        return "文字"
    name = field.lower()
    if "icon" in name or "thumb" in name:
        return "图标"
    if any(
        word in name for word in ("gap", "padding", "margin", "align", "layout", "columns", "label")
    ):
        return "排版与间距"
    return "尺寸与形状"


def _parameter_groups(fields):
    declared = dict.fromkeys(_parameter_group(field, rule) for field, rule in fields.items())
    return _PARAMETER_GROUPS + tuple(group for group in declared if group not in _PARAMETER_GROUPS)


class EditorPages:
    def draw_contents(self, workspace):
        from imgui_bundle import imgui as g

        self.draw_header(workspace)
        previous_page = self.page
        self.page = self.choose(workspace, "page", self.page, ("交互原型", "设计规范"))
        if self.page != previous_page:
            self._catalogue_scroll_pending = True
        previous_mode = self.mode
        self.mode = self.choose(
            workspace,
            "mode",
            self.mode,
            ("部件参数", "共享规范", "模块规范", "整体布局", "原生圆角"),
            labels={
                "部件参数": "部件",
                "共享规范": "共享",
                "模块规范": "模块",
                "整体布局": "布局",
                "原生圆角": "圆角",
            },
        )
        if self.mode != previous_mode:
            self.selected_field = None
            self.field_search = ""
        g.spacing()
        columns = (
            self.page == "设计规范"
            and self.mode == "部件参数"
            and g.get_content_region_avail().x >= 560 * workspace.s
        )
        if self.mode == "部件参数" and not columns:
            self.part_selector(workspace)
            self.draw_selection_hierarchy(workspace)
        elif self.mode == "模块规范":
            _, self.family = self.combo(
                "##family", self.family, tuple(FAMILY_LABELS), FAMILY_LABELS
            )
        if columns:
            self.draw_specification(workspace)
        else:
            self.draw_parameter_list(workspace)
        self.draw_footer(workspace)

    def draw_selection_hierarchy(self, workspace):
        from imgui_bundle import imgui as g

        related = {
            "assets_material": ("assets_material_list",),
            "assets_material_list": ("assets_material",),
            "output_row": ("output_list", "output_toolbar", "output_layout"),
            "output_list": ("output_row", "output_toolbar", "output_layout"),
            "output_toolbar": ("output_layout", "output_list", "output_row"),
            "output_layout": ("output_toolbar", "output_list", "output_row"),
            "timeline_model_group": ("select", "icon_button", "timeline_sheet"),
            "statusbar": ("status_context", "status_metrics", "status_separator"),
            "status_context": ("status_hints", "statusbar", "status_separator"),
            "status_hints": ("status_context", "statusbar", "status_separator"),
            "status_metrics": ("statusbar", "status_hints", "status_separator"),
            "status_separator": ("statusbar", "status_hints", "status_metrics"),
        }
        definition = self.editor.catalog[self.part]
        controls = definition.get("controlParts", {}).values()
        parent = definition.get("controlGroup")
        parts = list(
            dict.fromkeys(
                (
                    *controls,
                    *((parent,) if parent else ()),
                    *((definition["basePart"],) if parent else ()),
                    *self.selection_parts,
                    *related.get(self.part, ()),
                )
            )
        )
        owner = self.spacing_owner()
        if owner is not None:
            parts.insert(0, owner[0])
        parts = list(
            dict.fromkeys(
                part for part in parts if part in self.editor.catalog and part != self.part
            )
        )
        shared = [
            part for part in parts if self.editor.catalog[part].get("selectionScope") != "group"
        ]
        parts = [part for part in parts if part not in shared]
        if not parts and not shared:
            return
        # Short native windows reserve the remaining space for parameter input.
        # Related owners remain reachable in the same menu used for overflow.
        visible_parts = parts[:3] if g.get_content_region_avail().y >= 360 * workspace.s else []
        if visible_parts:
            g.text_disabled("关联分组")
        for index, part in enumerate(visible_parts):
            label = self.editor.catalog[part]["label"]
            if index:
                self.next_button_on_row(label)
            if self.button(label, "parent/" + part):
                self.part = part
                self.selected_field = None
                self.field_search = ""
                self.field_group = "全部参数"
        remaining_parts = parts[len(visible_parts) :]
        if remaining_parts:
            caption = "更多分组" if visible_parts else "关联分组"
            if visible_parts:
                self.next_button_on_row(caption)
            if self.button(caption, "more-parents"):
                g.open_popup("关联分组")
            if g.begin_popup("关联分组"):
                try:
                    for part in remaining_parts:
                        if g.selectable(self.editor.catalog[part]["label"] + "##" + part, False)[0]:
                            self.part = part
                            self.selected_field = None
                            self.field_search = ""
                            self.field_group = "全部参数"
                        self.record_item("parent/" + part)
                finally:
                    g.end_popup()
        if shared:
            opened = g.tree_node("全局控件类（影响所有分组）##shared-classes")
            self.record_item("shared-classes")
            if opened:
                try:
                    for part in shared:
                        if self.button(self.editor.catalog[part]["label"], "parent/" + part):
                            self.part = part
                            self.selected_field = None
                            self.field_search = ""
                            self.field_group = "全部参数"
                finally:
                    g.tree_pop()

    def draw_parameter_list(self, workspace):
        from imgui_bundle import imgui as g

        if (
            self.page == "设计规范"
            and self.mode == "部件参数"
            and g.get_content_region_avail().x < 560 * workspace.s
        ):
            self.specimens.draw(workspace.ui, self.part)

        available = g.get_content_region_avail().x
        filter_width = min(112 * workspace.s, available * 0.34) if self.mode == "部件参数" else 0
        g.set_next_item_width(
            max(1, available - filter_width - (g.get_style().item_spacing.x if filter_width else 0))
        )
        _, self.field_search = g.input_text_with_hint(
            "##field-filter", "搜索参数", self.field_search
        )
        self.record_item("field-filter")
        self.tooltip("按参数名称筛选，例如：行间距、字号、图标、对齐。")
        if filter_width:
            g.same_line()
            groups = ("全部参数", *_parameter_groups(self.editor.catalog[self.part]["fields"]))
            if self.field_group not in groups:
                self.field_group = "全部参数"
            _, self.field_group = self.combo(
                "##field-group",
                self.field_group,
                groups,
                key="field-group",
                width=filter_width,
            )
        height = max(1, g.get_content_region_avail().y - 30 * workspace.s)
        g.begin_child("参数逐项列表", (0, height))
        try:
            selection = (self.mode, self.part, self.family, self.field_search, self.field_group)
            if getattr(self, "_field_list_selection", None) != selection:
                g.set_scroll_y(0)
                self._field_list_selection = selection
            self.draw_fields(workspace)
        finally:
            g.end_child()

    def editor_action(self, action):
        try:
            result = action()
            self.runtime.poll(force=True)
            return result
        except (OSError, ValueError) as error:
            self.editor.error = "规范操作失败：" + str(error)
            return False

    def draw_specification(self, workspace):
        """Keep the catalogue and each field's definition visible together."""
        from imgui_bundle import imgui as g

        s = workspace.s
        height = max(1, g.get_content_region_avail().y - 27 * s)
        catalogue_width = min(178 * s, g.get_content_region_avail().x * 0.31)
        g.begin_child("部件目录", (catalogue_width, height), g.ChildFlags_.borders)
        try:
            g.text_unformatted("部件目录 · " + str(len(self.editor.catalog)))
            categories = list(dict.fromkeys(p["category"] for p in self.editor.catalog.values()))
            _, self.category = self.combo(
                "##category", self.category, ["全部部件", *categories], key="category"
            )
            g.set_next_item_width(-1)
            _, self.search = g.input_text_with_hint("##part-search", "搜索部件名称", self.search)
            self.record_item("part-search")
            g.separator()
            g.begin_child("部件条目", (0, 0))
            try:
                last_category = None
                for part in self.editor.catalog.values():
                    if self.category != "全部部件" and part["category"] != self.category:
                        continue
                    if (
                        self.search
                        and self.search.lower() not in (part["label"] + " " + part["id"]).lower()
                    ):
                        continue
                    if last_category != part["category"]:
                        last_category = part["category"]
                        g.spacing()
                        g.text_colored(rgba("primary"), last_category)
                    label = self.fit_label(part["label"], g.get_content_region_avail().x)
                    clicked, _ = g.selectable(
                        label + "##catalogue-" + part["id"], self.part == part["id"]
                    )
                    self.record_item("catalogue/" + part["id"])
                    self.tooltip(part["label"], shown=label)
                    if self._catalogue_scroll_pending and self.part == part["id"]:
                        g.set_scroll_here_y(0.35)
                        self._catalogue_scroll_pending = False
                    if clicked:
                        self.part = part["id"]
                        self.selected_field = None
                        self.selection_parts = []
                        self.field_search = ""
                        self.request_navigation(workspace)
            finally:
                g.end_child()
        finally:
            g.end_child()
        g.same_line()
        g.begin_child("部件规格", (0, height))
        try:
            g.begin_child(
                "部件说明", (0, max(1, min(250 * s, g.get_content_region_avail().y * 0.42)))
            )
            try:
                self.part_context(workspace)
            finally:
                g.end_child()
            self.draw_parameter_list(workspace)
        finally:
            g.end_child()

    def part_context(self, workspace):
        from imgui_bundle import imgui as g

        s = workspace.s
        definition = self.editor.catalog[self.part]
        info = get_part_help(self.part)
        g.push_font(workspace.ui.body, 18 * s)
        g.text_wrapped(definition["label"])
        g.pop_font()
        self.record_item("part-title")
        self.tooltip(info["context"])
        self.specimens.draw(workspace.ui, self.part)
        g.push_style_color(g.Col_.text, rgba("primary"))
        g.text_unformatted(
            str(len(definition["fields"])) + " 项 · " + str(len(self.native_instances)) + " 个实例"
        )
        g.pop_style_color()
        if definition.get("selectionScope") == "group":
            g.text_disabled("独立分组 · " + definition["label"])
        else:
            g.text_disabled("全局控件类 · 影响所有分组")
        if self.button("定位到原生界面", "locate", tooltip=info["preview"]["steps"]):
            self.request_navigation(workspace)
        self.next_button_on_row("恢复此部件")
        if self.button(
            "恢复此部件",
            "reset-part",
            enabled=bool(self.editor.spec.get("parts", {}).get(self.part)),
        ):
            self.change(("parts", self.part), MISSING)
        self.tooltip("；".join(info["preview"]["steps"]))
        if not self.native_instances:
            g.text_unformatted("点击定位，显示部件")
            if info.get("emptyReason"):
                self.tooltip(info["emptyReason"])

    def part_selector(self, workspace):
        from imgui_bundle import imgui as g

        s = workspace.s
        definition = self.editor.catalog[self.part]
        available = g.get_content_region_avail().x
        left = g.get_cursor_pos_x()
        side, gap = g.get_frame_height(), g.get_style().item_spacing.x
        g.set_next_item_width(max(1, available - side - gap))
        g.set_next_window_size_constraints((min(340 * s, available), 0), (600 * s, 460 * s))
        opened = g.begin_combo("##part", definition["label"])
        self.record_item("part")
        if opened:
            g.set_next_item_width(-1)
            if g.is_window_appearing():
                g.set_keyboard_focus_here()
            _, self.search = g.input_text_with_hint(
                "##part-search", "搜索部件名称或标识", self.search
            )
            self.record_item("part-search")
            categories = list(dict.fromkeys(p["category"] for p in self.editor.catalog.values()))
            _, self.category = self.combo(
                "##category", self.category, ["全部部件", *categories], key="category"
            )
            g.separator()
            g.begin_child("搜索部件", (0, 280 * s))
            last_category, matches = None, 0
            for part in self.editor.catalog.values():
                if self.category != "全部部件" and part["category"] != self.category:
                    continue
                if (
                    self.search
                    and self.search.lower() not in (part["label"] + " " + part["id"]).lower()
                ):
                    continue
                matches += 1
                if last_category != part["category"]:
                    last_category = part["category"]
                    g.spacing()
                    g.text_colored(rgba("secondary"), last_category)
                clicked, _ = g.selectable(
                    part["label"] + "##" + part["id"], self.part == part["id"], size=(0, 28 * s)
                )
                self.record_item("catalogue/" + part["id"])
                if clicked:
                    self.part = part["id"]
                    self.selected_field = None
                    self.selection_parts = []
                    self.field_search = ""
                    self.field_group = "全部参数"
                    self.request_navigation(workspace)
                    g.close_current_popup()
            if not matches:
                g.text_disabled("没有匹配的部件")
            g.end_child()
            g.end_combo()
        g.same_line(left + available - side)
        if self.tool_button(workspace, "target", "locate", "定位到原生界面"):
            self.request_navigation(workspace)
        g.align_text_to_frame_padding()
        scope = "当前分组" if definition.get("selectionScope") == "group" else "全局控件类"
        g.text_disabled(f"{scope} · {len(self.native_instances)} 处实例")
        reset_width = g.calc_text_size("恢复部件").x + 2 * g.get_style().frame_padding.x
        g.same_line(left + available - reset_width)
        if self.button(
            "恢复部件", "reset-part", enabled=bool(self.editor.spec.get("parts", {}).get(self.part))
        ):
            self.change(("parts", self.part), MISSING)
        g.spacing()
        g.text_wrapped(definition["label"].replace(" · ", "  ›  "))
        self.record_item("selection-path")
        g.spacing()

    def spacing_owner(self):
        control_group = self.editor.catalog[self.part].get("controlGroup")
        if control_group is not None:
            entry = self.editor.catalog[control_group]
            layout = entry.get("layoutSpacing", {})
            axes = ("rowGap",) if layout.get("kind") == "column" else ("columnGap", "rowGap")
            fields = tuple(field for field in axes if field in entry["fields"])
            if fields:
                return control_group, fields
        owners = {
            "status_context": ("statusbar", ("columnGap",)),
            "status_hints": ("status_context", ("columnGap",)),
            "output_row": ("output_list", ("rowGap",)),
            "select_option": ("select_popup", ("rowGap",)),
            "menu_item": ("menu", ("rowGap",)),
            "menu_item_content": ("menu", ("rowGap",)),
            "material": ("material_list", ("rowGap",)),
            "stat_row": ("data_list", ("rowGap", "columnGap")),
            "lane_label": ("timeline_sheet", ("rowGap",)),
            "timeline_lane": ("timeline_sheet", ("rowGap",)),
            "tree_label": ("tree_row", ("rowGap",)),
            "tree_expander": ("tree_row", ("rowGap",)),
        }
        if self.part in owners:
            return owners[self.part]
        if self.editor.catalog[self.part].get("selectionScope") != "group":
            return None
        for parent in self.selection_parts:
            if parent in (self.part, "application") or parent not in self.editor.catalog:
                continue
            entry = self.editor.catalog[parent]
            layout = entry.get("layoutSpacing", {})
            if not layout:
                continue
            axes = ("rowGap",) if layout["kind"] == "column" else ("columnGap", "rowGap")
            fields = tuple(axis for axis in axes if axis in entry["fields"])
            if fields:
                return parent, fields
        return None

    def draw_parent_spacing(self, workspace):
        from imgui_bundle import imgui as g

        owner = self.spacing_owner()
        if owner is None or self.field_group not in ("全部参数", "排版与间距"):
            return
        parent, fields = owner
        entry = self.editor.catalog[parent]
        matches = []
        for field in fields:
            rule = entry["fields"][field]
            label = "条目垂直间距" if field == "rowGap" else "条目水平间距"
            if self.field_matches(field, {**rule, "label": label + " item spacing"}):
                matches.append((field, rule, label))
        if not matches:
            return
        g.spacing()
        g.text_disabled("组内间距")
        self.tooltip(entry["label"], shown="组内间距")
        values = self.editor.effective(parent)
        for field, rule, label in matches:
            self.field(
                workspace,
                ("parts", parent, field),
                rule,
                values[field],
                help={**get_field_help(parent, field), "shortLabel": label},
            )

    def draw_part_field(self, workspace, field, fields, effective):
        self.field(
            workspace,
            ("parts", self.part, field),
            fields[field],
            effective[field],
            help=get_field_help(self.part, field),
            condition=condition_help(self.part, field, effective),
        )

    def draw_fields(self, workspace):
        from imgui_bundle import imgui as g

        if self.mode == "部件参数":
            self.draw_parent_spacing(workspace)
            area = get_part_help(self.part)["preview"]["area"]
            family = {
                "检查器": "inspector",
                "材质": "inspector",
                "颜色编辑": "inspector",
                "时间轴": "timeline",
                "姿态": "pose",
                "输出": "output",
                "设置": "settings",
                "顶栏菜单": "menus",
                "顶栏": "chrome",
                "活动栏": "chrome",
                "场景": "navigation",
                "视口": "viewport",
                "相机预览": "viewport",
            }.get(area)
            effective = self.editor.effective(self.part, family=family)
            fields = self.editor.catalog[self.part]["fields"]
            preferred = (
                "rowGap",
                "columnGap",
                "gap",
                "layout",
                "columns",
                "labelWidth",
                "labelGap",
                "align",
                "alignItems",
                "textAlign",
                "fontSize",
                "fontFamily",
                "textColor",
                "lineHeightMode",
                "lineHeight",
                "iconSize",
                "iconColor",
                "heightMode",
                "height",
                "minHeight",
                "widthMode",
                "width",
            )
            ordered = sorted(
                fields,
                key=lambda f: (
                    preferred.index(f) if f in preferred else len(preferred),
                    list(fields).index(f),
                ),
            )
            groups = {name: [] for name in _parameter_groups(fields)}
            for field in ordered:
                rule = fields[field]
                group = _parameter_group(field, rule)
                if self.field_group != "全部参数" and self.field_group != group:
                    continue
                if self.field_matches(field, rule):
                    groups[group].append(field)
            matched = sum(map(len, groups.values()))
            if not matched:
                g.text_disabled("没有匹配的参数")
            for group, names in groups.items():
                if not names:
                    continue
                g.push_id(self.part + group)
                g.spacing()
                if self.field_search:
                    g.set_next_item_open(True)
                opened = g.collapsing_header(
                    f"{group} · {len(names)}", flags=g.TreeNodeFlags_.default_open
                )
                self.record_item("field-section/" + group)
                if opened:
                    basic = [field for field in names if field not in _EDGE_FIELDS]
                    edges = [field for field in names if field in _EDGE_FIELDS]
                    for field in basic:
                        self.draw_part_field(workspace, field, fields, effective)
                    if edges:
                        selected = self.selected_field
                        if self.field_search or (
                            selected
                            and selected[:2] == ("parts", self.part)
                            and selected[2] in edges
                        ):
                            g.set_next_item_open(True)
                        count = sum(
                            field in self.editor.spec.get("parts", {}).get(self.part, {})
                            for field in edges
                        )
                        caption = "四边独立设置" + (f" · {count} 项覆盖" if count else "")
                        advanced = g.tree_node(caption + "##edge-overrides")
                        self.record_item("field-advanced/" + group)
                        if advanced:
                            try:
                                for field in edges:
                                    self.draw_part_field(workspace, field, fields, effective)
                            finally:
                                g.tree_pop()
                g.pop_id()
        elif self.mode in ("共享规范", "模块规范"):
            group = None
            for field, rule in SCHEMA["tokens"].items():
                if not self.field_matches(field, rule):
                    continue
                if group != rule["group"]:
                    group = rule["group"]
                    g.text_colored(rgba("primary"), group)
                if self.mode == "模块规范":
                    path = ("components", self.family, field)
                    value = (
                        self.editor.spec.get("components", {})
                        .get(self.family, {})
                        .get(field, self.editor.spec["tokens"][field])
                    )
                else:
                    path, value = ("tokens", field), self.editor.spec["tokens"][field]
                self.field(workspace, path, rule, value)
        elif self.mode == "整体布局":
            g.text_wrapped("默认分区尺寸。改动后重建停靠布局，随后仍可手动拖动分隔线。")
            for field, rule in SCHEMA["layout"].items():
                self.field(
                    workspace,
                    ("layout", field),
                    rule,
                    self.editor.spec["layout"][field],
                    help={
                        "summary": rule["description"],
                        "reference": rule["reference"],
                        "unit": "像素",
                    },
                )
        elif self.mode == "原生圆角":
            g.text_colored(rgba("primary"), "连续曲线")
            enabled = self.editor.spec["native"]["g3"]
            for field, rule in SCHEMA["native"].items():
                if not self.field_matches(field, rule):
                    continue
                self.field(
                    workspace,
                    ("native", field),
                    rule,
                    self.editor.spec["native"][field],
                    help={"summary": rule["description"]},
                    condition={
                        "enabled": field == "g3" or enabled,
                        "reason": "" if enabled else "G3 连续圆角已关闭。",
                        "enable": "开启 G3 连续圆角后调整。",
                    },
                )
            g.spacing()
            g.text_colored(rgba("primary"), "开关胶囊")
            for field in ("switchWidth", "switchHeight", "switchThumb"):
                rule = self.editor.catalog["switch"]["fields"][field]
                if self.field_matches(field, rule):
                    self.field(
                        workspace,
                        ("parts", "switch", field),
                        rule,
                        self.editor.effective("switch")[field],
                        help=get_field_help("switch", field),
                    )

    def field_matches(self, field, rule):
        aliases = {
            "rowGap": "行间距 行距 纵向间距 垂直间距",
            "columnGap": "列间距 列距 横向间距 水平间距",
            "fontSize": "字号 字体大小 文字大小",
            "iconSize": "图标尺寸 图标大小",
            "alignItems": "垂直对齐",
            "align": "水平对齐",
            "textAlign": "文字对齐",
            "paddingX": "左右内边距 水平内边距",
            "paddingY": "上下内边距 垂直内边距",
        }
        return (
            not self.field_search
            or self.field_search.lower()
            in (rule["label"] + " " + field + " " + aliases.get(field, "")).lower()
        )
