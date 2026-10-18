"""Render catalogue component references with the production control painters."""

from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.controls import NativePrimitives
from mojive.design.native.geometry import (
    Box,
    NativePartStyle,
    Rect,
    part_scope,
)
from mojive.design.specification import component_catalog


class ReferenceValues:
    """Use catalogue defaults in references while production retains its authored slots."""

    authored_defaults = False

    def __init__(self, runtime):
        self.owner = runtime

    def __getattr__(self, name):
        return getattr(self.owner, name)


class ComponentSpecimens:
    def __init__(self, runtime):
        self.runtime = ReferenceValues(runtime)
        self.values = {}
        self.last = {}
        self.long_text = False
        self.color_editors = {}
        self.objects = {}
        self.color_heights = {}

    def measure(self, ui, part, width):
        """Return a specimen's logical content extent before placing it on a plate."""
        definition = component_catalog()[part]
        if "controlGroup" in definition:
            with part_scope(definition["controlGroup"], inherit=False):
                return self.measure(ui, definition["basePart"], width)
        if "statusKey" in definition:
            from mojive.design.native.specimen_groups import status_readout_geometry

            field, draw, groups = status_readout_geometry(ui, part)
            separators = int(field.key != "backend") + int(field.key != "render")
            return (
                field.left_gap
                + field.width
                + field.right_gap
                + separators * groups.metric_options()["separator_width"],
                max(24 * ui.s, draw.text_size("M")[1]) + 12 * ui.s,
            )
        if "basePart" in definition:
            with group_scope(definition["groupOwner"]):
                return self.measure(ui, definition["basePart"], width)
        if part == "settings_nav":
            requested = ui.metric(part, "navWidth", ui.metric("settings_layout", "navWidth", 0))
            if requested:
                width = min(width, requested * ui.s)
        elif part == "settings_layout":
            requested = ui.metric(part, "windowWidth", 0)
            if requested:
                width = min(width, requested * ui.s)
        from mojive.design.native.specimen_groups import ACTIONS
        from mojive.ui.workspace.button_rows import measure_button_row
        from mojive.ui.workspace.labeled_input import measure_labeled_input

        s = ui.s
        engine = NativePrimitives(ui, self.runtime)
        if part == "form_popup":
            from mojive.design.native.specimen_forms import form_reference

            _toolbar, _ctx, layout = form_reference(self, ui)
            return layout.width, layout.height
        if part in ("scene_tools", "pose_tools"):
            from mojive.design.native.specimen_groups import filter_geometry

            return filter_geometry(ui, part, width)[:2]
        if part in (
            "statusbar",
            "status_context",
            "status_hints",
            "status_metrics",
            "status_separator",
        ):
            from mojive.ui.workspace.status_components import (
                StatusGroups,
                resolve_status_bar_layout,
            )

            style = ui.component_style(part)
            if part == "statusbar":
                return width, resolve_status_bar_layout(ui).height
            if part == "status_separator":
                height = (
                    style.get("height", 18)
                    + style.get("marginTop", 5)
                    + style.get("marginBottom", 5)
                )
                return min(width, 160 * s), height * s
            edges = StatusGroups(ui).edges(part) if part != "statusbar" else (0, 0, 0, 0)
            minimum = 24 * s + edges[1] + edges[3]
            return width, minimum
        if part == "button_row":
            geometry = measure_button_row(ui, width, [x[0] for x in ACTIONS[part]])
            natural = max(r.right for r in geometry.rectangles)
            pads = ui.component_style(part).edges("padding", s)
            width = min(width, natural + pads[0] + pads[2])
            return width, measure_button_row(ui, width, [x[0] for x in ACTIONS[part]]).outer_height
        if part in ACTIONS:
            from mojive.design.native.specimen_groups import action_layout

            box, _ = action_layout(ui, part, Rect(0, 0, width, 0))
            return box.outer.width, box.outer.height
        if part == "checkbox":
            extent = ui.metric("checkbox", "size", 18) * s
            return extent, extent
        if part == "switch":
            from mojive.ui.workspace.property_style import switch_form_layout

            width = min(width, 200 * s)
            return width, switch_form_layout(ui, width, "显示").height
        if part in ("select_popup", "select_option", "menu", "menu_item", "menu_item_content"):
            from mojive.design.native.specimen_groups import choice_size

            with ui.component_scope(part):
                return choice_size(self, ui, part, width)
        if part == "badge":
            from mojive.ui.workspace.status_components import StatusBadge

            return StatusBadge(ui).size("已暂停")
        if part == "control_metadata":
            style = ui.component_style(part)
            padding = style.edges("padding", s, (8, 3, 8, 3))
            return width, style.line_height(style.get("fontSize", 12) * 1.4) * s + padding[
                1
            ] + padding[3]
        if part in ("rgba", "color_channels"):
            layout = ui.numeric_layout(width, 4, parts=("rgba", part), column_options=(4, 2, 1))
            return width, layout.height
        if part == "color_input":
            style = ui.component_style(part)
            return min(width, style.get("width", 36) * s), style.get("height", 28) * s
        if part == "nav_widget":
            return min(width, 120 * s), 120 * s
        if part == "range_bounds":
            layout = measure_labeled_input(
                ui, width, "起点", (82 * s, engine.field_height("input_number") * s)
            )
            return layout.width, layout.height
        if part in (
            "input_number",
            "input_text",
            "select",
            "axis_input",
            "input_shell",
            "range",
            "checkbox",
            "switch",
        ):
            return width, engine.field_height(part) * s
        if part in ("icon_button", "tree_expander", "timeline_key"):
            return 28 * s, 28 * s
        if part in (
            "button",
            "tab",
            "filter",
            "menu_trigger",
            "menu_item",
            "link_button",
            "tree_label",
        ):
            return min(width, 220 * s), engine.field_height(part) * s
        if part in ("vector_row", "vector_group", "vector_heading", "joint"):
            layout = engine.form_layout(Rect(0, 0, width, 28 * s), 3, vector=True, axis=True)
            return width, max(r.bottom for r in [layout["label"], *layout["fields"]])
        if part == "form_row":
            layout = engine.form_layout(Rect(0, 0, width, 28 * s))
            return width, max(r.bottom for r in [layout["label"], *layout["fields"]])
        heights = {
            "textarea": 100,
            "color_editor": 420,
            "color_picker": 420,
            "color_input": 28,
            "tree_row": 140,
            "section": 100,
            "section_header": 100,
            "section_body": 100,
            "segment": 28,
            "stat_row": 36,
            "data_list": 168,
            "statistics_list": 168,
            "table": 150,
            "material": 300,
            "material_list": 300,
            "sensor": 160,
            "component_list": 160,
            "component_path": 160,
            "output_row": 124,
            "output_list": 124,
            "output_layout": 190,
            "menu": 116,
            "select_popup": 116,
            "select_option": 36,
            "menu_item_content": 116,
            "rail": 200,
            "toolbox": 200,
            "panel": 260,
            "panel_body": 160,
            "dialog": 320,
            "dialog_body": 160,
            "settings_layout": 320,
            "settings_nav": 240,
            "settings_page": 160,
            "application": 320,
            "workspace": 320,
            "camera_preview": 280,
            "camera_body": 220,
            "nav_widget": 150,
            "timeline_sheet": 190,
            "timeline_body": 206,
            "timeline_ruler": 190,
            "timeline_lane": 100,
            "lane_label": 100,
            "timeline_model_group": 36,
            "separator": 20,
            "empty": 108,
            "notice": 40,
            "status_context": 40,
            "status_hints": 40,
            "status_metrics": 40,
            "status_separator": 40,
        }
        height = heights.get(part, 48)
        if part in ("color_editor", "color_picker"):
            height = self.color_heights.get((part, width), width + 136 * s) / s
        return min(width, 48 * s) if part in ("rail", "toolbox") else width, height * s

    def draw(self, ui, part):
        width = imgui.get_content_region_avail().x
        w, h = self.measure(ui, part, max(1, width - 16 * ui.s))
        height = max(132 * ui.s, min(300 * ui.s, h + 16 * ui.s))
        imgui.begin_child(
            "Native component reference",
            (width, height),
            window_flags=imgui.WindowFlags_.horizontal_scrollbar,
        )
        try:
            pos = imgui.get_cursor_screen_pos()
            available = imgui.get_content_region_avail()
            result = self.paint(
                ui,
                part,
                Rect(
                    pos.x + max(0, (available.x - w) / 2),
                    pos.y + max(0, (available.y - h) / 2),
                    w,
                    h,
                ),
            )
            imgui.set_cursor_screen_pos((pos.x, pos.y))
            bottom = max(
                [
                    result["outer"][1] + result["outer"][3],
                    *(rect[1] + rect[3] for rect in result["children"]),
                ]
            )
            imgui.dummy((result["outer"][2] + 16 * ui.s, bottom - pos.y + 8 * ui.s))
        finally:
            imgui.end_child()

    def paint(self, ui, part, bounds):
        definition = component_catalog()[part]
        if "controlGroup" in definition:
            with part_scope(definition["controlGroup"], inherit=False):
                result = {**self.paint(ui, definition["basePart"], bounds), "part": part}
                self.last[part] = result
                return result
        if "basePart" in definition:
            with group_scope(definition["groupOwner"]):
                result = {**self.paint(ui, definition["basePart"], bounds), "part": part}
                self.last[part] = result
                return result
        engine = NativePrimitives(ui, self.runtime)
        style = NativePartStyle(self.runtime, part)
        s = ui.s
        key = "reference-" + part
        self.runtime.paint_context.paint_depth += 1
        try:
            if part == "form_popup":
                from mojive.design.native.specimen_forms import paint_form_reference

                start = len(ui.design.primitives.observed)
                box, children = paint_form_reference(self, ui, bounds)
                record = self._record(part, box, engine, children)
                record["controls"] = ui.design.primitives.observed[start:]
                return record
            if part in ("select_popup", "select_option", "menu", "menu_item", "menu_item_content"):
                from mojive.design.native.specimen_groups import choices

                with ui.component_scope(part):
                    box, children = choices(self, ui, engine, style, part, bounds)
                return self._record(part, box, engine, children)
            if part == "switch":
                from mojive.ui.workspace.property_style import property_toggle

                g = imgui
                g.set_cursor_screen_pos((bounds.x, bounds.y))
                primitives = ui.design.primitives
                start = len(primitives.observed)
                self.values[part] = property_toggle(
                    ui, key, "显示", self.values.get(part, True), width=bounds.width
                )
                record = self._record(part, Box(bounds, bounds, bounds, (0,) * 4, (0,) * 4), engine)
                record["controls"] = primitives.observed[start:]
                return record
            if part in (
                "button",
                "icon_button",
                "tab",
                "filter",
                "menu_trigger",
                "link_button",
                "tree_label",
                "tree_expander",
                "timeline_key",
            ):
                label = (
                    ""
                    if part in ("icon_button", "tree_expander", "timeline_key")
                    else (
                        "带有完整名称的组件操作按钮"
                        if self.long_text
                        else {
                            "button": "保存场景",
                            "tab": "属性",
                            "filter": "本体 11",
                            "menu_trigger": "文件",
                            "link_button": "打开文件夹",
                            "tree_label": "hinge_body",
                        }.get(part, "定位")
                    )
                )
                extent = style.get("size", 28) * s
                rect = Rect(
                    bounds.x,
                    bounds.y,
                    min(
                        bounds.width,
                        extent
                        if part in ("icon_button", "tree_expander", "timeline_key")
                        else 220 * s,
                    ),
                    style.get("height", extent / s) * s,
                )
                result = engine.button(
                    key,
                    rect,
                    label=label,
                    icon="right"
                    if part == "tree_expander"
                    else "diamond"
                    if part == "timeline_key"
                    else "scene",
                    parts=part,
                    filled=True,
                    active=self.values.get(part, False),
                )
                if result.changed:
                    self.values[part] = not self.values.get(part, False)
                return self._record(part, result.box, engine)
            if part in (
                "input_number",
                "axis_input",
                "input_text",
                "textarea",
                "select",
                "checkbox",
                "range",
                "input_shell",
            ):
                height = 100 if part == "textarea" else engine.field_height(part)
                rect = Rect(bounds.x, bounds.y, bounds.width, height * s)
                if part in ("input_number", "axis_input"):
                    result = engine.input_number(
                        key,
                        rect,
                        self.values.get(part, 1.25),
                        parts=part,
                        axis="X" if part == "axis_input" else None,
                    )
                elif part in ("input_text", "textarea"):
                    result = engine.input_text(
                        key,
                        rect,
                        self.values.get(part, "示例文字"),
                        parts=part,
                        multiline=part == "textarea",
                    )
                elif part == "select":
                    result = engine.select(
                        key,
                        rect,
                        self.values.get(part, "选项一"),
                        ("选项一", "选项二", "具有较长名称的选项三"),
                        parts=part,
                    )
                elif part == "checkbox":
                    result = engine.checkbox(key, rect.x, rect.y, self.values.get(part, True))
                elif part == "range":
                    result = engine.range(key, rect, self.values.get(part, 0.4), 0, 1)
                else:
                    result = engine.number_shell(key, rect, self.values.get(part, 1.25), unit="m")
                self.values[part] = result.value
                return self._record(part, result.box, engine)
            if part in (
                "form_row",
                "vector_row",
                "vector_group",
                "form_label",
                "vector_heading",
                "range_number",
                "range_bounds",
                "time_field",
                "joint",
            ):
                if part == "form_label":
                    result = engine.label(key, bounds, "位置 (m)", parts=part)
                    return self._record(part, result, engine)
                return self._form(ui, engine, style, part, bounds)
            if part in ("section", "section_header", "section_body"):
                box = style.box(*bounds.tuple(), scale=s)
                expanded = self.values.get(part, True)
                if ui.section(
                    "显示", box.content.x, box.content.y, box.content.width, open=expanded
                ):
                    expanded = not expanded
                self.values[part] = expanded
                children = [ui.hits["section-显示"]]
                if expanded:
                    rect = Rect(
                        box.content.x,
                        box.content.y + ui.section_offset(38) * s,
                        box.content.width,
                        engine.field_height("switch") * s,
                    )
                    result = engine.switch(
                        "reference-section-switch",
                        rect,
                        self.values.get(part + "/switch", True),
                        label="可见",
                    )
                    self.values[part + "/switch"] = result.value
                    children.append(result.box.frame.tuple())
                return self._record(part, box, engine, children)
            if part == "segment":
                box = style.box(*bounds.tuple(), scale=s)
                self.values[part] = ui.segmented(
                    key,
                    box.content.x,
                    box.content.y,
                    box.content.width / s,
                    ("外观", "碰撞", "叠加"),
                    self.values.get(part, 0),
                    icons=("visual", "collision", "both"),
                )
                return self._record(part, box, engine, [ui.hits[key]])
            if part in ("color_editor", "color_picker", "color_input"):
                return self._native_color(ui, engine, style, part, bounds)
            if part == "control_metadata":
                from mojive.ui.workspace.scene_panels import _control_detail

                _control_detail(ui, key, bounds.x, bounds.y, bounds.width, "hinge_motor · -1 … 1")
                frame = Rect(*ui.hits[key])
                padding = ui.component_style(part).edges("padding", s, (8, 3, 8, 3))
                box = Box(bounds, frame, frame.inset(*padding), padding, style.edges("margin", s))
                return self._record(part, box, engine, [frame.tuple()])
            from mojive.design.native.specimen_groups import GROUP_RENDERERS
            from mojive.design.native.specimen_surfaces import SURFACE_RENDERERS

            renderer = {**GROUP_RENDERERS, **SURFACE_RENDERERS}[part]
            if part == "statusbar":
                from mojive.ui.workspace.status_components import resolve_status_bar_layout

                with part_scope("statusbar"):
                    children = renderer(self, ui, engine, style, part, bounds)
                    box = resolve_status_bar_layout(ui).box(bounds.x, bounds.y, bounds.width)
                return self._record(part, box, engine, children)
            children = renderer(self, ui, engine, style, part, bounds)
            box = style.box(*bounds.tuple(), scale=s)
            if part == "timeline_body":
                from mojive.ui.workspace.timeline_body import body_geometry

                geometry = body_geometry(ui)
                padding = (*geometry.padding, *geometry.padding)
                box = Box(box.outer, box.frame, box.frame.inset(*padding), padding, box.margin)
            return self._record(part, box, engine, children)
        finally:
            self.runtime.paint_context.paint_depth -= 1

    def _record(self, part, box, engine, children=()):
        record = {
            "part": part,
            "outer": box.outer.tuple(),
            "frame": box.frame.tuple(),
            "content": box.content.tuple(),
            "children": list(children),
            "controls": list(engine.observed),
        }
        self.last[part] = record
        return record

    def _form(self, ui, engine, style, part, bounds):
        s = ui.s
        if part == "range_number":
            box = style.box(*bounds.tuple(), scale=s)
            edit = ui.rail(
                "reference-range-number",
                box.content.x,
                box.content.y,
                box.content.width,
                self.values.get(part, 1.25),
                (0, 10),
                initial=None,
                unit="m",
                show_reset=False,
            )
            self.values[part] = edit.value
            keys = ("track", "value", "unit")
            children = [ui.hits["reference-range-number-" + key] for key in keys]
            return self._record(part, box, engine, children)
        if part == "range_bounds":
            from mojive.ui.workspace.labeled_input import (
                draw_labeled_input,
                measure_labeled_input,
            )

            layout = measure_labeled_input(
                ui, bounds.width, "起点", (82 * s, engine.field_height("input_number") * s)
            )
            _, value = draw_labeled_input(
                ui,
                "reference-bound-value",
                bounds.x,
                bounds.y,
                layout,
                "起点",
                self.values.get(part, 0.0),
                format="%.3f s",
            )
            self.values[part] = value
            box = style.box(bounds.x, bounds.y, layout.width, layout.height, scale=s)
            children = [
                (bounds.x + r.x, bounds.y + r.y, r.width, r.height)
                for r in (layout.label, layout.field)
            ]
            return self._record(part, box, engine, children)
        if part == "time_field":
            from mojive.ui.workspace.time_readout import draw_time_readout

            box = style.box(*bounds.tuple(), scale=s)
            readout = draw_time_readout(
                ui,
                "reference-time",
                box.content.x,
                box.content.y,
                0.0,
                width=min(80, box.content.width / s),
            )
            return self._record(part, box, engine, (readout,))
        if part == "form_row":
            layout = engine.form_layout(bounds)
            box = style.box(*bounds.tuple(), scale=s)
            engine.label("reference-label", layout["label"], "质量 (kg)")
            result = engine.input_number(
                "reference-form-value", layout["fields"][0], self.values.get(part, 1.25)
            )
            self.values[part] = result.value
            return self._record(
                part, box, engine, [layout["label"].tuple(), result.box.frame.tuple()]
            )
        layout = engine.form_layout(bounds, 3, vector=True, axis=True)
        box = style.box(*bounds.tuple(), scale=s)
        engine.label("reference-label", layout["label"], "位置 (m)")
        for axis, rect in zip("XYZ", layout["fields"], strict=True):
            key = part + "/" + axis
            result = engine.input_number(
                "reference-axis-" + axis, rect, self.values.get(key, 1.25), axis=axis
            )
            self.values[key] = result.value
        return self._record(part, box, engine, [rect.tuple() for rect in layout["fields"]])

    def _native_color(self, ui, engine, style, part, bounds):
        from mojive.ui.workspace.color_editor import ColorEditor, ColorField, color_popup_geometry

        editor = self.color_editors.get(part)
        if editor is None:
            editor = self.color_editors[part] = (
                ColorField() if part == "color_input" else ColorEditor()
            )
        box = style.box(*bounds.tuple(), scale=ui.s)
        if part == "color_editor":
            padding = color_popup_geometry(ui).padding
            box = Box(box.outer, box.frame, box.frame.inset(*padding), padding, box.margin)
        values = self.values.get(part, (0.55, 0.68, 0.80, 1.0))
        key = "reference-" + part
        if part == "color_input":
            imgui.set_cursor_screen_pos((box.content.x, box.content.y))
            _, self.values[part] = editor.draw(ui, key, values, width=box.content.width)
        else:
            # The production editor consumes its own margins and padding. Keep
            # the reference box for observation without insetting its input twice.
            imgui.set_cursor_screen_pos((box.outer.x, box.outer.y))
            _, self.values[part] = editor.draw(ui, key, values, box.outer.width)
            padding = color_popup_geometry(ui).padding
            self.color_heights[part, bounds.width] = (
                imgui.get_cursor_screen_pos().y - bounds.y + padding[3]
            )
        children = [hit for name, hit in ui.hits.items() if name.startswith(key)]
        return self._record(part, box, engine, children)
