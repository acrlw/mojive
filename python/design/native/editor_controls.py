"""Compact native parameter controls and editing gestures."""

from contextlib import contextmanager

from mojive.design.editor import MISSING, path_value
from mojive.design.specification import DEFAULT
from mojive.ui.design_palette import rgba


class EditorControls:
    def choose(self, workspace, key, current, choices, *, labels=None):
        """Draw one mutually exclusive editor choice group on a common frame."""
        from imgui_bundle import imgui as g

        labels = labels or {}
        s = workspace.s
        from mojive.ui.controls import segmented_control
        from mojive.ui.workspace.widget_style import CONTROL_THEME

        g.push_style_var(g.StyleVar_.frame_padding, (3 * s, 4 * s))
        g.push_style_var(g.StyleVar_.frame_rounding, 3 * s)
        current = choices[
            segmented_control(
                "wb-" + key,
                tuple(labels.get(value, value) for value in choices),
                choices.index(current),
                width=g.get_content_region_avail().x,
                theme=CONTROL_THEME,
                on_item=lambda index: self.record_item(key + "/" + choices[index]),
            )
        ]
        g.pop_style_var(2)
        self.record_item(key)
        return current

    def record_item(self, key):
        from imgui_bundle import imgui

        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        role, _, path = key.partition("/")
        if (
            role in ("label", "value", "slider")
            and path.split("/", 1)[0] in ("parts", "tokens", "components", "layout", "native")
            and (
                imgui.is_item_activated()
                or (
                    imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
                    and imgui.is_mouse_clicked(0)
                )
            )
        ):
            self.selected_field = tuple(path.split("/"))

    @contextmanager
    def editor_scope(self, workspace):
        from imgui_bundle import imgui as g

        from mojive.design.native.geometry import CURRENT_NATIVE_PARTS

        parts = CURRENT_NATIVE_PARTS.set(())
        engine = self.runtime.paint_context
        engine.paint_depth += 1
        context = g.get_current_context()
        # The design editor stays usable while the actual scene previews a
        # popup. Restore all preview flags before submitting scene controls.
        popup_flags = [
            (data.window, data.window.flags)
            for data in context.open_popup_stack
            if data.window is not None and not self._editor_window(data.window)
        ]
        for window, flags in popup_flags:
            window.flags = int(flags) & ~int(g.WindowFlags_.modal | g.WindowFlags_.popup)
        if popup_flags:
            g.internal.update_hovered_window_and_capture_flags(g.get_io().mouse_pos)
        s = workspace.s
        g.push_font(workspace.ui.body, 14 * s)
        for key, value in (
            (g.StyleVar_.window_padding, (18 * s, 16 * s)),
            (g.StyleVar_.frame_padding, (8 * s, 7 * s)),
            (g.StyleVar_.item_spacing, (8 * s, 8 * s)),
            (g.StyleVar_.frame_rounding, 5 * s),
            (g.StyleVar_.selectable_text_align, (0, 0.5)),
            (g.StyleVar_.disabled_alpha, 0.65),
        ):
            g.push_style_var(key, value)
        colors = (
            (g.Col_.text, rgba("text")),
            (g.Col_.window_bg, rgba("background")),
            (g.Col_.frame_bg, rgba("field")),
            (g.Col_.button, rgba("field")),
            (g.Col_.button_hovered, rgba("field_hovered")),
            (g.Col_.button_active, rgba("field_active")),
            (g.Col_.frame_bg_hovered, rgba("field_hovered")),
            (g.Col_.frame_bg_active, rgba("field_active")),
            (g.Col_.header, (*rgba("header")[:3], 0.45)),
            (g.Col_.header_hovered, rgba("field_hovered")),
            (g.Col_.header_active, rgba("field_active")),
            (g.Col_.separator, (*rgba("border")[:3], 0.35)),
            (g.Col_.check_mark, rgba("primary")),
            (g.Col_.slider_grab, rgba("primary")),
            (g.Col_.slider_grab_active, rgba("primary_bright")),
        )
        for key, value in colors:
            g.push_style_color(key, value)
        try:
            yield
        finally:
            g.pop_style_color(len(colors))
            g.pop_style_var(6)
            g.pop_font()
            for window, flags in popup_flags:
                window.flags = flags
            if popup_flags:
                g.internal.update_hovered_window_and_capture_flags(g.get_io().mouse_pos)
            engine.paint_depth -= 1
            CURRENT_NATIVE_PARTS.reset(parts)

    @staticmethod
    def _editor_window(window):
        current = window
        while current is not None:
            if current.name.startswith("PythonDesignWorkbench"):
                return True
            current = current.parent_window_in_begin_stack
        return False

    def tooltip(self, text, *, shown=None):
        from imgui_bundle import imgui as g

        text = str(text).splitlines()[0] if text else ""
        if shown is None or not text or shown == text:
            return
        if g.is_item_hovered(g.HoveredFlags_.allow_when_disabled) or (
            g.get_io().nav_visible and g.is_item_focused()
        ):
            g.set_tooltip(text)

    def button(self, label, key, *, enabled=True, tooltip=""):
        from imgui_bundle import imgui as g

        g.begin_disabled(not enabled)
        clicked = g.button(label + "##wb-" + key)
        self.record_item(key)
        if label == "?":
            self.tooltip("参数说明", shown="?")
        g.end_disabled()
        return clicked

    @staticmethod
    def next_on_row(width):
        """Keep paired controls on one row only when the content column fits."""
        from imgui_bundle import imgui as g

        right = g.internal.get_current_window().work_rect.max.x
        if g.get_item_rect_max().x + g.get_style().item_spacing.x + width <= right + 0.1:
            g.same_line()

    def next_button_on_row(self, label):
        from imgui_bundle import imgui as g

        self.next_on_row(g.calc_text_size(label).x + 2 * g.get_style().frame_padding.x)

    def next_checkbox_on_row(self, label):
        from imgui_bundle import imgui as g

        self.next_on_row(
            g.get_frame_height() + g.get_style().item_inner_spacing.x + g.calc_text_size(label).x
        )

    def combo(self, label, current, values, labels=None, *, key=None, width=-1):
        from imgui_bundle import imgui as g

        labels = labels or {}

        def shown(v):
            return labels.get(str(v), labels.get(v, str(v)))

        changed = False
        g.set_next_item_width(width)
        g.set_next_window_size_constraints((0, 0), (800, 360))
        opened = g.begin_combo(label, shown(current))
        if key:
            self.record_item(key)
        text_width = max(
            0, g.get_item_rect_size().x - g.get_frame_height() - 2 * g.get_style().frame_padding.x
        )
        if opened:
            try:
                for value in values:
                    clicked, _ = g.selectable(shown(value) + "##" + str(value), current == value)
                    if current == value:
                        g.set_item_default_focus()
                    if clicked:
                        changed, current = True, value
            finally:
                g.end_combo()
        self.tooltip(shown(current), shown=self.fit_label(shown(current), text_width))
        return changed, current

    def change(self, path, value, *, gesture=False):
        try:
            if self.editor.change(path, value, gesture=gesture):
                self.actions.append(
                    {
                        "path": list(path),
                        "value": None if value is MISSING else value,
                        "removed": value is MISSING,
                        "revisionBefore": self.runtime.revision,
                    }
                )
                self.runtime.poll(force=True)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self.editor.error = str(exc)

    @staticmethod
    def pixel_field(path, rule, help):
        if rule.get("type", "number") != "number":
            return False
        if path[0] == "native" or path[-1] in ("columns", "pickerShare"):
            return False
        if path[0] in ("tokens", "layout"):
            return True
        return help.get("unit") == "像素"

    def field(self, workspace, path, rule, value, *, help=None, condition=None):
        from imgui_bundle import imgui as g

        help = help or {}
        condition = condition or {"enabled": True, "reason": "", "enable": ""}
        enabled = condition["enabled"]
        if path[0] == "parts" and path[1] == "workspace_presets":
            enabled = False
            condition = {"enabled": False, "reason": "当前主界面已移除旧布局预设。", "enable": ""}
        key, s = "/".join(path), workspace.s
        kind = rule.get("type", "number")
        short = {
            "rowGap": "行间距",
            "columnGap": "列间距",
            "gap": "间距",
            "fontSize": "字号",
            "iconSize": "图标大小",
            "alignItems": "纵向对齐",
            "textAlign": "文字对齐",
            "paddingX": "左右内距",
            "paddingY": "上下内距",
            "radius": "圆角",
        }
        caption = {
            ("select_option", "gap"): "图标与文字间距",
            ("select_popup", "rowGap"): "条目垂直间距",
            ("statusbar", "columnGap"): "状态组间距",
            ("status_context", "columnGap"): "状态与提示间距",
            ("status_hints", "columnGap"): "提示条目间距",
            ("status_metrics", "columnGap"): "指标条目间距",
            ("output_row", "columnGap"): "图标与正文间距",
            ("output_row", "gap"): "基础图文间距",
            ("output_list", "rowGap"): "条目间距",
            ("output_list", "gap"): "基础条目间距",
        }.get(tuple(path[1:]), help.get("shortLabel") or short.get(path[-1], rule["label"]))
        if path[0] == "parts" and path[1] in ("tree_row", "lane_label") and path[-1] == "radius":
            caption = "选中圆角"
        explicit = path_value(self.editor.spec, path) is not MISSING
        reset = MISSING
        if path[0] in ("tokens", "native", "layout"):
            reset = DEFAULT[path[0]][path[-1]]
            explicit = value != reset
        source = help.get("defaultHelp", "")
        from mojive.design.support import native_support

        support = (
            native_support()[path[1]]["fields"][path[2]]
            if path[0] == "parts" and len(path) == 3
            else None
        )
        if support and support["status"] == "reference":
            caption += " · 样例"
        full_help = "\n".join(
            filter(
                None,
                (
                    rule["label"],
                    support["description"] if support else "",
                    help.get("summary") or rule.get("description"),
                    help.get("reference"),
                    source,
                    condition.get("reason") if not enabled else "",
                    condition.get("enable") if not enabled else "",
                ),
            )
        )
        g.push_id(key)
        x, available = g.get_cursor_pos_x(), g.get_content_region_avail().x
        gap = g.get_style().item_spacing.x
        reset_width = info_width = g.get_frame_height()
        label_width = min(160 * s, available * 0.42)
        stacked = g.calc_text_size(caption).x > label_width
        control_width = max(
            1,
            available - (0 if stacked else label_width + gap) - reset_width - info_width - 2 * gap,
        )
        selected = getattr(self, "selected_field", None) == tuple(path)
        if g.selectable(
            caption + "##field-label",
            selected,
            size=(available if stacked else label_width, g.get_frame_height()),
        )[0]:
            self.selected_field = tuple(path)
        self.record_item("label/" + key)
        self.tooltip(rule["label"], shown=self.fit_label(caption, label_width))
        if not stacked:
            g.same_line(x + label_width + gap)
        g.begin_disabled(not enabled)
        gesture_ended = False
        if kind == "number":
            step = rule.get("step") or 1
            fmt = "%.0f" if step >= 1 else ("%.1f" if step >= 0.1 else "%.2f")
            g.set_next_item_width(control_width)
            entered, manual = g.input_float(
                "##value", float(value), 0, 0, fmt, g.InputTextFlags_.enter_returns_true
            )
            self.record_item("value/" + key)
        elif kind == "select":
            g.set_next_item_width(control_width)
            entered, manual = self.combo(
                "##value",
                value,
                rule["choices"],
                rule.get("choiceLabels"),
                key="value/" + key,
                width=control_width,
            )
        elif kind == "color":
            channels = [int(value[n : n + 2], 16) / 255 for n in range(1, len(value), 2)]
            if len(channels) == 3:
                channels.append(1)
            g.set_next_item_width(control_width)
            entered, channels = g.color_edit4(
                "##value", channels, g.ColorEditFlags_.no_inputs | g.ColorEditFlags_.no_tooltip
            )
            manual = "#" + "".join(f"{round(c * 255):02x}" for c in channels)
            self.record_item("value/" + key)
        else:
            entered, manual = g.checkbox("##value", value)
            self.record_item("value/" + key)
        self.tooltip(rule["label"], shown=self.fit_label(caption, label_width))
        g.end_disabled()
        g.same_line(x + available - reset_width - info_width - gap)
        if self.tool_button(
            workspace, "reset", "reset/" + key, source or "恢复默认值", enabled=explicit
        ):
            # Selecting on press can remove the preceding field's slider and
            # move this reset button before release, canceling its own click.
            self.selected_field = tuple(path)
            self.change(path, reset)
        g.same_line(x + available - info_width)
        if self.tool_button(workspace, "info", "help/" + key, "参数说明"):
            g.open_popup("字段说明")
        max_width = max(1.0, min(400 * s, g.get_io().display_size.x - 40 * s))
        g.set_next_window_size_constraints(
            (min(220 * s, max_width), 0), (max_width, max(1.0, g.get_io().display_size.y - 40 * s))
        )
        if g.begin_popup("字段说明"):
            # An auto-sized popup has no stable available width on its first frame.
            wrap_width = max(1.0, min(340 * s, max_width - 2 * g.get_style().window_padding.x))
            g.push_text_wrap_pos(g.get_cursor_pos_x() + wrap_width)
            g.text_wrapped(full_help)
            g.pop_text_wrap_pos()
            g.end_popup()
        if entered:
            result = manual
            if kind == "number":
                result = max(rule["min"], min(rule["max"], round(manual / step) * step))
                if step >= 1:
                    result = int(result)
            self.change(path, result, gesture=kind == "color")
        if kind == "number" and getattr(self, "selected_field", None) == tuple(path):
            g.begin_disabled(not enabled)
            g.set_next_item_width(available)
            changed, number = g.slider_float(
                "##slider", float(value), rule["min"], rule["max"], fmt, g.SliderFlags_.always_clamp
            )
            self.record_item("slider/" + key)
            gesture_active = g.is_item_active()
            gesture_ended = g.is_item_deactivated_after_edit()
            if changed:
                quantum = (
                    max(step, self.snap_step)
                    if self.snap and self.pixel_field(path, rule, help)
                    else step
                )
                result = max(rule["min"], min(rule["max"], round(number / quantum) * quantum))
                if step >= 1:
                    result = int(result)
                self.change(path, result, gesture=gesture_active)
            g.end_disabled()
            if gesture_ended or g.is_item_deactivated_after_edit():
                try:
                    self.editor.end_gesture()
                except OSError as error:
                    self.editor.error = str(error)
        if not enabled and path[0] == "parts" and rule.get("condition"):
            requirement = rule["condition"]
            parent = self.editor.catalog[path[1]]["fields"][requirement["field"]]
            shown = parent.get("choiceLabels", {}).get(
                str(requirement["value"]), str(requirement["value"])
            )
            if self.button("启用：" + shown, "enable/" + key, tooltip=full_help):
                self.change(("parts", path[1], requirement["field"]), requirement["value"])
        note = condition.get("enable") or condition.get("reason") if not enabled else ""
        if support:
            note = note or support.get("condition", "")
            if support["status"] == "reference":
                note = support["description"]
        typography = getattr(self, "preview_typography", None)
        if (
            path[0] == "parts"
            and path[-1] == "fontSize"
            and typography is not None
            and typography.size_for((path[1],)) is not None
        ):
            note = "字号由设置覆盖；切回“设计规范默认值”后生效。"
        if note and (not enabled or getattr(self, "selected_field", None) == tuple(path)):
            g.text_disabled(self.fit_label(note.splitlines()[0], available))
            self.tooltip(note, shown=self.fit_label(note.splitlines()[0], available))
        g.pop_id()

    @staticmethod
    def fit_label(label, width):
        from imgui_bundle import imgui as g

        if g.calc_text_size(label).x <= width:
            return label
        count = len(label)
        while count > 0 and g.calc_text_size(label[:count] + "…").x > width:
            count -= 1
        return label[:count] + "…"
