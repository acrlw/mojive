"""Keep workbench navigation and document actions separate from parameters."""

from imgui_bundle import imgui as g

from mojive.ui.design_palette import rgba


class EditorChrome:
    def tool_button(self, workspace, icon, key, caption, *, enabled=True, active=False):
        s = workspace.s
        side = g.get_frame_height()
        pos = g.get_cursor_screen_pos()
        g.begin_disabled(not enabled)
        clicked = g.invisible_button("##tool-" + key, (side, side))
        self.record_item(key)
        hovered = g.is_item_hovered(g.HoveredFlags_.allow_when_disabled)
        draw = g.get_window_draw_list()
        if active or (hovered and enabled):
            color = "field_active" if g.is_item_active() else "field_hovered"
            draw.add_rect_filled(
                pos, (pos.x + side, pos.y + side), g.get_color_u32(rgba(color)), 5 * s
            )
        ink = rgba("text" if enabled and hovered else "secondary")
        if not enabled:
            ink = (*ink[:3], 0.35)
        workspace.ui.icon(
            icon,
            pos.x + (side - 16 * s) / 2,
            pos.y + (side - 16 * s) / 2,
            16,
            tuple(round(c * 255) for c in ink),
            draw,
        )
        self.tooltip(caption, shown=icon)
        g.end_disabled()
        return clicked

    def draw_header(self, workspace):
        s = workspace.s
        left = g.get_cursor_pos_x()
        available = g.get_content_region_avail().x
        mode_label = "合并窗口" if self.detached else "分离窗口"
        width = g.calc_text_size(mode_label).x + 2 * g.get_style().frame_padding.x
        g.push_font(workspace.ui.body, 20 * s)
        g.text_unformatted("设计工作台")
        g.pop_font()
        g.same_line(left + available - width)
        if self.button(mode_label, "window-mode"):
            self.requested_detached = not self.detached

        pending = self.editor.spec != self.runtime.spec
        dirty = self.editor.spec != self.editor.saved_spec
        status = "应用中…" if pending else "未保存" if dirty else "已保存"
        g.align_text_to_frame_padding()
        g.text_colored(rgba("warning" if pending or dirty else "success"), status)
        actions_width = (
            g.calc_text_size("保存").x
            + 2 * g.get_style().frame_padding.x
            + 4 * g.get_frame_height()
            + 4 * g.get_style().item_spacing.x
        )
        g.same_line(max(left + 65 * s, left + available - actions_width))
        g.push_style_color(g.Col_.button, rgba("primary_dim"))
        if self.button("保存", "save"):
            self.editor_action(self.editor.save)
        g.pop_style_color()
        g.same_line()
        if self.tool_button(
            workspace, "undo", "undo", "撤销", enabled=bool(self.editor.undo_stack)
        ):
            self.editor_action(self.editor.undo)
        g.same_line()
        if self.tool_button(
            workspace, "redo", "redo", "重做", enabled=bool(self.editor.redo_stack)
        ):
            self.editor_action(self.editor.redo)
        g.same_line()
        if self.tool_button(workspace, "settings", "observation", "观察与吸附设置"):
            g.open_popup("观察设置")
        g.same_line()
        if self.tool_button(workspace, "more", "document-actions", "规范文件与导入、导出"):
            g.open_popup("规范文件")
        self.draw_observation(workspace)
        self.draw_document_actions(workspace)
        if self.watcher.error or self.editor.error:
            g.text_colored(rgba("danger"), self.editor.error or self.watcher.error)
        g.spacing()

    def draw_observation(self, workspace):
        if not g.begin_popup("观察设置"):
            return
        try:
            g.text_unformatted("观察设置")
            g.separator()
            _, self.show_bounds = g.checkbox("边界与中心线", self.show_bounds)
            self.record_item("show-bounds")
            _, self.pick = g.checkbox("Alt 点击选部件", self.pick)
            self.record_item("pick")
            g.spacing()
            g.separator()
            _, self.snap = g.checkbox("像素吸附", self.snap)
            self.record_item("snap-enabled")
            g.same_line()
            g.set_next_item_width(105 * workspace.s)
            _, index = g.combo(
                "##snap", (1, 2, 4, 8).index(self.snap_step), ["1 px", "2 px", "4 px", "8 px"]
            )
            self.snap_step = (1, 2, 4, 8)[index]
            self.record_item("snap-step")
        finally:
            g.end_popup()

    def draw_document_actions(self, workspace):
        s = workspace.s
        g.set_next_window_size_constraints(
            (min(360 * s, g.get_io().display_size.x), 0), (520 * s, 600 * s)
        )
        if not g.begin_popup("规范文件"):
            return
        try:
            g.text_unformatted("规范文件")
            g.separator()
            g.text_disabled("导入 / 导出路径")
            g.set_next_item_width(-1)
            _, self.file_path = g.input_text("##file", self.file_path)
            self.record_item("file-path")
            for label, key, action in (
                ("导出", "export", self.editor.export),
                ("导入", "import", self.editor.import_file),
            ):
                if key == "import":
                    g.same_line()
                if self.button(label, key):
                    try:
                        action(self.file_path)
                        self.runtime.poll(force=True)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        self.editor.error = str(error)
            g.spacing()
            if g.collapsing_header("存储位置"):
                g.push_text_wrap_pos(g.get_cursor_pos_x() + 330 * s)
                g.text_disabled("已保存")
                g.text_wrapped(str(self.editor.saved_path))
                g.text_disabled("实时预览")
                g.text_wrapped(str(self.editor.live_path))
                g.pop_text_wrap_pos()
            overrides = self.editor.overrides()
            if g.collapsing_header(f"部件覆盖 · {len(overrides)} 项"):
                g.begin_child("覆盖记录", (0, 180 * s))
                for record in overrides:
                    g.text_wrapped(f"{record['label']}：{record['value']}")
                g.end_child()
        finally:
            g.end_popup()

    def draw_footer(self, workspace):
        g.separator()
        selected = self.selected_field
        label = None
        if selected and selected[0] == "parts":
            label = (
                self.editor.catalog.get(selected[1], {})
                .get("fields", {})
                .get(selected[-1], {})
                .get("label")
            )
            if selected[1:] == ("output_list", "rowGap"):
                label = "条目间距"
        if selected and selected[0] == "native":
            from mojive.design.specification import SCHEMA

            label = SCHEMA["native"][selected[-1]]["label"]
        if label:
            g.text_colored(
                rgba("warning"),
                self.fit_label("正在调整 · " + label, g.get_content_region_avail().x),
            )
        else:
            g.text_disabled("实时预览 · Alt + 点击选取部件" if self.pick else "实时预览")
