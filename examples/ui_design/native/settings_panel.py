"""Reference settings navigation around the editor's shared settings actions."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.ui.panels.settings import (
    SettingsPanel,
    render_flag_label,
    responsive_flag_groups,
)

from .widgets import SAGE, TEXT

CATEGORIES = ("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals")
CATEGORY_ICONS = ("settings", "camera", "mouse", "shading", "video", "layers")
_CHECKBOX_PROPERTIES = (
    "Copy to clipboard",
    "Run simulation when recording starts",
    "Movable capsules",
    "Keep mode/unit",
    "Camera & light icons",
    "Selected frustum / light range",
    "Show contact points",
    "Use model color",
)


class ReferenceSettings(SettingsPanel):
    def __init__(self, study):
        super().__init__()
        self.study = study
        self._checkbox_labels = frozenset()

    def _control_padding(self, *, checkbox=False):
        # Shared settings actions size checkboxes from the current frame height.
        s = self.study.s
        height = 18 if checkbox else 28
        imgui.get_style().frame_padding = (
            8 * s,
            max(0, (height * s - imgui.get_font_size()) / 2),
        )

    def _begin_properties(self, str_id):
        self._control_padding()
        imgui.get_style().cell_padding = (0, 2 * self.study.s)
        width = imgui.get_content_region_avail().x
        compact = width < 350 * self.study.s
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table(str_id, 1 if compact else 2, flags):
            return False
        if not compact:
            imgui.table_setup_column(
                "label", imgui.TableColumnFlags_.width_fixed, min(184 * self.study.s, width * 0.42)
            )
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
        return True

    def _property(self, label):
        self._control_padding()
        s = self.study.s
        compact = imgui.table_get_column_count() == 1
        imgui.table_next_row(0, 0 if compact else 32 * s)
        imgui.table_next_column()
        if not compact:
            imgui.align_text_to_frame_padding()
        imgui.push_text_wrap_pos(0)
        imgui.text_disabled(label)
        imgui.pop_text_wrap_pos()
        imgui.table_next_column()
        if label in self._checkbox_labels:
            self._control_padding(checkbox=True)
            if not compact:
                imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 5 * s)
        imgui.set_next_item_width(-1)

    def _group_heading(self, title):
        self._control_padding()
        u, s = self.study.ui, self.study.s
        imgui.dummy((0, 12 * s))
        p = imgui.get_cursor_screen_pos()
        u.text(p.x, p.y, title, TEXT, 13, bold=True)
        u.line(p.x, p.y + 23 * s, imgui.get_content_region_avail().x)
        imgui.dummy((0, 30 * s))

    def _begin_toggle_grid(self, str_id, labels):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, 6 * self.study.s)
        width = imgui.get_content_region_avail().x
        required = max((imgui.calc_text_size(label).x for label in labels), default=0)
        required += 30 * self.study.s
        columns = max(1, min(2, int(width / max(1, required))))
        return imgui.begin_table(str_id, columns, imgui.TableFlags_.sizing_stretch_same)

    def _flag_table(self, ctx, table_id, flags, groups=2, *, translate_labels=True):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, 2 * self.study.s)
        groups = responsive_flag_groups(groups, imgui.get_content_region_avail().x, ctx.style_scale)
        if not imgui.begin_table(table_id, groups * 2, imgui.TableFlags_.sizing_stretch_prop):
            return
        for group in range(groups):
            imgui.table_setup_column(f"label {group}", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column(
                f"value {group}", imgui.TableColumnFlags_.width_fixed, 30 * ctx.style_scale
            )
        row_count = (len(flags) + groups - 1) // groups
        for row in range(row_count):
            imgui.table_next_row(0, 30 * ctx.style_scale)
            for group in range(groups):
                index = group * row_count + row
                imgui.table_next_column()
                if index >= len(flags):
                    imgui.table_next_column()
                    continue
                flag = flags[index]
                label = render_flag_label(flag, ctx.tr, localized=translate_labels)
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(label)
                imgui.table_next_column()
                self._flag_row(ctx, flag, label)
        imgui.end_table()
        self._control_padding()

    def _visual_groups(self, ctx):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, 6 * self.study.s)
        super()._visual_groups(ctx)
        self._control_padding()

    def _interaction(self, ctx):
        self._interaction_policy(ctx)
        self._selection_presentation(ctx)
        self._gizmo_settings(ctx)
        self._precise_input_settings(ctx)
        self._snap_settings(ctx)
        self._view_settings(ctx)
        self._perturb_settings(ctx)
        self._helper_settings(ctx)
        self._shortcut_settings(ctx)

    def _general(self, ctx):
        super()._general(replace(ctx, font_report=None))

    def draw(self, ctx):
        u, s = self.study.ui, self.study.s
        p = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail()
        width = available.x
        u.icon("settings", p.x, p.y + 1 * s, 20, SAGE)
        u.text(p.x + 30 * s, p.y, "Settings", TEXT, 18, bold=True)
        y = p.y + 48 * s
        narrow = width < 560 * s
        if narrow:
            columns = 3 if width >= 360 * s else 2
            cell_width = width / columns
            for n, category in enumerate(CATEGORIES):
                if u.button(
                    "settings-" + category,
                    p.x + n % columns * cell_width,
                    y + n // columns * 34 * s,
                    cell_width / s - 4,
                    30,
                    label=category,
                    align="center",
                    active=self._category == category,
                ):
                    self._category = category
            y += (len(CATEGORIES) // columns * 34 + 18) * s
            x, page_width = p.x, width
        else:
            imgui.set_cursor_screen_pos((p.x, y))
            imgui.push_style_color(imgui.Col_.child_bg, (26 / 255, 29 / 255, 32 / 255, 1))
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6 * s)
            imgui.begin_child("Reference settings navigation", (144 * s, 0))
            nav = imgui.get_cursor_screen_pos()
            for n, (category, icon) in enumerate(zip(CATEGORIES, CATEGORY_ICONS, strict=True)):
                if u.button(
                    "settings-" + category,
                    nav.x + 8 * s,
                    nav.y + (8 + n * 38) * s,
                    128,
                    32,
                    icon=icon,
                    label=category,
                    active=self._category == category,
                ):
                    self._category = category
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            x, page_width = p.x + 164 * s, width - 164 * s
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_child("Reference settings content", (page_width, max(s, available.y - y + p.y)))
        page = imgui.get_cursor_screen_pos()
        u.text(page.x, page.y, self._category, TEXT, 18, bold=True)
        u.line(page.x, page.y + 32 * s, page_width)
        imgui.set_cursor_screen_pos((page.x, page.y + 44 * s))
        self._checkbox_labels = frozenset(ctx.tr(label) for label in _CHECKBOX_PROPERTIES)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (8 * s, 7 * s))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, 4 * s))
        imgui.push_style_var(imgui.StyleVar_.cell_padding, (0, 2 * s))
        try:
            {
                "General": self._general,
                "Camera": self._camera_navigation,
                "Interaction": self._interaction,
                "Rendering": self._rendering,
                "Recording": self._recording,
                "MuJoCo Visuals": self._mujoco_visuals,
            }[self._category](ctx)
        finally:
            imgui.pop_style_var(3)
            imgui.end_child()
