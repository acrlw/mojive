"""Reference settings navigation around the editor's shared settings actions."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.ui.panels import padded_selectable, themed_checkbox
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

    def _property(self, label, *, control_part="input_number"):
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
        if control_part == "checkbox" or label in self._checkbox_labels:
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

    def _general(self, ctx):
        super()._general(replace(ctx, font_report=None, panels=None))
        if ctx.panels is None:
            return
        from .docking_study import BOTTOM_PANELS, LEFT_PANELS

        study = self.study
        self._group_heading(ctx.tr("Panels"))
        if not self._begin_properties("reference_settings_panels"):
            return
        self._property(ctx.tr("Sidebar"))
        sidebar = study.left_active if study.open[study.left_active] else None
        names = tuple(name for name in LEFT_PANELS if ctx.panels.get(name).enabled)
        changed, chosen = self._panel_choice(ctx, "settings-sidebar", names, sidebar)
        if changed:
            if chosen is None:
                for name in LEFT_PANELS:
                    ctx.panels.set_open(name, False)
            else:
                study.activate(chosen)
        self._property(ctx.tr("Bottom panel"))
        bottom = study.bottom_active if study.bottom_open else None
        names = tuple(name for name in BOTTOM_PANELS if ctx.panels.get(name).enabled)
        changed, chosen = self._panel_choice(ctx, "settings-bottom", names, bottom)
        if changed:
            if chosen is None:
                study.collapse_bottom()
            else:
                study.activate(chosen)
        self._property(ctx.tr("Inspector"))
        self._control_padding(checkbox=True)
        changed, opened = themed_checkbox(
            "##settings_inspector", study.open["Inspector"], ctx.theme
        )
        self._record_control("settings-inspector")
        if changed:
            ctx.panels.set_open("Inspector", opened)
        imgui.end_table()

    def _record_control(self, key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.study.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def _panel_choice(self, ctx, key, names, current):
        opened = imgui.begin_combo("##" + key, ctx.tr(current or "none"))
        self._record_control(key)
        chosen = current
        changed = False
        if opened:
            for name in (None, *names):
                clicked, _ = padded_selectable(ctx.tr(name or "none"), name == current)
                self._record_control(key + "-option-" + (name or "none"))
                if clicked:
                    chosen, changed = name, True
            imgui.end_combo()
        return changed, chosen

    def _header(self, x, y, width):
        u, s = self.study.ui, self.study.s
        u.role_icon("settings", x, y + 2 * s, 20 * s, 28 * s, role="title", c=SAGE)
        u.text(x + 28 * s, y + 8.5 * s, "Settings", TEXT, 15, bold=True)
        u.hits["settings-title"] = (
            x + 28 * s,
            y + 8.5 * s,
            u.measure("Settings", 15),
            15 * s,
        )
        closed = u.button(
            "settings-close", x + width - 28 * s, y + 2 * s, 28, 28, icon="close", tooltip="Close"
        )
        u.line(x, y + 32 * s, width)
        if closed:
            self.study.open["Settings"] = self.open = False
        return closed

    def draw(self, ctx):
        u, s = self.study.ui, self.study.s
        p = imgui.get_cursor_screen_pos()
        available = imgui.get_content_region_avail()
        width = available.x
        if self._header(p.x, p.y, width):
            return
        y = p.y + 44 * s
        narrow = width < 560 * s
        if narrow:
            minimum = max(u.measure(ctx.tr(category)) for category in CATEGORIES) + 16 * s
            columns = max(1, min(3, int(width / minimum)))
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
            button_width = max(
                128 * s, max(u.measure(ctx.tr(category)) for category in CATEGORIES) + 39 * s
            )
            nav_width = button_width + 16 * s
            imgui.set_cursor_screen_pos((p.x, y))
            imgui.push_style_color(imgui.Col_.child_bg, (26 / 255, 29 / 255, 32 / 255, 1))
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6 * s)
            imgui.begin_child("Reference settings navigation", (nav_width, 0))
            nav = imgui.get_cursor_screen_pos()
            for n, (category, icon) in enumerate(zip(CATEGORIES, CATEGORY_ICONS, strict=True)):
                if u.button(
                    "settings-" + category,
                    nav.x + 8 * s,
                    nav.y + (8 + n * 38) * s,
                    button_width / s,
                    32,
                    icon=icon,
                    label=category,
                    align="left",
                    active=self._category == category,
                ):
                    self._category = category
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            x, page_width = p.x + nav_width + 16 * s, width - nav_width - 16 * s
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_child("Reference settings content", (page_width, max(s, available.y - y + p.y)))
        page = imgui.get_cursor_screen_pos()
        u.text(page.x, page.y, self._category, TEXT, 15, bold=True)
        u.line(page.x, page.y + 32 * s, page_width)
        imgui.dummy((0, 44 * s - imgui.get_style().item_spacing.y))
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
