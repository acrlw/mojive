"""Reference settings navigation around the editor's shared settings actions."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.ui.panels import padded_selectable, themed_checkbox
from mojive.ui.panels.settings import (
    SettingsPanel,
    render_flag_label,
    responsive_flag_groups,
)

from .design import DESIGN, TITLE_SIZE
from .property_style import NativePropertyPresentation, advance, property_controls, property_section
from .widgets import SAGE, TEXT, NumericEditState

CATEGORIES = ("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals")
CATEGORY_ICONS = ("settings", "camera", "mouse", "shading", "video", "layers")
_CHECKBOX_PROPERTIES = (
    "Inspector",
    "Copy to clipboard",
    "Run simulation when recording starts",
    "Movable capsules",
    "Keep mode/unit",
    "Camera & light icons",
    "Selected frustum / light range",
    "Show contact points",
    "Use model color",
)


class WorkspaceSettings(SettingsPanel):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self._checkbox_labels = frozenset()
        self.fields = NativePropertyPresentation(workspace.ui, prefix="settings")
        self._number_origin = None
        self._number_committed = False

    def _control_padding(self, *, checkbox=False):
        # Shared settings actions size checkboxes from the current frame height.
        s = self.workspace.s
        height = DESIGN.checkbox_size if checkbox else DESIGN.control_height
        imgui.get_style().frame_padding = (
            DESIGN.frame_padding_x * s,
            max(0, (height * s - imgui.get_font_size()) / 2),
        )

    def _begin_properties(self, str_id):
        self._control_padding()
        imgui.get_style().cell_padding = (0, DESIGN.row_gap * self.workspace.s / 2)
        return self.fields.begin_table(str_id)

    def _property(self, label):
        self._control_padding()
        self.fields.row(label)
        if label in self._checkbox_labels:
            self._control_padding(checkbox=True)
            offset = (DESIGN.control_height - DESIGN.checkbox_size) * self.workspace.s / 2
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + offset)

    def _group_heading(self, title):
        self._control_padding()
        ui = self.workspace.ui
        p = imgui.get_cursor_screen_pos()
        ui.section(title, p.x, p.y, imgui.get_content_region_avail().x)
        advance(ui, p.x, p.y + 38 * ui.s)

    def _collapsing_header(self, label, flags=0):
        # The section painter owns its geometry. Preserve the following fields' padding.
        return property_section(self.workspace.ui, label.split("##", 1)[0], flags, item_id=label)

    def _drag_float(self, key, value, speed, low, high, fmt="%.3f", flags=0):
        changed, result = self.fields.numeric(key, value, speed, low, high, fmt)
        return self._finish_number(
            key, value, changed, result, self.workspace.ui.numeric_edits[key]
        )

    def _drag_int(self, key, value, speed, low, high, fmt="%d", flags=0):
        changed, result = self._drag_float(key, value, speed, low, high, "%.0f", flags)
        return changed, round(result)

    def _input_float(self, key, value, step=0, fast_step=0, fmt="%.3f", flags=0):
        return self._drag_float(key, value, step or 0.01, 0, 0, fmt)

    def _slider_float(self, key, value, low, high, fmt="%.3f", flags=0):
        edit = self.fields.value_rail(
            key, value, (low, high), initial=None, fmt=fmt, show_reset=False
        )
        active = imgui.is_item_active()
        deactivated = imgui.is_item_deactivated()
        state = NumericEditState(
            activated=edit.activated,
            active=active,
            deactivated=deactivated,
            committed=imgui.is_item_deactivated_after_edit(),
            canceled=(active or deactivated) and imgui.is_key_pressed(imgui.Key.escape, False),
        )
        return self._finish_number(key, value, edit.changed, edit.value, state)

    def _finish_number(self, key, value, changed, result, state):
        """Restore a canceled preview and keep it out of persistent preferences."""
        if state.activated:
            self._number_origin = key, value
        self._number_committed = state.committed and not state.canceled
        if self._number_origin is not None and self._number_origin[0] == key:
            if state.canceled:
                result = self._number_origin[1]
                changed = result != value
                self._number_origin = None
                imgui.internal.clear_active_id()
            elif state.deactivated:
                self._number_origin = None
        return changed, result

    def _value_committed(self):
        return self._number_committed

    def _slider_int(self, key, value, low, high, fmt="%d", flags=0):
        changed, result = self._slider_float(key, value, low, high, "%.0f", flags)
        return changed, round(result)

    def _color_edit3(self, key, values, flags=0):
        return self.fields.color(key, values)

    def _color_edit4(self, key, values, flags=0):
        return self.fields.color(key, values)

    def _color_committed(self):
        return self.fields.color_committed

    def _begin_toggle_grid(self, str_id, labels):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, DESIGN.row_gap * self.workspace.s / 2)
        width = imgui.get_content_region_avail().x
        required = max((imgui.calc_text_size(label).x for label in labels), default=0)
        required += 30 * self.workspace.s
        columns = max(1, min(2, int(width / max(1, required))))
        return imgui.begin_table(str_id, columns, imgui.TableFlags_.sizing_stretch_same)

    def _flag_table(self, ctx, table_id, flags, groups=2, *, translate_labels=True):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, DESIGN.row_gap * self.workspace.s / 2)
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
                label = render_flag_label(flag, ctx.tr, localized=True)
                imgui.align_text_to_frame_padding()
                imgui.text_disabled(label)
                imgui.table_next_column()
                self._flag_row(ctx, flag, label)
        imgui.end_table()
        self._control_padding()

    def _visual_groups(self, ctx):
        self._control_padding(checkbox=True)
        imgui.get_style().cell_padding = (0, DESIGN.row_gap * self.workspace.s / 2)
        super()._visual_groups(ctx)
        self._control_padding()

    def _general(self, ctx):
        super()._general(replace(ctx, font_report=None, panels=None))
        if ctx.panels is None:
            return
        from .constants import BOTTOM_PANELS, LEFT_PANELS

        workspace = self.workspace
        self._group_heading(ctx.tr("Panels"))
        if not self._begin_properties("reference_settings_panels"):
            return
        self._property(ctx.tr("Sidebar"))
        sidebar = workspace.left_active if workspace.open[workspace.left_active] else None
        names = tuple(
            name
            for name in LEFT_PANELS
            if ctx.panels.get(name) is not None and ctx.panels.get(name).enabled
        )
        changed, chosen = self._panel_choice(ctx, "settings-sidebar", names, sidebar)
        if changed:
            if chosen is None:
                for name in LEFT_PANELS:
                    ctx.panels.set_open(name, False)
            else:
                workspace.activate(chosen)
        self._property(ctx.tr("Bottom panel"))
        bottom = workspace.bottom_active if workspace.bottom_open else None
        names = tuple(
            name
            for name in BOTTOM_PANELS
            if ctx.panels.get(name) is not None and ctx.panels.get(name).enabled
        )
        changed, chosen = self._panel_choice(ctx, "settings-bottom", names, bottom)
        if changed:
            if chosen is None:
                workspace.collapse_bottom()
            else:
                workspace.activate(chosen)
        self._property(ctx.tr("Inspector"))
        self._control_padding(checkbox=True)
        changed, opened = themed_checkbox(
            "##settings_inspector", workspace.open["Inspector"], ctx.theme
        )
        self._record_control("settings-inspector")
        if changed:
            ctx.panels.set_open("Inspector", opened)
        imgui.end_table()

    def _record_control(self, key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.workspace.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

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
        u, s = self.workspace.ui, self.workspace.s
        u.role_icon("settings", x, y + 2 * s, 20 * s, 28 * s, role="title", c=SAGE)
        u.text(x + 28 * s, y + (32 - TITLE_SIZE) * s / 2, "Settings", TEXT, TITLE_SIZE, bold=True)
        u.hits["settings-title"] = (
            x + 28 * s,
            y + (32 - TITLE_SIZE) * s / 2,
            u.measure("Settings", TITLE_SIZE),
            TITLE_SIZE * s,
        )
        closed = u.button(
            "settings-close", x + width - 28 * s, y + 2 * s, 28, 28, icon="close", tooltip="Close"
        )
        u.line(x, y + 32 * s, width)
        if closed:
            self.workspace.open["Settings"] = self.open = False
        return closed

    def draw(self, ctx):
        u, s = self.workspace.ui, self.workspace.s
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
                    y + n // columns * (DESIGN.control_height + DESIGN.row_gap) * s,
                    cell_width / s - DESIGN.item_spacing_x,
                    DESIGN.control_height,
                    label=category,
                    align="center",
                    active=self._category == category,
                ):
                    self._category = category
            y += (
                (len(CATEGORIES) + columns - 1)
                // columns
                * (DESIGN.control_height + DESIGN.row_gap)
                + DESIGN.row_gap
            ) * s
            x, page_width = p.x, width
        else:
            button_width = max(
                128 * s, max(u.measure(ctx.tr(category)) for category in CATEGORIES) + 39 * s
            )
            nav_width = button_width + 2 * DESIGN.panel_padding_x * s
            imgui.set_cursor_screen_pos((p.x, y))
            imgui.push_style_color(imgui.Col_.child_bg, (26 / 255, 29 / 255, 32 / 255, 1))
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6 * s)
            imgui.begin_child("Reference settings navigation", (nav_width, 0))
            nav = imgui.get_cursor_screen_pos()
            for n, (category, icon) in enumerate(zip(CATEGORIES, CATEGORY_ICONS, strict=True)):
                if u.button(
                    "settings-" + category,
                    nav.x,
                    nav.y + n * (DESIGN.control_height + DESIGN.row_gap) * s,
                    button_width / s,
                    DESIGN.control_height,
                    icon=icon,
                    label=category,
                    align="left",
                    active=self._category == category,
                ):
                    self._category = category
            imgui.end_child()
            imgui.pop_style_var()
            imgui.pop_style_color()
            x, page_width = (
                p.x + nav_width + DESIGN.label_gap * s,
                width - nav_width - DESIGN.label_gap * s,
            )
        imgui.set_cursor_screen_pos((x, y))
        imgui.begin_child("Reference settings content", (page_width, max(s, available.y - y + p.y)))
        page = imgui.get_cursor_screen_pos()
        u.text(page.x, page.y, self._category, TEXT, TITLE_SIZE, bold=True)
        u.line(page.x, page.y + 32 * s, imgui.get_content_region_avail().x)
        imgui.dummy((0, 44 * s - imgui.get_style().item_spacing.y))
        self._checkbox_labels = frozenset(ctx.tr(label) for label in _CHECKBOX_PROPERTIES)
        try:
            with property_controls(u):
                self._control_padding()
                {
                    "General": self._general,
                    "Camera": self._camera_navigation,
                    "Interaction": self._interaction,
                    "Rendering": self._rendering,
                    "Recording": self._recording,
                    "MuJoCo Visuals": self._mujoco_visuals,
                }[self._category](ctx)
        finally:
            imgui.end_child()
