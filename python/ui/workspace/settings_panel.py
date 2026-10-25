"""Reference settings navigation around the editor's shared settings actions."""

from dataclasses import replace

from imgui_bundle import imgui

from mojive.ui.panels.settings import (
    SettingsPanel,
)
from mojive.ui.workspace.choice_popup import ChoicePopup
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.property_style import (
    NativePropertyPresentation,
    advance,
    property_section,
)
from mojive.ui.workspace.widgets import NumericEditState

CATEGORIES = ("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals")
CATEGORY_ICONS = ("general", "camera", "mouse", "shading", "video", "mujoco-visuals")
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
        self.fields = NativePropertyPresentation(workspace.ui, prefix="settings", fit_labels=True)
        self.choices = ChoicePopup(workspace.ui)
        self._number_origin = None
        self._number_committed = False
        self._automatic_width = None
        self._page_content_top = None

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

    def _end_properties(self):
        self.fields.end_table()

    def _property_rows(self, labels):
        return self.fields.clipped_rows(labels)

    def _combo(self, key, index, labels):
        return self.fields.combo(key, index, labels)

    def _tooltip(self, text, *, hovered=False):
        self.fields.tooltip(text, hovered=hovered)

    def _choice_list(self, key, index, labels, *, preview=None, enabled=None, tooltips=None):
        from mojive.ui.workspace.popup_style import menu_choice_width

        ui = self.workspace.ui
        preview = labels[index] if preview is None and index >= 0 else preview or ""
        selected = index
        if self.fields.begin_combo(
            key, preview, popup_width=lambda: menu_choice_width(ui, labels, index)
        ):
            self.choices.begin(key)
            try:
                for option, label in enumerate(labels):
                    available = enabled is None or enabled[option]
                    clicked, _ = self.choices.choice(
                        label,
                        option == index,
                        enabled=available,
                        tooltip=tooltips[option] if tooltips else "",
                    )
                    if clicked:
                        selected = option
            finally:
                self.choices.end()
        return selected != index, selected

    def _begin_combo(self, key, preview):
        return self.fields.begin_combo(key, preview)

    def _input_text(self, key, value, flags=0):
        return self.fields.input_text(key, value, flags)

    def _checkbox(self, key, value, theme=None):
        return self.fields.checkbox(key, value, theme)

    def _button(self, label):
        return self.fields.button(label)

    def _segmented(self, key, labels, index, theme=None, *, icons=None):
        return self.fields.segmented(key, labels, index, theme=theme, icons=icons)

    def _property(self, label, *, control_part="input_number"):
        if control_part == "input_number" and label in self._checkbox_labels:
            control_part = "checkbox"
        self._control_padding()
        self.fields.row(label, control_part=control_part)
        if control_part == "checkbox":
            self._control_padding(checkbox=True)

    def _readonly_value(self, label, value, *, muted=True):
        self._control_padding()
        self.fields.readonly_value(label, value, muted=muted)

    def _section_divider(self):
        # Section spacing and headings group the form. The window header owns
        # the sole rule, keeping every settings category's hierarchy consistent.
        return False

    def _group_heading(self, title):
        self._control_padding()
        ui = self.workspace.ui
        p = imgui.get_cursor_screen_pos()
        ui.section(
            title, p.x, p.y, imgui.get_content_region_avail().x, divider=self._section_divider()
        )
        advance(ui, p.x, p.y + ui.section_offset(38) * ui.s)

    def _collapsing_header(self, label, flags=0):
        # The section painter owns its geometry. Preserve the following fields' padding.
        return property_section(
            self.workspace.ui,
            label.split("##", 1)[0],
            flags,
            item_id=label,
            divider=self._section_divider(),
        )

    def _drag_float(self, key, value, speed, low, high, fmt="%.3f", flags=0, *, integer=False):
        changed, result = self.fields.numeric(key, value, speed, low, high, fmt, integer=integer)
        return self._finish_number(
            key, value, changed, result, self.workspace.ui.numeric_edits[key]
        )

    def _drag_int(self, key, value, speed, low, high, fmt="%d", flags=0):
        changed, result = self._drag_float(key, value, speed, low, high, fmt, flags, integer=True)
        return changed, round(result)

    def _input_float(self, key, value, step=0, fast_step=0, fmt="%.3f", flags=0):
        return self._drag_float(key, value, step or 0.01, 0, 0, fmt)

    def _slider_float(self, key, value, low, high, fmt="%.3f", flags=0, *, integer=False):
        edit = self.fields.value_rail(
            key, value, (low, high), initial=None, fmt=fmt, show_reset=False, integer=integer
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
        changed, result = self._slider_float(key, value, low, high, fmt, flags, integer=True)
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
        from mojive.ui.workspace.settings_visuals import flag_table

        flag_table(self, ctx, table_id, flags, groups)

    def _geometry_view(self, ctx):
        from mojive.ui.workspace.geometry_view import geometry_selector

        ui = self.workspace.ui
        if self._begin_properties("settings_geometry"):
            self._property(ctx.tr("Geometry view"), control_part="segment")
            position = imgui.get_cursor_screen_pos()
            geometry_selector(
                ui,
                ctx,
                position.x,
                position.y,
                self.fields._field_width(),
                key="settings-geometry",
            )
            self._end_properties()

    def _visual_groups(self, ctx):
        from mojive.ui.workspace.settings_visuals import visual_groups

        visual_groups(self, ctx)

    def _theme_settings(self, ctx):
        from mojive.design.themes import select_theme

        design = self.workspace.ui.design
        if design is None:
            return
        runtime = design.runtime
        names = ("dark", "light")
        current = runtime.active_theme
        self._property(ctx.tr("Theme"), control_part="segment")
        selected = names.index(current) if current in names else -1
        index = self._segmented(
            "##ui_theme",
            [ctx.tr("Dark theme"), ctx.tr("Light theme")],
            selected,
            icons=("moon", "sun"),
        )
        if index != selected:
            try:
                select_theme(runtime, names[index])
            except (OSError, ValueError) as error:
                ctx.session.report_message(str(error), level="error")

    def _general(self, ctx):
        super()._general(replace(ctx, font_report=None, panels=None))
        from mojive.ui.workspace.typography import draw_typography_settings

        draw_typography_settings(self, ctx)
        if ctx.panels is None:
            return
        from mojive.ui.workspace.constants import BOTTOM_PANELS, LEFT_PANELS

        workspace = self.workspace
        self._group_heading(ctx.tr("Panels"))
        if not self._begin_properties("reference_settings_panels"):
            return
        self._property(ctx.tr("Sidebar"), control_part="select")
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
        self._property(ctx.tr("Bottom panel"), control_part="select")
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
        self._property(ctx.tr("Inspector"), control_part="checkbox")
        self._control_padding(checkbox=True)
        changed, opened = self._checkbox(
            "##settings_inspector", workspace.open["Inspector"], ctx.theme
        )
        self._record_control("settings-inspector")
        if changed:
            ctx.panels.set_open("Inspector", opened)
        self._end_properties()

    def _record_control(self, key):
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.workspace.ui.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def _panel_choice(self, ctx, key, names, current):
        choices = (None, *names)
        changed, index = self._combo(
            "##" + key,
            choices.index(current) if current in choices else 0,
            tuple(ctx.tr(name or "none") for name in choices),
        )
        self._record_control(key)
        return changed, choices[index]

    def draw(self, ctx):
        from mojive.ui.workspace.settings_layout import draw_settings

        draw_settings(self, ctx)
