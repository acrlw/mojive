"""Renderer, visual group, and interaction settings."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.adapters.base import FrameNeeds
from mojive.render.backend import DebugView, RenderFlag
from mojive.ui.localization import LANGUAGE_LABELS, Language, parse_language
from mojive.ui.panels import (
    Panel,
    PanelContext,
    padded_selectable,
    pointer_hint,
    pointer_pressed,
    search_input,
    segmented_control,
    themed_checkbox,
)
from mojive.ui.panels.settings_pages import camera, interaction, recording, rendering, visuals
from mojive.ui.panels.settings_pages.catalog import (
    _CATEGORIES,
    _CATEGORY_WIDTH_PT,
    settings_category_label,
    settings_category_matches,
    settings_uses_stacked_layout,
)
from mojive.ui.panels.settings_pages.rendering import (
    flag_groups as flag_groups,
)
from mojive.ui.panels.settings_pages.rendering import (
    render_flag_label as render_flag_label,
)
from mojive.ui.panels.settings_pages.rendering import (
    responsive_flag_groups as responsive_flag_groups,
)
from mojive.ui.pointer_bindings import PointerAction
from mojive.ui.scrollbar_visibility import DEFAULT_HIDE_DELAY, MAX_HIDE_DELAY, MIN_HIDE_DELAY


class SettingsPanel(Panel):
    """Own navigation and draft text; category modules share these presentation hooks."""

    id = "settings"
    name = "Settings"
    default_open = False
    shortcut = "F9"
    modal = False
    dock_with = "Camera"

    def __init__(self) -> None:
        super().__init__()
        self._pointer_text = {}
        self._pointer_bindings_seen = None
        self._pointer_error = ""
        self._view = DebugView.SHADED
        self._message = ""
        self._category = "General"
        self._search = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        scale = ctx.style_scale
        stacked = settings_uses_stacked_layout(
            imgui.get_content_region_avail().x,
            scale,
        )
        if stacked:
            imgui.set_next_item_width(-1.0)
            changed, selected = self._choice_list(
                "##settings_category",
                _CATEGORIES.index(self._category),
                tuple(ctx.tr(settings_category_label(category)) for category in _CATEGORIES),
                enabled=tuple(
                    settings_category_matches(category, self._search) for category in _CATEGORIES
                ),
            )
            if changed:
                self._category = _CATEGORIES[selected]
            imgui.spacing()
        else:
            self._draw_category_rail(ctx, scale)
            imgui.same_line()

        imgui.begin_child(
            "settings_page",
            imgui.ImVec2(0.0, 0.0),
            imgui.ChildFlags_.always_use_window_padding,
        )
        imgui.set_next_item_width(-1.0)
        changed, self._search = search_input(
            "##settings_search",
            self._search,
            hint=ctx.tr("Search settings"),
            search_tooltip=ctx.tr("Search settings"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed and self._search and not settings_category_matches(self._category, self._search):
            self._category = next(
                (
                    category
                    for category in _CATEGORIES
                    if settings_category_matches(category, self._search)
                ),
                self._category,
            )
        imgui.spacing()
        imgui.text(ctx.tr(settings_category_label(self._category)))
        imgui.separator()
        if self._search and not settings_category_matches(self._category, self._search):
            imgui.text_disabled(ctx.tr("No matching settings"))
            imgui.end_child()
            return
        if self._category == "General":
            self._general(ctx)
        elif self._category == "Camera":
            self._camera_navigation(ctx)
        elif self._category == "Interaction":
            self._interaction(ctx)
        elif self._category == "Rendering":
            self._rendering(ctx)
        elif self._category == "Recording":
            self._recording(ctx)
        else:
            self._mujoco_visuals(ctx)
        imgui.end_child()

    def _draw_category_rail(self, ctx: PanelContext, scale: float) -> None:
        imgui.begin_child(
            "settings_categories",
            imgui.ImVec2(_CATEGORY_WIDTH_PT * scale, 0.0),
            0,
        )
        imgui.push_style_var(
            imgui.StyleVar_.selectable_text_align,
            imgui.ImVec2(0.5, 0.5),
        )
        for category in _CATEGORIES:
            matched = settings_category_matches(category, self._search)
            imgui.begin_disabled(not matched)
            selected, _ = padded_selectable(
                f"{ctx.tr(settings_category_label(category))}##settings_{category}",
                self._category == category,
            )
            if selected:
                self._category = category
            imgui.end_disabled()
        imgui.pop_style_var()
        imgui.end_child()

    def _general(self, ctx: PanelContext) -> None:
        t = ctx.tr
        languages = tuple(Language)
        current = parse_language(ctx.language)
        labels = [LANGUAGE_LABELS[language] for language in languages]
        if not self._begin_properties("settings_general"):
            return
        self._property(t("Language"), control_part="select")
        changed, index = self._combo("##ui_language", languages.index(current), labels)
        if changed and ctx.set_language is not None:
            ctx.apply_setting(ctx.set_language, languages[index].value)
        self._property(t("Model updates"), control_part="segment")
        selected = int(not ctx.live_model_updates)
        value = self._segmented(
            "model-updates",
            (t("Realtime"), t("Deferred")),
            selected,
            theme=ctx.theme,
        )
        if value != selected and ctx.set_live_model_updates is not None:
            ctx.apply_setting(ctx.set_live_model_updates, value == 0)
        self._tooltip(
            t("Realtime applies each edit. Deferred edits wait for Apply in the viewport.")
        )
        if ctx.set_scrollbar_hide_delay is not None:
            changed, delay, committed = self._number_setting(
                ctx,
                "##scrollbar_hide_delay",
                "Scrollbar hide delay",
                ctx.scrollbar_hide_delay,
                speed=0.1,
                low=MIN_HIDE_DELAY,
                high=MAX_HIDE_DELAY,
                fmt="%.1f s",
                default=DEFAULT_HIDE_DELAY,
                clamp=True,
                show_reset_hint=False,
            )
            if changed or committed:
                ctx.apply_setting(ctx.set_scrollbar_hide_delay, delay, persist=committed)
        if ctx.font_report is not None:
            self._readonly_value(t("UI font"), ctx.font_report.mono)
            self._readonly_value(t("CJK font"), ctx.font_report.cjk or ctx.tr("none"))
        self._end_properties()
        if ctx.panels is not None:
            self._group_heading(t("Panels"))
            panels = tuple(
                panel
                for panel in ctx.panels
                if panel.enabled and not panel.modal and panel.id != self.id
            )
            if self._begin_toggle_grid("settings_panels", tuple(t(panel.name) for panel in panels)):
                for panel in panels:
                    imgui.table_next_column()
                    changed, is_open = self._checkbox(
                        f"{t(panel.name)}###panel_open_{panel.id}", panel.open, ctx.theme
                    )
                    if changed:
                        ctx.panels.set_open(panel.id, is_open)
                self._end_properties()

    def show_category(self, category: str) -> None:
        if category in _CATEGORIES:
            self._category = category
            self._search = ""

    def _camera_navigation(self, ctx: PanelContext) -> None:
        return camera.draw(self, ctx)

    def _recording(self, ctx: PanelContext) -> None:
        return recording.draw(self, ctx)

    def _rendering(self, ctx: PanelContext) -> None:
        return rendering.draw(self, ctx)

    def _interaction(self, ctx: PanelContext) -> None:
        return interaction.draw(self, ctx)

    def _number_setting(
        self,
        ctx: PanelContext,
        identifier: str,
        label: str,
        value: float,
        *,
        speed: float,
        low: float,
        high: float,
        fmt: str,
        default: float,
        tooltip: str = "",
        clamp: bool = False,
        show_reset_hint: bool = True,
    ) -> tuple[bool, float, bool]:
        """Return preview and commit phases without owning or saving the setting."""
        self._property(ctx.tr(label))
        changed, edited = self._drag_float(
            identifier,
            float(value),
            speed,
            low,
            high,
            fmt,
            imgui.SliderFlags_.always_clamp if clamp else 0,
        )
        committed = self._value_committed()
        hovered = imgui.is_item_hovered()
        reset = hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET)
        if reset:
            changed, edited = True, default
        if hovered and (tooltip or show_reset_hint):
            self._tooltip(
                ctx.tr(tooltip)
                if tooltip
                else pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")),
                hovered=True,
            )
        # Idle float widgets may round-trip through float32. A commit on release
        # must retain the last accepted preview rather than replacing its precision.
        return changed, edited if changed else value, committed or reset

    def _geometry_view(self, ctx):
        from mojive.ui.geometry_view import draw_geometry_view

        draw_geometry_view(ctx.backend, ctx.tr, compact=True, theme=ctx.theme, combo=self._combo)

    def _mujoco_visuals(self, ctx: PanelContext) -> None:
        return visuals.draw(self, ctx)

    @staticmethod
    def _begin_toggle_grid(str_id: str, labels: tuple[str, ...]) -> bool:
        style = imgui.get_style()
        required = max((imgui.calc_text_size(label).x for label in labels), default=0.0)
        required += (
            imgui.get_frame_height() + style.item_inner_spacing.x + 2.0 * style.cell_padding.x
        )
        columns = max(1, min(4, int(imgui.get_content_region_avail().x / max(1.0, required))))
        return imgui.begin_table(str_id, columns, imgui.TableFlags_.sizing_stretch_same)

    @staticmethod
    def _end_properties():
        imgui.end_table()

    @staticmethod
    def _begin_properties(str_id: str) -> bool:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.pad_outer_x
        compact = imgui.get_content_region_avail().x < 24.0 * imgui.get_font_size()
        if not imgui.begin_table(str_id, 1 if compact else 2, flags):
            return False
        if not compact:
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 0.64)
        return True

    @staticmethod
    def _group_heading(title: str) -> None:
        imgui.spacing()
        imgui.text_disabled(title)
        imgui.separator()

    def _readonly_value(self, label: str, value: str, *, muted=True) -> None:
        self._property(label)
        imgui.align_text_to_frame_padding()
        if muted:
            imgui.push_style_color(
                imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled)
            )
        imgui.text_wrapped(str(value))
        if muted:
            imgui.pop_style_color()

    @staticmethod
    def _property(label: str, *, control_part: str = "input_number") -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        compact = imgui.table_get_column_count() == 1
        if not compact:
            imgui.align_text_to_frame_padding()
        imgui.text(label)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)

    def _visual_groups(self, ctx: PanelContext) -> None:
        return visuals.visual_groups(self, ctx)

    def _flag_row(self, ctx: PanelContext, flag: RenderFlag, display_label: str) -> None:
        return rendering.flag_row(self, ctx, flag, display_label)

    def _flag_table(
        self,
        ctx: PanelContext,
        table_id: str,
        flags: tuple[RenderFlag, ...],
        groups: int = 2,
        *,
        translate_labels: bool = True,
    ) -> None:
        return rendering.flag_table(
            self, ctx, table_id, flags, groups, translate_labels=translate_labels
        )

    def current_view(self, backend) -> DebugView:
        getter = getattr(backend, "get_debug_view", None)
        return getter() if callable(getter) else self._view

    @staticmethod
    def _combo(*args, **kwargs):
        return imgui.combo(*args, **kwargs)

    def _choice_list(self, key, index, labels, *, preview=None, enabled=None, tooltips=None):
        """Present optional disabled choices while retaining the panel's setting ownership."""
        preview = labels[index] if preview is None and index >= 0 else preview or ""
        selected = index
        if self._begin_combo(key, preview):
            for option, label in enumerate(labels):
                available = enabled is None or enabled[option]
                imgui.begin_disabled(not available)
                clicked, _ = padded_selectable(label, option == index)
                imgui.end_disabled()
                if clicked and available:
                    selected = option
                if (
                    tooltips
                    and tooltips[option]
                    and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
                ):
                    self._tooltip(tooltips[option], hovered=True)
            imgui.end_combo()
        return selected != index, selected

    @staticmethod
    def _begin_combo(*args, **kwargs):
        return imgui.begin_combo(*args, **kwargs)

    @staticmethod
    def _input_text(*args, **kwargs):
        return imgui.input_text(*args, **kwargs)

    @staticmethod
    def _checkbox(key, value, theme=None):
        from mojive.ui.theme import THEME

        return themed_checkbox(key, value, theme or THEME)

    @staticmethod
    def _button(*args, **kwargs):
        return imgui.button(*args, **kwargs)

    @staticmethod
    def _segmented(*args, **kwargs):
        return segmented_control(*args, **kwargs)

    @staticmethod
    def _collapsing_header(*args, **kwargs):
        return imgui.collapsing_header(*args, **kwargs)

    @staticmethod
    def _drag_float(*args, **kwargs):
        return imgui.drag_float(*args, **kwargs)

    @staticmethod
    def _drag_int(*args, **kwargs):
        return imgui.drag_int(*args, **kwargs)

    @staticmethod
    def _input_float(*args, **kwargs):
        return imgui.input_float(*args, **kwargs)

    @staticmethod
    def _slider_float(*args, **kwargs):
        return imgui.slider_float(*args, **kwargs)

    @staticmethod
    def _slider_int(*args, **kwargs):
        return imgui.slider_int(*args, **kwargs)

    @staticmethod
    def _color_edit3(*args, **kwargs):
        return imgui.color_edit3(*args, **kwargs)

    @staticmethod
    def _color_edit4(*args, **kwargs):
        return imgui.color_edit4(*args, **kwargs)

    @staticmethod
    def _value_committed():
        return imgui.is_item_deactivated_after_edit()

    @staticmethod
    def _color_committed():
        return imgui.is_item_deactivated_after_edit()
