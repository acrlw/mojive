"""Renderer, visual group, and interaction settings."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...render.backend import DebugView, RenderFlag
from ..localization import LANGUAGE_LABELS, Language, parse_language
from ..pointer_bindings import PointerAction
from . import (
    Panel,
    PanelContext,
    padded_selectable,
    pointer_hint,
    pointer_pressed,
    search_input,
    segmented_control,
    themed_checkbox,
)
from .settings_pages import camera, interaction, recording, rendering, visuals
from .settings_pages.catalog import (
    _CATEGORIES,
    _CATEGORY_WIDTH_PT,
    settings_category_matches,
    settings_uses_stacked_layout,
)
from .settings_pages.rendering import (
    flag_groups as flag_groups,
)
from .settings_pages.rendering import (
    render_flag_label as render_flag_label,
)
from .settings_pages.rendering import (
    responsive_flag_groups as responsive_flag_groups,
)


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
            if imgui.begin_combo("##settings_category", ctx.tr(self._category)):
                for category in _CATEGORIES:
                    matched = settings_category_matches(category, self._search)
                    imgui.begin_disabled(not matched)
                    selected, _ = padded_selectable(
                        f"{ctx.tr(category)}##settings_{category}",
                        self._category == category,
                    )
                    if selected:
                        self._category = category
                    imgui.end_disabled()
                imgui.end_combo()
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
        imgui.text(ctx.tr(self._category))
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
                f"{ctx.tr(category)}##settings_{category}", self._category == category
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
        self._property(t("Language"))
        changed, index = imgui.combo("##ui_language", languages.index(current), labels)
        if changed and ctx.set_language is not None:
            ctx.apply_setting(ctx.set_language, languages[index].value)
        self._property(t("Model updates"))
        selected = int(not ctx.live_model_updates)
        value = segmented_control(
            "model-updates",
            (t("Realtime"), t("Deferred")),
            selected,
            theme=ctx.theme,
        )
        if value != selected and ctx.set_live_model_updates is not None:
            ctx.apply_setting(ctx.set_live_model_updates, value == 0)
        imgui.set_item_tooltip(
            t("Realtime applies each edit. Deferred edits wait for Apply in the viewport.")
        )
        if ctx.font_report is not None:
            self._property(t("UI font"))
            imgui.text_disabled(ctx.font_report.mono)
            self._property(t("CJK font"))
            imgui.text_disabled(ctx.font_report.cjk or ctx.tr("none"))
        imgui.end_table()
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
                    changed, is_open = themed_checkbox(
                        f"{t(panel.name)}###panel_open_{panel.id}", panel.open, ctx.theme
                    )
                    if changed:
                        ctx.panels.set_open(panel.id, is_open)
                imgui.end_table()

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
        if hovered:
            imgui.set_tooltip(
                ctx.tr(tooltip)
                if tooltip
                else pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset"))
            )
        # Idle float widgets may round-trip through float32. A commit on release
        # must retain the last accepted preview rather than replacing its precision.
        return changed, edited if changed else value, committed or reset

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

    @staticmethod
    def _property(label: str) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        compact = imgui.table_get_column_count() == 1
        if not compact:
            imgui.align_text_to_frame_padding()
        width = imgui.get_content_region_avail().x
        text_width = imgui.calc_text_size(label).x
        if not compact:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, width - text_width))
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
