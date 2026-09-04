"""Shared panel protocols and UI controls."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...config import PanelConfig
from ..draw2d import ImguiDraw2D
from ..theme import THEME, Theme
from ..viewport_widgets import ToolHint, draw_projection_label

if TYPE_CHECKING:
    from ...render.backend import RenderBackend
    from ...session import Session


@dataclass
class PanelContext:
    session: Session
    backend: RenderBackend
    camera: Any = None

    model_camera_id: int = -1
    model_camera_view: Any = None
    select_model_camera: Any = None
    focus_node: Any = None
    focus_joint: Any = None
    request_rename: Any = None
    request_model_rename: Any = None
    request_texture_import: Any = None
    request_geometry_resource_import: Any = None
    request_model_asset_import: Any = None
    request_model_asset_replace: Any = None

    theme: Theme = THEME
    gizmo: Any = None
    view_cube: Any = None
    perturb: Any = None
    scene_entities: Any = None
    camera_preview: Any = None

    style_scale: float = 1.0

    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    dt: float = 0.0

    info: dict[str, Any] = field(default_factory=dict)

    status: str = ""
    # Each panel publishes its available grammar independently of hover.
    # PanelManager collects it by name; the application selects the clicked panel.
    status_hints: tuple[Any, ...] = ()
    status_hints_by_panel: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    panels: Any = None

    language: str = "en"
    translate: Any = None
    set_language: Any = None
    set_shadow_quality: Any = None
    interactions: Any = None
    set_interactions: Any = None
    selection_style: Any = None
    set_selection_style: Any = None
    set_precise_input_memory: Any = None
    set_view_selection_padding: Any = None
    viewport_overlay_scale: float = 1.0
    set_viewport_overlay_scale: Any = None
    viewport_overlays: Any = None
    set_viewport_overlays: Any = None
    set_viewport_capsule_scale: Any = None
    input_bindings: Any = None
    set_input_binding: Any = None
    reset_input_bindings: Any = None
    font_report: Any = None
    output: Any = None

    def submit(self, command: Any) -> Any:
        result = self.session.submit(command)
        if result.message:
            self.status = result.message
        return result

    def report(
        self,
        message: str,
        *,
        level: str = "warning",
        duration: float | None = 5.0,
    ) -> None:
        """Keep a panel diagnostic visible in the shared status channel."""
        self.status = str(message)
        self.session.report_message(self.status, level=level, duration=duration)

    def tr(self, value: str) -> str:
        return self.translate(value) if self.translate is not None else value


class Panel:
    id: str = ""
    name: str = ""

    default_open: bool = True
    shortcut: str = ""

    aliases: tuple[str, ...] = ()
    standalone: bool = False
    modal: bool = False
    closable: bool = True
    dock_with: str = ""
    initial_size: tuple[float, float] = (0.0, 0.0)

    def __init__(self) -> None:
        self.enabled = True
        self.open = self.default_open

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def needs(self) -> FrameNeeds:
        return self.frame_needs() if self.enabled and self.open else FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        raise NotImplementedError

    def finish_frame(self, ctx: PanelContext) -> None:
        pass

    def toggle(self) -> None:
        self.open = not self.open

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} open={self.open}>"


_EXPANDED: set[str] = set()


def slider_gesture(
    hovered: bool, right_clicked: bool, double_clicked: bool, shift: bool
) -> str | None:
    if not hovered:
        return None
    if right_clicked:
        return "expand" if shift else "reset"
    if double_clicked:
        return "copy"
    return None


def copyable_name_item(ctx: PanelContext, name: str, available_width: float) -> bool:
    """Expose a clipped row name and make its right-click action discoverable."""

    publish_status_hint(
        ctx,
        ToolHint("mouse", "right", ctx.tr("Copy name"), hint_id="panel.copy-name"),
    )
    hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
    if not hovered:
        return False
    visible_width = max(1.0, float(available_width) - 2.0 * imgui.get_style().frame_padding.x)
    if imgui.calc_text_size(name).x > visible_width:
        imgui.set_tooltip(name)
    copied = imgui.is_mouse_clicked(imgui.MouseButton_.right)
    if copied:
        imgui.set_clipboard_text(name)
    return bool(copied)


def state_vector_text(values) -> str:
    """Format one complete simulation vector for lossless clipboard reuse."""

    vector = np.asarray(values).reshape(-1)
    return "[" + ", ".join(repr(float(value)) for value in vector) + "]"


def copy_state_vector(values) -> bool:
    """Copy a simulation vector when the current adapter produced it."""

    if values is None:
        return False
    imgui.set_clipboard_text(state_vector_text(values))
    return True


def publish_status_hint(ctx: PanelContext, hint: ToolHint) -> None:
    """Publish one stable panel grammar entry without duplicating visible rows."""

    if hint.hint_id and any(existing.hint_id == hint.hint_id for existing in ctx.status_hints):
        return
    if hint not in ctx.status_hints:
        ctx.status_hints = (*ctx.status_hints, hint)


def publish_focus_item_hint(ctx: PanelContext) -> None:
    """Advertise the shared hierarchy/joint double-click focus gesture."""

    publish_status_hint(
        ctx,
        ToolHint(
            "mouse",
            "left",
            ctx.tr("Focus item"),
            "×2",
            hint_id="panel.focus-item",
        ),
    )


@dataclass
class ValueEdit:
    changed: bool = False
    value: float = 0.0
    copied: bool = False
    expanded: bool = False


def value_slider(
    label: str,
    value: float,
    lo: float,
    hi: float,
    *,
    initial: float | None = None,
    fmt: str = "%.4f",
    width: float = 0.0,
    more_hint: str = "more options",
) -> ValueEdit:
    if width:
        imgui.set_next_item_width(width)
    changed, new_value = imgui.slider_float(label, value, lo, hi, fmt)
    io = imgui.get_io()
    action = slider_gesture(
        imgui.is_item_hovered(),
        imgui.is_mouse_clicked(imgui.MouseButton_.right),
        imgui.is_mouse_double_clicked(imgui.MouseButton_.left),
        bool(io.key_shift),
    )
    out = ValueEdit(changed=changed, value=new_value, expanded=label in _EXPANDED)

    if action == "reset" and initial is not None:
        out.changed = True
        out.value = float(initial)
    elif action == "copy":
        imgui.set_clipboard_text(fmt % value)
        out.copied = True
    elif action == "expand" and more_hint:
        if label in _EXPANDED:
            _EXPANDED.discard(label)
        else:
            _EXPANDED.add(label)
        out.expanded = label in _EXPANDED

    tooltip = "Right-click reset · Double-click copy"
    if more_hint:
        tooltip += f" · Shift+right-click {more_hint}"
    imgui.set_item_tooltip(tooltip)
    return out


def is_expanded(label: str) -> bool:
    return label in _EXPANDED


def colored_text(color: tuple[float, float, float, float], text: str) -> None:
    imgui.text_colored(imgui.ImVec4(*color), text)


def search_input(
    str_id: str,
    value: str,
    *,
    hint: str = "",
    search_tooltip: str = "Search",
    clear_tooltip: str = "Clear search",
) -> tuple[bool, str]:
    """Draw a search field with consistent search and clear affordances."""

    if value:
        imgui.set_next_item_allow_overlap()
    if hint:
        changed, value = imgui.input_text_with_hint(str_id, hint, value)
    else:
        changed, value = imgui.input_text(str_id, value)

    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    cursor_after_input = imgui.get_cursor_screen_pos()
    color = imgui.color_convert_float4_to_u32(imgui.get_style_color_vec4(imgui.Col_.text_disabled))
    radius = max(2.5, (hi.y - lo.y) * 0.16)
    icon_step = max(radius * 3.6, hi.y - lo.y)
    search_center = imgui.ImVec2(
        hi.x - radius * 2.7,
        (lo.y + hi.y) * 0.5 - radius * 0.2,
    )
    draw_list = imgui.get_window_draw_list()
    draw_list.add_circle(search_center, radius, color, 12, 1.2)
    draw_list.add_line(
        imgui.ImVec2(
            search_center.x + radius * 0.70,
            search_center.y + radius * 0.70,
        ),
        imgui.ImVec2(
            search_center.x + radius * 1.55,
            search_center.y + radius * 1.55,
        ),
        color,
        1.2,
    )
    search_lo = imgui.ImVec2(search_center.x - icon_step * 0.5, lo.y)
    search_hi = imgui.ImVec2(search_center.x + icon_step * 0.5, hi.y)
    if imgui.is_mouse_hovering_rect(search_lo, search_hi):
        imgui.set_tooltip(search_tooltip)

    if value:
        clear_center = imgui.ImVec2(
            search_center.x - icon_step,
            (lo.y + hi.y) * 0.5,
        )
        clear_lo = imgui.ImVec2(clear_center.x - icon_step * 0.5, lo.y)
        clear_hi = imgui.ImVec2(clear_center.x + icon_step * 0.5, hi.y)
        imgui.set_cursor_screen_pos(clear_lo)
        identifier = str_id.partition("##")[2] or str_id
        clear_clicked = imgui.invisible_button(
            f"##clear_{identifier}",
            imgui.ImVec2(clear_hi.x - clear_lo.x, clear_hi.y - clear_lo.y),
        )
        clear_hovered = imgui.is_item_hovered()
        imgui.set_cursor_screen_pos(cursor_after_input)
        clear_color = (
            imgui.color_convert_float4_to_u32(imgui.get_style_color_vec4(imgui.Col_.text))
            if clear_hovered
            else color
        )
        arm = radius * 0.88
        draw_list.add_line(
            imgui.ImVec2(clear_center.x - arm, clear_center.y - arm),
            imgui.ImVec2(clear_center.x + arm, clear_center.y + arm),
            clear_color,
            1.4,
        )
        draw_list.add_line(
            imgui.ImVec2(clear_center.x + arm, clear_center.y - arm),
            imgui.ImVec2(clear_center.x - arm, clear_center.y + arm),
            clear_color,
            1.4,
        )
        if clear_hovered:
            imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
            imgui.set_tooltip(clear_tooltip)
        if clear_clicked:
            value = ""
            changed = True
            imgui.set_keyboard_focus_here(-1)
    return changed, value


def sort_order_glyph(
    rect: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return a compact list-and-arrow sorting glyph inside ``rect``."""

    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    left = x0 + width * 0.24
    arrow_x = x0 + width * 0.72
    ys = (y0 + height * 0.32, y0 + height * 0.50, y0 + height * 0.68)
    bars = tuple(
        ((left, y), (left + width * length, y))
        for y, length in zip(ys, (0.28, 0.21, 0.14), strict=True)
    )
    arrow_top = y0 + height * 0.28
    arrow_bottom = y0 + height * 0.70
    wing = width * 0.09
    return (
        *bars,
        ((arrow_x, arrow_top), (arrow_x, arrow_bottom)),
        ((arrow_x - wing, arrow_bottom - wing), (arrow_x, arrow_bottom)),
        ((arrow_x + wing, arrow_bottom - wing), (arrow_x, arrow_bottom)),
    )


def sort_order_tooltip(
    by_name: bool,
    state_order: str,
    translate=lambda value: value,
) -> str:
    """Describe only the active order; the button itself communicates switching."""

    current = translate("Name") if by_name else state_order
    return f"{translate('Order')}: {current}"


def sort_order_button(
    str_id: str,
    by_name: bool,
    *,
    state_order: str,
    translate=lambda value: value,
) -> tuple[bool, bool]:
    """Draw a compact state/name order toggle beside a search field."""

    size = imgui.get_frame_height()
    left_clicked = imgui.button(f"##sort_order_{str_id}", imgui.ImVec2(size, 0.0))
    hovered = imgui.is_item_hovered()
    right_clicked = hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right)
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    color_value = imgui.get_style_color_vec4(imgui.Col_.check_mark if by_name else imgui.Col_.text)
    color = imgui.color_convert_float4_to_u32(color_value)
    thickness = max(1.0, size * 0.045)
    draw = imgui.get_window_draw_list()
    for start, end in sort_order_glyph((lo.x, lo.y, hi.x, hi.y)):
        draw.add_line(imgui.ImVec2(*start), imgui.ImVec2(*end), color, thickness)
    imgui.set_item_tooltip(sort_order_tooltip(by_name, state_order, translate))
    changed = bool(left_clicked or right_clicked)
    return changed, (not by_name if changed else by_name)


def searchable_ordered_list_header(
    str_id: str,
    value: str,
    by_name: bool,
    *,
    hint: str,
    search_tooltip: str,
    clear_tooltip: str,
    state_order: str,
    translate=lambda text: text,
) -> tuple[bool, str, bool, bool]:
    """Draw one responsive search field followed by its list-order button."""

    spacing = imgui.get_style().item_spacing.x
    button_width = imgui.get_frame_height()
    available = imgui.get_content_region_avail().x
    imgui.set_next_item_width(max(1.0, available - spacing - button_width))
    search_changed, value = search_input(
        str_id,
        value,
        hint=hint,
        search_tooltip=search_tooltip,
        clear_tooltip=clear_tooltip,
    )
    imgui.same_line()
    sort_changed, by_name = sort_order_button(
        str_id,
        by_name,
        state_order=state_order,
        translate=translate,
    )
    return search_changed, value, sort_changed, by_name


def horizontal_wheel_target(
    current: float,
    maximum: float,
    wheel: float,
    wheel_horizontal: float = 0.0,
    *,
    step: float = 48.0,
) -> float:
    """Map ordinary wheel input onto a bounded horizontal scroll position."""

    delta = float(wheel_horizontal) - float(wheel)
    return min(max(0.0, float(current) + delta * float(step)), max(0.0, float(maximum)))


def horizontal_wheel_scroll(*, step: float = 48.0) -> bool:
    """Scroll the current horizontal child with either wheel axis."""

    if not imgui.is_window_hovered(imgui.HoveredFlags_.child_windows):
        return False
    maximum = float(imgui.get_scroll_max_x())
    if maximum <= 0.0:
        return False
    io = imgui.get_io()
    wheel = float(io.mouse_wheel)
    wheel_horizontal = float(io.mouse_wheel_h)
    if wheel == 0.0 and wheel_horizontal == 0.0:
        return False
    current = float(imgui.get_scroll_x())
    target = horizontal_wheel_target(
        current,
        maximum,
        wheel,
        wheel_horizontal,
        step=step,
    )
    if target == current:
        return False
    imgui.set_scroll_x(target)
    return True


def labeled(label: str, value: str) -> None:
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(label)
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text(value)


def begin_kv_table(str_id: str) -> bool:
    return imgui.begin_table(str_id, 2, imgui.TableFlags_.sizing_stretch_prop)


def themed_checkbox(
    label: str,
    value: bool,
    theme: Theme = THEME,
) -> tuple[bool, bool]:
    """Draw a neutral checkbox without the platform/default blue selected fill."""

    visible_label = label.partition("##")[0]
    size = imgui.get_frame_height()
    clicked = imgui.invisible_button(label, imgui.ImVec2(size, size))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    surface = (
        theme.bg_frame_active
        if active or value
        else theme.bg_frame_hovered
        if hovered
        else theme.bg_frame
    )
    draw = ImguiDraw2D()
    rounding = min(imgui.get_style().frame_rounding, size * 0.24)
    draw.rect_filled((lo.x, lo.y), (hi.x, hi.y), surface, rounding=rounding)
    draw.rect((lo.x, lo.y), (hi.x, hi.y), theme.border, 1.0, rounding=rounding)
    if value:
        draw.polyline(
            (
                (lo.x + size * 0.22, lo.y + size * 0.52),
                (lo.x + size * 0.43, lo.y + size * 0.72),
                (lo.x + size * 0.80, lo.y + size * 0.29),
            ),
            theme.primary_bright,
            max(1.5, size * 0.11),
        )
    if visible_label:
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        imgui.text(visible_label)
    return clicked, not value if clicked else value


def segmented_control(
    str_id: str,
    labels: tuple[str, ...],
    selected: int,
    *,
    width: float = 0.0,
    theme: Theme = THEME,
    icons: tuple[str, ...] | None = None,
) -> int:
    """Draw a compact mutually exclusive button row and return its selected index."""

    if not labels:
        return 0
    spacing = imgui.get_style().item_spacing.x
    available = width if width > 0.0 else imgui.get_content_region_avail().x
    item_width = max(1.0, (available - spacing * (len(labels) - 1)) / len(labels))
    result = min(max(0, int(selected)), len(labels) - 1)
    draw = ImguiDraw2D()
    for index, label in enumerate(labels):
        if index:
            imgui.same_line()
        is_selected = index == result
        if is_selected:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*theme.bg_frame_active))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*theme.primary_bright))
        icon = icons[index] if icons is not None and index < len(icons) else ""
        # Icon-bearing segments paint their complete contents explicitly.  Reserving
        # glyph space with leading text made ImGui center the whitespace rather than
        # the visible icon/text pair, which drifted at different fonts and UI scales.
        button_label = f"##{str_id}-{index}" if icon else f"{label}##{str_id}-{index}"
        clicked = imgui.button(button_label, imgui.ImVec2(item_width, 0.0))
        item_min = imgui.get_item_rect_min()
        item_max = imgui.get_item_rect_max()
        if is_selected:
            imgui.pop_style_color(2)
        if icon:
            base_color = theme.primary_bright if is_selected else theme.text
            color = (*base_color[:3], base_color[3] * imgui.get_style().alpha)
            glyph_scale = max(0.65, imgui.get_frame_height() / 24.0)
            draw_projection_label(
                draw,
                (item_min.x, item_min.y),
                (item_max.x, item_max.y),
                color,
                glyph_scale,
                icon,
                label,
            )
        if clicked:
            result = index
    return result


def button_width(label: str, minimum: float = 0.0) -> float:
    text = label.partition("##")[0]
    padding = 2.0 * imgui.get_style().frame_padding.x
    return max(float(minimum), imgui.calc_text_size(text).x + padding)


def button_row_layout(
    widths: tuple[float, ...], available: float, spacing: float
) -> tuple[bool, ...]:
    same_line: list[bool] = []
    used = 0.0
    for width in widths:
        inline = bool(same_line) and used + spacing + width <= available
        if inline:
            used += spacing + width
        else:
            used = width
        same_line.append(inline)
    return tuple(same_line)


@dataclass(frozen=True)
class PanelState:
    """Current availability and open state of one registered panel."""

    enabled: bool
    open: bool


class PanelManager:
    """Register panels and control them through stable, non-localized IDs."""

    def __init__(
        self,
        panels: list[Panel] | None = None,
        config: dict[str, PanelConfig] | None = None,
    ) -> None:
        self.panels: list[Panel] = list(panels) if panels is not None else default_panels()
        self._pending_config = dict(config or {})
        problems = validate_panels(self.panels)
        if problems:
            raise ValueError("Invalid panel configuration: " + "; ".join(problems))
        for panel in self.panels:
            self._apply_config(panel)

    def __iter__(self):
        return iter(self.panels)

    def get(self, panel_id: str) -> Panel | None:
        """Return a panel by stable ID, accepting its legacy title for compatibility."""

        value = str(panel_id)
        return next((p for p in self.panels if _panel_id(p) == value or p.name == value), None)

    def register(self, panel: Panel) -> None:
        """Register one custom panel and apply any deferred configuration."""

        if self.get(_panel_id(panel)) is not None:
            raise ValueError(f"Duplicate panel ID: {_panel_id(panel)}")
        self.panels.append(panel)
        problems = validate_panels(self.panels)
        if problems:
            self.panels.pop()
            raise ValueError("Invalid panel configuration: " + "; ".join(problems))
        self._apply_config(panel)

    def set_open(self, panel_id: str, open: bool) -> bool:
        panel = self.get(panel_id)
        if panel is None or not panel.enabled:
            return False
        panel.open = bool(open)
        return True

    def open(self, panel_id: str) -> bool:
        return self.set_open(panel_id, True)

    def close(self, panel_id: str) -> bool:
        return self.set_open(panel_id, False)

    def toggle(self, panel_id: str) -> bool:
        panel = self.get(panel_id)
        return False if panel is None else self.set_open(panel_id, not panel.open)

    def set_enabled(self, panel_id: str, enabled: bool) -> bool:
        panel = self.get(panel_id)
        if panel is None:
            return False
        panel.enabled = bool(enabled)
        if not panel.enabled:
            panel.open = False
        return True

    def enable(self, panel_id: str) -> bool:
        return self.set_enabled(panel_id, True)

    def disable(self, panel_id: str) -> bool:
        return self.set_enabled(panel_id, False)

    def state(self, panel_id: str) -> PanelState | None:
        panel = self.get(panel_id)
        return None if panel is None else PanelState(bool(panel.enabled), bool(panel.open))

    def states(self) -> dict[str, PanelState]:
        return {_panel_id(panel): PanelState(panel.enabled, panel.open) for panel in self.panels}

    def open_panel(self, panel_id: str) -> None:
        """Compatibility alias for callers using the previous method name."""

        self.open(panel_id)

    def _apply_config(self, panel: Panel) -> None:
        override = self._pending_config.pop(_panel_id(panel), None)
        if override is None:
            return
        if override.enabled is not None:
            panel.enabled = bool(override.enabled)
        if override.open is not None:
            panel.open = bool(override.open)
        if not panel.enabled:
            panel.open = False

    def frame_needs(self) -> FrameNeeds:
        needs = FrameNeeds.none()
        for p in self.panels:
            needs = needs.merge(p.needs())
        return needs

    def draw(self, ctx: PanelContext) -> None:
        ctx.panels = self
        ctx.status_hints_by_panel.clear()
        for p in self.panels:
            if not p.enabled or not p.open:
                continue
            translated = ctx.tr(p.name)
            title = p.name if translated == p.name else f"{translated}###{p.name}"
            if p.modal:
                self._draw_modal(p, ctx, title)
                continue
            # Keep hidden dock tabs addressable on their activation frame,
            # before ImGui begins submitting the newly selected tab's contents.
            ctx.status_hints_by_panel[p.name] = ()
            expanded, keep_open = self._begin_panel_window(
                p,
                title,
                ctx.style_scale,
                self._translated_panel_title(ctx.tr, p.dock_with),
            )
            ctx.status_hints = ()
            if expanded:
                p.draw(ctx)
                ctx.status_hints_by_panel[p.name] = tuple(ctx.status_hints)
            p.finish_frame(ctx)
            imgui.end()
            if keep_open is not None and not keep_open:
                p.open = False
                ctx.status_hints_by_panel.pop(p.name, None)

    def draw_shells(self, translate, style_scale: float) -> None:
        """Submit docked panel windows without reading application state."""

        for panel in self.panels:
            if not panel.enabled or not panel.open or panel.modal:
                continue
            translated = translate(panel.name)
            title = panel.name if translated == panel.name else f"{translated}###{panel.name}"
            _expanded, keep_open = self._begin_panel_window(
                panel,
                title,
                style_scale,
                self._translated_panel_title(translate, panel.dock_with),
            )
            imgui.end()
            if keep_open is not None and not keep_open:
                panel.open = False

    @staticmethod
    def _translated_panel_title(translate, name: str) -> str:
        if not name:
            return ""
        translated = translate(name)
        return name if translated == name else f"{translated}###{name}"

    def _begin_panel_window(
        self, panel: Panel, title: str, style_scale: float, dock_neighbor_title: str = ""
    ):
        flags = 0
        if panel.standalone:
            viewport = imgui.get_main_viewport()
            width, height = panel.initial_size
            if width > 0.0 and height > 0.0:
                imgui.set_next_window_size(
                    imgui.ImVec2(width * style_scale, height * style_scale),
                    imgui.Cond_.first_use_ever,
                )
            imgui.set_next_window_pos(
                viewport.get_center(),
                imgui.Cond_.first_use_ever,
                imgui.ImVec2(0.5, 0.5),
            )
            flags = imgui.WindowFlags_.no_docking.value
        elif panel.dock_with:
            self._dock_with_neighbor(panel.dock_with, dock_neighbor_title)
        result = imgui.begin(title, True if panel.closable else None, flags)
        return result

    @staticmethod
    def _dock_with_neighbor(name: str, translated_title: str = "") -> None:
        """Place a newly introduced panel beside an established saved-layout tab."""

        with suppress(AttributeError, TypeError):
            target = imgui.internal.find_window_by_name(translated_title or name)
            if target is None and translated_title != name:
                target = imgui.internal.find_window_by_name(name)
            if target is not None and target.dock_node is not None:
                imgui.set_next_window_dock_id(target.dock_node.id_, imgui.Cond_.first_use_ever)

    @staticmethod
    def _draw_modal(panel: Panel, ctx: PanelContext, title: str) -> None:
        if not imgui.is_popup_open(title):
            imgui.open_popup(title)
        viewport = imgui.get_main_viewport()
        width, height = panel.initial_size
        if width > 0.0 and height > 0.0:
            margin = 32.0 * ctx.style_scale
            imgui.set_next_window_size(
                imgui.ImVec2(
                    min(width * ctx.style_scale, viewport.work_size.x - margin),
                    min(height * ctx.style_scale, viewport.work_size.y - margin),
                ),
                imgui.Cond_.appearing.value,
            )
        imgui.set_next_window_pos(
            viewport.get_center(),
            imgui.Cond_.always.value,
            imgui.ImVec2(0.5, 0.5),
        )
        flags = imgui.WindowFlags_.no_docking.value | imgui.WindowFlags_.no_collapse.value
        visible, keep_open = imgui.begin_popup_modal(title, True, flags)
        if visible:
            panel.draw(ctx)
            panel.finish_frame(ctx)
            imgui.end_popup()
        if keep_open is not None and not keep_open:
            panel.open = False

    def poll_shortcuts(self, *, claimed_keys=frozenset(), keyboard_claimed: bool = False) -> None:
        if imgui.get_io().want_capture_keyboard and imgui.is_any_item_active():
            return
        if keyboard_claimed:
            return
        for p in self.panels:
            if not p.enabled:
                continue
            for spec in (p.shortcut, *p.aliases):
                if spec and spec.casefold() not in claimed_keys and _shortcut_pressed(spec):
                    p.toggle()
                    break

    def shortcut_table(self) -> tuple[tuple[str, str, bool], ...]:
        return tuple(
            (" / ".join(x for x in (p.shortcut, *p.aliases) if x), p.name, p.default_open)
            for p in self.panels
        )


def _shortcut_pressed(spec: str) -> bool:
    if spec == "?":
        return bool(imgui.get_io().key_shift) and imgui.is_key_pressed(imgui.Key.slash, False)
    key = getattr(imgui.Key, spec.lower(), None)
    return key is not None and imgui.is_key_pressed(key, False)


def validate_panels(panels: list[Panel]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, str] = {}
    names: set[str] = set()
    ids: set[str] = set()
    for p in panels:
        panel_id = _panel_id(p)
        if not panel_id:
            problems.append(f"{type(p).__name__} has no panel ID")
        elif panel_id in ids:
            problems.append(f"Duplicate panel ID: {panel_id}")
        ids.add(panel_id)
        if not p.name:
            problems.append(f"{type(p).__name__} has no name")
        elif p.name in names:
            problems.append(f"Duplicate panel name: {p.name}")
        names.add(p.name)

        for spec in (p.shortcut, *p.aliases):
            if not spec:
                continue
            if spec in seen:
                problems.append(f"Shortcut {spec} is shared by {seen[spec]} and {p.name}")
            seen[spec] = p.name
    return problems


def _panel_id(panel: Panel) -> str:
    """Resolve the stable ID, with a compatibility fallback for custom panels."""

    if panel.id:
        return str(panel.id)
    return str(panel.name).strip().casefold().replace(" ", "_")


# Kept for source compatibility; new public code should use PanelManager.
PanelSet = PanelManager


def default_panels() -> list[Panel]:
    from .assets import AssetsPanel
    from .camera import CameraPanel
    from .control import ControlPanel
    from .help import HelpPanel
    from .hierarchy import HierarchyPanel
    from .info import InfoPanel
    from .inspector import InspectorPanel
    from .joints import JointsPanel
    from .keyframes import KeyframesPanel
    from .output import OutputPanel
    from .plot import PlotPanel
    from .sensors import SensorsPanel
    from .settings import SettingsPanel
    from .stats import StatsPanel

    return [
        ControlPanel(),
        HierarchyPanel(),
        AssetsPanel(),
        InspectorPanel(),
        JointsPanel(),
        KeyframesPanel(),
        CameraPanel(),
        PlotPanel(),
        StatsPanel(),
        OutputPanel(),
        SettingsPanel(),
        SensorsPanel(),
        HelpPanel(),
        InfoPanel(),
    ]


__all__ = [
    "Panel",
    "PanelContext",
    "PanelManager",
    "PanelSet",
    "PanelState",
    "ValueEdit",
    "begin_kv_table",
    "colored_text",
    "copy_state_vector",
    "copyable_name_item",
    "default_panels",
    "is_expanded",
    "labeled",
    "publish_focus_item_hint",
    "publish_status_hint",
    "search_input",
    "segmented_control",
    "slider_gesture",
    "state_vector_text",
    "themed_checkbox",
    "validate_panels",
    "value_slider",
]
