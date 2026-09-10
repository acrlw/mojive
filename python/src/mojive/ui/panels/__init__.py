"""Shared panel protocols and UI controls."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ...config import PanelConfig
from ...input import physical_ctrl_super
from ..controls import (
    button_row_layout,
    button_width,
    padded_selectable,
    search_input,
    searchable_ordered_list_header,
    segmented_control,
    sort_order_button,
    sort_order_glyph,
    sort_order_tooltip,
    themed_checkbox,
)
from ..input_bindings import DEFAULT_INPUT_BINDINGS
from ..pointer_bindings import PointerAction
from ..theme import THEME, Theme
from ..viewport_widgets import ToolHint, pointer_tool_hint

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
    tracking: Any = None
    tracking_node_id: int | None = None
    track_node: Any = None
    set_camera_tracking: Any = None
    focus_node: Any = None
    focus_joint: Any = None
    request_rename: Any = None
    request_model_rename: Any = None
    request_texture_import: Any = None
    request_geometry_resource_import: Any = None
    request_model_asset_import: Any = None
    request_model_asset_replace: Any = None
    queue_model_edit: Any = None
    live_model_updates: bool = False
    set_live_model_updates: Any = None

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
    popup_owned_frame: bool = False
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
    viewport_layers: Any = None
    set_viewport_layers: Any = None
    recording_config: Any = None
    set_recording_config: Any = None
    recording: Any = None
    take_video_active: bool = False
    start_take_video: Any = None
    stop_recording: Any = None
    set_viewport_capsule_scale: Any = None
    input_bindings: Any = None
    set_input_binding: Any = None
    set_pointer_binding: Any = None
    set_navigation_preset: Any = None
    input_claim: Any = None
    reset_input_bindings: Any = None
    font_report: Any = None
    output: Any = None

    def submit(self, command: Any) -> Any:
        result = self.session.submit(command)
        if result.message:
            self.status = result.message
        return result

    def submit_model_edit(self, command: Any, completed=None) -> None:
        """Defer a rebuilding UI edit while retaining the synchronous Session API."""
        if self.queue_model_edit is not None:
            self.queue_model_edit(command, completed)
        else:
            result = self.submit(command)
            if completed is not None:
                completed(result)

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
        pointer_tool_hint(
            PointerAction.COPY_NAME,
            ctx.input_bindings or DEFAULT_INPUT_BINDINGS,
            ctx.tr("Copy name"),
            hint_id="panel.copy-name",
        ),
    )
    hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
    if not hovered:
        return False
    visible_width = max(1.0, float(available_width) - 2.0 * imgui.get_style().frame_padding.x)
    if imgui.calc_text_size(name).x > visible_width:
        imgui.set_tooltip(name)
    copied = pointer_pressed(ctx, PointerAction.COPY_NAME)
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

    if hint is None:
        return
    if hint.hint_id and any(existing.hint_id == hint.hint_id for existing in ctx.status_hints):
        return
    if hint not in ctx.status_hints:
        ctx.status_hints = (*ctx.status_hints, hint)


def activate_edit_gizmo(ctx: PanelContext, node, joint=None) -> None:
    """Explicit panel editing reveals a supported tool; plain selection stays passive."""
    if ctx.gizmo is None or (ctx.interactions is not None and not ctx.interactions.gizmo):
        return
    mode = "rotate" if joint is not None and joint.type in {"hinge", "ball"} else "translate"
    if ctx.gizmo.evaluate_mode(ctx.session, node, mode).ok:
        ctx.gizmo.set_mode(mode)


def publish_focus_item_hint(ctx: PanelContext) -> None:
    """Advertise the shared hierarchy/joint double-click focus gesture."""

    publish_status_hint(
        ctx,
        pointer_tool_hint(
            PointerAction.PANEL_FOCUS,
            ctx.input_bindings or DEFAULT_INPUT_BINDINGS,
            ctx.tr("Focus item"),
            hint_id="panel.focus-item",
        ),
    )


def pointer_pressed(ctx: PanelContext, action: PointerAction) -> bool:
    bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
    frame = (
        bindings.pointer_frame()
        if ctx.input_claim is None
        else bindings.pointer_frame(ctx.input_claim)
    )
    return bindings.pointer_match(action, frame, press=True) is not None


def pointer_hint(ctx: PanelContext, action: PointerAction, label: str) -> str:
    bindings = ctx.input_bindings or DEFAULT_INPUT_BINDINGS
    return f"{bindings.pointer_label(action)} · {label}"


@dataclass
class ValueEdit:
    changed: bool = False
    value: float = 0.0
    copied: bool = False
    expanded: bool = False
    activated: bool = False


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
    bindings=None,
) -> ValueEdit:
    if width:
        imgui.set_next_item_width(width)
    changed, new_value = imgui.slider_float(label, value, lo, hi, fmt)
    return value_edit(
        label,
        value,
        changed,
        new_value,
        initial=initial,
        fmt=fmt,
        more_hint=more_hint,
        bindings=bindings,
    )


def value_edit(
    label, value, changed, new_value, *, initial=None, fmt="%.4f", more_hint="", bindings=None
):
    """Apply mapped value actions to the last submitted numeric widget."""
    bindings = bindings or DEFAULT_INPUT_BINDINGS
    action = None
    if imgui.is_item_hovered():
        frame = bindings.pointer_frame()
        for candidate, name in (
            (PointerAction.VALUE_RESET, "reset"),
            (PointerAction.VALUE_COPY, "copy"),
            (PointerAction.VALUE_EXPAND, "expand"),
        ):
            if bindings.pointer_match(candidate, frame, press=True) is not None:
                action = name
                break
    out = ValueEdit(
        changed=changed,
        value=new_value,
        expanded=label in _EXPANDED,
        activated=imgui.is_item_activated(),
    )

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

    tooltip = f"{bindings.pointer_label(PointerAction.VALUE_RESET)}: reset · {bindings.pointer_label(PointerAction.VALUE_COPY)}: copy"
    if more_hint:
        tooltip += f" · {bindings.pointer_label(PointerAction.VALUE_EXPAND)}: {more_hint}"
    imgui.set_item_tooltip(tooltip)
    return out


def is_expanded(label: str) -> bool:
    return label in _EXPANDED


def colored_text(color: tuple[float, float, float, float], text: str) -> None:
    imgui.text_colored(imgui.ImVec4(*color), text)


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
        if open and hasattr(panel, "collapsed"):
            panel.collapsed = False
        return True

    def open(self, panel_id: str) -> bool:
        return self.set_open(panel_id, True)

    def close(self, panel_id: str) -> bool:
        return self.set_open(panel_id, False)

    def toggle(self, panel_id: str) -> bool:
        panel = self.get(panel_id)
        return (
            False
            if panel is None
            else self.set_open(panel_id, bool(getattr(panel, "collapsed", False)) or not panel.open)
        )

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
            if not p.enabled or not p.open or getattr(p, "collapsed", False):
                p.finish_frame(ctx)
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
        io = imgui.get_io()
        ctrl, super_key = physical_ctrl_super(io)
        if io.want_capture_keyboard and imgui.is_any_item_active():
            return
        if keyboard_claimed:
            return
        if any(
            held and key in claimed_keys
            for key, held in (
                ("ctrl", ctrl),
                ("super", super_key),
                ("alt", io.key_alt),
                ("shift", io.key_shift),
            )
        ):
            return
        for p in self.panels:
            if not p.enabled:
                continue
            for spec in (p.shortcut, *p.aliases):
                key = "slash" if spec == "?" else spec.casefold()
                if (
                    spec
                    and key not in claimed_keys
                    and spec.casefold() not in claimed_keys
                    and _shortcut_pressed(spec)
                ):
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
    from .layers import LayersPanel
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
        LayersPanel(),
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
    "button_row_layout",
    "button_width",
    "colored_text",
    "copy_state_vector",
    "copyable_name_item",
    "default_panels",
    "is_expanded",
    "labeled",
    "padded_selectable",
    "publish_focus_item_hint",
    "publish_status_hint",
    "search_input",
    "searchable_ordered_list_header",
    "segmented_control",
    "slider_gesture",
    "sort_order_button",
    "sort_order_glyph",
    "sort_order_tooltip",
    "state_vector_text",
    "themed_checkbox",
    "validate_panels",
    "value_slider",
]
