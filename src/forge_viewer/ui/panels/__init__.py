"""Shared panel protocols and UI controls."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from ..theme import THEME, Theme

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
    request_rename: Any = None
    request_texture_import: Any = None
    request_geometry_resource_import: Any = None
    request_model_asset_import: Any = None
    request_model_asset_replace: Any = None

    theme: Theme = THEME
    gizmo: Any = None
    view_cube: Any = None
    perturb: Any = None
    scene_entities: Any = None

    style_scale: float = 1.0

    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    dt: float = 0.0

    info: dict[str, Any] = field(default_factory=dict)

    status: str = ""

    panels: Any = None

    language: str = "en"
    translate: Any = None
    set_language: Any = None
    set_precise_input_memory: Any = None
    set_view_selection_padding: Any = None
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
        self.open = self.default_open

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def needs(self) -> FrameNeeds:
        return self.frame_needs() if self.open else FrameNeeds.none()

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
    elif action == "expand":
        if label in _EXPANDED:
            _EXPANDED.discard(label)
        else:
            _EXPANDED.add(label)
        out.expanded = label in _EXPANDED

    imgui.set_item_tooltip(f"Right-click reset · Double-click copy · Shift+right-click {more_hint}")
    return out


def is_expanded(label: str) -> bool:
    return label in _EXPANDED


def colored_text(color: tuple[float, float, float, float], text: str) -> None:
    imgui.text_colored(imgui.ImVec4(*color), text)


def labeled(label: str, value: str) -> None:
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text_disabled(label)
    imgui.table_next_column()
    imgui.text(value)


def begin_kv_table(str_id: str) -> bool:
    return imgui.begin_table(
        str_id, 2, imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.row_bg
    )


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


class PanelSet:
    def __init__(self, panels: list[Panel] | None = None) -> None:
        self.panels: list[Panel] = list(panels) if panels is not None else default_panels()
        problems = validate_panels(self.panels)
        if problems:
            raise ValueError("Invalid panel configuration: " + "; ".join(problems))

    def __iter__(self):
        return iter(self.panels)

    def get(self, name: str) -> Panel | None:
        return next((p for p in self.panels if p.name == name), None)

    def open_panel(self, name: str) -> None:
        panel = self.get(name)
        if panel is not None:
            panel.open = True

    def frame_needs(self) -> FrameNeeds:
        needs = FrameNeeds.none()
        for p in self.panels:
            needs = needs.merge(p.needs())
        return needs

    def draw(self, ctx: PanelContext) -> None:
        ctx.panels = self
        for p in self.panels:
            if not p.open:
                continue
            translated = ctx.tr(p.name)
            title = p.name if translated == p.name else f"{translated}###{p.name}"
            if p.modal:
                self._draw_modal(p, ctx, title)
                continue
            expanded, keep_open = self._begin_panel_window(
                p,
                title,
                ctx.style_scale,
                self._translated_panel_title(ctx.tr, p.dock_with),
            )
            if expanded:
                p.draw(ctx)
            p.finish_frame(ctx)
            imgui.end()
            if keep_open is not None and not keep_open:
                p.open = False

    def draw_shells(self, translate, style_scale: float) -> None:
        """Submit docked panel windows without reading application state."""

        for panel in self.panels:
            if not panel.open or panel.modal:
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
        return imgui.begin(title, True if panel.closable else None, flags)

    @staticmethod
    def _dock_with_neighbor(name: str, translated_title: str = "") -> None:
        """Place a newly introduced panel beside an established saved-layout tab."""

        with suppress(AttributeError, TypeError):
            target = imgui.internal.find_window_by_name(translated_title or name)
            if target is None and translated_title != name:
                target = imgui.internal.find_window_by_name(name)
            if target is not None and target.dock_node is not None:
                imgui.set_next_window_dock_id(target.dock_node.id, imgui.Cond_.first_use_ever)

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

    def poll_shortcuts(self) -> None:
        if imgui.get_io().want_capture_keyboard and imgui.is_any_item_active():
            return
        for p in self.panels:
            for spec in (p.shortcut, *p.aliases):
                if spec and _shortcut_pressed(spec):
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
    for p in panels:
        if not p.name:
            problems.append(f"{type(p).__name__} has no name")
        elif p.name in names:
            problems.append(f"Duplicate panel name: {p.name}")
        names.add(p.name)

        if not p.default_open and not p.shortcut:
            problems.append(f"{p.name} is closed by default and has no shortcut")

        for spec in (p.shortcut, *p.aliases):
            if not spec:
                continue
            if spec in seen:
                problems.append(f"Shortcut {spec} is shared by {seen[spec]} and {p.name}")
            seen[spec] = p.name
    return problems


def default_panels() -> list[Panel]:
    from .assets import AssetsPanel
    from .camera import CameraPanel
    from .control import ControlPanel
    from .help import HelpPanel
    from .hierarchy import HierarchyPanel
    from .info import InfoPanel
    from .inspector import InspectorPanel
    from .joints import JointsPanel
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
    "PanelSet",
    "ValueEdit",
    "begin_kv_table",
    "colored_text",
    "default_panels",
    "is_expanded",
    "labeled",
    "slider_gesture",
    "validate_panels",
    "value_slider",
]
