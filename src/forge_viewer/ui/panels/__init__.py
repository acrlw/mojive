"""Shared panel protocols and UI controls."""

from __future__ import annotations

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

    theme: Theme = THEME
    gizmo: Any = None
    perturb: Any = None

    style_scale: float = 1.0

    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    dt: float = 0.0

    info: dict[str, Any] = field(default_factory=dict)

    status: str = ""

    panels: Any = None

    def submit(self, command: Any) -> Any:
        result = self.session.submit(command)
        if result.message:
            self.status = result.message
        return result


class Panel:
    name: str = ""

    default_open: bool = True
    shortcut: str = ""

    aliases: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.open = self.default_open

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def needs(self) -> FrameNeeds:
        return self.frame_needs() if self.open else FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        raise NotImplementedError

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
            expanded, keep_open = imgui.begin(p.name, True)
            if expanded:
                p.draw(ctx)
            imgui.end()
            if keep_open is not None and not keep_open:
                p.open = False

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
    from .camera import CameraPanel
    from .control import ControlPanel
    from .help import HelpPanel
    from .hierarchy import HierarchyPanel
    from .info import InfoPanel
    from .inspector import InspectorPanel
    from .joints import JointsPanel
    from .plot import PlotPanel
    from .sensors import SensorsPanel
    from .settings import SettingsPanel
    from .stats import StatsPanel

    return [
        ControlPanel(),
        HierarchyPanel(),
        InspectorPanel(),
        JointsPanel(),
        CameraPanel(),
        PlotPanel(),
        StatsPanel(),
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
