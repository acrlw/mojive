"""Small production-widget comparisons with local, reversible style experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from imgui_bundle import imgui

from mojive.ui.controls import (
    begin_property_table,
    property_row,
    search_input,
    segmented_control,
    segmented_control_width,
)
from mojive.ui.icons import ICON_GRID, draw_icon, production_icon_metrics, production_icon_offset
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels.context import PanelContext
from mojive.ui.panels.value_cards import value_rail
from mojive.ui.theme import DEFAULT_CORNER_RADIUS, THEME

from .optical import OpticalStudy, draw_optical

CORNER_SCENES = ("Basic controls", "Inspector card", "Search & filters", "Action group")
CORNER_SCENE_KEYS = ("basic", "inspector", "filters", "actions")


@dataclass
class ComponentStudy:
    radius: float = 4.0
    inset: float = 2.0
    link_corners: bool = True
    guides: bool = False
    tab: int = 0
    optical: OpticalStudy = field(default_factory=OpticalStudy)
    search: str = ""
    position: float = 0.35
    angle: float = 0.25
    projection: int = 0
    scene: int = 0
    name: str = "Scene camera"
    enabled: bool = True
    exposure: float = 1.0
    filter_type: int = 0
    show_hidden: bool = False
    action: str = "No action selected"

    @property
    def camera_y(self) -> float:
        return self.optical.offset("helper-camera")[1]

    @camera_y.setter
    def camera_y(self, value: float) -> None:
        self.optical.offsets["helper-camera"] = (self.optical.offset("helper-camera")[0], value)


@contextmanager
def _style(*values):
    for name, value in values:
        imgui.push_style_var(name, value)
    try:
        yield
    finally:
        imgui.pop_style_var(len(values))


def _settings(study: ComponentStudy, scale: float) -> None:
    imgui.set_next_item_width(-1)
    _, study.scene = imgui.combo("##corner-scene", study.scene, CORNER_SCENES)
    controls = [
        ("Candidate radius", "radius", 0.0, 10.0, "%.1f px"),
        ("Inset (both columns)", "inset", 2.0, 12.0, "%.1f px"),
    ]
    if study.scene == 0:
        controls.append(("Camera extra Y", "camera_y", -2.0, 2.0, "%+.2f U"))
    columns = len(controls) if imgui.get_content_region_avail().x >= 760 * scale else 1
    if imgui.begin_table("##component-settings", columns):
        for label, attribute, low, high, fmt in controls:
            imgui.table_next_column()
            imgui.text(label)
            imgui.set_next_item_width(-1)
            _, value = imgui.slider_float(
                f"##component-{attribute}", getattr(study, attribute), low, high, fmt
            )
            setattr(study, attribute, value)
        imgui.end_table()
    _, study.link_corners = imgui.checkbox("Link outer radius to inset", study.link_corners)
    if study.scene == 0:
        if columns > 1:
            imgui.same_line()
        _, study.guides = imgui.checkbox("Show alignment guides", study.guides)
    if columns > 1:
        imgui.same_line()
    if imgui.button("Reset experiment"):
        defaults = ComponentStudy()
        for name in ("radius", "inset", "link_corners", "camera_y", "guides"):
            setattr(study, name, getattr(defaults, name))


def _fields(ctx: PanelContext, study: ComponentStudy) -> None:
    imgui.set_next_item_width(-1)
    _, study.search = search_input(
        "##component-search", study.search, hint="Search joints", draw=ctx.painter()
    )
    if begin_property_table("##component-fields", labels=("Position", "Angle")):
        for name, attribute, bounds, unit in (
            ("Position", "position", (-1.0, 1.0), "m"),
            ("Angle", "angle", (-1.2, 1.2), "rad"),
        ):
            property_row(name)
            edit = value_rail(
                ctx,
                f"##component-{attribute}",
                getattr(study, attribute),
                bounds,
                initial=0.0,
                unit=unit,
                fmt="%+.3f",
            )
            setattr(study, attribute, edit.value)
        imgui.end_table()


def _nested_frame(ctx: PanelContext, study: ComponentStudy, radius: float, candidate: bool):
    scale = ctx.style_scale
    outer = radius + study.inset if candidate and study.link_corners else radius
    imgui.text_disabled(f"Inner {radius:.1f} + inset {study.inset:.1f}  |  outer {outer:.1f}")
    labels = ("Perspective", "Orthographic")
    icons = ("persp", "ortho")
    inner_width = imgui.get_content_region_avail().x - 2 * study.inset * scale
    rows = 1 if segmented_control_width(labels, icons=icons) <= inner_width else len(labels)
    # This compact surface is a comparison fixture, not a replica of a viewer panel.
    # Zero border makes its WindowPadding the actual distance between the fills.
    with _style(
        (imgui.StyleVar_.child_rounding, outer * scale),
        (imgui.StyleVar_.child_border_size, 0.0),
        (imgui.StyleVar_.window_padding, (study.inset * scale, study.inset * scale)),
    ):
        imgui.push_style_color(imgui.Col_.child_bg, THEME.bg_header)
        opened = imgui.begin_child(
            "##component-nested",
            (0, rows * imgui.get_frame_height() + 2 * study.inset * scale),
            imgui.ChildFlags_.always_use_window_padding,
            imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        if opened:
            study.projection = segmented_control(
                "component-projection",
                labels,
                study.projection,
                theme=THEME,
                icons=icons,
                draw=ctx.painter(),
            )
        imgui.end_child()
        imgui.pop_style_color()


def _icons(ctx: PanelContext, study: ComponentStudy, candidate: bool) -> None:
    scale = ctx.style_scale
    imgui.text_disabled(
        f"Extra camera Y {study.camera_y:+.2f} U; Play / Reset unchanged"
        if candidate
        else "Production shapes and reviewed optical offsets"
    )
    draw = ctx.painter()
    origin = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    cell = width / 3
    size = 24 * scale
    for index, (name, label) in enumerate(
        (("playback-play", "Play"), ("playback-reset", "Reset"), ("helper-camera", "Camera"))
    ):
        center = (origin.x + cell * (index + 0.5), origin.y + 27 * scale)
        offset = study.camera_y * size / ICON_GRID if candidate and index == 2 else 0.0
        draw.circle_filled(center, 22 * scale, THEME.bg_frame)
        draw_icon(draw, (center[0], center[1] + offset), size, name, THEME.text)
        if study.guides:
            guide = (*THEME.warning[:3], 0.55)
            draw.line(
                (center[0] - 26 * scale, center[1]),
                (center[0] + 26 * scale, center[1]),
                guide,
                scale,
            )
            draw.line(
                (center[0], center[1] - 26 * scale),
                (center[0], center[1] + 26 * scale),
                guide,
                scale,
            )
            x0, y0, x1, y1 = production_icon_metrics(name).bounds
            ox, oy = production_icon_offset(name)
            unit = size / ICON_GRID
            draw.rect(
                (center[0] + (x0 + ox) * unit, center[1] + (y0 + oy) * unit + offset),
                (center[0] + (x1 + ox) * unit, center[1] + (y1 + oy) * unit + offset),
                (*THEME.text_disabled[:3], 0.6),
                scale,
            )
        text_width = draw.text_size(label)[0]
        draw.text((center[0] - text_width * 0.5, origin.y + 58 * scale), THEME.text, label)
    imgui.dummy((width, 86 * scale))


@contextmanager
def _surface(ctx, study, radius, candidate):
    scale = ctx.style_scale
    outer = radius + study.inset if candidate and study.link_corners else radius
    imgui.text_disabled(f"Inner {radius:.1f} + inset {study.inset:.1f}  |  outer {outer:.1f}")
    with _style(
        (imgui.StyleVar_.child_rounding, outer * scale),
        (imgui.StyleVar_.child_border_size, 0.0),
        (imgui.StyleVar_.window_padding, (study.inset * scale, study.inset * scale)),
    ):
        imgui.push_style_color(imgui.Col_.child_bg, THEME.bg_header)
        opened = imgui.begin_child(
            "##corner-surface",
            (0, 0),
            imgui.ChildFlags_.always_use_window_padding | imgui.ChildFlags_.auto_resize_y,
        )
        try:
            yield opened
        finally:
            imgui.end_child()
            imgui.pop_style_color()


def _inspector(ctx, study, radius, candidate):
    imgui.text("Camera / Inspector")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            imgui.set_next_item_width(-1)
            _, study.name = imgui.input_text("##corner-name", study.name)
            if begin_property_table(
                "##corner-properties", labels=("Projection", "Enabled", "Exposure")
            ):
                property_row("Projection")
                imgui.set_next_item_width(-1)
                _, study.projection = imgui.combo(
                    "##corner-projection", study.projection, ("Perspective", "Orthographic")
                )
                property_row("Enabled")
                _, study.enabled = imgui.checkbox("##corner-enabled", study.enabled)
                property_row("Exposure")
                imgui.set_next_item_width(-1)
                _, study.exposure = imgui.drag_float(
                    "##corner-exposure", study.exposure, 0.01, 0.0, 4.0
                )
                imgui.end_table()
            imgui.begin_disabled(not study.enabled)
            if imgui.button("Reset exposure", (-1, 0)):
                study.exposure = 1.0
            imgui.end_disabled()
    imgui.text_wrapped("Text, combo, checkbox and numeric input share one property surface.")


def _filters(ctx, study, radius, candidate):
    imgui.text("Scene / Search and filters")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            imgui.set_next_item_width(-1)
            _, study.search = search_input(
                "##corner-search", study.search, hint="Search scene", draw=ctx.painter()
            )
            study.filter_type = segmented_control(
                "corner-filter",
                ("All", "Cameras", "Lights"),
                study.filter_type,
                theme=THEME,
                draw=ctx.painter(),
            )
            _, study.show_hidden = imgui.checkbox("Show hidden objects", study.show_hidden)
            imgui.begin_disabled(
                not study.search and study.filter_type == 0 and not study.show_hidden
            )
            if imgui.button("Clear filters", (-1, 0)):
                study.search, study.filter_type, study.show_hidden = "", 0, False
            imgui.end_disabled()
    imgui.text_wrapped(
        "The search pill keeps its own radius; segmented controls use the local frame radius."
    )


def _actions(ctx, study, radius, candidate):
    imgui.text("Selection / Actions")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            for label in ("Focus selection", "Duplicate selection", "Delete selection"):
                if imgui.button(label, (-1, 0)):
                    study.action = label
            imgui.begin_disabled()
            imgui.button("Restore deleted object", (-1, 0))
            imgui.end_disabled()
    imgui.text_wrapped(study.action)
    imgui.text_wrapped(
        "Hover, keyboard focus and disabled states remain native. Actions only update this preview."
    )


def _column(ctx: PanelContext, study: ComponentStudy, candidate: bool) -> None:
    imgui.push_id("candidate" if candidate else "current")
    radius = study.radius if candidate else DEFAULT_CORNER_RADIUS
    with _style((imgui.StyleVar_.frame_rounding, radius * ctx.style_scale)):
        imgui.text("B / Candidate" if candidate else "A / Current widgets")
        imgui.text_disabled(f"Control radius {radius:.1f} px")
        imgui.separator()
        if study.scene == 0:
            imgui.text("Search and numeric fields")
            _fields(ctx, study)
            imgui.spacing()
            imgui.text("Nested frame specimen")
            _nested_frame(ctx, study, radius, candidate)
            imgui.spacing()
            imgui.text("Optical placement")
            _icons(ctx, study, candidate)
        else:
            (_inspector, _filters, _actions)[study.scene - 1](ctx, study, radius, candidate)
    imgui.pop_id()


def draw_components(state, scale: float) -> None:
    width = min(imgui.get_content_region_avail().x, 1040 * scale)
    if imgui.begin_child("##component-study", (width, 0)):
        _draw_study(state, scale)
    imgui.end_child()


def _draw_study(state, scale: float) -> None:
    study = state.components
    imgui.text("Component study")
    study.tab = segmented_control(
        "component-tab",
        ("Corners & controls", "Optical alignment"),
        study.tab,
        theme=THEME,
        draw=state.painter(),
    )
    ctx = PanelContext(
        None,
        None,
        theme=THEME,
        style_scale=scale,
        input_bindings=DEFAULT_INPUT_BINDINGS,
        painter=state.painter,
    )
    if study.tab == 1:
        draw_optical(ctx, study.optical)
        return
    imgui.text_wrapped(
        "Compare current Mojive widgets with local geometry changes. "
        "Both columns share values; edits stay in this probe."
    )
    _settings(study, scale)
    imgui.separator()
    columns = 2 if imgui.get_content_region_avail().x >= 800 * scale else 1
    if imgui.begin_table("##component-comparison", columns):
        for candidate in (False, True):
            imgui.table_next_column()
            _column(ctx, study, candidate)
        imgui.end_table()
    imgui.separator()
    imgui.text_wrapped(
        "The surfaces use native children and production controls with local style overrides."
    )
    if study.scene == 0:
        imgui.text_wrapped(
            "Camera extra Y adds to the reviewed production offset on the 24-unit icon grid. "
            "Hide guides to judge the actual visual balance."
        )
