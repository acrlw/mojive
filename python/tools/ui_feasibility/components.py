"""Small production-widget comparisons with local, reversible style experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from imgui_bundle import imgui

from mojive.adapters.base import NodeType, SceneNode
from mojive.ui.controls import (
    begin_property_table,
    property_row,
    search_input,
    segmented_control,
)
from mojive.ui.icons import ICON_GRID, draw_icon, production_icon_metrics, production_icon_offset
from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS
from mojive.ui.panels.context import PanelContext
from mojive.ui.panels.inspector.fields import _property_color_edit3, _vector_fields
from mojive.ui.panels.value_cards import value_card, value_rail
from mojive.ui.theme import DEFAULT_CORNER_RADIUS, THEME

from .optical import OpticalStudy, draw_optical, draw_optical_settings

CORNER_SCENES = (
    "Basic controls",
    "Inspector properties",
    "Search & filters",
    "Action group",
    "Transform properties",
    "Material editor",
    "Scene browser",
    "Joint controls",
    "Capture settings",
)
CORNER_SCENE_KEYS = (
    "basic",
    "inspector",
    "filters",
    "actions",
    "transform",
    "material",
    "scene",
    "joints",
    "capture",
)
_TRANSFORM_NODE = SceneNode(0, "Arm base", NodeType.LINK)
_SCENE_OBJECTS = (
    ("Floor", "Geometry"),
    ("Arm base", "Body"),
    ("Scene camera", "Camera"),
    ("Key light", "Light"),
    ("Target", "Body"),
)
_JOINTS = (
    ("Shoulder", (-1.5, 1.5), "rad"),
    ("Elbow", (-2.0, 2.0), "rad"),
    ("Gripper", (0.0, 0.08), "m"),
)


@dataclass
class ComponentStudy:
    radius: float = 4.0
    card_background: bool = False
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
    transform_position: tuple[float, ...] = (0.15, -0.2, 0.6)
    transform_rotation: tuple[float, ...] = (0.0, 30.0, 0.0)
    transform_scale: tuple[float, ...] = (1.0, 1.0, 1.0)
    transform_locked: bool = False
    material_name: str = "Sage enamel"
    material_color: tuple[float, ...] = (0.44, 0.58, 0.38)
    material_roughness: float = 0.35
    material_metallic: float = 0.15
    material_opacity: float = 1.0
    material_double_sided: bool = False
    scene_search: str = ""
    scene_filter: int = 0
    scene_show_hidden: bool = True
    scene_selected: int = 2
    scene_visible: list[bool] = field(default_factory=lambda: [True, True, True, False, True])
    joint_values: list[float] = field(default_factory=lambda: [0.35, -0.65, 0.04])
    joint_selected: int = 1
    joints_locked: bool = False
    capture_name: str = "camera-study"
    capture_format: int = 0
    capture_size: int = 1
    capture_quality: int = 90
    capture_transparent: bool = False
    capture_overlays: bool = True
    capture_message: str = "Adjust settings, then preview the output summary."

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
    if imgui.begin_table("##corner-navigation", 2):
        imgui.table_next_column()
        if imgui.button("Previous example", (-1, 0)):
            study.scene = (study.scene - 1) % len(CORNER_SCENES)
        imgui.table_next_column()
        if imgui.button("Next example", (-1, 0)):
            study.scene = (study.scene + 1) % len(CORNER_SCENES)
        imgui.end_table()
    imgui.text_disabled(f"Example {study.scene + 1} / {len(CORNER_SCENES)}")
    imgui.separator()
    imgui.text("Candidate surface")
    study.card_background = bool(
        segmented_control(
            "corner-surface-mode", ("Panel", "Card"), int(study.card_background), theme=THEME
        )
    )
    imgui.set_item_tooltip(
        "Panel keeps sections unfilled. Card adds an experimental background to B only."
    )
    controls = [("Control radius / B", "radius", 0.0, 10.0, "%.1f px")]
    if study.card_background:
        controls.append(("Card inset / B", "inset", 2.0, 12.0, "%.1f px"))
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
    if study.card_background:
        _, study.link_corners = imgui.checkbox("Link card radius to inset", study.link_corners)
        outer = study.radius + study.inset if study.link_corners else study.radius
        imgui.text_disabled(f"Card radius {outer:.1f} px")
    if study.scene == 0:
        if columns > 1:
            imgui.same_line()
        _, study.guides = imgui.checkbox("Show alignment guides", study.guides)
    if columns > 1:
        imgui.same_line()
    if imgui.button("Reset experiment"):
        defaults = ComponentStudy()
        for name in ("radius", "card_background", "inset", "link_corners", "camera_y", "guides"):
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
    with _surface(ctx, study, radius, candidate, item_id="##component-nested") as opened:
        if opened:
            study.projection = segmented_control(
                "component-projection",
                ("Perspective", "Orthographic"),
                study.projection,
                theme=THEME,
                icons=("persp", "ortho"),
                draw=ctx.painter(),
            )


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
def _surface(
    ctx, study, radius, candidate, *, item_id="##corner-surface", background=THEME.bg_header
):
    scale = ctx.style_scale
    card = candidate and study.card_background
    inset = study.inset if card else 0.0
    outer = (radius + inset if study.link_corners else radius) if card else 0.0
    # Keep child IDs stable across modes, but plain sections have no fill or inset.
    # The reference column never gains the experimental card's geometry.
    with _style(
        (imgui.StyleVar_.child_rounding, outer * scale),
        (imgui.StyleVar_.child_border_size, 0.0),
        (imgui.StyleVar_.window_padding, (inset * scale, inset * scale)),
    ):
        imgui.push_style_color(imgui.Col_.child_bg, background)
        opened = imgui.begin_child(
            item_id,
            (0, 0),
            imgui.ChildFlags_.always_use_window_padding | imgui.ChildFlags_.auto_resize_y,
            imgui.WindowFlags_.none if card else imgui.WindowFlags_.no_background,
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


def _transform(ctx, study, radius, candidate):
    imgui.text("Arm base / Transform")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            _, study.transform_locked = imgui.checkbox("Lock transform", study.transform_locked)
            edits = _vector_fields(
                ctx,
                _TRANSFORM_NODE,
                "##corner-transform",
                (
                    ("Position", study.transform_position, 0.01, "%.3f", None),
                    ("Rotation", study.transform_rotation, 0.5, "%.1f", None),
                    ("Scale", study.transform_scale, 0.01, "%.3f", (1, 1, 1)),
                ),
                editable=not study.transform_locked,
            )
            for attribute, (changed, values) in zip(
                ("transform_position", "transform_rotation", "transform_scale"), edits, strict=True
            ):
                if changed:
                    setattr(study, attribute, tuple(values))
            imgui.separator()
            imgui.begin_disabled(study.transform_locked)
            if imgui.button("Reset transform", (-1, 0)):
                study.transform_position = study.transform_rotation = (0.0, 0.0, 0.0)
                study.transform_scale = (1.0, 1.0, 1.0)
            imgui.end_disabled()
    imgui.text_wrapped(
        "Production XYZ fields adapt to panel width. Click an axis badge to reset that value. "
        "Position uses meters; rotation uses degrees."
    )


def _material(ctx, study, radius, candidate):
    imgui.text("Surface / Material")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            imgui.set_next_item_width(-1)
            _, study.material_name = imgui.input_text("##corner-material-name", study.material_name)
            if begin_property_table(
                "##corner-material", labels=("Base color", "Roughness", "Metallic")
            ):
                property_row("Base color")
                _, study.material_color = _property_color_edit3(
                    ctx, "##corner-material-color", study.material_color
                )
                for label, attribute in (
                    ("Roughness", "material_roughness"),
                    ("Metallic", "material_metallic"),
                ):
                    property_row(label)
                    _, value = imgui.slider_float(
                        f"##corner-{attribute}", getattr(study, attribute), 0.0, 1.0, "%.2f"
                    )
                    setattr(study, attribute, value)
                imgui.end_table()
            if imgui.collapsing_header("Surface options", imgui.TreeNodeFlags_.default_open):
                _, study.material_double_sided = imgui.checkbox(
                    "Double sided", study.material_double_sided
                )
                imgui.text("Opacity")
                imgui.set_next_item_width(-1)
                _, study.material_opacity = imgui.slider_float(
                    "##corner-opacity", study.material_opacity, 0.0, 1.0, "%.2f"
                )
            if imgui.button("Reset material", (-1, 0)):
                defaults = ComponentStudy()
                for attribute in (
                    "material_name",
                    "material_color",
                    "material_roughness",
                    "material_metallic",
                    "material_opacity",
                    "material_double_sided",
                ):
                    setattr(study, attribute, getattr(defaults, attribute))
    imgui.text_wrapped("Color picker, sliders and a collapsible property section.")


def _scene_browser(ctx, study, radius, candidate):
    imgui.text("Scene / Objects")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            imgui.set_next_item_width(-1)
            _, study.scene_search = search_input(
                "##corner-object-search",
                study.scene_search,
                hint="Search objects",
                draw=ctx.painter(),
            )
            study.scene_filter = segmented_control(
                "corner-object-filter",
                ("All", "Cameras", "Lights"),
                study.scene_filter,
                theme=THEME,
                draw=ctx.painter(),
            )
            _, study.scene_show_hidden = imgui.checkbox("Include hidden", study.scene_show_hidden)
            visible_count = 0
            imgui.push_style_color(
                imgui.Col_.header,
                THEME.bg_frame_active if candidate and study.card_background else THEME.bg_header,
            )
            imgui.push_style_var(imgui.StyleVar_.selectable_text_align, (0, 0.5))
            if imgui.begin_table("##corner-objects", 2, imgui.TableFlags_.row_bg):
                imgui.table_setup_column("Object", imgui.TableColumnFlags_.width_stretch)
                imgui.table_setup_column(
                    "Visible", imgui.TableColumnFlags_.width_fixed, imgui.get_frame_height()
                )
                for index, (name, kind) in enumerate(_SCENE_OBJECTS):
                    if study.scene_search.casefold() not in name.casefold():
                        continue
                    if study.scene_filter and kind != ("", "Camera", "Light")[study.scene_filter]:
                        continue
                    if not study.scene_show_hidden and not study.scene_visible[index]:
                        continue
                    visible_count += 1
                    imgui.table_next_row()
                    imgui.table_next_column()
                    if not study.scene_visible[index]:
                        imgui.push_style_color(imgui.Col_.text, THEME.text_disabled)
                    clicked, _ = imgui.selectable(
                        f"{name}##corner-object-{index}",
                        study.scene_selected == index,
                        size=(0, imgui.get_frame_height()),
                    )
                    if not study.scene_visible[index]:
                        imgui.pop_style_color()
                    if clicked:
                        study.scene_selected = index
                    imgui.set_item_tooltip(kind)
                    imgui.table_next_column()
                    _, study.scene_visible[index] = imgui.checkbox(
                        f"##corner-visible-{index}", study.scene_visible[index]
                    )
                    imgui.set_item_tooltip(f"Show {name}")
                imgui.end_table()
            imgui.pop_style_var()
            imgui.pop_style_color()
            if not visible_count:
                imgui.text_wrapped("No matching objects. Clear the search or change the filter.")
            imgui.text_disabled(f"{visible_count} of {len(_SCENE_OBJECTS)} objects")
            if imgui.button("Show all objects", (-1, 0)):
                study.scene_search, study.scene_filter, study.scene_show_hidden = "", 0, True
                study.scene_visible[:] = [True] * len(_SCENE_OBJECTS)
    imgui.spacing()
    name, kind = _SCENE_OBJECTS[study.scene_selected]
    imgui.text(f"Selection / {name}")
    with _surface(ctx, study, radius, candidate, item_id="##corner-selection") as opened:
        if opened and begin_property_table(
            "##corner-selection-fields", labels=("Type", "Visibility")
        ):
            property_row("Type")
            imgui.text(kind)
            property_row("Visibility")
            imgui.text("Visible" if study.scene_visible[study.scene_selected] else "Hidden")
            imgui.end_table()
    imgui.text_wrapped(
        "Filter, select and toggle visibility to compare populated and empty list states."
    )


def _joints(ctx, study, radius, candidate):
    imgui.text("Robot arm / Joints")
    _, study.joints_locked = imgui.checkbox("Lock joint editing", study.joints_locked)
    with _surface(ctx, study, radius, candidate, background=THEME.bg_child) as opened:
        if opened:
            for index, (name, bounds, unit) in enumerate(_JOINTS):
                with value_card(
                    ctx,
                    f"##corner-joint-card-{index}",
                    name,
                    f"{bounds[0]:g} to {bounds[1]:g} {unit}",
                    selected=study.joint_selected == index,
                ) as (selected, _):
                    if selected:
                        study.joint_selected = index
                    imgui.begin_disabled(study.joints_locked)
                    study.joint_values[index] = value_rail(
                        ctx,
                        f"##corner-joint-{index}",
                        study.joint_values[index],
                        bounds,
                        initial=0.0,
                        unit=unit,
                        fmt="%+.3f",
                    ).value
                    imgui.end_disabled()
            imgui.separator()
            imgui.begin_disabled(study.joints_locked)
            if imgui.button("Reset joint values", (-1, 0)):
                study.joint_values[:] = [0.0] * len(_JOINTS)
            imgui.end_disabled()
    imgui.text_wrapped(
        "Production value cards combine selection, a bounded rail, numeric entry, units and reset."
    )


def _capture(ctx, study, radius, candidate):
    imgui.text("Camera / Capture")
    sizes = ("1280 x 720", "1920 x 1080", "3840 x 2160")
    formats = ("PNG", "JPEG")
    with _surface(ctx, study, radius, candidate) as opened:
        if opened:
            imgui.set_next_item_width(-1)
            _, study.capture_name = imgui.input_text_with_hint(
                "##corner-capture-name", "Output name", study.capture_name
            )
            if begin_property_table("##corner-capture", labels=("Format", "Resolution", "Quality")):
                property_row("Format")
                study.capture_format = segmented_control(
                    "corner-capture-format",
                    formats,
                    study.capture_format,
                    theme=THEME,
                    draw=ctx.painter(),
                )
                property_row("Resolution")
                _, study.capture_size = imgui.combo(
                    "##corner-capture-size", study.capture_size, sizes
                )
                property_row("Quality")
                imgui.begin_disabled(study.capture_format == 0)
                _, study.capture_quality = imgui.slider_int(
                    "##corner-capture-quality", study.capture_quality, 1, 100, "%d %%"
                )
                imgui.end_disabled()
                imgui.end_table()
            if imgui.collapsing_header("Image options", imgui.TreeNodeFlags_.default_open):
                imgui.begin_disabled(study.capture_format != 0)
                _, study.capture_transparent = imgui.checkbox(
                    "Transparent background", study.capture_transparent
                )
                imgui.end_disabled()
                _, study.capture_overlays = imgui.checkbox(
                    "Include overlays", study.capture_overlays
                )
            imgui.begin_disabled(not study.capture_name.strip())
            if imgui.button("Preview capture", (-1, 0)):
                extension = ("png", "jpg")[study.capture_format]
                study.capture_message = (
                    f"{study.capture_name.strip()}.{extension} / {sizes[study.capture_size]} / "
                    + (
                        "Transparent"
                        if study.capture_format == 0 and study.capture_transparent
                        else "Opaque"
                    )
                    + (" / Overlays" if study.capture_overlays else " / No overlays")
                )
            imgui.end_disabled()
    imgui.spacing()
    imgui.text_wrapped(study.capture_message)
    imgui.text_wrapped(
        "PNG enables transparency; JPEG enables quality. Preview only; no file is written."
    )


def _column(ctx: PanelContext, study: ComponentStudy, candidate: bool) -> None:
    imgui.push_id("candidate" if candidate else "current")
    radius = study.radius if candidate else DEFAULT_CORNER_RADIUS
    with _style((imgui.StyleVar_.frame_rounding, radius * ctx.style_scale)):
        imgui.text("B / Candidate controls" if candidate else "A / Default controls")
        imgui.text_disabled(f"Control radius {radius:.1f} px")
        imgui.separator()
        if study.scene == 0:
            imgui.text("Search and numeric fields")
            _fields(ctx, study)
            imgui.spacing()
            imgui.text("Projection selector")
            _nested_frame(ctx, study, radius, candidate)
            imgui.spacing()
            imgui.text("Optical placement")
            _icons(ctx, study, candidate)
        else:
            (
                _inspector,
                _filters,
                _actions,
                _transform,
                _material,
                _scene_browser,
                _joints,
                _capture,
            )[study.scene - 1](ctx, study, radius, candidate)
    imgui.pop_id()


def draw_components(state, scale: float) -> None:
    if imgui.begin_child(
        "##component-study",
        (0, 0),
        window_flags=imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
    ):
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
    available = imgui.get_content_region_avail()
    side_by_side = available.x >= 980 * scale
    controls_height = (620 if study.tab == 1 else 450) * scale
    controls_size = (
        (300 * scale, 0) if side_by_side else (0, min(controls_height, available.y * 0.42))
    )
    # Only the panes scroll: changing a parameter must never move the preview
    # or remove its controls from view. Each tab retains its own scroll positions.
    imgui.push_id(study.tab)
    previous_scene = study.scene
    if imgui.begin_child(
        "##study-controls", controls_size, imgui.ChildFlags_.always_use_window_padding
    ):
        if study.tab == 1:
            draw_optical_settings(study.optical)
        else:
            imgui.text("Panel example")
            _settings(study, scale)
            imgui.text_wrapped("Both previews share values. Style changes stay in this probe.")
    imgui.end_child()
    if side_by_side:
        imgui.same_line()
    if study.scene != previous_scene:
        imgui.set_next_window_scroll((0, 0))
    imgui.push_style_color(
        imgui.Col_.child_bg, THEME.bg_window if study.tab == 0 else THEME.bg_child
    )
    if imgui.begin_child("##study-preview", (0, 0), imgui.ChildFlags_.always_use_window_padding):
        if study.tab == 1:
            draw_optical(ctx, study.optical)
        else:
            _draw_corner_preview(ctx, study)
    imgui.end_child()
    imgui.pop_style_color()
    imgui.pop_id()


def _draw_corner_preview(ctx: PanelContext, study: ComponentStudy) -> None:
    imgui.text_wrapped(
        "Sample layouts using Mojive controls. A keeps the default style; B previews your changes."
    )
    imgui.separator()
    columns = 2 if imgui.get_content_region_avail().x >= 560 * ctx.style_scale else 1
    if imgui.begin_table("##component-comparison", columns):
        for candidate in (False, True):
            imgui.table_next_column()
            _column(ctx, study, candidate)
        imgui.end_table()
    imgui.separator()
    imgui.text_wrapped(
        "Card experiment: only B adds section backgrounds and inset. A stays on the panel surface."
        if study.card_background
        else "Panel mode: sections share the panel background. Only controls and interaction states have fills."
    )
    if study.scene == 0:
        imgui.text_wrapped(
            "Camera extra Y adds to the reviewed production offset on the 24-unit icon grid. "
            "Hide guides to judge the actual visual balance."
        )
