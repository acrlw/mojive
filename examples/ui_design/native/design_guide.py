"""Browse the workspace specification through production controls and isolated edits."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from imgui_bundle import imgui as g
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore, atomic_write, read_spec
from mojive.ui.localization import Language, Localizer
from mojive.ui.scrollbar_visibility import NativeScrollbarVisibility
from mojive.ui.theme import THEME
from mojive.ui.window import WindowConfig
from mojive.ui.workspace.design import DESIGN, use_design
from mojive.ui.workspace.typography import Typography
from mojive.ui.workspace.widgets import UI

from .guide_localization import translate
from .guide_system import (
    GROUP_CONTAINERS,
    GROUPS,
    PALETTE,
    ROLES,
    VIEWER_LAYOUT,
    edit_group,
    grayscale_spec,
    proposal_spec,
    system_manifest,
)

FOUNDATIONS = (
    "Overview",
    "Typography",
    "Palette",
    "Spacing",
    "Icons",
    "Controls",
    "Corners",
    "Viewer layout",
    "Group laboratory",
)
PANELS = (
    "Scene",
    "Viewport",
    "Inspector",
    "Pose",
    "Control",
    "Assets",
    "Cameras",
    "Sensors",
    "Layers",
    "Timeline",
    "Output",
    "Statistics",
    "Curves",
    "Settings",
    "Menus",
    "Status",
)
PAGES = (*FOUNDATIONS, *PANELS)


class DesignGuide:
    def __init__(self, window, output, *, language="zh_CN", spec=None, typography=None):
        self.window, self.output = window, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.spec = proposal_spec() if spec is None else deepcopy(spec)
        self.initial_spec = deepcopy(self.spec)
        self.store = SpecificationStore(self.output / "preview.json")
        self.store.write(self.spec, save=True)
        self.runtime = DesignRuntime(self.store)
        self.ui = UI(window, portable_fonts=self.spec["tokens"]["font_family"] == "portable")
        self.ui.typography = Typography.resolve(typography)
        self.localizer = Localizer(Language(language))
        self.ui.translate = self.tr
        self.runtime.prepare_fonts(self.ui)
        self.ui.design = DesignPresentation(self.ui, self.runtime)
        self.theme = self.runtime.apply_surface(window, self.ui.s, THEME)
        self.page, self.group = "Overview", "Settings navigation"
        self.gray = False
        self.message = "Component reference · Local prototype interactions"
        self.selected = {}
        self.values = {}
        self.viewer_layout = deepcopy(VIEWER_LAYOUT)
        self.viewer_group = "Navigation buttons"
        self.hits = {}
        self.origin = (0, 0)
        self.width = 0
        self.animate = True
        self.clock = 0.0
        self.status_layout = None
        self.scrollbars = NativeScrollbarVisibility()
        self._dirty = False
        self.sample = None

    @property
    def s(self):
        return self.ui.s

    def tr(self, value, **values):
        text = translate(self.localizer, str(value))
        return text.format(**values) if values else text

    def ink(self, role):
        ink = self.runtime.ink(PALETTE[ROLES.get(role, role)])
        if self.gray:
            value = round(0.2126 * ink[0] + 0.7152 * ink[1] + 0.0722 * ink[2])
            return value, value, value, ink[3]
        return ink

    def rect(self, x, y, width, height):
        return Rect(
            self.origin[0] + x * self.s,
            self.origin[1] + y * self.s,
            width * self.s,
            height * self.s,
        )

    def text(self, x, y, text, role="text", size=14, *, bold=False, mono=False, width=None):
        self.ui.text(
            self.origin[0] + x * self.s,
            self.origin[1] + y * self.s,
            self.tr(text),
            self.ink(role),
            size,
            bold=bold,
            mono=mono,
            translate=False,
            font_resolved=True,
            color_resolved=True,
            max_width=None if width is None else width * self.s,
        )

    def box(self, x, y, width, height, role="panel", radius=8):
        self.ui.rect(*self.rect(x, y, width, height).tuple(), self.ink(role), radius)

    def line(self, x, y, width):
        r = self.rect(x, y, width, 1)
        self.ui.draw.add_rect_filled(
            (r.x, r.y),
            (r.right, r.bottom),
            g.get_color_u32(tuple(c / 255 for c in self.ink("divider"))),
        )

    def title(self, title, detail):
        self.text(0, 0, title, size=24, bold=True)
        self.text(0, 40, detail, "muted", width=self.width)

    def button(
        self,
        key,
        x,
        y,
        width,
        label="",
        *,
        icon=None,
        part="button",
        active=False,
        enabled=True,
        solid=False,
        filled=True,
        align="center",
        viewer_group=None,
        tooltip="",
    ):
        padding = self.viewer_layout[viewer_group] if viewer_group is not None else None
        insets = (
            tuple(padding["padding" + edge] for edge in ("Left", "Top", "Right", "Bottom"))
            if padding is not None
            else None
        )
        height = self.ui.design.primitives.field_height(part, DESIGN.control_height)
        if insets is not None:
            height = max(height, 20 + insets[1] + insets[3])
        result = self.ui.design.primitives.button(
            key,
            self.rect(x, y, width, height),
            self.tr(label),
            icon,
            parts=part,
            active=active,
            filled=filled and not solid,
            enabled=enabled,
            solid=solid,
            width_limit=width * self.s,
            translate=False,
            align=align,
            **(
                {"padding": tuple(value * self.s for value in insets)} if insets is not None else {}
            ),
            tooltip=self.tr(tooltip),
        )
        self.hits[key] = result.box.frame.tuple()
        self.hits[key + "/content"] = result.box.content.tuple()
        return result.changed, result.box.outer.height / self.s

    def number(self, key, x, y, width, value, *, axis=None, unit="", enabled=True):
        parts = ("input_number", "axis_input") if axis else "input_number"
        result = self.ui.design.primitives.input_number(
            key,
            self.rect(x, y, width, DESIGN.control_height),
            value,
            axis=axis,
            format="%.3g" + (" " + unit if unit else ""),
            enabled=enabled,
            parts=parts,
        )
        return result.value

    def set_group(self, field, value):
        self.spec = edit_group(self.spec, self.group, field, value)
        self._dirty = True

    def action(self, key, x, y, icon, tooltip, *, active=False, enabled=True, part="icon_button"):
        result = self.ui.design.primitives.button(
            key,
            self.rect(x, y, DESIGN.control_height, DESIGN.control_height),
            icon=icon,
            parts=part,
            active=active,
            enabled=enabled,
            tooltip=self.tr(tooltip),
        )
        self.hits[key] = result.box.frame.tuple()
        return result.changed

    def refresh(self):
        self.store.write(grayscale_spec(self.spec) if self.gray else self.spec, save=True)
        self.runtime.poll(force=True)
        self.runtime.prepare_fonts(self.ui)
        self.theme = self.runtime.apply_surface(self.window, self.s, self.theme)
        self._dirty = False

    def export(self):
        path = self.output / "proposed-spec.json"
        atomic_write(path, self.spec)
        manifest = system_manifest(self.spec)
        manifest["actual_font_faces"] = [source.label for source, _ in self.ui.font_sources]
        (self.output / "design-system.json").write_text(json.dumps(manifest, indent=2))
        (self.output / "viewer-layout.json").write_text(json.dumps(self.viewer_layout, indent=2))
        self.message = "Exported proposed-spec.json · Load explicitly in the production workbench"
        return path

    def record(self, key):
        lo, hi = g.get_item_rect_min(), g.get_item_rect_max()
        self.hits[key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)

    def viewer_select(self, key, x, y, width, value, choices):
        """Viewer-owned fields use the viewer's insets, outside sample component scopes."""
        rect = self.rect(x, y, width, g.get_frame_height() / self.s)
        result = self.ui.design.primitives.select(key, rect, value, choices)
        self.hits[key] = result.box.frame.tuple()
        return result.changed, result.value

    @contextmanager
    def region(self, key, x, y, width, height):
        """Keep independently scrollable content inside its own padded surface."""
        rect = self.rect(x, y, width, height)
        g.set_cursor_screen_pos((rect.x, rect.y))
        with self.scrollbars.panel_scope(key):
            # Reserve the gutter without painting an idle track or resizing neighbors.
            g.begin_child(
                key,
                (rect.width, rect.height),
                g.ChildFlags_.always_use_window_padding,
                g.WindowFlags_.always_vertical_scrollbar,
            )
            self.scrollbars.bind_current_window()
            origin = self.origin
            position = g.get_cursor_screen_pos()
            self.origin = (position.x, position.y)
            try:
                yield g.get_content_region_avail().x / self.s
            finally:
                self.origin = origin
                g.end_child()

    @contextmanager
    def editor_fields(self):
        """Apply guide editor padding without changing the embedded component specimen."""
        padding = self.viewer_layout["Parameter fields"]
        g.push_style_var(
            g.StyleVar_.frame_padding, (padding["paddingX"] * self.s, padding["paddingY"] * self.s)
        )
        try:
            yield
        finally:
            g.pop_style_var()

    @contextmanager
    def appearance(self):
        """Keep guide navigation spacing independent from the embedded workspace controls."""
        g.push_style_color(g.Col_.window_bg, tuple(c / 255 for c in self.ink("canvas")))
        g.push_style_color(g.Col_.child_bg, tuple(c / 255 for c in self.ink("panel")))
        g.push_style_var(
            g.StyleVar_.window_padding,
            tuple(self.viewer_layout["Content panes"]["padding" + axis] * self.s for axis in "XY"),
        )
        g.push_font(self.ui.body, DESIGN.body_size * self.s)
        try:
            yield
        finally:
            g.pop_font()
            g.pop_style_var()
            g.pop_style_color(2)

    def close(self):
        """Release the local panel session before the hosting window closes."""
        if self.sample is not None:
            self.sample.release()
            self.sample = None

    def show(self):
        """Present a completed first page before mapping the native window."""
        for _ in range(3):
            self.frame()
        self.window.show()

    def group_editor(self, x, y, width):
        part = GROUPS[self.group]
        style = self.ui.component_style(part)
        fields = self.runtime.catalog[part]["fields"]
        controls = (
            ("fontSize", "Font size", 12, 24),
            ("iconSize", "Icon size", 12, 32),
            ("width", "Button width", 16, 256),
            ("height", "Height", 24, 80),
            ("slotSize", "Tool slot", 24, 64),
            ("swatchSize", "Swatch size", 16, 64),
            ("paddingLeft", "Left padding", 0, 32),
            ("paddingRight", "Right padding", 0, 32),
            ("paddingTop", "Top padding", 0, 24),
            ("paddingBottom", "Bottom padding", 0, 24),
            ("gap", "Icon / text gap", 0, 24),
            ("columnGap", "Swatch / text", 0, 24),
            ("valueWidth", "Numeric width", 16, 80),
            ("labelGap", "Label / value", 0, 24),
            ("unitGap", "Value / unit", 0, 24),
            ("leftGap", "Left divider", 0, 24),
            ("rightGap", "Right divider", 0, 24),
        )
        controls = tuple(
            control
            for control in controls
            if control[0] in fields and (part != "assets_material" or control[0] != "gap")
        )
        container = GROUP_CONTAINERS.get(self.group)
        colors = tuple(field for field in ("textColor", "iconColor") if field in fields)
        control_height = g.get_frame_height() / self.s
        row_height = control_height + 12
        family_height = control_height + 28
        reset_padding = self.viewer_layout["Parameter buttons"]
        reset_height = max(32, 20 + reset_padding["paddingTop"] + reset_padding["paddingBottom"])
        height = (
            108
            + row_height * (len(controls) + (container is not None))
            + family_height * ("fontFamily" in fields)
            + row_height * len(colors)
            + reset_height
            + 24
        )
        self.box(x, y, width, height, "panel")
        self.text(x + 16, y + 16, "LOCAL GROUP", "tertiary", 12)
        self.text(x + 16, y + 40, self.group, size=16, bold=True, width=width - 32)
        self.text(x + 16, y + 68, part, "muted", 12, width=width - 32)
        yy = y + 108
        for field, caption, low, high in controls:
            self.text(x + 16, yy + (control_height - 14) / 2, caption, "muted", 14)
            g.set_cursor_screen_pos(
                (self.origin[0] + (x + 128) * self.s, self.origin[1] + yy * self.s)
            )
            g.set_next_item_width((width - 144) * self.s)
            changed, value = g.slider_int(
                "##guide-" + field,
                round(self.runtime.resolved_part(part, field)),
                low,
                high,
            )
            self.record("edit/" + field)
            if changed:
                self.set_group(field, value)
                if field == "width":
                    self.set_group("widthMode", "fixed")
            yy += row_height
        if container is not None:
            field = (
                "rowGap" if "rowGap" in self.runtime.catalog[container]["fields"] else "columnGap"
            )
            self.text(x + 16, yy + (control_height - 14) / 2, "Group spacing", "muted", 14)
            g.set_cursor_screen_pos(
                (self.origin[0] + (x + 128) * self.s, self.origin[1] + yy * self.s)
            )
            g.set_next_item_width((width - 144) * self.s)
            changed, value = g.slider_int(
                "##guide-group-gap", round(self.runtime.resolved_part(container, field)), 0, 32
            )
            self.record("edit/container-gap")
            if changed:
                self.set_group("container." + field, value)
            yy += row_height
        if "fontFamily" in fields:
            families = ("inherit", "system", "portable", "mono")
            self.text(x + 16, yy, "Font family", "muted", 14)
            changed, family = self.viewer_select(
                "edit/fontFamily",
                x + 16,
                yy + 20,
                width - 32,
                style.get("fontFamily", "inherit"),
                families,
            )
            if changed:
                self.set_group("fontFamily", family)
            yy += family_height
        for field, caption in (("textColor", "Text color"), ("iconColor", "Icon color")):
            if field not in fields:
                continue
            self.text(x + 16, yy + 8, caption, "muted", 14)
            value = style.get(field, "inherit")
            ink = (
                self.ink("text" if field == "textColor" else "muted")
                if value == "inherit"
                else tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
            )
            g.set_cursor_screen_pos(
                (self.origin[0] + (x + 128) * self.s, self.origin[1] + yy * self.s)
            )
            changed, rgb = g.color_edit3(
                "##guide-" + field, [c / 255 for c in ink[:3]], g.ColorEditFlags_.no_inputs
            )
            if changed:
                self.set_group(field, "#" + "".join(f"{round(c * 255):02X}" for c in rgb))
            yy += row_height
        if self.button(
            "reset-group",
            x + 16,
            yy + 8,
            width - 32,
            "Reset this group",
            viewer_group="Parameter buttons",
        )[0]:
            for target in (part, container):
                if target in self.initial_spec["parts"]:
                    self.spec["parts"][target] = deepcopy(self.initial_spec["parts"][target])
                elif target is not None:
                    self.spec["parts"].pop(target, None)
            self._dirty = True
        return yy + reset_height + 24

    def frame(self, *, readback=False):
        from .guide_foundations import draw_foundation
        from .guide_panels import draw_panel

        # Font-family installation must finish before ImGui locks the atlas.
        if self._dirty:
            self.refresh()
        if self.sample is None:
            from .guide_workspace import GuideWorkspace

            self.sample = GuideWorkspace(self)
            self.theme = self.runtime.apply_surface(self.window, self.s, self.theme)
        self.window.begin_frame()
        self.scrollbars.update()
        self.ui.s = self.window.style_scale
        self.ui.hits.clear()
        self.hits.clear()
        self.ui.design.begin_frame()
        self.runtime.begin_frame()
        if self.sample is not None:
            self.sample.prepare_frame()
        if self.animate:
            self.clock += g.get_io().delta_time
        with use_design(self.runtime.dimensions), self.appearance(), self.scrollbars.scope():
            self.draw_contents(draw_foundation, draw_panel)
            self.sample.finish_frame()
        return self.window.end_frame(readback=readback)

    def draw_contents(self, draw_foundation, draw_panel):
        self.origin = (0, 0)
        display = g.get_io().display_size
        width, height = display.x / self.s, display.y / self.s
        g.set_next_window_pos((0, 0))
        g.set_next_window_size(display)
        g.begin(
            "Mojive design guide",
            None,
            g.WindowFlags_.no_decoration
            | g.WindowFlags_.no_move
            | g.WindowFlags_.no_saved_settings
            | g.WindowFlags_.no_scrollbar
            | g.WindowFlags_.no_scroll_with_mouse,
        )
        try:
            self._draw_sections(draw_foundation, draw_panel, width, height)
        finally:
            g.end()

    def _draw_sections(self, draw_foundation, draw_panel, width, height):
        self.text(24, 20, "Mojive", "action", 20, bold=True)
        self.text(120, 22, "Design system", size=16, bold=True)
        self.text(24, 52, "DRAFT 01 · Native UI reference", "muted", 12)
        if self.button(
            "language",
            300,
            20,
            112,
            "English" if self.localizer.language is Language.SIMPLIFIED_CHINESE else "简体中文",
            viewer_group="Header actions",
        )[0]:
            self.localizer.set_language(
                Language.ENGLISH
                if self.localizer.language is Language.SIMPLIFIED_CHINESE
                else Language.SIMPLIFIED_CHINESE,
                persist=False,
            )
        padding = self.viewer_layout["Header actions"]
        action_width = max(
            128,
            max(
                self.ui.measure(self.tr(label), 14, translate=False, font_resolved=True) / self.s
                for label in ("Grayscale", "Animate", "Export draft")
            )
            + 24
            + padding["paddingLeft"]
            + padding["paddingRight"],
        )
        for index, (key, label, icon, active) in enumerate(
            (
                ("grayscale", "Grayscale", "shading", self.gray),
                ("animate", "Animate", "play", self.animate),
                ("export", "Export draft", "save", True),
            )
        ):
            if self.button(
                key,
                width - 24 - (3 * action_width + 16) + index * (action_width + 8),
                20,
                action_width,
                label,
                icon=icon,
                active=active,
                solid=key == "export",
                filled=False,
                viewer_group="Header actions",
            )[0]:
                if key == "grayscale":
                    self.gray = not self.gray
                    self._dirty = True
                elif key == "animate":
                    self.animate = not self.animate
                else:
                    self.export()
            self.hits[key] = self.ui.hits[key]
        self.line(24, 80, width - 48)
        self.line(24, height - 48, width - 48)
        self.text(24, height - 32, self.message, "muted", 12, width=width - 48)
        with self.region("Guide navigation", 24, 96, 208, height - 160) as available:
            y = 0
            for title, pages in (("FOUNDATIONS", FOUNDATIONS), ("PANEL PROTOTYPES", PANELS)):
                self.text(0, y, title, "tertiary", 12)
                y += 28
                for page in pages:
                    key = "page/" + page
                    clicked, item_height = self.button(
                        key,
                        0,
                        y,
                        available,
                        page,
                        active=self.page == page,
                        filled=False,
                        align="start",
                        viewer_group="Navigation buttons",
                    )
                    if clicked:
                        self.page = page
                    self.hits[key] = self.ui.hits[key]
                    y += item_height + self.viewer_layout["Navigation buttons"]["rowGap"]
                y += 20
            g.set_cursor_screen_pos((self.origin[0], self.origin[1] + y * self.s))
            g.dummy((available * self.s, 1))
        with self.region(
            "Guide page/" + self.page, 256, 96, width - 280, height - 160
        ) as available:
            self.width = available
            g.push_id(self.page)
            try:
                end = draw_foundation(self) if self.page in FOUNDATIONS else draw_panel(self)
                g.set_cursor_screen_pos((self.origin[0], self.origin[1] + end * self.s))
                g.dummy((self.width * self.s, 1))
            finally:
                g.pop_id()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--output", type=Path, default=Path("output/ui_design/guide"))
    parser.add_argument("--page", choices=PAGES, default="Overview")
    parser.add_argument("--language", choices=("zh_CN", "en"), default="zh_CN")
    parser.add_argument("--capture", action="store_true", help="Capture every page, then exit")
    parser.add_argument(
        "--capture-alignment",
        action="store_true",
        help="Check and capture section/body columns, pose and status typography, then exit",
    )
    parser.add_argument(
        "--capture-review",
        action="store_true",
        help="Capture focused production readouts, hints, friction fields and color popups",
    )
    parser.add_argument("--spec", type=Path, help="Preview an explicit workspace specification")
    parser.add_argument("--defaults", action="store_true", help="Preview shared factory defaults")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.width < 1000 or args.height < 700:
        parser.error("The guide requires at least 1000 x 700 logical points")
    if args.spec is not None and args.defaults:
        parser.error("--spec and --defaults are mutually exclusive")
    if sum((args.capture, args.capture_alignment, args.capture_review)) > 1:
        parser.error("Capture modes are mutually exclusive")
    typography = None
    if args.spec is not None:
        spec = read_spec(args.spec)
    elif args.capture or args.capture_alignment or args.capture_review or args.defaults:
        spec = proposal_spec()
    else:
        source = SpecificationStore().latest()
        if source["source"] is None and source["errors"]:
            parser.error(
                "No valid workspace specification: "
                + "; ".join(e["error"] for e in source["errors"])
            )
        spec = source["spec"]
        from mojive.ui.preferences import Preferences

        typography = Preferences.load().get("typography")
    with create_window(
        WindowConfig(
            title="Mojive · UI Design System",
            width=args.width,
            height=args.height,
            docking=False,
            ini_path="",
            show_on_start=False,
        ),
        args.backend,
    ) as window:
        guide = DesignGuide(
            window, args.output, language=args.language, spec=spec, typography=typography
        )
        try:
            guide.page = args.page
            if args.capture_review:
                from .guide_capture_review import capture_review

                guide.animate = False
                capture_review(guide)
                guide.export()
                print(f"Captured focused production review to {args.output}")
            elif args.capture_alignment:
                from .guide_capture_alignment import capture_alignment

                guide.animate = False
                capture_alignment(guide)
                guide.export()
                print(f"Captured production alignment checks to {args.output}")
            elif args.capture:
                report = {
                    "backend": args.backend,
                    "language": args.language,
                    "pages": {},
                    "fonts": [source.label for source, _ in guide.ui.font_sources],
                }
                guide.animate = False
                for page in PAGES:
                    guide.page = page
                    for _ in range(4):
                        pixels = guide.frame(readback=True)
                    name = page.lower().replace(" ", "-") + ".png"
                    Image.fromarray(pixels[::-1]).save(args.output / name)
                    report["pages"][page] = {"image": name, "controls": dict(guide.ui.hits)}
                guide.page = "Group laboratory"
                for group in GROUPS:
                    guide.group = group
                    for _ in range(4):
                        pixels = guide.frame(readback=True)
                    name = "group-" + group.lower().replace(" ", "-") + ".png"
                    Image.fromarray(pixels[::-1]).save(args.output / name)
                guide.page = "Viewer layout"
                for group in VIEWER_LAYOUT:
                    guide.viewer_group = group
                    for _ in range(4):
                        pixels = guide.frame(readback=True)
                    name = "viewer-" + group.lower().replace(" ", "-") + ".png"
                    Image.fromarray(pixels[::-1]).save(args.output / name)
                from .guide_capture import capture_interactions, capture_telemetry
                from .guide_capture_refinements import capture_refinements

                capture_interactions(guide)
                capture_telemetry(guide)
                capture_refinements(guide)
                guide.gray, guide.page = True, "Settings"
                guide.refresh()
                for _ in range(4):
                    pixels = guide.frame(readback=True)
                Image.fromarray(pixels[::-1]).save(args.output / "settings-grayscale.png")
                guide.export()
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
                print(f"Captured {len(PAGES)} design pages to {args.output}")
            else:
                guide.show()
                print("Mojive design guide ready: browse pages and edit local groups.", flush=True)
                while not window.should_close():
                    guide.frame()
        finally:
            guide.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
