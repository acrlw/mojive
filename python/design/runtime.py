"""Resolve one workspace's live design values without changing module globals."""

import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

from mojive.design.specification import BASELINE, DEFAULT, component_catalog, resolve_part_value
from mojive.design.store import SpecificationStore

TOKEN_METRICS = {
    "body_size": "body_font_size",
    "title_size": "title_font_size",
    "mono_size": "mono_font_size",
    "control_height": "control_height",
    "control_radius": "control_radius",
    "container_radius": "container_radius",
    "panel_padding_x": "panel_padding",
    "panel_padding_y": "panel_padding_y",
    "frame_padding_x": "frame_padding_x",
    "frame_padding_y": "frame_padding_y",
    "item_spacing_x": "item_spacing_x",
    "row_gap": "row_gap",
    "numeric_gap": "numeric_gap",
    "label_width": "label_width",
    "label_minimum": "label_minimum",
    "label_gap": "label_gap",
    "grab_radius": "radius_grab",
    "child_radius": "radius_child",
}
PART_METRICS = {
    "checkbox_size": ("checkbox", "size"),
    "switch_track_width": ("switch", "switchWidth"),
    "switch_track_height": ("switch", "switchHeight"),
    "switch_thumb_size": ("switch", "switchThumb"),
    "application_height": ("titlebar", "height"),
    "sidebar_width": ("rail", "width"),
    "timeline_column_gap": ("timeline_toolbar", "columnGap"),
    "timeline_row_gap": ("timeline_toolbar", "rowGap"),
    "timeline_lane_height": ("timeline_lane", "height"),
    "timeline_ruler_size": ("timeline_ruler", "fontSize"),
    "timeline_lane_size": ("timeline_lane", "fontSize"),
    "timeline_label_gap": ("lane_label", "columnGap"),
    "timeline_sheet_gap": ("timeline_sheet", "rowGap"),
    "output_row_gap": ("output_list", "rowGap"),
    "menu_item_height": ("menu_item", "height"),
    "menu_row_gap": ("menu", "rowGap"),
    "menu_padding_x": ("menu_item", "paddingX"),
    "navigation_size": ("nav_widget", "size"),
    "transport_padding_y": ("transport", "paddingY"),
    "range_thumb_size": ("range", "thumbSize"),
    "range_track_size": ("range", "trackSize"),
    "data_row_gap": ("data_list", "rowGap"),
    "camera_right": ("camera_preview", "right"),
    "camera_bottom": ("camera_preview", "bottom"),
    "file_receipt_padding": ("file_receipt", "paddingX"),
    "rail_entry_width": ("range_number", "numberWidth"),
}
PALETTE_TOKENS = {
    "text": "text_color",
    "secondary": "muted_color",
    "disabled": "dim_color",
    "primary": "sage_color",
    "warning": "amber_color",
    "background": "panel_color",
    "floating": "floating_color",
    "popup": "popup_color",
    "field": "input_color",
}
THEME_ROLES = {
    "text": "text",
    "text_disabled": "disabled",
    "primary": "primary",
    "bg_window": "background",
    "bg_child": "background",
    "bg_popup": "popup",
    "bg_frame": "field",
    "warning": "warning",
}


@lru_cache(maxsize=256)
def _palette_ink(value):
    return (
        *(int(value[index : index + 2], 16) for index in (1, 3, 5)),
        int(value[7:9], 16) if len(value) == 9 else 255,
    )


class DesignRuntime:
    def __init__(self, store=None):
        self.store = store or SpecificationStore()
        self.catalog = component_catalog()
        self.spec = deepcopy(DEFAULT)
        self.revision = 0
        self.error = ""
        self._stamp = None
        self._poll_at = 0
        self._inks = {}
        self._module_inks = {}
        self._module_themes = {}
        self._palette_families = frozenset()
        self.has_module_palette = False
        self._theme_revision = -1
        self._theme_key = None
        self._edit_cache = {}
        self._value_cache = {}
        self._style_cache = {}
        self._dimensions_key = None
        self._dimensions = None
        self._layout_signature = None
        self.observed = []
        self.used_fields = {}
        self.track_field_reads = False
        from mojive.design.native.geometry import NativePaintContext

        self.paint_context = NativePaintContext(self)
        self.authored_defaults = True
        self.workbench = None
        self.observe_components = False
        self._reset_inks()
        self.poll(force=True)

    def poll(self, *, force=False):
        if not force and time.monotonic() - self._poll_at < 0.12:
            return False
        self._poll_at = time.monotonic()
        envelope = self.store.latest()
        self.error = "; ".join(error["error"] for error in envelope["errors"])
        if envelope["source"] is None and (envelope["errors"] or self.spec != DEFAULT):
            self.error = self.error or "The saved specification and draft are missing"
            return False
        if self._stamp == envelope["revision"]:
            return False
        self.spec = envelope["spec"]
        self._edit_cache.clear()
        self._value_cache.clear()
        self._style_cache.clear()
        self._stamp = envelope["revision"]
        self.revision += 1
        self._reset_inks()
        return True

    def update_spec(self, value):
        self.spec = value
        self._edit_cache.clear()
        self._value_cache.clear()
        self._style_cache.clear()
        self.revision += 1
        self._reset_inks()

    def _reset_inks(self):
        from mojive.ui.design_palette import rgba8

        self._inks = {
            rgba8(role)[:3]: _palette_ink(self.value(token))
            for role, token in PALETTE_TOKENS.items()
        }
        self._module_inks.clear()
        self._module_themes.clear()
        self._palette_families = frozenset(
            family
            for family, values in self.spec.get("components", {}).items()
            if any(token in values for token in PALETTE_TOKENS.values())
        )
        self.has_module_palette = bool(self._palette_families)

    def ink(self, color, family=None):
        palette = self._inks
        if self.has_module_palette and family in self._palette_families:
            cached = self._module_inks.get(family)
            if cached is None:
                from mojive.ui.design_palette import rgba8

                palette = {}
                aliases = {}
                overrides = self.spec["components"][family]
                for role, token in PALETTE_TOKENS.items():
                    source = rgba8(role)[:3]
                    value = _palette_ink(self.value(token, family))
                    palette[source] = value
                    if token in overrides:
                        # Neutral Draw2D consumers receive the applied global
                        # Theme. Recognize that exact ink before class overrides.
                        aliases[self._inks[source]] = value
                cached = self._module_inks[family] = palette, aliases
            palette, aliases = cached
            replacement = aliases.get(tuple(color))
            if replacement is not None:
                return replacement
        replacement = palette.get(tuple(color[:3]))
        if replacement is None:
            return color
        return (*replacement[:3], round(color[3] * replacement[3] / 255))

    def theme_for(self, theme, family):
        """Reuse a module's existing palette for neutral status and output painters."""
        if not self.has_module_palette or family not in self._palette_families:
            return theme
        key = (id(theme), family)
        cached = self._module_themes.get(key)
        if cached is None:
            themed = replace(
                theme,
                **{
                    field: tuple(
                        v / 255 for v in _palette_ink(self.value(PALETTE_TOKENS[role], family))
                    )
                    for field, role in THEME_ROLES.items()
                },
            )
            cached = self._module_themes[key] = theme, themed
        return cached[1]

    def prepare_fonts(self, ui):
        """Prepare the requested faces in each surface's own native context."""
        ui.prepare_fonts(
            self.spec["tokens"]["font_family"],
            {
                values["fontFamily"]
                for values in self.spec.get("parts", {}).values()
                if values.get("fontFamily") in ("system", "portable")
            }
            | {
                values["font_family"]
                for values in self.spec.get("components", {}).values()
                if values.get("font_family") in ("system", "portable")
            },
        )

    def apply_theme(self, shell):
        shell.app.theme = self.apply_surface(shell.window, shell.s, shell.app.theme)

    def apply_surface(self, window, scale, theme):
        """Apply the same native type, geometry and theme to a workspace or guide surface."""
        from imgui_bundle import imgui

        from mojive.ui.design_palette import rgba8
        from mojive.ui.workspace.design import DEFAULT_DESIGN

        imgui.set_current_context(window._imgui_context)
        self._reset_inks()
        themed = replace(
            theme,
            border=tuple(channel / 255 for channel in _palette_ink(self.value("border_color"))),
            **{
                name: tuple(channel / 255 for channel in self.ink(rgba8(role)))
                for name, role in THEME_ROLES.items()
            },
        )
        window.apply_theme(themed)
        dimensions = self.dimensions(DEFAULT_DESIGN)
        style = imgui.get_style()
        style.font_size_base = dimensions.body_size * scale
        style.frame_rounding = dimensions.control_radius * scale
        style.window_rounding = dimensions.container_radius * scale
        style.popup_rounding = self.value("radius_popup") * scale
        style.child_rounding = dimensions.child_radius * scale
        style.window_padding = (
            dimensions.panel_padding_x * scale,
            dimensions.panel_padding_y * scale,
        )
        style.frame_padding = (
            dimensions.frame_padding_x * scale,
            dimensions.frame_padding_y * scale,
        )
        style.item_spacing = (dimensions.item_spacing_x * scale, dimensions.row_gap * scale)
        return themed

    def value(self, token, family=None):
        return (
            self.spec.get("components", {}).get(family, {}).get(token, self.spec["tokens"][token])
        )

    def resolved_part(self, part, field, family=None):
        key = (part, field, family)
        if key not in self._value_cache:
            self._value_cache[key] = resolve_part_value(
                self.spec, part, field, family=family, catalog=self.catalog
            )
        return self._value_cache[key]

    def edited(self, part, field, seen=(), family=None):
        key = (part, field, family)
        if key not in self._edit_cache:
            self._edit_cache[key] = self._edited(part, field, seen, family)
        return self._edit_cache[key]

    def _edited(self, part, field, seen=(), family=None):
        if (part, field) in seen:
            raise ValueError(f"Circular design default: {part}.{field}")
        values = self.spec.get("parts", {}).get(part, {})
        baseline = BASELINE.get("parts", {}).get(part, {})
        if values.get(field) != baseline.get(field):
            return True
        default = self.catalog.get(part, {}).get("fields", {}).get(field, {}).get("default")
        if isinstance(default, dict):
            if "field" in default:
                return self.edited(
                    default.get("part", part),
                    default["field"],
                    (*seen, (part, field)),
                    family=family,
                )
            if "token" in default:
                return self.value(default["token"], family) != BASELINE["tokens"][default["token"]]
            if "layout" in default:
                return (
                    self.spec["layout"][default["layout"]] != BASELINE["layout"][default["layout"]]
                )
        return False

    def metric(self, part, field, fallback, *, family=None):
        self.consume(part, field)
        if not self.edited(part, field, family=family):
            return fallback
        rule = self.catalog[part]["fields"][field]
        condition = rule.get("condition")
        if condition and self.resolved_part(part, condition["field"], family) != condition["value"]:
            return fallback
        return self.resolved_part(part, field, family)

    def dimensions(self, baseline):
        key = (baseline, self.revision)
        if key == self._dimensions_key:
            return self._dimensions
        values = {name: self.value(token) for name, token in TOKEN_METRICS.items()}
        values.update(
            {
                name: self.metric(part, field, getattr(baseline, name))
                for name, (part, field) in PART_METRICS.items()
            }
        )
        values.update(
            {
                name: self.spec["layout"][name]
                for name in ("scene_width", "inspector_width", "bottom_height")
            }
        )
        for name, part, field in (
            ("scene_width", "workspace", "sceneWidth"),
            ("inspector_width", "workspace", "inspectorWidth"),
            ("bottom_height", "workspace", "bottomHeight"),
            ("application_height", "application", "titleHeight"),
        ):
            values[name] = self.metric(part, field, values.get(name, getattr(baseline, name)))
        values["body_size"] = self.metric("application", "fontSize", values["body_size"])
        if self.spec["layout"]["application_height"] != BASELINE["layout"]["application_height"]:
            values["application_height"] = self.spec["layout"]["application_height"]
        self._dimensions = replace(baseline, **values)
        self._dimensions_key = key
        return self._dimensions

    def observes(self, parts):
        """Collect the selected component, or the full frame for picking and audits."""
        if (
            getattr(self, "observe_components", False)
            or self.track_field_reads
            or not self.authored_defaults
        ):
            return True
        bench = self.workbench
        if bench is None:
            return False
        selected = getattr(bench, "part", None)
        if selected is None or bench.pick or selected in parts:
            return True
        from mojive.design.native.geometry import CURRENT_NATIVE_PARTS

        return selected in CURRENT_NATIVE_PARTS.get()

    def begin_frame(self):
        self.observed.clear()
        self.used_fields.clear()

    def consume(self, part, field):
        """Record actual production reads only while a workbench is observing."""
        if self.track_field_reads:
            fields = self.used_fields.get(part)
            if fields is None:
                fields = self.used_fields[part] = set()
            fields.add(field)

    def consume_groups(self, groups):
        for part, fields in groups:
            used = self.used_fields.get(part)
            if used is None:
                self.used_fields[part] = set(fields)
            else:
                used.update(fields)

    @contextmanager
    def scope(self, shell):
        from mojive.ui.workspace.design import DEFAULT_DESIGN, use_design

        self.poll()
        self.begin_frame()
        dimensions = self.dimensions(DEFAULT_DESIGN)
        signature = tuple(
            getattr(dimensions, name)
            for name in (
                "scene_width",
                "inspector_width",
                "bottom_height",
                "application_height",
                "sidebar_width",
            )
        )
        if signature != self._layout_signature:
            if self._layout_signature is not None:
                shell.needs_layout = True
            self._layout_signature = signature
        from mojive.design.native.geometry import part_scope

        with use_design(self.dimensions), part_scope("application"):
            if self._theme_revision != self.revision:
                self.prepare_fonts(shell.ui)
                theme_key = (
                    tuple(self.spec["tokens"].items()),
                    shell.s,
                    tuple(
                        getattr(dimensions, field)
                        for field in (
                            "body_size",
                            "control_radius",
                            "container_radius",
                            "panel_padding_x",
                            "panel_padding_y",
                            "item_spacing_x",
                            "row_gap",
                        )
                    ),
                )
                if theme_key != self._theme_key:
                    self.apply_theme(shell)
                    self._theme_key = theme_key
                self._theme_revision = self.revision
            from imgui_bundle import imgui

            split = self.metric("workspace", "splitterWidth", 2) + self.metric(
                "workspace", "columnGap", 0
            )
            imgui.get_style().docking_separator_size = split * shell.s
            yield
