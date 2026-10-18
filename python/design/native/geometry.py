"""Catalogue-driven boxes, inheritance and equal-spacing layout."""

import contextvars
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from mojive.design.groups import scoped_parts
from mojive.design.native.style_cache import prepared_geometry
from mojive.design.specification import BASELINE, component_catalog, resolve_part_value

BUTTON_PARTS = frozenset(
    (
        "button",
        "icon_button",
        "tab",
        "filter",
        "menu_trigger",
        "menu_item",
        "link_button",
        "tree_label",
        "tree_expander",
        "material",
        "sensor",
        "timeline_key",
    )
)
CURRENT_NATIVE_PARTS = contextvars.ContextVar("design_native_parts", default=())
_CORNER_RADII = contextvars.ContextVar("design_native_corner_radii", default=())
INHERITED_TEXT_FIELDS = (
    "fontSize",
    "fontFamily",
    "textColor",
    "iconColor",
    "iconSize",
    "lineHeightMode",
    "lineHeight",
)
PRECISE_TEXT_PARENTS = frozenset(("joint", "timeline_details", "section_body", "select_popup"))


@contextmanager
def corner_radius_scope(radius_native, parts):
    """Constrain matching child corners to a parent's resolved native geometry."""
    parts = (parts,) if isinstance(parts, str) else tuple(parts)
    token = _CORNER_RADII.set((*_CORNER_RADII.get(), (parts, max(0, radius_native))))
    try:
        yield
    finally:
        _CORNER_RADII.reset(token)


def scoped_corner_radius(parts):
    """Return the innermost matching native radius, including an explicit zero."""
    for candidates, radius in reversed(_CORNER_RADII.get()):
        if any(part in parts for part in candidates):
            return radius
    return None


@contextmanager
def independent_corner_scope():
    """Measure a separate popup root without inheriting its trigger's corners."""
    token = _CORNER_RADII.set(())
    try:
        yield
    finally:
        _CORNER_RADII.reset(token)


# Higher values take precedence when a field delegates to another field.
RESOLVED_VALUE, AUTHORED_FALLBACK, INACTIVE_FIELD = range(3)


def budget_margins(margins, width, height):
    """Keep opposite margins proportional when their parent budget is exhausted."""
    left, top, right, bottom = margins
    fx = min(1, max(0, width) / (left + right)) if left + right > 0 else 1
    fy = min(1, max(0, height) / (top + bottom)) if top + bottom > 0 else 1
    return left * fx, top * fy, right * fx, bottom * fy


class NativePaintContext:
    """Keep nested reference painting separate from production component defaults."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.paint_depth = 0

    def inherited(self, field, fallback):
        for part in reversed(CURRENT_NATIVE_PARTS.get()):
            if self.runtime.edited(part, field):
                return self.runtime.metric(part, field, fallback)
        return fallback


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def tuple(self):
        return self.x, self.y, self.width, self.height

    def inset(self, left=0.0, top=0.0, right=0.0, bottom=0.0):
        left = min(left, max(0.0, self.width))
        top = min(top, max(0.0, self.height))
        return Rect(
            self.x + left,
            self.y + top,
            max(0.0, self.width - left - right),
            max(0.0, self.height - top - bottom),
        )


@dataclass(frozen=True, slots=True)
class Box:
    """Margin, visible frame and content in the same physical coordinate system."""

    outer: Rect
    frame: Rect
    content: Rect
    padding: tuple[float, float, float, float]
    margin: tuple[float, float, float, float]


class NativePartStyle:
    """Resolve nested control specializations from the shared component catalogue."""

    def __init__(self, runtime, part_ids, family=None, *, authored_defaults=False):
        self._compiled = None
        self.runtime = runtime
        self.spec = runtime.spec if hasattr(runtime, "spec") else runtime
        self.catalog = getattr(runtime, "catalog", None) or component_catalog()
        self.part_ids = scoped_parts(part_ids)
        self.family = family
        self.authored_defaults = authored_defaults
        self.values: dict[str, Any] = {}
        self.origins: dict[str, str] = {}
        self.edge_sources: dict[str, str] = {}
        self._field_rules = {}
        cache = getattr(runtime, "_style_cache", None)
        identity = (self.part_ids, family)
        template = cache.get(identity) if cache is not None else None
        if template is None:
            self._explicit_fields = frozenset(
                key for part in self.part_ids for key in self.spec.get("parts", {}).get(part, {})
            )
            # Explicit base edits outrank a specialization's inherited default.
            for part_id in self.part_ids:
                for key in self.catalog.get(part_id, {}).get("fields", {}):
                    self.values[key] = resolve_part_value(
                        self.spec, part_id, key, family=family, catalog=self.catalog
                    )
                    self.origins[key] = part_id
            for part_id in self.part_ids:
                for key, value in self.spec.get("parts", {}).get(part_id, {}).items():
                    self.values[key], self.origins[key] = value, part_id
            self._inherit_edges()
            if cache is not None:
                cache[identity] = (
                    self.values.copy(),
                    self.origins.copy(),
                    self.edge_sources.copy(),
                    self._explicit_fields,
                    self._field_rules,
                )
        else:
            self.values, self.origins = template[0].copy(), template[1].copy()
            self.edge_sources = template[2].copy()
            self._explicit_fields, self._field_rules = template[3], template[4]

    def freeze(self):
        """Compile immutable field policies after parent and control overrides."""
        self._dependency_parts = self._dependencies()
        self._dependency_values = {
            part: dict(self.spec.get("parts", {}).get(part, {}))
            for part in (*self._dependency_parts, *CURRENT_NATIVE_PARTS.get())
        }
        self._token_values = dict(self.spec["tokens"])
        self._layout_values = dict(self.spec["layout"])
        self._family_values = dict(self.spec.get("components", {}).get(self.family, {}))
        compiled = {}
        for key in self.values:
            self._compile_field(key, compiled)
        self.record_values = dict(self.values)
        self.record_origins = dict(self.origins)
        self._variants = {}
        self._geometry_cache = {}
        self._collecting = None
        self._compiled = compiled
        self.values = MappingProxyType(self.values)
        self.origins = MappingProxyType(self.origins)
        self.edge_sources = MappingProxyType(self.edge_sources)
        return self

    def _compile_field(self, key, compiled):
        if key in compiled:
            return compiled[key]
        if key not in self.values:
            result = (None, INACTIVE_FIELD, ())
        elif (
            key in ("rowGap", "columnGap")
            and self.has("gap")
            and self.explicit("gap")
            and not self.explicit(key)
        ):
            result = self._compile_field("gap", compiled)
        else:
            origin = self.origins[key]
            condition, use_fallback, alias = self._field_rule(origin, key)
            reads = ((origin, key),) if self.authored_defaults and origin != "parent_corner" else ()
            fallback_mode = (
                AUTHORED_FALLBACK if self.authored_defaults and use_fallback else RESOLVED_VALUE
            )
            if condition and self.values.get(condition["field"]) != condition["value"]:
                result = (None, INACTIVE_FIELD, reads)
            elif alias:
                value, mode, nested = self._compile_field(alias, compiled)
                result = (value, max(mode, fallback_mode), reads + nested)
            else:
                result = (self.values[key], fallback_mode, reads)
        compiled[key] = result
        return result

    def compatible(self, spec):
        """Unrelated component edits keep their prepared geometry warm."""
        if self._token_values != spec["tokens"] or self._layout_values != spec["layout"]:
            return False
        if self._family_values != spec.get("components", {}).get(self.family, {}):
            return False
        return all(
            values == spec.get("parts", {}).get(part, {})
            for part, values in self._dependency_values.items()
        )

    def _dependencies(self):
        dependencies = set(self.part_ids)
        pending = list(dependencies)
        while pending:
            part = pending.pop()
            for rule in self.catalog.get(part, {}).get("fields", {}).values():
                default = rule.get("default")
                source = default.get("part", part) if isinstance(default, dict) else part
                if source not in dependencies:
                    dependencies.add(source)
                    pending.append(source)
        return tuple(dependencies)

    def with_overrides(self, overrides):
        """Keep control-local typography and slot sizes out of shared templates."""
        if not overrides:
            return self
        key = tuple(overrides.items())
        cached = self._variants.get(key)
        if cached is not None:
            return cached
        style = object.__new__(type(self))
        style.__dict__ = self.__dict__.copy()
        style.values, style.origins = dict(self.values), dict(self.origins)
        style.edge_sources = dict(self.edge_sources)
        for field, (value, origin) in overrides.items():
            style.values[field], style.origins[field] = value, origin
        if len(self._variants) >= 128:
            self._variants.clear()
        self._variants[key] = style.freeze()
        return style

    def _inherit_edges(self):
        for prefix, axis in (("padding", "paddingX"), ("margin", "marginX")):
            for edge in ("Left", "Right"):
                self._inherit_edge(prefix + edge, axis)
        for prefix, axis in (("padding", "paddingY"), ("margin", "marginY")):
            for edge in ("Top", "Bottom"):
                self._inherit_edge(prefix + edge, axis)

    def _inherit_edge(self, key, axis):
        if not self.explicit(key) and self.explicit(axis):
            self.values[key], self.origins[key] = self.values[axis], self.origins[axis]
            self.edge_sources[key] = axis

    def explicit(self, key):
        return key in self._explicit_fields

    def has(self, key):
        return key in self.values

    @property
    def edited(self):
        return any(self.spec.get("parts", {}).get(part) for part in self.part_ids) or any(
            self.origins.get(field) not in self.part_ids
            and field in self.spec.get("parts", {}).get(self.origins.get(field), {})
            for field in INHERITED_TEXT_FIELDS
        )

    def get(self, key, fallback=None):
        if self._compiled is not None:
            item = self._compiled.get(key)
            if item is None:
                return fallback
            value, mode, reads = item
            if self._collecting is not None:
                self._collecting.update(reads)
            elif getattr(self.runtime, "track_field_reads", False):
                for origin, field in reads:
                    self.runtime.consume(origin, field)
            if mode == INACTIVE_FIELD or (mode == AUTHORED_FALLBACK and fallback is not None):
                return fallback
            return value
        if key not in self.values:
            return fallback
        if (
            key in ("rowGap", "columnGap")
            and self.has("gap")
            and self.explicit("gap")
            and not self.explicit(key)
        ):
            return self.get("gap", fallback)
        origin = self.origins[key]
        if self.authored_defaults and hasattr(self.runtime, "consume"):
            self.runtime.consume(origin, key)
        identity = (origin, key)
        policy = self._field_rules.get(identity)
        if policy is None:
            policy = self._field_rules[identity] = self._field_rule(origin, key)
        condition, use_fallback, alias = policy
        if condition and self.values.get(condition["field"]) != condition["value"]:
            return fallback
        if self.authored_defaults and use_fallback and fallback is not None:
            return fallback
        if alias:
            return self.get(alias, fallback)
        return self.values[key]

    def _field_rule(self, origin, key):
        """Compile revision-stable inheritance; values and conditions stay live.

        Rules follow the resolved origin because parent typography may replace
        it on an individual control. Runtime spec updates clear the style cache.
        """
        if origin == "parent_corner":
            return None, False, None
        rule = self.catalog.get(origin, {}).get("fields", {}).get(key, {})
        condition = rule.get("condition")
        default = rule.get("default")
        reference = default if isinstance(default, dict) else {}
        source = reference.get("field")
        inherited = source is not None and self.explicit(source)
        mode_activated = condition and self.explicit(condition["field"])
        token_default, token_part = reference, origin
        # Panel-owned defaults delegate to a shared field. Keep the shared
        # field's token activation, so global and family edits still apply.
        while "field" in token_default:
            token_part = token_default.get("part", token_part)
            source_rule = (
                self.catalog.get(token_part, {}).get("fields", {}).get(token_default["field"], {})
            )
            source_default = source_rule.get("default")
            token_default = source_default if isinstance(source_default, dict) else {}
        token = token_default.get("token")
        token_changed = (
            token is not None
            and self.spec.get("components", {})
            .get(self.family, {})
            .get(token, self.spec["tokens"][token])
            != BASELINE["tokens"][token]
        )
        parent_explicit = origin not in self.part_ids and key in self.spec.get("parts", {}).get(
            origin, {}
        )
        use_fallback = not (
            self.explicit(key)
            or key in self.edge_sources
            or inherited
            or mode_activated
            or token_changed
            or parent_explicit
        )
        alias = (
            source
            if not self.explicit(key)
            and source != key
            and reference.get("part", origin) in self.part_ids
            and inherited
            else None
        )
        return condition, use_fallback, alias

    def dimension(self, dimension, fallback):
        mode_key = dimension + "Mode"
        if self.has(mode_key) and self.get(mode_key) != "fixed":
            return fallback
        return self.get(dimension, fallback)

    @prepared_geometry
    def edges(self, prefix, scale=1.0, fallback=(0.0, 0.0, 0.0, 0.0)):
        keys = (
            ("paddingLeft", "paddingTop", "paddingRight", "paddingBottom")
            if prefix == "padding"
            else ("marginLeft", "marginTop", "marginRight", "marginBottom")
        )
        left, top, right, bottom = (self.get(key, fallback[n]) for n, key in enumerate(keys))
        if prefix == "padding" and not all(key in self.values for key in keys):
            x, y = self.get("paddingX"), self.get("paddingY")
            left = left if keys[0] in self.values or x is None else x
            right = right if keys[2] in self.values or x is None else x
            top = top if keys[1] in self.values or y is None else y
            bottom = bottom if keys[3] in self.values or y is None else y
        return left * scale, top * scale, right * scale, bottom * scale

    def line_height(self, fallback=18.0):
        if self.get("lineHeightMode", "auto") == "fixed":
            return self.get("lineHeight", fallback)
        return self.get("fontSize", fallback / 1.4) * 1.4

    @prepared_geometry
    def box(self, x, y, width, height, *, scale=1.0, padding=(0.0, 0.0, 0.0, 0.0), intrinsic=None):
        """Coordinates and supplied dimensions are physical; catalogue values are points."""
        width = self.dimension("width", width / scale) * scale
        height = self.dimension("height", height / scale) * scale
        pad = self.edges("padding", scale, tuple(v / scale for v in padding))
        margin = self.edges("margin", scale)
        if any(part in ("button", "icon_button") for part in self.part_ids) and any(
            self.explicit(field)
            for field in (
                "paddingX",
                "paddingY",
                "paddingLeft",
                "paddingRight",
                "paddingTop",
                "paddingBottom",
            )
        ):
            # Small authored reset slots must allocate edited padding in auto
            # mode. The same box call measures both parent budgets and paint.
            if self.get("widthMode", "auto") == "auto":
                width = max(width, pad[0] + pad[2])
            if self.get("heightMode", "auto") == "auto":
                height = max(height, pad[1] + pad[3])
        if intrinsic is not None:
            iw, ih = intrinsic
            # Fixed heights include padding; auto controls grow to fit content.
            if not self.has("widthMode") or self.get("widthMode") == "auto":
                width = max(width, iw + pad[0] + pad[2])
            if not self.explicit("height") or any(part in BUTTON_PARTS for part in self.part_ids):
                height = max(height, ih + pad[1] + pad[3])
        min_height = self.get("minHeight", 0.0) * scale
        height = max(height, min_height)
        frame = Rect(x + margin[0], y + margin[1], max(0.0, width), max(0.0, height))
        outer = Rect(
            x, y, frame.width + margin[0] + margin[2], frame.height + margin[1] + margin[3]
        )
        return Box(outer, frame, frame.inset(*pad), pad, margin)

    def align_rect(self, content, width, height, *, horizontal=None, vertical=None):
        horizontal = horizontal or self.get("align", self.get("textAlign", "start"))
        vertical = vertical or self.get("alignItems", "center")
        fx = (
            0.5
            if horizontal == "center"
            else 1.0
            if horizontal in ("end", "right", "flex-end")
            else 0.0
        )
        fy = (
            0.5
            if vertical == "center"
            else 1.0
            if vertical in ("end", "bottom", "flex-end")
            else 0.0
        )
        return Rect(
            content.x + max(0.0, content.width - width) * fx,
            content.y + max(0.0, content.height - height) * fy,
            width,
            height,
        )


@dataclass(frozen=True, slots=True)
class LayoutChild:
    width: float
    height: float
    margin: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    grow: float = 0.0


def equal_spacing_layout(
    rect,
    children: Iterable[LayoutChild],
    gap=0.0,
    *,
    direction="row",
    align="start",
    align_items="center",
    wrap=False,
):
    """Place each margin box with one adjustable gap, retaining all child extents."""
    children = tuple(children)
    if not children:
        return ()
    horizontal = direction == "row"
    major_available = rect.width if horizontal else rect.height
    minor_available = rect.height if horizontal else rect.width
    major = [
        c.width + c.margin[0] + c.margin[2] if horizontal else c.height + c.margin[1] + c.margin[3]
        for c in children
    ]
    minor = [
        c.height + c.margin[1] + c.margin[3] if horizontal else c.width + c.margin[0] + c.margin[2]
        for c in children
    ]
    if wrap and horizontal:
        # Rows wrap as complete margin boxes. The same gap separates rows.
        result, row, consumed, top = [], [], 0.0, rect.y

        def finish(items, row_height):
            return equal_spacing_layout(
                Rect(rect.x, top, rect.width, row_height),
                items,
                gap,
                direction="row",
                align=align,
                align_items=align_items,
            )

        for child, extent in zip(children, major, strict=True):
            if row and consumed + gap + extent > rect.width:
                height = max(c.height + c.margin[1] + c.margin[3] for c in row)
                result.extend(finish(row, height))
                top += height + gap
                row, consumed = [], 0.0
            consumed += extent + (gap if row else 0.0)
            row.append(child)
        if row:
            result.extend(finish(row, max(c.height + c.margin[1] + c.margin[3] for c in row)))
        return tuple(result)
    natural = sum(major) + gap * (len(children) - 1)
    spare = max(0.0, major_available - natural)
    grow = sum(c.grow for c in children)
    if grow:
        major = [extent + spare * c.grow / grow for c, extent in zip(children, major, strict=True)]
        spare = 0.0
    actual_gap = (
        gap + spare / (len(children) - 1) if align == "space-between" and len(children) > 1 else gap
    )
    cursor = spare * (
        0.5 if align == "center" else 1.0 if align in ("end", "right", "bottom") else 0.0
    )
    out = []
    for child, extent, cross in zip(children, major, minor, strict=True):
        offset = max(0.0, minor_available - cross) * (
            0.5
            if align_items == "center"
            else 1.0
            if align_items in ("end", "bottom", "right")
            else 0.0
        )
        ml, mt, mr, mb = child.margin
        width, height = (
            (max(0.0, extent - ml - mr), child.height)
            if horizontal
            else (child.width, max(0.0, extent - mt - mb))
        )
        if align_items == "stretch":
            if horizontal:
                height = max(0.0, minor_available - mt - mb)
            else:
                width = max(0.0, minor_available - ml - mr)
        out.append(
            Rect(
                rect.x + (cursor if horizontal else offset) + ml,
                rect.y + (offset if horizontal else cursor) + mt,
                width,
                height,
            )
        )
        cursor += extent + actual_gap
    return tuple(out)


@dataclass(slots=True)
class PrimitiveResult:
    value: Any
    changed: bool
    box: Box
    activated: bool = False
    active: bool = False
    deactivated: bool = False
    committed: bool = False
    canceled: bool = False
    reset: bool = False


@dataclass(frozen=True, slots=True)
class SliderGeometry:
    style: NativePartStyle
    box: Box
    placements: tuple[Rect, Rect]
    source_rects: tuple[Rect, Rect]
    gap: float


@contextmanager
def part_scope(*part_ids, inherit=True):
    """Assign a real component identity at an exact native call boundary."""
    parents = CURRENT_NATIVE_PARTS.get() if inherit else ()
    token = CURRENT_NATIVE_PARTS.set(tuple(dict.fromkeys((*parents, *scoped_parts(part_ids)))))
    try:
        yield
    finally:
        CURRENT_NATIVE_PARTS.reset(token)


def classify_button(key, label, override=()):
    override = (override,) if isinstance(override, str) else tuple(override)
    return _classify_button(str(key), bool(label), override)


@lru_cache(maxsize=4096)
def _classify_button(key, label, override):
    if key.startswith(("receipt-", "link-")):
        return ("link_button",)
    if key == "material-color":
        # This input opens the existing color editor; it is not a material card.
        return ("color_input",)
    selected = tuple(part for part in override if part in BUTTON_PARTS)
    if selected:
        return selected
    if key.startswith("tab-") or key in ("timeline-tab", "output-tab"):
        return ("tab",)
    if key.startswith("expand-"):
        return ("tree_expander",)
    if key.startswith("entity-"):
        return ("tree_label",)
    if key.startswith("material-"):
        return ("material",)
    if key.startswith("top-menu-"):
        return ("menu_trigger",)
    return ("button",) if label else ("icon_button",)
