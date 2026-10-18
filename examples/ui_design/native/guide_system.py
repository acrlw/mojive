"""Shared workspace specification and isolated guide editing metadata."""

from colorsys import rgb_to_hls
from copy import deepcopy

from mojive.design.runtime import PALETTE_TOKENS
from mojive.design.specification import DEFAULT, component_catalog, validate_spec
from mojive.ui.design_palette import PALETTE as WORKSPACE_PALETTE
from mojive.ui.design_palette import rgba8

SPACING = (4, 8, 12, 16, 24, 32, 48, 64)
TYPE_SCALE = (12, 14, 16, 20, 24)
VIEWER_LAYOUT = {
    "Navigation buttons": {
        "paddingLeft": 12,
        "paddingRight": 12,
        "paddingTop": 6,
        "paddingBottom": 6,
        "rowGap": 4,
    },
    "Header actions": {"paddingLeft": 12, "paddingRight": 12, "paddingTop": 6, "paddingBottom": 6},
    "Parameter buttons": {
        "paddingLeft": 12,
        "paddingRight": 12,
        "paddingTop": 6,
        "paddingBottom": 6,
    },
    "Parameter fields": {"paddingX": 12, "paddingY": 9},
    "Content panes": {"paddingX": 24, "paddingY": 24},
}
GRAY = tuple(
    (name, None)
    for name in (
        "child",
        "background",
        "field",
        "field_hovered",
        "field_active",
        "disabled",
        "secondary",
        "text",
    )
)
SAGE = tuple((name, None) for name in ("primary_dim", "primary", "primary_bright"))
SEMANTIC = tuple((name, None) for name in ("info", "warning", "danger"))
TYPE_ROLES = (
    ("Window heading", 20, "Emphasis", 28, "UI Design System"),
    ("Panel heading", 16, "Emphasis", 24, "Transform / 变换"),
    ("Section heading", 14, "Emphasis", 20, "Position / 位置"),
    ("Body / controls", 14, "Regular", 20, "Camera navigation / 相机导航"),
    ("Supporting text", 12, "Regular", 16, "World coordinates / 世界坐标系"),
    ("Numeric data", 14, "Numeric", 20, "-0.125  59.9  1,024"),
)
REFERENCES = (
    (
        "Apple · Human Interface Guidelines",
        "https://developer.apple.com/design/human-interface-guidelines/",
    ),
    ("Refactoring UI · Wathan & Schoger", "https://refactoringui.com/"),
    ("LovroPodobnik · ui-refactor", "https://github.com/LovroPodobnik/refactoring-ui-skill"),
    ("s0xDk · refactoring-ui", "https://github.com/s0xDk/refactoring-ui-skill"),
    ("Lucide · 24-unit outline masters", "https://github.com/lucide-icons/lucide/tree/1.55.0"),
    ("Mojive · G3 geometry", "docs/how-to/ui-corners.md"),
)


PALETTE = {name: rgba8(name) for name in WORKSPACE_PALETTE}
ROLES = {
    "canvas": "child",
    "panel": "background",
    "raised": "popup",
    "field": "field",
    "text": "text",
    "muted": "secondary",
    "tertiary": "disabled",
    "border": "border",
    "divider": "border",
    "action": "primary",
    "selected": "selected",
}

GROUPS = {
    "Scene tree": "scene_tree_row_tree_label",
    "Scene filters": "scene_type_filters_filter",
    "Settings navigation": "settings_nav_button",
    "Inspector tabs": "inspector_tabs_tab",
    "Timeline actions": "timeline_toolbar_button",
    "Material cards": "assets_material",
    "Viewport tools": "toolbox",
    "Render metric": "status_render",
}
GROUP_CONTAINERS = {
    "Scene tree": "scene_tree_row",
    "Scene filters": "scene_type_filters",
    "Settings navigation": "settings_nav",
    "Inspector tabs": "inspector_tabs",
    "Timeline actions": "timeline_toolbar",
    "Material cards": "assets_material_list",
}


def hex_ink(ink):
    return "#" + "".join(f"{value:02X}" for value in ink[:3])


def contrast(first, second):
    def luminance(ink):
        values = [c / 255 for c in ink[:3]]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in values]
        return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    low, high = sorted((luminance(first), luminance(second)))
    return (high + 0.05) / (low + 0.05)


def grayscale_spec(spec):
    """Convert preview inks only, preserving the exported authored colors."""
    if isinstance(spec, dict):
        return {key: grayscale_spec(value) for key, value in spec.items()}
    if isinstance(spec, str) and spec.startswith("#") and len(spec) in (7, 9):
        rgb = [int(spec[i : i + 2], 16) for i in (1, 3, 5)]
        gray = round(
            sum(c * weight for c, weight in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True))
        )
        return f"#{gray:02X}{gray:02X}{gray:02X}" + spec[7:]
    return deepcopy(spec)


def proposal_spec():
    """Return an isolated copy of the production defaults, without preview overrides."""
    return deepcopy(DEFAULT)


def edit_group(spec, group, field, value):
    """Change one declared owner, never its shared defaults or sibling groups."""
    if field.startswith("container."):
        part, field = GROUP_CONTAINERS[group], field.removeprefix("container.")
    else:
        part = GROUPS[group]
    if field not in component_catalog()[part]["fields"]:
        raise ValueError(f"{part} has no {field} field")
    edited = deepcopy(spec)
    edited["parts"].setdefault(part, {})[field] = value
    return validate_spec(edited)


def hsl_caption(ink):
    hue, lightness, saturation = rgb_to_hls(*(c / 255 for c in ink[:3]))
    return f"HSL {hue * 360:.0f} {saturation * 100:.0f}% {lightness * 100:.0f}%"


def system_manifest(spec):
    from .guide_icons import icon_catalogue

    palette_tokens = {rgba8(role)[:3]: token for role, token in PALETTE_TOKENS.items()}
    colors = {}
    for name, _ in (*GRAY, *SAGE, *SEMANTIC):
        token = palette_tokens.get(PALETTE[name][:3])
        value = spec["tokens"][token] if token is not None else hex_ink(PALETTE[name])
        colors[name] = {"hex": value, "token": token}
    return {
        "version": 1,
        "status": "Workspace specification snapshot; guide edits remain local",
        "spacing_pt": SPACING,
        "type_scale_pt": TYPE_SCALE,
        "geometry": {
            "rounded_rectangle_smoothing": spec["native"]["g3_smoothing"],
            "capsule_smoothing": spec["native"]["g3_capsule_smoothing"],
            "nested_corner_starting_point": "inner = max(0, outer - inset)",
            "scene_filter_shape": "capsule",
            "switch_painter": "mojive.ui.interaction_paint.draw_switch",
        },
        "typography": [
            {
                "role": role,
                "size_pt": size,
                "line_height_pt": line,
                "weight": "Semibold" if weight == "Emphasis" else "Regular",
                "family_role": weight,
            }
            for role, size, weight, line, _ in TYPE_ROLES
        ],
        "colors": colors,
        "workspace_tokens": deepcopy(spec["tokens"]),
        "semantic_roles": ROLES,
        "group_owners": {
            name: {
                "part": part,
                "overrides": spec["parts"].get(part, {}),
                "container": GROUP_CONTAINERS.get(name),
            }
            for name, part in GROUPS.items()
        },
        "sources": dict(REFERENCES),
        "icons": [
            {
                "name": entry.name,
                "label": entry.label,
                "family": entry.family,
                "renderer": "production" if entry.native else "workspace",
            }
            for entry in icon_catalogue()
        ],
    }
