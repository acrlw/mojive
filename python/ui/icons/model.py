"""Canonical icon families, style values and reviewed production defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

from ..overlay_geometry import (
    DEFAULT_RESET_HEAD_SCALE,
    OVERLAY_GEOMETRY,
)
from ..severity_icons import SEVERITY_FRAME_PADDING

ICON_GRID = 24.0
ICON_STROKE = 1.75
ICON_MIN_STROKE = 1.0
ICON_MAX_STROKE = 2.5
# Keep ordinary non-tool families two grid units inside their circular slot.
# Viewport tools override this below so their glyphs remain close to the frame.
ICON_BOUND_DIAMETER = ICON_GRID
ICON_MIN_CLEARANCE = 0.5
ICON_DEFAULT_PADDING = 2.0
ICON_MAX_PADDING = 4.0
ROTATE_FRAME_PADDING = 0.65
ROTATE_FRAME_STROKE_OVERSHOOT = ICON_STROKE * 0.5
ROTATE_FRINGE_MAX = 1.0
ROTATE_FRINGE_GAP_FRACTION = 0.5
ICON_ROTATE_RING_GAP_RATIO = 1.0
ICON_ROTATE_RING_CAP = "round"
MORE_ARM_RATIO = 0.94
STATUS_MOUSE_DEFAULT_WIDTH = OVERLAY_GEOMETRY.hint_mouse_width
REVIEW_LOCKED_ICONS = frozenset(("status-info", "status-warning", "status-error"))
REVIEW_LOCKED_PADDING = SEVERITY_FRAME_PADDING
STROKE_SCALE_LOCKED_ICONS = REVIEW_LOCKED_ICONS | frozenset(
    ("status-mouse-left", "status-mouse-right", "status-mouse-wheel")
)
BOX_CENTERED_ICONS = frozenset(
    (
        "tool-snap",
        "key-follow-off",
        "key-follow-page",
        "key-follow-locked",
        "playback-previous",
        "playback-next",
        "playback-more",
        "transport-first",
        "transport-previous",
        "transport-next",
        "transport-last",
        "transport-more",
    )
)
RING_CENTERED_ICONS = frozenset(("playback-reset",))
ICON_ALIGNMENT_CHOICES = ("circle", "box")
ICON_ALIGNMENT_EDITABLE_ICONS = frozenset(("key-snapshot", "helper-camera", "helper-light"))
ICON_GLYPH_ALIGNMENT_DEFAULTS = dict.fromkeys(ICON_ALIGNMENT_EDITABLE_ICONS, "box")
RESET_RING_CENTER = (0.0, 0.47)
ICON_FAMILIES = (
    (
        "Viewport tools",
        (
            ("Move", "tool-move"),
            ("Rotate", "tool-rotate"),
            ("Scale", "tool-scale"),
            ("World", "tool-world"),
            ("Body", "tool-body"),
            ("Snap", "tool-snap"),
        ),
    ),
    (
        "Viewport playback",
        (
            ("Previous", "playback-previous"),
            ("Play", "playback-play"),
            ("Pause", "playback-pause"),
            ("Next", "playback-next"),
            ("Reset", "playback-reset"),
            ("Record", "playback-record"),
            ("Stop", "playback-stop"),
            ("More", "playback-more"),
        ),
    ),
    (
        "Keyframe transport",
        (
            ("First", "transport-first"),
            ("Previous", "transport-previous"),
            ("Play", "transport-play"),
            ("Pause", "transport-pause"),
            ("Next", "transport-next"),
            ("Last", "transport-last"),
            ("Loop", "transport-loop"),
            ("Record", "transport-record"),
            ("Stop", "transport-stop"),
            ("More", "transport-more"),
        ),
    ),
    (
        "Keyframes",
        (
            ("Snapshot", "key-snapshot"),
            ("Keyframe", "key-keyframe"),
            ("Add", "key-add"),
            ("Clear", "key-clear"),
            ("Previous key", "key-previous"),
            ("Next key", "key-next"),
            ("Fit", "key-fit"),
            ("Follow", "key-follow"),
            ("Follow off", "key-follow-off"),
            ("Follow page", "key-follow-page"),
            ("Follow locked", "key-follow-locked"),
            ("View", "key-view"),
        ),
    ),
    (
        "Panels",
        (
            ("Search", "panel-search"),
            ("Sort", "panel-sort"),
            ("Clear", "panel-clear"),
            ("Visible", "panel-visible"),
            ("Hidden", "panel-hidden"),
            ("Perspective", "panel-perspective"),
            ("Orthographic", "panel-orthographic"),
            ("Disclosure right", "panel-right"),
            ("Disclosure down", "panel-down"),
        ),
    ),
    (
        "Scene helpers",
        (
            ("Camera", "helper-camera"),
            ("Light", "helper-light"),
        ),
    ),
    (
        "Status & input",
        (
            ("Information", "status-info"),
            ("Warning", "status-warning"),
            ("Error", "status-error"),
            ("Mouse left", "status-mouse-left"),
            ("Mouse right", "status-mouse-right"),
            ("Mouse wheel", "status-mouse-wheel"),
        ),
    ),
)

# Every group owns circular padding so component sizing can be tuned independently.
# Placement follows the explicit geometric anchor declared below.
ICON_GROUP_LAYOUT_DEFAULTS = {
    "Viewport tools": 0.5,
    "Viewport playback": ICON_DEFAULT_PADDING,
    "Keyframe transport": ICON_DEFAULT_PADDING,
    "Keyframes": ICON_DEFAULT_PADDING,
    "Panels": ICON_DEFAULT_PADDING,
    "Scene helpers": ICON_DEFAULT_PADDING,
    "Status & input": ICON_DEFAULT_PADDING,
}
ICON_GLYPH_PADDING_DEFAULTS = {
    "tool-move": 1.0,
    "tool-rotate": ROTATE_FRAME_PADDING,
    "tool-scale": 1.0,
    "tool-world": 1.0,
    "tool-body": 1.0,
    "tool-snap": 1.0,
    "playback-previous": 4.0,
    "playback-play": 3.0,
    "playback-next": 4.0,
    "playback-more": 4.0,
    "transport-first": 4.0,
    "transport-previous": 4.0,
    "transport-play": 3.0,
    "transport-next": 4.0,
    "transport-last": 4.0,
    "transport-more": 4.0,
    "key-keyframe": 4.0,
    "key-add": 4.0,
    "key-clear": 4.0,
    "key-previous": 4.0,
    "key-next": 4.0,
    "key-snapshot": 0.5,
    "key-fit": 0.5,
    "key-follow": 0.5,
    "key-follow-off": 0.5,
    "key-follow-page": 0.5,
    "key-follow-locked": 0.5,
    "key-view": 0.5,
    "panel-search": 4.0,
    "panel-sort": 4.0,
    "panel-clear": 4.0,
    "panel-visible": 4.0,
    "panel-hidden": 4.0,
    "panel-perspective": 4.0,
    "panel-orthographic": 4.0,
    "panel-right": 3.0,
    "panel-down": 3.0,
    "helper-camera": 0.5,
    "helper-light": 0.5,
}
ICON_GROUP_STROKE_DEFAULTS = dict.fromkeys(ICON_GROUP_LAYOUT_DEFAULTS, ICON_STROKE)
ICON_GLYPH_STROKE_DEFAULTS = {
    "tool-move": 1.0,
    "tool-rotate": 1.0,
    "tool-scale": 1.0,
    "tool-world": 1.0,
    "tool-body": 1.0,
    "tool-snap": 1.0,
    "playback-previous": 2.0,
    "playback-next": 2.0,
    "playback-reset": 1.5,
    "playback-more": 2.0,
    "transport-previous": 2.0,
    "transport-next": 2.0,
    "transport-loop": 1.5,
    "transport-more": 2.0,
    "key-snapshot": 1.25,
    "key-keyframe": 1.5,
    "key-add": 1.5,
    "key-clear": 1.5,
    "key-previous": 1.5,
    "key-next": 1.5,
    "key-fit": 1.5,
    "key-follow": 1.5,
    "key-follow-off": 1.5,
    "key-follow-page": 1.5,
    "key-follow-locked": 1.5,
    "key-view": 1.5,
    "panel-search": 1.25,
    "panel-sort": 1.25,
    "panel-clear": 1.5,
    "panel-visible": 1.0,
    "panel-hidden": 1.0,
    "panel-perspective": 1.5,
    "panel-orthographic": 1.5,
    "helper-camera": 1.25,
    "helper-light": 1.25,
}

# Reviewed optical translations on the 24-unit grid, after geometric fitting.
# Keep these separate from shape metrics so labels and hit targets stay fixed.
ICON_GLYPH_OFFSET_DEFAULTS = {
    "tool-snap": (0.0, 0.83),
    "playback-previous": (-1.0, 0.0),
    "playback-next": (1.0, 0.0),
    "playback-more": (0.0, 1.0),
    "transport-previous": (-1.0, 0.0),
    "transport-next": (1.0, 0.0),
    "transport-more": (0.0, 1.0),
    "key-snapshot": (0.0, -0.84),
    "panel-search": (1.06, 1.06),
    "panel-sort": (-0.54, 0.41),
    "panel-perspective": (0.27, 0.0),
    "helper-camera": (0.0, -0.84),
    "helper-light": (0.0, -0.65),
}


def production_icon_offset(name: str) -> tuple[float, float]:
    """Return the reviewed optical translation, independent of geometry and size."""
    return ICON_GLYPH_OFFSET_DEFAULTS.get(name, (0.0, 0.0))


@dataclass(frozen=True)
class IconTuning:
    """Authored shape controls that are independent of fit and stroke."""

    move_head_scale: float = 0.85
    scale_handle_scale: float = 1.15
    snap_endpoint_scale: float = 1.3
    key_fit_arm_length: float = 4.5
    reset_head_scale: float = DEFAULT_RESET_HEAD_SCALE

    def __post_init__(self) -> None:
        if not math.isfinite(self.reset_head_scale) or self.reset_head_scale <= 0:
            raise ValueError("reset head scale must be finite and positive")


ICON_TUNING_DEFAULTS = IconTuning()


@dataclass(frozen=True)
class IconStyle:
    """Frozen production inputs for one reviewed icon."""

    padding: float | None
    stroke_width: float
    alignment: str | None
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP
    tuning: IconTuning = ICON_TUNING_DEFAULTS


# Closely related marks share one fitted master so their authored dimensions
# remain comparable after placement. More is a shorter, rotated Previous at the
# same stroke scale. Record shares Stop's fitted scale and carries its requested
# two-percent reduction in the authored circle. All mouse states reuse the Left
# shell scale because their production outer rectangle has one fixed size.
ICON_LAYOUT_REFERENCES = {
    "playback-more": "playback-previous",
    "playback-record": "playback-stop",
    "transport-more": "transport-previous",
    "transport-record": "transport-stop",
    "status-mouse-right": "status-mouse-left",
    "status-mouse-wheel": "status-mouse-left",
}

ICON_LIBRARY_TABS = (
    "Overview",
    "UI context",
    "Capsules",
    "Keyframe follow",
    *(label for label, _icons in ICON_FAMILIES),
)
ICON_GROUP_BY_SLUG = {
    label.casefold().replace(" ", "-").replace("&", "and"): label for label in ICON_LIBRARY_TABS
}


@cache
def production_icon_style(name: str) -> IconStyle:
    """Resolve one immutable runtime style without per-frame dictionary work."""

    group = icon_component_group(name)
    return IconStyle(
        padding=ICON_GLYPH_PADDING_DEFAULTS.get(name, ICON_GROUP_LAYOUT_DEFAULTS[group]),
        stroke_width=ICON_GLYPH_STROKE_DEFAULTS.get(name, ICON_GROUP_STROKE_DEFAULTS[group]),
        alignment=ICON_GLYPH_ALIGNMENT_DEFAULTS.get(name),
    )


def icon_family(label: str):
    """Return one family by its user-facing label."""

    if label == "Keyframe follow":
        return tuple((mode.title(), f"key-follow-{mode}") for mode in ("off", "page", "locked"))
    return next(icons for family, icons in ICON_FAMILIES if family == label)


def icon_component_group(name: str) -> str:
    """Return the actual UI component group that owns one candidate."""

    if name.startswith("tool-"):
        return "Viewport tools"
    if name.startswith("playback-"):
        return "Viewport playback"
    if name.startswith("transport-"):
        return "Keyframe transport"
    if name.startswith("key-"):
        return "Keyframes"
    if name.startswith("panel-"):
        return "Panels"
    if name.startswith("helper-"):
        return "Scene helpers"
    if name.startswith("status-"):
        return "Status & input"
    raise ValueError(f"unknown concept icon: {name!r}")
