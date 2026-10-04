"""Shared vector icons for runtime UI and design tools.

Shapes, fitting and submission have distinct owners; this module exposes their public API.
"""

from .drawing import IconStrokePath as IconStrokePath
from .drawing import draw_concept_icon as draw_concept_icon
from .drawing import draw_control_icon as draw_control_icon
from .drawing import draw_icon as draw_icon
from .drawing import draw_icon_label as draw_icon_label
from .drawing import production_helper_strokes as production_helper_strokes
from .layout import IconMetrics as IconMetrics
from .layout import icon_alignment_anchor as icon_alignment_anchor
from .layout import icon_alignment_center as icon_alignment_center
from .layout import icon_metrics as icon_metrics
from .layout import minimum_enclosing_circle as minimum_enclosing_circle
from .layout import production_icon_metrics as production_icon_metrics
from .model import BOX_CENTERED_ICONS as BOX_CENTERED_ICONS
from .model import ICON_ALIGNMENT_CHOICES as ICON_ALIGNMENT_CHOICES
from .model import ICON_ALIGNMENT_EDITABLE_ICONS as ICON_ALIGNMENT_EDITABLE_ICONS
from .model import ICON_BOUND_DIAMETER as ICON_BOUND_DIAMETER
from .model import ICON_DEFAULT_PADDING as ICON_DEFAULT_PADDING
from .model import ICON_FAMILIES as ICON_FAMILIES
from .model import ICON_GLYPH_ALIGNMENT_DEFAULTS as ICON_GLYPH_ALIGNMENT_DEFAULTS
from .model import ICON_GLYPH_OFFSET_DEFAULTS as ICON_GLYPH_OFFSET_DEFAULTS
from .model import ICON_GLYPH_PADDING_DEFAULTS as ICON_GLYPH_PADDING_DEFAULTS
from .model import ICON_GLYPH_STROKE_DEFAULTS as ICON_GLYPH_STROKE_DEFAULTS
from .model import ICON_GRID as ICON_GRID
from .model import ICON_GROUP_BY_SLUG as ICON_GROUP_BY_SLUG
from .model import ICON_GROUP_LAYOUT_DEFAULTS as ICON_GROUP_LAYOUT_DEFAULTS
from .model import ICON_GROUP_STROKE_DEFAULTS as ICON_GROUP_STROKE_DEFAULTS
from .model import ICON_LAYOUT_REFERENCES as ICON_LAYOUT_REFERENCES
from .model import ICON_LIBRARY_TABS as ICON_LIBRARY_TABS
from .model import ICON_MAX_PADDING as ICON_MAX_PADDING
from .model import ICON_MAX_STROKE as ICON_MAX_STROKE
from .model import ICON_MIN_CLEARANCE as ICON_MIN_CLEARANCE
from .model import ICON_MIN_STROKE as ICON_MIN_STROKE
from .model import ICON_ROTATE_RING_CAP as ICON_ROTATE_RING_CAP
from .model import ICON_ROTATE_RING_GAP_RATIO as ICON_ROTATE_RING_GAP_RATIO
from .model import ICON_STROKE as ICON_STROKE
from .model import ICON_TUNING_DEFAULTS as ICON_TUNING_DEFAULTS
from .model import MORE_ARM_RATIO as MORE_ARM_RATIO
from .model import RESET_RING_CENTER as RESET_RING_CENTER
from .model import REVIEW_LOCKED_ICONS as REVIEW_LOCKED_ICONS
from .model import REVIEW_LOCKED_PADDING as REVIEW_LOCKED_PADDING
from .model import RING_CENTERED_ICONS as RING_CENTERED_ICONS
from .model import ROTATE_FRAME_PADDING as ROTATE_FRAME_PADDING
from .model import ROTATE_FRAME_STROKE_OVERSHOOT as ROTATE_FRAME_STROKE_OVERSHOOT
from .model import ROTATE_FRINGE_GAP_FRACTION as ROTATE_FRINGE_GAP_FRACTION
from .model import ROTATE_FRINGE_MAX as ROTATE_FRINGE_MAX
from .model import STATUS_MOUSE_DEFAULT_WIDTH as STATUS_MOUSE_DEFAULT_WIDTH
from .model import STROKE_SCALE_LOCKED_ICONS as STROKE_SCALE_LOCKED_ICONS
from .model import IconStyle as IconStyle
from .model import IconTuning as IconTuning
from .model import icon_component_group as icon_component_group
from .model import icon_family as icon_family
from .model import production_icon_offset as production_icon_offset
from .model import production_icon_style as production_icon_style
