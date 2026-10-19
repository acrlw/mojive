"""Panel slots and viewport actions used by the native shell."""

from mojive.ui.input_bindings import InputAction

from .design import DESIGN

_PANEL_SLOTS = (
    ("hierarchy", "Scene"),
    ("joints", "Pose"),
    ("control", "Control"),
    ("assets", "Assets"),
    ("camera", "Cameras"),
    ("sensors", "Sensors"),
    ("layers", "Layers"),
    ("output", "Output"),
    ("stats", "Statistics"),
    ("settings", "Settings"),
)
PANEL_ICONS = {
    "hierarchy": "scene",
    "joints": "pose",
    "control": "control",
    "assets": "assets",
    "camera": "camera",
    "sensors": "sensor",
    "layers": "layers",
    "output": "terminal",
    "stats": "performance",
    "settings": "settings",
    "inspector": "inspect",
    "keyframes": "key",
    "plot": "waveform",
    "help": "help",
    "info": "info",
}
PANELS = tuple(name for _panel_id, name in _PANEL_SLOTS)
ICONS = tuple(PANEL_ICONS[panel_id] for panel_id, _name in _PANEL_SLOTS)
LEFT_PANELS = PANELS[:7]
BOTTOM_PANELS = ("Timeline", "Output", "Statistics", "Plot")
BOTTOM_HEIGHT_PT = DESIGN.bottom_height
VIEWPORT_TOOLS = (
    ("select", None),
    ("move", InputAction.GIZMO_TRANSLATE),
    ("rotate", InputAction.GIZMO_ROTATE),
    ("scale", InputAction.GIZMO_DIMENSIONS),
)
