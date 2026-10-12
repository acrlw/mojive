"""Panel slots and viewport actions used by the native shell."""

from mojive.ui.input_bindings import InputAction

from .design import DESIGN

PANELS = (
    "Scene",
    "Pose",
    "Control",
    "Assets",
    "Cameras",
    "Sensors",
    "Layers",
    "Output",
    "Statistics",
    "Settings",
)
ICONS = (
    "scene",
    "pose",
    "control",
    "assets",
    "camera",
    "chart",
    "layers",
    "terminal",
    "chart",
    "settings",
)
LEFT_PANELS = PANELS[:7]
BOTTOM_PANELS = ("Timeline", "Output", "Statistics", "Plot")
BOTTOM_HEIGHT_PT = DESIGN.bottom_height
VIEWPORT_TOOLS = (
    ("select", None),
    ("move", InputAction.GIZMO_TRANSLATE),
    ("rotate", InputAction.GIZMO_ROTATE),
    ("scale", InputAction.GIZMO_DIMENSIONS),
)
