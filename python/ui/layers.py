"""Viewport content groups over the existing retained debug layers."""

from contextlib import contextmanager

from ..config import ViewportLayers

LAYER_LABELS = (
    ("viewport_ui", "Viewport controls"),
    ("gizmos", "Transform and joint gizmos"),
    ("selection", "Selection feedback"),
    ("helpers", "Camera and light helpers"),
    ("perturbation", "Perturbation guides"),
    ("debug_3d", "3D debug drawings"),
    ("debug_2d", "2D canvas drawings"),
)

_DEFAULT_LAYERS = ViewportLayers()


def debug_layer_group(name: str) -> str:
    """Map built-in helper layers and public canvas layers to their content group."""
    if name.startswith("canvas2d:"):
        return "debug_2d"
    if name.startswith("ui.gizmo."):
        return "gizmos"
    if name == "ui.selection":
        return "selection"
    if name in ("ui.scene_entities", "ui.scene_entity_icons"):
        return "helpers"
    if name.startswith("ui.perturb."):
        return "perturbation"
    return "debug_3d"


@contextmanager
def visible_debug_layers(draw, layers: ViewportLayers):
    """Apply view visibility for one draw without changing the publisher's choices."""
    if draw is None or layers == _DEFAULT_LAYERS:
        yield
        return
    hidden = []
    try:
        for layer in draw.layers():
            if layer.visible and (
                not getattr(layers, debug_layer_group(layer.name))
                or layer.name in layers.hidden_debug_layers
            ):
                hidden.append(layer)
                layer.visible = False
        yield
    finally:
        for layer in hidden:
            layer.visible = True
