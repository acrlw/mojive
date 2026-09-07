"""Live viewport visibility controls, independent from recording."""

from dataclasses import replace

from imgui_bundle import imgui

from ..layers import LAYER_LABELS, debug_layer_group
from . import Panel, PanelContext, themed_checkbox


class LayersPanel(Panel):
    id = "layers"
    name = "Layers"
    default_open = False
    dock_with = "Inspector"

    def draw(self, ctx: PanelContext) -> None:
        layers = ctx.viewport_layers
        if layers is None:
            return
        imgui.text_wrapped(ctx.tr("Choose the content visible in the viewport."))
        imgui.spacing()
        for name, label in LAYER_LABELS:
            changed, value = themed_checkbox(ctx.tr(label), getattr(layers, name), ctx.theme)
            if changed:
                layers = replace(layers, **{name: value})
                ctx.set_viewport_layers(layers)
        draw = ctx.backend.debug
        if draw is None:
            return
        items = [
            layer
            for layer in draw.layers()
            if debug_layer_group(layer.name) in ("debug_3d", "debug_2d")
            and (layer.primitives or layer.name in layers.hidden_debug_layers)
        ]
        if not items:
            return
        imgui.spacing()
        imgui.separator_text(ctx.tr("Named drawing layers"))
        for layer in items:
            group = debug_layer_group(layer.name)
            imgui.begin_disabled(not getattr(layers, group))
            label = layer.name.removeprefix("canvas2d:")
            changed, value = themed_checkbox(
                f"{label}##drawing_{layer.name}",
                layer.name not in layers.hidden_debug_layers,
                ctx.theme,
            )
            if changed:
                hidden = tuple(name for name in layers.hidden_debug_layers if name != layer.name)
                if not value:
                    hidden += (layer.name,)
                layers = replace(layers, hidden_debug_layers=hidden)
                ctx.set_viewport_layers(layers)
            imgui.end_disabled()
