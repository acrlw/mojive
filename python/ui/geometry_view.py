"""Geometry presentation controls for the scene tree and inspected link."""

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.render.geometry import GeometryView, geometry_view, set_geometry_view
from mojive.ui.controls import segmented_control, segmented_control_width
from mojive.ui.theme import THEME


def draw_geometry_view(backend, translate, *, compact=False, theme=THEME, node=None, submit=None):
    """Choose the scene preset or override the inspected link's own geometry."""
    t = translate
    views = tuple(GeometryView)
    current = (
        geometry_view(backend) if node is None else (node.geometry_view or GeometryView.DEFAULT)
    )
    first = "Default" if node is None else "Follow scene"
    labels = tuple(t(label) for label in (first, "Visual", "Collision", "Both"))
    selected = views.index(current)
    available = imgui.get_content_region_avail().x
    if compact and available < segmented_control_width(labels):
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##geometry_view", labels[selected]):
            for i, label in enumerate(labels):
                if imgui.selectable(label, selected == i)[0]:
                    selected = i
            imgui.end_combo()
    else:
        selected = segmented_control("geometry_view", labels, selected, theme=theme)
    if views[selected] != current:
        if node is None:
            set_geometry_view(backend, views[selected])
        else:
            submit(cmd.SetGeometryView(node.node_id, views[selected] if selected else None))
    if imgui.is_item_hovered():
        hint = (
            "Scene geometry"
            if node is None
            else "Only this geometry"
            if node.type is NodeType.GEOM
            else "Only this link's geometry"
        )
        imgui.set_tooltip(t(hint))
