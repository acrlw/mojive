"""Complete, cached hierarchy browsing across scene structure changes."""

from dataclasses import replace
from types import SimpleNamespace

from mojive.adapters.base import NodeType, SceneNode
from mojive.ui.panels.hierarchy import HierarchyPanel


def test_large_hierarchy_keeps_every_root_and_filter_match():
    nodes = [SceneNode(i, f"node-{i}", NodeType.LINK) for i in range(2500)]
    ctx = SimpleNamespace(session=SimpleNamespace(structure_generation=1, nodes=nodes))
    panel = HierarchyPanel()
    panel._refresh(ctx)
    rows = panel._visible_rows()
    assert [node for node, _, _ in rows] == nodes
    assert panel._visible_rows() is rows
    panel._filter = "node-"
    filtered = panel._visible_rows()
    assert [node for node, _, _ in filtered] == nodes
    assert panel._visible_rows() is filtered
    panel._filter = "node-2499"
    assert panel._visible_rows() == [(nodes[-1], 0, True)]
    panel._type_filter = "joint"
    assert panel._visible_rows() == []


def test_expansion_changes_invalidate_cached_rows_without_recursive_traversal():
    nodes = [
        SceneNode(i, f"node-{i}", NodeType.LINK, parent=i - 1, children=[i + 1] if i < 1199 else [])
        for i in range(1200)
    ]
    ctx = SimpleNamespace(session=SimpleNamespace(structure_generation=1, nodes=nodes))
    panel = HierarchyPanel()
    panel._refresh(ctx)
    panel._open_state.update({node.node_id: True for node in nodes})
    assert len(panel._visible_rows()) == len(nodes)
    assert panel._visible_rows()[-1] == (nodes[-1], 1199, True)
    panel._open_state[0] = False
    assert panel._visible_rows() == [(nodes[0], 0, False)]


def test_recycled_node_indices_do_not_preserve_batch_selection_or_expansion():
    nodes = [SceneNode(i, f"joint-{i}", NodeType.JOINT) for i in range(3)]
    session = SimpleNamespace(structure_generation=1, nodes=nodes)
    ctx = SimpleNamespace(session=session)
    panel = HierarchyPanel()
    panel._refresh(ctx)
    panel._batch_selected = {1, 2}
    panel._open_state = {1: True, 2: False}
    panel._visible_rows()
    session.nodes = [nodes[0], replace(nodes[2], node_id=1)]
    session.structure_generation += 1
    panel._refresh(ctx)
    assert not panel._batch_selected
    assert not panel._open_state
    assert [node for node, _, _ in panel._visible_rows()] == session.nodes
