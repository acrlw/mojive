"""Stable hierarchy queries do not scan the scene repeatedly between edits."""

from types import SimpleNamespace

from mojive.adapters.base import NodeType, SceneNode
from mojive.ui.panels.hierarchy import HierarchyPanel


def test_search_reuses_results_and_invalidates_on_query_type_and_structure():
    nodes = [SceneNode(1, "Left arm", NodeType.LINK), SceneNode(2, "Left joint", NodeType.JOINT)]
    session = SimpleNamespace(nodes=nodes, structure_generation=1)
    ctx = SimpleNamespace(session=session)
    panel = HierarchyPanel()
    panel._refresh(ctx)
    panel._filter = "LEFT"
    rows = panel._visible_rows()
    assert [node for node, _depth, _leaf in rows] == nodes

    class MustNotScan:
        def __iter__(self):
            raise AssertionError("Unchanged hierarchy search traversed the scene again")

    names, panel._search_names = panel._search_names, MustNotScan()
    assert panel._visible_rows() is rows
    panel._search_names = names
    panel._type_filter = "joint"
    assert [row[0] for row in panel._visible_rows()] == [nodes[1]]
    panel._filter = "right"
    assert panel._visible_rows() == []
    nodes[1].name = "Right joint"
    session.structure_generation += 1
    panel._refresh(ctx)
    assert [row[0] for row in panel._visible_rows()] == [nodes[1]]
    session.nodes = []
    session.structure_generation += 1
    panel._refresh(ctx)
    assert panel._visible_rows() == []


def test_filtered_search_keeps_matches_beyond_the_former_row_budget():
    nodes = [SceneNode(i, f"arm{i}", NodeType.LINK) for i in range(1500)]
    panel = HierarchyPanel()
    panel._refresh(SimpleNamespace(session=SimpleNamespace(nodes=nodes, structure_generation=1)))
    panel._filter = "arm"
    assert [row[0] for row in panel._visible_rows()] == nodes
    panel._filter = "arm1499"
    assert [row[0] for row in panel._visible_rows()] == [nodes[-1]]
