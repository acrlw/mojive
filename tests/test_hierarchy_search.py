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
    hits, truncated = panel._filtered_nodes(nodes)
    assert hits == tuple(nodes) and not truncated

    class MustNotScan:
        def __iter__(self):
            raise AssertionError("Unchanged hierarchy search traversed the scene again")

    assert panel._filtered_nodes(MustNotScan())[0] is hits
    panel._type_filter = "joint"
    assert panel._filtered_nodes(nodes)[0] == (nodes[1],)
    panel._filter = "right"
    assert panel._filtered_nodes(nodes) == ((), False)
    assert panel._filtered_nodes(MustNotScan()) == ((), False)
    nodes[1].name = "Right joint"
    session.structure_generation += 1
    panel._refresh(ctx)
    assert panel._filtered_nodes(nodes)[0] == (nodes[1],)
    session.nodes = []
    session.structure_generation += 1
    panel._refresh(ctx)
    assert panel._filtered_nodes([]) == ((), False)


def test_search_cache_retains_only_the_visible_budget_and_updates_truncation():
    nodes = [SceneNode(i, f"arm{i}", NodeType.LINK) for i in range(10)]
    panel = HierarchyPanel()
    panel._refresh(SimpleNamespace(session=SimpleNamespace(nodes=nodes, structure_generation=1)))
    panel._filter = "arm"
    panel._row_budget = 3
    assert panel._filtered_nodes(nodes) == (tuple(nodes[:3]), True)
    panel._row_budget = 10
    assert panel._filtered_nodes(nodes) == (tuple(nodes), False)
