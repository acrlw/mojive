from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mojive.ui.viewport_widgets import draw_status, draw_tool_glyph

PROBE_PATH = Path(__file__).resolve().parents[1] / "design/tools/render_ui_feasibility.py"


@pytest.mark.parametrize(
    ("name", "function"),
    (("draw_tool_glyph", draw_tool_glyph), ("draw_status", draw_status)),
)
def test_shared_widget_calls_match_runtime_signatures(name, function):
    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"), filename=str(PROBE_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]

    assert calls, f"no {name} calls found in {PROBE_PATH}"
    signature = inspect.signature(function)
    for call in calls:
        assert all(keyword.arg is not None for keyword in call.keywords)
        positional = [object()] * len(call.args)
        keywords = {keyword.arg: object() for keyword in call.keywords if keyword.arg is not None}
        try:
            signature.bind(*positional, **keywords)
        except TypeError as exc:
            pytest.fail(f"{PROBE_PATH}:{call.lineno}: {name}{signature}: {exc}")
