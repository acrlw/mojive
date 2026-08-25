"""Static checks for the user-facing example programs."""

from __future__ import annotations

import ast
from pathlib import Path


def test_python_examples_are_import_safe_and_have_main_entry_points() -> None:
    examples = Path(__file__).parents[1] / "examples"
    scripts = sorted(examples.glob("*.py"))
    assert scripts
    for script in scripts:
        module = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        assert "main" in functions, script.name
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in module.body
        ), script.name
