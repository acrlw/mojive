from __future__ import annotations

import ast
from io import StringIO
from pathlib import Path

from forge_viewer.log import configure, get_logger


def test_loguru_output_is_compact_and_component_scoped():
    stream = StringIO()
    configure(stream=stream)

    get_logger("test").info("Ready with {} object", 1)

    line = stream.getvalue()
    assert "[forge/test]" in line
    assert "INFO" in line
    assert "Ready with 1 object" in line
    assert "20" not in line  # no timestamp noise


def test_runtime_does_not_fall_back_to_standard_logging():
    root = Path(__file__).parents[1] / "src" / "forge_viewer"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        uses_logging = any(
            (isinstance(node, ast.Import) and any(name.name == "logging" for name in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "logging")
            for node in ast.walk(tree)
        )
        assert not uses_logging, f"{path} imports standard logging"
