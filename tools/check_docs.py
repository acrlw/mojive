"""Validate documentation entry points and executable example coverage."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODULES = (
    "src/forge_viewer/types.py",
    "src/forge_viewer/commands.py",
    "src/forge_viewer/adapters/base.py",
    "src/forge_viewer/adapters/conformance.py",
    "src/forge_viewer/scene.py",
    "src/forge_viewer/composition.py",
    "src/forge_viewer/session.py",
    "src/forge_viewer/renderer.py",
    "src/forge_viewer/render/backend.py",
    "src/forge_viewer/render/debugdraw.py",
    "src/forge_viewer/recording.py",
    "src/forge_viewer/remote.py",
    "src/forge_viewer/control_rpc.py",
    "src/forge_viewer/scene_io.py",
    "src/forge_viewer/workspace_io.py",
)
SNIPPET = re.compile(r'--8<--\s+["\']([^"\']+)["\']')


def public_entry_errors(relative: str) -> list[str]:
    """Return undocumented public top-level definitions in one API module."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    errors = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or ast.get_docstring(node):
            continue
        errors.append(f"{relative}:{node.lineno}: public entry {node.name!r} has no docstring")
    return errors


def example_errors() -> list[str]:
    """Return syntax, module-docstring, entry-point, and catalog errors."""
    catalog = (ROOT / "docs/guides/examples.md").read_text(encoding="utf-8")
    errors = []
    for path in sorted((ROOT / "examples").glob("*.py")):
        relative = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            compile(tree, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{relative}:{exc.lineno}: {exc.msg}")
            continue
        if not ast.get_docstring(tree):
            errors.append(f"{relative}: module docstring is required")
        if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            errors.append(f"{relative}: main() entry point is required")
        if path.name not in catalog:
            errors.append(f"{relative}: missing from docs/guides/examples.md")
    return errors


def snippet_errors() -> list[str]:
    """Return documentation snippets that reference missing files."""
    errors = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for reference in SNIPPET.findall(text):
            if not (ROOT / reference).is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing snippet {reference!r}")
    return errors


def main() -> int:
    """Run focused documentation checks without importing graphics dependencies."""
    errors = [error for module in PUBLIC_MODULES for error in public_entry_errors(module)]
    errors.extend(example_errors())
    errors.extend(snippet_errors())
    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Documentation checks passed: {len(PUBLIC_MODULES)} API modules and examples catalog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
