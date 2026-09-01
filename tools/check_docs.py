"""Validate documentation entry points and executable example coverage."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from mojive.cli import build_parser
from mojive.render.backend import DebugView, RenderFlag

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODULES = (
    "src/mojive/types.py",
    "src/mojive/commands.py",
    "src/mojive/adapters/base.py",
    "src/mojive/adapters/conformance.py",
    "src/mojive/scene.py",
    "src/mojive/composition.py",
    "src/mojive/session.py",
    "src/mojive/renderer.py",
    "src/mojive/render/backend.py",
    "src/mojive/render/debugdraw.py",
    "src/mojive/recording.py",
    "src/mojive/remote.py",
    "src/mojive/control_rpc.py",
    "src/mojive/scene_io.py",
    "src/mojive/workspace_io.py",
)
SNIPPET = re.compile(r'--8<--\s+["\']([^"\']+)["\']')
ASSET_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(assets/[A-Za-z0-9_./-]+\.(?:xml|urdf|png|jpg|jpeg|obj|stl|msh|ply))"
)
CURRENT_DOCS = (ROOT / "README.md", ROOT / "examples/README.md")
CONFIG_ENV_MODULES = (
    "src/mojive/composition.py",
    "src/mojive/renderer.py",
    "src/mojive/ui/app.py",
    "src/mojive/ui/fonts.py",
    "src/mojive/ui/localization.py",
    "src/mojive/ui/window.py",
    "src/mojive/ui/window_wgpu.py",
)
MOJIVE_ENV = re.compile(r'["\'](MOJIVE_[A-Z0-9_]+)["\']')


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
    catalogs = {
        relative: (ROOT / relative).read_text(encoding="utf-8")
        for relative in ("docs/guides/examples.md", "examples/README.md")
    }
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
        for catalog_path, catalog in catalogs.items():
            if path.name not in catalog:
                errors.append(f"{relative}: missing from {catalog_path}")
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


def cli_reference_errors() -> list[str]:
    """Return CLI commands missing from the hand-written command reference."""
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    reference_path = ROOT / "docs/reference/cli.md"
    reference = reference_path.read_text(encoding="utf-8")
    errors = []
    for command in sorted(subparsers.choices):
        if f"### `{command}`" not in reference:
            errors.append(
                f"{reference_path.relative_to(ROOT)}: missing command heading for {command!r}"
            )
    for value in (*RenderFlag, *DebugView):
        if f"`{value.value}`" not in reference:
            errors.append(
                f"{reference_path.relative_to(ROOT)}: missing renderer value {value.value!r}"
            )
    return errors


def configuration_reference_errors() -> list[str]:
    """Return runtime Mojive variables missing from the configuration reference."""
    reference_path = ROOT / "docs/reference/configuration.md"
    reference = reference_path.read_text(encoding="utf-8")
    variables = set()
    for relative in CONFIG_ENV_MODULES:
        variables.update(MOJIVE_ENV.findall((ROOT / relative).read_text(encoding="utf-8")))
    return [
        f"{reference_path.relative_to(ROOT)}: missing environment variable {variable!r}"
        for variable in sorted(variables)
        if f"`{variable}`" not in reference
    ]


def asset_reference_errors() -> list[str]:
    """Return user-facing Markdown references to unavailable repository assets."""
    errors = []
    paths = (*CURRENT_DOCS, *sorted((ROOT / "docs").rglob("*.md")))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for reference in sorted(set(ASSET_REFERENCE.findall(text))):
            if not (ROOT / reference).is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing asset {reference!r}")
    return errors


def stale_usage_errors() -> list[str]:
    """Return known obsolete user-facing invocation and UI descriptions."""
    forbidden = {
        ".venv/bin/python": "use `uv run python` in user-facing commands",
        ".venv/bin/mojive": "use `uv run mojive` in user-facing commands",
        "centered modal Settings": "Settings is a dockable non-modal panel",
        "Selecting a camera opens its live preview": "camera preview is disabled by default",
    }
    errors = []
    paths = (*CURRENT_DOCS, *sorted((ROOT / "docs").rglob("*.md")))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for stale, replacement in forbidden.items():
            if stale in text:
                errors.append(f"{path.relative_to(ROOT)}: {stale!r} is obsolete; {replacement}")
    return errors


def main() -> int:
    """Run focused documentation checks without importing graphics dependencies."""
    errors = [error for module in PUBLIC_MODULES for error in public_entry_errors(module)]
    errors.extend(example_errors())
    errors.extend(snippet_errors())
    errors.extend(cli_reference_errors())
    errors.extend(configuration_reference_errors())
    errors.extend(asset_reference_errors())
    errors.extend(stale_usage_errors())
    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        f"Documentation checks passed: {len(PUBLIC_MODULES)} API modules, "
        "CLI reference, examples, snippets, and asset links"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
