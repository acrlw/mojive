from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "forge_viewer"


def _module_name(path: Path) -> str:
    """Return the import name represented by a source path."""

    try:
        rel = path.resolve().relative_to(SRC).with_suffix("")
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["forge_viewer", *parts])


def _imports(path: Path) -> set[str]:
    """Collect absolute import names without importing the module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    full = _module_name(path)
    pkg = full if path.name == "__init__.py" else full.rsplit(".", 1)[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    out.add(node.module)
                    out.update(f"{node.module}.{a.name}" for a in node.names)
                continue
            base = pkg.split(".")

            base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
            prefix = ".".join([*base, node.module]) if node.module else ".".join(base)
            out.add(prefix)
            out.update(f"{prefix}.{a.name}" for a in node.names)
    return out


def _files(*subdirs: str) -> list[Path]:
    """Return Python source paths below selected package directories."""

    roots = [SRC / s for s in subdirs] if subdirs else [SRC]
    return sorted(p for root in roots for p in root.rglob("*.py") if root.exists())


def _hits(imports: set[str], prefix: str) -> set[str]:
    """Return imports equal to or below a package prefix."""

    return {i for i in imports if i == prefix or i.startswith(prefix + ".")}


# ---------------------------------------------------------------------------


def test_render_layer_does_not_import_ui():
    """Render modules remain independent of the UI layer."""

    bad = {}
    for path in _files("render"):
        hit = _hits(_imports(path), "forge_viewer.ui")
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad


def test_ui_layer_does_not_import_concrete_backend():
    """UI modules depend on render contracts instead of backend internals."""

    allowed_prefixes = ("forge_viewer.render.backend", "forge_viewer.render.debugdraw")
    bad = {}
    for path in _files("ui"):
        hit = _hits(_imports(path), "forge_viewer.render")
        hit = {h for h in hit if not h.startswith(allowed_prefixes)}

        hit = {h for h in hit if h != "forge_viewer.render"}
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad


def test_render_layer_does_not_import_physics():
    """Render modules remain independent of physics packages."""

    physics = ("mujoco", "newton", "warp")
    bad = {}
    for path in _files("render"):
        hit = {i for i in _imports(path) for p in physics if i == p or i.startswith(p + ".")}
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad


def test_shared_vocabulary_is_dependency_free():
    """Shared contracts remain importable without window or render dependencies."""

    forbidden = (
        "forge_viewer.render",
        "forge_viewer.ui",
        "forge_viewer.adapters",
        "moderngl",
        "glfw",
        "imgui",
    )
    bad = {}
    for name in ("types.py", "math3d.py", "commands.py"):
        path = SRC / name
        if not path.exists():
            continue
        hit = {i for i in _imports(path) for f in forbidden if i == f or i.startswith(f + ".")}
        if hit:
            bad[name] = sorted(hit)
    assert not bad


def test_adapters_do_not_import_render_internals():
    """Adapters publish shared scene contracts without backend imports."""

    bad = {}
    for path in _files("adapters"):
        hit = _hits(_imports(path), "forge_viewer.render.forge")
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad


def test_this_scan_needs_no_gpu_and_no_optional_deps():
    """Architecture checks remain runnable in the fast CPU layer."""

    hit = _imports(Path(__file__))
    external = {i.split(".")[0] for i in hit} - {"ast", "pathlib", "pytest", "__future__"}
    assert not external


@pytest.mark.parametrize("pkg", ["render", "ui", "adapters"])
def test_every_package_has_docstring(pkg: str):
    """Each architectural package documents its role at module level."""

    init = SRC / pkg / "__init__.py"
    assert init.exists()
    tree = ast.parse(init.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree)


@pytest.mark.parametrize("name", ["types.py", "commands.py", "math3d.py"])
def test_public_contract_definitions_have_docstrings(name: str):
    """Public shared definitions provide API documentation at their declaration."""

    tree = ast.parse((SRC / name).read_text(encoding="utf-8"), filename=name)
    missing = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and ast.get_docstring(node) is None
    ]
    assert not missing


def test_scene_adapter_base_methods_have_docstrings():
    """The default adapter contract documents every public operation."""

    path = SRC / "adapters" / "base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SceneAdapterBase"
    )
    missing = [
        node.name
        for node in adapter.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and ast.get_docstring(node) is None
    ]
    assert not missing
