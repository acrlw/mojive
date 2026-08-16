from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "forge_viewer"


def _module_name(path: Path) -> str:

    try:
        rel = path.resolve().relative_to(SRC).with_suffix("")
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["forge_viewer", *parts])


def _imports(path: Path) -> set[str]:

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
    roots = [SRC / s for s in subdirs] if subdirs else [SRC]
    return sorted(p for root in roots for p in root.rglob("*.py") if root.exists())


def _hits(imports: set[str], prefix: str) -> set[str]:
    return {i for i in imports if i == prefix or i.startswith(prefix + ".")}


# ---------------------------------------------------------------------------


def test_render_layer_does_not_import_ui():

    bad = {}
    for path in _files("render"):
        hit = _hits(_imports(path), "forge_viewer.ui")
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad, f"渲染层 import 了 UI 层：{bad}"


def test_ui_layer_does_not_import_concrete_backend():

    allowed_prefixes = ("forge_viewer.render.backend", "forge_viewer.render.debugdraw")
    bad = {}
    for path in _files("ui"):
        hit = _hits(_imports(path), "forge_viewer.render")
        hit = {h for h in hit if not h.startswith(allowed_prefixes)}

        hit = {h for h in hit if h != "forge_viewer.render"}
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad, f"UI 层 import 了具体后端：{bad}"


def test_render_layer_does_not_import_physics():

    physics = ("mujoco", "newton", "warp")
    bad = {}
    for path in _files("render"):
        hit = {i for i in _imports(path) for p in physics if i == p or i.startswith(p + ".")}
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad, f"渲染层 import 了物理库：{bad}"


def test_shared_vocabulary_is_dependency_free():

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
    assert not bad, f"共同词汇表依赖了具体层：{bad}"


def test_adapters_do_not_import_render_internals():

    bad = {}
    for path in _files("adapters"):
        hit = _hits(_imports(path), "forge_viewer.render.forge")
        if hit:
            bad[str(path.relative_to(SRC))] = sorted(hit)
    assert not bad, f"适配器 import 了渲染器内部：{bad}"


def test_this_scan_needs_no_gpu_and_no_optional_deps():

    hit = _imports(Path(__file__))
    external = {i.split(".")[0] for i in hit} - {"ast", "pathlib", "pytest", "__future__"}
    assert not external, f"分层扫描引入了额外依赖：{sorted(external)}"


@pytest.mark.parametrize("pkg", ["render", "ui", "adapters"])
def test_every_package_has_docstring(pkg: str):

    init = SRC / pkg / "__init__.py"
    assert init.exists(), f"{pkg} 缺 __init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{pkg}/__init__.py 没有 docstring"
