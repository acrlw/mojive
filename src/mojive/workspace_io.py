"""Portable documents for model composition and Mojive scene authoring."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene_io import FORMAT as SCENE_FORMAT
from .scene_io import LEGACY_FORMATS as LEGACY_SCENE_FORMATS
from .scene_io import scene_from_document, scene_to_document

FORMAT = "mojive.workspace"
LEGACY_FORMATS = frozenset({"forge-viewer.workspace"})
VERSION = 1


@dataclass(frozen=True)
class MissingResource:
    """One unresolved model reference in a Mojive workspace document."""

    model_index: int
    model_id: int
    model_name: str
    reference: str
    expected_path: Path


@dataclass(frozen=True)
class ResourceRepair:
    """Result of rewriting missing model references in a workspace document."""

    repaired: int
    missing: tuple[MissingResource, ...]


def save_workspace(workspace, path: str | Path) -> Path:
    """Write model composition, resource roots, and authored entities as JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    resource_roots = _unique_paths((target.parent, *workspace.resource_roots))
    models = []
    for item in workspace.scene_models():
        model = {
            "id": item.model_id,
            "name": item.name,
            "path": _resource_path(item.path, resource_roots),
            "position": list(item.position),
            "rotation": [list(row) for row in item.rotation],
        }
        xml = workspace.scene_model_xml(item.model_id)
        if xml is not None:
            model["mjcf"] = xml
        models.append(model)
    document = {
        "format": FORMAT,
        "version": VERSION,
        "resource_roots": [_relative_path(root, target.parent) for root in resource_roots],
        "models": models,
        "scene": scene_to_document(workspace.scene),
    }
    if not any(item.model_id == 0 for item in workspace.scene_models()):
        root_mjcf = workspace.scene_model_xml(0)
        if root_mjcf is not None:
            document["root_mjcf"] = root_mjcf
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def load_workspace(workspace, path: str | Path) -> None:
    """Replace a workspace with models and entities loaded from a document."""
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    format_name = document.get("format")
    if format_name == SCENE_FORMAT or format_name in LEGACY_SCENE_FORMATS:
        workspace.primary.new_scene()
        workspace.scene = scene_from_document(document)
        workspace.set_resource_roots(())
        return
    if format_name != FORMAT and format_name not in LEGACY_FORMATS:
        raise ValueError(f"Unsupported Mojive workspace format in {source}")
    if document.get("version") != VERSION:
        raise ValueError(f"Unsupported Mojive workspace format in {source}")
    missing = _missing_resource_entries(document, source)
    if missing:
        names = ", ".join(item.reference for item in missing)
        raise FileNotFoundError(f"Missing workspace resources: {names}")
    workspace.primary.new_scene()
    if root_mjcf := document.get("root_mjcf"):
        workspace.primary.set_scene_model_xml(0, str(root_mjcf))
    resource_roots = _document_resource_roots(document, source.parent)
    workspace.set_resource_roots(tuple(root for root in resource_roots if root != source.parent))
    for model in document.get("models", ()):
        model_path = _resolve_resource(str(model["path"]), source.parent, resource_roots)
        rotation = np.asarray(model.get("rotation", np.eye(3)), np.float32).reshape(3, 3)
        model_id = workspace.primary.add_scene_model(
            model_path,
            np.asarray(model.get("position", (0.0, 0.0, 0.0)), np.float32),
            rotation,
        )
        if model_id < 0:
            raise RuntimeError(f"Failed to add {model_path.name}")
        if xml := model.get("mjcf"):
            workspace.primary.set_scene_model_xml(model_id, xml)
    workspace.scene = scene_from_document(document["scene"])


def missing_resources(path: str | Path) -> tuple[Path, ...]:
    """Return unresolved model paths referenced by a workspace."""
    return tuple(item.expected_path for item in missing_resource_entries(path))


def missing_resource_entries(path: str | Path) -> tuple[MissingResource, ...]:
    """Return structured details for unresolved workspace model references."""
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    return _missing_resource_entries(document, source)


def relocate_workspace_resource(
    path: str | Path, model_index: int, replacement: str | Path
) -> ResourceRepair:
    """Replace one missing model reference with a caller-selected file."""

    source = Path(path).expanduser().resolve()
    selected = Path(replacement).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Replacement resource is unavailable: {selected}")
    document = json.loads(source.read_text(encoding="utf-8"))
    models = document.get("models", ())
    if not 0 <= int(model_index) < len(models):
        raise IndexError(f"Workspace model index is unavailable: {model_index}")
    models[int(model_index)]["path"] = _relative_path(selected, source.parent)
    _write_document(source, document)
    return ResourceRepair(1, _missing_resource_entries(document, source))


def repair_workspace_resources(path: str | Path, search_root: str | Path) -> ResourceRepair:
    """Repair every unambiguous missing model reference found below a directory."""

    source = Path(path).expanduser().resolve()
    root = Path(search_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Resource directory is unavailable: {root}")
    document = json.loads(source.read_text(encoding="utf-8"))
    missing = _missing_resource_entries(document, source)
    candidates = {
        name: tuple(item.resolve() for item in root.rglob(name) if item.is_file())
        for name in {Path(item.reference).name for item in missing}
    }
    repaired = 0
    for item in missing:
        replacement = _resource_replacement(item, root, candidates[Path(item.reference).name])
        if replacement is None:
            continue
        document["models"][item.model_index]["path"] = _relative_path(replacement, source.parent)
        repaired += 1
    if repaired:
        _write_document(source, document)
    return ResourceRepair(repaired, _missing_resource_entries(document, source))


def _relative_path(path: Path, directory: Path) -> str:
    try:
        return path.resolve().relative_to(directory.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()


def _resource_path(path: Path, roots: tuple[Path, ...]) -> str:
    resolved = path.expanduser().resolve()
    candidates = []
    for root in roots:
        try:
            candidates.append(resolved.relative_to(root).as_posix())
        except ValueError:
            continue
    if candidates:
        return min(candidates, key=lambda value: (value.count("/"), len(value)))
    return _relative_path(resolved, roots[0])


def _document_resource_roots(document: dict, directory: Path) -> tuple[Path, ...]:
    values = document.get("resource_roots", (".",))
    roots = tuple(_absolute_path(str(value), directory) for value in values)
    return _unique_paths((directory, *roots))


def _resolve_resource(value: str, directory: Path, roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = tuple((root / path).resolve() for root in roots)
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _absolute_path(value: str, directory: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _unique_paths(paths) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _missing_resource_entries(document: dict, source: Path) -> tuple[MissingResource, ...]:
    resource_roots = _document_resource_roots(document, source.parent)
    missing = []
    for index, item in enumerate(document.get("models", ())):
        reference = str(item["path"])
        candidate = _resolve_resource(reference, source.parent, resource_roots)
        if candidate.is_file():
            continue
        missing.append(
            MissingResource(
                model_index=index,
                model_id=int(item.get("id", index)),
                model_name=str(item.get("name", Path(reference).stem)),
                reference=reference,
                expected_path=candidate,
            )
        )
    return tuple(missing)


def _resource_replacement(
    missing: MissingResource, root: Path, candidates: tuple[Path, ...]
) -> Path | None:
    reference = Path(missing.reference).expanduser()
    if not reference.is_absolute():
        exact = (root / reference).resolve()
        if exact.is_file():
            return exact
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    reference_parts = tuple(part.casefold() for part in reference.parts)
    ranked = []
    for candidate in candidates:
        relative_parts = tuple(part.casefold() for part in candidate.relative_to(root).parts)
        score = 0
        for expected, actual in zip(
            reversed(reference_parts), reversed(relative_parts), strict=False
        ):
            if expected != actual:
                break
            score += 1
        ranked.append((score, candidate))
    best_score = max(score for score, _candidate in ranked)
    best = [candidate for score, candidate in ranked if score == best_score]
    return best[0] if len(best) == 1 and best_score > 1 else None


def _write_document(path: Path, document: dict) -> None:
    temporary: Path | None = None
    mode = stat.S_IMODE(path.stat().st_mode)
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            json.dump(document, output, indent=2)
            temporary = Path(output.name)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
