"""Portable documents for model composition and Forge scene authoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .scene_io import scene_from_document, scene_to_document

FORMAT = "forge-viewer.workspace"
VERSION = 1


def save_workspace(workspace, path: str | Path) -> Path:
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
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def load_workspace(workspace, path: str | Path) -> None:
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("format") == "forge-viewer.scene":
        workspace.primary.new_scene()
        workspace.scene = scene_from_document(document)
        workspace.set_resource_roots(())
        return
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError(f"Unsupported Forge workspace format in {source}")
    workspace.primary.new_scene()
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
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    resource_roots = _document_resource_roots(document, source.parent)
    return tuple(
        candidate
        for item in document.get("models", ())
        if not (
            candidate := _resolve_resource(str(item["path"]), source.parent, resource_roots)
        ).is_file()
    )


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
