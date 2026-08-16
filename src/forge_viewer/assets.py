from __future__ import annotations

import os
from pathlib import Path

ASSET_SUFFIXES: tuple[str, ...] = (".xml", ".urdf", ".mjcf")


class AssetNotFoundError(FileNotFoundError):
    pass


def assets_dir() -> Path:

    here = Path(__file__).resolve().parent
    for candidate in (here.parent.parent / "assets", here / "assets"):
        if candidate.is_dir():
            return candidate
    return here.parent.parent / "assets"


def list_assets() -> list[str]:

    root = assets_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix in ASSET_SUFFIXES)


def resolve(name: str) -> Path:

    if not name:
        raise AssetNotFoundError(_message("", "asset name is empty"))

    text = os.path.expanduser(name)
    as_path = Path(text)
    if as_path.is_file():
        return as_path.resolve()

    root = assets_dir()
    direct = root / as_path.name
    if as_path.suffix and direct.is_file():
        return direct.resolve()
    if not as_path.suffix:
        for suffix in ASSET_SUFFIXES:
            guess = root / (as_path.name + suffix)
            if guess.is_file():
                return guess.resolve()

    raise AssetNotFoundError(_message(name))


def _message(name: str, reason: str = "") -> str:
    root = assets_dir()
    head = reason or f"Asset {name!r} was not found as a path or in {root}"
    names = list_assets()
    if not names:
        return f"{head}. The asset directory is empty."
    return f"{head}. Available assets: {', '.join(names)}"
