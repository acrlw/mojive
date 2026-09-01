"""Scene adapter discovery, availability, and construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SceneAdapter


PHYSICS_MODULES: dict[str, tuple[str, str]] = {
    "mujoco": ("mujoco", "MuJoCo"),
    "newton": ("newton", "Newton"),
    "toy": ("mojive.adapters.toy", "Toy physics"),
}


def physics_of(backend_name: str) -> str:
    return str(backend_name).split("-", 1)[0]


def physics_available(physics: str) -> tuple[bool, str]:
    entry = PHYSICS_MODULES.get(physics)
    if entry is None:
        return (
            False,
            f"Unknown physics backend {physics!r}. Available: {', '.join(sorted(PHYSICS_MODULES))}",
        )
    module, label = entry
    if importlib.util.find_spec(module) is None:
        return False, f"{label} is not installed (Python package {module})"
    return True, ""


def make_adapter(backend_name: str, asset_path: str | Path | None = None) -> SceneAdapter:
    """Create a registered physics adapter and optionally load an asset."""

    physics = physics_of(backend_name)
    ok, reason = physics_available(physics)
    if not ok:
        raise RuntimeError(f"Backend {backend_name!r} is unavailable: {reason}")

    if physics == "mujoco":
        from .mujoco_adapter import MuJoCoAdapter

        adapter: SceneAdapter = MuJoCoAdapter()
    elif physics == "toy":
        from .toy import ToyPhysicsAdapter

        adapter = ToyPhysicsAdapter()
    else:
        raise RuntimeError(f"Backend {backend_name!r} has no adapter implementation")

    if asset_path is not None:
        adapter.load(_resolve_asset(asset_path))
    return adapter


def _resolve_asset(asset: str | Path) -> Path:
    try:
        from ..assets import resolve
    except ImportError:
        path = Path(asset)
        if not path.exists():
            raise FileNotFoundError(f"Asset {asset!r} was not found") from None
        return path
    return Path(resolve(asset))
