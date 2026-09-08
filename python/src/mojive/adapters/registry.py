"""Scene adapter discovery, availability, and construction."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SceneAdapter


PHYSICS_MODULES: dict[str, tuple[str, str]] = {
    "mujoco": ("mujoco", "MuJoCo"),
    "newton": ("newton", "Newton"),
    "toy": ("mojive.adapters.toy", "Toy physics"),
}


# Factories are process-local and may close over engine configuration. Built-in
# factories stay lazy so importing the registry never imports a physics engine.
_ADAPTER_FACTORIES: dict[str, tuple[Callable[[], SceneAdapter], str]] = {}


def register_adapter(
    name: str, factory: Callable[[], SceneAdapter], *, label: str | None = None
) -> None:
    """Register a zero-argument adapter factory without replacing an existing name."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError("Adapter names must use lowercase letters, digits, underscores or hyphens")
    if name in PHYSICS_MODULES or name in _ADAPTER_FACTORIES or name == "mujoco-classic":
        raise ValueError(f"Adapter {name!r} is already registered or reserved")
    if not callable(factory):
        raise TypeError("Adapter factory must be callable")
    _ADAPTER_FACTORIES[name] = (factory, label or name)


def unregister_adapter(name: str) -> None:
    """Remove a process-local registration; built-in adapters cannot be removed."""
    del _ADAPTER_FACTORIES[name]


def registered_adapters() -> tuple[tuple[str, str], ...]:
    """Return custom adapter names and display labels in registration order."""
    return tuple((name, label) for name, (_factory, label) in _ADAPTER_FACTORIES.items())


def physics_of(backend_name: str) -> str:
    name = str(backend_name)
    return name if name in _ADAPTER_FACTORIES else name.split("-", 1)[0]


def physics_available(physics: str) -> tuple[bool, str]:
    if physics in _ADAPTER_FACTORIES:
        return True, ""
    entry = PHYSICS_MODULES.get(physics)
    if entry is None:
        return (
            False,
            f"Unknown physics backend {physics!r}. Available: {', '.join(sorted(PHYSICS_MODULES.keys() | _ADAPTER_FACTORIES.keys()))}",
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

    if physics in _ADAPTER_FACTORIES:
        adapter: SceneAdapter = _ADAPTER_FACTORIES[physics][0]()
    elif physics == "mujoco":
        from .mujoco_adapter import MuJoCoAdapter

        adapter = MuJoCoAdapter()
    elif physics == "toy":
        from .toy import ToyPhysicsAdapter

        adapter = ToyPhysicsAdapter()
    else:
        raise RuntimeError(f"Backend {backend_name!r} has no adapter implementation")

    if asset_path is not None:
        try:
            adapter.load(_resolve_asset(asset_path))
        except Exception:
            adapter.release()
            raise
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
