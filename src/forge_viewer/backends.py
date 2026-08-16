from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .adapters.registry import PHYSICS_MODULES, make_adapter, physics_available, physics_of

if TYPE_CHECKING:
    from .adapters.base import SceneAdapter

__all__ = [
    "BackendInfo",
    "available_backends",
    "backend_info",
    "default_backend",
    "make_adapter",
]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    physics: str
    renderer: str
    available: bool
    reason: str = ""

    role: str = ""


_RENDER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "forge": ("moderngl", "glfw"),
    "mjr_": ("mujoco",),
    "newton-gl": ("newton",),
}

_MATRIX: tuple[tuple[str, str, str, str], ...] = (
    # (name, physics, renderer, role)
    ("mujoco", "MuJoCo", "forge", "primary backend"),
    ("mujoco-classic", "MuJoCo", "mjr_", "reference renderer"),
    ("toy", "Toy physics", "forge", "adapter conformance example"),
    ("newton", "Newton", "newton-gl", "planned forge adapter"),
)


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    return [n for n in names if importlib.util.find_spec(n) is None]


def _classic_blocked() -> str:

    if sys.platform == "darwin":
        return (
            "MuJoCo's legacy renderer requires an OpenGL compatibility profile. "
            "Run reference comparisons on Linux or Windows."
        )
    return ""


def backend_info(name: str) -> BackendInfo:
    for entry in _MATRIX:
        if entry[0] == name:
            return _describe(*entry)
    known = ", ".join(e[0] for e in _MATRIX)
    return BackendInfo(
        name=name,
        physics="?",
        renderer="?",
        available=False,
        reason=f"Unknown backend {name!r}. Available backends: {known}",
    )


def _describe(name: str, physics_label: str, renderer: str, role: str) -> BackendInfo:
    reasons: list[str] = []

    ok, why = physics_available(physics_of(name))
    if not ok:
        reasons.append(why)

    physics_module = PHYSICS_MODULES.get(physics_of(name), ("", ""))[0]
    missing = [
        n for n in _missing_modules(_RENDER_REQUIREMENTS.get(renderer, ())) if n != physics_module
    ]
    if missing:
        reasons.append(f"renderer {renderer} is missing {', '.join(missing)}")

    if renderer == "mjr_":
        blocked = _classic_blocked()
        if blocked:
            reasons.append(blocked)

    if renderer == "newton-gl" and not reasons:
        reasons.append("adapter is not implemented")
        return BackendInfo(name, physics_label, renderer, False, "；".join(reasons), role)

    return BackendInfo(
        name=name,
        physics=physics_label,
        renderer=renderer,
        available=not reasons,
        reason="；".join(reasons),
        role=role,
    )


def available_backends() -> list[BackendInfo]:

    return [_describe(*entry) for entry in _MATRIX]


def default_backend() -> str:

    infos = available_backends()
    for info in infos:
        if info.name == "mujoco" and info.available:
            return info.name
    for info in infos:
        if info.available:
            return info.name
    return "mujoco"


def make_backend_adapter(backend_name: str, asset_path: str | Path | None = None) -> SceneAdapter:

    info = backend_info(backend_name)
    if not info.available:
        raise RuntimeError(f"Backend {backend_name!r} is unavailable: {info.reason}")
    return make_adapter(backend_name, asset_path)
