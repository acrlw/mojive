"""Forge render-pass registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PASS_ORDER: tuple[str, ...] = (
    "shadow",
    "reflect",
    "opaque",
    "id",
    "skybox",
    "tendon",
    "transparent",
    "outline",
    "debug",
    "gizmo",
    "present",
)

PassFactory = Callable[[], Any]

_REGISTRY: dict[str, PassFactory] = {}


def register_pass(name: str, factory: PassFactory) -> None:
    if name not in PASS_ORDER:
        raise ValueError(f"Unknown pass {name!r}. Available passes: {PASS_ORDER}")
    _REGISTRY[name] = factory


def registered() -> dict[str, PassFactory]:
    return dict(_REGISTRY)
