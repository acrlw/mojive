"""Resolve the renderer consistently for interactive and offscreen entry points."""

from __future__ import annotations

import os


def render_backend_name(renderer: str | None = None) -> str:
    """Prefer an explicit renderer, then MOJIVE_RENDERER, then legacy settings."""
    requested = renderer
    if requested is None:
        requested = os.environ.get(
            "MOJIVE_RENDERER",
            os.environ.get("MOJIVE_BACKEND", os.environ.get("FORGE_VIEWER_BACKEND", "")),
        )
    requested = requested.strip().lower()
    requested = {"forge": "opengl", "webgpu": "wgpu"}.get(requested, requested)
    if requested not in {"", "opengl", "wgpu"}:
        raise ValueError(f"Unsupported renderer: {requested!r}; expected 'opengl' or 'wgpu'")
    return requested or "opengl"
