"""WGSL shader loading for the webgpu backend.

Deliberately minimal counterpart to ``render.forge.programs``: WGSL sources
live in ``shaders/*.wgsl`` next to this module and are read from disk;
``load_wgsl`` concatenates shared chunks (WGSL has no preprocessor).  No hot
reload and no include graph — pass the chunk names explicitly.
"""

from __future__ import annotations

from pathlib import Path

_SHADER_DIR = Path(__file__).parent / "shaders"


def load_wgsl(*names: str) -> str:
    """Read and concatenate WGSL sources from the shaders directory."""
    return "\n".join((_SHADER_DIR / name).read_text(encoding="utf-8") for name in names)
