"""WGSL source loading and hot-reload tracking."""

from __future__ import annotations

from pathlib import Path

_SHADER_DIR = Path(__file__).parent / "shaders"


def load_wgsl(*names: str) -> str:
    """Read and concatenate WGSL sources from the shaders directory."""
    return "\n".join((_SHADER_DIR / name).read_text(encoding="utf-8") for name in names)


class WgslWatch:
    """Mtime tracking for a fixed set of WGSL sources.

    The change-detection half of forge's ``ProgramCache.reload_changed``:
    paths resolve against the module shaders directory at call time, and
    ``changed`` reports any edit (or removal) since the last ``mark``.
    """

    def __init__(self, *names: str) -> None:
        self.names = names
        self._mtimes: dict[Path, float] = {}
        self.mark()

    def _mtimes_now(self) -> dict[Path, float]:
        paths = (_SHADER_DIR / name for name in self.names)
        return {p: p.stat().st_mtime for p in paths if p.exists()}

    def changed(self) -> bool:
        return self._mtimes_now() != self._mtimes

    def mark(self) -> None:
        self._mtimes = self._mtimes_now()
