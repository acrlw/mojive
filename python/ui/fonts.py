"""ImGui atlas setup using shared, backend-independent font sources."""

from __future__ import annotations

import os
from pathlib import Path

from mojive.text.sources import (
    _MONO,
    _REMOTE_MONO,
    BASE_SIZE_PT,
    DOWNLOAD_TIMEOUT_S,
    FontReport,
    _resolve,
    _resolve_cjk,
)


def _bundled_mono() -> tuple[str, int] | None:
    try:
        from imgui_bundle import imgui_bundle_folder

        p = Path(imgui_bundle_folder()) / "assets" / "fonts" / "Inconsolata-Medium.ttf"
        return (str(p), 0) if p.is_file() else None
    except Exception:
        return None


def load(
    imgui,
    io,
    *,
    size_pt: float = BASE_SIZE_PT,
    allow_download: bool = True,
    timeout: float = DOWNLOAD_TIMEOUT_S,
) -> FontReport:
    rep = FontReport(size_pt=size_pt)
    io.fonts.clear()

    def cfg(*, merge: bool = False, index: int = 0):
        c = imgui.ImFontConfig()
        c.merge_mode = merge
        c.font_no = index
        return c

    mono, label = _resolve(
        _MONO,
        _REMOTE_MONO,
        prefer_remote=True,
        allow_download=allow_download,
        timeout=timeout,
        notes=rep.notes,
    )
    if mono is None:
        mono, label = (_bundled_mono(), "Inconsolata bundled with imgui")
        if mono is not None:
            rep.notes.append("Falling back to Inconsolata bundled with imgui")
    if mono is None:
        io.fonts.add_font_default(cfg())
        rep.notes.append("No monospace font is available; using the proportional built-in font")
    else:
        io.fonts.add_font_from_file_ttf(mono[0], size_pt, cfg(index=mono[1]))
        rep.mono = label
        rep.mono_path, rep.mono_index = mono

    cjk, label = _resolve_cjk(
        os.environ.get("MOJIVE_CJK_FONT"),
        allow_download=allow_download,
        timeout=timeout,
        notes=rep.notes,
    )
    if cjk is None:
        rep.notes.append("No CJK font is available; CJK text may render as missing glyphs")
    else:
        io.fonts.add_font_from_file_ttf(cjk[0], size_pt, cfg(merge=True, index=cjk[1]))
        rep.cjk = label
        rep.cjk_path, rep.cjk_index = cjk
    return rep
