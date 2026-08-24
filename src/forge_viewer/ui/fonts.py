"""UI font discovery, download, and atlas setup."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_MONO: tuple[tuple[str, int], ...] = (
    ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/Monaco.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ("C:/Windows/Fonts/consola.ttf", 0),
)


_CJK: tuple[tuple[str, int], ...] = (
    (str(Path.home() / "Library/Fonts/SourceHanMonoSC-Regular.otf"), 0),
    (str(Path.home() / "Library/Fonts/SourceHanSansSC-Regular.otf"), 0),
    ("/Library/Fonts/SourceHanMonoSC-Regular.otf", 0),
    ("/Library/Fonts/SourceHanSansSC-Regular.otf", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/usr/share/fonts/opentype/source-han-mono/SourceHanMonoSC-Regular.otf", 0),
    ("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansMonoCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("C:/Windows/Fonts/SourceHanMonoSC-Regular.otf", 0),
    ("C:/Windows/Fonts/SourceHanSansSC-Regular.otf", 0),
    ("C:/Windows/Fonts/msyh.ttc", 0),
)


@dataclass(frozen=True)
class Remote:
    label: str
    url: str
    sha256: str
    size: int
    filename: str


_REMOTE_MONO = Remote(
    "JetBrains Mono",
    "https://github.com/JetBrains/JetBrainsMono/raw/v2.304/fonts/ttf/JetBrainsMono-Regular.ttf",
    "a0bf60ef0f83c5ed4d7a75d45838548b1f6873372dfac88f71804491898d138f",
    273900,
    "JetBrainsMono-Regular.ttf",
)
_REMOTE_CJK = Remote(
    "Noto Sans SC",
    "https://github.com/notofonts/noto-cjk/raw/Sans2.004/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
    "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9",
    8331336,
    "NotoSansSC-Regular.otf",
)

DOWNLOAD_TIMEOUT_S = 20.0


def cache_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "forge-viewer" / "fonts"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(remote: Remote, *, timeout: float = DOWNLOAD_TIMEOUT_S) -> tuple[Path | None, str]:
    dst = cache_dir() / remote.filename
    if dst.is_file() and dst.stat().st_size == remote.size and _digest(dst) == remote.sha256:
        return dst, f"{remote.label} loaded from cache"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(remote.url, timeout=timeout) as r:
                shutil.copyfileobj(r, tmp)
        got = _digest(tmp_path)
        if got != remote.sha256:
            tmp_path.unlink(missing_ok=True)
            return None, f"{remote.label} checksum mismatch ({got[:12]}...); download discarded"
        tmp_path.replace(dst)
        return dst, f"{remote.label} downloaded ({remote.size / 1e6:.1f} MB)"
    except Exception as e:
        return None, f"{remote.label} download failed: {type(e).__name__}: {e}"


BASE_SIZE_PT = 14.0


@dataclass
class FontReport:
    mono: str = "built-in default"
    cjk: str = ""
    mono_path: str = ""
    mono_index: int = 0
    cjk_path: str = ""
    cjk_index: int = 0
    size_pt: float = BASE_SIZE_PT
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        cjk = self.cjk or "none"
        return f"Font {self.mono} {self.size_pt:g}pt; CJK {cjk}"


def _first_existing(cands: tuple[tuple[str, int], ...]) -> tuple[str, int] | None:
    for path, idx in cands:
        if Path(path).is_file():
            return path, idx
    return None


def _bundled_mono() -> tuple[str, int] | None:
    try:
        from imgui_bundle import imgui_bundle_folder

        p = Path(imgui_bundle_folder()) / "assets" / "fonts" / "Inconsolata-Medium.ttf"
        return (str(p), 0) if p.is_file() else None
    except Exception:
        return None


def _resolve(
    local: tuple[tuple[str, int], ...],
    remote: Remote,
    *,
    prefer_remote: bool,
    allow_download: bool,
    timeout: float,
    notes: list[str],
) -> tuple[tuple[str, int] | None, str]:
    def remote_first() -> tuple[tuple[str, int] | None, str]:
        if not allow_download:
            notes.append(f"{remote.label} is unavailable and downloads are disabled")
            return None, ""
        path, why = fetch(remote, timeout=timeout)
        notes.append(why)
        return ((str(path), 0), remote.label) if path is not None else (None, "")

    def local_first() -> tuple[tuple[str, int] | None, str]:
        found = _first_existing(local)
        return (found, Path(found[0]).name) if found is not None else (None, "")

    order = (remote_first, local_first) if prefer_remote else (local_first, remote_first)
    for step in order:
        got, label = step()
        if got is not None:
            return got, label
    return None, ""


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

    configured_cjk = os.environ.get("FORGE_VIEWER_CJK_FONT")
    cjk_candidates = (
        ((str(Path(configured_cjk).expanduser()), 0), *_CJK) if configured_cjk else _CJK
    )
    cjk, label = _resolve(
        cjk_candidates,
        _REMOTE_CJK,
        prefer_remote=False,
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
