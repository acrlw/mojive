"""UI font selection policy."""

from pathlib import Path

from forge_viewer.ui import fonts


def test_explicit_cjk_font_has_priority(tmp_path):
    configured = tmp_path / "custom.otf"
    configured.write_bytes(b"font")
    notes = []

    resolved, label = fonts._resolve_cjk(
        str(configured), allow_download=False, timeout=0.1, notes=notes
    )

    assert resolved == (str(configured), 0)
    assert label == "custom.otf"


def test_noto_download_precedes_system_fallback(monkeypatch, tmp_path):
    downloaded = tmp_path / "NotoSansSC-Regular.otf"
    downloaded.write_bytes(b"font")
    fallback = tmp_path / "SourceHanSansSC-Regular.otf"
    fallback.write_bytes(b"font")
    monkeypatch.setattr(fonts, "_NOTO_CJK", ())
    monkeypatch.setattr(fonts, "_CJK_FALLBACK", ((str(fallback), 0),))
    monkeypatch.setattr(fonts, "fetch", lambda *_args, **_kwargs: (downloaded, "downloaded"))
    notes = []

    resolved, label = fonts._resolve_cjk(None, allow_download=True, timeout=0.1, notes=notes)

    assert resolved == (str(downloaded), 0)
    assert label == "Noto Sans SC"
    assert notes == ["downloaded"]


def test_system_cjk_is_used_when_noto_is_unavailable(monkeypatch, tmp_path):
    fallback = tmp_path / "SourceHanSansSC-Regular.otf"
    fallback.write_bytes(b"font")
    monkeypatch.setattr(fonts, "_NOTO_CJK", ())
    monkeypatch.setattr(fonts, "_CJK_FALLBACK", ((str(fallback), 0),))
    notes = []

    resolved, label = fonts._resolve_cjk(None, allow_download=False, timeout=0.1, notes=notes)

    assert resolved == (str(fallback), 0)
    assert label == Path(fallback).name
