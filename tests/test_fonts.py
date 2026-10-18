"""UI font selection policy."""

from pathlib import Path

import pytest

from mojive.text import sources as fonts


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
    monkeypatch.setattr(fonts, "_BUNDLED_CJK", tmp_path / "missing.ttf")
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


def test_bundled_misans_is_available_without_network_or_system_font_installation(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("Bundled CJK must not download a replacement")

    monkeypatch.setattr(fonts, "fetch", no_download)
    resolved, label = fonts._resolve_cjk(None, allow_download=False, timeout=0.1, notes=[])
    assert label == "MiSans"
    assert Path(resolved[0]).name == "MiSans-Regular.ttf"


def test_system_cjk_is_used_when_noto_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(fonts, "_BUNDLED_CJK", tmp_path / "missing.ttf")
    monkeypatch.setattr(fonts, "cache_dir", lambda: tmp_path / "cache")
    fallback = tmp_path / "SourceHanSansSC-Regular.otf"
    fallback.write_bytes(b"font")
    monkeypatch.setattr(fonts, "_NOTO_CJK", ())
    monkeypatch.setattr(fonts, "_CJK_FALLBACK", ((str(fallback), 0),))
    notes = []

    resolved, label = fonts._resolve_cjk(None, allow_download=False, timeout=0.1, notes=notes)

    assert resolved == (str(fallback), 0)
    assert label == Path(fallback).name


def test_downloads_disabled_still_accepts_a_verified_cached_font(monkeypatch, tmp_path):
    import hashlib

    data = b"cached font resource"
    cached = tmp_path / "font.ttf"
    cached.write_bytes(data)
    remote = fonts.Remote(
        "Test font",
        "https://example.invalid/font.ttf",
        hashlib.sha256(data).hexdigest(),
        len(data),
        cached.name,
    )
    monkeypatch.setattr(fonts, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        fonts.urllib.request,
        "urlopen",
        lambda *a, **kw: pytest.fail("Font discovery attempted a download"),
    )
    found, note = fonts.fetch(remote, allow_download=False)
    assert found == cached
    assert "loaded from cache" in note
    cached.write_bytes(b"damaged font")
    found, note = fonts.fetch(remote, allow_download=False)
    assert found is None
    assert "downloads are disabled" in note


@pytest.mark.parametrize("outcome", ("partial-download", "checksum-mismatch", "success"))
def test_font_download_cleans_temporary_files_and_replaces_cache_only_on_success(
    monkeypatch, tmp_path, outcome
):
    import hashlib
    import io

    data = b"verified font resource"
    remote = fonts.Remote(
        "Test font",
        "https://example.invalid/font.ttf",
        hashlib.sha256(data).hexdigest(),
        len(data),
        "font.ttf",
    )
    cached = tmp_path / remote.filename
    cached.write_bytes(b"prior cache contents")
    monkeypatch.setattr(fonts, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        fonts.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(data if outcome == "success" else b"incomplete"),
    )
    if outcome == "partial-download":

        def interrupted_copy(source, destination):
            destination.write(source.read(4))
            raise OSError("Download interrupted")

        monkeypatch.setattr(fonts.shutil, "copyfileobj", interrupted_copy)

    found, note = fonts.fetch(remote)
    assert tuple(tmp_path.iterdir()) == (cached,)
    if outcome == "success":
        assert found == cached
        assert cached.read_bytes() == data
        assert "downloaded" in note
    else:
        assert found is None
        assert cached.read_bytes() == b"prior cache contents"
        assert ("failed" if outcome == "partial-download" else "checksum mismatch") in note
