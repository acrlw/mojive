"""Source download failures must not poison subsequent native viewer launches."""

import hashlib
import io

import pytest

from mojive.tools import build_imgui


@pytest.fixture
def source_download(monkeypatch):
    payload = b"complete upstream source archive"
    monkeypatch.setattr(build_imgui, "SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def open_source(url, *, timeout):
        assert url == build_imgui.URL and timeout > 0
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(build_imgui.urllib.request, "urlopen", open_source)
    return payload, calls


def test_valid_cached_source_needs_no_network(tmp_path, source_download):
    payload, calls = source_download
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(payload)
    build_imgui.ensure_source_archive(archive)
    assert calls == []


@pytest.mark.parametrize("cached", [False, True])
def test_missing_or_truncated_source_is_replaced_and_reused(tmp_path, source_download, cached):
    payload, calls = source_download
    archive = tmp_path / "source.tar.gz"
    if cached:
        archive.write_bytes(payload[:8])
    build_imgui.ensure_source_archive(archive)
    build_imgui.ensure_source_archive(archive)
    assert archive.read_bytes() == payload
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [archive]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_download_preserves_cache_and_allows_retry(
    tmp_path, monkeypatch, source_download, cached, interrupted
):
    payload, _ = source_download
    archive = tmp_path / "source.tar.gz"
    if cached:
        archive.write_bytes(b"old incomplete download")

    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell() and interrupted:
                raise TimeoutError("download interrupted")
            return super().read(size)

    original_open = build_imgui.urllib.request.urlopen
    monkeypatch.setattr(
        build_imgui.urllib.request, "urlopen", lambda *_a, **_k: BrokenResponse(payload[:8])
    )
    with pytest.raises((TimeoutError, RuntimeError), match=r"interrupted|checksum mismatch"):
        build_imgui.ensure_source_archive(archive)
    if cached:
        assert archive.read_bytes() == b"old incomplete download"
    else:
        assert not archive.exists()
    assert list(tmp_path.iterdir()) == ([archive] if cached else [])
    monkeypatch.setattr(build_imgui.urllib.request, "urlopen", original_open)
    build_imgui.ensure_source_archive(archive)
    assert archive.read_bytes() == payload
