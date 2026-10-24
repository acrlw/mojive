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


def test_patch_reads_utf8_sources_on_non_utf8_hosts(tmp_path, monkeypatch):
    path = tmp_path / "source.cpp"
    path.write_bytes("// \u2014 upstream comment\nold\n".encode("utf-8"))
    original = type(path).read_text

    def local_read(self, encoding=None, errors=None):
        return original(self, encoding=encoding or "cp936", errors=errors)

    monkeypatch.setattr(type(path), "read_text", local_read)
    build_imgui.replace_source(path, "old", "new")
    build_imgui.replace_source(path, "old", "new")
    assert path.read_bytes() == "// \u2014 upstream comment\nnew\n".encode("utf-8")


@pytest.fixture
def binding_sources(tmp_path):
    root = tmp_path / "external/bindings_generation/cpp"
    root.mkdir(parents=True)
    registrations, sources = [], []
    for name, variable, initializer, directory in build_imgui.UNUSED_BINDINGS:
        registrations.append(
            f'    _register_submodule("{name}");\n'
            f'    auto module_{variable} =  m.def_submodule("{name}");\n'
            f"    py_init_module_{initializer}(module_{variable});"
        )
        sources.append(f"    ${{external_dir}}/{directory}/bindings/pybind_{initializer}.cpp")
    entry = root / "pybind_imgui_bundle.cpp"
    entry.write_text("\n".join(registrations), encoding="utf-8")
    cmake = root / "all_pybind_files.cmake"
    cmake.write_text("\n".join(sources), encoding="utf-8")
    return tmp_path, entry, cmake


def test_binding_trim_is_repeatable_and_preserves_required_core(binding_sources):
    root, entry, cmake = binding_sources
    core = "\n    py_init_module_imgui(module_imgui);\n    py_init_module_portable_file_dialogs(module_pfd);"
    entry.write_text(entry.read_text() + core)
    build_imgui.trim_unused_bindings(root)
    result = entry.read_text(), cmake.read_text()
    build_imgui.trim_unused_bindings(root)
    assert result == (entry.read_text(), cmake.read_text())
    assert entry.read_text().endswith(core)
    assert "m.def_submodule" not in result[0]
    assert "${external_dir}" not in result[1]


def test_binding_trim_rejects_upstream_registration_drift(binding_sources):
    root, entry, cmake = binding_sources
    entry.write_text(entry.read_text().replace("py_init_module_hello_imgui", "renamed_initializer"))
    with pytest.raises(RuntimeError, match="unexpected upstream binding registration: hello_imgui"):
        build_imgui.trim_unused_bindings(root)
    assert "${external_dir}" in cmake.read_text()
