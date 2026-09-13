"""Shader compilation yields UI frames and never publishes an obsolete edit."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from mojive.render.native.programs import ShaderReload


def test_compile_does_not_block_checks_and_obsolete_generation_is_not_published(tmp_path):
    sources = tmp_path / "sources"
    binaries = tmp_path / "shaders"
    sources.mkdir()
    binaries.mkdir()
    source = sources / "vertex.sc"
    source.write_text("first")
    applied = []
    watcher = ShaderReload(SimpleNamespace(reload_shaders=lambda: applied.append(True)), binaries)
    watcher.sources = sources
    watcher.enable()
    entered, resume = threading.Event(), threading.Event()

    def compile_source():
        entered.set()
        assert resume.wait(5)
        return watcher._files(), ""

    watcher._compile = compile_source
    source.write_text("second version")
    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            caller.submit(watcher.check).result(timeout=1)
            assert entered.wait(1)
            # A second UI frame completes while the shader compiler is blocked.
            caller.submit(watcher.check).result(timeout=1)
            assert not applied
            source.write_text("third version arrives during compilation")
            resume.set()
            watcher._build_future.result(timeout=1)
            watcher.check()
            assert not applied
            watcher.check()
            watcher._build_future.result(timeout=1)
            watcher.check()
            assert len(applied) == 1
        finally:
            resume.set()
            watcher.close()
    watcher._next_check = 0
    source.write_text("after close")
    watcher.check()
    assert watcher._build_future is None


def test_close_stops_owned_compiler_process(tmp_path, monkeypatch):
    import os
    import subprocess

    started = threading.Event()
    stopped = threading.Event()

    class Process:
        returncode = None
        pid = 12345

        def __enter__(self):
            started.set()
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, timeout):
            if stopped.wait(0.01):
                return "", ""
            raise subprocess.TimeoutExpired("compiler", timeout)

        def terminate(self):
            self.returncode = -15
            stopped.set()

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    if os.name == "posix":
        monkeypatch.setattr(os, "killpg", lambda pid, sig: stopped.set())
    watcher = ShaderReload(SimpleNamespace(), tmp_path)
    watcher.sources = tmp_path
    watcher.enable()
    (tmp_path / "source.sc").write_text("changed")
    watcher.check()
    assert started.wait(1)
    started_at = time.monotonic()
    watcher.close()
    assert stopped.is_set()
    assert time.monotonic() - started_at < 1
