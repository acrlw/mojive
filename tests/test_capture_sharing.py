"""Saved captures, clipboard ownership, and nonblocking publication contracts."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mojive.capture import RecordingPhase
from mojive.config import RecordingConfig
from mojive.ui import clipboard
from mojive.ui.app import ViewerApp


@pytest.fixture
def app():
    value = ViewerApp.__new__(ViewerApp)
    value._released = False
    value._capture_requests = []
    value._capture_tasks = []
    value._viewport_recording_phase = RecordingPhase.IDLE
    value.recording_config = RecordingConfig()
    value.localizer = SimpleNamespace(text=lambda key: key)
    value.messages = []
    value.copies = []
    value.session = SimpleNamespace(
        report_message=lambda text, **kwargs: value.messages.append((text, kwargs)),
        document_id="scene",
        document_revision=1,
        structure_generation=0,
        frame=SimpleNamespace(step=0, time=0),
    )
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    value._surface_image = lambda *args, **kwargs: pixels

    def copy(path, kind):
        # Publication is permitted only after the actual output is readable.
        if kind == "image":
            with Image.open(path) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)
        else:
            assert path.read_bytes() == b"finalized"
        value.copies.append((path.resolve(), kind))

    value._capture_clipboard = SimpleNamespace(submit=copy)
    return value


def test_multiple_requests_save_every_file_and_share_one_surface_readback(app, tmp_path):
    reads = []
    read = app._surface_image
    app._surface_image = lambda *args: reads.append(args) or read(*args)
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        assert app.request_capture(path) == path
    app._finish_capture_and_recording(None, 0)
    assert app.copies == [(path, "image") for path in paths]
    assert len(reads) == 1
    assert not app._capture_requests


def test_default_names_do_not_collide_in_the_same_second(app, monkeypatch):
    import mojive.ui.app.capture as capture

    monkeypatch.setattr(capture.time, "strftime", lambda _: "20260916-123456")
    paths = {app.request_capture() for _ in range(100)}
    assert len(paths) == 100
    assert all(path.parent == Path("output") and path.suffix == ".png" for path in paths)


def test_closed_viewer_rejects_screenshot_and_recording(app):
    app._released = True
    with pytest.raises(RuntimeError, match="closed"):
        app.request_capture()
    with pytest.raises(RuntimeError, match="closed"):
        app.start_recording()
    with pytest.raises(RuntimeError, match="closed"):
        app.request_capture_async().result()


def test_failed_save_does_not_copy_and_later_capture_recovers(app, tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied")
    app.request_capture(parent / "image.png")
    app.request_capture(tmp_path / "valid.png")
    app._finish_capture_and_recording(None, 0)
    assert len(app.copies) == 1
    assert [message[1]["level"] for message in app.messages] == ["error", "success"]


def test_memory_capture_does_not_touch_clipboard_but_file_capture_does(app, tmp_path):
    memory = app.request_capture_async(memory=True)
    file = app.request_capture_async(tmp_path / "capture.png")
    canceled = app.request_capture_async(tmp_path / "canceled.png")
    canceled.cancel()
    app._finish_capture_and_recording(None, 0)
    assert memory.result()["image"].shape == (2, 3, 3)
    assert file.result()["path"] == str(tmp_path / "capture.png")
    assert app.copies == [(tmp_path / "capture.png", "image")]
    assert not (tmp_path / "canceled.png").exists()


def test_opt_out_preserves_save_without_copy(app, tmp_path):
    app.recording_config = RecordingConfig(copy_to_clipboard=False)
    app.request_capture(tmp_path / "capture.png")
    app._finish_capture_and_recording(None, 0)
    assert (tmp_path / "capture.png").exists()
    assert not app.copies
    assert RecordingConfig.from_mapping(asdict(app.recording_config)) == app.recording_config
    assert RecordingConfig.from_mapping({}).copy_to_clipboard
    assert RecordingConfig.from_mapping({"copy_to_clipboard": "false"}).copy_to_clipboard


@pytest.mark.parametrize("outcome", ("success", "failed", "canceled"))
def test_video_copies_only_after_successful_finalization(app, tmp_path, outcome):
    path = app.start_recording(tmp_path / "video.mp4", countdown=0)

    def close():
        assert not app.copies
        if outcome == "failed":
            raise RuntimeError("finalization failed")
        path.write_bytes(b"finalized")

    if outcome != "canceled":
        app._viewport_recorder = SimpleNamespace(close=close)
        app._viewport_recording_frames = 1
    app.stop_recording()
    assert app.copies == ([(path, "file")] if outcome == "success" else [])


def test_native_file_targets_are_uris_and_images_are_png(tmp_path):
    path = tmp_path / "中文 space #.jpg"
    Image.new("RGB", (2, 3), "red").save(path)
    assert clipboard.image_png(path).startswith(b"\x89PNG\r\n\x1a\n")
    targets = clipboard.file_targets(path)
    assert targets["text/uri-list"] == (path.as_uri() + "\r\n").encode()
    assert targets["x-special/gnome-copied-files"].startswith(b"copy\nfile:///")
    assert b"%20" in targets["text/uri-list"] and b"%23" in targets["text/uri-list"]


def test_clipboard_copy_never_blocks_submission_and_coalesces_to_latest(tmp_path):
    entered, release = threading.Event(), threading.Event()
    copied = []

    def copy(path, kind):
        copied.append(path.name)
        entered.set()
        assert release.wait(3)

    publisher = clipboard.CaptureClipboard(copy)
    try:
        publisher.submit(tmp_path / "first.png", "image")
        assert entered.wait(3)
        publisher.submit(tmp_path / "second.png", "image")
        publisher.submit(tmp_path / "last.mp4", "file")
        assert publisher.poll() is None
        release.set()
        result = publisher.close()
        assert copied == ["first.png", "last.mp4"]
        assert result.path.name == "last.mp4" and result.kind == "file" and not result.error
        assert publisher.close() is None
        with pytest.raises(RuntimeError, match="closed"):
            publisher.submit(tmp_path / "after.png", "image")
    finally:
        release.set()
        publisher.close()


def test_manual_text_wins_over_an_inflight_capture_on_the_ui_thread(tmp_path):
    from mojive.ui.window import _install_glfw_clipboard_callbacks

    entered, release = threading.Event(), threading.Event()
    native_text, copied, writes = ["previous"], [], []
    ui_thread = threading.get_ident()

    def copy_file(path, kind):
        entered.set()
        assert release.wait(3)
        copied.append(path.name)
        native_text[0] = None  # File targets do not contain an X11 text selection.

    def set_text(_window, text):
        writes.append(threading.get_ident())
        native_text[0] = text

    publisher = clipboard.CaptureClipboard(copy_file)
    platform = SimpleNamespace()
    api = SimpleNamespace(
        get_clipboard_string=lambda _window: native_text[0], set_clipboard_string=set_text
    )
    try:
        publisher.submit(tmp_path / "first.mp4", "file")
        assert entered.wait(3)
        publisher.submit(tmp_path / "superseded.mp4", "file")
        _install_glfw_clipboard_callbacks(
            api, SimpleNamespace(get_platform_io=lambda: platform), publisher
        )
        platform.platform_set_clipboard_text_fn(None, "first path")
        platform.platform_set_clipboard_text_fn(None, "latest path")
        assert not release.is_set()  # Manual copy does not wait on a desktop worker.
        assert platform.platform_get_clipboard_text_fn(None) == "latest path"
        release.set()
        publisher.close()
        assert copied == ["first.mp4"]
        assert native_text[0] == "latest path"
        assert writes == [ui_thread]
        assert platform.platform_get_clipboard_text_fn(None) == "latest path"
    finally:
        release.set()
        publisher.close()


def test_failed_clipboard_keeps_saved_file_and_reports_actionable_receipt(app, tmp_path):
    path = tmp_path / "capture.png"

    def fail(path, kind):
        raise RuntimeError("No desktop clipboard is available")

    app._capture_clipboard = clipboard.CaptureClipboard(fail)
    try:
        app.request_capture(path)
        app._finish_capture_and_recording(None, 0)
        deadline = time.monotonic() + 3
        while len(app.messages) < 2 and time.monotonic() < deadline:
            app._poll_capture_clipboard()
            time.sleep(0.001)
        assert path.exists()
        text, receipt = app.messages[-1]
        assert "Clipboard copy failed" in text and "No desktop clipboard" in text
        assert receipt["level"] == "warning" and receipt["copy_text"] == str(path)
    finally:
        app._capture_clipboard.close()


@pytest.mark.parametrize("kind", ("image", "file"))
def test_wayland_publishes_typed_content_without_shell(monkeypatch, tmp_path, kind):
    path = tmp_path / "中文 ' $ image.png"
    Image.new("RGB", (2, 3)).save(path)
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: "/usr/bin/wl-copy")
    monkeypatch.setattr(clipboard, "_run", lambda args, **kwargs: calls.append((args, kwargs)))
    clipboard.copy_saved_file(path, kind)
    command, options = calls[0]
    assert command == ["wl-copy", "--type", "image/png" if kind == "image" else "text/uri-list"]
    assert "shell" not in options
    assert options["input"] == (
        clipboard.image_png(path) if kind == "image" else (path.as_uri() + "\r\n").encode()
    )


@pytest.mark.parametrize("platform", ("darwin", "win32"))
def test_platform_commands_pass_paths_as_data(monkeypatch, tmp_path, platform):
    path = tmp_path / "中文 ' $ file.mp4"
    path.write_bytes(b"video")
    calls = []
    monkeypatch.setattr(clipboard.sys, "platform", platform)
    monkeypatch.setattr(clipboard, "_run", lambda args, **kwargs: calls.append((args, kwargs)))
    clipboard.copy_saved_file(path, "file")
    command, options = calls[0]
    assert "shell" not in options
    if platform == "darwin":
        assert command[-2:] == [str(path), "file"]
        assert str(path) not in command[4]
    else:
        assert "-STA" in command
        assert options["env"]["MOJIVE_CLIPBOARD_PATH"] == str(path)


def test_clipboard_timeout_has_a_short_error(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(clipboard.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="within 5 seconds"):
        clipboard._run(["clipboard"])


def test_x11_startup_failure_reaps_only_its_owned_helper(monkeypatch, tmp_path):
    import io

    events = []
    process = SimpleNamespace(
        stdout=io.BytesIO(b"Cannot connect to the X11 desktop clipboard\n"),
        kill=lambda: events.append("kill"),
        wait=lambda: events.append("wait"),
    )
    monkeypatch.setattr(clipboard.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(clipboard.select, "select", lambda *args: ([process.stdout], [], []))
    with pytest.raises(RuntimeError, match="Cannot connect"):
        clipboard._copy_x11(tmp_path / "capture.png", "image")
    assert events == ["kill", "wait"] and process.stdout.closed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX clipboard owner daemon")
@pytest.mark.integration
def test_copy_command_does_not_wait_for_daemon_inherited_stderr(tmp_path):
    import os
    import signal

    pid_file = tmp_path / "owner.pid"
    script = """import os, sys, time
from pathlib import Path
pid = os.fork()
if pid:
    Path(sys.argv[1]).write_text(str(pid))
else:
    time.sleep(30)
"""
    try:
        clipboard._run([sys.executable, "-c", script, str(pid_file)])
        assert pid_file.is_file()
    finally:
        if pid_file.is_file():
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
