"""Desktop image/file sharing, isolated from rendering and UI frame pacing."""

from __future__ import annotations

import base64
import io
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


def image_png(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as image, io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def file_targets(path: Path) -> dict[str, bytes]:
    uri = path.resolve().as_uri()
    return {
        "text/uri-list": (uri + "\r\n").encode(),
        "x-special/gnome-copied-files": ("copy\n" + uri).encode(),
        "application/x-kde-cutselection": b"0",
    }


def _run(args, **kwargs) -> None:
    # wl-copy forks a selection owner that retains stderr. A pipe would make
    # communicate() wait for that owner to exit even after copying succeeded.
    with tempfile.TemporaryFile() as errors:
        try:
            result = subprocess.run(
                args, stdout=subprocess.DEVNULL, stderr=errors, timeout=5, check=False, **kwargs
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Desktop clipboard did not respond within 5 seconds") from exc
        if result.returncode:
            errors.seek(0, 2)
            errors.seek(max(0, errors.tell() - 4096))
            raise RuntimeError(
                errors.read().decode(errors="replace").strip() or "Clipboard rejected the copy"
            )


def _copy_x11(path: Path, kind: str) -> None:
    # GTK owns the X selection in an isolated process. It must outlive this call
    # (and the viewer) until another application takes ownership of CLIPBOARD.
    process = subprocess.Popen(
        [sys.executable, "-m", "mojive.ui._clipboard_linux", kind, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GDK_BACKEND": "x11"},
        start_new_session=True,
    )
    try:
        if not select.select([process.stdout], [], [], 5)[0]:
            raise RuntimeError("Clipboard initialization timed out")
        response = process.stdout.readline().decode(errors="replace").strip()
        if response != "READY":
            raise RuntimeError(response or "Clipboard helper exited before taking ownership")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    threading.Thread(target=process.wait, name="mojive-clipboard-owner", daemon=True).start()


def copy_saved_file(path: Path, kind: str) -> None:
    """Copy an image or file using native MIME/file-drop types, never plain path text."""
    path = path.resolve(strict=True)
    if kind not in ("image", "file"):
        raise ValueError("clipboard kind must be image or file")
    if sys.platform.startswith("linux"):
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            mime = "image/png" if kind == "image" else "text/uri-list"
            payload = image_png(path) if kind == "image" else file_targets(path)[mime]
            _run(["wl-copy", "--type", mime], input=payload)
        elif os.environ.get("DISPLAY"):
            _copy_x11(path, kind)
        else:
            raise RuntimeError("No desktop clipboard is available; Wayland requires wl-copy")
    elif sys.platform == "darwin":
        script = """ObjC.import('AppKit');
function run(argv) {
    const item = argv[1] === 'image'
        ? $.NSImage.alloc.initWithContentsOfFile(argv[0])
        : $.NSURL.fileURLWithPath(argv[0]);
    if (!item) throw Error('Cannot load clipboard item');
    const board = $.NSPasteboard.generalPasteboard;
    board.clearContents;
    if (!board.writeObjects([item])) throw Error('Clipboard rejected the copy');
}"""
        _run(["osascript", "-l", "JavaScript", "-e", script, str(path), kind])
    elif sys.platform == "win32":
        script = """$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
if ($env:MOJIVE_CLIPBOARD_KIND -eq 'image') {
    $image = [System.Drawing.Image]::FromFile($env:MOJIVE_CLIPBOARD_PATH)
    try { [System.Windows.Forms.Clipboard]::SetDataObject($image, $true) }
    finally { $image.Dispose() }
} else {
    $files = New-Object System.Collections.Specialized.StringCollection
    [void]$files.Add($env:MOJIVE_CLIPBOARD_PATH)
    [System.Windows.Forms.Clipboard]::SetFileDropList($files)
}"""
        _run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-EncodedCommand",
                base64.b64encode(script.encode("utf-16le")).decode(),
            ],
            env={**os.environ, "MOJIVE_CLIPBOARD_PATH": str(path), "MOJIVE_CLIPBOARD_KIND": kind},
        )
    else:
        raise RuntimeError(f"Clipboard sharing is unavailable on {sys.platform}")


@dataclass(frozen=True)
class CopyResult:
    path: Path
    kind: str
    error: str = ""


class CaptureClipboard:
    """UI-owned publisher; one active copy and at most one newer pending selection."""

    def __init__(self, copy=copy_saved_file):
        self._copy = copy
        self._executor = None
        self._future = None
        self._pending = None
        self._pending_text: tuple[str, Callable[[str], None]] | None = None
        self._closed = False

    def submit(self, path: Path, kind: str) -> None:
        if self._closed:
            raise RuntimeError("The clipboard publisher is closed")
        # A clipboard has one selection; coalesce bursts without an unbounded
        # worker queue or allowing an older copy to finish after a newer one.
        self._pending_text = None
        self._pending = (path.resolve(), kind)
        if self._future is None:
            self._start_pending()

    @property
    def pending_text(self) -> str | None:
        """Expose the latest local text while an older desktop copy finishes."""
        return self._pending_text[0] if self._pending_text is not None else None

    def submit_text(self, text: str, publish: Callable[[str], None]) -> None:
        """Publish on the UI thread after any older file/image clipboard transfer."""
        if self._closed:
            raise RuntimeError("The clipboard publisher is closed")
        self._pending = None
        if self._future is None:
            publish(text)
        else:
            self._pending_text = (text, publish)

    def _start_pending(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mojive-clipboard"
            )
        self._future = self._executor.submit(self._publish, *self._pending)
        self._pending = None

    def _publish(self, path: Path, kind: str) -> CopyResult:
        try:
            self._copy(path, kind)
        except Exception as exc:
            return CopyResult(path, kind, str(exc))
        return CopyResult(path, kind)

    def poll(self) -> CopyResult | None:
        if self._future is None or not self._future.done():
            return None
        result = self._future.result()
        self._future = None
        if self._pending_text is not None:
            text, publish = self._pending_text
            self._pending_text = None
            publish(text)
            return None
        if self._pending is not None:
            self._start_pending()
            return None
        return result

    def close(self) -> CopyResult | None:
        self._closed = True
        result = None
        while self._future is not None:
            self._future.result()
            result = self.poll()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        return result
