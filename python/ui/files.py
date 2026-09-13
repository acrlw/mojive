"""Resolve local output paths and reveal files without blocking the frame loop."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from threading import Thread

from ..log import get_logger

log = get_logger("ui.files")
_PATH_START = re.compile(r"(?<![\w/])(?:/|[A-Za-z]:[\\/]|\.{1,2}/|~/|output/)")


def message_path(text: str, copy_text: str | None = None) -> Path | None:
    """Prefer explicit path metadata, then resolve a path embedded in a message."""
    candidates = ([copy_text] if copy_text else []) + [
        text[match.start() :].splitlines()[0] for match in _PATH_START.finditer(text)
    ]
    for candidate in candidates:
        while candidate:
            value = candidate.strip().strip("\"'")
            for spelling in (value, value.rstrip(".,;:!?，。；：！？)]}")):
                try:
                    path = Path(spelling).expanduser()
                    if path.exists():
                        return path.resolve()
                except (OSError, ValueError):
                    pass
            candidate = candidate.rsplit(" ", 1)[0] if " " in candidate else ""
    return None


def _reveal(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            command = ["open", "-R", str(path)]
        elif sys.platform == "win32":
            command = ["explorer", "/select,", str(path)]
        else:
            command = [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.freedesktop.FileManager1",
                "--object-path",
                "/org/freedesktop/FileManager1",
                "--method",
                "org.freedesktop.FileManager1.ShowItems",
                f"['{path.as_uri()}']",
                "",
            ]
        try:
            subprocess.run(
                command, check=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError):
            if sys.platform in ("darwin", "win32"):
                raise
            # File managers without ShowItems can still open the containing directory.
            subprocess.run(
                ["xdg-open", str(path if path.is_dir() else path.parent)],
                check=True,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Cannot reveal {}: {}", path, exc)


def reveal_path(path: Path) -> None:
    Thread(target=_reveal, args=(path,), name="mojive-reveal-file", daemon=True).start()
