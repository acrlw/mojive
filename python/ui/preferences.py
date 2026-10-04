"""Persistent editor preferences shared by UI configuration and localization."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def settings_path() -> Path:
    override = os.environ.get("MOJIVE_SETTINGS")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "mojive" / "settings.json"


@dataclass
class Preferences:
    """Own the saved values and their destination; callers resolve domain defaults."""

    path: Path | None = None
    values: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Preferences:
        path = settings_path() if path is None else path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            value = {}
        return cls(path, value if isinstance(value, dict) else {})

    def get(self, name: str, default: object = None) -> object:
        return self.values.get(str(name), default)

    def update(self, values: dict[str, object], *, persist: bool = True) -> None:
        updated = self.values | {str(name): value for name, value in values.items()}
        if persist and self.path is not None:
            self._save(updated)
        # Keep references exposed through Localizer's compatibility API valid.
        self.values.update(updated)

    def _save(self, values: dict[str, object]) -> None:
        payload = json.dumps(values, ensure_ascii=False, indent=2) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            # A failed write must not truncate the last usable settings file.
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
