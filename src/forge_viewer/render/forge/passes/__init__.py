from __future__ import annotations

import importlib
import pkgutil

from ....log import get_logger

log = get_logger("passes")

_SKIP = {"base"}

_loaded: list[str] = []
_failed: dict[str, str] = {}
_done = False


def load_all(force: bool = False) -> tuple[str, ...]:

    global _done
    if _done and not force:
        return tuple(_loaded)
    _loaded.clear()
    _failed.clear()
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name in _SKIP:
            continue
        try:
            importlib.import_module(f"{__name__}.{info.name}")
        except Exception as e:
            _failed[info.name] = f"{type(e).__name__}: {e}"
            log.error("Pass module {} failed to load: {}", info.name, e)
        else:
            _loaded.append(info.name)
    _done = True
    return tuple(_loaded)


def loaded() -> tuple[str, ...]:

    return tuple(_loaded)


def failed() -> dict[str, str]:

    return dict(_failed)
