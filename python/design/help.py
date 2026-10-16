"""Read concise Chinese parameter help for the native workbench."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache

from mojive.design.resources import ROOT

HELP_PATH = ROOT / "parameter-help.json"


@lru_cache(maxsize=1)
def _help():
    return json.loads(HELP_PATH.read_text(encoding="utf-8"))


def get_part_help(part):
    """Return a component's visual context, preview instructions and field help."""
    return deepcopy(_help()["parts"][part])


def get_field_help(part, field):
    """Return the meaning and visible effect of one catalogue field."""
    return deepcopy(_help()["parts"][part]["fields"][field])


def condition_help(part, field, effective):
    """Explain a field's disabled mode using already resolved component values."""
    condition = get_field_help(part, field).get("condition")
    if condition is None:
        return {"enabled": True, "reason": "此项可直接调整。", "enable": ""}
    value = effective.get(condition["field"])
    enabled = value == condition["value"]
    return {
        "enabled": enabled,
        "reason": "当前模式已启用此项。" if enabled else condition["disabledReason"],
        "enable": "" if enabled else condition["enabledWhen"],
    }
