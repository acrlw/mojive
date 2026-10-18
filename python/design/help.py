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
    from mojive.design.specification import component_catalog

    definition = component_catalog()[part]
    result = deepcopy(
        _help()["parts"][definition.get("helpPart", definition.get("basePart", part))]
    )
    result["label"] = definition["label"]
    if "previewArea" in definition:
        result["preview"]["area"] = definition["previewArea"]
    if "controlGroup" in definition:
        parent = get_part_help(definition["controlGroup"])
        result["preview"] = parent["preview"]
        result["context"] = "仅调整「" + parent["label"] + "」内的同类控件。"
    if "groupOwner" in definition:
        result["context"] = definition["label"] + "；只调整此面板内的同组组件。"
    for field, rule in definition["fields"].items():
        if field not in result["fields"] or ("helpPart" in definition and rule.get("description")):
            result["fields"][field] = get_field_help(part, field)
    return result


def get_field_help(part, field):
    """Return the meaning and visible effect of one catalogue field."""
    from mojive.design.specification import component_catalog

    definition = component_catalog()[part]
    base = definition.get("helpPart", definition.get("basePart", part))
    # A specialization's precise geometry description outranks shared help.
    if "helpPart" in definition and definition["fields"].get(field, {}).get("description"):
        return _appearance_help(part, field)
    existing = _help()["parts"][base]["fields"].get(field)
    return deepcopy(existing) if existing is not None else _appearance_help(part, field)


def _appearance_help(part, field):
    from mojive.design.specification import component_catalog

    rule = component_catalog()[part]["fields"][field]
    return {
        "summary": rule.get("description", rule["label"]),
        "reference": "仅作用于此类组件及其组内内容；其他组件组保留各自设置。",
        "unit": "像素" if rule["type"] == "number" else "",
        "shortLabel": rule["label"],
        "defaultHelp": "恢复此项删除局部覆盖，重新继承所属分组或全局默认值。",
    }


def condition_help(part, field, effective):
    """Explain a field's disabled mode using already resolved component values."""
    if (part, field) == ("search", "radius"):
        return {
            "enabled": False,
            "reason": get_field_help(part, field)["summary"],
            "enable": "调整搜索框高度、字号或内边距后，圆角会随实际高度自动更新。",
        }
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
