"""Describe concrete native consumers and component reference availability."""

from functools import lru_cache

from mojive.design.consumer_support import (
    PRODUCTION_CONTRACTS,
    UNAVAILABLE_FIELDS,
    UNAVAILABLE_PARTS,
    field_condition,
)
from mojive.design.runtime import PART_METRICS
from mojive.design.specification import component_catalog


def native_consumers(catalog=None):
    catalog = catalog or component_catalog()
    result = {}
    for part, field in PART_METRICS.values():
        result.setdefault(part, {})[field] = [
            {"source": "ui/workspace/design.py", "description": "生产共享尺寸"}
        ]
    for part, binding in PRODUCTION_CONTRACTS.items():
        result.setdefault(part, {}).update(
            {
                field: [{"source": binding.source, "description": "生产组件尺寸与排版"}]
                for field in binding.fields
                if field in catalog.get(part, {}).get("fields", {})
            }
        )
    for part in UNAVAILABLE_PARTS:
        result.pop(part, None)
    for part, field in UNAVAILABLE_FIELDS:
        result.get(part, {}).pop(field, None)
    return result


@lru_cache(maxsize=1)
def native_support():
    catalog = component_catalog()
    consumers = native_consumers(catalog)
    return {
        part: {
            "label": definition["label"],
            "fields": {
                field: {
                    "label": rule["label"],
                    "status": "applied" if field in consumers.get(part, {}) else "reference",
                    "nativeLabel": "生产界面" if field in consumers.get(part, {}) else "仅示例",
                    "description": "同类组件同步。"
                    if field in consumers.get(part, {})
                    else UNAVAILABLE_FIELDS.get((part, field))
                    or UNAVAILABLE_PARTS.get(part)
                    or "此字段尚未接入生产组件。",
                    "condition": field_condition(part, field, rule),
                    "consumers": consumers.get(part, {}).get(field, []),
                }
                for field, rule in definition["fields"].items()
            },
        }
        for part, definition in catalog.items()
    }
