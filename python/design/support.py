"""Describe concrete native consumers and component reference availability."""

from functools import lru_cache

from mojive.design.consumer_support import (
    PRODUCTION_CONTRACTS,
    UNAVAILABLE_FIELDS,
    UNAVAILABLE_PARTS,
    field_condition,
)
from mojive.design.runtime import PART_METRICS
from mojive.design.specification import SCHEMA, component_catalog


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
    for part, definition in catalog.items():
        if "statusKey" in definition:
            result[part] = {
                field: [
                    {
                        "source": "ui/workspace/status_components.py",
                        "description": "生产指标独立布局",
                    }
                ]
                for field in definition["fields"]
            }
        base = definition.get("basePart")
        if base is not None:
            result[part] = {field: list(rows) for field, rows in result.get(base, {}).items()}
        for kind in definition.get("appearance", ()):
            for field in SCHEMA["componentAppearance"][kind]:
                result.setdefault(part, {}).setdefault(
                    field,
                    [
                        {
                            "source": "ui/workspace/widgets.py",
                            "description": "组件分组文字与图标绘制",
                        }
                    ],
                )
    for part in UNAVAILABLE_PARTS:
        result.pop(part, None)
    for part, field in UNAVAILABLE_FIELDS:
        result.get(part, {}).pop(field, None)
    # Search keeps the serialized legacy field, but its painter owns a capsule.
    result.get("search", {}).pop("radius", None)
    return result


@lru_cache(maxsize=1)
def native_support():
    catalog = component_catalog()
    consumers = native_consumers(catalog)
    support = {
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
    support["search"]["fields"]["radius"].update(
        status="derived",
        nativeLabel="随高度计算",
        description=catalog["search"]["fields"]["radius"]["description"],
        condition="固定胶囊形；旧圆角值保留，但不独立参与绘制。",
    )
    return support
