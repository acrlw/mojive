"""Validate and resolve shared native component specifications."""

import json
import math
import re
from functools import lru_cache

from mojive.design.groups import add_group_definitions
from mojive.design.resources import ROOT

SCHEMA = json.loads((ROOT / "tokens-schema.json").read_text())
DEFAULT = json.loads((ROOT / "design-spec.json").read_text())
# Authored painter fallbacks predate the adopted default preset. Compare edits
# against that baseline so tuned defaults also override the original geometry.
BASELINE = json.loads((ROOT / "design-baseline.json").read_text())
FAMILIES = set(SCHEMA["componentFamilies"])
LOCAL_FIELDS = {
    "fontSize": (9, 36),
    "lineHeight": (9, 60),
    **dict.fromkeys(("marginLeft", "marginRight", "marginTop", "marginBottom"), (-80, 80)),
    **dict.fromkeys(("paddingLeft", "paddingRight", "paddingTop", "paddingBottom"), (0, 80)),
    "gap": (0, 64),
    "iconSize": (8, 64),
    "height": (12, 600),
    "width": (12, 1200),
    "borderRadius": (0, 100),
    "offsetX": (-600, 600),
    "offsetY": (-600, 600),
}
CATALOG_PATH = ROOT / "component-catalog.json"


@lru_cache(maxsize=1)
def component_catalog():
    """Read the same metadata used by the component specimens and workbench."""
    entries = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("部件目录必须为数组")
    catalog = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("部件目录的标识无效")
        if entry["id"] in catalog or not isinstance(entry.get("fields"), dict):
            raise ValueError("部件目录含重复标识或无效字段")
        # Shared appearance rules keep each component's capabilities explicit
        # without duplicating font and color metadata throughout the catalogue.
        appearance = {}
        for kind in entry.get("appearance", ()):
            appearance.update(SCHEMA["componentAppearance"][kind])
        entry["fields"] = {**appearance, **entry["fields"]}
        catalog[entry["id"]] = entry
    add_group_definitions(catalog)
    return catalog


def validate_parts(parts):
    """Reject unknown, non-finite and out-of-range component properties."""
    if not isinstance(parts, dict):
        raise ValueError("部件规格必须为对象")
    if not parts:
        return parts
    catalog = component_catalog()
    for part_id, values in parts.items():
        definition = catalog.get(part_id)
        if definition is None:
            raise ValueError(f"未知部件：{part_id}")
        if not isinstance(values, dict):
            raise ValueError(f"{definition['label']}：部件参数必须为对象")
        for field, value in values.items():
            rule = definition["fields"].get(field)
            if rule is None:
                raise ValueError(f"{definition['label']}：未知参数 {field}")
            name = f"{definition['label']} / {rule['label']}"
            if rule["type"] == "number":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not rule["min"] <= value <= rule["max"]
                    or not math.isfinite(value)
                    or (rule.get("step") == 1 and int(value) != value)
                ):
                    raise ValueError(f"{name}：需要 {rule['min']} 至 {rule['max']} 的有限数字")
            elif rule["type"] == "select":
                if not any(
                    type(value) is type(choice) and value == choice for choice in rule["choices"]
                ):
                    raise ValueError(f"{name}：需要从有效选项中选择")
            elif rule["type"] == "color":
                if not _valid_color(value, inherit=rule.get("inherit", False)):
                    raise ValueError(f"{name}：需要 inherit 或六位、八位十六进制颜色值")
            else:
                raise ValueError(f"{name}：目录字段类型无效")
    return parts


def resolve_part_value(spec, part_id, field, *, family=None, catalog=None, seen=()):
    """Resolve explicit values and catalogue token/layout/field defaults consistently."""
    catalog = component_catalog() if catalog is None else catalog
    values = spec.get("parts", {}).get(part_id, {})
    if field in values:
        return values[field]
    identity = (part_id, field)
    if identity in seen:
        raise ValueError(f"部件默认字段循环：{part_id}.{field}")
    default = catalog[part_id]["fields"][field]["default"]
    if isinstance(default, dict):
        if "field" in default:
            return resolve_part_value(
                spec,
                default.get("part", part_id),
                default["field"],
                family=family,
                catalog=catalog,
                seen=(*seen, identity),
            )
        if "token" in default:
            tokens = {**spec["tokens"], **spec.get("components", {}).get(family, {})}
            return tokens[default["token"]]
        if "layout" in default:
            return spec["layout"][default["layout"]]
        raise ValueError(f"部件默认字段来源无效：{part_id}.{field}")
    return default


def _valid_color(value, *, inherit=False):
    return isinstance(value, str) and (
        (inherit and value == "inherit")
        or re.fullmatch(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?", value) is not None
    )


def validate_tokens(tokens):

    if not isinstance(tokens, dict):
        raise ValueError("参数必须为对象")
    for key, value in tokens.items():
        if key not in SCHEMA["tokens"]:
            raise ValueError(f"未知参数：{key}")
        rule = SCHEMA["tokens"][key]
        if rule["type"] == "number":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not rule["min"] <= value <= rule["max"]
            ):
                raise ValueError(f"{rule['label']}：需要 {rule['min']} 至 {rule['max']}")
        elif rule["type"] == "color":
            if not _valid_color(value):
                raise ValueError(f"{rule['label']}：需要六位或八位十六进制颜色值")
        elif value not in rule["choices"]:
            raise ValueError(f"{rule['label']}：需要从有效选项中选择")


def validate_spec(value):
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("需要第 1 版设计规范")
    from mojive.design.themes import PRESETS

    palette = value.get("palette", {})
    if not isinstance(palette, dict) or any(
        role not in PRESETS["dark"] or not _valid_color(ink) for role, ink in palette.items()
    ):
        raise ValueError("Invalid theme palette")
    validate_tokens(value.get("tokens"))
    if set(value["tokens"]) != set(DEFAULT["tokens"]):
        raise ValueError("保存的规范必须包含全部全局参数")
    if not isinstance(value.get("components", {}), dict):
        raise ValueError("模块覆盖必须为对象")
    for family, tokens in value.get("components", {}).items():
        if family not in FAMILIES:
            raise ValueError(f"未知模块：{family}")
        validate_tokens(tokens)
    elements = value.get("elements", {})
    if not isinstance(elements, dict):
        raise ValueError("元素覆盖必须为对象")
    for selector, record in elements.items():
        if (
            not isinstance(selector, str)
            or not selector.strip()
            or len(selector) > 500
            or any(c in selector for c in "{}")
        ):
            raise ValueError("元素选择器无效")
        if not isinstance(record, dict) or not isinstance(record.get("styles", {}), dict):
            raise ValueError("元素样式必须为对象")
        for key, number in record.get("styles", {}).items():
            if (
                key not in LOCAL_FIELDS
                or isinstance(number, bool)
                or not isinstance(number, (int, float))
            ):
                raise ValueError(f"元素参数无效：{key}")
            low, high = LOCAL_FIELDS[key]
            if not low <= number <= high:
                raise ValueError(f"{key}：需要 {low} 至 {high}")
    layout = value.get("layout")
    if not isinstance(layout, dict) or set(layout) != set(SCHEMA["layout"]):
        raise ValueError("保存的规范必须包含全部布局参数")
    for key, number in layout.items():
        rule = SCHEMA["layout"].get(key)
        if (
            rule is None
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not rule["min"] <= number <= rule["max"]
        ):
            raise ValueError(f"布局参数无效：{key}")
    native = value.get("native", {})
    if (
        not isinstance(native, dict)
        or not isinstance(native.get("g3"), bool)
        or isinstance(native.get("g3_smoothing"), bool)
        or not isinstance(native.get("g3_smoothing"), (int, float))
        or not SCHEMA["native"]["g3_smoothing"]["min"]
        <= native["g3_smoothing"]
        <= SCHEMA["native"]["g3_smoothing"]["max"]
    ):
        raise ValueError("原生圆角设置无效")
    native = {**DEFAULT["native"], **native}
    smoothing = native["g3_capsule_smoothing"]
    if (
        isinstance(smoothing, bool)
        or not isinstance(smoothing, (int, float))
        or not 0 <= smoothing <= 1
        or not isinstance(native["g3_capsule_optical"], bool)
    ):
        raise ValueError("原生胶囊设置无效")
    tools = value.get("tools", {})
    if (
        not isinstance(tools, dict)
        or not isinstance(tools.get("snap"), bool)
        or isinstance(tools.get("snap_step"), bool)
        or not isinstance(tools.get("snap_step"), (int, float))
        or not 1 <= tools["snap_step"] <= 32
    ):
        raise ValueError("吸附设置无效")
    return {
        **value,
        "native": native,
        "parts": validate_parts(value.get("parts", {})),
    }
