"""Declare panel-owned component groups without duplicating control schemas."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from mojive.design.resources import ROOT


@lru_cache(maxsize=1)
def component_groups():
    return json.loads((ROOT / "component-groups.json").read_text(encoding="utf-8"))


def group_part(owner, base):
    return component_groups()[owner]["id"] + "_" + base


CURRENT_GROUPS = ContextVar("design_component_groups", default=())


@contextmanager
def group_scope(owner):
    """Keep a panel's local geometry and appearance inside its drawing boundary."""
    if owner not in component_groups():
        raise ValueError(f"Unknown component group owner: {owner}")
    token = CURRENT_GROUPS.set((*CURRENT_GROUPS.get(), owner))
    try:
        yield
    finally:
        CURRENT_GROUPS.reset(token)


@lru_cache(maxsize=2048)
def _scoped_parts(parts, owners):
    result = []
    for part in parts:
        result.append(part)
        for owner in reversed(owners):
            if part in component_groups()[owner]["parts"]:
                result.append(group_part(owner, part))
                break
    return tuple(dict.fromkeys(result))


def scoped_parts(parts):
    parts = (parts,) if isinstance(parts, str) else tuple(parts)
    owners = CURRENT_GROUPS.get()
    return _scoped_parts(parts, owners) if owners else parts


def add_group_definitions(catalog):
    """Alias existing field defaults while retaining independently saved overrides."""
    for owner, group in component_groups().items():
        for base, label in group["parts"].items():
            source = catalog[base]
            identity = group_part(owner, base)
            if identity in catalog:
                raise ValueError(f"Duplicate component group: {identity}")
            catalog[identity] = {
                **source,
                "id": identity,
                "label": group["label"] + " · " + label,
                "selectionScope": "group",
                "groupOwner": owner,
                "basePart": base,
                "previewArea": group["area"],
                "selectors": [],
                "fields": {
                    field: {**rule, "default": {"part": base, "field": field}}
                    for field, rule in source["fields"].items()
                },
            }
            if base in group.get("controlKinds", {}):
                catalog[identity]["controlKinds"] = group["controlKinds"][base]
    shared_groups = {base for group in component_groups().values() for base in group["parts"]}
    for base in shared_groups:
        catalog[base].pop("selectionScope", None)
    # A group's container and its controls have separate boxes. Give controls
    # class fields instead of inheriting the container's geometry.
    for parent, definition in tuple(catalog.items()):
        controls = {}
        for base in definition.get("controlKinds", ()):
            source = catalog[base]
            fields = source["fields"]
            if definition.get("basePart", parent) == "segment":
                # Contiguous segments keep outer spacing on their shared shell.
                fields = {
                    field: rule for field, rule in fields.items() if not field.startswith("margin")
                }
            shared = (
                catalog.get(definition.get("basePart"), {}).get("controlParts", {}).get(base, base)
            )
            identity = parent + "_" + base
            controls[base] = identity
            catalog[identity] = {
                **source,
                "id": identity,
                "label": definition["label"] + " · " + source["label"],
                "selectionScope": definition.get("selectionScope", "class"),
                "controlGroup": parent,
                "basePart": base,
                "helpPart": base,
                "controlDefaults": shared,
                "selectors": [],
                "fields": {
                    field: {**rule, "default": {"part": shared, "field": field}}
                    for field, rule in fields.items()
                },
            }
        if controls:
            definition["controlParts"] = controls


@lru_cache(maxsize=4096)
def control_parts(parts, parents):
    """Use the closest declared control group for measurement and painting alike."""
    from mojive.design.specification import component_catalog

    catalog = component_catalog()
    if any("controlGroup" in catalog.get(part, {}) for part in parts):
        return parts
    for parent in reversed(parents):
        controls = catalog.get(parent, {}).get("controlParts", {})
        for base in reversed(parts):
            if base in controls:
                local = controls[base]
                chain = [local]
                while "controlGroup" in catalog.get(catalog[chain[-1]].get("controlDefaults"), {}):
                    chain.append(catalog[chain[-1]]["controlDefaults"])
                return (*parts, *reversed(chain))
    return parts
