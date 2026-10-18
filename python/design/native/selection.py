"""Choose local authoring groups before shared control classes."""


def selection_parts(records, catalog):
    """Order the closest semantic groups first, retaining every related class."""
    ordered = sorted(records, key=lambda record: record["hit"][2] * record["hit"][3])
    parts = tuple(
        dict.fromkeys(
            part for record in ordered for part in reversed(record["parts"]) if part in catalog
        )
    )
    groups = [part for part in parts if catalog[part].get("selectionScope") == "group"]
    return groups + [part for part in parts if part not in groups]
