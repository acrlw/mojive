"""Read legacy drafts without publishing inactive controls as current specifications."""

RETIRED_PARTS = frozenset(
    {
        "agent_message",
        "agent",
        "selection_caption",
        "camera_footer",
        "workspace_presets",
        "agent_messages",
        "panel_footer",
    }
)
RETIRED_FIELDS = frozenset(
    {
        ("textarea", "textAlign"),
        ("badge", "textAlign"),
        ("dialog_footer", "rowGap"),
        ("dialog_footer", "alignItems"),
        ("textarea", "lineHeight"),
        ("link_button", "gap"),
        ("textarea", "lineHeightMode"),
        ("dialog_footer", "gap"),
        ("menu_trigger", "gap"),
        ("dialog_footer", "columnGap"),
        ("color_picker", "radius"),
        ("section", "radius"),
    }
)


def current_parts(parts):
    if not isinstance(parts, dict):
        return parts
    return {
        part: {
            field: value for field, value in values.items() if (part, field) not in RETIRED_FIELDS
        }
        if isinstance(values, dict)
        else values
        for part, values in parts.items()
        if part not in RETIRED_PARTS
    }
