"""Select production states in which a conditional design field is observable."""

from mojive.design.editor import MISSING, path_value


def set_parameter(audit, part, field, value):
    """Use the same native editor for a field's supporting presentation state."""
    bench = audit.bench
    previous = bench.part, bench.field_search
    try:
        bench.part = part
        rule = audit.runtime.catalog[part]["fields"][field]
        audit.input.set_field(part, field, rule, value)
    finally:
        bench.part, bench.field_search = previous
        audit.input.frames(2)


def support_fixed_width(audit, part, width):
    """Enable the numeric editor, then restore the width mode's true prior state."""
    original = audit.runtime.resolved_part(part, "widthMode")
    override = path_value(audit.bench.editor.spec, ("parts", part, "widthMode"))
    set_parameter(audit, part, "widthMode", "fixed")
    set_parameter(audit, part, "width", width)
    if override is MISSING:
        previous = audit.bench.part, audit.bench.field_search
        try:
            audit.bench.part = part
            audit.input.restore(part, "widthMode")
        finally:
            audit.bench.part, audit.bench.field_search = previous
            audit.input.frames(2)
    else:
        set_parameter(audit, part, "widthMode", original)


def prepare_presentation(audit, part, field):
    runtime = audit.runtime
    fields = runtime.catalog[part]["fields"]
    context = {}
    # Isolate parent inheritance from the preset's explicit child overrides.
    resets = []
    if field in ("paddingX", "paddingY"):
        resets.extend(
            (part, edge)
            for edge in (
                ("paddingLeft", "paddingRight")
                if field == "paddingX"
                else ("paddingTop", "paddingBottom")
            )
        )
    if part in ("toolbox", "rail", "panel_header") and field == "iconSize":
        resets.append(("icon_button", "iconSize"))
    if part == "application" and field in ("titleHeight", "statusHeight"):
        resets.append(("titlebar" if field == "titleHeight" else "statusbar", "height"))
    if field == "gap":
        resets.extend((part, axis) for axis in ("rowGap", "columnGap") if axis in fields)
    for owner, name in resets:
        if path_value(audit.bench.editor.spec, ("parts", owner, name)) is MISSING:
            continue
        previous = audit.bench.part, audit.bench.field_search
        try:
            audit.bench.part = owner
            audit.input.restore(owner, name)
        finally:
            audit.bench.part, audit.bench.field_search = previous
            audit.input.frames(2)
        context[owner + "." + name] = "inherited for this experiment"
    if part == "select_popup" and field == "widthMode":
        support_fixed_width(audit, part, 1100)
        context[part + ".width"] = 1100
    if part == "button_row" and field == "alignItems":
        set_parameter(audit, part, "height", 70)
        context[part + ".height"] = 70
    if part == "range_bounds" and field == "rowGap":
        set_parameter(audit, part, "columnGap", 100)
        context[part + ".columnGap"] = 100
    if part in ("button", "menu_trigger") and field == "widthMode":
        support_fixed_width(audit, "menu_trigger", 60)
        context["menu_trigger.width"] = 60
    if part == "select" and field == "widthMode":
        support_fixed_width(audit, part, 80)
        context[part + ".width"] = 80
    if part in ("panel", "panel_body") and field in ("widthMode", "heightMode"):
        original = runtime.resolved_part(part, field)
        override = path_value(audit.bench.editor.spec, ("parts", part, field))
        dimension, value = ("width", 160) if field == "widthMode" else ("height", 80)
        set_parameter(audit, part, field, "fixed")
        set_parameter(audit, part, dimension, value)
        if override is MISSING:
            audit.input.restore(part, field)
        else:
            set_parameter(audit, part, field, original)
        context[part + "." + dimension] = value
    if field in ("lineHeight", "lineHeightMode"):
        alignment = fields.get("alignItems")
        if alignment and runtime.resolved_part(part, "alignItems") == "center":
            set_parameter(audit, part, "alignItems", "start")
            context[part + ".alignItems"] = "start"
    if field == "lineHeightMode" and "lineHeight" in fields:
        original = runtime.resolved_part(part, field)
        override = path_value(audit.bench.editor.spec, ("parts", part, field))
        set_parameter(audit, part, field, "fixed")
        height = 60 if part == "application" else 40
        set_parameter(audit, part, "lineHeight", height)
        if override is MISSING:
            audit.input.restore(part, field)
        else:
            set_parameter(audit, part, field, original)
        context[part + ".lineHeight"] = height
    if field == "alignItems" and part == "range_number":
        set_parameter(audit, "input_number", "height", 44)
        context["input_number.height"] = 44
    if field == "alignItems" and part == "form_row":
        set_parameter(audit, part, "height", 60)
        context[part + ".height"] = 60
    if field == "alignItems" and part in ("viewport_toolbar", "timeline_toolbar", "output_toolbar"):
        set_parameter(audit, "button", "fontSize", 30)
        context["button.fontSize"] = 30
    if field == "rowGap" and part in ("rgba", "color_channels"):
        set_parameter(audit, "rgba", "columns", 2)
        context["rgba.columns"] = 2
    if field == "rowGap" and part == "vector_group":
        set_parameter(audit, part, "columns", 2)
        context[part + ".columns"] = 2
    if field == "rowGap" and part == "vector_row":
        set_parameter(audit, part, "layout", "top")
        context[part + ".layout"] = "top"
    if field == "rowGap" and part == "button_row":
        set_parameter(audit, "button", "widthMode", "fixed")
        set_parameter(audit, "button", "width", 180)
        context.update({"button.widthMode": "fixed", "button.width": 180})
    if part == "button_row" and field in ("gap", "columnGap", "align"):
        context["floatingInspectorWidth"] = audit.scene_state.get("floatingInspectorWidth")
    if part == "tabs" and field == "align":
        set_parameter(audit, "tab", "widthMode", "fixed")
        set_parameter(audit, "tab", "width", 56)
        context.update({"tab.widthMode": "fixed", "tab.width": 56})
    if part == "menu_trigger" and field == "align":
        set_parameter(audit, part, "widthMode", "fixed")
        set_parameter(audit, part, "width", 100)
        context.update({part + ".widthMode": "fixed", part + ".width": 100})
    if part == "segment" and field == "align":
        audit.viewer.app.workspace.activate("Scene")
        audit.input.frames(3)
        context["contentLayout"] = "Scene geometry-view choices"
    if part == "menu_item_content" and field in ("gap", "columnGap"):
        support_fixed_width(audit, "menu", 240)
        set_parameter(audit, "menu", "widthMode", "fixed")
        context.update({"menu.path": "Edit", "menu.width": 240, "menu.widthMode": "fixed"})
    if part == "camera_body" and field in ("fit", "heightMode"):
        original_mode = runtime.resolved_part(part, "heightMode")
        set_parameter(audit, "camera_preview", "width", 300)
        set_parameter(audit, part, "marginLeft", 4)
        set_parameter(audit, part, "marginRight", 8)
        set_parameter(audit, part, "heightMode", "fixed")
        set_parameter(audit, part, "height", 160)
        if field == "heightMode":
            set_parameter(audit, part, "heightMode", original_mode)
        context[part + ".height"] = 160
        context[part + ".heightMode"] = runtime.resolved_part(part, "heightMode")
        context.update(
            {"camera_preview.width": 300, part + ".marginLeft": 4, part + ".marginRight": 8}
        )
    if part == "timeline_key" and field in ("align", "alignItems"):
        size = runtime.catalog[part]["fields"]["iconSize"]["min"]
        set_parameter(audit, part, "iconSize", size)
        context[part + ".iconSize"] = size
    if part == "data_list" and field == "columnGap":
        set_parameter(audit, "stat_row", "align", "start")
        context["stat_row.align"] = "start"
    if part == "stat_row" and field in ("gap", "columnGap"):
        set_parameter(audit, part, "align", "start")
        context[part + ".align"] = "start"
    if part == "output_row" and field == "alignItems":
        set_parameter(audit, part, "height", 72)
        context[part + ".height"] = 72
    if part == "output_toolbar" and field == "alignItems":
        audit.viewer.app.workspace.open["Inspector"] = False
        audit.input.frames(3)
        context["Inspector.visible"] = False
    if part == "output_toolbar" and field == "rowGap":
        panel = audit.viewer.app.panels.get("output")
        if not panel.show_search:
            x, y, width, height = audit.viewer.app.workspace.ui.hits["output-search"]
            audit.input.event("add_mouse_pos_event", x + width / 2, y + height / 2)
            audit.input.event("add_mouse_button_event", 0, True)
            audit.input.event("add_mouse_button_event", 0, False)
            audit.input.frames(3)
        context["output.searchVisible"] = panel.show_search
    if part == "vector_row" and field in ("labelWidth", "labelGap", "columnGap"):
        set_parameter(audit, part, "layout", "left")
        context[part + ".layout"] = "left"
    return context
