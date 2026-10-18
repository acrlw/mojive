"""Concrete group specimens composed from the production native controls."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

from imgui_bundle import imgui as g

from mojive.design.native.geometry import Rect, corner_radius_scope
from mojive.design.specification import component_catalog
from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.button_rows import draw_button_row
from mojive.ui.workspace.component_layout import component_flow, control_frame_size, control_size
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.geometry_view import GEOMETRY_ICONS, GEOMETRY_VIEWS
from mojive.ui.workspace.group_corners import ACTION_PARTS, action_corners, centered_action_corners
from mojive.ui.workspace.menu_geometry import native_edges, native_unit, popup_padding, popup_radius
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import menu_row_size, paint_menu_row
from mojive.ui.workspace.popup_style import popup_box
from mojive.ui.workspace.rail_geometry import (
    expand_rail_box,
    rail_corners,
    rail_dimensions,
    rail_padding,
)
from mojive.ui.workspace.statistics import NativeStatisticsPresentation
from mojive.ui.workspace.window_chrome import surface_ink

# Every entry describes an actual application operation, not a numbered placeholder.
ACTIONS = {
    "button_row": (("定位", "frame"), ("复制", "copy"), ("删除", "delete")),
    "tabs": (("属性", None), ("物理", None), ("材质", None), ("关节", None)),
    "bottom_header": (
        ("时间轴", None),
        ("曲线", "waveform"),
        ("统计", "performance"),
        ("输出", "terminal"),
    ),
    "type_filters": (("本体 11", "body"), ("几何体 20", "geom"), ("关节 11", "joint")),
    "panel_tools": tuple(
        (view.value.title(), icon)
        for view, icon in zip(GEOMETRY_VIEWS, GEOMETRY_ICONS, strict=True)
    ),
    "menu_bar": (("文件", None), ("编辑", None), ("添加", None), ("视图", None)),
    "title_actions": (("", "inspect"), ("", "settings")),
    "document_title": (("joint_types.xml", "file"),),
    "viewport_toolbar": (("透视", "perspective"), ("", "shading"), ("", "layers")),
    "transport": (
        ("", "previous"),
        ("播放", "play"),
        ("", "right"),
        ("0.000 s", None),
        ("", "reset"),
    ),
    "rail": (("", "scene"), ("", "pose"), ("", "sliders"), ("", "folder"), ("", "camera")),
    "toolbox": (("", "select"), ("", "move"), ("", "rotate"), ("", "scale"), ("", "frame")),
    "output_toolbar": (
        ("信息 3", "info"),
        ("警告 1", "warn"),
        ("错误 0", "err"),
        ("", "search"),
        ("", "copy"),
        ("", "delete"),
    ),
    "timeline_toolbar": (
        ("", "transport-first"),
        ("播放", "play"),
        ("", "transport-last"),
        ("录制", "record"),
        ("快照", "snapshot"),
    ),
    "timeline_details": (
        ("关键帧  0.250 s", "diamond"),
        ("定位", "frame"),
        ("复制", "copy"),
        ("删除", "delete"),
    ),
    "pending_edits": (("2 项修改待应用", None), ("应用", "check"), ("撤销", "reset")),
    "file_receipt": (("scene.xml", "file"), ("打开文件夹", "folder"), ("", "close")),
    "dialog_footer": (("完成", "check"),),
}

# Other action groups are transparent in production; their controls do not gain
# an enclosing rounded plate just because they are shown in the catalogue.
ACTION_SURFACES = frozenset(
    ("rail", "toolbox", "viewport_toolbar", "transport", "pending_edits", "file_receipt")
)
ACTION_PADDING = {
    "toolbox": (4,) * 4,
    "viewport_toolbar": (4,) * 4,
    "transport": (6,) * 4,
    "pending_edits": (8,) * 4,
    "file_receipt": (4,) * 4,
}


def filter_geometry(ui, part, width):
    """Measure the production search field and optional geometry segments."""
    with ui.component_scope("panel_tools", part):
        search = ui.component_style(("input_text", "search"))
        margins = search.edges("margin", ui.s)
        height = (
            ui.design.primitives.field_height(("input_text", "search"), DESIGN.control_height)
            * ui.s
            + margins[1]
            + margins[3]
        )
        style = ui.component_style(("panel_tools", part))
        gap = style.get("rowGap", style.get("gap", 8 if part == "scene_tools" else 10)) * ui.s
        segment_height = 0
        if part == "scene_tools":
            segment_height = ui.segment_height()
        return width, height + gap + segment_height, height, gap


def filters(owner, ui, engine, style, part, bounds):
    with ui.component_scope("panel_tools", part):
        _, _, search_height, gap = filter_geometry(ui, part, bounds.width)
        key = "reference-" + part
        _, owner.values[key] = ui.search(
            key,
            bounds.x,
            bounds.y,
            bounds.width,
            owner.values.get(key, ""),
            hint=ui.translate("Search scene" if part == "scene_tools" else "Search joints"),
        )
        children = [ui.hits[key]]
        if part == "scene_tools":
            segment_key = key + "-geometry"
            owner.values[segment_key] = ui.segmented(
                segment_key,
                bounds.x,
                bounds.y + search_height + gap,
                bounds.width / ui.s,
                tuple(view.value.title() for view in GEOMETRY_VIEWS),
                owner.values.get(segment_key, 0),
                icons=GEOMETRY_ICONS,
            )
            children.append(ui.hits[segment_key])
        return children


def action_layout(ui, part, bounds):
    vertical = part in ("rail", "toolbox")
    style = ui.component_style(part)
    padding = tuple(
        (style.get(field) or 0) * ui.s
        for field in ("paddingLeft", "paddingTop", "paddingRight", "paddingBottom")
    )
    if part in ACTION_PADDING:
        padding = action_corners(ui, part, ACTION_PADDING[part]).padding
    elif part == "rail":
        padding = rail_padding(ui)
    column_gap = style.get("columnGap") if style.has("columnGap") else style.get("gap", 6)
    row_gap = style.get("rowGap") if style.has("rowGap") else style.get("gap", 6)
    items = ACTIONS[part]
    sizes = []
    for index, (label, icon) in enumerate(items):
        slot = (
            rail_dimensions(ui)[0]
            if part == "rail"
            else style.get("slotSize", 36)
            if vertical
            else 28
            if part in ACTION_SURFACES
            else 32
        )
        w = slot if vertical or not label else ui.measure(label) / ui.s + (40 if icon else 24)
        sizes.append(
            control_size(
                ui,
                "reference-" + part + "/" + str(index),
                w,
                slot,
                label=label,
                icon=icon,
                **(
                    {
                        "icon_size": style.get("iconSize", 24 if part == "rail" else 20),
                        "icon_role": "activity" if part == "rail" else "toolbar",
                    }
                    if part in ("rail", "toolbox")
                    else {}
                ),
            )
            if part in ACTION_SURFACES
            else (w * ui.s, slot * ui.s)
        )
        if part == "transport" and index == 3:
            sizes[-1] = (w * ui.s, DESIGN.control_height * ui.s)
    if part == "transport":
        padding = centered_action_corners(
            action_corners(ui, part, ACTION_PADDING[part]), sizes
        ).padding
    box, flow = component_flow(
        ui,
        part,
        bounds.x,
        bounds.y,
        min(bounds.width, rail_dimensions(ui)[1] * ui.s)
        if part == "rail"
        else min(bounds.width, max(v[0] for v in sizes) + padding[0] + padding[2])
        if part == "toolbox"
        else min(
            bounds.width,
            sum(v[0] for v in sizes)
            + max(0, len(sizes) - 1) * column_gap * ui.s
            + padding[0]
            + padding[2],
        ),
        sizes,
        key="reference-" + part,
        padding=padding,
        column_gap=column_gap,
        row_gap=row_gap,
        observe=False,
        resolved_padding=padding if part in ACTION_SURFACES else None,
    )
    if part == "rail":
        box = expand_rail_box(ui, box)
        rectangles = tuple(
            replace(rect, width=min(size[0], box.content.width))
            for rect, size in zip(flow.rectangles, sizes, strict=True)
        )
        flow = replace(flow, rectangles=rectangles, width=max(r.width for r in rectangles))
    return box, flow


def actions(owner, ui, engine, style, part, bounds):
    items = ACTIONS[part]
    if part == "button_row":
        g.set_cursor_screen_pos((bounds.x, bounds.y))
        changed = draw_button_row(
            ui, "reference-actions", [v[0] for v in items], available=bounds.width
        )
        if any(changed):
            owner.values[part] = items[changed.index(True)][0]
        boxes = [ui.hits["reference-actions/" + str(i)] for i in range(len(items))]
        return boxes
    box, flow = action_layout(ui, part, bounds)
    corners = action_corners(ui, part, ACTION_PADDING[part]) if part in ACTION_PADDING else None
    if part == "transport":
        corners = centered_action_corners(
            corners, tuple((r.width, r.height) for r in flow.rectangles)
        )
    radius = rail_corners(ui, box)[0] if part == "rail" else corners.radius if corners else 0
    child_radius = (
        rail_corners(ui, box)[2] if part == "rail" else corners.inner_radius if corners else None
    )
    if part in ACTION_SURFACES:
        ui.rect(*box.frame.tuple(), rgba8("child"), r=radius / ui.s)
    children = []
    scope = (
        corner_radius_scope(child_radius, ACTION_PARTS)
        if child_radius is not None
        else nullcontext()
    )
    with scope:
        for i, ((label, icon), rect) in enumerate(zip(items, flow.rectangles, strict=True)):
            if part == "transport" and i == 3:
                ui.component_text(part, rect, label, rgba8("text"))
                children.append(rect.tuple())
                continue
            if part in ACTION_SURFACES:
                width, height = control_frame_size(
                    ui,
                    "reference-" + part + "/" + str(i),
                    rect.width,
                    rect.height,
                    label=label,
                    icon=icon,
                )
                rect = replace(rect, width=width * ui.s, height=height * ui.s)
            result = engine.button(
                "reference-" + part + "/" + str(i),
                rect,
                label=label,
                icon=icon,
                parts="icon_button"
                if part in ACTION_SURFACES and not label
                else "tab"
                if part in ("tabs", "bottom_header")
                else "button",
                active=owner.values.get(part, 0) == i,
                filled=True,
                radius=None,
                **(
                    {
                        "icon_size": style.get("iconSize", 24 if part == "rail" else 20),
                        "icon_role": "activity" if part == "rail" else "toolbar",
                    }
                    if part in ("rail", "toolbox")
                    else {}
                ),
            )
            if result.changed:
                owner.values[part] = i
            children.append(result.box.frame.tuple())
    return children


def choice_rows(owner, part):
    icon = "scene" if owner.values.get(part + "/icon", False) else None
    if part in ("menu", "menu_item", "menu_item_content"):
        entries = (
            Item(label="打开场景", shortcut="Ctrl+O", icon="folder"),
            Item(
                label="带有完整名称的组件操作按钮"
                if part == "menu_item" and owner.long_text
                else "保存场景",
                shortcut="Ctrl+S",
                icon="save",
            ),
            Item(label="撤销", shortcut="Ctrl+Z", icon="undo", enabled=False),
        )
        return entries[1:2] if part == "menu_item" else entries
    entries = ("简体中文", "English", "日本語") if part == "select_popup" else ("简体中文",)
    return tuple(
        Item(
            label=label,
            checked=owner.values.get(part, entries[0]) == label
            if part == "select_popup"
            else bool(owner.values.get(part)),
            icon=icon,
        )
        for label in entries
    )


def choice_size(owner, ui, part, width):
    container = part not in ("select_option", "menu_item")
    row_part = "select_option" if part.startswith("select_") else "menu_item"
    pad = popup_padding(ui, part) if container else (0,) * 4
    with (
        ui.corner_scope(max(0, popup_radius(ui, part, pad) - pad[0]), row_part)
        if part in ("menu", "select_popup")
        else nullcontext()
    ):
        sizes = [menu_row_size(ui, row, part=row_part) for row in choice_rows(owner, part)]
    height = sum(h for _w, h in sizes)
    if container:
        style = ui.component_style(part)
        margin = native_edges(style, "margin", ui.s)
        height += max(0, len(sizes) - 1) * native_unit(style.get("rowGap", 6) * ui.s)
        height += pad[1] + pad[3] + margin[1] + margin[3]
        width = min(width, style.dimension("width", width / ui.s) * ui.s)
        height = style.dimension("height", height / ui.s) * ui.s
    return width, height


def choices(owner, ui, engine, style, part, bounds):
    popup = part not in ("select_option", "menu_item")
    row_part = "select_option" if part.startswith("select_") else "menu_item"
    box = popup_box(ui, bounds, part=part) if popup else None
    if box is not None:
        ui.rect(
            *box.frame.tuple(),
            surface_ink(ui, part, "popup"),
            r=popup_radius(ui, part, box.padding) / ui.s,
            color_resolved=True,
        )
    content = box.content if box is not None else bounds
    y, boxes = content.y, []
    g.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
    try:
        with (
            ui.corner_scope(max(0, popup_radius(ui, part, box.padding) - box.padding[0]), row_part)
            if part in ("menu", "select_popup")
            else nullcontext()
        ):
            for index, row in enumerate(choice_rows(owner, part)):
                g.set_cursor_screen_pos((content.x, y))
                result = paint_menu_row(
                    ui,
                    row,
                    "reference-choice/" + str(index),
                    content_right=content.right,
                    part=row_part,
                )
                if result.activated:
                    owner.values[part] = row.label if popup else not row.checked
                if not popup:
                    ui.hits["reference-" + part] = result.box.frame.tuple()
                boxes.append(result.box.frame.tuple())
                y = (
                    result.box.outer.bottom + native_unit(style.get("rowGap", 6) * ui.s)
                    if popup
                    else y
                )
    finally:
        g.pop_clip_rect()
    if box is not None:
        ui.record_component_box(part, box, key="reference-" + part, fields=boxes)
    return box if box is not None else result.box, boxes


def status_readout_geometry(ui, part):
    """Measure one reference using the complete production telemetry layout."""
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.viewport_widgets.status_metrics import _status_performance_layout
    from mojive.ui.workspace.status_components import StatusGroups

    groups = StatusGroups(ui)
    draw = ImguiDraw2D(ui.draw)
    text = groups.text_draw(draw, "status_metrics")
    mode = "steps" if part == "status_steps" else "time"
    styles = groups.metric_styles(draw, mode)
    layout = _status_performance_layout(
        text,
        0,
        ui.s,
        "OpenGL",
        0.002,
        59.8,
        metric_text="Steps 130" if mode == "steps" else "Time 0.26 s",
        metric_mode=mode,
        step=130,
        sim_time=0.26,
        show_physics=True,
        physics_hz=120,
        text_budget=groups.text_budget,
        metric_styles=styles,
        group_gap=groups.gap("status_metrics"),
        **groups.metric_options(),
    )
    key = (
        "metric"
        if part in ("status_steps", "status_time")
        else component_catalog()[part]["statusKey"]
    )
    field = next(field for field in layout.fields if field.key == key)
    return field, styles[key].draw if key in styles else text, groups


def status(owner, ui, engine, style, part, bounds):
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.theme import THEME
    from mojive.ui.viewport_widgets.hints import draw_tool_hints, tool_hints_height
    from mojive.ui.viewport_widgets.model import ToolHint
    from mojive.ui.viewport_widgets.status import draw_status
    from mojive.ui.workspace.status_components import (
        StatusBadge,
        StatusGroups,
        resolve_status_bar_layout,
    )

    if "statusKey" in component_catalog()[part]:
        from mojive.ui.viewport_widgets.status_metrics import _StatusNumbers

        field, draw, groups = status_readout_geometry(ui, part)
        separator_width = groups.metric_options()["separator_width"]
        leading_width = separator_width if field.key != "backend" else 0
        offset = bounds.x + leading_width + field.left_gap - field.x
        field = replace(
            field,
            x=field.x + offset,
            content_x=field.content_x + offset,
            value_x=field.value_x + offset,
            unit_x=field.unit_x + offset,
        )
        center = bounds.y + bounds.height / 2
        _StatusNumbers(draw).paint(field, center, THEME.text_disabled)
        if field.key != "backend":
            groups.divider(draw, bounds.x + separator_width / 2, center, bounds.height, THEME)
        if field.key != "render":
            groups.divider(
                draw,
                field.x + field.width + field.right_gap + separator_width / 2,
                center,
                bounds.height,
                THEME,
            )
        groups.record_readout(
            field,
            (field.x, bounds.y, field.width, bounds.height),
            "steps" if part == "status_steps" else "time",
        )
        return [(field.x, bounds.y, field.width, bounds.height)]
    groups, badge = StatusGroups(ui), StatusBadge(ui, keycap=True)
    draw = ImguiDraw2D(ui.draw)
    center = bounds.y + bounds.height / 2
    if part == "status_separator":
        groups.divider(draw, bounds.x + bounds.width / 2, center, bounds.height, THEME)
        return []
    if part == "status_hints":
        hints = (ToolHint("key", "Esc", "取消选择"), ToolHint("key", "Shift", "吸附"))
        edges, fields = groups.edges(part), []
        height = tool_hints_height(draw, ui.s, hints, keycap=badge)
        width = draw_tool_hints(
            draw,
            (bounds.x + edges[0], center + (edges[1] - edges[3]) / 2),
            THEME,
            ui.s,
            hints,
            keycap=badge,
            group_gap=groups.gap(part),
            record_item=fields.append,
            separator_draw=lambda x: groups.divider(draw, x, center, bounds.height, THEME),
        )
        groups.record(
            part,
            (
                bounds.x,
                center - (height + edges[1] + edges[3]) / 2,
                width + edges[0] + edges[2],
                height + edges[1] + edges[3],
            ),
            fields,
        )
        return fields
    surface = None
    if part == "statusbar":
        box = resolve_status_bar_layout(ui, badge=badge, groups=groups).box(
            bounds.x, bounds.y, bounds.width
        )
        bounds, surface = box.content, box.frame.tuple()
    draw_status(
        draw,
        (bounds.x, bounds.y),
        bounds.width,
        bounds.height,
        THEME,
        ui.s,
        selected="",
        state="paused" if part in ("statusbar", "status_context") else "external",
        sim_time=0.26,
        step=130,
        metric_mode="steps" if part == "status_steps" else "time",
        show_physics=True,
        backend="OpenGL" if part != "status_context" else "",
        show_metrics=part != "status_context",
        dt=0.002,
        fps=60,
        presentation=groups,
        surface_rect=surface,
        keycap=badge,
        tool_hints=(ToolHint("key", "Esc", "取消选择"), ToolHint("key", "Shift", "吸附"))
        if part == "status_context"
        else (),
    )
    return groups.fields


def native_chrome(owner, ui, engine, style, part, bounds):
    from mojive.ui.workspace.window_chrome import native_chrome as chrome_scope

    g.set_cursor_screen_pos((bounds.x, bounds.y))
    boxes = []
    with chrome_scope(ui, part):
        if part == "window_title":
            g.text_unformatted(ui.translate("Help"))
            lo, hi = g.get_item_rect_min(), g.get_item_rect_max()
            return [(lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)]
        if g.begin_tab_bar("reference-dock-tabs"):
            try:
                for label in ("Timeline", "Output", "Statistics", "Plot"):
                    selected, _ = g.begin_tab_item(ui.translate(label))
                    lo, hi = g.get_item_rect_min(), g.get_item_rect_max()
                    boxes.append((lo.x, lo.y, hi.x - lo.x, hi.y - lo.y))
                    if selected:
                        g.end_tab_item()
            finally:
                g.end_tab_bar()
    return boxes


def data(owner, ui, engine, style, part, bounds):
    g.set_cursor_screen_pos((bounds.x, bounds.y))
    presenter = NativeStatisticsPresentation(SimpleNamespace(ui=ui, s=ui.s))
    presenter._cursor = g.ImVec2(bounds.x, bounds.y)
    # A child fixes the same width budget consumed by the production table painter.
    g.push_style_var(g.StyleVar_.window_padding, (0, 0))
    g.begin_child(
        "reference-data", (bounds.width, bounds.height), window_flags=g.WindowFlags_.no_background
    )
    g.pop_style_var()
    try:
        presenter._cursor = g.get_cursor_screen_pos()
        if part == "table":
            presenter._table(
                "reference-pass-table",
                ("渲染通道", "CPU (ms)", "GPU (ms)"),
                (
                    ("Opaque", "1.240", "0.860"),
                    ("Shadow", "0.370", "0.210"),
                    ("Overlay", "0.180", "0.120"),
                ),
            )
        else:
            rows = (
                ("绘制调用", "126"),
                ("实例数量", "52"),
                ("三角形", "42,816"),
                ("帧耗时", "8.25 ms"),
            )
            presenter._pairs("reference-statistics", rows[:1] if part == "stat_row" else rows)
    finally:
        g.end_child()
    return [bounds.tuple()]


def material(owner, ui, engine, style, part, bounds):
    from mojive.ui.workspace.material_library import draw_material_library

    workspace = SimpleNamespace(
        ui=ui, s=ui.s, material=owner.values.get("material", 0), app=SimpleNamespace(entity=None)
    )
    draw_material_library(workspace, bounds.x, bounds.y, bounds.width)
    owner.values["material"] = workspace.material
    return [hit for key, hit in ui.hits.items() if key.startswith("material-")]


def tree(owner, ui, engine, style, part, bounds):
    s = ui.s
    gap = style.get("rowGap", 4) * s
    height = max(28, style.get("height", 28)) * s
    rows = [
        ("world", "scene", 0),
        ("free_body", "body", 1),
        ("free_box", "geom", 2),
        ("hinge_limited", "joint", 1),
    ]
    children = []
    for index, (label, icon, depth) in enumerate(rows):
        if index > 0 and not owner.values.get("tree/open", True):
            break
        row = Rect(bounds.x, bounds.y + index * (height + gap), bounds.width, height)
        hovered = g.is_mouse_hovering_rect((row.x, row.y), (row.right, row.bottom))
        selected = index in owner.values.get("tree/selected", {1})
        if selected or hovered:
            ui.rect(*row.tuple(), rgba8("selected" if selected else "field_hovered"), r=6)
        x = row.x + depth * style.get("indent", 14) * s
        if index == 0:
            result = engine.button(
                "reference-tree-expand",
                Rect(x, row.y, 28 * s, height),
                icon="down" if owner.values.get("tree/open", True) else "right",
                parts="tree_expander",
            )
            if result.changed:
                owner.values["tree/open"] = not owner.values.get("tree/open", True)
        name = Rect(x + 28 * s, row.y, max(1, row.right - x - 60 * s), height)
        result = engine.button(
            "reference-tree-row/" + str(index),
            name,
            label=label,
            icon=icon,
            parts="tree_label",
            active=selected,
        )
        if result.changed:
            current = owner.values.setdefault("tree/selected", {1})
            if g.get_io().key_shift:
                owner.values["tree/selected"] = set(
                    range(min(current | {index}), max(current | {index}) + 1)
                )
            else:
                owner.values["tree/selected"] = {index}
        visible = owner.values.get("tree/visible/" + str(index), True)
        if hovered or not visible:
            result = engine.button(
                "reference-tree-eye/" + str(index),
                Rect(row.right - 28 * s, row.y, 28 * s, height),
                icon="eye" if visible else "eyeoff",
                parts="icon_button",
                filled=True,
            )
            if result.changed:
                owner.values["tree/visible/" + str(index)] = not visible
        children.append(row.tuple())
    return children


def rows(owner, ui, engine, style, part, bounds):
    if part in ("component_list", "component_path"):
        from mojive.ui.workspace.model_components import component_actions

        entries = owner.values.setdefault(
            part + "/rows",
            ["hinge_pos (position)", "slide_pos (position)"]
            if part == "component_list"
            else ["1. site · attachment", "2. geom · hinge_anchor"],
        )
        g.set_cursor_screen_pos((bounds.x, bounds.y))
        g.begin_child("reference-model-components", (bounds.width, bounds.height))
        try:
            for index, label in enumerate(tuple(entries)):
                labels = ("编辑", "删除") if part == "component_list" else ("上移", "下移", "删除")
                enabled = (
                    (True, True)
                    if part == "component_list"
                    else (index > 0, index < len(entries) - 1, True)
                )
                result = component_actions(
                    ui, part, "reference-model/" + str(index), label, labels, enabled
                )
                if result[-1]:
                    entries.remove(label)
                elif result[0] and part == "component_list":
                    owner.values[part + "/editing"] = index
                elif part == "component_path" and (result[0] or result[1]):
                    other = index - 1 if result[0] else index + 1
                    entries[index], entries[other] = entries[other], entries[index]
            edit = owner.values.get(part + "/editing")
            if edit is not None and edit < len(entries):
                g.set_next_item_width(-1)
                _, entries[edit] = g.input_text("##component-name", entries[edit])
        finally:
            g.end_child()
        return [bounds.tuple()]
    if part == "sensor":
        import numpy as np

        from mojive.adapters.base import SensorInfo
        from mojive.ui.workspace.sensors_panel import WorkspaceSensors

        g.set_cursor_screen_pos((bounds.x, bounds.y))
        g.begin_child("reference-sensor", (bounds.width, bounds.height))
        try:
            if "sensor-preview" not in owner.objects:
                panel = WorkspaceSensors(SimpleNamespace(ui=ui, s=ui.s))
                ctx = SimpleNamespace(
                    tr=ui.translate,
                    panels=None,
                    session=SimpleNamespace(
                        sensor_infos=(
                            SensorInfo(0, "joint_position", "jointpos", 0, 1),
                            SensorInfo(1, "position_xyz", "framepos", 1, 3),
                            SensorInfo(2, "component_readings", "user", 4, 14),
                        ),
                        frame=SimpleNamespace(sensors=np.arange(18, dtype=float) * 0.125),
                    ),
                )
                owner.objects["sensor-preview"] = panel, ctx
            panel, ctx = owner.objects["sensor-preview"]
            panel.draw(ctx)
        finally:
            g.end_child()
        return [bounds.tuple()]
    from mojive.ui.messages import OutputBuffer
    from mojive.ui.theme import THEME
    from mojive.ui.workspace.menus import Menus
    from mojive.ui.workspace.output_panel import WorkspaceOutput

    key = "output-sample/" + part
    if key not in owner.objects:
        workspace = SimpleNamespace(ui=ui, s=ui.s)
        workspace.menus = Menus(workspace, ())
        output = OutputBuffer()
        for stamp, message, level in (
            ("14:18:41", "已载入 joint_types.xml", "info"),
            ("14:18:45", "已清除选择", "info"),
            ("14:19:24", "未选择录制片段", "warning"),
        ):
            output.write(message, timestamp=stamp, level=level)
        owner.objects[key] = WorkspaceOutput(workspace), output
    presenter, output = owner.objects[key]
    ctx = SimpleNamespace(
        output=output,
        style_scale=ui.s,
        language="zh_CN",
        tr=ui.translate,
        theme=THEME,
        input_bindings=None,
        input_claim=None,
    )
    g.set_cursor_screen_pos((bounds.x, bounds.y))
    g.begin_child("reference-output", (bounds.width, bounds.height))
    try:
        if part == "output_layout":
            presenter.draw(ctx)
        else:
            presenter._draw_entries(ctx, output.entries(), False)
    finally:
        g.end_child()
    return presenter._row_rectangles


GROUP_RENDERERS = {
    **dict.fromkeys(("window_title", "dock_tabs"), native_chrome),
    **dict.fromkeys(
        (
            "status_backend",
            "status_steps",
            "status_time",
            "status_delta",
            "status_physics",
            "status_render",
        ),
        status,
    ),
    **dict.fromkeys(("scene_tools", "pose_tools"), filters),
    **dict.fromkeys(
        ("statusbar", "status_context", "status_hints", "status_metrics", "status_separator"),
        status,
    ),
    **dict.fromkeys(ACTIONS, actions),
    **dict.fromkeys(("stat_row", "data_list", "table", "statistics_list"), data),
    **dict.fromkeys(("material", "material_list"), material),
    **dict.fromkeys(
        (
            "sensor",
            "component_list",
            "component_path",
            "output_row",
            "output_list",
            "output_layout",
        ),
        rows,
    ),
    "tree_row": tree,
}
