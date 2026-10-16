"""Concrete group specimens composed from the production native controls."""

from types import SimpleNamespace

from imgui_bundle import imgui as g

from mojive.design.native.geometry import Rect
from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.button_rows import draw_button_row
from mojive.ui.workspace.component_layout import component_flow
from mojive.ui.workspace.menu_item import Item
from mojive.ui.workspace.menu_rows import menu_row_size, paint_menu_row
from mojive.ui.workspace.popup_style import popup_box
from mojive.ui.workspace.statistics import NativeStatisticsPresentation

# Every entry describes an actual application operation, not a numbered placeholder.
ACTIONS = {
    "button_row": (("定位", "frame"), ("复制", "copy"), ("删除", "delete")),
    "tabs": (("属性", None), ("物理", None), ("材质", None), ("关节", None)),
    "bottom_header": (
        ("时间轴", "chart"),
        ("曲线", "chart"),
        ("统计", "chart"),
        ("输出", "terminal"),
    ),
    "type_filters": (("本体 11", "body"), ("几何体 20", "geom"), ("关节 11", "joint")),
    "panel_tools": (("外观", "visual"), ("碰撞", "collision"), ("叠加", "both")),
    "menu_bar": (("文件", None), ("编辑", None), ("添加", None), ("视图", None)),
    "title_actions": (("", "inspect"), ("", "settings")),
    "document_title": (("joint_types.xml", "file"),),
    "viewport_toolbar": (("透视", "camera"), ("", "shading"), ("", "layers")),
    "transport": (
        ("", "previous"),
        ("播放", "play"),
        ("", "right"),
        ("0.000 s", None),
        ("", "reset"),
    ),
    "rail": (("", "scene"), ("", "joint"), ("", "sliders"), ("", "folder"), ("", "camera")),
    "toolbox": (("", "select"), ("", "move"), ("", "rotate"), ("", "scale"), ("", "cube")),
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
        ("快照", "camera"),
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


def action_layout(ui, part, bounds):
    vertical = part in ("rail", "toolbox")
    items = ACTIONS[part]
    sizes = []
    for label, icon in items:
        w = 32 if vertical or not label else ui.measure(label) / ui.s + (40 if icon else 24)
        sizes.append((w * ui.s, 32 * ui.s))
    box, flow = component_flow(
        ui,
        part,
        bounds.x,
        bounds.y,
        min(bounds.width, 48 * ui.s)
        if vertical
        else min(
            bounds.width,
            sum(v[0] for v in sizes)
            + max(0, len(sizes) - 1) * ui.metric(part, "columnGap", 6) * ui.s
            + sum(ui.component_style(part).edges("padding", ui.s)[::2]),
        ),
        sizes,
        key="reference-" + part,
        column_gap=6,
        row_gap=6,
        observe=False,
    )
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
    if part not in ("menu_bar", "document_title", "status_hints", "button_row"):
        ui.rect(*box.frame.tuple(), rgba8("child"), r=style.get("radius", 8))
    children = []
    for i, ((label, icon), rect) in enumerate(zip(items, flow.rectangles, strict=True)):
        result = engine.button(
            "reference-" + part + "/" + str(i),
            rect,
            label=label,
            icon=icon,
            parts="tab" if part in ("tabs", "bottom_header") else "button",
            active=owner.values.get(part, 0) == i,
            filled=True,
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
                icon="file",
            ),
            Item(label="撤销", shortcut="Ctrl+Z", icon="reset", enabled=False),
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
    sizes = [menu_row_size(ui, row, part=row_part) for row in choice_rows(owner, part)]
    height = sum(h for _w, h in sizes)
    if container:
        style = ui.component_style(part)
        pad, margin = style.edges("padding", ui.s, (12, 6, 12, 6)), style.edges("margin", ui.s)
        height += max(0, len(sizes) - 1) * style.get("rowGap", 6) * ui.s
        height += pad[1] + pad[3] + margin[1] + margin[3]
        width = min(width, style.dimension("width", width / ui.s) * ui.s)
        height = style.dimension("height", height / ui.s) * ui.s
    return width, height


def choices(owner, ui, engine, style, part, bounds):
    popup = part not in ("select_option", "menu_item")
    row_part = "select_option" if part.startswith("select_") else "menu_item"
    box = popup_box(ui, bounds, part=part) if popup else None
    if box is not None:
        ui.rect(*box.frame.tuple(), rgba8("popup"), r=style.get("radius", 8))
    content = box.content if box is not None else bounds
    y, boxes = content.y, []
    g.push_clip_rect((content.x, content.y), (content.right, content.bottom), True)
    try:
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
            y = result.box.outer.bottom + style.get("rowGap", 6) * ui.s if popup else y
    finally:
        g.pop_clip_rect()
    if box is not None:
        ui.record_component_box(part, box, key="reference-" + part, fields=boxes)
    return box if box is not None else result.box, boxes


def status(owner, ui, engine, style, part, bounds):
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.theme import THEME
    from mojive.ui.viewport_widgets.hints import draw_tool_hints, tool_hints_height
    from mojive.ui.viewport_widgets.model import ToolHint
    from mojive.ui.viewport_widgets.status import draw_status
    from mojive.ui.workspace.status_components import StatusBadge, StatusGroups

    groups, badge = StatusGroups(ui), StatusBadge(ui)
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
    if part == "statusbar":
        bounds = style.box(*bounds.tuple(), scale=ui.s).content
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
        metric_mode="time",
        backend="OpenGL" if part != "status_context" else "",
        show_metrics=part != "status_context",
        dt=0.002,
        fps=60,
        presentation=groups,
        keycap=badge,
        tool_hints=(ToolHint("key", "Esc", "取消选择"), ToolHint("key", "Shift", "吸附"))
        if part == "status_context"
        else (),
    )
    return groups.fields


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
        from mojive.ui.workspace.sensors_panel import SensorPresentation

        g.set_cursor_screen_pos((bounds.x, bounds.y))
        g.begin_child("reference-sensor", (bounds.width, bounds.height), g.ChildFlags_.borders)
        try:
            presenter = SensorPresentation(ui, prefix="reference-sensor")
            if presenter.begin_table("reference-sensor-values"):
                for label, value in (
                    ("名称", "joint_position"),
                    ("类型", "jointpos"),
                    ("维数", "1"),
                    ("当前值", "0.785 rad"),
                ):
                    presenter.labeled(label, value)
                presenter.end_table()
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
