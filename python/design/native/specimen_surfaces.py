"""Native panel, timeline and viewport specimens with editable application content."""

import math

from imgui_bundle import imgui as g

from mojive.design.native.geometry import Rect
from mojive.design.native.specimen_groups import actions, tree
from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.timeline_header import model_header_layout
from mojive.ui.workspace.widgets import color
from mojive.ui.workspace.window_chrome import surface_ink


def heading(owner, ui, engine, style, part, bounds):
    labels = {
        "panel_header": "场景",
        "inspector_header": "hinge_body",
        "dialog_header": "设置",
        "camera_header": "Camera · joints",
        "titlebar": "Mojive · joint_types.xml",
    }
    box = style.box(*bounds.tuple(), scale=ui.s)
    c, s = box.content, ui.s
    result = engine.button(
        "reference-heading",
        Rect(c.x, c.y, max(1, c.width - 72 * s), 32 * s),
        label=labels[part],
        icon="camera" if part == "camera_header" else "body",
        parts="tree_label",
    )
    boxes = [result.box.frame.tuple()]
    for i, icon in enumerate(("pin", "close")):
        result = engine.button(
            "reference-heading/" + icon,
            Rect(c.right - (2 - i) * 32 * s, c.y, 28 * s, 28 * s),
            icon=icon,
            parts="icon_button",
        )
        if result.changed:
            owner.values[part + "/" + icon] = not owner.values.get(part + "/" + icon, False)
        boxes.append(result.box.frame.tuple())
    ui.line(c.x, c.y + 40 * s, c.width)
    return boxes


def form(owner, ui, engine, style, part, bounds):
    c, s = style.box(*bounds.tuple(), scale=ui.s).content, ui.s
    labels = (
        ("语言", "滚动条隐藏延迟", "显示检查器")
        if part.startswith(("settings", "dialog"))
        else ("偏航", "俯仰", "距离")
    )
    children = []
    for index, label in enumerate(labels):
        rect = Rect(c.x, c.y + index * 48 * s, c.width, 36 * s)
        layout = engine.form_layout(rect)
        engine.label("reference-form-label/" + str(index), layout["label"], label)
        field = layout["fields"][0]
        if label == "语言":
            r = engine.select(
                "reference-language",
                field,
                owner.values.get("language", "简体中文"),
                ("简体中文", "English"),
            )
            owner.values["language"] = r.value
        elif label == "显示检查器":
            r = engine.switch("reference-inspector", field, owner.values.get("inspector", True))
            owner.values["inspector"] = r.value
        else:
            key = part + "/" + label
            r = engine.number_shell(
                "reference-form/" + label,
                field,
                owner.values.get(key, 1.0),
                unit="s" if "延迟" in label else "m" if label == "距离" else "°",
            )
            owner.values[key] = r.value
        children.extend([layout["label"].tuple(), r.box.frame.tuple()])
    return children


def panels(owner, ui, engine, style, part, bounds):
    s = ui.s
    ui.rect(*bounds.tuple(), surface_ink(ui, part, "background"), r=8, color_resolved=True)
    inner = bounds.inset(16 * s, 16 * s, 16 * s, 16 * s)
    if part in ("panel_body", "dialog_body", "settings_page"):
        return form(owner, ui, engine, style, part, inner)
    if part == "settings_nav":
        boxes = []
        for i, (label, icon) in enumerate(
            (
                ("常规", "general"),
                ("相机", "camera"),
                ("交互", "mouse"),
                ("渲染", "shading"),
                ("录制", "video"),
            )
        ):
            r = engine.button(
                "reference-settings-nav/" + str(i),
                Rect(inner.x, inner.y + i * 40 * s, inner.width, 32 * s),
                label=label,
                icon=icon,
                active=owner.values.get(part, 0) == i,
                parts="tab",
                filled=True,
            )
            if r.changed:
                owner.values[part] = i
            boxes.append(r.box.frame.tuple())
        return boxes
    if part in ("settings_layout", "dialog"):
        children = heading(
            owner,
            ui,
            engine,
            ui.component_style("dialog_header"),
            "dialog_header",
            Rect(inner.x, inner.y, inner.width, 48 * s),
        )
        requested = ui.metric(
            "settings_nav", "navWidth", ui.metric("settings_layout", "navWidth", 0)
        )
        nav_w = min(requested * s or 126 * s, inner.width / 2)
        y = inner.y + 52 * s
        children += panels(
            owner,
            ui,
            engine,
            ui.component_style("settings_nav"),
            "settings_nav",
            Rect(inner.x, y, nav_w, inner.height - 52 * s),
        )
        children += form(
            owner,
            ui,
            engine,
            ui.component_style("settings_page"),
            "settings_page",
            Rect(inner.x + nav_w + 16 * s, y, inner.width - nav_w - 16 * s, inner.height - 52 * s),
        )
        return children
    children = heading(
        owner,
        ui,
        engine,
        ui.component_style("panel_header"),
        "panel_header",
        Rect(inner.x, inner.y, inner.width, 48 * s),
    )
    children += form(
        owner,
        ui,
        engine,
        style,
        part,
        Rect(inner.x, inner.y + 56 * s, inner.width, inner.height - 56 * s),
    )
    return children


def timeline(owner, ui, engine, style, part, bounds):
    if part == "timeline_body":
        from mojive.ui.workspace.timeline_body import body_corner_scope

        with body_corner_scope(ui):
            return _timeline(owner, ui, engine, style, part, bounds)
    return _timeline(owner, ui, engine, style, part, bounds)


def _timeline(owner, ui, engine, style, part, bounds):
    s = ui.s
    box = style.box(*bounds.tuple(), scale=s)
    c = box.content
    if part == "timeline_body":
        from mojive.ui.workspace.timeline_body import body_geometry

        geometry = body_geometry(ui)
        c = box.frame.inset(*geometry.padding, *geometry.padding)
        ui.rect(
            *box.frame.tuple(),
            surface_ink(ui, "timeline_body", "background"),
            r=geometry.radius / s,
            color_resolved=True,
        )
    label_width = min(ui.metric("timeline_sheet", "labelWidth", 154) * s, c.width * 0.4)
    header, model = model_header_layout(ui, c.x, c.y, label_width, observe=True)
    header_height = max(header.outer.height, ui.metric("timeline_ruler", "height", 28) * s)
    header_gap = ui.metric("timeline_sheet", "headerGap", 8) * s
    row_height = ui.metric("timeline_lane", "height", 36) * s
    rows = []
    if part in ("timeline_body", "timeline_model_group", "timeline_sheet", "timeline_ruler"):
        r = engine.select(
            "reference-model",
            model.rectangles[0],
            owner.values.get("model", "joint_types"),
            ("joint_types", "scene_002"),
        )
        owner.values["model"] = r.value
        r = engine.button(
            "reference-add-key", model.rectangles[1], icon="plus", parts="icon_button"
        )
        if r.changed:
            owner.values["key-count"] = owner.values.get("key-count", 3) + 1
        rows.extend(rect.tuple() for rect in model.rectangles)
    left = c.x + label_width + 10 * s
    right = c.right
    line_width = max(1, right - left)
    start = c.y + header_height + header_gap
    if part == "timeline_model_group":
        return rows
    count = 1 if part in ("timeline_lane", "lane_label") else 3
    for index, label in enumerate(("关键帧", "片段", "快照")[:count]):
        y = start + index * (row_height + ui.metric("timeline_sheet", "rowGap", 2) * s)
        r = engine.button(
            "reference-lane/" + str(index),
            Rect(c.x, y, label_width, row_height),
            label=label,
            icon=("diamond", "circle", "snap")[index],
            parts="lane_label",
            active=owner.values.get("lane", 0) == index,
            filled=True,
        )
        if r.changed:
            owner.values["lane"] = index
        rows.append(r.box.frame.tuple())
        ui.line(left, y + row_height, right - left)
    if part != "lane_label":
        for i in range(6):
            x = left + i * line_width / 5
            ui.text(
                x + 3 * s,
                c.y + (header_height - 14 * s) / 2,
                f"{i / 10:.1f}",
                rgba8("secondary"),
                12,
                mono=True,
                translate=False,
            )
            ui.draw.add_line(
                (x, c.y + header_height), (x, start + count * row_height), color(rgba8("border")), s
            )
        for i in range(owner.values.get("key-count", 3)):
            x = left + line_width * (i + 1) / (owner.values.get("key-count", 3) + 1)
            r = engine.button(
                "reference-key/" + str(i),
                Rect(x - 10 * s, start + 8 * s, 20 * s, 20 * s),
                icon="diamond",
                parts="timeline_key",
                active=owner.values.get("key", 0) == i,
            )
            if r.changed:
                owner.values["key"] = i
            rows.append(r.box.frame.tuple())
        mouse = g.get_io().mouse_pos
        region = Rect(left, c.y, line_width, header_height)
        g.set_cursor_screen_pos((region.x, region.y))
        g.invisible_button("reference-ruler", (region.width, region.height))
        ui.hits["reference-ruler"] = region.tuple()
        if g.is_item_active():
            owner.values["time"] = min(0.5, max(0, (mouse.x - left) / line_width * 0.5))
        x = left + line_width * owner.values.get("time", 0.25) / 0.5
        ui.draw.add_line((x, c.y), (x, start + count * row_height), color(rgba8("danger")), s)
    return rows


def scene(owner, ui, bounds):
    """Paint an orbitable scene diagram inside the same viewport boundary."""
    if getattr(owner, "scene_drawer", None) is not None:
        owner.scene_drawer(owner, ui, bounds)
        return
    s = ui.s
    ui.rect(*bounds.tuple(), (41, 48, 58, 255), r=0)
    ui.draw.push_clip_rect((bounds.x, bounds.y), (bounds.right, bounds.bottom), True)
    try:
        cx, cy = bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.60
        k = min(bounds.width, bounds.height) * 0.19
        yaw = owner.values.get("scene/yaw", -0.55)

        def project(x, y, z):
            a, b = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
            return cx + a * k, cy + b * k * 0.42 - z * k

        for a in range(-8, 9):
            ui.draw.add_line(project(a, -8, 0), project(a, 8, 0), color((72, 82, 96, 255)), s)
            ui.draw.add_line(project(-8, a, 0), project(8, a, 0), color((72, 82, 96, 255)), s)
        for offset, tint in ((-1.2, (224, 137, 72, 255)), (1.1, (121, 168, 202, 255))):
            points = [
                project(x + offset, y, z)
                for x, y, z in (
                    (-0.45, -0.45, 0),
                    (0.45, -0.45, 0),
                    (0.45, 0.45, 0),
                    (-0.45, 0.45, 0),
                    (-0.45, -0.45, 1),
                    (0.45, -0.45, 1),
                    (0.45, 0.45, 1),
                    (-0.45, 0.45, 1),
                )
            ]
            for indices, factor in (((0, 1, 5, 4), 0.85), ((1, 2, 6, 5), 0.65), ((4, 5, 6, 7), 1)):
                ui.draw.add_quad_filled(
                    *(points[i] for i in indices),
                    color((*tuple(round(v * factor) for v in tint[:3]), 255)),
                )
        g.set_cursor_screen_pos((bounds.x, bounds.y))
        g.invisible_button("reference-scene", (bounds.width, bounds.height))
        if g.is_item_active():
            owner.values["scene/yaw"] = yaw + g.get_io().mouse_delta.x * 0.01
        ui.hits["reference-scene"] = bounds.tuple()
    finally:
        ui.draw.pop_clip_rect()


def viewport(owner, ui, engine, style, part, bounds):
    s = ui.s
    if part == "nav_widget":
        from mojive.ui.camera import OrbitCamera
        from mojive.ui.imgui_draw import ImguiDraw2D
        from mojive.ui.viewcube import ViewCube

        cube = owner.objects.setdefault("viewcube", ViewCube(1.2))
        cam = owner.objects.setdefault("camera", OrbitCamera())
        cube.update(cam.view(), bounds.tuple(), tuple(g.get_io().mouse_pos), s)
        cube.draw(ImguiDraw2D(ui.draw), s)
        return [bounds.tuple()]
    if part in ("camera_preview", "camera_body"):
        if part == "camera_preview":
            heading(
                owner,
                ui,
                engine,
                ui.component_style("camera_header"),
                "camera_header",
                Rect(bounds.x, bounds.y, bounds.width, 44 * s),
            )
            image = bounds.inset(6 * s, 48 * s, 6 * s, 6 * s)
        else:
            image = bounds
        scene(owner, ui, image)
        return [bounds.tuple()]
    ui.rect(*bounds.tuple(), rgba8("child"), r=8)
    top = Rect(bounds.x + 8 * s, bounds.y + 8 * s, bounds.width - 16 * s, 36 * s)
    actions(owner, ui, engine, ui.component_style("menu_bar"), "menu_bar", top)
    left = Rect(
        bounds.x + 12 * s,
        bounds.y + 56 * s,
        min(160 * s, bounds.width * 0.3),
        bounds.height - 72 * s,
    )
    tree(owner, ui, engine, ui.component_style("tree_row"), "tree_row", left)
    right = Rect(
        left.right + 12 * s, left.y, max(1, bounds.right - left.right - 24 * s), left.height
    )
    scene(owner, ui, right)
    actions(
        owner,
        ui,
        engine,
        ui.component_style("viewport_toolbar"),
        "viewport_toolbar",
        Rect(right.x + 8 * s, right.y + 8 * s, right.width - 16 * s, 36 * s),
    )
    return [bounds.tuple()]


def small(owner, ui, engine, style, part, bounds):
    s = ui.s
    if part == "search":
        _changed, value = ui.search(
            "reference-search",
            bounds.x,
            bounds.y,
            bounds.width,
            owner.values.get(part, ""),
            hint="搜索场景",
        )
        owner.values[part] = value
    elif part in ("rgba", "color_channels"):
        layout = ui.numeric_layout(bounds.width, 4, parts=("rgba", part), column_options=(4, 2, 1))
        for axis, (dx, dy, width, height) in zip("RGBA", layout.fields, strict=True):
            r = engine.input_number(
                "reference-channel/" + axis,
                Rect(bounds.x + dx, bounds.y + dy, width, height),
                owner.values.get("rgba/" + axis, 0.5 if axis != "A" else 1),
                axis=axis,
                lo=0,
                hi=1,
            )
            owner.values["rgba/" + axis] = r.value
    elif part in ("visual_group", "visual_group_item"):
        for i in range(6 if part == "visual_group" else 1):
            key = "visual/" + str(i)
            r = engine.checkbox(
                "reference-" + key,
                bounds.x + i * 48 * s,
                bounds.y,
                owner.values.get(key, True),
                label=str(i),
            )
            owner.values[key] = r.value
    elif part == "badge":
        from mojive.ui.theme import THEME
        from mojive.ui.workspace.status_components import StatusBadge

        badge = StatusBadge(ui)
        badge.draw(bounds.x, bounds.y + bounds.height / 2, "已暂停", THEME)
    elif part == "separator":
        ui.line(bounds.x, bounds.y + bounds.height / 2, bounds.width)
    elif part == "empty":
        ui.role_icon(
            "upload",
            bounds.x + bounds.width / 2 - 24 * s,
            bounds.y,
            48 * s,
            48 * s,
            role="activity",
        )
        engine.label(
            "reference-empty",
            Rect(bounds.x, bounds.y + 60 * s, bounds.width, 28 * s),
            "拖入场景文件以开始",
            parts="empty",
        )
    elif part == "notice":
        ui.rect(*bounds.tuple(), rgba8("field"), r=6)
        ui.icon("info", bounds.x + 12 * s, bounds.y + 10 * s, 16, rgba8("info"))
        ui.text(
            bounds.x + 40 * s,
            bounds.y + 9 * s,
            "场景已保存",
            max_width=bounds.width - 52 * s,
        )
    return [bounds.tuple()]


SURFACE_RENDERERS = {
    **dict.fromkeys(
        ("panel_header", "inspector_header", "dialog_header", "camera_header", "titlebar"), heading
    ),
    **dict.fromkeys(
        (
            "panel",
            "panel_body",
            "dialog",
            "dialog_body",
            "settings_layout",
            "settings_nav",
            "settings_page",
        ),
        panels,
    ),
    **dict.fromkeys(
        (
            "timeline_body",
            "timeline_model_group",
            "timeline_sheet",
            "timeline_ruler",
            "timeline_lane",
            "lane_label",
        ),
        timeline,
    ),
    **dict.fromkeys(
        ("application", "workspace", "camera_preview", "camera_body", "nav_widget"), viewport
    ),
    **dict.fromkeys(
        (
            "search",
            "rgba",
            "color_channels",
            "visual_group",
            "visual_group_item",
            "badge",
            "separator",
            "empty",
            "notice",
        ),
        small,
    ),
}
