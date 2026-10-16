"""Native visual plates for palette, type, interaction ink and spacing."""

from imgui_bundle import imgui as g

from mojive.ui.design_palette import PALETTE, rgba8
from mojive.ui.interaction_paint import ItemPaint, draw_switch
from mojive.ui.workspace.design import DEFAULT_DESIGN
from mojive.ui.workspace.widgets import color


def title(ui, text, subtitle=""):
    g.push_font(ui.bold, 24 * ui.s)
    g.text_unformatted(text)
    g.pop_font()
    if subtitle:
        g.text_colored(tuple(v / 255 for v in rgba8("secondary")), subtitle)
    g.spacing()


def foundation_board(ui, runtime):
    title(ui, "视觉基础", "低亮度背景 · 绿色强调 · 清晰的内容层级")
    groups = (
        ("品牌与强调", ("brand", "primary", "primary_bright", "primary_dim", "selected")),
        ("表面与边界", ("background", "child", "popup", "field", "border")),
        ("交互与文本", ("field_hovered", "field_active", "text", "secondary", "disabled")),
        ("语义状态", ("info", "success", "warning", "danger", "purple")),
    )
    assigned = {role for _, roles in groups for role in roles}
    remaining = tuple(role for role in PALETTE if role not in assigned)
    if remaining:
        groups += (("辅助层级", remaining),)
    s = ui.s
    for heading, roles in groups:
        g.text_unformatted(heading)
        width = g.get_content_region_avail().x
        cols = max(1, min(5, int(width / (128 * s))))
        if g.begin_table("palette-" + heading, cols):
            for role in roles:
                g.table_next_column()
                x, y = g.get_cursor_screen_pos()
                w = max(1, g.get_content_region_avail().x - 8 * s)
                ui.rect(x, y, w, 42 * s, rgba8(role), r=6)
                g.dummy((w, 46 * s))
                g.text_unformatted(PALETTE[role].upper())
                g.text_unformatted(role)
            g.end_table()
        g.spacing()
    g.separator()
    title(ui, "字体与尺度")
    for label, font, value in (
        ("界面正文 · 场景、检查器与设置", ui.body, runtime.value("body_font_size")),
        ("分组标题 · 本体惯性与动力学", ui.bold, runtime.value("title_font_size")),
        ("数值与单位 · −12.375 m / 42.0°", ui.mono, runtime.value("mono_font_size")),
    ):
        g.push_font(font, value * s)
        g.text_unformatted(label)
        g.pop_font()
        g.same_line()
        g.text_unformatted(f"{value:g} px")
    g.spacing()
    g.separator()
    title(ui, "坐标轴与 Gizmo")
    from mojive.interaction.gizmo_palette import (
        JOINT_COLOR,
        JOINT_DARK_COLOR,
        JOINT_HIGHLIGHT_COLOR,
    )
    from mojive.ui.theme import AXIS_COLORS

    colors = [
        *((axis.upper(), AXIS_COLORS[axis]) for axis in "xyz"),
        ("关节", JOINT_COLOR),
        ("关节高亮", JOINT_HIGHLIGHT_COLOR),
        ("关节暗部", JOINT_DARK_COLOR),
    ]
    if g.begin_table("independent-colors", 3):
        for label, ink in colors:
            g.table_next_column()
            x, y = g.get_cursor_screen_pos()
            tint = tuple(round(v * 255) for v in ink)
            ui.rect(x, y, min(150 * s, g.get_content_region_avail().x), 32 * s, tint)
            g.dummy((100 * s, 38 * s))
            g.text_unformatted(label + "  #" + "".join(f"{v:02X}" for v in tint[:3]))
        g.end_table()
    g.spacing()
    g.separator()
    title(ui, "图标与命中区域")
    from mojive.ui.workspace.widget_icons import GLYPHS, ICON_ROLES

    for role, spec in ICON_ROLES.items():
        g.text_unformatted(f"{role}:  图形 {spec.canvas_size:g} px / 槽位 {spec.slot_size:g} px")
    columns = max(1, int(g.get_content_region_avail().x / (120 * s)))
    if g.begin_table("icon-catalogue", columns):
        for name in GLYPHS:
            g.table_next_column()
            x, y = g.get_cursor_screen_pos()
            ui.role_icon(name, x, y, 28 * s, 28 * s, role="toolbar")
            g.dummy((28 * s, 30 * s))
            g.text_unformatted(name)
        g.end_table()


def state_board(ui, runtime):
    title(ui, "交互状态", "状态保持相同尺寸；焦点、选择与禁用分别表达")
    s = ui.s
    design = runtime.dimensions(DEFAULT_DESIGN)
    columns = ("常态", "悬停", "按下", "禁用")
    paints = (
        ItemPaint(),
        ItemPaint(hovered=True),
        ItemPaint(active=True),
        ItemPaint(disabled=True),
    )
    if not g.begin_table("state-ink", 4):
        return
    for label, paint in zip(columns, paints, strict=True):
        g.table_next_column()
        g.text_unformatted(label)
        x, y = g.get_cursor_screen_pos()
        w = max(1, min(168 * s, g.get_content_region_avail().x - 4 * s))
        g.begin_disabled(paint.disabled)
        ui.rect(x, y, w, design.control_height * s, paint.color(), r=design.control_radius)
        ui.text(x + 8 * s, y + 6 * s, "1.250 m", size=design.mono_size, mono=True)
        g.dummy((w, 44 * s))
        for value in (False, True):
            x, y = g.get_cursor_screen_pos()
            draw_switch(
                ui,
                x,
                y,
                design.switch_track_width * s,
                design.switch_track_height * s,
                design.switch_thumb_size * s,
                value,
                paint,
            )
            g.dummy((w, 38 * s))
        g.end_disabled()
    g.end_table()
    g.spacing()
    g.text_unformatted("输入背景与开关使用生产界面的 ItemPaint / draw_switch。")


def layout_board(ui, runtime):
    title(ui, "尺寸与层级", "逻辑像素；1× 与 2× 同比例缩放")
    s, d = ui.s, runtime.dimensions(DEFAULT_DESIGN)
    metrics = (
        ("控件行高", d.control_height),
        ("控件圆角", d.control_radius),
        ("面板横向内距", d.panel_padding_x),
        ("面板纵向内距", d.panel_padding_y),
        ("行间距", d.row_gap),
        ("分量间距", d.numeric_gap),
        ("标签到输入列", d.label_gap),
        ("开关宽度", d.switch_track_width),
        ("开关高度", d.switch_track_height),
        ("开关圆点", d.switch_thumb_size),
    )
    for label, value in metrics:
        x, y = g.get_cursor_screen_pos()
        ui.text(x, y + 4 * s, label)
        ui.rect(x + 160 * s, y + 6 * s, value * s, 12 * s, rgba8("primary_dim"), r=2)
        ui.text(x + 230 * s, y + 4 * s, f"{value:g} px", mono=True)
        g.dummy((320 * s, 26 * s))
    g.spacing()
    g.text_unformatted("容器 → 分组 → 同级控件")
    x, y = g.get_cursor_screen_pos()
    w = min(g.get_content_region_avail().x, 720 * s)
    h = 170 * s
    ui.rect(x, y, w, h, rgba8("child"), r=8)
    inset = 12 * s
    ui.rect(x + inset, y + inset, w - 2 * inset, 50 * s, rgba8("field"), r=6)
    ui.text(x + 24 * s, y + 28 * s, "模型选择  +  添加按钮", size=16)
    ui.text(x + 24 * s, y + 72 * s, "headerGap", c=rgba8("primary"))
    ui.rect(x + inset, y + 98 * s, w - 2 * inset, 58 * s, rgba8("selected"), r=6)
    ui.text(x + 24 * s, y + 118 * s, "关键帧轨道", size=16)
    ui.draw.add_rect((x, y), (x + w, y + h), color(rgba8("border")), 8 * s)
    g.dummy((w, h + 12 * s))
    g.text_wrapped(
        "组内使用 columnGap；分组到轨道使用 headerGap；内距与外距分别归属父级和当前部件。"
    )
