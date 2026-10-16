"""Reviewable state and composition contracts for the native design reference."""

from dataclasses import asdict, dataclass

from mojive.design.help import get_part_help
from mojive.design.specification import DEFAULT, SCHEMA, component_catalog, resolve_part_value
from mojive.design.support import native_support
from mojive.ui.design_palette import PALETTE


@dataclass(frozen=True)
class Contract:
    title: str
    parts: tuple[str, ...]
    states: tuple[str, ...]
    cases: tuple[str, ...]
    rule: str


def contract(title, parts, states, cases, rule):
    return Contract(
        title, tuple(parts.split()), tuple(states.split(" / ")), tuple(cases.split(" / ")), rule
    )


CONTRACTS = (
    contract(
        "动作与选择",
        "button icon_button tab filter menu_trigger menu_item link_button tree_label tree_expander timeline_key",
        "常态 / 悬停 / 按下 / 键盘焦点 / 禁用 / 选中",
        "文字 / 图标 / 图标与文字 / 长文本省略 / 窄宽度 / 点击与键盘激活",
        "状态不改变外框尺寸；禁用时不执行动作；省略文字与无标签图标提供简短提示。",
    ),
    contract(
        "文本编辑",
        "input_text textarea search",
        "空值 / 常态 / 悬停 / 编辑 / 键盘焦点 / 只读 / 禁用",
        "中文 / 多行 / 长文本 / 选中与复制 / 清空 / Enter 提交 / Escape 退出",
        "编辑使用原生光标、选择和输入法；标签左对齐，多行输入保留原生行距。",
    ),
    contract(
        "数值编辑",
        "input_number axis_input input_shell range_number range_bounds time_field range",
        "常态 / 悬停 / 按下 / 编辑 / 键盘焦点 / 禁用",
        "负数 / 零 / 小数 / 上下限 / 单位 / 拖动 / 直接输入 / 恢复初值",
        "数值与单位共用输入框预算；滑块与输入值同步；轴标签独立响应恢复操作。",
    ),
    contract(
        "布尔选择",
        "checkbox switch visual_group_item",
        "关闭 / 开启 / 悬停 / 按下 / 键盘焦点 / 禁用",
        "开与关 / 有标签 / 无标签 / 长标签 / 禁用的开与关",
        "控件在输入列左对齐，纵向居中；开关轨道、圆点与命中区域分别计量。",
    ),
    contract(
        "下拉选择",
        "select select_popup select_option",
        "常态 / 悬停 / 按下 / 展开 / 键盘焦点 / 禁用",
        "已选项 / 长选项 / 弹层滚动 / 键盘选择 / Escape 收起 / 窄宽度",
        "箭头与文字共享框内宽度；展开状态沿用按下色；菜单至少与触发框等宽。",
    ),
    contract(
        "表单与向量",
        "form_row form_label vector_row vector_group vector_heading joint",
        "标签同排 / 标签独立一行 / 分量同排 / 分量纵排 / 子控件状态",
        "XYZ / 单值 / 长标签 / 窄与宽往返 / 1×与2× / 参数即时调整",
        "父级分配标签与值列，子级使用剩余宽度；相同宽度得到相同布局；标签始终左对齐。",
    ),
    contract(
        "折叠分组",
        "section section_header section_body",
        "展开 / 收起 / 悬停 / 按下 / 子控件状态",
        "短标题 / 长标题 / 恢复按钮 / 连续分组 / 滚动后展开",
        "标题、内容与分隔线分别拥有间距；收起后不保留内容高度；不添加整行厚背景。",
    ),
    contract(
        "数据与消息",
        "badge notice empty stat_row data_list table material sensor component_list component_path output_row ",
        "常态 / 空数据 / 信息 / 警告 / 错误 / 子控件状态",
        "短内容 / 长内容 / 多行 / 无数据 / 数值与文本 / 可操作附件",
        "语义颜色用于状态识别；正文与数值字体分开配置；条目高度包含内容与内距。",
    ),
    contract(
        "集合与分组",
        "button_row tabs segment type_filters material_list  visual_group panel_tools output_list statistics_list color_channels menu_item_content menu_bar title_actions document_title status_context status_hints status_metrics",
        "单项 / 多项 / 选中项 / 换行 / 子控件状态",
        "组内间距 / 组间间距 / 内容溢出 / 筛选无结果 / 可用宽度变化",
        "容器定义行列间距；子项定义自身尺寸；组内间距不替代相邻组之间的外距。",
    ),
    contract(
        "树形列表",
        "tree_row",
        "常态 / 悬停 / 选中 / 展开 / 收起 / 可见 / 隐藏 / 禁用",
        "根节点 / 子节点 / 长名称 / 多选 / Shift 范围选择 / 可见性切换",
        "可见节点仅在行悬停时显示眼睛；隐藏节点常驻隐藏图标；图标保留独立交互状态。",
    ),
    contract(
        "面板与弹层",
        "panel panel_header inspector_header panel_body  menu dialog dialog_header dialog_body dialog_footer settings_layout settings_nav settings_page ",
        "打开 / 关闭 / 浮动 / 窄布局 / 子控件状态",
        "标题与工具 / 内容滚动 / 关闭 / 长标签 / 内容超高 / 窗口缩放",
        "标题、正文、底栏分别分配边距；设置标签按内容预算，输入框使用剩余空间。",
    ),
    contract(
        "颜色编辑",
        "color_editor rgba color_input color_picker",
        "常态 / 悬停 / 编辑 / 弹层展开 / 禁用",
        "RGB / RGBA / HEX / 透明度 / 通道同排与换行 / 重置",
        "通道组计入标签、色块、列距与可用宽度；颜色选择与通道输入同步。",
    ),
    contract(
        "应用与视口",
        "application workspace titlebar rail viewport_toolbar transport toolbox nav_widget  camera_preview camera_header camera_body  pending_edits file_receipt statusbar bottom_header ",
        "默认布局 / 紧凑布局 / 面板开关 / 子控件状态",
        "最小可用尺寸 / 面板分隔 / 合并与独立窗口 / 中英文 / 1×与2× / 相机预览",
        "应用布局与组件布局分层；视口保持实际预览尺寸；浮层在视口内定位。",
    ),
    contract(
        "时间轴",
        "timeline_toolbar timeline_model_group timeline_sheet timeline_ruler timeline_lane lane_label timeline_details",
        "暂停 / 播放 / 选中 / 录制 / 空轨道 / 子控件状态",
        "工具栏换行 / 模型选择与添加 / 刻度与标题同高 / 标题到轨道间距 / 帧定位 / 起止时间",
        "模型选择与添加按钮组成一组；组与刻度共享标题高度；轨道另设 headerGap。",
    ),
    contract(
        "输出工具",
        "output_toolbar output_layout",
        "常态 / 筛选选中 / 悬停 / 按下 / 禁用",
        "信息警告错误 / 搜索 / 复制 / 自动跟随 / 清空 / 空结果",
        "过滤计数与内容同步；工具条与日志列表分开管理间距；日志字体独立设置。",
    ),
    contract(
        "分隔",
        "separator status_separator",
        "常态 / 显示 / 隐藏",
        "水平 / 竖向 / 内缩 / 不同厚度",
        "分隔线只承担层级划分；厚度、两端缩进与相邻内容间距分别定义。",
    ),
)

# These behaviors span component families and are not individual size parameters.
BEHAVIORS = (
    (
        "滚动条",
        "滚动显示 → 停止滚动 → 延迟后直接隐藏 → 边缘悬停唤醒",
        "各面板独立计时；普通点击或移动不刷新滚动计时；设置可调整隐藏延迟。",
        "panel_body",
    ),
    (
        "提示",
        "完整标签 → 无提示；省略标签／无标签图标 → 简短提示",
        "分类标签保留含义提示；不在普通表单悬停时堆叠说明。",
        "filter",
    ),
    (
        "可见性",
        "可见 → 行悬停显示眼睛 → 点击隐藏 → 隐藏图标常驻",
        "眼睛命中区域不改变行宽或选择范围；常态、悬停、按下、禁用分别响应。",
        "tree_row",
    ),
    (
        "多选",
        "起始节点 → Shift + 终止节点 → 可见顺序范围",
        "范围由当前展开的树顺序确定；折叠后重新计算可见节点。",
        "tree_row",
    ),
    (
        "表单响应",
        "宽布局 → 标签换行 → 分量纵排 → 反向恢复",
        "按当前可用宽度计算，不依赖上一次布局；标签左对齐，值列共同起点。",
        "vector_row",
    ),
    (
        "输入单位",
        "拖动滑块 ↔ 输入数值与单位",
        "单位随值显示在同一框内，° 与 m 不占独立按钮；滑块保留可操作长度。",
        "range_number",
    ),
    (
        "键盘操作",
        "Tab 定位 → 编辑／激活 → Escape 退出",
        "输入保留光标与选区；禁用控件不提交修改；控件焦点与选中状态分别表达。",
        "input_text",
    ),
    (
        "分组排版",
        "父容器 → 同级分组 → 组内控件",
        "宽度扣除内距与列距后再分配；容器的内距不重复叠加到子控件。",
        "timeline_model_group",
    ),
    (
        "字体",
        "界面标签 / 数值输入 / 输出日志",
        "输入与输出可以分别设置字体及字号；数值保持对齐；单位与数值同步缩放。",
        "output_row",
    ),
    (
        "实时规范",
        "调整 → 即时预览 → 恢复／撤销 → 保存",
        "同类组件共享参数；生产消费者、适用条件与未接入字段逐项列出。",
        "input_number",
    ),
    (
        "窗口与缩放",
        "合并工作台 ↔ 独立工作台；1× ↔ 2×",
        "逻辑尺寸不变，绘制与命中边界同步缩放；预览窗口独立维持实际运行尺寸。",
        "workspace",
    ),
    (
        "Gizmo",
        "普通 / 悬停 / 拖动；本体坐标系 / 世界坐标系",
        "Gizmo 保留独立配色，界面主色调整不改变轴与交互高亮语义。",
        "toolbox",
    ),
)


def contract_index(catalog=None):
    catalog = component_catalog() if catalog is None else catalog
    index = {}
    for group in CONTRACTS:
        for part in group.parts:
            if part in index:
                raise ValueError(f"Duplicate reference contract: {part}")
            index[part] = group
    if index.keys() != catalog.keys():
        raise ValueError(f"Reference contract mismatch: {sorted(index.keys() ^ catalog.keys())}")
    return index


def sample_variants(part):
    """Content variants supported by the reference's native value owner."""
    if part in ("input_text", "textarea"):
        return (
            {
                "默认": "示例文字",
                "空文本": "",
                "长文本": "场景中的具有较长名称的组件 / component_001",
                "多行": "第一行\n第二行\n第三行",
            }
            if part == "textarea"
            else {
                "默认": "示例文字",
                "空文本": "",
                "长文本": "场景中的具有较长名称的组件 / component_001",
            }
        )
    if part in ("switch", "checkbox", "section", "section_header", "section_body"):
        return {"开启": True, "关闭": False}
    if part in ("range", "range_number"):
        maximum = 1.0 if part == "range" else 10.0
        return {"中间值": maximum / 2, "下限": 0.0, "上限": maximum}
    if part in ("input_number", "axis_input", "input_shell", "range_bounds"):
        return {"默认": 1.25, "零": 0.0, "负数": -12.375, "大数": 1234567.89}
    if part == "select":
        return {"首项": "选项一", "次项": "选项二", "长选项": "具有较长名称的选项三"}
    return {}


def reference_entry(part):
    if part == "textarea":
        return {"area": "模型源码", "steps": ["打开模型源码编辑器，编辑与滚动多行文本。"]}
    return get_part_help(part)["preview"]


def reference_manifest(spec=None):
    """Export every field together with its applicability and native consumer."""
    spec = DEFAULT if spec is None else spec
    catalog, support, contracts = component_catalog(), native_support(), contract_index()
    parts = {}
    for part, definition in catalog.items():
        parts[part] = {
            "label": definition["label"],
            "category": definition["category"],
            "contract": asdict(contracts[part]),
            "entry": reference_entry(part),
            "variants": sample_variants(part),
            "fields": {
                field: {
                    **rule,
                    **support[part]["fields"][field],
                    "activation": rule.get("condition"),
                    "value": resolve_part_value(spec, part, field, catalog=catalog),
                }
                for field, rule in definition["fields"].items()
            },
        }
    fields = [f for p in parts.values() for f in p["fields"].values()]
    return {
        "parts": parts,
        "behaviors": BEHAVIORS,
        "palette": PALETTE,
        "globalSpecifications": {
            group: {
                key: {**definition, "value": spec[group][key]}
                for key, definition in SCHEMA[group].items()
            }
            for group in ("tokens", "layout", "native")
        },
        "counts": {
            "parts": len(parts),
            "fields": len(fields),
            "production": sum(f["status"] == "applied" for f in fields),
            "reference": sum(f["status"] == "reference" for f in fields),
        },
    }
