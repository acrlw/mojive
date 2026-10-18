"""Explicit production consumers; catalogue additions require a reviewed binding."""

from dataclasses import dataclass

PADDING = "paddingX paddingY paddingLeft paddingRight paddingTop paddingBottom"
MARGIN = "marginLeft marginRight marginTop marginBottom"
TEXT = "fontSize lineHeightMode lineHeight"
SPACING = "gap rowGap columnGap"
FIELD = f"height {TEXT} {PADDING} {MARGIN} radius textAlign widthMode width"
BUTTON = f"height {TEXT} {PADDING} {MARGIN} gap radius iconSize align alignItems widthMode width"
HEADER = f"height {TEXT} {PADDING} {MARGIN} gap iconSize alignItems columnGap"
CONTAINER = f"{PADDING} {MARGIN} {SPACING}"
TOOLBAR_SEPARATOR = "separatorThickness separatorInset separatorGap separatorColor"


@dataclass(frozen=True, slots=True)
class ConsumerContract:
    source: str
    fields: frozenset[str]


def contract(source, fields):
    return ConsumerContract(source, frozenset(fields.split()))


# A field appears here only when its production painter or parent measurement
# consumes it. Parsing the specification and reference samples are not consumers.
PRODUCTION_CONTRACTS = {
    "window_title": contract("ui/workspace/window_chrome.py", "fontSize paddingX paddingY"),
    "dock_tabs": contract("ui/workspace/window_chrome.py", "fontSize paddingX paddingY gap"),
    "button": contract("design/native/controls.py", BUTTON),
    "icon_button": contract(
        "design/native/controls.py",
        f"height width {PADDING} {MARGIN} radius iconSize align alignItems",
    ),
    "input_text": contract("design/native/controls.py", FIELD),
    "textarea": contract(
        "ui/workspace/inspector_style.py",
        f"fontSize {PADDING} {MARGIN} radius minHeight widthMode width",
    ),
    "input_number": contract("design/native/controls.py", FIELD),
    "select": contract("design/native/controls.py", FIELD),
    "search": contract(
        "design/native/controls.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap radius iconSize",
    ),
    "axis_input": contract(
        "design/native/controls.py",
        f"height {TEXT} axisWidth paddingX paddingLeft paddingRight {MARGIN} textAlign radius",
    ),
    "range": contract("ui/panels/value_cards.py", "height thumbSize trackSize"),
    "checkbox": contract("design/native/controls.py", "size radius"),
    "switch": contract(
        "design/native/controls.py",
        f"height {TEXT} {MARGIN} gap columnGap switchWidth switchHeight switchThumb",
    ),
    "form_row": contract(
        "ui/workspace/numeric_fields.py",
        f"height {TEXT} {MARGIN} labelWidth labelGap alignItems rowGap columnGap",
    ),
    "form_label": contract("design/native/controls.py", f"{TEXT} {MARGIN} textAlign"),
    "input_shell": contract(
        "design/native/controls.py",
        f"height {PADDING} {MARGIN} gap columnGap radius unitSize alignItems",
    ),
    "range_number": contract(
        "ui/workspace/numeric_fields.py", f"{MARGIN} gap columnGap numberWidth alignItems"
    ),
    "vector_row": contract(
        "ui/workspace/numeric_fields.py",
        f"{TEXT} {MARGIN} labelGap labelWidth layout rowGap columnGap",
    ),
    "vector_group": contract("ui/workspace/numeric_fields.py", f"{MARGIN} {SPACING} columns"),
    "section": contract("ui/workspace/property_groups.py", MARGIN),
    "section_header": contract("ui/workspace/widgets.py", HEADER),
    "section_body": contract("ui/workspace/property_groups.py", CONTAINER),
    "button_row": contract("ui/workspace/button_rows.py", f"{CONTAINER} height align alignItems"),
    "segment": contract("ui/workspace/widgets.py", f"{PADDING} {MARGIN} gap columnGap align"),
    "rgba": contract("ui/workspace/numeric_fields.py", f"{MARGIN} {SPACING} marginY columns"),
    "color_editor": contract(
        "ui/workspace/color_editor.py", f"{MARGIN} {SPACING} pickerHeight pickerShare"
    ),
    "color_picker": contract("ui/workspace/color_editor.py", f"height {PADDING} {MARGIN}"),
    "color_input": contract(
        "ui/workspace/menu_color.py", f"width height {PADDING} {MARGIN} radius"
    ),
    "color_channels": contract("ui/workspace/inspector_style.py", "gap rowGap"),
    "panel": contract(
        "ui/workspace/panel_host.py", f"widthMode width heightMode height {MARGIN} rowGap radius"
    ),
    "panel_header": contract("ui/workspace/panel_host.py", HEADER),
    "inspector_header": contract(
        "ui/workspace/inspector_heading.py",
        f"minHeight {TEXT} {PADDING} {MARGIN} gap iconSize alignItems columnGap",
    ),
    "panel_body": contract(
        "ui/workspace/panel_host.py", f"widthMode width heightMode height {CONTAINER}"
    ),
    "tabs": contract(
        "ui/workspace/inspector_heading.py", f"{PADDING} {MARGIN} gap columnGap align"
    ),
    "tab": contract("design/native/controls.py", BUTTON),
    "settings_layout": contract(
        "ui/workspace/settings_layout.py", f"windowWidth navWidth minHeight {MARGIN} {SPACING}"
    ),
    "settings_nav": contract("ui/workspace/settings_layout.py", f"navWidth {CONTAINER} radius"),
    "settings_page": contract(
        "ui/workspace/settings_layout.py", f"{TEXT} {PADDING} {MARGIN} titleSize titleGap rowGap"
    ),
    "material": contract(
        "ui/workspace/material_library.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap radius swatchSize iconSize alignItems",
    ),
    "material_list": contract(
        "ui/workspace/material_library.py", f"{PADDING} {MARGIN} gap rowGap actionGap"
    ),
    "sensor": contract(
        "ui/workspace/sensors_panel.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap radius iconSize alignItems valueSize detailSize",
    ),
    "joint": contract("ui/workspace/pose_panel.py", f"{TEXT} {CONTAINER}"),
    "vector_heading": contract("design/native/controls.py", "gap columnGap"),
    "range_bounds": contract("ui/workspace/labeled_input.py", CONTAINER),
    "visual_group": contract("ui/workspace/settings_visuals.py", SPACING),
    "visual_group_item": contract("ui/workspace/settings_visuals.py", "gap columnGap"),
    "component_list": contract(
        "ui/workspace/model_components.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap alignItems",
    ),
    "component_path": contract(
        "ui/workspace/model_components.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap alignItems",
    ),
    "tree_row": contract(
        "ui/workspace/navigation.py",
        f"height {TEXT} {MARGIN} radius indent paddingY paddingTop paddingBottom gap columnGap rowGap",
    ),
    "tree_label": contract(
        "design/native/controls.py",
        f"widthMode width height {TEXT} {PADDING} {MARGIN} gap iconSize align alignItems",
    ),
    "tree_expander": contract(
        "design/native/controls.py", f"height width {PADDING} {MARGIN} iconSize align alignItems"
    ),
    "titlebar": contract(
        "ui/workspace/chrome.py", f"height {TEXT} {PADDING} {MARGIN} gap columnGap alignItems"
    ),
    "rail": contract(
        "ui/workspace/chrome.py", f"width {PADDING} {MARGIN} gap rowGap slotSize iconSize"
    ),
    "filter": contract("ui/workspace/navigation.py", BUTTON),
    "type_filters": contract("ui/workspace/navigation.py", SPACING),
    "viewport_toolbar": contract(
        "ui/workspace/viewport_controls.py", f"{PADDING} {MARGIN} gap columnGap top left alignItems"
    ),
    "toolbox": contract(
        "ui/workspace/viewport_controls.py",
        f"{PADDING} {MARGIN} gap rowGap radius left slotSize iconSize {TOOLBAR_SEPARATOR}",
    ),
    "transport": contract(
        "ui/workspace/viewport.py",
        f"{PADDING} {MARGIN} gap columnGap radius top alignItems {TOOLBAR_SEPARATOR}",
    ),
    "nav_widget": contract("ui/workspace/viewcube.py", "size right top"),
    "pending_edits": contract(
        "ui/workspace/notices.py",
        f"{TEXT} {PADDING} {MARGIN} gap columnGap radius bottom alignItems",
    ),
    "notice": contract(
        "ui/workspace/notices.py", f"{TEXT} {PADDING} {MARGIN} gap columnGap radius iconSize"
    ),
    "empty": contract(
        "ui/workspace/notices.py", f"{TEXT} {PADDING} {MARGIN} gap rowGap iconSize textAlign"
    ),
    "timeline_toolbar": contract("ui/workspace/timeline_panel.py", f"{CONTAINER} align alignItems"),
    "timeline_model_group": contract(
        "ui/workspace/timeline_header.py",
        f"columnGap {MARGIN} paddingLeft paddingRight paddingTop paddingBottom",
    ),
    "timeline_ruler": contract(
        "ui/workspace/timeline_track.py", f"height {TEXT} {MARGIN} labelTop"
    ),
    "timeline_lane": contract("ui/workspace/timeline_track.py", f"height {TEXT} {MARGIN}"),
    "timeline_sheet": contract(
        "ui/workspace/timeline_track.py", f"height labelWidth rightInset {MARGIN} rowGap headerGap"
    ),
    "lane_label": contract(
        "ui/workspace/timeline_track.py",
        f"{TEXT} {PADDING} {MARGIN} gap columnGap iconSize alignItems radius",
    ),
    "timeline_key": contract(
        "ui/workspace/timeline_track.py",
        f"size top {PADDING} {MARGIN} radius iconSize align alignItems",
    ),
    "timeline_details": contract(
        "ui/workspace/timeline_details.py",
        f"height {TEXT} {PADDING} {MARGIN} {SPACING} iconSize alignItems",
    ),
    "output_layout": contract("ui/workspace/output_panel.py", "toolbarGap listGap"),
    "output_toolbar": contract("ui/workspace/output_panel.py", f"{TEXT} {CONTAINER} alignItems"),
    "output_row": contract(
        "ui/workspace/output_panel.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap iconSize alignItems textAlign",
    ),
    "output_list": contract("ui/workspace/output_panel.py", "gap rowGap"),
    "menu": contract(
        "ui/workspace/menus.py",
        f"widthMode width heightMode height {TEXT} {PADDING} {MARGIN} radius rowGap",
    ),
    "form_popup": contract(
        "ui/workspace/form_popup.py", f"fontSize {PADDING} radius rowGap columnGap groupGap"
    ),
    "menu_item": contract("ui/workspace/menus.py", BUTTON),
    "menu_item_content": contract("ui/workspace/menu_rows.py", "gap columnGap"),
    "select_popup": contract(
        "ui/workspace/popup_style.py",
        f"widthMode width heightMode height {TEXT} {PADDING} {MARGIN} radius rowGap",
    ),
    "select_option": contract("ui/workspace/menu_rows.py", BUTTON + " checkGap"),
    "menu_trigger": contract(
        "ui/workspace/menus.py",
        f"height {TEXT} {PADDING} {MARGIN} radius align alignItems widthMode width",
    ),
    "menu_bar": contract("ui/workspace/menus.py", "gap columnGap"),
    "document_title": contract("ui/workspace/chrome.py", "gap columnGap"),
    "title_actions": contract("ui/workspace/chrome.py", "gap columnGap"),
    "status_context": contract(
        "ui/workspace/status_components.py", f"gap columnGap {PADDING} {MARGIN}"
    ),
    "status_hints": contract(
        "ui/workspace/status_components.py", f"gap columnGap {PADDING} {MARGIN}"
    ),
    "status_metrics": contract(
        "ui/workspace/status_components.py",
        f"gap columnGap {PADDING} {MARGIN} labelGap unitGap deltaValueWidth physicsValueWidth "
        "stepsValueWidth timeValueWidth renderValueWidth tooltipPaddingX tooltipPaddingY tooltipRowGap",
    ),
    "status_separator": contract(
        "ui/workspace/status_components.py",
        "visibility thickness heightMode height marginTop marginBottom",
    ),
    "bottom_header": contract("ui/workspace/viewport.py", "gap columnGap"),
    "time_field": contract("ui/workspace/time_readout.py", "gap columnGap"),
    "dialog": contract(
        "ui/workspace/menu_dialogs.py", f"width maxWidth {TEXT} {MARGIN} radius rowGap"
    ),
    "dialog_header": contract(
        "ui/workspace/menu_dialogs.py",
        f"{TEXT} {PADDING} {MARGIN} gap columnGap alignItems titleSize subtitleSize",
    ),
    "dialog_body": contract(
        "ui/workspace/menu_dialogs.py", f"{TEXT} {PADDING} {MARGIN} gap rowGap"
    ),
    "dialog_footer": contract(
        "ui/workspace/menu_dialogs.py", f"{PADDING} {MARGIN} align alignItems"
    ),
    "file_receipt": contract(
        "ui/workspace/notices.py",
        f"{TEXT} {PADDING} {MARGIN} radius gap columnGap widthMode width",
    ),
    "link_button": contract(
        "ui/workspace/notices.py",
        f"height {TEXT} {PADDING} {MARGIN} radius align alignItems widthMode width",
    ),
    "statusbar": contract(
        "ui/app/status.py",
        f"height {TEXT} {PADDING} {MARGIN} gap columnGap iconSize alignItems",
    ),
    "badge": contract(
        "ui/workspace/status_components.py", f"{TEXT} {PADDING} {MARGIN} radius textAlign"
    ),
    "camera_preview": contract(
        "ui/workspace/camera_preview.py", f"width radius right bottom {MARGIN} rowGap"
    ),
    "camera_header": contract(
        "ui/workspace/camera_preview.py",
        f"{TEXT} {PADDING} {MARGIN} gap columnGap iconSize alignItems",
    ),
    "camera_body": contract("ui/workspace/camera_preview.py", f"height heightMode fit {MARGIN}"),
    "stat_row": contract(
        "ui/workspace/statistics.py", f"height {TEXT} {PADDING} {MARGIN} gap columnGap align"
    ),
    "data_list": contract(
        "ui/workspace/statistics.py", f"{TEXT} {PADDING} {MARGIN} {SPACING} labelWidth valueSize"
    ),
    "table": contract("ui/workspace/statistics.py", f"{TEXT} {PADDING} {MARGIN} textAlign"),
    "statistics_list": contract("ui/workspace/statistics.py", "gap rowGap"),
    "application": contract("ui/workspace/chrome.py", f"{TEXT} titleHeight statusHeight rowGap"),
    "workspace": contract(
        "design/runtime.py", "sceneWidth inspectorWidth bottomHeight splitterWidth columnGap"
    ),
    "separator": contract("ui/workspace/widgets.py", "thickness margin"),
    "panel_tools": contract("ui/workspace/navigation.py", "gap rowGap columnGap"),
    "scene_tools": contract("ui/workspace/navigation.py", "gap rowGap columnGap"),
    "pose_tools": contract("ui/workspace/navigation.py", "gap rowGap"),
}


UNAVAILABLE_PARTS = {}
UNAVAILABLE_FIELDS = {}

CONDITIONS = {
    ("button", "height"): "同类部件默认高度可能不同；至少容纳文字、图标和内距。",
    ("input_number", "height"): "不能小于文字行高与上下内距。",
    ("input_number", "lineHeight"): "固定行高；超过最小控件高度才增高。",
    ("input_number", "lineHeightMode"): "固定模式采用指定行高。",
    ("axis_input", "height"): "不能小于文字行高与上下内距。",
    ("axis_input", "lineHeight"): "固定行高；超过最小控件高度才增高。",
    ("axis_input", "lineHeightMode"): "固定模式采用指定行高。",
    ("input_text", "height"): "不能小于文字行高与上下内距。",
    ("textarea", "minHeight"): "组件短字段增高；源码编辑器受弹窗剩余空间限制。",
    ("select", "height"): "不能小于文字行高与上下内距。",
    ("form_row", "height"): "不能小于标签和控件实际高度；XYZ 换行后最小高度增大。",
    ("form_row", "labelWidth"): "标签和值左右同排时；宽度不足则转到上方。",
    ("form_row", "labelGap"): "左右同排时；独立列距优先。",
    ("form_row", "columnGap"): "标签和值左右同排时。",
    ("form_row", "alignItems"): "控件高度小于行内可用高度时。",
    ("vector_row", "labelWidth"): "左侧标签布局；宽度不足时转到上方。",
    ("vector_row", "labelGap"): "左侧标签布局；独立列距优先。",
    ("vector_row", "columnGap"): "左侧标签与值之间。",
    ("switch", "gap"): "带标签的开关行；独立列距优先。",
    ("switch", "marginRight"): "预留开关右侧空白；左对齐且空间充足时不移动图形或标签。",
    ("switch", "fontSize"): "带标签或配对外部标签的开关行。",
    ("switch", "columnGap"): "标签与开关的最小留白；短标签视觉位置不变。",
    ("button", "gap"): "同时显示图标和文字时。",
    ("button", "iconSize"): "含图标的按钮。",
    ("button", "lineHeight"): "固定行高；文字竖向留白充足时。",
    ("input_shell", "unitSize"): "带单位的数值输入。",
    ("input_shell", "gap"): "带单位；独立列间距优先。",
    ("input_shell", "columnGap"): "数值与单位之间。",
    ("input_shell", "alignItems"): "数值行与输入框高度不同时。",
    ("range_number", "alignItems"): "滑块和输入框高度不同时。",
    ("rgba", "rowGap"): "通道换行后。",
    ("vector_group", "rowGap"): "分量换行后。",
    ("button_row", "rowGap"): "按钮换行后。",
    ("button_row", "columnGap"): "两个或更多按钮同排时。",
    ("button_row", "alignItems"): "最小组高超过内容高度，或同行按钮高度不同时。",
    ("button_row", "height"): "最小组高；低于按钮与内边距总高时使用内容高度。",
    ("range_bounds", "rowGap"): "标签与输入框换行后。",
    ("segment", "align"): "内容布局；等宽布局占满整行。",
    ("section_header", "iconSize"): "可折叠分组的展开箭头。",
    ("section_header", "gap"): "标题与展开箭头的最小留白；独立列距优先。",
    ("section_header", "columnGap"): "标题与展开箭头的最小留白；短标题视觉位置不变。",
    ("panel", "radius"): "浮动窗口。",
    ("panel", "rowGap"): "自定义面板标题与内容之间。",
    ("panel_body", "gap"): "普通表单行之间；专有列表使用自身间距。",
    ("panel_body", "rowGap"): "普通表单行之间；专有列表使用自身间距。",
    ("panel_header", "iconSize"): "设置标题等含前置图标的栏。",
    ("tab", "iconSize"): "含图标的窄屏标签。",
    ("tab", "radius"): "悬停或选中标签的背景。",
    ("settings_nav", "columnGap"): "窄窗口的多列导航。",
    ("component_list", "height"): "至少容纳标题和操作按钮。",
    ("component_path", "height"): "至少容纳标题和操作按钮。",
    ("material", "iconSize"): "选中材质卡片的勾选图标。",
    ("color_input", "width"): "默认占满列宽；窄框仅显示色块，悬停查看色值。",
    ("output_row", "alignItems"): "行高大于文字与图标高度时。",
    ("timeline_details", "rowGap"): "选中信息换行或显示错误时。",
    ("menu", "radius"): "弹窗外框圆角。",
    ("menu", "width"): "命令菜单的固定宽度。",
    ("select_popup", "width"): "下拉弹窗至少与输入框等宽。",
    ("select_option", "gap"): "选项包含图标时，调整图标与文字的留白。",
    ("menu_item", "radius"): "悬停背景。",
    ("filter", "radius"): "选中或悬停背景。",
    ("lane_label", "radius"): "选中或悬停背景。",
    ("tree_row", "radius"): "选中或悬停背景。",
    ("menu_item_content", "gap"): "文字与快捷键的最小间距；留白充足时不移动。",
    ("menu_item_content", "columnGap"): "文字与快捷键的最小间距；留白充足时不移动。",
    ("visual_group_item", "gap"): "可视组列距或标签到复选框的最小留白。",
    ("visual_group_item", "columnGap"): "可视组列距或标签到复选框的最小留白。",
    ("timeline_sheet", "height"): "至少容纳刻度栏和三个轨道。",
    ("timeline_key", "top"): "显式设置后固定位置；默认垂直居中。",
    ("timeline_key", "marginLeft"): "影响密集帧间距；不改变帧时刻。",
    ("timeline_key", "marginRight"): "影响密集帧间距；不改变帧时刻。",
    ("timeline_key", "align"): "图标小于内容区时，在固定时刻标记框内对齐。",
    ("timeline_key", "alignItems"): "图标小于内容区时，在固定时刻标记框内对齐。",
    ("data_list", "columnGap"): "标签与数值的最小间隔；两端对齐且留白充足时不移动。",
    ("camera_body", "height"): "固定高度；受视口剩余区域限制。",
    ("camera_body", "fit"): "固定高度或预览区与相机比例不同时。",
    ("application", "statusHeight"): "未单独指定状态栏高度时。",
    ("application", "titleHeight"): "未单独指定顶栏高度时。",
    ("statusbar", "height"): "不能小于文字、徽章与上下内距。",
    ("dialog_header", "fontSize"): "标题、副标题未单独指定字号时。",
    ("dialog_header", "columnGap"): "标题与关闭按钮的最小留白；窄窗口或长标题时可见。",
}

PART_CONDITIONS = {
    "textarea": "模型源码编辑器或模型组件的长文本／多行字段。",
    "table": "后端提供逐渲染通道计时数据时。",
    "output_list": "存在输出历史时。",
    "timeline_key": "存在关键帧或快照时。",
    "timeline_details": "选中单帧、快照或显示错误时。",
    "camera_preview": "启用相机预览时。",
    "camera_header": "启用相机预览时。",
    "file_receipt": "文件保存或输出回执。",
    "link_button": "文件保存或输出回执。",
    "notice": "操作完成、失败等临时结果。",
    "empty": "没有场景或拖入文件时。",
    "document_title": "标题栏足够宽时。",
}


def field_condition(part, field, rule):
    condition = CONDITIONS.get((part, field), "")
    if condition:
        return condition
    requirement = rule.get("condition")
    if requirement:
        return (
            "固定宽度模式。"
            if requirement["field"] == "widthMode"
            else "固定高度模式。"
            if requirement["field"] == "heightMode"
            else "固定行高模式。"
        )
    if part in ("component_list", "component_path"):
        return "场景源提供 model.components 能力时。"
    if part in PART_CONDITIONS:
        return PART_CONDITIONS[part]
    if field in ("gap", "rowGap", "columnGap"):
        return "独立行／列间距优先于通用间距。" if field == "gap" else ""
    if field == "lineHeight":
        return "固定行高模式。"
    return ""
