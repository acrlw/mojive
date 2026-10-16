"""Classify each production instance without counting parsed values as paint effects."""

from mojive.design.consumer_support import UNAVAILABLE_FIELDS, UNAVAILABLE_PARTS

TEXT_FIELDS = frozenset(("fontSize", "lineHeight", "lineHeightMode", "textAlign"))
BUTTON_PARTS = frozenset(
    (
        "button",
        "icon_button",
        "tab",
        "filter",
        "menu_trigger",
        "menu_item",
        "link_button",
        "tree_label",
        "tree_expander",
    )
)


def identity(record):
    return f"{record.get('windowId', '')}:{record['key']}"


def physical_change(before, after, fields):
    return any(before[field] != after[field] for field in before.keys() & after.keys() & fields)


def inactive_reason(before, after, part, field, spec, scale):
    declared = UNAVAILABLE_FIELDS.get((part, field)) or UNAVAILABLE_PARTS.get(part)
    if declared:
        return declared
    if part == "select_option" and field == "gap" and not before.get("iconRect"):
        return "此选项没有前置图标；勾选标记与文字之间的间距由 checkGap 控制。"
    owned_spacing = before.get("ownedSpacingPart")
    if owned_spacing and field in ("gap", "rowGap", "columnGap"):
        return f"此列表的间距由 {owned_spacing} 控制。"
    consumed = before.get("consumedFields", {}).get(part)
    if consumed is None and (part == "application" or before.get("parts") == [part]):
        consumed = before.get("fieldsConsumed")
    if consumed is not None and field not in consumed:
        owned_spacing = before.get("ownedSpacingPart")
        if owned_spacing and field in ("gap", "rowGap", "columnGap"):
            return f"此列表的间距由 {owned_spacing} 控制。"
        return "此生产角色没有对应布局属性，参数作用于其他同类角色。"
    if part == "segment" and field == "align" and before.get("layout") == "equal":
        return "等宽选项占满整行，外部对齐只在内容宽度布局生效。"
    if part == "panel" and field == "radius" and before.get("docked") is True:
        return "停靠面板连接到工作区，仅浮动窗口绘制外框圆角。"
    if part == "material" and field == "iconSize" and before.get("selected") is False:
        return "未选中材料卡片不显示勾选图标。"
    if part == "tree_row" and field == "indent" and before.get("depth") == 0:
        return "根层项目没有缩进；该参数移动深度大于零的项目。"
    if part == "timeline_lane" and field in TEXT_FIELDS and before.get("hasLaneText") is False:
        return "这条轨道没有片段文字；字号和行高作用于含文字的轨道。"
    if (
        part == "menu_item_content"
        and field in ("gap", "columnGap")
        and not before.get("shortcutRect")
    ):
        return "此菜单项没有快捷键；该间距用于文字与快捷键之间。"
    kind = before.get("kind")
    if field == "radius" and before.get("background") is False:
        return "此实例不绘制背景。"
    if field in ("labelWidth", "labelGap", "columnGap") and before.get("layout") == "top":
        return "标签独占上方一行，该横向尺寸只在左侧标签布局生效。"
    if (
        field == "height"
        and before.get("minContentHeight") is not None
        and after["hit"][3] <= before["minContentHeight"] + 0.1
    ):
        return "设置的行高仍低于实际子控件最低高度，当前按子内容撑高。"
    if field == "alignItems" and before.get("minContentHeight") is not None:
        row_height = before["hit"][3]
        if before.get("layout") == "top":
            row_height -= (before.get("labelRect") or (0, 0, 0, 0))[3]
        margins = before.get("margin", (0, 0, 0, 0))
        row_height -= margins[1] + margins[3]
        if row_height <= before["minContentHeight"] + 0.1:
            return "子控件已占满输入区高度，没有纵向对齐余量。"
    if field in ("gap", "columnGap") and kind == "switch" and not before.get("labelRect"):
        return "此实例仅为开关图形；标签间距由同部件的 label-row 实例控制。"
    if field in TEXT_FIELDS and kind == "switch" and not before.get("labelRect"):
        return "此实例仅为开关图形；文字由同部件的标签行绘制。"
    if kind == "button":
        if field in TEXT_FIELDS and not before.get("label"):
            return "此实例仅显示图标，没有文字。"
        if field == "iconSize" and not before.get("icon"):
            return "此实例没有图标。"
        if (
            field in ("gap", "columnGap")
            and part in BUTTON_PARTS
            and not (before.get("label") and before.get("icon"))
        ):
            return "此实例没有同时显示图标与文字。"
        if field == "radius" and "background" in before:
            if not before["background"]:
                return "此实例不绘制背景。"
            if not any(before.get(state) for state in ("active", "filled", "hovered")):
                return "此实例未选中、未悬停，也没有常驻背景。"
    origin = after.get("styleOrigins", {}).get(field)
    if origin and origin != part and field in spec.get("parts", {}).get(origin, {}):
        return f"此实例由 {origin}.{field} 的显式覆盖控制。"
    if field in ("paddingX", "paddingY"):
        edges = (
            ("paddingLeft", "paddingRight")
            if field == "paddingX"
            else ("paddingTop", "paddingBottom")
        )
        origins = after.get("styleOrigins", {})
        if all(edge in spec.get("parts", {}).get(origins.get(edge), {}) for edge in edges):
            return "两侧内距已有独立覆盖，优先于该轴内距。"
    content = before.get("content")
    values = before.get("fieldValues", {})
    if field == "alignItems" and content:
        icon_height = (before.get("iconRect") or (0, 0, 0, 0))[3]
        text_height = (before.get("textRect") or (0, 0, 0, 0))[3]
        if values.get("lineHeightMode") == "fixed":
            text_height = max(text_height, values.get("lineHeight", 0) * scale)
        if content[3] <= max(icon_height, text_height) + 0.1:
            return "内容已占满当前高度，没有纵向对齐余量。"
    if field in ("align", "textAlign") and content:
        icon_width = (before.get("iconRect") or (0, 0, 0, 0))[2]
        text_width = (before.get("textRect") or (0, 0, 0, 0))[2]
        gap = before.get("gap", 0) if icon_width and text_width else 0
        if content[2] <= icon_width + text_width + gap + 0.1:
            return "内容已占满当前宽度，没有横向对齐余量。"
    fields = before.get("fields")
    if fields and field == "rowGap" and len({round(item[1], 2) for item in fields}) == 1:
        return "分量在同一行，当前没有行间距。"
    if fields and field in ("gap", "columnGap") and len(fields) == 1:
        return "当前只有一个分量，没有列间距。"
    return ""


def instance_effects(before, after, part, field, spec, scale, physical_fields, effect):
    """Keep every previously visible instance and distinguish clipping from removal."""
    initial = {identity(record): record for record in before["records"]}
    visible = {identity(record): record for record in after["records"]}
    submitted = {identity(record): record for record in after.get("rawRecords", after["records"])}
    geometry = set(effect["changedGeometryKeys"])
    drawing = set(effect["changedDrawKeys"])
    result = []
    for key, record in initial.items():
        row = {"instance": key, "kind": record["kind"]}
        current = visible.get(key)
        if current is None:
            current = submitted.get(key)
            if current is None:
                row.update(status="removed", reason="修改后此实例不再提交生产绘制记录。")
            else:
                row.update(
                    status="not-visible",
                    reason="实例仍提交，但移出当前滚动或工作台遮挡后的可见区域。",
                    geometryChanged=physical_change(record, current, physical_fields),
                )
        elif key in geometry or key in drawing:
            row.update(
                status="changed", geometryChanged=key in geometry, drawChanged=key in drawing
            )
        else:
            reason = inactive_reason(record, current, part, field, spec, scale)
            row.update(
                status="inactive" if reason else "unchanged",
                reason=reason or "此可见实例的实际几何与绘制均未改变。",
            )
        result.append(row)
    result.extend(
        {"instance": key, "kind": visible[key]["kind"], "status": "newly-visible"}
        for key in visible.keys() - initial.keys()
    )
    return result


def field_effect_status(effects):
    """Require every applicable baseline instance, retaining any loss of visibility."""
    baseline = [row for row in effects if row["status"] != "newly-visible"]
    if not baseline:
        return "absent", "定位后没有此类型的可见生产组件绘制记录。"
    statuses = {row["status"] for row in baseline}
    if statuses & {"removed", "not-visible"}:
        return "visibility-loss", "修改使原有实例消失或移出可见区域，不能完成逐实例绘制验收。"
    if "unchanged" in statuses:
        if "changed" in statuses:
            return "partial-effect", "部分同类实例未变化，尚未证明所有适用实例同步生效。"
        return "unchanged", "字段值已提交，但适用生产实例的几何与绘制均未改变。"
    if "changed" in statuses:
        return "observed", "所有适用可见实例均观察到实际几何或绘制变化。"
    return "inactive", "当前各实例均受已记录的具体显示条件或独立覆盖限制。"
