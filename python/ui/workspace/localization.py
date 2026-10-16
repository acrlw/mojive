"""Translate native workspace copy through the product catalogue."""

from mojive.ui.localization import Language

_ZH_CN = {
    "Interaction guide": "交互指南",
    "Drag an empty area": "拖动空白处",
    "Middle mouse drag": "按中键拖动",
    "Mouse wheel": "滚轮",
    "Transform tools": "变换工具",
    "World / body frame": "世界／本体坐标系",
    "Frame all / focus": "显示全部／聚焦",
    "Precise gizmo input": "精确变换输入",
    "Double-click a handle": "双击手柄",
    "Undo / redo": "撤销／重做",
    "Simulation playback": "仿真播放",
    "Dock panels": "停靠面板",
    "Drag a panel tab": "拖动面板标签",
    "Render Backend": "渲染后端",
    "Passes": "渲染阶段",
    "Removed keyframe": "已删除关键帧",
}

_ZH_PREFIXES = (
    ("Updated keyframe ", "已更新关键帧 "),
    ("Added keyframe ", "已添加关键帧 "),
)


def workspace_text(localizer, value):
    if localizer.language is Language.SIMPLIFIED_CHINESE:
        if value in _ZH_CN:
            return _ZH_CN[value]
        for prefix, translated in _ZH_PREFIXES:
            if value.startswith(prefix):
                return translated + value[len(prefix) :]
        suffix = " / double-click"
        if value.endswith(suffix):
            return value[: -len(suffix)] + " / 双击"
    return localizer.text(value)


def statistics_caption(localizer, value):
    """Translate backend note labels without changing the visual Frames category."""
    if value == "Frames" and localizer.language is Language.SIMPLIFIED_CHINESE:
        return "帧数"
    return workspace_text(localizer, value)
