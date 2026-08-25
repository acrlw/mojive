"""Editor language selection and localized UI text."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Language(StrEnum):
    ENGLISH = "en"
    SIMPLIFIED_CHINESE = "zh_CN"


LANGUAGE_LABELS: dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.SIMPLIFIED_CHINESE: "简体中文",
}


_ZH_CN = {
    "File": "文件",
    "Edit": "编辑",
    "Entity": "实体",
    "Create": "创建",
    "New Scene": "新建场景",
    "Open Scene...": "打开场景…",
    "Save": "保存",
    "Save As...": "另存为…",
    "Open Model (MJCF / URDF)...": "打开模型（MJCF / URDF）…",
    "Add Models (MJCF / URDF)...": "添加模型（MJCF / URDF）…",
    "Remove Model": "移除模型",
    "Reload Model": "重新加载模型",
    "Resource Directories": "资源目录",
    "Add Directory...": "添加目录…",
    "Remove": "移除",
    "Quit": "退出",
    "Undo": "撤销",
    "Redo": "重做",
    "Settings...": "设置…",
    "Duplicate": "复制",
    "Rename": "重命名",
    "Delete": "删除",
    "Box": "立方体",
    "Sphere": "球体",
    "Cylinder": "圆柱体",
    "Cone": "圆锥体",
    "Plane": "平面",
    "Point Light": "点光源",
    "Camera": "相机",
    "Untitled": "未命名",
    "Viewport": "视口",
    "Control": "控制",
    "Hierarchy": "层级",
    "Inspector": "检查器",
    "Joints": "关节",
    "Plot": "曲线",
    "Stats": "统计",
    "Settings": "设置",
    "General": "常规",
    "MuJoCo Visuals": "MuJoCo 可视化",
    "Close": "关闭",
    "Sensors": "传感器",
    "Help": "帮助",
    "Info": "信息",
    "Application": "应用程序",
    "Language": "语言",
    "UI font": "界面字体",
    "CJK font": "中日韩字体",
    "Rendering": "渲染",
    "Interaction": "交互",
    "Backend": "后端",
    "Graphics device": "图形设备",
    "Gizmo style": "Gizmo 样式",
    "Gizmo orientation": "Gizmo 坐标系",
    "Rotation dial projection": "旋转刻度盘投影",
    "Scene helpers": "场景辅助图形",
    "Forge render flags": "Forge 渲染开关",
    "3D gizmo": "3D Gizmo",
    "Use the flat 2D overlay": "使用平面 2D 叠加层",
    "World frame (T)": "世界坐标系（T）",
    "position snap (Shift)": "位置吸附（Shift）",
    "rotation snap (Shift)": "旋转吸附（Shift）",
    "rotation tick scale": "旋转刻度大小",
    "perturb corner radius": "扰动轮廓圆角",
    "scene entity helpers": "场景实体辅助图形",
    "selected influence volumes": "选中实体的影响范围",
    "visual groups": "可视组",
    "BVH depth": "BVH 深度",
    "source": "来源",
    "free": "编辑器相机",
    "missing": "缺失",
    "Return to Editor Camera": "返回编辑器相机",
    "View Through Camera": "切换到此相机",
    "Lock gizmo while simulation runs": "模拟运行时锁定 Gizmo",
    "Pin": "固定",
    "Pinned": "已固定",
    "Lock": "锁定相机",
    "Locked": "已锁定",
    "model camera follows scene kinematics": "模型相机跟随场景运动",
    "presets": "预设",
    "front": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "top": "上",
    "bottom": "下",
    "iso": "等轴测",
    "frame all": "显示全部",
    "bookmarks and snapshots": "书签与场景快照",
    "camera bookmark": "相机书签",
    "scene snapshot": "场景快照",
    "copy camera snapshot": "复制相机快照",
    "copy qpos": "复制 qpos",
    "copy reproduction state": "复制复现状态",
    "orthographic": "正交投影",
    "position": "位置",
    "target": "目标",
    "up": "上方向",
    "vertical fov": "垂直视场角",
    "near": "近平面",
    "far": "远平面",
    "ortho height": "正交高度",
}


def settings_path() -> Path:
    override = os.environ.get("FORGE_VIEWER_SETTINGS")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "forge-viewer" / "settings.json"


def parse_language(value: Language | str) -> Language:
    if isinstance(value, Language):
        return value
    normalized = str(value).strip().split(":", 1)[0].split(".", 1)[0].replace("-", "_").lower()
    if normalized in {"zh", "zh_cn", "zh_hans", "zh_chs"}:
        return Language.SIMPLIFIED_CHINESE
    if normalized in {"en", "en_us", "en_gb"}:
        return Language.ENGLISH
    raise ValueError(f"Unsupported language: {value}")


def _read_language(path: Path) -> Language:
    requested = os.environ.get("FORGE_VIEWER_LANGUAGE")
    if requested:
        try:
            return parse_language(requested)
        except ValueError:
            return Language.ENGLISH
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("language", Language.ENGLISH)
        return parse_language(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return Language.ENGLISH


@dataclass
class Localizer:
    language: Language = Language.ENGLISH
    path: Path | None = None

    @classmethod
    def load(cls) -> Localizer:
        path = settings_path()
        return cls(_read_language(path), path)

    def text(self, value: str) -> str:
        if self.language is Language.SIMPLIFIED_CHINESE:
            return _ZH_CN.get(value, value)
        return value

    def set_language(self, value: Language | str, *, persist: bool = True) -> None:
        self.language = parse_language(value)
        if not persist or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"language": self.language.value}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
