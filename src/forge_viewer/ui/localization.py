"""Editor preference persistence and localized UI text."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
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
    "View": "视图",
    "Window": "窗口",
    "Help": "帮助",
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
    "Frame All": "显示全部",
    "Capture Viewport": "截取视口",
    "Scene Helpers": "场景辅助图形",
    "Influence Volumes": "影响范围",
    "Reset Layout": "重置布局",
    "Interaction Reference": "交互参考",
    "About": "关于",
    "Duplicate": "复制",
    "Rename": "重命名",
    "Delete": "删除",
    "Box": "立方体",
    "Sphere": "球体",
    "Cylinder": "圆柱体",
    "Cone": "圆锥体",
    "Plane": "平面",
    "Site": "站点",
    "Point Light": "点光源",
    "Camera": "相机",
    "Untitled": "未命名",
    "Viewport": "视口",
    "Control": "控制",
    "Hierarchy": "层级",
    "Assets": "资源",
    "Inspector": "检查器",
    "Joints": "关节",
    "Keyframes": "关键帧",
    "Plot": "曲线",
    "Stats": "统计",
    "Output": "输出",
    "Settings": "设置",
    "General": "常规",
    "MuJoCo Visuals": "MuJoCo 可视化",
    "Close": "关闭",
    "Sensors": "传感器",
    "Info": "信息",
    "Application": "应用程序",
    "Language": "语言",
    "UI font": "界面字体",
    "CJK font": "中日韩字体",
    "Rendering": "渲染",
    "Interaction": "交互",
    "Shortcuts": "快捷键",
    "Reset shortcuts": "恢复默认快捷键",
    "A key already in use swaps the two actions": "选择已占用的按键时会交换两个动作",
    "Play / Pause": "播放 / 暂停",
    "Frame selection": "聚焦选中项",
    "Move tool": "移动工具",
    "Rotate tool": "旋转工具",
    "World / Body": "世界 / 本体坐标系",
    "Snap": "吸附",
    "Play": "播放",
    "Pause": "暂停",
    "Step": "单步",
    "Pause to step": "暂停后单步",
    "Stop": "停止",
    "Move": "移动",
    "Rotate": "旋转",
    "Orbit": "环绕",
    "Pan": "平移",
    "Zoom": "缩放",
    "Frame": "聚焦",
    "Type value": "输入数值",
    "Drag": "拖动",
    "Push": "推移",
    "Twist": "扭转",
    "Running": "运行中",
    "Paused": "已暂停",
    "Static": "静态",
    "Time": "时间",
    "Steps": "步数",
    "No selection": "未选择",
    "Click to show steps": "单击显示步数",
    "Click to show time": "单击显示时间",
    "Right-click to copy exact value": "右键复制精确值",
    "Perturb": "物理扰动",
    "Constrain X": "约束 X 轴",
    "Constrain Y": "约束 Y 轴",
    "Constrain Z": "约束 Z 轴",
    "Fly forward": "向前飞行",
    "Fly backward": "向后飞行",
    "Fly right": "向右飞行",
    "Fly left": "向左飞行",
    "Fly up": "向上飞行",
    "Fly down": "向下飞行",
    "Backend": "后端",
    "Graphics device": "图形设备",
    "Debug view": "调试视图",
    "Labels": "标签",
    "Frames": "坐标架",
    "Gizmo style": "Gizmo 样式",
    "Gizmo orientation": "Gizmo 坐标系",
    "Scene helpers": "场景辅助图形",
    "Forge render flags": "Forge 渲染开关",
    "3D gizmo": "3D Gizmo",
    "Use the flat 2D overlay": "使用平面 2D 叠加层",
    "World frame (T)": "世界坐标系（T）",
    "position snap (Shift)": "位置吸附（Shift）",
    "rotation snap (Shift)": "旋转吸附（Shift）",
    "rotation tick scale": "旋转刻度大小",
    "view selection padding": "选中项视图留白",
    "1x is a tight fit; larger values move the view farther away": (
        "1x 为紧密适配；数值越大，视图距离越远"
    ),
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
    "Editor Camera": "编辑器相机",
    "View Camera": "查看相机",
    "Show Camera Preview": "显示相机预览",
    "Lock gizmo while simulation runs": "模拟运行时锁定 Gizmo",
    "gizmo lock": "Gizmo 锁定",
    "camera preview": "相机预览",
    "preview": "预览",
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
    "camera bookmarks": "相机书签",
    "camera bookmark": "相机书签",
    "Stored in": "存储于",
    "Copy all": "全部复制",
    "Copy shown": "复制筛选结果",
    "Copy": "复制",
    "Clear": "清空",
    "Clear search": "清除搜索",
    "Search hierarchy": "搜索层级",
    "Search joints": "搜索关节",
    "Search actuators": "搜索执行器",
    "Name": "名称",
    "Order": "排序",
    "Search output": "搜索输出",
    "Search assets": "搜索资源",
    "Search settings": "搜索设置",
    "messages": "条消息",
    "Copy message": "复制消息",
    "Filter text or component...": "筛选文本或组件…",
    "Filter assets...": "筛选资源…",
    "All levels": "全部级别",
    "No matching joints": "没有匹配的关节",
    "No matching actuators": "没有匹配的执行器",
    "actuators": "执行器",
    "equality": "等式约束",
    "no actuators": "没有执行器",
    "ctrl not produced this frame": "本帧没有控制量",
    "cannot write joint positions": "无法写入关节位置",
    "Pause simulation to edit joint positions": "暂停模拟后才能编辑关节位置",
    "qpos not produced this frame": "本帧没有关节位置数据",
    "no joints": "没有关节",
    "Selected link": "已选连杆",
    "joints total": "个关节",
    "Previous": "上一页",
    "Next": "下一页",
    "page": "页",
    "Using gizmo": "正在使用 Gizmo",
    "Use gizmo": "使用 Gizmo",
    "Choose which direct joint the viewport gizmo edits": "选择视口 Gizmo 要编辑的直属关节",
    "Hinge joint": "旋转关节",
    "Slide joint": "滑动关节",
    "Ball joint": "球形关节",
    "Free joint": "自由关节",
    "dof": "自由度",
    "Mode": "模式",
    "Relative": "相对",
    "Absolute": "绝对",
    "Copy error": "复制错误",
    "Right-click to copy XYZ": "右键复制 XYZ",
    "Click to reset to 0": "单击重置为 0",
    "Read only": "只读",
    "Drag to edit · right-click to copy": "拖动编辑 · 右键复制",
    "Read only · right-click to copy": "只读 · 右键复制",
    "Info and above": "信息及以上",
    "Warnings and errors": "警告和错误",
    "Errors only": "仅错误",
    "orthographic": "正交投影",
    "projection": "投影",
    "persp": "透视",
    "ortho": "正交",
    "Remember precise input choices": "记住精确输入选项",
    "Reuse the last relative/absolute mode and angle unit across editor sessions": (
        "跨编辑器会话沿用上次的相对/绝对模式和角度单位"
    ),
    "position": "位置",
    "target": "目标",
    "up": "上方向",
    "vertical fov": "垂直视场角",
    "near": "近平面",
    "far": "远平面",
    "ortho height": "正交高度",
    "camera view": "相机视图",
    "camera transform": "相机变换",
    "camera projection": "相机投影",
    "camera behavior": "相机行为",
    "light properties": "灯光属性",
    "light transform": "灯光变换",
    "light attenuation": "灯光衰减",
    "light behavior": "灯光行为",
    "enabled": "启用",
    "type": "类型",
    "color": "颜色",
    "intensity": "强度",
    "specular": "高光",
    "ambient": "环境光",
    "texture": "纹理",
    "cast shadow": "投射阴影",
    "position (local)": "位置（局部）",
    "direction (local)": "方向（局部）",
    "range (0 = unlimited)": "范围（0=无限）",
    "range": "范围",
    "0 means unlimited": "0 表示无限",
    "attenuation": "衰减系数",
    "cutoff": "截止角",
    "falloff exponent": "衰减指数",
    "source radius": "光源半径",
    "add a cube texture to illuminate the scene": "添加立方体纹理以照亮场景",
    "appearance": "外观",
    "instance color": "实例颜色",
    "assigned material": "指定材质",
    "inline appearance": "内联外观",
    "asset browser": "资源浏览器",
    "create material": "创建材质",
    "material actions": "材质操作",
    "texture import": "纹理导入",
    "environment textures": "环境纹理",
    "shared material": "共享材质",
    "material properties": "材质属性",
    "base color": "基础颜色",
    "preset": "预设",
    "Presets...": "预设…",
    "emission": "自发光",
    "shininess": "光泽度",
    "reflectance": "反射率",
    "metallic override": "覆盖金属度",
    "metallic": "金属度",
    "roughness override": "覆盖粗糙度",
    "roughness": "粗糙度",
    "texture repeat": "纹理重复",
    "uniform texture scale": "统一纹理缩放",
    "none": "无",
    "Open in Assets": "在资源面板中打开",
    "New material": "新建材质",
    "Duplicate material": "复制材质",
    "Import texture": "导入纹理",
    "Import cube texture": "导入立方体纹理",
    "Import skybox texture": "导入天空盒纹理",
    "Create or assign a shared material to edit its properties": (
        "创建或指定共享材质后才能编辑其属性"
    ),
    "geometry shape and resource": "几何体形状与资源",
    "geometry dimensions": "几何体尺寸",
    "geometry type": "几何体类型",
    "resource": "资源",
    "resource actions": "资源操作",
    "changes": "更改",
    "import": "导入",
    "Import and assign mesh": "导入并指定网格",
    "Import and assign height field": "导入并指定高度场",
    "contact properties": "接触属性",
    "friction (slide spin roll)": "摩擦（滑动 / 旋转 / 滚动）",
    "contact dimension": "接触维度",
    "collision type mask": "碰撞类型掩码",
    "collision affinity mask": "碰撞匹配掩码",
    "contact priority": "接触优先级",
    "contact margin": "接触边距",
    "contact gap": "接触间隙",
    "solver mix": "求解器混合系数",
    "solver reference": "求解器参考参数",
    "impedance min / max / width": "阻抗最小值 / 最大值 / 宽度",
    "impedance midpoint / power": "阻抗中点 / 幂次",
    "adhesion": "黏附力",
    "surface linear velocity": "表面线速度",
    "surface angular velocity": "表面角速度",
    "mass, group, and fluid": "质量、分组与流体",
    "visual group": "可视组",
    "mass source": "质量来源",
    "density": "密度",
    "inertia distribution": "惯量分布",
    "ellipsoid fluid interaction": "椭球流体交互",
    "fluid blunt / slender / angular": "流体系数：钝体 / 细长体 / 角向",
    "fluid Kutta / Magnus": "流体系数：Kutta / Magnus",
    "transform": "变换",
    "rotation": "旋转",
    "velocity": "速度",
    "linear velocity": "线速度",
    "angular velocity": "角速度",
    "body inertial and dynamics": "本体惯性与动力学",
    "inertia mode": "惯性模式",
    "auto from geoms": "由几何体自动计算",
    "diagonal": "对角惯量",
    "full tensor": "完整惯性张量",
    "mass": "质量",
    "inertial position": "惯性位置",
    "inertial rotation": "惯性旋转",
    "diagonal inertia": "对角惯量",
    "tensor xx / yy / zz": "张量 xx / yy / zz",
    "tensor xy / xz / yz": "张量 xy / xz / yz",
    "gravity compensation": "重力补偿",
    "mocap body": "动作捕捉本体",
    "sleep policy": "休眠策略",
    "automatic": "自动",
    "never": "从不",
    "allowed": "允许",
    "initially asleep": "初始休眠",
    "Mass and inertia are derived from attached geoms": "质量和惯量由附属几何体计算",
    "Body-frame rotation of diagonal principal axes; full tensors derive this rotation": (
        "对角主轴相对本体坐标系的旋转；完整张量会自动推导此旋转"
    ),
    "joint properties": "关节属性",
    "advanced joint properties": "高级关节属性",
    "axis": "轴",
    "Axis is expressed in the body frame": "该轴使用本体坐标系表示",
    "limited": "启用限位",
    "limit angle": "限位角度",
    "damping": "阻尼",
    "stiffness": "刚度",
    "qpos address": "qpos 地址",
    "group": "分组",
    "armature": "附加转子惯量",
    "friction loss": "摩擦损耗",
    "reference": "参考位置",
    "spring reference": "弹簧参考位置",
    "limit margin": "限位边距",
    "limit solver reference": "限位求解器参考参数",
    "limit impedance min / max / width": "限位阻抗最小值 / 最大值 / 宽度",
    "limit impedance midpoint / power": "限位阻抗中点 / 幂次",
    "friction solver reference": "摩擦求解器参考参数",
    "friction impedance min / max / width": "摩擦阻抗最小值 / 最大值 / 宽度",
    "friction impedance midpoint / power": "摩擦阻抗中点 / 幂次",
    "actuator force limit": "执行器力限位",
    "actuator force range": "执行器力范围",
    "actuator gravity compensation": "执行器重力补偿",
    "Auto enables the limit when a valid range is authored; unlimited ignores the range": (
        "自动模式在范围有效时启用限位；无限制模式忽略该范围"
    ),
    "MuJoCo sleep policies apply only to movable root bodies": (
        "MuJoCo 休眠策略仅适用于可移动的根本体"
    ),
    "Pause the simulation to edit model body properties": "暂停模拟后才能编辑模型本体属性",
    "Apply": "应用",
    "Revert": "还原",
    "Apply rebuilds the model once": "应用时仅重建一次模型",
    "Joint gizmo": "关节 Gizmo",
    "nothing selected": "未选择任何对象",
    "click an object in the viewport or the Hierarchy panel": "在视口或层级面板中单击对象",
    "node id": "节点 ID",
    "object id": "对象 ID",
    "body": "本体",
    "posable": "可变换",
    "not pickable": "不可拾取",
    "yes": "是",
    "no": "否",
    "no pose this frame": "本帧没有位姿数据",
    "gizmo hidden": "Gizmo 已隐藏",
    "body frame": "本体坐标系",
    "world frame": "世界坐标系",
    "material": "材质",
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


def _read_settings(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _read_language(settings: dict[str, object]) -> Language:
    requested = os.environ.get("FORGE_VIEWER_LANGUAGE")
    if requested:
        try:
            return parse_language(requested)
        except ValueError:
            return Language.ENGLISH
    try:
        value = settings.get("language", Language.ENGLISH)
        return parse_language(value)
    except (ValueError, TypeError):
        return Language.ENGLISH


@dataclass
class Localizer:
    language: Language = Language.ENGLISH
    path: Path | None = None
    preferences: dict[str, object] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Localizer:
        path = settings_path()
        preferences = _read_settings(path)
        return cls(_read_language(preferences), path, preferences)

    def text(self, value: str) -> str:
        if self.language is Language.SIMPLIFIED_CHINESE:
            return _ZH_CN.get(value, value)
        return value

    def set_language(self, value: Language | str, *, persist: bool = True) -> None:
        self.language = parse_language(value)
        if persist:
            self.set_preferences({"language": self.language.value})

    def preference(self, name: str, default: object = None) -> object:
        return self.preferences.get(str(name), default)

    def set_preferences(self, values: dict[str, object], *, persist: bool = True) -> None:
        self.preferences.update({str(name): value for name, value in values.items()})
        if not persist or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.preferences, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
