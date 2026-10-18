"""Settings navigation, search terms and responsive category layout."""

_CATEGORIES = ("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals")

_CATEGORY_WIDTH_PT = 132.0

_PAGE_MIN_WIDTH_PT = 224.0

_COLUMN_GAP_PT = 8.0

_CATEGORY_SEARCH_TERMS = {
    "General": (
        "language",
        "ui font",
        "cjk font",
        "numeric output typography 字体 字号 数值 输出",
        "model realtime rebuild apply",
        "scrollbar hide delay 滚动条 隐藏 延迟",
    ),
    "Camera": (
        "focus distance margin transition duration easing curve zoom speed mode minimum maximum limits 相机 聚焦 距离 过渡 时长 曲线 缩放 速度 上限 下限",
    ),
    "Recording": (
        "video capture clipboard copy countdown delay frame rate fps viewport layers quality crf bitrate encoding chroma",
    ),
    "Interaction": (
        "gizmo",
        "built-in interactions camera orbit pan dolly fly view cube",
        "scene picking selection focus empty click escape",
        "playback panel shortcuts physics perturbation",
        "selection presentation highlight outline coordinate frame label bounds",
        "style",
        "orientation",
        "overlay size",
        "shortcuts key bindings remap reset",
        "reuse mode unit",
        "snap position rotation tick scale",
        "view selection padding",
        "perturb corner radius drag force twist torque strength scale 力倍率 力矩倍率 扰动",
        "helpers entities influence volumes",
    ),
    "Rendering": (
        "backend graphics device scene lights shadow casters quality performance balanced high",
        "debug view labels frames",
        "renderer render flags outline tonemap msaa mesh lod detail",
        "viewport appearance editor ground scene low contrast plane 地面 外观 低对比 平面",
    ),
    "MuJoCo Visuals": (
        "content visualization 内容 可视化 visual groups bvh depth geometry both collision color opacity",
        "mjt rnd flag shadow wireframe reflection additive skybox fog haze cull face",
        "mjt vis flag joint actuator camera light contact force split inertia bvh contact color shape sphere cylinder point 接触点 颜色 样式 球 圆柱",
    ),
}


def settings_category_label(category: str) -> str:
    """Keep persisted group identities stable while exposing backend-neutral labels."""
    return "Content visualization" if category == "MuJoCo Visuals" else category


def settings_category_matches(category: str, query: str) -> bool:
    """Return whether a settings search can be satisfied by one category."""

    tokens = query.casefold().split()
    if not tokens:
        return True
    searchable = " ".join(
        (category, settings_category_label(category), *_CATEGORY_SEARCH_TERMS.get(category, ()))
    ).casefold()
    return all(token in searchable for token in tokens)


def settings_uses_stacked_layout(available_width: float, style_scale: float) -> bool:
    """Keep the settings page usable when scaled columns no longer fit side by side."""

    minimum_width = _CATEGORY_WIDTH_PT + _COLUMN_GAP_PT + _PAGE_MIN_WIDTH_PT
    return float(available_width) < minimum_width * float(style_scale)
