"""Renderer, visual group, and interaction settings."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.capture import CaptureSurface
from mojive.config import CAMERA_FOCUS_EASINGS, CAMERA_NAVIGATION_RANGES, CameraNavigationConfig
from mojive.types import ContactStyle

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from ...render.backend import DebugView, FrameMode, LabelMode, RenderFlag, ShadowQuality
from ..geometry_view import draw_geometry_view
from ..gizmo import (
    DEFAULT_ROTATION_SNAP_DEG,
    DEFAULT_ROTATION_TICK_SCALE,
    DEFAULT_TRANSLATION_SNAP_M,
)
from ..input_bindings import InputAction, input_action_name, key_choices
from ..localization import LANGUAGE_LABELS, Language, parse_language, render_note_text
from ..perturb import OUTLINE_CORNER_RADIUS_PT
from ..pointer_bindings import NAVIGATION_PRESETS, POINTER_ACTION_NAMES, PointerAction
from ..viewcube import (
    DEFAULT_SELECTION_PADDING,
    MAX_SELECTION_PADDING,
    MIN_SELECTION_PADDING,
)
from ..viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    MAX_VIEWPORT_CAPSULE_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_CAPSULE_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
)
from . import (
    Panel,
    PanelContext,
    padded_selectable,
    pointer_hint,
    pointer_pressed,
    search_input,
    segmented_control,
    themed_checkbox,
)

_RND_FLAGS: tuple[RenderFlag, ...] = (
    RenderFlag.SHADOW,
    RenderFlag.WIREFRAME,
    RenderFlag.REFLECTION,
    RenderFlag.ADDITIVE,
    RenderFlag.SKYBOX,
    RenderFlag.FOG,
    RenderFlag.HAZE,
    RenderFlag.CULL_FACE,
)
_VIS_FLAGS: tuple[RenderFlag, ...] = (
    RenderFlag.CONVEXHULL,
    RenderFlag.TEXTURE,
    RenderFlag.JOINT,
    RenderFlag.ACTUATOR,
    RenderFlag.ACTIVATION,
    RenderFlag.CAMERA,
    RenderFlag.LIGHT,
    RenderFlag.RANGEFINDER,
    RenderFlag.CONSTRAINT,
    RenderFlag.STATIC,
    RenderFlag.SKIN,
    RenderFlag.FLEXFACE,
    RenderFlag.FLEXSKIN,
    RenderFlag.FLEXVERT,
    RenderFlag.FLEXEDGE,
    RenderFlag.CONTACTPOINT,
    RenderFlag.CONTACTFORCE,
    RenderFlag.CONTACTSPLIT,
    RenderFlag.ISLAND,
    RenderFlag.AUTOCONNECT,
    RenderFlag.TENDON,
    RenderFlag.TRANSPARENT,
    RenderFlag.COM,
    RenderFlag.INERTIA,
    RenderFlag.SCLINERTIA,
    RenderFlag.BODYBVH,
    RenderFlag.MESHBVH,
)

_CATEGORIES = ("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals")
_CATEGORY_WIDTH_PT = 132.0
_PAGE_MIN_WIDTH_PT = 224.0
_COLUMN_GAP_PT = 8.0
_CATEGORY_SEARCH_TERMS = {
    "General": ("language", "ui font", "cjk font", "model realtime rebuild apply"),
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
        "perturb corner radius",
        "helpers entities influence volumes",
    ),
    "Rendering": (
        "backend graphics device scene lights shadow casters quality performance balanced high",
        "debug view labels frames",
        "renderer render flags outline tonemap msaa mesh lod detail",
    ),
    "MuJoCo Visuals": (
        "visual groups bvh depth geometry both collision color opacity",
        "mjt rnd flag shadow wireframe reflection additive skybox fog haze cull face",
        "mjt vis flag joint actuator camera light contact force split inertia bvh contact color shape sphere cylinder point 接触点 颜色 样式 球 圆柱",
    ),
}


def settings_category_matches(category: str, query: str) -> bool:
    """Return whether a settings search can be satisfied by one category."""

    tokens = query.casefold().split()
    if not tokens:
        return True
    searchable = " ".join((category, *_CATEGORY_SEARCH_TERMS.get(category, ()))).casefold()
    return all(token in searchable for token in tokens)


def settings_uses_stacked_layout(available_width: float, style_scale: float) -> bool:
    """Keep the settings page usable when scaled columns no longer fit side by side."""

    minimum_width = _CATEGORY_WIDTH_PT + _COLUMN_GAP_PT + _PAGE_MIN_WIDTH_PT
    return float(available_width) < minimum_width * float(style_scale)


def responsive_flag_groups(requested: int, available_width: float, style_scale: float) -> int:
    """Reduce render-flag columns before translated labels start clipping."""

    fit = int(float(available_width) / max(96.0 * float(style_scale), 1.0))
    return max(1, min(int(requested), fit))


def render_flag_label(flag: RenderFlag, translate, *, localized: bool) -> str:
    """Keep official MuJoCo tokens intact and name renderer display controls."""

    if flag == RenderFlag.MESH_LOD:
        return translate("Mesh LOD")
    return translate(flag.value) if localized else flag.value


def flag_groups() -> tuple[tuple[str, tuple[RenderFlag, ...]], ...]:
    rest = tuple(
        f
        for f in RenderFlag
        if f not in _RND_FLAGS
        and f not in _VIS_FLAGS
        and f not in (RenderFlag.VISUAL_GEOMETRY, RenderFlag.COLLISION_GEOMETRY)
    )
    return (
        ("mjtRndFlag", _RND_FLAGS),
        ("mjtVisFlag", _VIS_FLAGS),
        ("renderer", rest),
    )


class SettingsPanel(Panel):
    id = "settings"
    name = "Settings"
    default_open = False
    shortcut = "F9"
    modal = False
    dock_with = "Camera"

    def __init__(self) -> None:
        super().__init__()
        self._pointer_text = {}
        self._pointer_bindings_seen = None
        self._pointer_error = ""
        self._view = DebugView.SHADED
        self._message = ""
        self._category = "General"
        self._search = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        scale = ctx.style_scale
        stacked = settings_uses_stacked_layout(
            imgui.get_content_region_avail().x,
            scale,
        )
        if stacked:
            imgui.set_next_item_width(-1.0)
            if imgui.begin_combo("##settings_category", ctx.tr(self._category)):
                for category in _CATEGORIES:
                    matched = settings_category_matches(category, self._search)
                    imgui.begin_disabled(not matched)
                    selected, _ = padded_selectable(
                        f"{ctx.tr(category)}##settings_{category}",
                        self._category == category,
                    )
                    if selected:
                        self._category = category
                    imgui.end_disabled()
                imgui.end_combo()
            imgui.spacing()
        else:
            self._draw_category_rail(ctx, scale)
            imgui.same_line()

        imgui.begin_child(
            "settings_page",
            imgui.ImVec2(0.0, 0.0),
            imgui.ChildFlags_.always_use_window_padding,
        )
        imgui.set_next_item_width(-1.0)
        changed, self._search = search_input(
            "##settings_search",
            self._search,
            hint=ctx.tr("Search settings"),
            search_tooltip=ctx.tr("Search settings"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed and self._search and not settings_category_matches(self._category, self._search):
            self._category = next(
                (
                    category
                    for category in _CATEGORIES
                    if settings_category_matches(category, self._search)
                ),
                self._category,
            )
        imgui.spacing()
        imgui.text(ctx.tr(self._category))
        imgui.separator()
        if self._search and not settings_category_matches(self._category, self._search):
            imgui.text_disabled(ctx.tr("No matching settings"))
            imgui.end_child()
            return
        if self._category == "General":
            self._general(ctx)
        elif self._category == "Camera":
            self._camera_navigation(ctx)
        elif self._category == "Interaction":
            self._interaction(ctx)
        elif self._category == "Rendering":
            self._rendering(ctx)
        elif self._category == "Recording":
            self._recording(ctx)
        else:
            self._mujoco_visuals(ctx)
        imgui.end_child()

    def _draw_category_rail(self, ctx: PanelContext, scale: float) -> None:
        imgui.begin_child(
            "settings_categories",
            imgui.ImVec2(_CATEGORY_WIDTH_PT * scale, 0.0),
            0,
        )
        imgui.push_style_var(
            imgui.StyleVar_.selectable_text_align,
            imgui.ImVec2(0.5, 0.5),
        )
        for category in _CATEGORIES:
            matched = settings_category_matches(category, self._search)
            imgui.begin_disabled(not matched)
            selected, _ = padded_selectable(
                f"{ctx.tr(category)}##settings_{category}", self._category == category
            )
            if selected:
                self._category = category
            imgui.end_disabled()
        imgui.pop_style_var()
        imgui.end_child()

    def _general(self, ctx: PanelContext) -> None:
        t = ctx.tr
        languages = tuple(Language)
        current = parse_language(ctx.language)
        labels = [LANGUAGE_LABELS[language] for language in languages]
        if not self._begin_properties("settings_general"):
            return
        self._property(t("Language"))
        changed, index = imgui.combo("##ui_language", languages.index(current), labels)
        if changed and ctx.set_language is not None:
            ctx.set_language(languages[index].value)
        self._property(t("Model updates"))
        selected = int(not ctx.live_model_updates)
        value = segmented_control(
            "model-updates",
            (t("Realtime"), t("Deferred")),
            selected,
            theme=ctx.theme,
        )
        if value != selected and ctx.set_live_model_updates is not None:
            ctx.set_live_model_updates(value == 0)
        imgui.set_item_tooltip(
            t("Realtime applies each edit. Deferred edits wait for Apply in the viewport.")
        )
        if ctx.font_report is not None:
            self._property(t("UI font"))
            imgui.text_disabled(ctx.font_report.mono)
            self._property(t("CJK font"))
            imgui.text_disabled(ctx.font_report.cjk or ctx.tr("none"))
        imgui.end_table()
        if ctx.panels is not None:
            self._group_heading(t("Panels"))
            panels = tuple(
                panel
                for panel in ctx.panels
                if panel.enabled and not panel.modal and panel.id != self.id
            )
            if self._begin_toggle_grid("settings_panels", tuple(t(panel.name) for panel in panels)):
                for panel in panels:
                    imgui.table_next_column()
                    changed, is_open = themed_checkbox(
                        f"{t(panel.name)}###panel_open_{panel.id}", panel.open, ctx.theme
                    )
                    if changed:
                        ctx.panels.set_open(panel.id, is_open)
                imgui.end_table()

    def show_category(self, category: str) -> None:
        if category in _CATEGORIES:
            self._category = category
            self._search = ""

    def _camera_navigation(self, ctx: PanelContext) -> None:
        if ctx.camera is None or ctx.set_camera_navigation is None:
            return
        t = ctx.tr
        config = ctx.camera.navigation
        defaults = CameraNavigationConfig()

        def number(name, label, speed, fmt, tooltip):
            nonlocal config
            self._property(t(label))
            low, high = CAMERA_NAVIGATION_RANGES[name]
            if name == "min_distance":
                high = config.max_distance
            elif name == "max_distance":
                low = config.min_distance
            changed, value = imgui.drag_float(
                f"##camera_{name}",
                getattr(config, name),
                speed,
                low,
                high,
                fmt,
                imgui.SliderFlags_.always_clamp,
            )
            committed = imgui.is_item_deactivated_after_edit()
            reset = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.VALUE_RESET)
            if reset:
                changed, value = True, min(high, max(low, getattr(defaults, name)))
            imgui.set_item_tooltip(t(tooltip))
            if changed:
                try:
                    config = replace(config, **{name: value})
                except ValueError as error:
                    ctx.report(str(error))
                else:
                    ctx.set_camera_navigation(config, persist=False)
            if committed or reset:
                ctx.set_camera_navigation(config)

        def choice(name, label, choices):
            nonlocal config
            self._property(t(label))
            current = getattr(config, name)
            if imgui.begin_combo(f"##camera_{name}", dict(choices)[current]):
                for value, text in choices:
                    if imgui.selectable(text, current == value)[0]:
                        config = replace(config, **{name: value})
                        ctx.set_camera_navigation(config)
                imgui.end_combo()

        self._group_heading(t("Focus"))
        if self._begin_properties("settings_camera_focus"):
            number(
                "focus_margin",
                "Focus distance",
                0.01,
                "%.2fx",
                "1x fits the object tightly; larger values leave more space around it.",
            )
            number(
                "focus_duration",
                "Transition duration",
                0.01,
                "%.2f s",
                "Duration of object and joint focus. Set to 0 for an instant transition.",
            )
            choice(
                "focus_easing",
                "Transition curve",
                tuple(
                    zip(
                        CAMERA_FOCUS_EASINGS,
                        ("Linear", "Smoothstep", "Smootherstep", "Ease out cubic"),
                        strict=True,
                    )
                ),
            )
            imgui.end_table()
        self._group_heading(t("Zoom"))
        if self._begin_properties("settings_camera_zoom"):
            choice(
                "zoom_mode", "Zoom mode", (("proportional", "Proportional"), ("linear", "Linear"))
            )
            imgui.set_item_tooltip(
                t(
                    "Proportional slows near the pivot. Linear uses a fixed step based on the framed scene size."
                )
            )
            number(
                "zoom_speed",
                "Zoom speed",
                0.05,
                "%.2fx",
                "Applies to the mouse wheel and zoom dragging.",
            )
            number(
                "min_distance",
                "Minimum distance",
                0.001,
                "%.4g",
                "World units; orthographic zoom uses the equivalent perspective distance.",
            )
            number(
                "max_distance",
                "Maximum distance",
                1.0,
                "%.4g",
                "World units; orthographic zoom uses the equivalent perspective distance.",
            )
            imgui.end_table()
        if imgui.button(t("Reset camera navigation")):
            ctx.set_camera_navigation(defaults)

    def _recording(self, ctx: PanelContext) -> None:
        config = ctx.recording_config
        if config is None:
            return
        if self._begin_properties("settings_recording"):
            self._property(ctx.tr("Copy to clipboard"))
            changed, value = imgui.checkbox("##capture_clipboard", config.copy_to_clipboard)
            imgui.set_item_tooltip(
                ctx.tr("Copy screenshots as images and videos as files after saving.")
            )
            if changed:
                config = replace(config, copy_to_clipboard=value)
                ctx.set_recording_config(config)
            self._property(ctx.tr("Run simulation when recording starts"))
            clock_control = ctx.session.adapter.caps.clock_control
            imgui.begin_disabled(not clock_control)
            changed, value = imgui.checkbox(
                "##recording_run_simulation", config.run_simulation and clock_control
            )
            imgui.end_disabled()
            if not clock_control and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(ctx.tr("Simulation is controlled by the external application."))
            if changed:
                config = replace(config, run_simulation=value)
                ctx.set_recording_config(config)
            self._property(ctx.tr("Countdown (s)"))
            changed, value = imgui.input_float(
                "##recording_delay", config.countdown, 1.0, 5.0, "%.1f"
            )
            imgui.set_item_tooltip(ctx.tr("Set to 0 to start after menus close."))
            if changed:
                config = replace(config, countdown=value)
                ctx.set_recording_config(config)
            self._property(ctx.tr("Take end hold (s)"))
            changed, value = imgui.input_float(
                "##recording_end_hold", config.end_hold, 0.5, 5.0, "%.1f"
            )
            imgui.set_item_tooltip(
                ctx.tr("Take videos stop automatically after holding the final frame.")
            )
            if changed:
                config = replace(config, end_hold=value)
                ctx.set_recording_config(config)
            self._property(ctx.tr("Video frame rate"))
            changed, value = imgui.input_float("##recording_fps", config.fps, 1.0, 10.0, "%.1f")
            if changed:
                config = replace(config, fps=value)
                ctx.set_recording_config(config)
            self._property(ctx.tr("Default capture area"))
            surfaces = tuple(CaptureSurface)
            labels = ("Scene Only", "Viewport with UI", "Entire Window")
            selected = surfaces.index(config.surface)
            if imgui.begin_combo("##recording_surface", ctx.tr(labels[selected])):
                for index, label in enumerate(labels):
                    clicked, _ = imgui.selectable(ctx.tr(label), index == selected)
                    if clicked:
                        config = replace(config, surface=surfaces[index])
                        ctx.set_recording_config(config)
                imgui.end_combo()
            imgui.set_item_tooltip(
                ctx.tr("Viewport recording follows the visibility choices in the Layers panel.")
            )
            imgui.end_table()
        self._group_heading(ctx.tr("Video encoding"))
        if self._begin_properties("settings_video_encoding"):
            self._property(ctx.tr("Rate control"))
            modes = (("quality", "Quality priority"), ("bitrate", "Target bitrate"))
            label = next(label for value, label in modes if value == config.rate_control)
            if imgui.begin_combo("##recording_rate_control", ctx.tr(label)):
                for value, label in modes:
                    if imgui.selectable(ctx.tr(label), value == config.rate_control)[0]:
                        config = replace(config, rate_control=value)
                        ctx.set_recording_config(config)
                imgui.end_combo()
            imgui.set_item_tooltip(ctx.tr("Encoding settings apply to the next recording."))
            if config.rate_control == "quality":
                self._property(ctx.tr("Quality (CRF)"))
                changed, value = imgui.slider_int("##recording_crf", config.crf, 0, 51)
                imgui.set_item_tooltip(ctx.tr("Lower CRF gives higher quality and larger files."))
                if changed:
                    config = replace(config, crf=value)
                    ctx.set_recording_config(config)
            else:
                self._property(ctx.tr("Target bitrate (Mbps)"))
                changed, value = imgui.input_float(
                    "##recording_bitrate", config.bitrate_mbps, 1.0, 5.0, "%.1f"
                )
                imgui.set_item_tooltip(
                    ctx.tr("Target average bitrate; actual bitrate varies with the scene.")
                )
                if changed:
                    config = replace(config, bitrate_mbps=value)
                    ctx.set_recording_config(config)
            self._property(ctx.tr("Encoding speed"))
            presets = (("fast", "Fast"), ("medium", "Balanced"), ("slow", "Slow"))
            label = next(label for value, label in presets if value == config.encoder_preset)
            if imgui.begin_combo("##recording_preset", ctx.tr(label)):
                for value, label in presets:
                    if imgui.selectable(ctx.tr(label), value == config.encoder_preset)[0]:
                        config = replace(config, encoder_preset=value)
                        ctx.set_recording_config(config)
                imgui.end_combo()
            imgui.set_item_tooltip(
                ctx.tr(
                    "Slower encoding spends more CPU time on compression and may slow live recording."
                )
            )
            self._property(ctx.tr("Color sampling"))
            formats = (("yuv420p", "Compatible (4:2:0)"), ("yuv444p", "Full chroma (4:4:4)"))
            label = next(label for value, label in formats if value == config.pixel_format)
            if imgui.begin_combo("##recording_pixel_format", ctx.tr(label)):
                for value, label in formats:
                    if imgui.selectable(ctx.tr(label), value == config.pixel_format)[0]:
                        config = replace(config, pixel_format=value)
                        ctx.set_recording_config(config)
                imgui.end_combo()
            imgui.set_item_tooltip(
                ctx.tr("4:4:4 preserves fine color detail but requires a compatible player.")
            )
            imgui.end_table()

    def _rendering(self, ctx: PanelContext) -> None:
        t = ctx.tr
        caps = ctx.backend.caps
        if caps.shadows:
            self._group_heading(t("Shadows"))
            if self._begin_properties("settings_shadow_quality"):
                self._property(t("Shadow quality"))
                getter = getattr(ctx.backend, "get_shadow_quality", None)
                try:
                    current = (
                        ShadowQuality(getter()) if getter is not None else ShadowQuality.BALANCED
                    )
                except (TypeError, ValueError):
                    current = ShadowQuality.BALANCED
                qualities = tuple(ShadowQuality)
                imgui.begin_group()
                index = segmented_control(
                    "shadow-quality",
                    tuple(t(label) for label in ("Performance", "Balanced", "High")),
                    qualities.index(current),
                    theme=ctx.theme,
                )
                imgui.end_group()
                imgui.set_item_tooltip(
                    t("Higher quality smooths close-up shadow edges but costs more GPU time")
                )
                selected = qualities[index]
                if selected is not current and ctx.set_shadow_quality is not None:
                    ctx.set_shadow_quality(selected)
                imgui.end_table()

        renderer_flags = flag_groups()[-1][1]
        if renderer_flags:
            self._group_heading(t("Renderer flags"))
            self._flag_table(ctx, "renderer_flags", renderer_flags, groups=3)

        flags = imgui.TreeNodeFlags_.default_open if self._search else 0
        if imgui.collapsing_header(
            f"{t('Debug')}###render_debug", flags
        ) and self._begin_properties("settings_render_debug"):
            self._property(t("Debug view"))
            self._debug_view(ctx)
            self._property(t("Labels"))
            self._label_mode(ctx)
            self._property(t("Frames"))
            self._frame_mode(ctx)
            imgui.end_table()

        self._group_heading(t("Backend info"))
        if self._begin_properties("settings_render_backend"):
            self._property(t("Backend"))
            imgui.text_disabled(caps.name)
            if caps.gl_version:
                self._property(t("Graphics device"))
                imgui.text_wrapped(f"{caps.gl_version}  {caps.renderer}")
            light_notes = ctx.backend.stats.notes
            for name in ("scene lights", "shadow casters"):
                if name in light_notes:
                    self._property(t(name))
                    imgui.text_disabled(render_note_text(light_notes[name], t))
            imgui.end_table()

    def _interaction(self, ctx: PanelContext) -> None:
        t = ctx.tr
        if ctx.interactions is not None:
            self._interaction_policy(ctx)
        if ctx.selection_style is not None:
            self._selection_presentation(ctx)
        if ctx.gizmo is not None:
            self._group_heading(t("Gizmo"))
            if self._begin_properties("settings_interaction_gizmo"):
                self._property(t("Style"))
                style_index = segmented_control(
                    "gizmo-style",
                    ("2D", "3D"),
                    1 if ctx.gizmo.style == "3d" else 0,
                    theme=ctx.theme,
                )
                ctx.gizmo.set_style("3d" if style_index == 1 else "2d")
                self._property(t("Orientation"))
                frame_index = segmented_control(
                    "gizmo-frame",
                    (t("Body"), t("World")),
                    1 if ctx.gizmo.space == "world" else 0,
                    theme=ctx.theme,
                )
                ctx.gizmo.set_space("world" if frame_index == 1 else "body")
                self._property(t("Overlay size"))
                changed, overlay_scale = imgui.drag_float(
                    "##viewport_overlay_scale",
                    float(ctx.viewport_overlay_scale),
                    0.02,
                    MIN_VIEWPORT_OVERLAY_SCALE,
                    MAX_VIEWPORT_OVERLAY_SCALE,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                committed = imgui.is_item_deactivated_after_edit()
                reset = False
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed = True
                    reset = True
                    overlay_scale = DEFAULT_VIEWPORT_OVERLAY_SCALE
                if hovered:
                    imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
                if (changed or committed or reset) and ctx.set_viewport_overlay_scale is not None:
                    ctx.set_viewport_overlay_scale(
                        overlay_scale,
                        persist=committed or reset,
                    )
                if ctx.viewport_overlays is not None:
                    for name, label, value in (
                        ("playback", "Capsule size", ctx.viewport_overlays.playback_scale),
                    ):
                        self._property(t(label))
                        changed, relative_scale = imgui.drag_float(
                            f"##viewport_{name}_scale",
                            float(value),
                            0.02,
                            MIN_VIEWPORT_CAPSULE_SCALE,
                            MAX_VIEWPORT_CAPSULE_SCALE,
                            "%.2fx",
                        )
                        hovered = imgui.is_item_hovered()
                        committed = imgui.is_item_deactivated_after_edit()
                        reset = False
                        if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                            changed, relative_scale, reset = True, 1.0, True
                        if hovered:
                            imgui.set_tooltip(
                                pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset"))
                            )
                        if (
                            changed or committed or reset
                        ) and ctx.set_viewport_capsule_scale is not None:
                            ctx.set_viewport_capsule_scale(
                                name,
                                relative_scale,
                                persist=committed or reset,
                            )
                    self._property(t("Status duration"))
                    changed, duration = imgui.slider_float(
                        "##viewport_status_duration",
                        ctx.viewport_overlays.status_duration,
                        3.0,
                        5.0,
                        "%.1f s",
                    )
                    if changed and ctx.set_viewport_overlays is not None:
                        ctx.set_viewport_overlays(
                            replace(ctx.viewport_overlays, status_duration=duration), persist=True
                        )
                    self._property(t("Movable capsules"))
                    changed, movable = themed_checkbox(
                        "##viewport_capsules_movable",
                        bool(ctx.viewport_overlays.movable),
                        ctx.theme,
                    )
                    if changed and ctx.set_viewport_overlays is not None:
                        ctx.set_viewport_overlays(
                            replace(ctx.viewport_overlays, movable=movable), persist=True
                        )
                imgui.end_table()

            self._group_heading(t("Input"))
            if self._begin_properties("settings_interaction_input"):
                self._property(t("Keep mode/unit"))
                changed, remember = themed_checkbox(
                    "##remember_precise_input_choices",
                    bool(ctx.gizmo.remember_precise_input_choices),
                    ctx.theme,
                )
                imgui.set_item_tooltip(
                    t("Reuse the last relative/absolute mode and angle unit across editor sessions")
                )
                if changed:
                    if ctx.set_precise_input_memory is not None:
                        ctx.set_precise_input_memory(remember)
                    else:
                        ctx.gizmo.remember_precise_input_choices = remember
                imgui.end_table()

            if ctx.input_bindings is not None:
                self._shortcut_settings(ctx)

            self._group_heading(t("Snap · Shift"))
            if self._begin_properties("settings_interaction_snap"):
                self._property(t("Position"))
                changed, step = imgui.drag_float(
                    "##position_snap",
                    float(ctx.gizmo.translation_snap_m),
                    0.01,
                    0.01,
                    100.0,
                    "%.3f m",
                )
                hovered = imgui.is_item_hovered()
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed, step = True, DEFAULT_TRANSLATION_SNAP_M
                if hovered:
                    imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
                if changed:
                    ctx.gizmo.translation_snap_m = step

                self._property(t("Rotation"))
                changed, step = imgui.drag_float(
                    "##rotation_snap",
                    float(ctx.gizmo.rotation_snap_deg),
                    0.1,
                    0.5,
                    180.0,
                    "%.1f deg",
                )
                hovered = imgui.is_item_hovered()
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed, step = True, DEFAULT_ROTATION_SNAP_DEG
                if hovered:
                    imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
                if changed:
                    ctx.gizmo.rotation_snap_deg = step

                self._property(t("Tick scale"))
                changed, tick_scale = imgui.drag_float(
                    "##rotation_tick_scale",
                    float(ctx.gizmo.rotation_tick_scale),
                    0.05,
                    0.5,
                    3.0,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed, tick_scale = True, DEFAULT_ROTATION_TICK_SCALE
                if hovered:
                    imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
                if changed:
                    ctx.gizmo.rotation_tick_scale = tick_scale
                imgui.end_table()

        if ctx.view_cube is not None:
            self._group_heading(t("View"))
            if self._begin_properties("settings_interaction_view"):
                self._property(t("Padding"))
                changed, padding = imgui.drag_float(
                    "##view_selection_padding",
                    float(ctx.view_cube.selection_padding),
                    0.02,
                    MIN_SELECTION_PADDING,
                    MAX_SELECTION_PADDING,
                    "%.2fx",
                )
                hovered = imgui.is_item_hovered()
                committed = imgui.is_item_deactivated_after_edit()
                reset = False
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed, padding = True, DEFAULT_SELECTION_PADDING
                    reset = True
                if hovered:
                    imgui.set_tooltip(
                        t("1x is a tight fit; larger values move the view farther away")
                    )
                if changed:
                    ctx.view_cube.selection_padding = padding
                if ctx.set_view_selection_padding is not None and (committed or reset):
                    ctx.set_view_selection_padding(ctx.view_cube.selection_padding)
                imgui.end_table()

        if ctx.perturb is not None:
            self._group_heading(t("Perturb"))
            if self._begin_properties("settings_interaction_perturb"):
                self._property(t("Corner radius"))
                changed, radius = imgui.drag_float(
                    "##perturb_corner_radius",
                    float(ctx.perturb.outline_corner_radius_pt),
                    0.1,
                    0.0,
                    24.0,
                    "%.1f px",
                )
                hovered = imgui.is_item_hovered()
                if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                    changed = True
                    radius = OUTLINE_CORNER_RADIUS_PT
                if hovered:
                    imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
                if changed:
                    ctx.perturb.outline_corner_radius_pt = radius
                imgui.end_table()

        if ctx.scene_entities is not None:
            self._group_heading(t("Helpers"))
            if self._begin_properties("settings_interaction_helpers"):
                self._property(t("Camera & light icons"))
                changed, visible = themed_checkbox(
                    "##scene_entity_helpers",
                    ctx.scene_entities.visible,
                    ctx.theme,
                )
                if changed:
                    ctx.scene_entities.visible = visible
                self._property(t("Selected frustum / light range"))
                imgui.begin_disabled(not ctx.scene_entities.visible)
                changed, influence = themed_checkbox(
                    "##selected_influence_volumes",
                    ctx.scene_entities.show_influence,
                    ctx.theme,
                )
                imgui.end_disabled()
                if changed:
                    ctx.scene_entities.show_influence = influence
                imgui.end_table()

    def _interaction_policy(self, ctx: PanelContext) -> None:
        t = ctx.tr
        config = ctx.interactions
        self._group_heading(t("Built-in interactions"))

        def update_top(attribute: str, value: bool) -> None:
            nonlocal config
            config = replace(config, **{attribute: value})
            ctx.interactions = config
            if ctx.set_interactions is not None:
                ctx.set_interactions(config)

        def update_camera(attribute: str, value: bool) -> None:
            nonlocal config
            config = replace(config, camera=replace(config.camera, **{attribute: value}))
            ctx.interactions = config
            if ctx.set_interactions is not None:
                ctx.set_interactions(config)

        def update_selection(attribute: str, value: bool) -> None:
            nonlocal config
            config = replace(
                config,
                selection=replace(config.selection, **{attribute: value}),
            )
            ctx.interactions = config
            if ctx.set_interactions is not None:
                ctx.set_interactions(config)

        rows = (
            ("Camera orbit", config.camera.orbit, update_camera, "orbit"),
            ("Camera pan", config.camera.pan, update_camera, "pan"),
            ("Camera dolly", config.camera.dolly, update_camera, "dolly"),
            ("Camera fly keys", config.camera.fly, update_camera, "fly"),
            ("View cube", config.camera.view_cube, update_camera, "view_cube"),
            ("Scene picking", config.selection.pick, update_selection, "pick"),
            (
                "Clear selection on empty click",
                config.selection.clear_on_empty,
                update_selection,
                "clear_on_empty",
            ),
            (
                "Clear selection with Escape",
                config.selection.clear_with_escape,
                update_selection,
                "clear_with_escape",
            ),
            (
                "Focus on double-click",
                config.selection.focus_on_double_click,
                update_selection,
                "focus_on_double_click",
            ),
            (
                "Pick when focusing viewport",
                config.selection.pick_on_focus,
                update_selection,
                "pick_on_focus",
            ),
            ("Gizmo input", config.gizmo, update_top, "gizmo"),
            ("Physics perturbation", config.perturb, update_top, "perturb"),
            (
                "Playback shortcuts",
                config.playback_shortcuts,
                update_top,
                "playback_shortcuts",
            ),
            ("Panel shortcuts", config.panel_shortcuts, update_top, "panel_shortcuts"),
        )
        if not self._begin_toggle_grid(
            "settings_builtin_interactions", tuple(t(row[0]) for row in rows)
        ):
            return
        for label, current, callback, attribute in rows:
            if attribute == "perturb" and not ctx.session.adapter.caps.perturb:
                continue
            imgui.table_next_column()
            changed, value = themed_checkbox(
                f"{t(label)}###interaction_{attribute}", current, ctx.theme
            )
            if changed:
                callback(attribute, value)
        imgui.end_table()

    def _selection_presentation(self, ctx: PanelContext) -> None:
        t = ctx.tr
        style = ctx.selection_style
        self._group_heading(t("Selection presentation"))
        rows = (
            ("highlight", "Highlight fill"),
            ("outline", "Outline"),
            ("gizmo", "Gizmo"),
            ("frame", "Coordinate frame"),
            ("label", "Name label"),
            ("bounds", "Bounds"),
        )
        if not self._begin_toggle_grid(
            "settings_selection_presentation", tuple(t(label) for _, label in rows)
        ):
            return
        for attribute, label in rows:
            imgui.table_next_column()
            changed, value = themed_checkbox(
                f"{t(label)}###selection_{attribute}", getattr(style, attribute), ctx.theme
            )
            if changed:
                style = replace(style, **{attribute: value})
                ctx.selection_style = style
                if ctx.set_selection_style is not None:
                    ctx.set_selection_style(style)
        imgui.end_table()

    def _shortcut_settings(self, ctx: PanelContext) -> None:
        """Draw conflict-free viewport key remapping from the shared map."""

        t = ctx.tr
        choices = key_choices()
        labels = tuple(t(choice.label) for choice in choices)
        identifiers = tuple(choice.identifier for choice in choices)
        self._group_heading(t("Shortcuts"))
        if not self._begin_properties("settings_interaction_shortcuts"):
            return
        for action in InputAction:
            if action is InputAction.PERTURB and not ctx.session.adapter.caps.perturb:
                continue
            self._property(t(input_action_name(action)))
            current = ctx.input_bindings.key_id(action)
            index = identifiers.index(current)
            changed, index = imgui.combo(
                f"##shortcut_{action.value}",
                index,
                labels,
            )
            imgui.set_item_tooltip(t("A key already in use swaps the two actions"))
            if changed and ctx.set_input_binding is not None:
                try:
                    ctx.set_input_binding(action, identifiers[index])
                    self._pointer_error = ""
                except ValueError as error:
                    self._pointer_error = str(error)
        self._property("")
        if imgui.button(t("Reset shortcuts")) and ctx.reset_input_bindings is not None:
            ctx.reset_input_bindings()
        imgui.end_table()
        self._mouse_shortcuts(ctx)

    def _mouse_shortcuts(self, ctx: PanelContext) -> None:
        t = ctx.tr
        self._group_heading(t("Mouse gestures"))
        if ctx.input_bindings is not self._pointer_bindings_seen:
            self._pointer_bindings_seen = ctx.input_bindings
            self._pointer_text = {
                action: "; ".join(
                    chord.identifier() for chord in ctx.input_bindings.pointer_chords(action)
                )
                for action in PointerAction
            }
        if self._begin_properties("settings_navigation_preset"):
            self._property(t("Navigation preset"))
            if imgui.begin_combo("##navigation_preset", t("Choose preset...")):
                for name in NAVIGATION_PRESETS:
                    if imgui.selectable(name, False)[0] and ctx.set_navigation_preset is not None:
                        try:
                            ctx.set_navigation_preset(name)
                            self._pointer_error = ""
                        except ValueError as error:
                            self._pointer_error = str(error)
                imgui.end_combo()
            imgui.end_table()
        imgui.text_wrapped(
            t(
                "Separate alternatives with ;. Example: alt+left; middle. Add :double for a double-click. Press Enter to apply; empty unbinds."
            )
        )
        if self._begin_properties("settings_mouse_gestures"):
            for action in PointerAction:
                if action.value.startswith("perturb.") and not ctx.session.adapter.caps.perturb:
                    continue
                self._property(t(POINTER_ACTION_NAMES[action]))
                submitted, value = imgui.input_text(
                    f"##mouse_{action.value}",
                    self._pointer_text[action],
                    imgui.InputTextFlags_.enter_returns_true,
                )
                self._pointer_text[action] = value
                if submitted and ctx.set_pointer_binding is not None:
                    try:
                        ctx.set_pointer_binding(
                            action, tuple(part.strip() for part in value.split(";") if part.strip())
                        )
                        self._pointer_error = ""
                    except ValueError as error:
                        self._pointer_error = str(error)
            imgui.end_table()
        if self._pointer_error:
            imgui.text_wrapped(self._pointer_error)

    def _mujoco_visuals(self, ctx: PanelContext) -> None:
        imgui.text(ctx.tr("Geometry view"))
        draw_geometry_view(ctx.backend, ctx.tr, compact=True, theme=ctx.theme)
        self._geometry_style(ctx)
        self._contact_style(ctx)
        self._visual_groups(ctx)
        self._bvh_depth(ctx)

        for title, flags in flag_groups()[:2]:
            if not flags:
                continue
            if not imgui.collapsing_header(title, imgui.TreeNodeFlags_.default_open):
                continue
            self._flag_table(ctx, f"settings_{title}", flags, translate_labels=False)

        if self._message:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._message)

    def _contact_style(self, ctx: PanelContext) -> None:
        t = ctx.tr
        if not imgui.collapsing_header(
            f"{t('Contact points')}###contact_style", imgui.TreeNodeFlags_.default_open
        ):
            return
        style = ctx.backend.get_contact_style()
        if self._begin_properties("contact_style_properties"):
            self._property(t("Show contact points"))
            changed, enabled = themed_checkbox(
                "##contact_enabled", ctx.backend.get_flag(RenderFlag.CONTACTPOINT), ctx.theme
            )
            if changed:
                ctx.backend.set_flag(RenderFlag.CONTACTPOINT, enabled)
            self._property(t("Shape"))
            if imgui.begin_combo("##contact_shape", style.shape.title()):
                for shape in ("cylinder", "sphere", "point"):
                    if imgui.selectable(shape.title(), style.shape == shape)[0]:
                        style = replace(style, shape=shape)
                imgui.end_combo()
            self._property(t("Use model color"))
            changed, use_model_color = themed_checkbox(
                "##contact_model_color", style.use_model_color, ctx.theme
            )
            if changed:
                style = replace(style, use_model_color=use_model_color)
            self._property(t("Color"))
            imgui.begin_disabled(style.use_model_color)
            changed, color = imgui.color_edit4("##contact_color", style.color)
            imgui.end_disabled()
            if changed:
                style = replace(style, color=tuple(color))
            imgui.set_item_tooltip(t("Island visualization overrides contact colors."))
            self._property(t("Size"))
            changed, scale = imgui.slider_float(
                "##contact_scale", style.scale, 0.1, 10, "%.2fx", imgui.SliderFlags_.always_clamp
            )
            if changed:
                try:
                    style = replace(style, scale=scale)
                except ValueError as error:
                    ctx.report(str(error))
            imgui.end_table()
        if imgui.button(t("Reset contact style")):
            style = ContactStyle()
        if style != ctx.backend.get_contact_style():
            setter = ctx.set_contact_style or ctx.backend.set_contact_style
            setter(style)

    def _geometry_style(self, ctx: PanelContext) -> None:
        if not imgui.collapsing_header(
            f"{ctx.tr('Both appearance')}###geometry_style", imgui.TreeNodeFlags_.default_open
        ):
            return
        style = ctx.backend.get_geometry_style()
        if self._begin_properties("geometry_style_properties"):
            self._property(ctx.tr("Collision color"))
            changed, color = imgui.color_edit3("##collision_color", style.collision_color)
            if changed:
                style = replace(style, collision_color=tuple(color))
            for name, label in (
                ("visual_opacity", "Visual opacity"),
                ("collision_opacity", "Collision opacity"),
            ):
                self._property(ctx.tr(label))
                changed, opacity = imgui.slider_float(
                    f"##{name}", getattr(style, name), 0.0, 1.0, "%.2f"
                )
                if changed:
                    style = replace(style, **{name: opacity})
            imgui.end_table()
        if style != ctx.backend.get_geometry_style():
            setter = ctx.set_geometry_style or ctx.backend.set_geometry_style
            setter(style)

    @staticmethod
    def _begin_toggle_grid(str_id: str, labels: tuple[str, ...]) -> bool:
        style = imgui.get_style()
        required = max((imgui.calc_text_size(label).x for label in labels), default=0.0)
        required += (
            imgui.get_frame_height() + style.item_inner_spacing.x + 2.0 * style.cell_padding.x
        )
        columns = max(1, min(4, int(imgui.get_content_region_avail().x / max(1.0, required))))
        return imgui.begin_table(str_id, columns, imgui.TableFlags_.sizing_stretch_same)

    @staticmethod
    def _begin_properties(str_id: str) -> bool:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.pad_outer_x
        compact = imgui.get_content_region_avail().x < 24.0 * imgui.get_font_size()
        if not imgui.begin_table(str_id, 1 if compact else 2, flags):
            return False
        if not compact:
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch.value, 0.36)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value, 0.64)
        return True

    @staticmethod
    def _group_heading(title: str) -> None:
        imgui.spacing()
        imgui.text_disabled(title)
        imgui.separator()

    @staticmethod
    def _property(label: str) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        compact = imgui.table_get_column_count() == 1
        if not compact:
            imgui.align_text_to_frame_padding()
        width = imgui.get_content_region_avail().x
        text_width = imgui.calc_text_size(label).x
        if not compact:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, width - text_width))
        imgui.text(label)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)

    def _visual_groups(self, ctx: PanelContext) -> None:
        groups = ctx.session.visual_groups()
        if not groups:
            return
        if imgui.collapsing_header(
            f"{ctx.tr('visual groups')}###visual_groups", imgui.TreeNodeFlags_.default_open
        ):
            flags = imgui.TableFlags_.sizing_stretch_same
            if imgui.begin_table("mujoco_visual_groups", 7, flags):
                imgui.table_setup_column(
                    "category",
                    imgui.TableColumnFlags_.width_fixed,
                    84.0 * ctx.style_scale,
                )
                for index in range(6):
                    imgui.table_setup_column(str(index), imgui.TableColumnFlags_.width_stretch, 1.0)
                row_height = imgui.get_frame_height()

                def centered_text(value: str, *, disabled: bool = False) -> None:
                    imgui.align_text_to_frame_padding()
                    available = imgui.get_content_region_avail().x
                    text_width = imgui.calc_text_size(value).x
                    imgui.set_cursor_pos_x(
                        imgui.get_cursor_pos_x() + max(0.0, (available - text_width) * 0.5)
                    )
                    if disabled:
                        imgui.text_disabled(value)
                    else:
                        imgui.text(value)

                imgui.table_next_row(0, row_height)
                for label in ("category", "0", "1", "2", "3", "4", "5"):
                    imgui.table_next_column()
                    centered_text(label, disabled=True)
                for family in groups:
                    imgui.table_next_row(0, row_height)
                    imgui.table_next_column()
                    centered_text(family.category)
                    for i, visible in enumerate(family.visible):
                        imgui.table_next_column()
                        available = imgui.get_content_region_avail().x
                        checkbox_size = imgui.get_frame_height()
                        imgui.set_cursor_pos_x(
                            imgui.get_cursor_pos_x() + max(0.0, (available - checkbox_size) * 0.5)
                        )
                        changed, value = themed_checkbox(
                            f"##visual_group_{family.category}_{i}", visible, ctx.theme
                        )
                        if changed:
                            ctx.submit(cmd.SetVisualGroup(family.category, i, value))
                imgui.end_table()
        imgui.separator()

    def _bvh_depth(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        if not (
            backend.caps.supports(RenderFlag.BODYBVH) or backend.caps.supports(RenderFlag.MESHBVH)
        ):
            return
        if self._begin_properties("settings_bvh_depth"):
            self._property(ctx.tr("BVH depth"))
            changed, depth = imgui.drag_int(
                "##bvh_depth", backend.get_bvh_depth(), 1.0, 0, 64, "%d"
            )
            hovered = imgui.is_item_hovered()
            if hovered and pointer_pressed(ctx, PointerAction.VALUE_RESET):
                changed, depth = True, 0
            if hovered:
                imgui.set_tooltip(pointer_hint(ctx, PointerAction.VALUE_RESET, ctx.tr("Reset")))
            if changed:
                backend.set_bvh_depth(depth)
            imgui.end_table()
        imgui.separator()

    def _flag_row(self, ctx: PanelContext, flag: RenderFlag, display_label: str) -> None:
        caps = ctx.backend.caps
        supported = caps.supports(flag)
        imgui.begin_disabled(not supported)
        changed, value = themed_checkbox(
            f"##rf_{flag.value}", ctx.backend.get_flag(flag), ctx.theme
        )
        imgui.end_disabled()
        if not supported:
            imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{display_label}”")
            return
        if flag == RenderFlag.MESH_LOD:
            imgui.set_item_tooltip(
                ctx.tr("Adaptive display detail; depth and picking use the displayed mesh")
            )
        if changed and not ctx.backend.set_flag(flag, value):
            self._message = (
                f"{display_label} · {ctx.tr('backend refused the change')} ({caps.name})"
            )

    def _flag_table(
        self,
        ctx: PanelContext,
        table_id: str,
        flags: tuple[RenderFlag, ...],
        groups: int = 2,
        *,
        translate_labels: bool = True,
    ) -> None:
        groups = responsive_flag_groups(
            groups,
            imgui.get_content_region_avail().x,
            ctx.style_scale,
        )
        table_flags = imgui.TableFlags_.sizing_stretch_prop
        if not imgui.begin_table(table_id, groups * 2, table_flags):
            return
        for group in range(groups):
            imgui.table_setup_column(f"label {group}", imgui.TableColumnFlags_.width_stretch, 1.0)
            imgui.table_setup_column(
                f"value {group}",
                imgui.TableColumnFlags_.width_fixed,
                28.0 * ctx.style_scale,
            )
        row_count = (len(flags) + groups - 1) // groups
        for row in range(row_count):
            imgui.table_next_row()
            for group in range(groups):
                index = group * row_count + row
                imgui.table_next_column()
                if index >= len(flags):
                    imgui.table_next_column()
                    continue
                flag = flags[index]
                imgui.align_text_to_frame_padding()
                width = imgui.get_content_region_avail().x
                display_label = render_flag_label(
                    flag,
                    ctx.tr,
                    localized=translate_labels,
                )
                label_width = imgui.calc_text_size(display_label).x
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, width - label_width))
                imgui.text(display_label)
                imgui.table_next_column()
                self._flag_row(ctx, flag, display_label)
        imgui.end_table()

    def current_view(self, backend) -> DebugView:
        getter = getattr(backend, "get_debug_view", None)
        return getter() if callable(getter) else self._view

    def _debug_view(self, ctx: PanelContext) -> None:
        caps = ctx.backend.caps
        current = self.current_view(ctx.backend)
        imgui.set_next_item_width(-1)
        if not imgui.begin_combo("##debugview", current.value):
            return

        for view in DebugView:
            ok = (view in caps.debug_views) if caps.debug_views else False
            imgui.begin_disabled(not ok)
            selected, _ = padded_selectable(view.value, view is current)
            imgui.end_disabled()
            if not ok:
                imgui.set_item_tooltip(f"{caps.name} {ctx.tr('does not implement')} “{view.value}”")
            elif selected:
                if ctx.backend.set_debug_view(view):
                    self._view = view
                    self._message = ""
                else:
                    self._message = (
                        f"{view.value} · {ctx.tr('backend refused the change')} ({caps.name})"
                    )
        imgui.end_combo()

    def _label_mode(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        label = backend.get_label_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##label_mode", label.value):
            for mode in LabelMode:
                supported = mode in backend.caps.label_modes
                imgui.begin_disabled(not supported)
                selected, _ = padded_selectable(mode.value, mode is label)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_label_mode(mode)
            imgui.end_combo()

    def _frame_mode(self, ctx: PanelContext) -> None:
        backend = ctx.backend
        frame = backend.get_frame_mode()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##frame_mode", frame.value):
            for mode in FrameMode:
                supported = mode in backend.caps.frame_modes
                imgui.begin_disabled(not supported)
                selected, _ = padded_selectable(mode.value, mode is frame)
                imgui.end_disabled()
                if selected and supported:
                    backend.set_frame_mode(mode)
            imgui.end_combo()
