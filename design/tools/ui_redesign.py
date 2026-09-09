"""Interactive layout candidates; all edits remain in the feasibility session."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import pi, sqrt

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.curves2d import CORNER_SMOOTHING, smooth_line_cap, smooth_polygon_corners
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import MeshShape
from mojive.ui.draw2d import ImguiDraw2D
from mojive.ui.localization import Localizer, parse_language
from mojive.ui.panels import PanelContext
from mojive.ui.panels.inspector import InspectorPanel
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets import (
    OVERLAY_GEOMETRY,
    PLAYBACK_HALF_HEIGHT_PT,
    PLAYBACK_RESET_SCALE,
    TOOL_GLYPH_SCALE,
    draw_playback_glyph,
    draw_tool_glyph,
)

if __package__:
    from .ui_capsule_geometry import capsule_layout, draw_capsule_shell
else:
    from ui_capsule_geometry import capsule_layout, draw_capsule_shell

RESET_GLYPH_SCALE = 0.88
# Match the stop square's nominal area for comparable visual weight across recording states.
RECORD_GLYPH_RADIUS = 2 * PLAYBACK_HALF_HEIGHT_PT * PLAYBACK_RESET_SCALE / sqrt(pi)

TRANSLATIONS = {
    "Overview": "总览",
    "Inspector": "属性",
    "Control": "控制",
    "Keys": "快捷键",
    "Geometry": "几何规范",
    "Selected": "已选中",
    "No selection": "未选中",
    "Move": "移动",
    "Rotate": "旋转",
    "Dimensions": "尺寸",
    "World": "世界系",
    "Body": "体系",
    "Snap": "吸附",
    "Previous frame": "步退",
    "Step": "步进",
    "Play": "播放",
    "Pause": "暂停",
    "Reset simulation": "复位仿真",
    "Record take": "录制 Take",
    "Stop recording": "停止录制",
    "Recording options": "录制选项",
    "Take": "Take",
    "Video": "视频",
    "Record video": "录制视频",
    "Recording settings": "录制设置",
    "Close": "关闭",
    "Frame": "帧",
    "Paused": "已暂停",
    "Running": "运行中",
    "Hold RMB to fly": "按住右键飞行",
    "Snap locked": "吸附已锁定",
    "Snap held": "按住吸附",
    "Snap off": "吸附关闭",
    "Scene overview": "场景概览",
    "Select the object to edit its properties.": "选中对象以编辑属性。",
    "Select object": "选中对象",
    "Name": "名称",
    "Cancel": "取消",
    "Transform": "变换",
    "Position": "位置",
    "Rotation": "旋转",
    "Material": "材质",
    "Color": "颜色",
    "Restore initial": "恢复初始值",
    "Set zero": "归零",
    "Copy IDs": "复制 ID",
    "Details": "详情",
    "Search actuators": "搜索执行器",
    "Actuators": "执行器",
    "No matching actuators": "没有匹配的执行器",
    "Restore all": "全部恢复初始值",
    "Restore all initial controls?": "恢复所有控制输入的初始值？",
    "Restore": "恢复",
    "Read only": "只读",
    "Unlimited": "无界",
    "Control input": "控制输入",
    "No physical unit is declared for these controls.": "这些控制输入未声明物理单位。",
    "Text input owns keyboard shortcuts.": "文本输入时由输入框接管快捷键。",
    "Viewport": "视口",
    "Action": "操作",
    "Binding": "键位",
    "Scope": "作用范围",
    "Fly camera": "飞行相机",
    "Viewport + RMB held": "视口内按住右键",
    "Keyboard candidates apply only inside this probe.": "候选键位仅在此原型内生效。",
    "Playback and Tools share their thickness, not their length.": "播放与工具胶囊等厚，长度各自适配。",
    "Changes here do not update the scene or preferences.": "这里的修改仅用于原型预览。",
    "Hold Shift; click to lock": "按住 Shift；点击可锁定",
    "Double-click or press F2 to rename": "双击或按 F2 重命名",
    "Enter or leave the field to apply; Escape to cancel": "Enter 或失焦确认；Esc 取消",
    "Drag; Ctrl+click to type": "拖动；Ctrl+点击输入",
    "Duration": "时长",
    "seconds": "秒",
    "Record": "录制",
    "Object": "对象",
    "Restore initial; right-click for zero": "恢复初始值；右键可归零",
}


class InspectorPreview:
    """Use the current production Inspector against an isolated in-memory scene."""

    def __init__(self, name, position, color):
        scene = Scene()
        item = scene.add(MeshShape.BOX, name=name, position=position, color=color)
        self.session = Session(StaticSceneAdapter(scene))
        self.session.submit(cmd.Select(item.object_id))
        self.panel = InspectorPanel()
        self.localizer = Localizer()
        self.context = PanelContext(self.session, None, translate=self.localizer.text)

    def draw_properties(self, scale, language):
        self.context.style_scale = scale
        self.localizer.language = parse_language(language)
        self.session.tick(self.panel.frame_needs())
        node = self.session.selected_node
        self.panel._transform(self.context, node)
        self.panel._gizmo_reason(self.context, node)
        self.panel._velocity(self.context, node)
        self.panel._body_properties(self.context, node)
        self.panel._material(self.context, node)
        self.panel.finish_frame(self.context)


@dataclass
class RedesignState:
    language: str = "en"
    section: str = "Overview"
    selected: bool = True
    tool: str = "move"
    space: str = "world"
    snap_locked: bool = False
    snap_held: bool = False
    flying: bool = False
    frame: int = 24
    playing: bool = False
    recording: str = ""
    recording_mode: str = "take"
    duration: float = 10.0
    name: str = "torso"
    name_draft: str = "torso"
    renaming: bool = False
    rename_focus: bool = False
    rename_active: bool = False
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.48])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    color: list[float] = field(default_factory=lambda: [0.6, 0.59, 0.78, 1.0])
    controls: list[float] = field(default_factory=lambda: [0.42, -1.10, 2.50])
    search: str = ""
    read_only: bool = False
    rects: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    inspector: InspectorPreview | None = None
    _clock: float = 0.0

    def tr(self, text: str) -> str:
        return TRANSLATIONS.get(text, text) if self.language == "zh" else text

    def rename(self) -> None:
        self.renaming = self.rename_focus = True
        self.rename_active = False
        self.name_draft = self.name

    def finish_rename(self, *, cancel=False):
        name = self.name_draft.strip()
        if not cancel and name:
            self.name = name
            if self.inspector is not None:
                node = self.inspector.session.selected_node
                self.inspector.session.submit(cmd.RenameSceneEntity(node.object_id, name))
        self.renaming = self.rename_active = self.rename_focus = False


def _remember(state, key):
    a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    state.rects[key] = (a.x, a.y, b.x, b.y)


def _button(state, key, text, size=None):
    clicked = imgui.button(f"{state.tr(text)}##redesign-{key}", size or imgui.ImVec2(0, 0))
    _remember(state, key)
    return clicked


def _hint(state, text, key=""):
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_normal):
        imgui.set_tooltip(state.tr(text) + (f"  [{key}]" if key else ""))


@lru_cache(maxsize=64)
def reset_glyph_path(stroke: float, smoothing: float = CORNER_SMOOTHING):
    """Construct a counterclockwise arrow within Geometry's normalized icon circle."""
    radius = OVERLAY_GEOMETRY.icon_radius - stroke / 2
    angles = np.radians(np.linspace(140, -140, 65))
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.array((radial[-1, 1], -radial[-1, 0]))
    outer, inner = (radius + stroke / 2) * radial, (radius - stroke / 2) * radial
    head = np.array(
        (
            (radius + 2 * stroke) * radial[-1],
            radius * radial[-1] + 4 * stroke * tangent,
            (radius - 2 * stroke) * radial[-1],
        )
    )
    cap = smooth_line_cap(
        radius * radial[0], (-radial[0, 1], radial[0, 0]), stroke, smoothing=smoothing
    )
    outline = np.vstack((outer, head, inner[::-1], cap))
    # The head and arc share one silhouette, rounded with the existing G3 primitive.
    outline = smooth_polygon_corners(
        outline, stroke * 0.28, tuple(range(len(outer), len(outer) + 3)), smoothing=smoothing
    )
    outline *= (
        OVERLAY_GEOMETRY.icon_radius * RESET_GLYPH_SCALE / np.linalg.norm(outline, axis=1).max()
    )
    return tuple(map(tuple, outline.tolist()))


def draw_reset_glyph(draw, center, color, scale, stroke=OVERLAY_GEOMETRY.tool_stroke):
    draw.fringed_concave_fill(
        tuple(
            (center[0] + x * scale, center[1] + y * scale)
            for x, y in reset_glyph_path(stroke, draw.corner_smoothing)
        ),
        color,
    )


def draw_recording_glyph(draw, center, color, scale, *, recording=False):
    if recording:
        draw_playback_glyph(draw, center, color, scale, "stop", smoothing=draw.corner_smoothing)
    else:
        draw.circle_filled(center, RECORD_GLYPH_RADIUS * scale, color)


@lru_cache(maxsize=64)
def expand_glyph_path(stroke: float, smoothing: float = CORNER_SMOOTHING):
    """Join two rectangular arms, rounding every exposed corner with the shared G3 profile."""
    offset = stroke / sqrt(2)
    outline = (
        (-4 - offset, -2),
        (0, 2 + offset),
        (4 + offset, -2),
        (4, -2 - offset),
        (0, 2 - offset),
        (-4, -2 - offset),
    )
    corners = tuple(range(len(outline)))
    path = smooth_polygon_corners(
        outline,
        stroke / 2,
        corners,
        smoothing=smoothing,
        convex_only=False,
        corner_radii=dict.fromkeys(corners, stroke / 2),
    )
    return tuple(map(tuple, path.tolist()))


def draw_expand_glyph(draw, center, color, scale, stroke=OVERLAY_GEOMETRY.tool_stroke):
    draw.fringed_concave_fill(
        tuple(
            (center[0] + x * scale, center[1] + y * scale)
            for x, y in expand_glyph_path(stroke, draw.corner_smoothing)
        ),
        color,
    )


def _capsule(state, origin, scale, geometry, circular_button, vertical=False):
    items = (
        (
            ("move", "Move", "G"),
            ("rotate", "Rotate", "R"),
            ("dimensions", "Dimensions", "E"),
            ("frame", "World" if state.space == "world" else "Body", "T"),
            ("snap", "Hold Shift; click to lock", "Shift"),
        )
        if vertical
        else (
            ("previous", "Previous frame", "Left"),
            ("play", "Pause" if state.playing else "Play", "Space"),
            ("step", "Step", "Right"),
            ("reset", "Reset simulation", ""),
            (
                "record",
                "Stop recording"
                if state.recording
                else "Record take"
                if state.recording_mode == "take"
                else "Record video",
                "",
            ),
            ("menu", "Recording options", ""),
        )
    )
    centers, length = capsule_layout(len(items), (3,) if vertical else (3, 4), geometry)
    icon_radius = geometry.overlay_icon_radius
    state_radius = icon_radius + geometry.overlay_radial_step
    shell_radius = state_radius + geometry.overlay_radial_step
    thickness = 2 * shell_radius
    w, h = (thickness, length) if vertical else (length, thickness)
    x, y = origin
    draw = ImguiDraw2D(imgui.get_window_draw_list()).with_corner_smoothing(
        geometry.tool_smoothing if vertical else geometry.playback_smoothing
    )
    draw_capsule_shell(draw, x, y, w * scale, h * scale, scale, geometry)
    state.rects["tools" if vertical else "playback"] = (x, y, x + w * scale, y + h * scale)
    for boundary in (3,) if vertical else (3, 4):
        p = (centers[boundary - 1] + centers[boundary]) / 2 * scale
        a, b = (
            (shell_radius - geometry.divider_width / 2) * scale,
            (shell_radius + geometry.divider_width / 2) * scale,
        )
        draw.line(
            (x + a, y + p) if vertical else (x + p, y + a),
            (x + b, y + p) if vertical else (x + p, y + b),
            (*THEME.border[:3], 0.72),
            scale,
        )
    for index, (kind, label, key) in enumerate(items):
        cell = geometry.overlay_center_step * scale
        center = (
            (x + shell_radius * scale, y + centers[index] * scale)
            if vertical
            else (x + centers[index] * scale, y + shell_radius * scale)
        )
        on = (
            (state.tool == kind or (kind == "snap" and (state.snap_locked or state.snap_held)))
            if vertical
            else (
                (kind == "play" and state.playing) or (kind == "record" and bool(state.recording))
            )
        )
        disabled = (
            (vertical and not state.selected)
            or (kind == "previous" and state.frame == 0)
            or (kind in ("previous", "step") and state.playing)
        )

        def icon(target, center, color, icon_scale, _surface, kind=kind, disabled=disabled):
            if disabled:
                color = (*color[:3], color[3] * imgui.get_style().disabled_alpha)
            if vertical:
                draw_tool_glyph(
                    target,
                    center,
                    color,
                    icon_scale,
                    kind,
                    state.space,
                    replace(
                        OVERLAY_GEOMETRY,
                        tool_stroke=geometry.tool_stroke_width,
                        rotate_ring_gap_ratio=geometry.rotate_ring_gap_ratio,
                        rotate_ring_cap=geometry.rotate_ring_cap,
                    ),
                    smoothing=geometry.tool_smoothing,
                )
            elif kind == "reset":
                draw_reset_glyph(target, center, color, icon_scale, geometry.tool_stroke_width)
            elif kind == "record":
                draw_recording_glyph(
                    target, center, THEME.danger, icon_scale, recording=bool(state.recording)
                )
            elif kind == "menu":
                draw_expand_glyph(target, center, color, icon_scale, geometry.tool_stroke_width)
            else:
                draw_playback_glyph(
                    target,
                    center,
                    color,
                    icon_scale,
                    "pause" if kind == "play" and state.playing else kind,
                    smoothing=geometry.playback_smoothing,
                )

        imgui.begin_disabled(disabled)
        clicked = circular_button(
            draw,
            f"##redesign-{kind}",
            (center[0] - cell / 2, center[1] - cell / 2),
            icon,
            selected=on,
            cell_size=geometry.overlay_center_step,
            state_radius=state_radius,
            icon_radius=icon_radius * (TOOL_GLYPH_SCALE if vertical else 1),
            icon_scale=scale * icon_radius / OVERLAY_GEOMETRY.icon_radius,
            show_icon_bound=geometry.show_icon_bounds,
            show_state_circle=geometry.show_state_circles,
            scale=scale,
        )
        _remember(state, kind)
        _hint(state, label, key)
        imgui.end_disabled()
        if clicked:
            if kind in ("move", "rotate", "dimensions"):
                state.tool = kind
            elif kind == "frame":
                state.space = "body" if state.space == "world" else "world"
            elif kind == "snap":
                state.snap_locked = not state.snap_locked
            elif kind == "play":
                state.playing = not state.playing
            elif kind == "previous":
                state.frame = max(0, state.frame - 1)
            elif kind == "step":
                state.frame += 1
            elif kind == "reset":
                state.frame, state.playing, state.recording = 0, False, ""
            elif kind == "record":
                state.recording = "" if state.recording else state.recording_mode
            elif kind == "menu":
                imgui.open_popup("redesign-recording-menu")
    return w * scale, h * scale


def _recording_menu(state):
    settings = False
    if imgui.begin_popup("redesign-recording-menu"):
        for mode, label in (("take", "Record take"), ("video", "Record video")):
            imgui.begin_disabled(bool(state.recording))
            if _button(state, f"record-{mode}", label):
                state.recording_mode = mode
                state.recording = mode
                imgui.close_current_popup()
            imgui.end_disabled()
        imgui.separator()
        settings = _button(state, "recording-settings", "Recording settings")
        if settings:
            imgui.close_current_popup()
        imgui.end_popup()
    if settings:
        imgui.open_popup("redesign-recording-settings")
    if imgui.begin_popup("redesign-recording-settings"):
        imgui.text(state.tr("Recording settings"))
        _, state.duration = imgui.input_float(state.tr("Duration"), state.duration, 1, 5, "%.1f")
        state.duration = max(0.1, state.duration)
        if _button(state, "close-recording-settings", "Close"):
            imgui.close_current_popup()
        imgui.end_popup()


def _viewport(state, scale, geometry, circular_button):
    width = imgui.get_content_region_avail().x
    shell = 2 * (geometry.overlay_icon_radius + 2 * geometry.overlay_radial_step)
    _, tool_length = capsule_layout(5, (3,), geometry)
    height = max(350, 14 + shell + 18 + tool_length + 48) * scale
    pos = imgui.get_cursor_screen_pos()
    x, y = pos.x, pos.y
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    draw.rect_filled((x, y), (x + width, y + height), (0.137, 0.149, 0.165, 1), rounding=6 * scale)
    # Reserve layout space without intercepting overlay buttons.
    imgui.dummy(imgui.ImVec2(width, height))
    _remember(state, "viewport")
    hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
        pos, imgui.ImVec2(x + width, y + height)
    )
    popup = imgui.is_popup_open(
        "", imgui.PopupFlags_.any_popup_id | imgui.PopupFlags_.any_popup_level
    )
    io = imgui.get_io()
    state.flying = hovered and io.mouse_down[1] and not io.want_text_input and not popup
    state.snap_held = hovered and io.key_shift and not io.want_text_input and not popup
    if hovered and not io.want_text_input and not popup and not io.key_ctrl and not io.key_super:
        owner = imgui.get_id("redesign-viewport-keys")

        def pressed(key):
            imgui.internal.set_key_owner(
                key, owner, imgui.internal.InputFlagsPrivate_.lock_this_frame
            )
            return imgui.internal.is_key_pressed(key, 0, owner)

        if not state.flying and state.selected:
            for key, tool in (
                (imgui.Key.g, "move"),
                (imgui.Key.r, "rotate"),
                (imgui.Key.e, "dimensions"),
            ):
                if pressed(key):
                    state.tool = tool
            if pressed(imgui.Key.t):
                state.space = "body" if state.space == "world" else "world"
            if pressed(imgui.Key.f2):
                state.rename()
        if not state.flying:
            if pressed(imgui.Key.space):
                state.playing = not state.playing
            if not state.playing:
                if pressed(imgui.Key.right_arrow):
                    state.frame += 1
                if pressed(imgui.Key.left_arrow) or pressed(imgui.Key.backspace):
                    state.frame = max(0, state.frame - 1)
    if state.playing:
        state._clock += io.delta_time
        if state._clock >= 1 / 30:
            state.frame += int(state._clock * 30)
            state._clock %= 1 / 30
    for offset in range(7):
        gy = y + (120 + offset * 27) * scale
        draw.line(
            (x + 80 * scale, gy), (x + width - 15 * scale, gy), (*THEME.border[:3], 0.45), scale
        )
    cx, cy = x + width * 0.56, y + height * 0.55
    if state.selected:
        draw_tool_glyph(
            draw,
            (cx, cy),
            THEME.primary,
            3.0 * scale,
            state.tool,
            state.space,
            smoothing=geometry.tool_smoothing,
        )
    draw.text(
        (x + 16 * scale, y + height - 31 * scale),
        THEME.text_disabled,
        state.tr("Hold RMB to fly")
        if state.flying
        else state.tr("Selected") + ": " + state.name
        if state.selected
        else state.tr("No selection"),
    )
    _, length = capsule_layout(6, (3, 4), geometry)
    _capsule(
        state, (x + (width - length * scale) / 2, y + 14 * scale), scale, geometry, circular_button
    )
    if state.selected:
        _capsule(
            state,
            (x + 14 * scale, y + (14 + shell + 18) * scale),
            scale,
            geometry,
            circular_button,
            True,
        )
    _recording_menu(state)
    imgui.set_cursor_screen_pos(imgui.ImVec2(x, y + height + 8 * scale))
    status = f"{state.tr('Running' if state.playing else 'Paused')}  ·  {state.tr('Frame')} {state.frame}"
    if state.recording:
        status += f"  ·  REC {state.recording}"
    imgui.text(status)
    imgui.text_disabled(
        state.tr(
            "Snap held" if state.snap_held else "Snap locked" if state.snap_locked else "Snap off"
        )
    )


def _inspector(state, scale):
    if not state.selected:
        imgui.text(state.tr("Scene overview"))
        imgui.text_wrapped(state.tr("Select the object to edit its properties."))
        if _button(state, "select-object", "Select object"):
            state.selected = True
        return
    if state.inspector is None:
        state.inspector = InspectorPreview(state.name, state.position, state.color)
    node = state.inspector.session.selected_node
    origin = imgui.get_cursor_screen_pos()
    available = imgui.get_content_region_avail().x
    height = imgui.get_frame_height()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    draw.circle_filled(
        (origin.x + 6 * scale, origin.y + height / 2), 5 * scale, THEME.node_color(node.type)
    )
    imgui.dummy(imgui.ImVec2(14 * scale, height))
    imgui.same_line()
    name_width = min(
        max(60 * scale, imgui.calc_text_size(state.name).x + 16 * scale),
        max(1, available - 120 * scale),
    )
    if state.renaming:
        if state.rename_focus:
            imgui.set_keyboard_focus_here()
            state.rename_focus = False
        imgui.set_next_item_width(max(name_width, available - 120 * scale))
        submitted, state.name_draft = imgui.input_text(
            "##redesign-name",
            state.name_draft,
            imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all,
        )
        _remember(state, "name-input")
        active = imgui.is_item_active()
        state.rename_active |= active
        _hint(state, "Enter or leave the field to apply; Escape to cancel")
        if imgui.is_key_pressed(imgui.Key.escape, False):
            state.finish_rename(cancel=True)
        elif submitted or (state.rename_active and (not active or imgui.get_io().app_focus_lost)):
            state.finish_rename()
    else:
        imgui.push_style_var(imgui.StyleVar_.frame_border_size, scale)
        imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*THEME.primary))
        _button(state, "name", state.name, imgui.ImVec2(name_width, height))
        imgui.pop_style_color()
        imgui.pop_style_var()
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            state.rename()
        _hint(state, "Double-click or press F2 to rename", "F2")
    imgui.same_line()
    badge_pos = imgui.get_cursor_screen_pos()
    badge_width = imgui.calc_text_size(str(node.type)).x + 14 * scale
    draw.rect_filled(
        (badge_pos.x, badge_pos.y + 2 * scale),
        (badge_pos.x + badge_width, badge_pos.y + height - 2 * scale),
        THEME.bg_frame_active,
        rounding=height / 2,
    )
    draw.text(
        (badge_pos.x + 7 * scale, badge_pos.y + imgui.get_style().frame_padding.y),
        THEME.text_disabled,
        str(node.type),
    )
    imgui.dummy(imgui.ImVec2(badge_width, height))
    imgui.same_line()
    imgui.set_cursor_screen_pos(imgui.ImVec2(origin.x + available - height, origin.y))
    if _button(state, "details", "", imgui.ImVec2(height, height)):
        imgui.open_popup("redesign-details")
    for offset in (-3, 0, 3):
        draw.circle_filled(
            (origin.x + available - height / 2, origin.y + height / 2 + offset * scale),
            scale,
            THEME.text_disabled,
        )
    ids = f"node {node.node_id} · object {node.object_id} · body {node.body_index}"
    imgui.push_font(None, 11 * scale)
    imgui.text_disabled(ids + (" · posable" if node.posable else ""))
    imgui.pop_font()
    if imgui.begin_popup("redesign-details"):
        imgui.text(ids)
        if _button(state, "copy-ids", "Copy IDs"):
            imgui.set_clipboard_text(ids)
        imgui.end_popup()
    imgui.separator()
    state.inspector.draw_properties(scale, state.language)
    frame = state.inspector.session.frame
    state.position = frame.body_xpos[node.body_index].tolist()
    state.rotation = state.inspector.panel._rotation_euler.tolist()


CONTROL_ROWS = (
    ("hip_motor", -2.0, 2.0, 0.42),
    ("knee_motor", -4.0, 4.0, -1.10),
    ("custom_drive", None, None, 2.50),
)


def _control_row(state, index, scale):
    name, lo, hi, initial = CONTROL_ROWS[index]
    imgui.push_id(f"redesign-control-{index}")
    imgui.text(name)
    imgui.same_line()
    imgui.text_disabled(f"{lo:g} … {hi:g}" if lo is not None else state.tr("Unlimited"))
    width = imgui.get_content_region_avail().x
    imgui.begin_disabled(state.read_only)
    if lo is not None:
        # Retain native slider keyboard/navigation behavior under a thin rail.
        for slot in (
            imgui.Col_.frame_bg,
            imgui.Col_.frame_bg_hovered,
            imgui.Col_.frame_bg_active,
            imgui.Col_.slider_grab,
            imgui.Col_.slider_grab_active,
        ):
            imgui.push_style_color(slot, imgui.ImVec4(0, 0, 0, 0))
        imgui.set_next_item_width(max(48 * scale, width - 148 * scale))
        _, state.controls[index] = imgui.slider_float(
            "##rail", state.controls[index], lo, hi, "", imgui.SliderFlags_.no_input
        )
        _remember(state, f"control-rail-{index}")
        imgui.pop_style_color(5)
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw = ImguiDraw2D(imgui.get_window_draw_list())
        left, right, cy = a.x + 7 * scale, b.x - 7 * scale, (a.y + b.y) / 2
        value_x = left + (right - left) * (state.controls[index] - lo) / (hi - lo)
        zero_x = left + (right - left) * (-lo) / (hi - lo)
        alpha = 0.35 if state.read_only else 1
        draw.line((left, cy), (right, cy), THEME.bg_frame_active, 4 * scale)
        draw.line((zero_x, cy), (value_x, cy), (*THEME.primary_dim[:3], alpha), 4 * scale)
        draw.line((zero_x, cy - 4 * scale), (zero_x, cy + 4 * scale), THEME.text_disabled, scale)
        draw.circle_filled((value_x, cy), 5 * scale, (*THEME.primary[:3], alpha))
        imgui.same_line()
    imgui.set_next_item_width(100 * scale if lo is not None else width - 38 * scale)
    _, state.controls[index] = imgui.drag_float(
        "##value",
        state.controls[index],
        0.01,
        lo or 0,
        hi or 0,
        "%+.3f",
        imgui.SliderFlags_.always_clamp if lo is not None else 0,
    )
    _remember(state, f"control-value-{index}")
    _hint(state, "Drag; Ctrl+click to type")
    imgui.same_line()
    if _button(state, f"control-restore-{index}", "", imgui.ImVec2(28 * scale, 0)):
        state.controls[index] = initial
    _hint(state, "Restore initial; right-click for zero")
    a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_reset_glyph(
        ImguiDraw2D(imgui.get_window_draw_list()),
        ((a.x + b.x) / 2, (a.y + b.y) / 2),
        (*THEME.text[:3], 0.35 if state.read_only else 1),
        0.7 * scale,
    )
    if imgui.begin_popup_context_item("zero-or-restore"):
        if _button(state, f"control-zero-{index}", "Set zero"):
            state.controls[index] = 0.0
            imgui.close_current_popup()
        if _button(state, f"control-restore-menu-{index}", "Restore initial"):
            state.controls[index] = initial
            imgui.close_current_popup()
        imgui.end_popup()
    imgui.end_disabled()
    imgui.separator()
    imgui.pop_id()


def _control(state, scale):
    imgui.set_next_item_width(-1)
    _, state.search = imgui.input_text_with_hint(
        "##redesign-search", state.tr("Search actuators"), state.search
    )
    _remember(state, "control-search")
    _, state.read_only = imgui.checkbox(
        state.tr("Read only") + "##redesign-readonly", state.read_only
    )
    _remember(state, "readonly")
    imgui.same_line()
    imgui.begin_disabled(state.read_only)
    if _button(state, "restore-all", "Restore all"):
        imgui.open_popup("redesign-restore-all")
    imgui.end_disabled()
    if imgui.begin_popup("redesign-restore-all"):
        imgui.text(state.tr("Restore all initial controls?"))
        if _button(state, "confirm-restore", "Restore"):
            state.controls = [row[3] for row in CONTROL_ROWS]
            imgui.close_current_popup()
        imgui.same_line()
        if _button(state, "cancel-restore", "Cancel"):
            imgui.close_current_popup()
        imgui.end_popup()
    indices = [
        i for i, row in enumerate(CONTROL_ROWS) if state.search.casefold() in row[0].casefold()
    ]
    imgui.text(f"{state.tr('Actuators')}  {len(indices)} / {len(CONTROL_ROWS)}")
    imgui.separator()
    for index in indices:
        _control_row(state, index, scale)
    if not indices:
        imgui.text_disabled(state.tr("No matching actuators"))
    imgui.text_wrapped(state.tr("No physical unit is declared for these controls."))


def _keys(state):
    imgui.text_wrapped(state.tr("Keyboard candidates apply only inside this probe."))
    imgui.text_wrapped(state.tr("Text input owns keyboard shortcuts."))
    if imgui.begin_table(
        "##redesign-keys", 3, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
    ):
        for title in ("Action", "Binding", "Scope"):
            imgui.table_setup_column(state.tr(title))
        imgui.table_headers_row()
        for label, key, scope in (
            ("Move", "G", "Viewport"),
            ("Rotate", "R", "Viewport"),
            ("Dimensions", "E", "Viewport"),
            ("World", "T", "Viewport"),
            ("Snap", "Shift", "Viewport"),
            ("Play", "Space", "Viewport"),
            ("Previous frame", "Left / Backspace", "Viewport"),
            ("Step", "Right", "Viewport"),
            ("Name", "F2", "Viewport"),
            ("Fly camera", "W A S D Q E", "Viewport + RMB held"),
        ):
            imgui.table_next_row()
            for text in (state.tr(label), key, state.tr(scope)):
                imgui.table_next_column()
                imgui.text_wrapped(text)
        imgui.end_table()


def draw_redesign(state: RedesignState, scale: float, geometry, *, circular_button) -> None:
    state.rects.clear()
    for index, section in enumerate(("Overview", "Inspector", "Control", "Keys")):
        if index:
            imgui.same_line()
        if _button(state, f"section-{section}", section):
            state.section = section
    if _button(state, "geometry", "Geometry"):
        geometry.page = "Geometry"
    imgui.same_line()
    if _button(state, "language", "中文" if state.language == "en" else "English"):
        state.language = "zh" if state.language == "en" else "en"
    imgui.same_line()
    _, state.selected = imgui.checkbox(state.tr("Selected"), state.selected)
    if state.renaming and (
        not state.selected
        or state.section not in ("Overview", "Inspector")
        or geometry.page != "Redesign"
    ):
        state.finish_rename()
    imgui.separator()
    size = imgui.get_content_region_avail()
    if imgui.begin_child(f"##redesign-content-{state.section}", size, imgui.ChildFlags_.none):
        if state.section == "Overview":
            wide = imgui.get_content_region_avail().x > 900 * scale
            if wide and imgui.begin_table(
                "##redesign-overview", 2, imgui.TableFlags_.sizing_stretch_prop
            ):
                imgui.table_setup_column("Viewport", imgui.TableColumnFlags_.width_stretch, 0.62)
                imgui.table_setup_column("Inspector", imgui.TableColumnFlags_.width_stretch, 0.38)
                imgui.table_next_column()
                _viewport(state, scale, geometry, circular_button)
                imgui.table_next_column()
                imgui.text(state.tr("Inspector"))
                _inspector(state, scale)
                imgui.end_table()
            elif not wide:
                _viewport(state, scale, geometry, circular_button)
                imgui.separator_text(state.tr("Inspector"))
                _inspector(state, scale)
            imgui.spacing()
            imgui.text_wrapped(
                state.tr("Playback and Tools share their thickness, not their length.")
            )
            imgui.text_disabled(state.tr("Changes here do not update the scene or preferences."))
        elif state.section == "Inspector":
            _inspector(state, scale)
        elif state.section == "Control":
            _control(state, scale)
        else:
            _keys(state)
    imgui.end_child()
