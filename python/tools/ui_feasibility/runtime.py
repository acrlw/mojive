"""Render or interact with the UI feasibility workspace on a selected backend."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.render.selection import render_backend_name
from mojive.ui.icons import (
    ICON_ALIGNMENT_CHOICES,
    ICON_ALIGNMENT_EDITABLE_ICONS,
    ICON_FAMILIES,
    ICON_GROUP_BY_SLUG,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_MAX_PADDING,
    ICON_MAX_STROKE,
    ICON_MIN_CLEARANCE,
    ICON_MIN_STROKE,
    ICON_ROTATE_RING_CAP,
    icon_component_group,
)
from mojive.ui.window import WindowConfig

from .fixtures import CORNER_CONTROLS, DEFAULT_OUTPUT, PROBE_BASE_SIZE, _apply_concept_theme
from .layout import _probe_window_size
from .state import ProbeState
from .workspace import _draw_workspace


def render(
    output: Path,
    width: int,
    height: int,
    *,
    interactive: bool,
    initial_page: str,
    initial_geometry_tab: str,
    initial_icon_group: str,
    initial_rotate_cap: str,
    ui_scale: float,
    interactive_fps: float,
    initial_smoothing: float | None = None,
    initial_imgui_radius: float | None = None,
    initial_tool_stroke: float | None = None,
    initial_rotate_gap_ratio: float | None = None,
    initial_reset_head_scale: float | None = None,
    initial_playback_zoom: float | None = None,
    initial_icon_padding: float | None = None,
    initial_icon_stroke: float | None = None,
    initial_icon_alignment: str | None = None,
    redesign_language: str = "en",
    redesign_section: str = "Overview",
    capsule_outline: str = "Soft white",
    preview_icon_library: bool = False,
    renderer: str | None = None,
    icon_glyph: str | None = None,
) -> None:
    renderer = render_backend_name(renderer)
    if icon_glyph is not None:
        initial_page, initial_geometry_tab = "Geometry", "Icon library"
        initial_icon_group = icon_component_group(icon_glyph)
    window_width, window_height = _probe_window_size(width, height, ui_scale)
    window = create_window(
        WindowConfig(
            title="Mojive UI feasibility",
            width=window_width,
            height=window_height,
            # The probe is UI, not a GPU benchmark. Interactive mode is paced,
            # and DrawList antialiasing already covers its vector edges.
            vsync=interactive,
            docking=False,
            ini_path="",
            show_on_start=False,
            samples=0,
            ui_scale=ui_scale,
        ),
        renderer,
    )
    try:
        _apply_concept_theme(window.style_scale)
        state = ProbeState(
            renderer=renderer,
            page=initial_page,
            geometry_tab=initial_geometry_tab,
            icon_library_tab=initial_icon_group,
            icon_glyph=icon_glyph,
            rotate_ring_cap=initial_rotate_cap,
            capsule_outline=capsule_outline,
            preview_icon_library=preview_icon_library,
        )
        state.redesign.language = redesign_language
        state.redesign.section = redesign_section
        if initial_smoothing is not None:
            for _, name in CORNER_CONTROLS:
                setattr(state, name, initial_smoothing)
        if initial_imgui_radius is not None:
            state.imgui_rounding = initial_imgui_radius
        if initial_tool_stroke is not None:
            state.tool_stroke_width = initial_tool_stroke
        if initial_rotate_gap_ratio is not None:
            state.rotate_ring_gap_ratio = initial_rotate_gap_ratio
        if initial_reset_head_scale is not None:
            state.reset_head_scale = initial_reset_head_scale
        if initial_playback_zoom is not None:
            state.construction_playback_scale = initial_playback_zoom
        initial_adjustment_group = (
            initial_icon_group
            if initial_icon_group in ICON_GROUP_LAYOUT_DEFAULTS
            else {
                "Capsules": "Viewport playback",
                "UI context": "Keyframes",
                "Keyframe follow": "Keyframes",
            }.get(initial_icon_group, state.icon_adjustment_group)
        )
        state.icon_adjustment_group = initial_adjustment_group
        if initial_icon_padding is not None:
            state.set_icon_padding_for(initial_adjustment_group, initial_icon_padding)
        if initial_icon_stroke is not None:
            state.set_icon_stroke_for(initial_adjustment_group, initial_icon_stroke)
        if initial_icon_alignment is not None:
            for name in ICON_ALIGNMENT_EDITABLE_ICONS:
                state.set_icon_alignment_for_glyph(name, initial_icon_alignment)
        if interactive:
            window.show()
            frame_period = 1.0 / interactive_fps
            try:
                while not window.should_close():
                    frame_started = time.perf_counter()
                    window.begin_frame()
                    _draw_workspace(window, state)
                    if state.page != "Redesign" and imgui.is_key_pressed(imgui.Key.escape, False):
                        if state.joint_value_open:
                            state.joint_value_open = False
                        elif state.value_open:
                            state.value_open = False
                        elif not imgui.is_any_item_active():
                            window.request_close()
                    window.end_frame()
                    remaining = frame_period - (time.perf_counter() - frame_started)
                    if remaining > 0.0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                pass
            return

        pixels = None
        for _ in range(4):
            window.begin_frame()
            _draw_workspace(window, state)
            pixels = window.end_frame(readback=True)
        if pixels is None:
            raise RuntimeError("ImGui framebuffer readback returned no pixels")
        image = Image.fromarray(np.asarray(pixels)[::-1], "RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
    finally:
        if "state" in locals() and state.timeline_session is not None:
            state.timeline_session.release()
        window.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer",
        choices=("opengl", "wgpu", "bgfx"),
        default=None,
        help="Window renderer; defaults to MOJIVE_RENDERER or OpenGL",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=PROBE_BASE_SIZE[0])
    parser.add_argument("--height", type=int, default=PROBE_BASE_SIZE[1])
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.0,
        help=(
            "Logical UI scale used for paths, strokes, controls, and text; "
            "the capture expands above 2x and oversized concept canvases scroll"
        ),
    )
    parser.add_argument(
        "--page",
        choices=("workspace", "panels", "geometry", "redesign"),
        default="workspace",
        help="Initial probe page; interactive mode can switch from the Probe menu",
    )
    parser.add_argument("--redesign-language", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--capsule-outline", choices=("neutral-gray", "soft-white"), default="soft-white"
    )
    parser.add_argument(
        "--redesign-section",
        choices=("overview", "inspector", "control", "keys"),
        default="overview",
    )
    parser.add_argument(
        "--geometry-tab",
        choices=(
            "corners",
            "playback",
            "tools",
            "icons",
            "hints",
            "gizmos",
            "helpers",
            "status",
            "diagnostics",
            "shell",
            "panels",
            "workspaces",
        ),
        default="playback",
        help="Initial non-closeable tab on the geometry page",
    )
    parser.add_argument(
        "--icon-group",
        choices=tuple(ICON_GROUP_BY_SLUG),
        default="overview",
        help="Initial family on the concept-only Icon library geometry tab",
    )
    parser.add_argument(
        "--icon-glyph",
        choices=tuple(name for _family, icons in ICON_FAMILIES for _label, name in icons),
        help="Capture one glyph at every review size without an oversized family canvas",
    )
    parser.add_argument(
        "--rotate-cap",
        choices=("butt", "round"),
        default=ICON_ROTATE_RING_CAP,
        help="Initial Rotate inner-ring cap style",
    )
    parser.add_argument(
        "--tool-stroke",
        type=float,
        default=None,
        help="Initial Tool glyph stroke, from 1.0 to 2.2 logical pixels",
    )
    parser.add_argument(
        "--rotate-gap-ratio",
        type=float,
        default=None,
        help=("Initial Rotate crossing gap as a stroke ratio, from 0.25 to 1.00"),
    )
    parser.add_argument(
        "--playback-zoom",
        type=float,
        default=None,
        help="Initial Playback construction zoom, from 1.5 to 4.0",
    )
    parser.add_argument(
        "--reset-head-scale",
        type=float,
        default=None,
        help="Shared reset/refresh arrowhead size, from 0.65 to 2.0 times the original",
    )
    parser.add_argument(
        "--icon-padding",
        type=float,
        default=None,
        help=(
            f"Initial circular glyph padding for --icon-group, from {ICON_MIN_CLEARANCE:g} "
            f"to {ICON_MAX_PADDING:g} grid units"
        ),
    )
    parser.add_argument(
        "--icon-stroke",
        type=float,
        default=None,
        help=(
            f"Initial Icon Library stroke for --icon-group, from {ICON_MIN_STROKE:g} "
            f"to {ICON_MAX_STROKE:g} grid units"
        ),
    )
    parser.add_argument(
        "--icon-alignment",
        choices=ICON_ALIGNMENT_CHOICES,
        default=None,
        help="Initial Box/Circle placement for Snapshot, Camera, and Light review",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a real ImGui window and run until it is closed",
    )
    parser.add_argument(
        "--preview-icon-library",
        action="store_true",
        help="Use Icon Library candidates throughout the selected feasibility page",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Maximum interactive refresh rate (default: 30)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Initial corner smoothing for all element groups, from 0 to 1",
    )
    parser.add_argument(
        "--imgui-radius",
        type=float,
        default=None,
        help="Initial ImGui radius in logical pixels, from 0 to 16",
    )
    args = parser.parse_args()
    if args.imgui_radius is not None and not 0.0 <= args.imgui_radius <= 16.0:
        parser.error("--imgui-radius must be between 0 and 16")
    if args.smoothing is not None and not 0.0 <= args.smoothing <= 1.0:
        parser.error("--smoothing must be between 0 and 1")
    if args.tool_stroke is not None and not 1.0 <= args.tool_stroke <= 2.2:
        parser.error("--tool-stroke must be between 1.0 and 2.2")
    if args.rotate_gap_ratio is not None and not 0.25 <= args.rotate_gap_ratio <= 1.0:
        parser.error("--rotate-gap-ratio must be between 0.25 and 1.00")
    if args.reset_head_scale is not None and not 0.65 <= args.reset_head_scale <= 2.0:
        parser.error("--reset-head-scale must be between 0.65 and 2.0")
    if args.playback_zoom is not None and not 1.5 <= args.playback_zoom <= 4.0:
        parser.error("--playback-zoom must be between 1.5 and 4.0")
    if (
        args.icon_padding is not None
        and not ICON_MIN_CLEARANCE <= args.icon_padding <= ICON_MAX_PADDING
    ):
        parser.error(
            f"--icon-padding must be between {ICON_MIN_CLEARANCE:g} and {ICON_MAX_PADDING:g}"
        )
    if args.icon_stroke is not None and not ICON_MIN_STROKE <= args.icon_stroke <= ICON_MAX_STROKE:
        parser.error(f"--icon-stroke must be between {ICON_MIN_STROKE:g} and {ICON_MAX_STROKE:g}")
    if not 0.5 <= args.ui_scale <= 4.0:
        parser.error("--ui-scale must be between 0.5 and 4.0")
    if not 15.0 <= args.fps <= 240.0:
        parser.error("--fps must be between 15 and 240")
    output = args.output.resolve()
    render(
        output,
        args.width,
        args.height,
        interactive=args.interactive,
        initial_page=args.page.title(),
        initial_geometry_tab={
            "corners": "Corners",
            "playback": "Playback",
            "tools": "Tools",
            "icons": "Icon library",
            "hints": "Hints & input",
            "gizmos": "Transform gizmos",
            "helpers": "Joint & helpers",
            "status": "Status",
            "diagnostics": "Diagnostics",
            "shell": "Shell & settings",
            "panels": "Panels",
            "workspaces": "Workspaces",
        }[args.geometry_tab],
        initial_icon_group=ICON_GROUP_BY_SLUG[args.icon_group],
        icon_glyph=args.icon_glyph,
        initial_rotate_cap=args.rotate_cap,
        ui_scale=args.ui_scale,
        interactive_fps=args.fps,
        initial_smoothing=args.smoothing,
        initial_imgui_radius=args.imgui_radius,
        initial_tool_stroke=args.tool_stroke,
        initial_rotate_gap_ratio=args.rotate_gap_ratio,
        initial_reset_head_scale=args.reset_head_scale,
        initial_playback_zoom=args.playback_zoom,
        initial_icon_padding=args.icon_padding,
        initial_icon_stroke=args.icon_stroke,
        initial_icon_alignment=args.icon_alignment,
        redesign_language=args.redesign_language,
        redesign_section=args.redesign_section.title(),
        capsule_outline=args.capsule_outline.replace("-", " ").capitalize(),
        preview_icon_library=args.preview_icon_library,
        renderer=args.renderer,
    )
    print("interactive probe closed" if args.interactive else output)
