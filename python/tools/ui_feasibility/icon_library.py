"""Icon family, component-context and multi-size geometry review pages."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial

import numpy as np
from imgui_bundle import imgui

from mojive.ui.controls import segmented_control, segmented_control_width
from mojive.ui.icons import (
    ICON_ALIGNMENT_CHOICES,
    ICON_ALIGNMENT_EDITABLE_ICONS,
    ICON_BOUND_DIAMETER,
    ICON_DEFAULT_PADDING,
    ICON_FAMILIES,
    ICON_GRID,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_LIBRARY_TABS,
    ICON_MAX_PADDING,
    ICON_MAX_STROKE,
    ICON_MIN_CLEARANCE,
    ICON_MIN_STROKE,
    ICON_ROTATE_RING_CAP,
    ICON_ROTATE_RING_GAP_RATIO,
    ICON_STROKE,
    ICON_TUNING_DEFAULTS,
    REVIEW_LOCKED_ICONS,
    STATUS_MOUSE_DEFAULT_WIDTH,
    STROKE_SCALE_LOCKED_ICONS,
    IconTuning,
    draw_concept_icon,
    icon_alignment_anchor,
    icon_alignment_center,
    icon_family,
    icon_metrics,
    minimum_enclosing_circle,
)
from mojive.ui.keyframe_editor import controls as keyframes_panel_module
from mojive.ui.keyframe_editor.controls import _draw_command_icon
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels.filters import severity_icon, severity_meshes
from mojive.ui.text_layout import text_line_y
from mojive.ui.theme import THEME

from .fixtures import _ICON_REVIEW_SIZES, CONCEPT_THEME
from .layout import _deferred_icon_group_slider, _even_slider, _stepped_slider, _wrapped_tabs
from .state import ProbeState
from .tuning import _draw_icon_group_combo, _icon_values_text
from .widgets import (
    _circular_icon_button,
    _draw_icon_library_label,
    _draw_playback,
    _draw_tool_column,
)


def _concept_icon_color(name: str):
    if name == "status-info":
        return CONCEPT_THEME.info
    if name == "status-warning":
        return CONCEPT_THEME.warning
    if name == "status-error":
        return CONCEPT_THEME.danger
    return CONCEPT_THEME.text


def _draw_concept_icon_specimen(
    draw,
    center,
    size: float,
    name: str,
    scale: float,
    padding: float = ICON_DEFAULT_PADDING,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
    centroid: tuple[float, float] | None = None,
) -> None:
    """Draw one candidate inside the shared circular placement boundary."""

    guide_radius = size * ICON_BOUND_DIAMETER / ICON_GRID * 0.5
    # Keep review guides behind the candidate. Rotate deliberately places its
    # outer-ring centerline on the orange circle; drawing the guide afterward
    # recolored the center of that stroke and falsely made the frame look thin.
    # The square has exactly the orange circle's diameter.
    draw.rect(
        (center[0] - guide_radius, center[1] - guide_radius),
        (center[0] + guide_radius, center[1] + guide_radius),
        (*CONCEPT_THEME.text_disabled[:3], 0.34),
        max(0.6, 0.72 * scale),
    )
    draw.circle(
        center,
        guide_radius,
        (*CONCEPT_THEME.warning[:3], 0.72),
        max(0.75, 0.9 * scale),
        segments=max(32, round(guide_radius * 4.0)),
    )
    if centroid is not None:
        for dx, dy in ((4 * scale, 0), (0, 4 * scale)):
            draw.line(
                (center[0] - dx, center[1] - dy),
                (center[0] + dx, center[1] + dy),
                CONCEPT_THEME.text_disabled,
                scale,
            )
    center = (center[0] + offset[0] * size / ICON_GRID, center[1] + offset[1] * size / ICON_GRID)
    if name.startswith("status-") and name.removeprefix("status-") in {
        "info",
        "warning",
        "error",
    }:
        # Output already has a reviewed production family. Show that exact
        # painter here so the concept library cannot silently drift from it.
        severity_icon(
            draw,
            center,
            size,
            name.removeprefix("status-"),
            _concept_icon_color(name),
        )
    else:
        draw_concept_icon(
            draw,
            center,
            size,
            name,
            _concept_icon_color(name),
            padding=padding,
            accent_color=(CONCEPT_THEME.primary if name.startswith("status-mouse-") else None),
            mouse_width=mouse_width,
            stroke_width=stroke_width,
            rotate_ring_gap_ratio=rotate_ring_gap_ratio,
            rotate_ring_cap=rotate_ring_cap,
            tuning=tuning,
            alignment=alignment,
        )
    if centroid is not None:
        draw.circle_filled(
            (
                center[0] + centroid[0] * size / ICON_GRID,
                center[1] + centroid[1] * size / ICON_GRID,
            ),
            2.0 * scale,
            CONCEPT_THEME.warning,
        )


@lru_cache(maxsize=256)
def _icon_review_metrics(
    name: str,
    padding: float = ICON_DEFAULT_PADDING,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
    stroke_width: float = ICON_STROKE,
    rotate_ring_gap_ratio: float = ICON_ROTATE_RING_GAP_RATIO,
    rotate_ring_cap: str = ICON_ROTATE_RING_CAP,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
) -> tuple[float, float, float, float, float]:
    """Return circle clearance plus enclosing-circle and box centers."""

    if name.startswith("status-") and name.removeprefix("status-") in {
        "info",
        "warning",
        "error",
    }:
        meshes = severity_meshes(ICON_GRID, name.removeprefix("status-"))
        points = np.concatenate([np.asarray(mesh[0], np.float64) for mesh in meshes])
        bounds = (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
        enclosing_center, enclosing_radius = minimum_enclosing_circle(points)
        return (
            ICON_BOUND_DIAMETER * 0.5 - enclosing_radius,
            enclosing_center[0],
            enclosing_center[1],
            (bounds[0] + bounds[2]) * 0.5,
            (bounds[1] + bounds[3]) * 0.5,
        )
    metrics = icon_metrics(
        name,
        padding=padding,
        mouse_width=mouse_width,
        stroke_width=stroke_width,
        rotate_ring_gap_ratio=rotate_ring_gap_ratio,
        rotate_ring_cap=rotate_ring_cap,
        tuning=tuning,
        alignment=alignment,
    )
    box_x, box_y = metrics.center_offset
    return (
        metrics.circular_clearance,
        metrics.enclosing_center[0],
        metrics.enclosing_center[1],
        box_x,
        box_y,
    )


def _draw_icon_library_overview(
    draw,
    origin,
    scale: float,
    state: ProbeState,
) -> None:
    card_width = 468.0 * scale
    card_height = 244.0 * scale
    gap_x = 20.0 * scale
    gap_y = 18.0 * scale
    clip_min = imgui.get_window_draw_list().get_clip_rect_min()
    clip_max = imgui.get_window_draw_list().get_clip_rect_max()
    for family_index, (family, icons) in enumerate(ICON_FAMILIES):
        column = family_index % 3
        row = family_index // 3
        x0 = origin[0] + column * (card_width + gap_x)
        y0 = origin[1] + row * (card_height + gap_y)
        x1, y1 = x0 + card_width, y0 + card_height
        if y1 < clip_min.y or y0 > clip_max.y:
            continue
        draw.rect_filled((x0, y0), (x1, y1), CONCEPT_THEME.bg_child, rounding=8.0 * scale)
        draw.rect(
            (x0, y0),
            (x1, y1),
            CONCEPT_THEME.border,
            max(0.75, scale),
            rounding=8.0 * scale,
        )
        draw.text((x0 + 16.0 * scale, y0 + 12.0 * scale), CONCEPT_THEME.text, family)
        draw.text(
            (x0 + 16.0 * scale, y0 + 34.0 * scale),
            CONCEPT_THEME.text_disabled,
            f"{len(icons)} candidates · 20 pt",
        )

        columns = min(5, len(icons))
        cell_width = (card_width - 28.0 * scale) / columns
        for icon_index, (label, name) in enumerate(icons):
            icon_column = icon_index % columns
            icon_row = icon_index // columns
            center = (
                x0 + 14.0 * scale + cell_width * (icon_column + 0.5),
                y0 + (86.0 + icon_row * 76.0) * scale,
            )
            _draw_concept_icon_specimen(
                draw,
                center,
                20.0 * scale,
                name,
                scale,
                state.icon_padding_for_glyph(name),
                state.concept_mouse_width(),
                state.icon_stroke_for_glyph(name),
                state.rotate_ring_gap_ratio,
                state.rotate_ring_cap,
                state.icon_tuning(),
                state.icon_alignment_for_glyph(name),
                offset=state.icon_offset(name),
                centroid=state.icon_centroid(name) if state.show_icon_centroids else None,
            )
            draw.centered_label(
                label,
                (center[0], center[1] + 29.0 * scale),
                CONCEPT_THEME.text_disabled,
                cell_width - 8.0 * scale,
            )


def _draw_context_row(
    draw,
    origin,
    width: float,
    row_height: float,
    icon_size: float,
    icon_name: str,
    label: str,
    detail: str,
    scale: float,
    padding: float,
    stroke_width: float = ICON_STROKE,
    *,
    production_kind: str | None = None,
    tuning: IconTuning = ICON_TUNING_DEFAULTS,
    alignment: str | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Place one candidate in current Mojive row metrics with text guides."""

    x0, y0 = float(origin[0]), float(origin[1])
    width *= scale
    row_height *= scale
    icon_size *= scale
    center_y = y0 + row_height * 0.5
    draw.rect_filled(
        (x0, y0),
        (x0 + width, y0 + row_height),
        CONCEPT_THEME.bg_frame,
        rounding=5.0 * scale,
    )
    icon_center = (x0 + 22.0 * scale, center_y)
    text_x = x0 + 40.0 * scale
    text_y = text_line_y(draw, center_y)
    reference = draw.text_ink_bounds("H")
    cap_y = text_y + (reference[1] if reference else 0.0)
    baseline_y = text_y + float(imgui.get_font_baked().ascent)
    guide_right = x0 + width - 250.0 * scale
    draw.line(
        (x0 + 8.0 * scale, center_y),
        (guide_right, center_y),
        (*CONCEPT_THEME.warning[:3], 0.18),
        max(0.5, 0.7 * scale),
    )
    draw.line(
        (text_x, cap_y),
        (guide_right, cap_y),
        (*CONCEPT_THEME.primary[:3], 0.48),
        max(0.5, 0.7 * scale),
    )
    draw.line(
        (text_x, baseline_y),
        (guide_right, baseline_y),
        (*CONCEPT_THEME.info[:3], 0.52),
        max(0.5, 0.7 * scale),
    )
    if production_kind is None:
        draw_concept_icon(
            draw,
            (
                icon_center[0] + offset[0] * icon_size / ICON_GRID,
                icon_center[1] + offset[1] * icon_size / ICON_GRID,
            ),
            icon_size,
            icon_name,
            CONCEPT_THEME.text,
            padding=padding,
            stroke_width=stroke_width,
            tuning=tuning,
            alignment=alignment,
        )
    else:
        _draw_command_icon(draw, icon_center, production_kind, CONCEPT_THEME.text, scale)
    draw.circle(
        icon_center,
        icon_size * 0.5,
        (*CONCEPT_THEME.warning[:3], 0.72),
        max(0.75, 0.9 * scale),
        segments=max(32, round(icon_size * 2.0)),
    )
    draw.text((text_x, text_y), CONCEPT_THEME.text, label, pixel_snap=False)
    draw.text(
        (x0 + width - 230.0 * scale, text_y),
        CONCEPT_THEME.text_disabled,
        detail,
        pixel_snap=False,
    )


def _draw_icon_context_page(
    draw,
    origin,
    scale: float,
    state: ProbeState,
) -> None:
    """Review candidate alignment in Mojive's actual row and font metrics."""

    x0, y0 = float(origin[0]), float(origin[1])
    draw.text((x0, y0 + 10.0 * scale), CONCEPT_THEME.text, "Alignment in UI")
    draw.text(
        (x0, y0 + 34.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Candidates use current Mojive row heights and font metrics; orange guides are review-only.",
    )
    draw.text(
        (x0, y0 + 58.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Green = H cap line · blue = text baseline · faint amber = row and icon-slot center",
    )

    card_x = x0 - 10.0 * scale
    card_width = 1370.0 * scale
    first_y = y0 + 96.0 * scale
    draw.rect_filled(
        (card_x, first_y),
        (card_x + card_width, first_y + 250.0 * scale),
        (*CONCEPT_THEME.bg_frame[:3], 0.55),
        rounding=6.0 * scale,
    )
    draw.text(
        (x0 + 10.0 * scale, first_y + 14.0 * scale),
        CONCEPT_THEME.text,
        "Camera and light masters · switchable Box / Circle placement",
    )
    draw.text(
        (x0 + 10.0 * scale, first_y + 38.0 * scale),
        CONCEPT_THEME.text_disabled,
        "The family-row Align control applies the selected visible box or minimum-circle center here.",
    )
    samples = (
        ("Keyframes", "key-snapshot"),
        ("Scene helpers", "helper-camera"),
    )
    for index, (label, name) in enumerate(samples):
        center = (x0 + (390.0 + index * 520.0) * scale, first_y + 132.0 * scale)
        draw_concept_icon(
            draw,
            state.icon_center(name, center, 112.0 * scale),
            112.0 * scale,
            name,
            CONCEPT_THEME.text,
            padding=state.icon_padding_for_glyph(name),
            stroke_width=state.icon_stroke_for_glyph(name),
            tuning=state.icon_tuning(),
            alignment=state.icon_alignment_for_glyph(name),
        )
        draw.circle(
            center,
            56.0 * scale,
            (*CONCEPT_THEME.warning[:3], 0.72),
            max(0.75, 0.9 * scale),
            segments=max(64, round(112.0 * scale * 2.0)),
        )
        draw.centered_label(
            label,
            (center[0], first_y + 215.0 * scale),
            CONCEPT_THEME.text_disabled,
            360.0 * scale,
        )

    second_y = first_y + 274.0 * scale
    draw.text((x0, second_y), CONCEPT_THEME.text, "Current UI metrics")
    draw.text(
        (x0, second_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Every candidate slot and label ink share the row center; the selected geometric anchor is then placed on that slot.",
    )
    row_x = x0 + 10.0 * scale
    row_width = 1320.0
    _draw_context_row(
        draw,
        (row_x, second_y + 62.0 * scale),
        row_width,
        28.0,
        16.0,
        "key-keyframe",
        "Capture snapshot",
        "production · 28 pt row / 16 pt slot",
        scale,
        state.icon_padding_for_glyph("key-keyframe"),
        state.icon_stroke_for_glyph("key-keyframe"),
        production_kind="key",
        tuning=state.icon_tuning(),
        alignment=state.icon_alignment_for_glyph("key-keyframe"),
    )
    _draw_context_row(
        draw,
        (row_x, second_y + 112.0 * scale),
        row_width,
        28.0,
        16.0,
        "key-snapshot",
        "Capture snapshot",
        f"candidate · {state.icon_alignment_for_glyph('key-snapshot')} anchor",
        scale,
        state.icon_padding_for_glyph("key-snapshot"),
        state.icon_stroke_for_glyph("key-snapshot"),
        tuning=state.icon_tuning(),
        alignment=state.icon_alignment_for_glyph("key-snapshot"),
        offset=state.icon_offset("key-snapshot"),
    )
    _draw_context_row(
        draw,
        (row_x, second_y + 162.0 * scale),
        row_width,
        26.0,
        14.0,
        "helper-camera",
        "camera0",
        "candidate · 26 pt hierarchy row",
        scale,
        state.icon_padding_for_glyph("helper-camera"),
        state.icon_stroke_for_glyph("helper-camera"),
        tuning=state.icon_tuning(),
        alignment=state.icon_alignment_for_glyph("helper-camera"),
        offset=state.icon_offset("helper-camera"),
    )
    _draw_context_row(
        draw,
        (row_x, second_y + 212.0 * scale),
        row_width,
        26.0,
        14.0,
        "helper-light",
        "light0",
        "candidate · 26 pt hierarchy row",
        scale,
        state.icon_padding_for_glyph("helper-light"),
        state.icon_stroke_for_glyph("helper-light"),
        tuning=state.icon_tuning(),
        alignment=state.icon_alignment_for_glyph("helper-light"),
        offset=state.icon_offset("helper-light"),
    )

    note_y = second_y + 270.0 * scale
    draw.text((x0, note_y), CONCEPT_THEME.text, "Review rule")
    draw.text(
        (x0, note_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "The row center is shared by the icon slot and the visible text ink; then the chosen glyph anchor is placed on it.",
    )
    draw.text(
        (x0, note_y + 48.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Snapshot, Camera, and Light can compare Box or Circle; fixed families retain their documented anchor.",
    )


def _draw_icon_family_detail(
    draw,
    origin,
    family: str,
    scale: float,
    state: ProbeState,
) -> None:
    icons = icon_family(family)
    if state.icon_glyph is not None:
        icons = tuple(item for item in icons if item[1] == state.icon_glyph)
    mouse_width = state.concept_mouse_width()
    centers = tuple(origin[0] + offset * scale for offset in (330.0, 535.0, 765.0, 1085.0))
    header_y = origin[1] + 10.0 * scale
    draw.text((origin[0], header_y), CONCEPT_THEME.text, family)
    draw.text(
        (origin[0], header_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Orange circle and gray square share one center · each glyph reports its declared anchor",
    )
    for center_x, size in zip(centers, _ICON_REVIEW_SIZES, strict=True):
        draw.centered_label(
            f"{int(size)} pt",
            (center_x, header_y + 62.0 * scale),
            CONCEPT_THEME.text_disabled,
            90.0 * scale,
        )

    rows_y = header_y + 98.0 * scale
    row_height = (max(_ICON_REVIEW_SIZES) + 20.0) * scale
    right = origin[0] + 1370.0 * scale
    clip_min = imgui.get_window_draw_list().get_clip_rect_min()
    clip_max = imgui.get_window_draw_list().get_clip_rect_max()
    for row, (label, name) in enumerate(icons):
        center_y = rows_y + row_height * row + row_height * 0.5
        row_top = center_y - row_height * 0.5
        row_bottom = center_y + row_height * 0.5
        if row_bottom < clip_min.y or row_top > clip_max.y:
            continue
        if row % 2 == 0:
            draw.rect_filled(
                (origin[0] - 10.0 * scale, row_top),
                (right, row_bottom),
                (*CONCEPT_THEME.bg_frame[:3], 0.55),
                rounding=5.0 * scale,
            )
        draw.text(
            (origin[0] + 10.0 * scale, center_y - 51.0 * scale),
            CONCEPT_THEME.text,
            label,
        )
        padding = state.icon_padding_for_glyph(name)
        stroke_width = state.icon_stroke_for_glyph(name)
        alignment = state.icon_alignment_for_glyph(name)
        control_x = origin[0] + 52.0 * scale
        control_width = 150.0 * scale
        imgui.push_id(name)
        draw.text(
            (origin[0] + 10.0 * scale, center_y - 20.0 * scale),
            CONCEPT_THEME.text_disabled,
            "Pad",
        )
        imgui.set_cursor_screen_pos(imgui.ImVec2(float(control_x), float(center_y - 27.0 * scale)))
        imgui.set_next_item_width(control_width)
        padding_locked = name in REVIEW_LOCKED_ICONS
        minimum_padding = 0.0 if name == "tool-rotate" else ICON_MIN_CLEARANCE
        imgui.begin_disabled(padding_locked)
        changed, padding = _stepped_slider(
            "##glyph-padding",
            padding,
            minimum_padding,
            ICON_MAX_PADDING,
            "%.2f u",
        )
        imgui.end_disabled()
        if changed and not padding_locked:
            state.set_icon_padding_for_glyph(name, padding)

        draw.text(
            (origin[0] + 10.0 * scale, center_y + 14.0 * scale),
            CONCEPT_THEME.text_disabled,
            "Stroke",
        )
        imgui.set_cursor_screen_pos(imgui.ImVec2(float(control_x), float(center_y + 7.0 * scale)))
        imgui.set_next_item_width(control_width)
        stroke_locked = name in STROKE_SCALE_LOCKED_ICONS or name in {
            "playback-play",
            "playback-pause",
            "playback-record",
            "playback-stop",
            "transport-first",
            "transport-play",
            "transport-pause",
            "transport-last",
            "transport-record",
            "transport-stop",
            "panel-right",
            "panel-down",
        }
        imgui.begin_disabled(stroke_locked)
        changed, stroke_width = _stepped_slider(
            "##glyph-stroke",
            stroke_width,
            ICON_MIN_STROKE,
            ICON_MAX_STROKE,
            "%.2f u",
        )
        imgui.end_disabled()
        if changed and not stroke_locked:
            state.set_icon_stroke_for_glyph(name, stroke_width)

        shape_control = None
        if name == "tool-rotate":
            shape_control = (
                "Gap",
                "##rotate-gap-ratio",
                state.rotate_ring_gap_ratio,
                0.25,
                1.0,
                "%.2f × stroke",
                0.05,
            )
        elif name == "tool-move":
            shape_control = (
                "Head",
                "##move-head-scale",
                state.move_head_scale,
                0.65,
                1.35,
                "%.2f × original",
                0.05,
            )
        elif name == "tool-scale":
            shape_control = (
                "Blocks",
                "##scale-handle-scale",
                state.scale_handle_scale,
                0.65,
                1.35,
                "%.2f × original",
                0.05,
            )
        elif name == "tool-snap":
            shape_control = (
                "Ends",
                "##snap-endpoint-scale",
                state.snap_endpoint_scale,
                0.65,
                1.50,
                "%.2f × original",
                0.05,
            )
        elif name == "key-fit":
            shape_control = (
                "Arms",
                "##key-fit-arm-length",
                state.key_fit_arm_length,
                2.5,
                5.5,
                "%.2f u",
                0.10,
            )
        elif name == "playback-reset":
            shape_control = (
                "Head",
                "##reset-head-scale",
                state.reset_head_scale,
                0.65,
                2.0,
                "%.2f × original",
                0.05,
            )
        if shape_control is not None:
            label, item_id, value, minimum, maximum, display_format, step = shape_control
            draw.text(
                (origin[0] + 10.0 * scale, center_y + 48.0 * scale),
                CONCEPT_THEME.text_disabled,
                label,
            )
            imgui.set_cursor_screen_pos(
                imgui.ImVec2(float(control_x), float(center_y + 41.0 * scale))
            )
            imgui.set_next_item_width(control_width)
            changed, value = _stepped_slider(
                item_id,
                value,
                minimum,
                maximum,
                display_format,
                step=step,
            )
            if name == "playback-reset":
                imgui.set_item_tooltip(
                    "Shared reset/refresh arrowhead size. Scales the head length and width; "
                    "the Stroke control sets the arc weight."
                )
            if changed:
                if name == "tool-rotate":
                    state.rotate_ring_gap_ratio = value
                elif name == "tool-move":
                    state.move_head_scale = value
                elif name == "tool-scale":
                    state.scale_handle_scale = value
                elif name == "tool-snap":
                    state.snap_endpoint_scale = value
                elif name == "playback-reset":
                    state.reset_head_scale = value
                else:
                    state.key_fit_arm_length = value
        if name in ICON_ALIGNMENT_EDITABLE_ICONS:
            draw.text(
                (origin[0] + 10.0 * scale, center_y + 48.0 * scale),
                CONCEPT_THEME.text_disabled,
                "Align",
            )
            imgui.set_cursor_screen_pos(
                imgui.ImVec2(float(control_x), float(center_y + 41.0 * scale))
            )
            imgui.set_next_item_width(control_width)
            changed, alignment_index = imgui.combo(
                "##glyph-alignment",
                ICON_ALIGNMENT_CHOICES.index(alignment or "circle"),
                tuple(value.title() for value in ICON_ALIGNMENT_CHOICES),
            )
            if changed:
                alignment = ICON_ALIGNMENT_CHOICES[alignment_index]
                state.set_icon_alignment_for_glyph(name, alignment)
        imgui.set_cursor_screen_pos(
            imgui.ImVec2(float(origin[0] + 216.0 * scale), float(center_y + 7.0 * scale))
        )
        if imgui.button("Default", imgui.ImVec2(82.0 * scale, 0.0)):
            state.icon_padding_by_glyph.pop(name, None)
            state.icon_stroke_by_glyph.pop(name, None)
            state.icon_alignment_by_glyph.pop(name, None)
            state.set_icon_manual_offset(name, None)
            if name == "tool-rotate":
                state.rotate_ring_gap_ratio = ICON_ROTATE_RING_GAP_RATIO
            elif name == "tool-move":
                state.move_head_scale = ICON_TUNING_DEFAULTS.move_head_scale
            elif name == "tool-scale":
                state.scale_handle_scale = ICON_TUNING_DEFAULTS.scale_handle_scale
            elif name == "tool-snap":
                state.snap_endpoint_scale = ICON_TUNING_DEFAULTS.snap_endpoint_scale
            elif name == "key-fit":
                state.key_fit_arm_length = ICON_TUNING_DEFAULTS.key_fit_arm_length
            elif name == "playback-reset":
                state.reset_head_scale = ICON_TUNING_DEFAULTS.reset_head_scale
            padding = state.icon_padding_for_glyph(name)
            stroke_width = state.icon_stroke_for_glyph(name)
            alignment = state.icon_alignment_for_glyph(name)
        meta_x = origin[0] + 1160.0 * scale
        offset = list(state.icon_manual_offset(name))
        for axis in range(2):
            control_y = center_y + (8.0 + 28.0 * axis) * scale
            draw.text((meta_x, control_y + 6.0 * scale), CONCEPT_THEME.text_disabled, "XY"[axis])
            imgui.set_cursor_screen_pos((meta_x + 22.0 * scale, control_y))
            imgui.set_next_item_width(178.0 * scale)
            imgui.begin_disabled(state.icon_auto_align)
            changed, offset[axis] = imgui.drag_float(
                f"##glyph-offset-{'xy'[axis]}",
                offset[axis],
                v_speed=0.01,
                v_min=-4.0,
                v_max=4.0,
                format="%+.2f U",
                flags=imgui.SliderFlags_.always_clamp,
            )
            imgui.end_disabled()
            if changed:
                state.set_icon_manual_offset(name, tuple(offset))
            imgui.set_item_tooltip(
                "Manual offset on the 24-unit grid. +X is right; +Y is down. "
                "Drag to fine-tune; double-click or Ctrl+click to type a value. "
                "Auto align temporarily previews the algorithm without changing this value."
            )
        imgui.pop_id()

        offset = state.icon_offset(name)
        centroid = state.icon_centroid(name) if state.show_icon_centroids else None

        for center_x, size in zip(centers, _ICON_REVIEW_SIZES, strict=True):
            _draw_concept_icon_specimen(
                draw,
                (center_x, center_y),
                size * scale,
                name,
                scale,
                padding,
                mouse_width,
                stroke_width,
                state.rotate_ring_gap_ratio,
                state.rotate_ring_cap,
                state.icon_tuning(),
                alignment,
                offset=offset,
                centroid=centroid,
            )

        clearance = _icon_review_metrics(
            name,
            padding,
            mouse_width,
            stroke_width,
            state.rotate_ring_gap_ratio,
            state.rotate_ring_cap,
            state.icon_tuning(),
            alignment,
        )[0]
        anchor = icon_alignment_anchor(name, alignment)
        anchor_x, anchor_y = icon_alignment_center(
            name,
            padding,
            mouse_width,
            stroke_width,
            state.rotate_ring_gap_ratio,
            state.rotate_ring_cap,
            state.icon_tuning(),
            alignment,
        )
        placement_label, placement_value = (
            ("frame", padding) if name == "tool-rotate" else ("pad", clearance)
        )
        for line, line_y in (
            (f"{name} · {placement_label} {placement_value:.2f}u", -54.0),
            (f"{anchor}  {anchor_x:+.2f},{anchor_y:+.2f}u", -34.0),
            (
                f"{'Auto' if state.icon_auto_align else 'Manual'}  {offset[0]:+.2f},{offset[1]:+.2f} U",
                -14.0,
            ),
        ):
            draw.text(
                (meta_x, center_y + line_y * scale),
                CONCEPT_THEME.text_disabled,
                line,
            )


def _draw_capsule_context_page(draw, origin, scale: float, state: ProbeState) -> None:
    """Show Icon Library candidates in their actual capsule cells and state circles."""

    x0, y0 = origin
    draw.text((x0, y0), CONCEPT_THEME.text, "Actual capsule placement")
    draw.text(
        (x0, y0 + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Amber = icon slot · green = state circle · component defaults plus per-glyph overrides",
    )

    def preview_state(**changes):
        return replace(
            state,
            redesign=replace(state.redesign),
            preview_icon_library=True,
            show_icon_bounds=True,
            show_state_circles=True,
            show_construction_notes=False,
            **changes,
        )

    playback_x = x0 + 24.0 * scale
    playback_y = y0 + 82.0 * scale
    draw.text((playback_x, playback_y - 28.0 * scale), CONCEPT_THEME.text, "Playback · 1×")
    imgui.push_id("icon-library-capsule-playback-1x")
    _draw_playback(draw, (playback_x, playback_y), scale, preview_state(playing=False))
    imgui.pop_id()

    playback_2x_y = playback_y + 116.0 * scale
    draw.text(
        (playback_x, playback_2x_y - 28.0 * scale),
        CONCEPT_THEME.text,
        "Playback · 2× · playing",
    )
    imgui.push_id("icon-library-capsule-playback-2x")
    _draw_playback(draw, (playback_x, playback_2x_y), scale * 2.0, preview_state(playing=True))
    imgui.pop_id()

    recording_y = playback_2x_y + 140.0 * scale
    draw.text(
        (playback_x, recording_y - 28.0 * scale),
        CONCEPT_THEME.text,
        "Playback · 1× · recording",
    )
    recording_state = preview_state(playing=True)
    recording_state.redesign.recording = "video"
    imgui.push_id("icon-library-capsule-playback-recording")
    _draw_playback(draw, (playback_x, recording_y), scale, recording_state)
    imgui.pop_id()

    metrics_x = x0 + 820.0 * scale
    draw.text((metrics_x, playback_y - 28.0 * scale), CONCEPT_THEME.text, "Placement diagnostics")
    draw.text(
        (metrics_x, playback_y - 4.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Declared anchor after the current manual or automatic offset",
    )
    for index, name in enumerate(
        (
            "playback-previous",
            "playback-play",
            "playback-pause",
            "playback-next",
            "playback-reset",
            "playback-record",
            "playback-stop",
            "playback-more",
        )
    ):
        anchor = icon_alignment_anchor(name)
        anchor_center = icon_alignment_center(
            name,
            state.icon_padding_for_glyph(name),
            stroke_width=state.icon_stroke_for_glyph(name),
            rotate_ring_gap_ratio=state.rotate_ring_gap_ratio,
            rotate_ring_cap=state.rotate_ring_cap,
        )
        offset = state.icon_offset(name)
        anchor_center = tuple(a + b for a, b in zip(anchor_center, offset, strict=True))
        draw.text(
            (metrics_x, playback_y + (30.0 + index * 25.0) * scale),
            CONCEPT_THEME.text_disabled,
            (f"{name:<20} {anchor:<6} {anchor_center[0]:+0.2f}, {anchor_center[1]:+0.2f}u"),
        )

    semantic_y = playback_y + 252.0 * scale
    draw.text((metrics_x, semantic_y), CONCEPT_THEME.text, "Applied semantic states")
    semantic_specs = (
        ("Default", "playback-play", "off", False),
        ("Hover", "playback-play", "hover", False),
        ("Press", "playback-next", "press", False),
        ("Selected", "playback-pause", "selected", False),
        ("Record", "playback-record", "off", True),
        ("Stop", "playback-stop", "selected", True),
    )
    for index, (label, name, interaction, danger) in enumerate(semantic_specs):
        position = (metrics_x + index * 92.0 * scale, semantic_y + 28.0 * scale)

        def semantic_icon(
            target,
            center,
            color,
            icon_scale,
            _surface,
            *,
            name=name,
            danger=danger,
        ):
            draw_concept_icon(
                target,
                state.icon_center(name, center, 20.0 * icon_scale),
                20.0 * icon_scale,
                name,
                THEME.viewport.record if danger else color,
                padding=state.icon_padding_for_glyph(name),
                stroke_width=state.icon_stroke_for_glyph(name),
                tuning=state.icon_tuning(),
                alignment=state.icon_alignment_for_glyph(name),
            )

        _circular_icon_button(
            draw,
            f"##capsule-semantic-{index}",
            position,
            semantic_icon,
            selected=interaction == "selected",
            cell_size=42.0,
            state_radius=18.0,
            icon_radius=10.0,
            icon_scale=scale,
            forced_interaction=interaction,
            scale=scale,
        )
        draw.centered_label(
            label,
            (position[0] + 21.0 * scale, semantic_y + 80.0 * scale),
            CONCEPT_THEME.text_disabled,
            82.0 * scale,
        )

    tools_y = y0 + 590.0 * scale
    draw.text((playback_x, tools_y), CONCEPT_THEME.text, "Viewport tools · actual vertical capsule")
    draw.text(
        (playback_x, tools_y + 24.0 * scale),
        CONCEPT_THEME.text_disabled,
        "World and Body use the same cell center; their internal strokes stop at the shell's inner edge.",
    )
    for index, (space, label) in enumerate((("world", "World"), ("body", "Body"))):
        tool_x = playback_x + index * 250.0 * scale
        tool_y = tools_y + 72.0 * scale
        draw.text((tool_x, tool_y - 26.0 * scale), CONCEPT_THEME.text_disabled, label)
        imgui.push_id(f"icon-library-capsule-tools-{space}")
        _draw_tool_column(
            draw,
            (tool_x, tool_y),
            scale * 1.5,
            preview_state(gizmo_space=space),
        )
        imgui.pop_id()

    draw.text(
        (metrics_x, tools_y),
        CONCEPT_THEME.text,
        "Application rules",
    )
    draw.text(
        (metrics_x, tools_y + 28.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Default uses Text; hover, press, and selected use Primary Bright.",
    )
    draw.text(
        (metrics_x, tools_y + 54.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Record and Stop keep the Danger red foreground in every interaction state.",
    )
    draw.text(
        (metrics_x, tools_y + 80.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Offsets move glyphs inside fixed capsule cells and state circles.",
    )


def _draw_follow_mode_preview(origin, scale: float, state: ProbeState) -> None:
    """Review compact production segments with both localized tooltip sets."""
    modes = ("off", "page", "locked")
    icons = tuple(name for _label, name in icon_family("Keyframe follow"))
    imgui.push_style_var(
        imgui.StyleVar_.frame_padding,
        (8 * scale, max(0, (28 * scale - imgui.get_font_size()) * 0.5)),
    )
    for column, (heading, language) in enumerate(
        (("English", Language.ENGLISH), ("简体中文", Language.SIMPLIFIED_CHINESE))
    ):
        localizer = Localizer(language)
        labels = tuple(localizer.text(text) for text in keyframes_panel_module.FOLLOW_MODE_TOOLTIPS)
        imgui.set_cursor_screen_pos((origin[0] + column * 660 * scale, origin[1]))
        imgui.begin_group()
        imgui.text_disabled(heading)
        selected = segmented_control(
            f"follow-icon-preview-{column}",
            labels,
            modes.index(state.timeline_panel.editor.follow_mode),
            width=segmented_control_width(labels, icons=icons, show_labels=False),
            theme=CONCEPT_THEME,
            icons=icons,
            icon_label_drawer=partial(_draw_icon_library_label, state=state),
            show_labels=False,
        )
        if modes[selected] != state.timeline_panel.editor.follow_mode:
            state.timeline_panel.editor.set_follow_mode(modes[selected])
        imgui.end_group()
    imgui.pop_style_var()
    imgui.set_cursor_screen_pos((origin[0], origin[1] + 64 * scale))
    imgui.text_disabled("Off: keep view  |  Page: jump at edge  |  Locked: hold playhead position")


def _draw_icon_library_page(
    draw, origin, available_width: float, scale: float, state: ProbeState
) -> None:
    """Render concept-only icon families without changing production painters."""

    x0, y0 = float(origin[0]), float(origin[1])
    draw.text((x0 + 42.0 * scale, y0 + 2.0 * scale), CONCEPT_THEME.text, "Mojive icon library")
    draw.text(
        (x0 + 42.0 * scale, y0 + 25.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Candidate vectors · 24-unit circular placement bound",
    )
    imgui.set_cursor_screen_pos(imgui.ImVec2(float(x0 + 258.0 * scale), float(y0 + 8.0 * scale)))
    if imgui.button("Copy icon parameters", imgui.ImVec2(178.0 * scale, 0.0)):
        imgui.set_clipboard_text(_icon_values_text(state))
    imgui.set_item_tooltip(
        "Copy geometry, every manual X/Y offset, and automatic comparison settings for review."
    )
    controls_x = x0 + max(450.0 * scale, available_width - 940.0 * scale)
    draw.text(
        (controls_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Group",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(controls_x + 58.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(190.0 * scale)
    group = _draw_icon_group_combo("##icon-library-group", state)
    padding_x = controls_x + 268.0 * scale
    draw.text(
        (padding_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Glyph pad",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(padding_x + 72.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(130.0 * scale)
    _deferred_icon_group_slider(
        "##icon-library-padding",
        state,
        group,
        kind="padding",
    )
    imgui.set_item_tooltip(
        "Circular clearance between candidate geometry and the orange placement circle. "
        "Rotate and the reviewed Info/Warning/Error family remain locked. Release to apply the group."
    )
    stroke_x = controls_x + 488.0 * scale
    draw.text(
        (stroke_x, y0 + 17.0 * scale),
        CONCEPT_THEME.text_disabled,
        "Glyph stroke",
    )
    imgui.set_cursor_screen_pos(
        imgui.ImVec2(float(stroke_x + 90.0 * scale), float(y0 + 8.0 * scale))
    )
    imgui.set_next_item_width(122.0 * scale)
    _deferred_icon_group_slider(
        "##icon-library-stroke",
        state,
        group,
        kind="stroke",
    )
    imgui.set_item_tooltip(
        "Default visual weight for the selected component group. Individual glyph rows can "
        "override it; reviewed Pause, First/Last, severity, and mouse geometry stay unchanged. "
        "Release to apply the group."
    )
    if group == "Status & input":
        mouse_x = controls_x + 718.0 * scale
        draw.text(
            (mouse_x, y0 + 17.0 * scale),
            CONCEPT_THEME.text_disabled,
            "Mouse W",
        )
        imgui.set_cursor_screen_pos(
            imgui.ImVec2(float(mouse_x + 80.0 * scale), float(y0 + 8.0 * scale))
        )
        imgui.set_next_item_width(135.0 * scale)
        state.hint_mouse_width = _even_slider(
            "##icon-library-mouse-width", state.hint_mouse_width, 12, 24
        )
        imgui.set_item_tooltip(
            "Visible mouse width used by Status & input specimens and whole-UI previews."
        )
    imgui.set_cursor_screen_pos(imgui.ImVec2(x0 + 42.0 * scale, y0 + 54.0 * scale))
    state.icon_library_tab = _wrapped_tabs(
        tuple(
            (label, "icon-library", label.casefold().replace(" ", "-").replace("&", "and"))
            for label in ICON_LIBRARY_TABS
        ),
        state.icon_library_tab,
        max(1.0, available_width - 84.0 * scale),
        gap=4.0 * scale,
    )
    if state.icon_library_tab in ICON_GROUP_LAYOUT_DEFAULTS:
        state.icon_adjustment_group = state.icon_library_tab
    elif state.icon_library_tab == "Keyframe follow":
        state.icon_adjustment_group = "Keyframes"
    imgui.set_cursor_screen_pos((x0 + 42.0 * scale, imgui.get_cursor_screen_pos().y))
    _, state.icon_auto_align = imgui.checkbox("Auto align", state.icon_auto_align)
    imgui.set_item_tooltip(
        "Temporarily preview alpha-weighted centroid alignment. Manual X/Y values are preserved."
    )
    imgui.same_line()
    _, state.show_icon_centroids = imgui.checkbox("Centroid guides", state.show_icon_centroids)
    imgui.same_line()
    _, state.link_mirrored_icon_offsets = imgui.checkbox(
        "Link mirrored offsets", state.link_mirrored_icon_offsets
    )
    imgui.set_item_tooltip(
        "Editing either partner mirrors X and copies Y: Playback Previous/Next, Transport "
        "Previous/Next and First/Last, Previous/Next key, Mouse Left/Right. "
        "Existing values stay unchanged until an edit or reset. Auto align remains independent."
    )
    imgui.same_line()
    imgui.begin_disabled(not state.icon_auto_align)
    imgui.align_text_to_frame_padding()
    imgui.text("Strength")
    imgui.same_line()
    imgui.set_next_item_width(200.0 * scale)
    _, strength = _stepped_slider(
        "##library-alignment-strength",
        state.icon_alignment_strength * 100,
        0,
        250,
        "%.0f %%",
        step=1,
    )
    state.icon_alignment_strength = strength / 100
    imgui.end_disabled()
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + 42.0 * scale)
    imgui.text_disabled(
        "X/Y: manual offsets in 24-unit grid. Auto align previews the algorithm; amber dots mark ink centroids."
    )
    content_y = float(imgui.get_cursor_screen_pos().y) + 14.0 * scale
    content_origin = (x0 + 42.0 * scale, content_y)
    if state.icon_library_tab == "Overview":
        _draw_icon_library_overview(draw, content_origin, scale, state)
    elif state.icon_library_tab == "UI context":
        _draw_icon_context_page(draw, content_origin, scale, state)
    elif state.icon_library_tab == "Capsules":
        _draw_capsule_context_page(draw, content_origin, scale, state)
    else:
        if state.icon_library_tab == "Keyframe follow":
            _draw_follow_mode_preview(content_origin, scale, state)
            content_origin = (content_origin[0], content_origin[1] + 110.0 * scale)
        _draw_icon_family_detail(
            draw,
            content_origin,
            state.icon_library_tab,
            scale,
            state,
        )
