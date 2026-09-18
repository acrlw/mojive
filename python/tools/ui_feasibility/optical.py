"""Context comparisons and cached silhouette diagnostics for production icons."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.ui.icons import (
    ICON_FAMILIES,
    ICON_GRID,
    draw_icon,
    draw_icon_label,
    icon_alignment_anchor,
    minimum_enclosing_circle,
    production_icon_metrics,
    production_icon_offset,
)
from mojive.ui.theme import THEME

from .icon_alignment import rasterized_ink, update_manual_offset

FAMILIES = {
    family: tuple(
        item for item in icons if item[1] not in {"status-info", "status-warning", "status-error"}
    )
    for family, icons in ICON_FAMILIES
}
SPECIMENS = tuple(item for icons in FAMILIES.values() for item in icons)
SPECIMEN_INDICES = {name: index for index, (_, name) in enumerate(SPECIMENS)}
_FAMILY_NAMES = tuple(FAMILIES)
_FAMILY_BY_GLYPH = {name: family for family, icons in FAMILIES.items() for _, name in icons}
_CANVAS = 80.0
_RESOLUTION = 256


@dataclass
class OpticalStudy:
    selected: int = SPECIMEN_INDICES["helper-camera"]
    offsets: dict[str, tuple[float, float]] = field(default_factory=dict)
    guides: bool = False
    blurred: bool = True
    sigma: float = 3.0
    threshold: float = 0.25
    auto_align: bool = False
    alignment_strength: float = 1.0
    link_mirrored_offsets: bool = False

    @property
    def family(self) -> str:
        return _FAMILY_BY_GLYPH[SPECIMENS[self.selected][1]]

    def set_offset(self, name: str, value: tuple[float, float] | None) -> None:
        update_manual_offset(self.offsets, name, value, link_mirrored=self.link_mirrored_offsets)

    def offset(self, name: str, candidate: bool = True) -> tuple[float, float]:
        return self.offsets.get(name, (0.0, 0.0)) if candidate else (0.0, 0.0)

    def effective_offset(self, name: str, candidate: bool = True) -> tuple[float, float]:
        if candidate and self.auto_align:
            x, y = _ink_mask(name)[1]
            return -x * self.alignment_strength, -y * self.alignment_strength
        return self.offset(name, candidate)

    def export(self) -> str:
        offsets = (
            {name: self.effective_offset(name) for _, name in SPECIMENS}
            if self.auto_align
            else self.offsets
        )
        values = {
            "grid_units": ICON_GRID,
            "alignment": "weighted-centroid" if self.auto_align else "manual",
            "icon_offsets": offsets,
            "link_mirrored_offsets": self.link_mirrored_offsets,
        }
        if self.auto_align:
            values["alignment_strength"] = self.alignment_strength
        return json.dumps(values, indent=2)


@lru_cache(maxsize=len(SPECIMENS))
def _ink_mask(name: str):
    return rasterized_ink(name)


def _gaussian_blur(alpha: np.ndarray, sigma_pixels: float) -> np.ndarray:
    """Convolve a normalized Gaussian in X/Y, with transparent padding and a 4-sigma kernel."""
    if sigma_pixels == 0:
        return alpha
    radius = int(np.ceil(4 * sigma_pixels))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_pixels) ** 2)
    kernel /= kernel.sum()
    result = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, alpha)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, result)


@lru_cache(maxsize=32)
def _blurred_mask(name: str, sigma: float):
    return _gaussian_blur(_ink_mask(name)[0], sigma * _RESOLUTION / _CANVAS)


@lru_cache(maxsize=64)
def silhouette(name: str, sigma: float = 3.0, threshold: float = 0.25):
    """Return Gaussian alpha, ink centroid and the circle of thresholded sample centers."""
    blurred = _blurred_mask(name, sigma)
    support = blurred >= float(blurred.max()) * threshold
    boundary = []
    for y, row in enumerate(support):
        xs = np.flatnonzero(row)
        if len(xs):
            for x in (xs[0], xs[-1]):
                boundary.append(
                    (
                        (x + 0.5) * _CANVAS / _RESOLUTION - _CANVAS / 2,
                        (y + 0.5) * _CANVAS / _RESOLUTION - _CANVAS / 2,
                    )
                )
    circle = minimum_enclosing_circle(boundary)
    return blurred, _ink_mask(name)[1], circle


@lru_cache(maxsize=32)
def _blur_quads(name, sigma, zoom):
    # Interpolate coverage between samples, rather than enlarging flat pixel blocks.
    # Cached colored quads use ImGui's existing draw path on either renderer.
    resolution = 128
    mask = np.asarray(
        Image.fromarray(_blurred_mask(name, sigma).astype(np.float32)).resize(
            (resolution, resolution), Image.Resampling.BILINEAR
        )
    )
    rgba = imgui.get_color_u32(THEME.text)
    alpha = np.rint(mask * (rgba >> 24)).astype(np.uint32)
    colors = (alpha << 24) | (rgba & 0xFFFFFF)
    visible = (alpha[:-1, :-1] | alpha[:-1, 1:] | alpha[1:, :-1] | alpha[1:, 1:]) != 0
    step = _CANVAS / resolution * zoom
    return tuple(
        (
            (x + 0.5 - resolution / 2) * step,
            (y + 0.5 - resolution / 2) * step,
            step,
            int(colors[y, x]),
            int(colors[y, x + 1]),
            int(colors[y + 1, x + 1]),
            int(colors[y + 1, x]),
        )
        for y, x in np.argwhere(visible)
    )


def _draw_blur(study, name, center, zoom):
    draw_list = imgui.get_window_draw_list()
    for x, y, step, c00, c10, c11, c01 in _blur_quads(name, study.sigma, zoom):
        draw_list.add_rect_filled_multi_color(
            (center[0] + x, center[1] + y),
            (center[0] + x + step, center[1] + y + step),
            c00,
            c10,
            c11,
            c01,
        )


def _guide(draw, center, extent, scale):
    color = (*THEME.warning[:3], 0.5)
    x, y = center
    draw.line((x - extent, y), (x + extent, y), color, scale)
    draw.line((x, y - extent), (x, y + extent), color, scale)


def _glyph(draw, center, size, name, study, candidate):
    dx, dy = study.effective_offset(name, candidate)
    draw_icon(
        draw,
        (center[0] + dx * size / ICON_GRID, center[1] + dy * size / ICON_GRID),
        size,
        name,
        THEME.text,
    )


def _settings(study, scale):
    columns = 4 if imgui.get_content_region_avail().x >= 650 * scale else 1
    _, study.auto_align = imgui.checkbox("Auto-align weighted centroid", study.auto_align)
    imgui.set_item_tooltip(
        "Correct each glyph's alpha-weighted ink centroid using the strength below. "
        "Manual offsets are preserved."
    )
    if columns > 1:
        imgui.same_line()
    if imgui.button("Previous icon"):
        study.selected = (study.selected - 1) % len(SPECIMENS)
    imgui.same_line()
    if imgui.button("Next icon"):
        study.selected = (study.selected + 1) % len(SPECIMENS)
    imgui.begin_disabled(not study.auto_align)
    imgui.align_text_to_frame_padding()
    imgui.text("Alignment strength")
    imgui.same_line()
    imgui.set_next_item_width(-1)
    _, percent = imgui.slider_float(
        "##optical-strength",
        study.alignment_strength * 100,
        0.0,
        250.0,
        "%.0f %%",
        imgui.SliderFlags_.always_clamp,
    )
    study.alignment_strength = percent / 100
    imgui.set_item_tooltip(
        "0%: production position. 50%: half correction. "
        "100%: centroid at control center. 250%: 2.5x correction. "
        "Strength scales a fixed direction; switch off auto-align for independent X/Y adjustment."
    )
    imgui.end_disabled()
    if imgui.begin_table("##optical-settings", columns):
        imgui.table_next_column()
        imgui.text("Family")
        imgui.set_next_item_width(-1)
        changed, family_index = imgui.combo(
            "##optical-family", _FAMILY_NAMES.index(study.family), _FAMILY_NAMES
        )
        if changed:
            study.selected = SPECIMEN_INDICES[FAMILIES[_FAMILY_NAMES[family_index]][0][1]]
        family_icons = FAMILIES[study.family]
        family_indices = tuple(SPECIMEN_INDICES[name] for _, name in family_icons)
        imgui.table_next_column()
        imgui.text("Choose icon")
        imgui.set_next_item_width(-1)
        _, glyph_index = imgui.combo(
            "##optical-glyph",
            family_indices.index(study.selected),
            [label for label, _ in family_icons],
        )
        study.selected = family_indices[glyph_index]
        name = SPECIMENS[study.selected][1]
        values = list(study.effective_offset(name))
        for axis in range(2):
            imgui.table_next_column()
            imgui.text(f"{'Auto' if study.auto_align else 'Candidate'} {'XY'[axis]} offset")
            imgui.set_next_item_width(-1)
            imgui.begin_disabled(study.auto_align)
            changed, values[axis] = imgui.slider_float(
                f"##optical-{'xy'[axis]}", values[axis], -4.0, 4.0, "%+.2f U"
            )
            if changed:
                study.set_offset(name, tuple(values))
            imgui.end_disabled()
        imgui.end_table()
    _, study.guides = imgui.checkbox("Alignment guides", study.guides)
    imgui.same_line()
    _, study.blurred = imgui.checkbox("Blur diagnostic", study.blurred)
    if columns > 1:
        imgui.same_line()
    _, study.link_mirrored_offsets = imgui.checkbox(
        "Link mirrored offsets", study.link_mirrored_offsets
    )
    imgui.set_item_tooltip(
        "Manual edits mirror X and copy Y to Previous/Next, First/Last, or Mouse Left/Right. Existing values stay until edited."
    )
    if columns > 1:
        imgui.same_line()
    imgui.begin_disabled(study.auto_align)
    if imgui.button("Reset selected"):
        study.set_offset(SPECIMENS[study.selected][1], None)
    imgui.end_disabled()
    imgui.same_line()
    if imgui.button("Copy offsets"):
        imgui.set_clipboard_text(study.export())
    if study.blurred and imgui.begin_table("##blur-settings", 2 if columns > 1 else 1):
        imgui.table_next_column()
        imgui.text("Gaussian blur / sigma")
        imgui.set_next_item_width(-1)
        _, study.sigma = imgui.slider_float("##optical-sigma", study.sigma, 0.0, 6.0, "%.2f U")
        imgui.table_next_column()
        imgui.text("Circle contour / fraction of peak")
        imgui.set_next_item_width(-1)
        _, percent = imgui.slider_float(
            "##optical-threshold", study.threshold * 100, 5.0, 80.0, "%.0f %%"
        )
        study.threshold = percent / 100
        imgui.end_table()


def _contexts(ctx, study, candidate):
    draw, scale = ctx.painter(), ctx.style_scale
    label, name = SPECIMENS[study.selected]
    imgui.text("Buttons / 16, 20, 24 px")
    origin = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    for index, size in enumerate((16, 20, 24)):
        x = origin.x + width * (index + 0.5) / 3
        for row in range(2):
            center = (x, origin.y + (22 + row * 46) * scale)
            if row == 0:
                draw.circle_filled(center, 20 * scale, THEME.bg_frame)
            else:
                draw.rect_filled(
                    (x - 20 * scale, center[1] - 20 * scale),
                    (x + 20 * scale, center[1] + 20 * scale),
                    THEME.bg_frame,
                    rounding=4 * scale,
                )
            _glyph(draw, center, size * scale, name, study, candidate)
            if study.guides:
                _guide(draw, center, 20 * scale, scale)
    imgui.dummy((width, 94 * scale))
    imgui.text("Icon + text / fixed text placement")
    for text in (label, "相机" if name == "helper-camera" else "示例标签"):
        lo = imgui.get_cursor_screen_pos()
        hi = (lo.x + width, lo.y + imgui.get_frame_height())
        draw.rect_filled(lo, hi, THEME.bg_frame, rounding=4 * scale)
        offset = tuple(
            value * 16 * scale / ICON_GRID for value in study.effective_offset(name, candidate)
        )
        draw_icon_label(draw, lo, hi, THEME.text, scale, name, text, icon_offset=offset)
        if study.guides:
            draw.line(
                (lo.x, (lo.y + hi[1]) / 2),
                (hi[0], (lo.y + hi[1]) / 2),
                (*THEME.warning[:3], 0.5),
                scale,
            )
        imgui.dummy((width, imgui.get_frame_height()))
    family_icons = FAMILIES[study.family]
    imgui.text(f"Toolbar / {study.family}")
    origin = imgui.get_cursor_screen_pos()
    count = 6
    for index, (_, glyph) in enumerate(family_icons):
        center = (
            origin.x + width * (index % count + 0.5) / min(count, len(family_icons)),
            origin.y + (18 + 36 * (index // count)) * scale,
        )
        draw.circle_filled(center, 15 * scale, THEME.bg_header if glyph == name else THEME.bg_frame)
        _glyph(draw, center, 20 * scale, glyph, study, candidate)
    imgui.dummy((width, (36 * ((len(family_icons) + count - 1) // count) + 4) * scale))


def _diagnostic(ctx, study, candidate):
    draw, scale = ctx.painter(), ctx.style_scale
    name = SPECIMENS[study.selected][1]
    imgui.text("Enlarged geometry / Gaussian blur" if study.blurred else "3x geometry")
    origin = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    count = 2 if study.blurred else 1
    zoom = min(3 * scale, (width / count - 8 * scale) / _CANVAS)
    extent = _CANVAS * zoom / 2
    dx, dy = study.effective_offset(name, candidate)
    for index in range(count):
        center = (
            origin.x + width * (index + 0.5) / (2 if study.blurred else 1),
            origin.y + extent + 4 * scale,
        )
        shifted = (center[0] + dx * zoom, center[1] + dy * zoom)
        draw.circle_filled(center, 20 * zoom, THEME.bg_frame)
        if index == 0:
            _glyph(draw, center, ICON_GRID * zoom, name, study, candidate)
            if study.guides:
                x0, y0, x1, y1 = production_icon_metrics(name).bounds
                ox, oy = production_icon_offset(name)
                draw.rect(
                    (shifted[0] + (x0 + ox) * zoom, shifted[1] + (y0 + oy) * zoom),
                    (shifted[0] + (x1 + ox) * zoom, shifted[1] + (y1 + oy) * zoom),
                    (*THEME.text_disabled[:3], 0.8),
                    scale,
                )
        else:
            _draw_blur(study, name, shifted, zoom)
            if study.guides:
                _, _, (circle, radius) = silhouette(name, study.sigma, study.threshold)
                circle_center = (shifted[0] + circle[0] * zoom, shifted[1] + circle[1] * zoom)
                draw.circle(
                    (shifted[0] + circle[0] * zoom, shifted[1] + circle[1] * zoom),
                    radius * zoom,
                    (*THEME.text_disabled[:3], 0.8),
                    scale,
                )
                draw.rect(
                    (circle_center[0] - 2 * scale, circle_center[1] - 2 * scale),
                    (circle_center[0] + 2 * scale, circle_center[1] + 2 * scale),
                    THEME.text,
                    scale,
                )
        if study.guides:
            _guide(draw, center, 20 * zoom, scale)
            centroid = _ink_mask(name)[1]
            draw.circle_filled(
                (shifted[0] + centroid[0] * zoom, shifted[1] + centroid[1] * zoom),
                2 * scale,
                THEME.warning,
            )
    imgui.dummy((width, extent * 2 + 8 * scale))


def _column(ctx, study, candidate):
    imgui.push_id("optical-candidate" if candidate else "optical-current")
    imgui.text(
        (
            f"B / Centroid correction {study.alignment_strength:.0%}"
            if study.auto_align
            else "B / Candidate"
        )
        if candidate
        else "A / Production"
    )
    name = SPECIMENS[study.selected][1]
    dx, dy = study.effective_offset(name, candidate)
    imgui.text_disabled(f"Offset {dx:+.2f}, {dy:+.2f} U")
    imgui.separator()
    _contexts(ctx, study, candidate)
    _diagnostic(ctx, study, candidate)
    imgui.pop_id()


def draw_optical(ctx, study):
    imgui.text_wrapped(
        f"Review {len(SPECIMENS)} glyphs by family, or step through every glyph with Previous/Next. "
        "Offsets use the 24-unit grid; +X is right, +Y is down."
    )
    _settings(study, ctx.style_scale)
    imgui.separator()
    columns = 2 if imgui.get_content_region_avail().x >= 800 * ctx.style_scale else 1
    if imgui.begin_table("##optical-comparison", columns):
        for candidate in (False, True):
            imgui.table_next_column()
            _column(ctx, study, candidate)
        imgui.end_table()
    name = SPECIMENS[study.selected][1]
    imgui.text_wrapped(
        f"Production anchor: {icon_alignment_anchor(name)}. "
        "Guides: cross = control center; box = visible bounds; amber dot = ink centroid; "
        "small square = circle center. The contour threshold changes the circle, not the blur."
    )
    if study.blurred:
        circle, _ = silhouette(name, study.sigma, study.threshold)[2]
        imgui.text_wrapped(
            f"Circle-center trial offset: X {-circle[0]:+.2f}, Y {-circle[1]:+.2f} U (not applied). "
            "Gaussian blur spreads and fades ink. Its whole-image centroid stays unchanged "
            "without clipping; a thresholded contour can have a different center."
        )
    imgui.text_wrapped(
        "Strength: 0% = production position, 100% = centroid at control center, >100% = overshoot. "
        "Auto-align preserves manual offsets; blur and contour do not affect alignment. "
        "Hide guides to judge small controls. Copy offsets exports this experiment only."
    )
