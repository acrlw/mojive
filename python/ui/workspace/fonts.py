"""Resolve matching Latin and Simplified Chinese faces without downloads."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imgui_bundle
from imgui_bundle import imgui

from mojive.text.sources import FontSource, resolve_font_sources


@dataclass(frozen=True, slots=True)
class FontFace(FontSource):
    """Use a synthesized heading weight only when the family has no installed bold face."""

    embolden: bool = False


def font_vertical_offset(source, size, config, *, ascent=None):
    """Center the actual face's cap height in native single-line text controls."""
    return _font_vertical_offset(
        source.path,
        source.index,
        float(size),
        config.extra_size_scale,
        config.font_loader_flags,
        config.rasterizer_density,
        imgui.get_io().display_framebuffer_scale.x,
        ascent,
        imgui.get_io().fonts.font_loader_name,
    )


@lru_cache(maxsize=96)
def _font_vertical_offset(
    path, index, size, ratio, flags, density, framebuffer_density, ascent, loader
):
    # Cache metrics, never a font or context. Measuring through the current loader
    # also covers installed SF named instances and synthesized heading weights.
    atlas = imgui.get_io().fonts
    config = imgui.ImFontConfig()
    config.font_no = index
    config.extra_size_scale = ratio
    config.font_loader_flags = flags
    config.rasterizer_density = density
    probe = atlas.add_font_from_file_ttf(path, size, config)
    try:
        baked = probe.get_font_baked(size, framebuffer_density)
        glyph = baked.find_glyph_no_fallback(ord("H"))
        if glyph is None:
            raise ValueError(f"Design font has no cap-height reference: {path}")
        center = (glyph.y0 + glyph.y1) / 2
        if ascent is not None:
            center += ascent - baked.ascent
        return baked.size / 2 - center
    finally:
        atlas.remove_font(probe)


def font_em_ratio(source: FontSource) -> float:
    """Convert FreeType's ascender/descent height to the known family's em size."""
    name = Path(source.path).name
    if name.startswith(("NotoSansCJK-", "NotoSansSC-", "SourceHanSansSC-")):
        return 1448 / 1000
    if name in ("Roboto-Regular.ttf", "Roboto-Bold.ttf"):
        return 2400 / 2048
    return 1.0


def resolve_cjk_faces(source):
    """Match body and heading weights and select the SC face in Noto's collection."""
    if source is None:
        return None, None
    path = Path(source.path)
    index = 2 if path.name.startswith("NotoSansCJK-") and path.suffix == ".ttc" else source.index
    family = {
        "NotoSansCJK": "Noto Sans CJK SC",
        "NotoSansSC": "Noto Sans SC",
        "SourceHanSansSC": "Source Han Sans SC",
    }.get(path.name.split("-", 1)[0], source.label)
    regular = FontFace(str(path), index, family + " Regular")
    bold = path.with_name(path.name.replace("Regular", "Bold"))
    if bold != path and bold.is_file():
        return regular, FontFace(str(bold), index, family + " Bold")
    return regular, FontFace(str(path), index, family + " Bold (synthesized)", True)


def resolve_fonts(*, portable=False, cjk=None):
    """Return body, heading and numeric sources with their em-size conversion."""
    sans = Path("/System/Library/Fonts/SFNS.ttf")
    mono = Path("/System/Library/Fonts/SFNSMono.ttf")
    if not portable and sans.is_file() and mono.is_file():
        # SF uses FreeType named instances and ascent/descent sizing on this family.
        return (
            (FontSource(str(sans), 4 << 16, "SF Regular"), 2412 / 2048),
            (FontSource(str(sans), 6 << 16, "SF Semibold"), 2412 / 2048),
            (FontSource(str(mono), 2 << 16, "SF Mono Regular"), 2412 / 2048),
        )
    bundled = Path(imgui_bundle.__file__).parent / "assets/fonts"
    try:
        numeric = resolve_font_sources(allow_download=False)[0]
    except FileNotFoundError:
        numeric = FontSource(str(bundled / "Inconsolata-Medium.ttf"), 0, "Inconsolata")
    sources = (
        FontSource(str(bundled / "Roboto/Roboto-Regular.ttf"), 0, "Roboto Regular"),
        FontSource(str(bundled / "Roboto/Roboto-Bold.ttf"), 0, "Roboto Bold"),
        numeric,
    )
    if (
        not portable
        and cjk is not None
        and Path(cjk.path).name.startswith(("NotoSansCJK-", "NotoSansSC-", "SourceHanSansSC-"))
    ):
        regular, heading = resolve_cjk_faces(cjk)
        sources = (regular, heading, numeric)
    for source in sources:
        if not Path(source.path).is_file():
            raise FileNotFoundError(f"Design workspace font is unavailable: {source.path}")
    # ImGui FreeType requests REAL_DIM, not nominal em height. Keep numeric
    # metrics compact while making body and heading sizes match their labels.
    return tuple(
        (source, font_em_ratio(source) if index < 2 else 1.0)
        for index, source in enumerate(sources)
    )
