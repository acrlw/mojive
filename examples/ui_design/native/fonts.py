"""Preserve the macOS reference and resolve portable fonts without downloads."""

from pathlib import Path

import imgui_bundle

from mojive.text.sources import FontSource, resolve_font_sources


def resolve_fonts(*, portable=False):
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
    for source in sources:
        if not Path(source.path).is_file():
            raise FileNotFoundError(f"Design study font is unavailable: {source.path}")
    # Roboto and the numeric font use ordinary em sizing, unlike the SF reference.
    return tuple((source, 1.0) for source in sources)
