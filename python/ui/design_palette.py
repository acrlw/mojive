"""Expose Mojive's existing native colors to the parameter editor and painters."""

from mojive.ui import theme


def _hex(ink):
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in ink[:3])


PALETTE = {
    "brand": _hex(theme.PRIMARY),
    "primary": _hex(theme.PRIMARY),
    "primary_bright": _hex(theme.PRIMARY_BRIGHT),
    "primary_dim": _hex(theme.PRIMARY_DIM),
    "selected": _hex(theme.BG_HEADER),
    "background": "#1E2125",
    "floating": "#252A2F",
    "child": "#1A1D20",
    "popup": "#2D353D",
    "field": "#2B2F34",
    "field_hovered": "#363B41",
    "field_active": "#40474E",
    "header": _hex(theme.BG_HEADER),
    "border": _hex(theme.BORDER),
    "text": "#E6E8EB",
    "secondary": "#A9AFB7",
    "disabled": "#8E98A5",
    "info": _hex(theme.INFO),
    "success": _hex(theme.PRIMARY),
    "warning": "#E8B04F",
    "danger": _hex(theme.DANGER),
    "purple": _hex(theme.ACCENT_PURPLE),
}


def rgba8(role: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = PALETTE[role].removeprefix("#")
    return (*(int(value[index : index + 2], 16) for index in (0, 2, 4)), alpha)


def rgba(role: str) -> tuple[float, float, float, float]:
    return tuple(channel / 255 for channel in rgba8(role))
