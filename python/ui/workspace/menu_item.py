"""One native menu row; availability is supplied by the menu catalogue."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    label: str
    action: str = ""
    icon: str = ""
    shortcut: str = ""
    enabled: bool = True
    checked: bool = False
    submenu: str = ""
    reason: str = ""
