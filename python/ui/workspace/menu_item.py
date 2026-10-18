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
    checkable: bool = False

    def __post_init__(self):
        if self.checked and not self.checkable:
            object.__setattr__(self, "checkable", True)
