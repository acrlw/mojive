"""Expose registered panel visibility without maintaining a second open-state store."""

from collections.abc import MutableMapping

from .constants import PANELS


class PanelVisibility(MutableMapping):
    """Keep native names as a view over the public Panel.open/enabled state."""

    def __init__(self, panels):
        self.panels = panels
        self.names = (*PANELS, "Inspector", "Timeline", "Plot")

    def __getitem__(self, name):
        if name not in self.names:
            raise KeyError(name)
        panel = self.panels.get(name)
        return panel is not None and panel.enabled and panel.open

    def __setitem__(self, name, value):
        if name not in self.names:
            raise KeyError(name)
        panel = self.panels.get(name)
        if panel is not None:
            panel.open = bool(value) and panel.enabled

    def __delitem__(self, name):
        self[name] = False

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)
