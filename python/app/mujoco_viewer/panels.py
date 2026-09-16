"""Map MuJoCo's two UI switches to discoverable Mojive dock groups."""

from dataclasses import replace

from mojive.config import LayoutConfig, PanelConfig, ViewerConfig

LEFT_PANELS = ("hierarchy", "assets", "keyframes", "output", "stats", "plot", "help", "info")
RIGHT_PANELS = ("control", "joints", "camera", "settings", "sensors", "inspector", "layers")


def viewer_config(config, show_left_ui, show_right_ui):
    if not isinstance(show_left_ui, bool) or not isinstance(show_right_ui, bool):
        raise TypeError("show_left_ui and show_right_ui must be bools")
    config = config or ViewerConfig(layout=LayoutConfig(persistence=False))
    panels = dict(config.panels)
    for shown, names in ((show_left_ui, LEFT_PANELS), (show_right_ui, RIGHT_PANELS)):
        if not shown:
            for name in names:
                panels[name] = replace(panels.get(name, PanelConfig()), open=False)
    return replace(config, panels=panels, threaded_physics=False)


def install_key_callback(viewer, callback):
    import glfw

    # Preserve deliberate user remaps; new users get the familiar middle-button dolly.
    if not viewer.app.localizer.preference("input_bindings", {}):
        viewer.app.set_navigation_preset("MuJoCo", persist=False)
    groups = (LEFT_PANELS, RIGHT_PANELS)
    saved = {}
    for names in groups:
        panels = [viewer.panels.get(name) for name in names]
        visible = any(panel is not None and panel.open for panel in panels)
        saved[names] = {
            name: (panel.open if visible else panel.default_open)
            for name, panel in zip(names, panels, strict=True)
            if panel is not None
        }

    def key_event(window, key, scancode, action, mods):
        if previous is not None:
            previous(window, key, scancode, action, mods)
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_TAB:
            names = RIGHT_PANELS if mods & glfw.MOD_SHIFT else LEFT_PANELS
            panels = [viewer.panels.get(name) for name in names]
            shown = any(panel is not None and panel.enabled and panel.open for panel in panels)
            if shown:
                saved[names] = {
                    name: panel.open
                    for name, panel in zip(names, panels, strict=True)
                    if panel is not None
                }
            for name in names:
                viewer.panels.set_open(name, False if shown else saved[names].get(name, False))
        if callback is not None:
            callback(key)

    previous = glfw.set_key_callback(viewer.window._window, key_event)
