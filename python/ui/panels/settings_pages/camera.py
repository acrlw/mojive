"""Camera settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.config import CAMERA_FOCUS_EASINGS, CAMERA_NAVIGATION_RANGES, CameraNavigationConfig
from mojive.ui.panels import (
    PanelContext,
    pointer_pressed,
)
from mojive.ui.pointer_bindings import PointerAction


def draw(panel, ctx: PanelContext) -> None:
    if ctx.camera is None or ctx.set_camera_navigation is None:
        return
    t = ctx.tr
    config = ctx.camera.navigation
    defaults = CameraNavigationConfig()

    def number(name, label, speed, fmt, tooltip):
        nonlocal config
        panel._property(t(label))
        low, high = CAMERA_NAVIGATION_RANGES[name]
        if name == "min_distance":
            high = config.max_distance
        elif name == "max_distance":
            low = config.min_distance
        changed, value = panel._drag_float(
            f"##camera_{name}",
            getattr(config, name),
            speed,
            low,
            high,
            fmt,
            imgui.SliderFlags_.always_clamp,
        )
        committed = panel._value_committed()
        reset = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.VALUE_RESET)
        if reset:
            changed, value = True, min(high, max(low, getattr(defaults, name)))
        panel._tooltip(t(tooltip))
        if changed:
            try:
                config = replace(config, **{name: value})
            except ValueError as error:
                ctx.report(str(error))
            else:
                ctx.apply_setting(ctx.set_camera_navigation, config, persist=False)
        if committed or reset:
            ctx.apply_setting(ctx.set_camera_navigation, config)

    def choice(name, label, choices):
        nonlocal config
        panel._property(t(label), control_part="select")
        current = getattr(config, name)
        values, labels = zip(*choices, strict=True)
        changed, selected = panel._combo(
            f"##camera_{name}", values.index(current), tuple(t(text) for text in labels)
        )
        if changed:
            config = replace(config, **{name: values[selected]})
            ctx.apply_setting(ctx.set_camera_navigation, config)

    panel._group_heading(t("Focus"))
    if panel._begin_properties("settings_camera_focus"):
        number(
            "focus_margin",
            "Focus distance",
            0.01,
            "%.2fx",
            "1x fits the object tightly; larger values leave more space around it.",
        )
        number(
            "focus_duration",
            "Transition duration",
            0.01,
            "%.2f s",
            "Duration of object and joint focus. Set to 0 for an instant transition.",
        )
        choice(
            "focus_easing",
            "Transition curve",
            tuple(
                zip(
                    CAMERA_FOCUS_EASINGS,
                    ("Linear", "Smoothstep", "Smootherstep", "Ease out cubic"),
                    strict=True,
                )
            ),
        )
        panel._end_properties()
    panel._group_heading(t("Zoom"))
    if panel._begin_properties("settings_camera_zoom"):
        choice("zoom_mode", "Zoom mode", (("proportional", "Proportional"), ("linear", "Linear")))
        panel._tooltip(
            t(
                "Proportional slows near the pivot. Linear uses a fixed step based on the framed scene size."
            )
        )
        number(
            "zoom_speed",
            "Zoom speed",
            0.05,
            "%.2fx",
            "Applies to the mouse wheel and zoom dragging.",
        )
        number(
            "min_distance",
            "Minimum distance",
            0.001,
            "%.4g",
            "World units; orthographic zoom uses the equivalent perspective distance.",
        )
        number(
            "max_distance",
            "Maximum distance",
            1.0,
            "%.4g",
            "World units; orthographic zoom uses the equivalent perspective distance.",
        )
        panel._end_properties()
    if panel._button(t("Reset camera navigation")):
        ctx.apply_setting(ctx.set_camera_navigation, defaults)
