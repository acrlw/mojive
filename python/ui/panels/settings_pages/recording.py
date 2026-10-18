"""Recording settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.capture import CaptureSurface
from mojive.ui.panels import (
    PanelContext,
)


def draw(panel, ctx: PanelContext) -> None:
    config = ctx.recording_config
    if config is None:
        return

    def number(identifier, label, attribute, step, fast_step, tooltip=""):
        nonlocal config
        panel._property(ctx.tr(label))
        changed, value = panel._input_float(
            f"##{identifier}", getattr(config, attribute), step, fast_step, "%.1f"
        )
        committed = panel._value_committed()
        if tooltip:
            panel._tooltip(ctx.tr(tooltip))
        if changed:
            config = replace(config, **{attribute: value})
        if changed or committed:
            ctx.apply_setting(ctx.set_recording_config, config, persist=committed)

    def choice(identifier, attribute, choices):
        nonlocal config
        values, labels = zip(*choices, strict=True)
        changed, selected = panel._combo(
            identifier,
            values.index(getattr(config, attribute)),
            tuple(ctx.tr(label) for label in labels),
        )
        if changed:
            config = replace(config, **{attribute: values[selected]})
            ctx.apply_setting(ctx.set_recording_config, config)

    if panel._begin_properties("settings_recording"):
        panel._property(ctx.tr("Copy to clipboard"), control_part="checkbox")
        changed, value = panel._checkbox("##capture_clipboard", config.copy_to_clipboard)
        panel._tooltip(ctx.tr("Copy screenshots as images and videos as files after saving."))
        if changed:
            config = replace(config, copy_to_clipboard=value)
            ctx.apply_setting(ctx.set_recording_config, config)
        panel._property(ctx.tr("Run simulation when recording starts"), control_part="checkbox")
        clock_control = ctx.session.adapter.caps.clock_control
        imgui.begin_disabled(not clock_control)
        changed, value = panel._checkbox(
            "##recording_run_simulation", config.run_simulation and clock_control
        )
        imgui.end_disabled()
        if not clock_control and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            panel._tooltip(
                ctx.tr("Simulation is controlled by the external application."), hovered=True
            )
        if changed:
            config = replace(config, run_simulation=value)
            ctx.apply_setting(ctx.set_recording_config, config)
        number(
            "recording_delay",
            "Countdown (s)",
            "countdown",
            1.0,
            5.0,
            "Set to 0 to start after menus close.",
        )
        number(
            "recording_end_hold",
            "Take end hold (s)",
            "end_hold",
            0.5,
            5.0,
            "Take videos stop automatically after holding the final frame.",
        )
        number("recording_fps", "Video frame rate", "fps", 1.0, 10.0)
        panel._property(ctx.tr("Default capture area"), control_part="select")
        surfaces = tuple(CaptureSurface)
        labels = ("Scene Only", "Viewport with UI", "Entire Window")
        choice("##recording_surface", "surface", tuple(zip(surfaces, labels, strict=True)))
        panel._tooltip(
            ctx.tr("Viewport recording follows the visibility choices in the Layers panel.")
        )
        panel._end_properties()
    panel._group_heading(ctx.tr("Video encoding"))
    if panel._begin_properties("settings_video_encoding"):
        panel._property(ctx.tr("Rate control"), control_part="select")
        modes = (("quality", "Quality priority"), ("bitrate", "Target bitrate"))
        choice("##recording_rate_control", "rate_control", modes)
        panel._tooltip(ctx.tr("Encoding settings apply to the next recording."))
        if config.rate_control == "quality":
            panel._property(ctx.tr("Quality (CRF)"))
            changed, value = panel._slider_int("##recording_crf", config.crf, 0, 51)
            committed = panel._value_committed()
            panel._tooltip(ctx.tr("Lower CRF gives higher quality and larger files."))
            if changed:
                config = replace(config, crf=value)
            if changed or committed:
                ctx.apply_setting(ctx.set_recording_config, config, persist=committed)
        else:
            number(
                "recording_bitrate",
                "Target bitrate (Mbps)",
                "bitrate_mbps",
                1.0,
                5.0,
                "Target average bitrate; actual bitrate varies with the scene.",
            )
        panel._property(ctx.tr("Encoding speed"), control_part="select")
        presets = (("fast", "Fast"), ("medium", "Balanced"), ("slow", "Slow"))
        choice("##recording_preset", "encoder_preset", presets)
        panel._tooltip(
            ctx.tr(
                "Slower encoding spends more CPU time on compression and may slow live recording."
            )
        )
        panel._property(ctx.tr("Color sampling"), control_part="select")
        formats = (("yuv420p", "Compatible (4:2:0)"), ("yuv444p", "Full chroma (4:4:4)"))
        choice("##recording_pixel_format", "pixel_format", formats)
        panel._tooltip(
            ctx.tr("4:4:4 preserves fine color detail but requires a compatible player.")
        )
        panel._end_properties()
