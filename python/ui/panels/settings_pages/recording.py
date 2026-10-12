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
            imgui.set_item_tooltip(ctx.tr(tooltip))
        if changed:
            config = replace(config, **{attribute: value})
        if changed or committed:
            ctx.apply_setting(ctx.set_recording_config, config, persist=committed)

    if panel._begin_properties("settings_recording"):
        panel._property(ctx.tr("Copy to clipboard"))
        changed, value = imgui.checkbox("##capture_clipboard", config.copy_to_clipboard)
        imgui.set_item_tooltip(
            ctx.tr("Copy screenshots as images and videos as files after saving.")
        )
        if changed:
            config = replace(config, copy_to_clipboard=value)
            ctx.apply_setting(ctx.set_recording_config, config)
        panel._property(ctx.tr("Run simulation when recording starts"))
        clock_control = ctx.session.adapter.caps.clock_control
        imgui.begin_disabled(not clock_control)
        changed, value = imgui.checkbox(
            "##recording_run_simulation", config.run_simulation and clock_control
        )
        imgui.end_disabled()
        if not clock_control and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(ctx.tr("Simulation is controlled by the external application."))
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
        panel._property(ctx.tr("Default capture area"))
        surfaces = tuple(CaptureSurface)
        labels = ("Scene Only", "Viewport with UI", "Entire Window")
        selected = surfaces.index(config.surface)
        if imgui.begin_combo("##recording_surface", ctx.tr(labels[selected])):
            for index, label in enumerate(labels):
                clicked, _ = imgui.selectable(ctx.tr(label), index == selected)
                if clicked:
                    config = replace(config, surface=surfaces[index])
                    ctx.apply_setting(ctx.set_recording_config, config)
            imgui.end_combo()
        imgui.set_item_tooltip(
            ctx.tr("Viewport recording follows the visibility choices in the Layers panel.")
        )
        imgui.end_table()
    panel._group_heading(ctx.tr("Video encoding"))
    if panel._begin_properties("settings_video_encoding"):
        panel._property(ctx.tr("Rate control"))
        modes = (("quality", "Quality priority"), ("bitrate", "Target bitrate"))
        label = next(label for value, label in modes if value == config.rate_control)
        if imgui.begin_combo("##recording_rate_control", ctx.tr(label)):
            for value, label in modes:
                if imgui.selectable(ctx.tr(label), value == config.rate_control)[0]:
                    config = replace(config, rate_control=value)
                    ctx.apply_setting(ctx.set_recording_config, config)
            imgui.end_combo()
        imgui.set_item_tooltip(ctx.tr("Encoding settings apply to the next recording."))
        if config.rate_control == "quality":
            panel._property(ctx.tr("Quality (CRF)"))
            changed, value = panel._slider_int("##recording_crf", config.crf, 0, 51)
            committed = panel._value_committed()
            imgui.set_item_tooltip(ctx.tr("Lower CRF gives higher quality and larger files."))
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
        panel._property(ctx.tr("Encoding speed"))
        presets = (("fast", "Fast"), ("medium", "Balanced"), ("slow", "Slow"))
        label = next(label for value, label in presets if value == config.encoder_preset)
        if imgui.begin_combo("##recording_preset", ctx.tr(label)):
            for value, label in presets:
                if imgui.selectable(ctx.tr(label), value == config.encoder_preset)[0]:
                    config = replace(config, encoder_preset=value)
                    ctx.apply_setting(ctx.set_recording_config, config)
            imgui.end_combo()
        imgui.set_item_tooltip(
            ctx.tr(
                "Slower encoding spends more CPU time on compression and may slow live recording."
            )
        )
        panel._property(ctx.tr("Color sampling"))
        formats = (("yuv420p", "Compatible (4:2:0)"), ("yuv444p", "Full chroma (4:4:4)"))
        label = next(label for value, label in formats if value == config.pixel_format)
        if imgui.begin_combo("##recording_pixel_format", ctx.tr(label)):
            for value, label in formats:
                if imgui.selectable(ctx.tr(label), value == config.pixel_format)[0]:
                    config = replace(config, pixel_format=value)
                    ctx.apply_setting(ctx.set_recording_config, config)
            imgui.end_combo()
        imgui.set_item_tooltip(
            ctx.tr("4:4:4 preserves fine color detail but requires a compatible player.")
        )
        imgui.end_table()
