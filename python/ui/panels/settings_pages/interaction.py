"""Interaction settings; apply changes through PanelContext callbacks."""

from __future__ import annotations

from dataclasses import replace

from imgui_bundle import imgui

from mojive.ui.gizmo import (
    DEFAULT_ROTATION_SNAP_DEG,
    DEFAULT_ROTATION_TICK_SCALE,
    DEFAULT_TRANSLATION_SNAP_M,
)
from mojive.ui.panels import (
    PanelContext,
)
from mojive.ui.panels.settings_pages import bindings
from mojive.ui.perturb import DEFAULT_PERTURB_SCALE, MAX_PERTURB_SCALE, OUTLINE_CORNER_RADIUS_PT
from mojive.ui.viewcube import (
    DEFAULT_SELECTION_PADDING,
    MAX_SELECTION_PADDING,
    MIN_SELECTION_PADDING,
)
from mojive.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    MAX_VIEWPORT_CAPSULE_SCALE,
    MAX_VIEWPORT_OVERLAY_SCALE,
    MIN_VIEWPORT_CAPSULE_SCALE,
    MIN_VIEWPORT_OVERLAY_SCALE,
)


def draw(panel, ctx: PanelContext) -> None:
    if ctx.interactions is not None:
        interaction_policy(panel, ctx)
    if ctx.selection_style is not None:
        selection_presentation(panel, ctx)
    if ctx.gizmo is not None:
        gizmo_settings(panel, ctx)
        precise_input_settings(panel, ctx)
        if ctx.input_bindings is not None:
            bindings.draw(panel, ctx)
        snap_settings(panel, ctx)
    if ctx.view_cube is not None:
        view_settings(panel, ctx)
    if ctx.perturb is not None:
        perturb_settings(panel, ctx)
    if ctx.scene_entities is not None:
        helper_settings(panel, ctx)


def gizmo_settings(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    panel._group_heading(t("Gizmo"))
    if not panel._begin_properties("settings_interaction_gizmo"):
        return
    panel._property(t("Style"), control_part="segment")
    style = panel._segmented(
        "gizmo-style", ("2D", "3D"), 1 if ctx.gizmo.style == "3d" else 0, theme=ctx.theme
    )
    ctx.gizmo.set_style("3d" if style == 1 else "2d")
    panel._property(t("Orientation"), control_part="segment")
    frame = panel._segmented(
        "gizmo-frame",
        (t("Body"), t("World")),
        1 if ctx.gizmo.space == "world" else 0,
        theme=ctx.theme,
    )
    ctx.gizmo.set_space("world" if frame == 1 else "body")
    changed, value, committed = panel._number_setting(
        ctx,
        "##viewport_overlay_scale",
        "Overlay size",
        ctx.viewport_overlay_scale,
        speed=0.02,
        low=MIN_VIEWPORT_OVERLAY_SCALE,
        high=MAX_VIEWPORT_OVERLAY_SCALE,
        fmt="%.2fx",
        default=DEFAULT_VIEWPORT_OVERLAY_SCALE,
    )
    if (changed or committed) and ctx.set_viewport_overlay_scale is not None:
        ctx.apply_setting(ctx.set_viewport_overlay_scale, value, persist=committed)
    if ctx.viewport_overlays is not None:
        capsule_settings(panel, ctx)
    panel._end_properties()


def capsule_settings(panel, ctx: PanelContext) -> None:
    overlays = ctx.viewport_overlays
    changed, value, committed = panel._number_setting(
        ctx,
        "##viewport_playback_scale",
        "Capsule size",
        overlays.playback_scale,
        speed=0.02,
        low=MIN_VIEWPORT_CAPSULE_SCALE,
        high=MAX_VIEWPORT_CAPSULE_SCALE,
        fmt="%.2fx",
        default=1.0,
    )
    if (changed or committed) and ctx.set_viewport_capsule_scale is not None:
        ctx.apply_setting(ctx.set_viewport_capsule_scale, "playback", value, persist=committed)
        overlays = replace(overlays, playback_scale=value, tool_scale=value)
    panel._property(ctx.tr("Status duration"))
    changed, duration = panel._slider_float(
        "##viewport_status_duration", overlays.status_duration, 3.0, 5.0, "%.1f s"
    )
    committed = panel._value_committed()
    if changed:
        overlays = replace(overlays, status_duration=duration)
    if (changed or committed) and ctx.set_viewport_overlays is not None:
        ctx.apply_setting(ctx.set_viewport_overlays, overlays, persist=committed)
    panel._property(ctx.tr("Movable capsules"), control_part="checkbox")
    changed, movable = panel._checkbox(
        "##viewport_capsules_movable", bool(overlays.movable), ctx.theme
    )
    if changed and ctx.set_viewport_overlays is not None:
        ctx.apply_setting(ctx.set_viewport_overlays, replace(overlays, movable=movable))


def precise_input_settings(panel, ctx: PanelContext) -> None:
    panel._group_heading(ctx.tr("Input"))
    if not panel._begin_properties("settings_interaction_input"):
        return
    panel._property(ctx.tr("Keep mode/unit"), control_part="checkbox")
    changed, remember = panel._checkbox(
        "##remember_precise_input_choices",
        bool(ctx.gizmo.remember_precise_input_choices),
        ctx.theme,
    )
    panel._tooltip(
        ctx.tr("Reuse the last relative/absolute mode and angle unit across editor sessions")
    )
    if changed:
        if ctx.set_precise_input_memory is not None:
            ctx.apply_setting(ctx.set_precise_input_memory, remember)
        else:
            ctx.gizmo.remember_precise_input_choices = remember
    panel._end_properties()


def snap_settings(panel, ctx: PanelContext) -> None:
    panel._group_heading(ctx.tr("Snap · Shift"))
    if not panel._begin_properties("settings_interaction_snap"):
        return
    for identifier, label, attribute, speed, low, high, fmt, default in (
        (
            "position_snap",
            "Position",
            "translation_snap_m",
            0.01,
            0.01,
            100.0,
            "%.3f m",
            DEFAULT_TRANSLATION_SNAP_M,
        ),
        (
            "rotation_snap",
            "Rotation",
            "rotation_snap_deg",
            0.1,
            0.5,
            180.0,
            "%.1f deg",
            DEFAULT_ROTATION_SNAP_DEG,
        ),
        (
            "rotation_tick_scale",
            "Tick scale",
            "rotation_tick_scale",
            0.05,
            0.5,
            3.0,
            "%.2fx",
            DEFAULT_ROTATION_TICK_SCALE,
        ),
    ):
        changed, value, _ = panel._number_setting(
            ctx,
            f"##{identifier}",
            label,
            getattr(ctx.gizmo, attribute),
            speed=speed,
            low=low,
            high=high,
            fmt=fmt,
            default=default,
        )
        if changed:
            setattr(ctx.gizmo, attribute, value)
    panel._end_properties()


def view_settings(panel, ctx: PanelContext) -> None:
    panel._group_heading(ctx.tr("View"))
    if not panel._begin_properties("settings_interaction_view"):
        return
    changed, value, committed = panel._number_setting(
        ctx,
        "##view_selection_padding",
        "Padding",
        ctx.view_cube.selection_padding,
        speed=0.02,
        low=MIN_SELECTION_PADDING,
        high=MAX_SELECTION_PADDING,
        fmt="%.2fx",
        default=DEFAULT_SELECTION_PADDING,
        tooltip="1x is a tight fit; larger values move the view farther away",
    )
    if changed or committed:
        if ctx.set_view_selection_padding is not None:
            ctx.apply_setting(ctx.set_view_selection_padding, value, persist=committed)
        elif changed:
            ctx.view_cube.selection_padding = value
    panel._end_properties()


def perturb_settings(panel, ctx: PanelContext) -> None:
    panel._group_heading(ctx.tr("Perturb"))
    if not panel._begin_properties("settings_interaction_perturb"):
        return
    imgui.begin_disabled(not ctx.session.adapter.caps.supports("physics.perturb_strength"))
    for attribute, label in (
        ("force_scale", "Drag force scale"),
        ("torque_scale", "Twist torque scale"),
    ):
        changed, value, committed = panel._number_setting(
            ctx,
            f"##perturb_{attribute}",
            label,
            getattr(ctx.perturb, attribute),
            speed=0.05,
            low=0.0,
            high=MAX_PERTURB_SCALE,
            fmt="%.2fx",
            default=DEFAULT_PERTURB_SCALE,
            clamp=True,
            tooltip="1x uses the default force or torque; 0x disables it. Saved for future sessions.",
        )
        if changed or committed:
            if ctx.set_perturb_strength is not None:
                force = value if attribute == "force_scale" else ctx.perturb.force_scale
                torque = value if attribute == "torque_scale" else ctx.perturb.torque_scale
                ctx.apply_setting(ctx.set_perturb_strength, force, torque, persist=committed)
            elif changed:
                setattr(ctx.perturb, attribute, value)
    imgui.end_disabled()
    changed, value, _ = panel._number_setting(
        ctx,
        "##perturb_corner_radius",
        "Corner radius",
        ctx.perturb.outline_corner_radius_pt,
        speed=0.1,
        low=0.0,
        high=24.0,
        fmt="%.1f px",
        default=OUTLINE_CORNER_RADIUS_PT,
    )
    if changed:
        ctx.perturb.outline_corner_radius_pt = value
    panel._end_properties()


def helper_settings(panel, ctx: PanelContext) -> None:
    panel._group_heading(ctx.tr("Helpers"))
    if not panel._begin_properties("settings_interaction_helpers"):
        return
    panel._property(ctx.tr("Camera & light icons"), control_part="checkbox")
    changed, visible = panel._checkbox(
        "##scene_entity_helpers", ctx.scene_entities.visible, ctx.theme
    )
    if changed:
        ctx.scene_entities.visible = visible
    panel._property(ctx.tr("Selected frustum / light range"), control_part="checkbox")
    imgui.begin_disabled(not ctx.scene_entities.visible)
    changed, influence = panel._checkbox(
        "##selected_influence_volumes", ctx.scene_entities.show_influence, ctx.theme
    )
    imgui.end_disabled()
    if changed:
        ctx.scene_entities.show_influence = influence
    panel._end_properties()


def interaction_policy(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    config = ctx.interactions
    panel._group_heading(t("Built-in interactions"))

    def update_top(attribute: str, value: bool) -> None:
        nonlocal config
        config = replace(config, **{attribute: value})
        ctx.interactions = config
        if ctx.set_interactions is not None:
            ctx.apply_setting(ctx.set_interactions, config)

    def update_camera(attribute: str, value: bool) -> None:
        nonlocal config
        config = replace(config, camera=replace(config.camera, **{attribute: value}))
        ctx.interactions = config
        if ctx.set_interactions is not None:
            ctx.apply_setting(ctx.set_interactions, config)

    def update_selection(attribute: str, value: bool) -> None:
        nonlocal config
        config = replace(
            config,
            selection=replace(config.selection, **{attribute: value}),
        )
        ctx.interactions = config
        if ctx.set_interactions is not None:
            ctx.apply_setting(ctx.set_interactions, config)

    rows = (
        ("Camera orbit", config.camera.orbit, update_camera, "orbit"),
        ("Camera pan", config.camera.pan, update_camera, "pan"),
        ("Camera dolly", config.camera.dolly, update_camera, "dolly"),
        ("Camera fly keys", config.camera.fly, update_camera, "fly"),
        ("View cube", config.camera.view_cube, update_camera, "view_cube"),
        ("Scene picking", config.selection.pick, update_selection, "pick"),
        (
            "Clear selection on empty click",
            config.selection.clear_on_empty,
            update_selection,
            "clear_on_empty",
        ),
        (
            "Clear selection with Escape",
            config.selection.clear_with_escape,
            update_selection,
            "clear_with_escape",
        ),
        (
            "Focus on double-click",
            config.selection.focus_on_double_click,
            update_selection,
            "focus_on_double_click",
        ),
        (
            "Pick when focusing viewport",
            config.selection.pick_on_focus,
            update_selection,
            "pick_on_focus",
        ),
        ("Gizmo input", config.gizmo, update_top, "gizmo"),
        ("Physics perturbation", config.perturb, update_top, "perturb"),
        (
            "Playback shortcuts",
            config.playback_shortcuts,
            update_top,
            "playback_shortcuts",
        ),
        ("Panel shortcuts", config.panel_shortcuts, update_top, "panel_shortcuts"),
    )
    if not panel._begin_toggle_grid(
        "settings_builtin_interactions", tuple(t(row[0]) for row in rows)
    ):
        return
    for label, current, callback, attribute in rows:
        if attribute == "perturb" and not ctx.session.adapter.caps.perturb:
            continue
        imgui.table_next_column()
        changed, value = panel._checkbox(
            f"{t(label)}###interaction_{attribute}", current, ctx.theme
        )
        if changed:
            callback(attribute, value)
    panel._end_properties()


def selection_presentation(panel, ctx: PanelContext) -> None:
    t = ctx.tr
    style = ctx.selection_style
    panel._group_heading(t("Selection presentation"))
    rows = (
        ("highlight", "Highlight fill"),
        ("outline", "Outline"),
        ("gizmo", "Gizmo"),
        ("frame", "Coordinate frame"),
        ("label", "Name label"),
        ("bounds", "Bounds"),
    )
    if not panel._begin_toggle_grid(
        "settings_selection_presentation", tuple(t(label) for _, label in rows)
    ):
        return
    for attribute, label in rows:
        imgui.table_next_column()
        changed, value = panel._checkbox(
            f"{t(label)}###selection_{attribute}", getattr(style, attribute), ctx.theme
        )
        if changed:
            style = replace(style, **{attribute: value})
            ctx.selection_style = style
            if ctx.set_selection_style is not None:
                ctx.apply_setting(ctx.set_selection_style, style)
    panel._end_properties()
