"""Inspector: environment."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    SceneNode,
)
from mojive.render.backend import RenderFlag
from mojive.scene.state import apply_camera_bookmark
from mojive.types import DEFAULT_HEADLIGHT, Environment, LightType, TextureType
from mojive.ui.controls import begin_property_table, property_row
from mojive.ui.panels import (
    PanelContext,
    button_width,
    segmented_control,
)
from mojive.ui.panels.value_cards import value_rail

from .fields import (
    _begin_property_table,
    _property_color_edit3,
    _property_control_row,
    _property_section,
    _property_vector_row,
    _vector_fields,
)


class _Environment:
    """Private environment methods of InspectorPanel; state belongs to its owner."""

    def _light(self, ctx: PanelContext, node: SceneNode) -> None:
        source = ctx.session.source
        index = node.light_index
        if source is None or not 0 <= index < len(source.lights.lights):
            imgui.text_disabled(ctx.tr("light data is unavailable"))
            return
        light = source.lights.lights[index]
        changed = False
        active = light.active
        kind_index = int(light.type)
        light_type = light.type
        diffuse = light.diffuse
        specular = light.specular
        ambient = light.ambient
        position = light.position
        direction = light.direction
        image_intensity = light.intensity
        texture = light.texture
        cast_shadow = light.cast_shadow

        if _property_section(ctx, "light properties") and _begin_property_table(
            "insp_light_properties"
        ):
            _property_control_row(ctx, "enabled")
            active_changed, active = imgui.checkbox("##light_enabled", light.active)
            _property_control_row(ctx, "type")
            kind_changed, kind_index = imgui.combo(
                "##light_type",
                int(light.type),
                ("directional", "point", "spot", "area", "image"),
            )
            changed |= active_changed or kind_changed
            light_type = LightType(kind_index)

            if light_type is LightType.IMAGE:
                _property_control_row(ctx, "intensity")
                intensity_changed, image_intensity = imgui.drag_float(
                    "##light_intensity", light.intensity, 50.0, 0.0, 100000.0, "%.0f"
                )
                textures = [
                    name
                    for name, item in source.textures.items()
                    if item.type in (TextureType.CUBE, TextureType.SKYBOX)
                ]
                texture_index = textures.index(light.texture) if light.texture in textures else 0
                texture_changed = False
                _property_control_row(ctx, "texture")
                if textures:
                    texture_changed, texture_index = imgui.combo(
                        "##light_texture", texture_index, textures
                    )
                    texture = textures[texture_index]
                else:
                    imgui.text_disabled(ctx.tr("add a cube texture to illuminate the scene"))
                changed |= intensity_changed or texture_changed
            else:
                intensity = float(np.max(light.diffuse))
                color = light.diffuse / intensity if intensity > 0.0 else np.ones(3, np.float32)
                _property_control_row(ctx, "color")
                color_changed, color = _property_color_edit3(ctx, "##light_color", color)
                _property_control_row(ctx, "intensity")
                intensity_changed, intensity = imgui.drag_float(
                    "##light_intensity", intensity, 0.01, 0.0, 10.0, "%.2f"
                )
                diffuse = np.asarray(color, np.float32) * intensity
                _property_control_row(ctx, "specular")
                specular_changed, specular = _property_color_edit3(
                    ctx, "##light_specular", light.specular
                )
                _property_control_row(ctx, "ambient")
                ambient_changed, ambient = _property_color_edit3(
                    ctx, "##light_ambient", light.ambient
                )
                _property_control_row(ctx, "cast shadow")
                shadow_changed, cast_shadow = imgui.checkbox(
                    "##light_cast_shadow", light.cast_shadow
                )
                changed |= (
                    color_changed
                    or intensity_changed
                    or specular_changed
                    or ambient_changed
                    or shadow_changed
                )
            imgui.end_table()

        if _property_section(ctx, "light transform") and _begin_property_table(
            "insp_light_transform"
        ):
            pos_changed, position = _property_vector_row(
                ctx,
                node,
                "position (local)",
                "light_position",
                position,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=0.0,
                fmt="%.3f",
            )
            dir_changed, direction = _property_vector_row(
                ctx,
                node,
                "direction (local)",
                "light_direction",
                direction,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=0.0,
                fmt="%.3f",
                reset_values=np.array((0.0, 0.0, -1.0)),
            )
            imgui.end_table()
            changed |= pos_changed or dir_changed

        range_changed = cutoff_changed = exponent_changed = attenuation_changed = False
        area_changed = False
        light_range, cutoff, exponent = light.range, light.cutoff, light.exponent
        area_radius = light.area_radius
        attenuation = light.attenuation
        local_light = light_type in (LightType.POINT, LightType.SPOT, LightType.AREA)
        if (
            local_light
            and _property_section(ctx, "light attenuation")
            and _begin_property_table("insp_light_attenuation")
        ):
            _property_control_row(ctx, "range")
            range_changed, light_range = imgui.drag_float(
                "##light_range", light.range, 0.05, 0.0, 10000.0, "%.2f"
            )
            imgui.set_item_tooltip(ctx.tr("0 means unlimited"))
            attenuation_changed, attenuation = _property_vector_row(
                ctx,
                node,
                "attenuation",
                "light_attenuation",
                light.attenuation,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=100.0,
                fmt="%.3f",
                reset_values=np.array((1.0, 0.0, 0.0)),
            )
            if light_type is LightType.SPOT:
                _property_control_row(ctx, "cutoff")
                cutoff_changed, cutoff = imgui.drag_float(
                    "##light_cutoff", light.cutoff, 0.25, 0.1, 89.9, "%.1f deg"
                )
                _property_control_row(ctx, "falloff exponent")
                exponent_changed, exponent = imgui.drag_float(
                    "##light_exponent", light.exponent, 0.1, 0.0, 100.0, "%.1f"
                )
            if light_type is LightType.AREA:
                _property_control_row(ctx, "source radius")
                area_changed, area_radius = imgui.drag_float(
                    "##light_area_radius", light.area_radius, 0.01, 0.0, 1000.0, "%.3f"
                )
            imgui.end_table()

        if (
            ctx.session.adapter.caps.simulation
            and _property_section(ctx, "light behavior")
            and _begin_property_table("insp_light_behavior")
        ):
            _property_control_row(ctx, "gizmo lock")
            lock_changed, locked = imgui.checkbox(
                "##light_gizmo_lock", ctx.session.entity_gizmo_lock_enabled(node)
            )
            if lock_changed:
                ctx.session.set_entity_gizmo_lock(node, locked)
            imgui.set_item_tooltip(ctx.tr("Lock gizmo while simulation runs"))
            imgui.end_table()
        changed |= (
            range_changed
            or cutoff_changed
            or exponent_changed
            or attenuation_changed
            or area_changed
        )
        if changed:
            self._submit_edit(
                ctx,
                cmd.SetLight(
                    index,
                    replace(
                        light,
                        type=light_type,
                        active=active,
                        diffuse=diffuse,
                        specular=np.asarray(specular, np.float32),
                        ambient=np.asarray(ambient, np.float32),
                        position=np.asarray(position, np.float32),
                        direction=np.asarray(direction, np.float32),
                        attenuation=np.clip(np.asarray(attenuation, np.float32), 0.0, 100.0),
                        range=float(light_range),
                        area_radius=float(area_radius),
                        cutoff=float(cutoff),
                        exponent=float(exponent),
                        texture=texture,
                        intensity=float(image_intensity),
                        cast_shadow=cast_shadow,
                    ),
                ),
            )

    def _environment(self, ctx: PanelContext) -> None:
        source = ctx.session.source
        if source is None:
            imgui.text_disabled(ctx.tr("environment data is unavailable"))
            return
        environment = source.lights.environment()
        changed = False
        cube_textures = [
            name
            for name, item in source.textures.items()
            if item.type in (TextureType.CUBE, TextureType.SKYBOX)
        ]
        skyboxes = [None, *cube_textures]
        skybox_index = skyboxes.index(source.skybox) if source.skybox in skyboxes else 0
        if _property_section(ctx, "skybox") and _begin_property_table("environment_skybox"):
            _property_control_row(ctx, "enabled")
            self._render_flag(ctx, RenderFlag.SKYBOX, "##environment-skybox-enabled")
            _property_control_row(ctx, "texture")
            skybox_changed, skybox_index = imgui.combo(
                "##environment-skybox-texture",
                skybox_index,
                [name or ctx.tr("none") for name in skyboxes],
            )
            imgui.end_table()
            if skybox_changed:
                self._submit_edit(ctx, cmd.SetSkybox(skyboxes[skybox_index]))

        ambient = environment.ambient
        if _property_section(ctx, "ambient light") and _begin_property_table("environment_ambient"):
            _property_control_row(ctx, "color")
            ambient_changed, ambient = _property_color_edit3(
                ctx, "##environment-ambient-color", ambient
            )
            changed |= ambient_changed
            imgui.end_table()

        headlight_enabled = environment.headlight is not None
        headlight = environment.headlight or DEFAULT_HEADLIGHT
        intensity = float(np.max(headlight.diffuse))
        color = headlight.diffuse / intensity if intensity > 0.0 else np.ones(3, np.float32)
        specular = headlight.specular
        headlight_ambient = headlight.ambient
        if _property_section(ctx, "headlight") and _begin_property_table("environment_headlight"):
            _property_control_row(ctx, "enabled")
            enabled_changed, headlight_enabled = imgui.checkbox(
                "##environment-headlight-enabled", headlight_enabled
            )
            _property_control_row(ctx, "color")
            color_changed, color = _property_color_edit3(
                ctx, "##environment-headlight-color", color
            )
            _property_control_row(ctx, "intensity")
            intensity_changed, intensity = imgui.drag_float(
                "##environment-headlight-intensity", intensity, 0.01, 0.0, 10.0, "%.2f"
            )
            _property_control_row(ctx, "specular")
            specular_changed, specular = _property_color_edit3(
                ctx, "##environment-headlight-specular", specular
            )
            _property_control_row(ctx, "ambient")
            headlight_ambient_changed, headlight_ambient = _property_color_edit3(
                ctx, "##environment-headlight-ambient", headlight_ambient
            )
            imgui.end_table()
            changed |= (
                enabled_changed
                or color_changed
                or intensity_changed
                or specular_changed
                or headlight_ambient_changed
            )

        fog_color = environment.fog_color
        fog_start = environment.fog_start
        fog_end = environment.fog_end
        if _property_section(ctx, "fog") and _begin_property_table("environment_fog"):
            _property_control_row(ctx, "enabled")
            self._render_flag(ctx, RenderFlag.FOG, "##environment-fog-enabled")
            _property_control_row(ctx, "color")
            fog_color_changed, fog_color = _property_color_edit3(
                ctx, "##environment-fog-color", fog_color
            )
            _property_control_row(ctx, "start")
            fog_start_changed, fog_start = imgui.drag_float(
                "##environment-fog-start", fog_start, 0.05, 0.0, 1e6, "%.2f m"
            )
            _property_control_row(ctx, "end")
            fog_end_changed, fog_end = imgui.drag_float(
                "##environment-fog-end", fog_end, 0.05, 0.0, 1e6, "%.2f m"
            )
            imgui.end_table()
            changed |= fog_color_changed or fog_start_changed or fog_end_changed

        haze_mode = int(environment.horizon_haze)
        haze_color = environment.haze_color
        haze_density = environment.haze_density
        haze_slices = environment.horizon_haze_slices
        if _property_section(ctx, "haze") and _begin_property_table("environment_haze"):
            _property_control_row(ctx, "enabled")
            self._render_flag(ctx, RenderFlag.HAZE, "##environment-haze-enabled")
            _property_control_row(ctx, "Mode")
            mode_changed, haze_mode = imgui.combo(
                "##environment-haze-mode",
                haze_mode,
                (ctx.tr("volumetric"), ctx.tr("horizon")),
            )
            horizon_haze = bool(haze_mode)
            _property_control_row(ctx, "color")
            haze_color_changed, haze_color = _property_color_edit3(
                ctx, "##environment-haze-color", haze_color
            )
            _property_control_row(ctx, "radius" if horizon_haze else "density")
            haze_format = "%.4f" if horizon_haze else "%.4f / m"
            haze_density_changed, haze_density = imgui.drag_float(
                "##environment-haze-density",
                haze_density,
                0.001,
                0.0,
                100.0,
                haze_format,
            )
            slices_changed = False
            if horizon_haze:
                _property_control_row(ctx, "slices")
                slices_changed, haze_slices = imgui.drag_int(
                    "##environment-haze-slices", haze_slices, 1.0, 3, 512, "%d"
                )
            imgui.end_table()
            changed |= haze_color_changed or haze_density_changed or mode_changed or slices_changed

        horizon_haze = bool(haze_mode)

        if changed:
            self._submit_edit(
                ctx,
                cmd.SetEnvironment(
                    Environment(
                        headlight=(
                            replace(
                                headlight,
                                diffuse=np.asarray(color, np.float32) * float(intensity),
                                specular=np.asarray(specular, np.float32),
                                ambient=np.asarray(headlight_ambient, np.float32),
                                active=True,
                            )
                            if headlight_enabled
                            else None
                        ),
                        ambient=np.asarray(ambient, np.float32),
                        fog_color=np.asarray(fog_color, np.float32),
                        fog_start=float(fog_start),
                        fog_end=float(fog_end),
                        haze_color=np.asarray(haze_color, np.float32),
                        haze_density=float(haze_density),
                        horizon_haze=horizon_haze,
                        horizon_haze_slices=int(haze_slices),
                    )
                ),
            )

    @staticmethod
    def _render_flag(ctx: PanelContext, flag: RenderFlag, label: str) -> None:
        supported = ctx.backend.caps.supports(flag)
        imgui.begin_disabled(not supported)
        changed, value = imgui.checkbox(label, ctx.backend.get_flag(flag))
        imgui.end_disabled()
        if changed:
            ctx.backend.set_flag(flag, value)

    def _camera(self, ctx: PanelContext, node: SceneNode) -> None:
        index = node.camera_index
        if not 0 <= index < len(ctx.session.cameras):
            imgui.text_disabled(ctx.tr("camera data is unavailable"))
            return
        info = ctx.session.cameras[index]
        view = ctx.session.camera_view(info.camera_id)
        if view is None:
            imgui.text_disabled(ctx.tr("camera view is unavailable"))
            return

        self._camera_transfer(ctx, info)
        identity = (info.camera_id, ctx.session.structure_generation)
        if self._camera_initial_key != identity:
            self._camera_initial_key, self._camera_initial_view = identity, view
        initial = self._camera_initial_view

        eye = np.asarray(view.eye, np.float64).copy()
        target = np.asarray(view.target, np.float64).copy()
        up = np.asarray(view.up, np.float64).copy()
        eye_changed = target_changed = up_changed = False
        fov_changed = near_changed = far_changed = ortho_changed = False
        fov = float(view.fov_y)
        near = float(view.near)
        far = float(view.far)
        orthographic = view.orthographic
        height_changed = False
        ortho_height = view.ortho_height

        if _property_section(ctx, "camera transform"):
            (eye_changed, eye), (target_changed, target), (up_changed, up) = _vector_fields(
                ctx,
                node,
                "insp_camera_transform",
                (
                    (ctx.tr("position"), eye, 0.01, "%.4f", initial.eye),
                    (ctx.tr("target"), target, 0.01, "%.4f", initial.target),
                    (ctx.tr("up"), up, 0.01, "%.4f", initial.up),
                ),
            )

        if _property_section(ctx, "camera projection") and begin_property_table(
            "inspector_camera_projection"
        ):
            fields = []
            for label, value, bounds, default, unit, fmt in (
                (
                    "vertical fov",
                    fov,
                    (np.radians(1.0), np.radians(179.0)),
                    initial.fov_y,
                    "rad",
                    "%.2f",
                ),
                ("near", near, (1e-5, far - 1e-5), initial.near, "m", "%.5f"),
                ("far", far, (near + 1e-5, 1e7), initial.far, "m", "%.3f"),
            ):
                property_row(ctx.tr(label))
                fields.append(
                    value_rail(
                        ctx,
                        f"##inspector-camera-{label}",
                        value,
                        bounds,
                        initial=default,
                        fmt=fmt,
                        show_reset=False,
                        unit=unit,
                        angular_degrees=self._camera_angular_degrees,
                        toggle_unit=self._toggle_camera_angle_unit,
                    )
                )
            fov_changed, fov = fields[0].changed, fields[0].value
            near_changed, near = fields[1].changed, fields[1].value
            far_changed, far = fields[2].changed, fields[2].value
            property_row(ctx.tr("projection"))
            imgui.set_next_item_width(-1)
            projection = 1 if orthographic else 0
            supported = ctx.backend.caps.orthographic
            imgui.begin_disabled(not supported)
            selected_projection = segmented_control(
                f"camera-inspector-projection-{node.node_id}",
                (ctx.tr("persp"), ctx.tr("ortho")),
                projection,
                theme=ctx.theme,
                icons=("persp", "ortho"),
            )
            imgui.end_disabled()
            if not supported:
                imgui.set_item_tooltip(
                    f"{ctx.backend.caps.name} {ctx.tr('has no orthographic projection')}"
                )
            elif selected_projection != projection:
                orthographic = selected_projection == 1
                ortho_changed = True
            if orthographic:
                property_row(ctx.tr("ortho height"))
                height_edit = value_rail(
                    ctx,
                    "##camera_ortho_height",
                    float(view.ortho_height),
                    (1e-4, 1e6),
                    initial=initial.ortho_height,
                    fmt="%.4f",
                    show_reset=False,
                    unit="m",
                )
                height_changed, ortho_height = height_edit.changed, height_edit.value
            imgui.end_table()

        if _property_section(ctx, "camera behavior") and _begin_property_table(
            "insp_camera_behavior"
        ):
            if ctx.select_model_camera is not None:
                active = ctx.model_camera_id == info.camera_id
                label = "Editor Camera" if active else "View Camera"
                _property_control_row(ctx, "camera view")
                if imgui.button(ctx.tr(label), imgui.ImVec2(-1.0, 0.0)):
                    ctx.select_model_camera(-1 if active else info.camera_id)
                imgui.set_item_tooltip(
                    ctx.tr("Return to Editor Camera" if active else "View Through Camera")
                )

            if ctx.session.adapter.caps.simulation:
                _property_control_row(ctx, "gizmo lock")
                changed, locked = imgui.checkbox(
                    "##camera-gizmo-lock",
                    ctx.session.entity_gizmo_lock_enabled(node),
                )
                imgui.set_item_tooltip(ctx.tr("Lock gizmo while simulation runs"))
                if changed:
                    ctx.session.set_entity_gizmo_lock(node, locked)

            if ctx.camera_preview is not None:
                _property_control_row(ctx, "preview")
                changed, enabled = imgui.checkbox(
                    "##camera_preview_enabled",
                    bool(ctx.camera_preview.enabled),
                )
                imgui.set_item_tooltip(ctx.tr("Show Camera Preview"))
                if changed:
                    ctx.camera_preview.set_enabled(enabled)
            imgui.end_table()

        if any(
            (
                eye_changed,
                target_changed,
                up_changed,
                fov_changed,
                near_changed,
                far_changed,
                ortho_changed,
                height_changed,
            )
        ):
            self._submit_edit(
                ctx,
                cmd.SetSceneCamera(
                    info.camera_id,
                    replace(
                        view,
                        eye=np.asarray(eye, np.float32),
                        target=np.asarray(target, np.float32),
                        up=np.asarray(up, np.float32),
                        fov_y=float(fov),
                        near=float(near),
                        far=max(float(far), float(near) + 1e-5),
                        orthographic=orthographic,
                        ortho_height=float(ortho_height),
                    ),
                ),
            )

    def _toggle_camera_angle_unit(self):
        self._camera_angular_degrees = not self._camera_angular_degrees

    def _camera_transfer(self, ctx, camera_info):
        sources = [(-1, ctx.tr("Editor Camera"))] + [
            (camera.camera_id, camera.name)
            for camera in ctx.session.cameras
            if camera.camera_id != camera_info.camera_id
        ]
        ids = [item[0] for item in sources]
        if self._camera_sync_source not in ids:
            self._camera_sync_source = -1
        width = imgui.get_content_region_avail().x
        gap = imgui.get_style().item_spacing.x
        sync_label, paste_label = ctx.tr("Sync view"), ctx.tr("Paste bookmark")
        actions_width = button_width(sync_label) + button_width(paste_label) + gap
        inline = width >= actions_width + 120 * ctx.style_scale + gap
        imgui.set_next_item_width(max(1, width - actions_width - gap) if inline else -1)
        changed, slot = imgui.combo(
            "##camera-sync-source",
            ids.index(self._camera_sync_source),
            tuple(item[1] for item in sources),
        )
        if changed:
            self._camera_sync_source = ids[slot]
        if inline:
            imgui.same_line()
        imgui.begin_disabled(self._camera_sync_source < 0 and ctx.camera is None)
        sync = imgui.button(sync_label + "##camera-sync")
        imgui.end_disabled()
        if sync:
            source = (
                ctx.camera.view()
                if self._camera_sync_source < 0
                else ctx.session.camera_view(self._camera_sync_source)
            )
            if source is not None:
                self._submit_edit(ctx, cmd.SetSceneCamera(camera_info.camera_id, source))
        if actions_width <= width:
            imgui.same_line()
        if imgui.button(paste_label + "##camera-paste"):
            try:
                bookmark = json.loads(imgui.get_clipboard_text())
                if not isinstance(bookmark, dict):
                    raise ValueError("The clipboard does not contain a camera bookmark")
                source = apply_camera_bookmark(bookmark, None)
                vectors = (source.eye, source.target, source.up)
                if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
                    raise ValueError("Camera position and direction must be finite XYZ vectors")
                if (
                    not np.isfinite(source.proj_matrix()).all()
                    or not 0 < source.near < source.far
                    or not 0 < source.fov_y < np.pi
                    or np.linalg.norm(np.cross(source.target - source.eye, source.up)) < 1e-8
                ):
                    raise ValueError("Camera bookmark has an invalid projection or direction")
                self._submit_edit(ctx, cmd.SetSceneCamera(camera_info.camera_id, source))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                ctx.report(str(error), level="error")
        imgui.separator()

    @staticmethod
    def _entity_gizmo_lock(ctx: PanelContext, node: SceneNode) -> None:
        if not ctx.session.adapter.caps.simulation:
            return
        changed, locked = imgui.checkbox(
            ctx.tr("Lock gizmo while simulation runs"),
            ctx.session.entity_gizmo_lock_enabled(node),
        )
        if changed:
            ctx.session.set_entity_gizmo_lock(node, locked)
