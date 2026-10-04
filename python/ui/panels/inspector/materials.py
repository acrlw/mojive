"""Material property rows and the asset actions requested by one Inspector frame."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import SceneSource
from mojive.types import Material
from mojive.ui.panels import PanelContext

from .fields import _property_button_row, _property_color_edit4, _property_control_row
from .support import _MATERIAL_PRESETS, _unique_component_name


@dataclass(frozen=True)
class _MaterialAssetActions:
    """Deferred actions; drawing rows never rebuilds their source underneath them."""

    assignment: cmd.SetGeometryMaterial | None = None
    creation: cmd.AddModelMaterial | None = None
    open_assets: bool = False
    import_texture: bool = False
    import_cube: bool = False
    import_skybox: bool = False


def _material_asset_rows(
    ctx: PanelContext,
    source: SceneSource,
    node_id: int,
    material_index: int,
    compatible_materials: tuple[int, ...],
    *,
    model_id: int,
    assigned: bool,
    editable: bool,
    assignment_editable: bool,
) -> _MaterialAssetActions | None:
    assignment = creation = None
    open_assets = False
    material = source.materials[material_index]
    assignment_label = material.name if assigned else ctx.tr("inline appearance")
    if not editable:
        imgui.begin_disabled()
    _property_control_row(ctx, "assigned material")
    if not assignment_editable:
        imgui.begin_disabled()
    if imgui.begin_combo("##assigned_material", assignment_label):
        selected, _ = imgui.selectable(ctx.tr("inline appearance"), not assigned)
        if selected and assigned:
            assignment = cmd.SetGeometryMaterial(node_id, -1)
        for candidate in compatible_materials:
            candidate_material = source.materials[candidate]
            selected, _ = imgui.selectable(
                candidate_material.name or f"{ctx.tr('material')} {candidate}",
                candidate == material_index,
            )
            if selected and candidate != material_index:
                assignment = cmd.SetGeometryMaterial(node_id, candidate)
        imgui.end_combo()
    if not assignment_editable:
        imgui.end_disabled()

    if assigned and ctx.panels is not None:
        (open_assets,) = _property_button_row(ctx, "asset browser", ("Open in Assets",))
    if not assignment_editable:
        imgui.begin_disabled()
    (create,) = _property_button_row(ctx, "create material", ("New material",))
    if not assigned or not assignment_editable:
        imgui.begin_disabled()
    (duplicate,) = _property_button_row(ctx, "material actions", ("Duplicate material",))
    if not assigned or not assignment_editable:
        imgui.end_disabled()
    if not assignment_editable:
        imgui.end_disabled()
    if create or duplicate:
        prefix = f"opengl_{model_id}_"
        existing = {
            source.materials[index].name.removeprefix(prefix) for index in compatible_materials
        }
        creation = cmd.AddModelMaterial(
            node_id,
            _unique_component_name("material", existing),
            material_index if duplicate and assigned else -1,
        )
    can_import_texture = assigned and ctx.request_texture_import is not None
    if not can_import_texture:
        imgui.begin_disabled()
    (import_texture,) = _property_button_row(ctx, "texture import", ("Import texture",))
    if not can_import_texture:
        imgui.end_disabled()
    if ctx.request_texture_import is None:
        imgui.begin_disabled()
    import_cube, import_skybox = _property_button_row(
        ctx, "environment textures", ("Import cube texture", "Import skybox texture")
    )
    if ctx.request_texture_import is None:
        imgui.end_disabled()
    if not editable:
        imgui.end_disabled()
    if (
        assignment is None
        and creation is None
        and not (open_assets or import_texture or import_cube or import_skybox)
    ):
        return None
    return _MaterialAssetActions(
        assignment, creation, open_assets, import_texture, import_cube, import_skybox
    )


def _material_override_rows(
    ctx: PanelContext, name: str, value: float, enabled_default: float
) -> tuple[bool, float]:
    _property_control_row(ctx, f"{name} override")
    toggle_changed, enabled = imgui.checkbox(f"##material_{name}_override", value >= 0.0)
    if toggle_changed:
        value = enabled_default if enabled else -1.0
    _property_control_row(ctx, name)
    if not enabled:
        imgui.begin_disabled()
    changed, edited = imgui.drag_float(
        f"##material_{name}", max(0.0, value), 0.01, 0.0, 1.0, "%.2f"
    )
    if not enabled:
        imgui.end_disabled()
    return toggle_changed or changed, float(edited) if enabled else value


def _material_surface_rows(ctx: PanelContext, material: Material, material_index: int) -> Material:
    """Return the original material when its surface controls have not changed."""
    _property_control_row(ctx, "shared material")
    imgui.align_text_to_frame_padding()
    imgui.text(material.name or str(material_index))
    _property_control_row(ctx, "base color")
    changed, rgba = _property_color_edit4(ctx, "##material_base_color", material.rgba)
    emission, specular = material.emission, material.specular
    shininess, reflectance = material.shininess, material.reflectance
    _property_control_row(ctx, "preset")
    if imgui.begin_combo("##material_preset", ctx.tr("Presets...")):
        for preset, values in _MATERIAL_PRESETS.items():
            selected, _ = imgui.selectable(preset, False)
            if selected:
                emission, specular, shininess, reflectance = values
                changed = True
        imgui.end_combo()
    _property_control_row(ctx, "emission")
    edited, emission = imgui.drag_float("##material_emission", emission, 0.01, 0.0, 10.0, "%.2f")
    changed |= edited
    _property_control_row(ctx, "specular")
    edited, specular = imgui.drag_float("##material_specular", specular, 0.01, 0.0, 1.0, "%.2f")
    changed |= edited
    _property_control_row(ctx, "shininess")
    edited, shininess = imgui.drag_float("##material_shininess", shininess, 0.01, 0.0, 1.0, "%.2f")
    changed |= edited
    _property_control_row(ctx, "reflectance")
    edited, reflectance = imgui.drag_float(
        "##material_reflectance", reflectance, 0.01, 0.0, 1.0, "%.2f"
    )
    changed |= edited
    edited, metallic = _material_override_rows(ctx, "metallic", material.metallic, 0.0)
    changed |= edited
    edited, roughness = _material_override_rows(ctx, "roughness", material.roughness, 0.5)
    changed |= edited
    if not changed:
        return material
    return replace(
        material,
        rgba=np.asarray(rgba, np.float32),
        emission=float(emission),
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
        metallic=metallic,
        roughness=roughness,
    )


def _material_texture_rows(
    ctx: PanelContext, material: Material, source: SceneSource, *, model_id: int, model_assets: bool
) -> Material:
    """Edit texture sampling without changing any surface parameters."""
    texture = material.texture
    texture_changed = False
    _property_control_row(ctx, "texture")
    if imgui.begin_combo("##material_texture", texture or ctx.tr("none")):
        compatible_textures = (
            ctx.session.model_texture_names(model_id) if model_assets else tuple(source.textures)
        )
        for candidate in (None, *compatible_textures):
            selected, _ = imgui.selectable(candidate or ctx.tr("none"), candidate == texture)
            if selected:
                texture = candidate
                texture_changed = True
        imgui.end_combo()
    _property_control_row(ctx, "texture repeat")
    repeat_changed, repeat = imgui.drag_float2(
        "##material_texture_repeat", material.tex_repeat, 0.05, 0.01, 1000.0, "%.2f"
    )
    _property_control_row(ctx, "uniform texture scale")
    uniform_changed, uniform = imgui.checkbox("##material_texture_uniform", material.tex_uniform)
    if not (texture_changed or repeat_changed or uniform_changed):
        return material
    return replace(
        material,
        texture=texture,
        tex_repeat=np.asarray(repeat, np.float32),
        tex_uniform=bool(uniform),
    )
