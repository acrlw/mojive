"""Inspector: geometry."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    NodeType,
    SceneNode,
)
from mojive.scene.geometry import geometry_dimensions, geometry_size_from_dimensions
from mojive.ui.panels import (
    PanelContext,
)

from .fields import (
    _begin_property_table,
    _property_button_row,
    _property_color_edit4,
    _property_control_row,
    _property_section,
    _property_vector_row,
)
from .support import (
    _MATERIAL_PRESETS,
    _unique_component_name,
)


class _Geometry:
    """Private geometry methods of InspectorPanel; state belongs to its owner."""

    def _material(self, ctx: PanelContext, node: SceneNode) -> None:
        if not imgui.collapsing_header(ctx.tr("material")):
            return
        src = ctx.session.source
        if src is None or node.body_index < 0 or len(src.geom_body) == 0:
            imgui.text_disabled(ctx.tr("no geometry"))
            return
        instances = (
            np.flatnonzero(np.asarray(src.geom_node) == node.node_id)
            if node.type in (NodeType.GEOM, NodeType.SITE)
            else np.flatnonzero(np.asarray(src.geom_body) == node.body_index)
        )
        if len(instances) == 0:
            imgui.text_disabled(ctx.tr("no geometry on this body"))
            return
        groups: dict[int, list[int]] = {}
        for instance in instances:
            node_id = int(src.geom_node[instance]) if instance < len(src.geom_node) else -1
            groups.setdefault(node_id, []).append(int(instance))
        imgui.text_disabled(f"{len(groups)} {ctx.tr('geometry components')}")
        for node_id, group in list(groups.items())[:8]:
            self._geometry_material(ctx, node_id, group)

    def _geometry_material(self, ctx: PanelContext, node_id: int, instances: list[int]) -> None:
        src = ctx.session.source
        assert src is not None
        first = instances[0]
        material_index = src.geom_material[first] if first < len(src.geom_material) else -1
        if not 0 <= material_index < len(src.materials):
            imgui.text_disabled(ctx.tr("material data is unavailable"))
            return
        material = src.materials[material_index]
        scene_node = ctx.session.node(node_id)
        label = scene_node.name if scene_node is not None else f"geometry {node_id}"
        imgui.push_id(node_id)
        opened = imgui.collapsing_header(
            f"{label}##component", imgui.TreeNodeFlags_.default_open if len(instances) == 1 else 0
        )
        if not opened:
            imgui.pop_id()
            return

        self._geometry_shape_properties(ctx, node_id)

        shape = src.geom_mesh[first].shape
        infinite_plane = bool(src.geom_infinite_plane[first])
        size_editor = geometry_dimensions(shape, src.geom_size[first])
        editable_size = (
            size_editor is not None
            and not infinite_plane
            and scene_node is not None
            and (
                (ctx.session.adapter.caps.topology_editing and scene_node.source_editable)
                or (scene_node.model_id < 0 and ctx.session.adapter.caps.scene_authoring)
            )
        )
        if infinite_plane:
            imgui.text_disabled(ctx.tr("infinite plane"))
        elif size_editor is not None and _property_section(ctx, "geometry dimensions"):
            if not editable_size:
                imgui.begin_disabled()
            dimension_label, dimensions = size_editor.label, size_editor.array()
            size_changed = False
            if _begin_property_table("insp_geometry_dimensions"):
                if len(dimensions) == 1:
                    _property_control_row(ctx, dimension_label)
                    size_changed, scalar = imgui.drag_float(
                        "##geometry_dimension",
                        float(dimensions[0]),
                        0.05,
                        0.002,
                        1000000.0,
                        "%.3f",
                    )
                    dimensions = np.array((scalar,), np.float32)
                elif len(dimensions) == 2:
                    _property_control_row(ctx, dimension_label)
                    size_changed, dimensions = imgui.drag_float2(
                        "##geometry_dimensions_2d",
                        dimensions,
                        0.05,
                        0.002,
                        1000000.0,
                        "%.3f",
                    )
                elif scene_node is not None:
                    size_changed, dimensions = _property_vector_row(
                        ctx,
                        scene_node,
                        dimension_label,
                        "geometry_dimensions",
                        dimensions,
                        editable=editable_size,
                        speed=0.05,
                        lo=0.002,
                        hi=1000000.0,
                        fmt="%.3f",
                        reset_values=np.ones(3, np.float64),
                    )
                else:
                    _property_control_row(ctx, dimension_label)
                    size_changed, dimensions = imgui.drag_float3(
                        "##geometry_dimensions_3d",
                        dimensions,
                        0.05,
                        0.002,
                        1000000.0,
                        "%.3f",
                    )
                imgui.end_table()
            if not editable_size:
                imgui.end_disabled()
            hint = (
                "Full authored primitive dimensions"
                if editable_size
                else "Edit model geometry dimensions in its source"
            )
            imgui.set_item_tooltip(ctx.tr(hint))
            if size_changed and editable_size:
                self._submit_edit(
                    ctx,
                    cmd.SetGeometrySize(
                        node_id,
                        geometry_size_from_dimensions(shape, src.geom_size[first], dimensions),
                    ),
                )

        self._geometry_contact_properties(ctx, node_id)
        self._geometry_advanced_properties(ctx, node_id)

        model_id = scene_node.model_id if scene_node is not None else -1
        model_assets = bool(model_id >= 0 and ctx.session.adapter.caps.model_assets)
        assignment_editable = bool(scene_node is not None and scene_node.source_editable)
        compatible_materials = ctx.session.model_material_indices(model_id) if model_assets else ()
        asset_editable = bool(
            not model_assets or not ctx.session.adapter.caps.simulation or ctx.session.paused
        )
        assigned = not model_assets or material_index in compatible_materials
        prefix = f"opengl_{model_id}_" if model_assets else ""
        assignment_choice: int | None = None
        color_changed = rgba_changed = preset_changed = False
        emission_changed = specular_changed = shininess_changed = False
        reflectance_changed = metallic_toggle_changed = metallic_changed = False
        roughness_toggle_changed = roughness_changed = texture_changed = False
        repeat_changed = uniform_changed = False
        rgba = np.asarray(src.geom_rgba[first], np.float32)
        material_rgba = np.asarray(material.rgba, np.float32)
        emission = material.emission
        specular = material.specular
        shininess = material.shininess
        reflectance = material.reflectance
        metallic = material.metallic
        roughness = material.roughness
        texture = material.texture
        tex_repeat = np.asarray(material.tex_repeat, np.float32)
        tex_uniform = material.tex_uniform
        create = duplicate = import_texture = False
        import_cube = import_skybox = open_assets = False

        if _property_section(ctx, "appearance") and _begin_property_table(
            "insp_geometry_appearance"
        ):
            _property_control_row(ctx, "instance color")
            color_changed, rgba = _property_color_edit4(ctx, "##geometry_instance_color", rgba)

            if model_assets:
                assignment_label = material.name if assigned else ctx.tr("inline appearance")
                if not asset_editable:
                    imgui.begin_disabled()
                _property_control_row(ctx, "assigned material")
                if not assignment_editable:
                    imgui.begin_disabled()
                if imgui.begin_combo("##assigned_material", assignment_label):
                    selected, _ = imgui.selectable(ctx.tr("inline appearance"), not assigned)
                    if selected and assigned:
                        assignment_choice = -1
                    for candidate in compatible_materials:
                        candidate_material = src.materials[candidate]
                        selected, _ = imgui.selectable(
                            candidate_material.name or f"{ctx.tr('material')} {candidate}",
                            candidate == material_index,
                        )
                        if selected and candidate != material_index:
                            assignment_choice = candidate
                    imgui.end_combo()
                if not assignment_editable:
                    imgui.end_disabled()

                if assigned and ctx.panels is not None:
                    (open_assets,) = _property_button_row(ctx, "asset browser", ("Open in Assets",))
                existing_materials = {
                    src.materials[index].name.removeprefix(prefix) for index in compatible_materials
                }
                new_name = _unique_component_name("material", existing_materials)
                if not assignment_editable:
                    imgui.begin_disabled()
                (create,) = _property_button_row(ctx, "create material", ("New material",))
                if not assigned or not assignment_editable:
                    imgui.begin_disabled()
                (duplicate,) = _property_button_row(
                    ctx, "material actions", ("Duplicate material",)
                )
                if not assigned or not assignment_editable:
                    imgui.end_disabled()
                if not assignment_editable:
                    imgui.end_disabled()
                can_import_texture = assigned and ctx.request_texture_import is not None
                if not can_import_texture:
                    imgui.begin_disabled()
                (import_texture,) = _property_button_row(ctx, "texture import", ("Import texture",))
                if not can_import_texture:
                    imgui.end_disabled()
                can_import_environment = ctx.request_texture_import is not None
                if not can_import_environment:
                    imgui.begin_disabled()
                import_cube, import_skybox = _property_button_row(
                    ctx,
                    "environment textures",
                    ("Import cube texture", "Import skybox texture"),
                )
                if not can_import_environment:
                    imgui.end_disabled()
                if not asset_editable:
                    imgui.end_disabled()

            if assigned:
                _property_control_row(ctx, "shared material")
                imgui.align_text_to_frame_padding()
                imgui.text(material.name or str(material_index))
                _property_control_row(ctx, "base color")
                rgba_changed, material_rgba = _property_color_edit4(
                    ctx, "##material_base_color", material.rgba
                )
                _property_control_row(ctx, "preset")
                if imgui.begin_combo("##material_preset", ctx.tr("Presets...")):
                    for preset, values in _MATERIAL_PRESETS.items():
                        selected, _ = imgui.selectable(preset, False)
                        if selected:
                            emission, specular, shininess, reflectance = values
                            preset_changed = True
                    imgui.end_combo()
                _property_control_row(ctx, "emission")
                emission_changed, emission = imgui.drag_float(
                    "##material_emission", emission, 0.01, 0.0, 10.0, "%.2f"
                )
                _property_control_row(ctx, "specular")
                specular_changed, specular = imgui.drag_float(
                    "##material_specular", specular, 0.01, 0.0, 1.0, "%.2f"
                )
                _property_control_row(ctx, "shininess")
                shininess_changed, shininess = imgui.drag_float(
                    "##material_shininess", shininess, 0.01, 0.0, 1.0, "%.2f"
                )
                _property_control_row(ctx, "reflectance")
                reflectance_changed, reflectance = imgui.drag_float(
                    "##material_reflectance", reflectance, 0.01, 0.0, 1.0, "%.2f"
                )
                _property_control_row(ctx, "metallic override")
                metallic_toggle_changed, metallic_enabled = imgui.checkbox(
                    "##material_metallic_override", metallic >= 0.0
                )
                if metallic_toggle_changed:
                    metallic = 0.0 if metallic_enabled else -1.0
                _property_control_row(ctx, "metallic")
                if not metallic_enabled:
                    imgui.begin_disabled()
                metallic_changed, metallic_value = imgui.drag_float(
                    "##material_metallic", max(0.0, metallic), 0.01, 0.0, 1.0, "%.2f"
                )
                if metallic_enabled:
                    metallic = float(metallic_value)
                if not metallic_enabled:
                    imgui.end_disabled()
                _property_control_row(ctx, "roughness override")
                roughness_toggle_changed, roughness_enabled = imgui.checkbox(
                    "##material_roughness_override", roughness >= 0.0
                )
                if roughness_toggle_changed:
                    roughness = 0.5 if roughness_enabled else -1.0
                _property_control_row(ctx, "roughness")
                if not roughness_enabled:
                    imgui.begin_disabled()
                roughness_changed, roughness_value = imgui.drag_float(
                    "##material_roughness", max(0.0, roughness), 0.01, 0.0, 1.0, "%.2f"
                )
                if roughness_enabled:
                    roughness = float(roughness_value)
                if not roughness_enabled:
                    imgui.end_disabled()
                _property_control_row(ctx, "texture")
                if imgui.begin_combo("##material_texture", texture or ctx.tr("none")):
                    compatible_textures = (
                        ctx.session.model_texture_names(model_id)
                        if model_assets
                        else tuple(src.textures)
                    )
                    for candidate in (None, *compatible_textures):
                        selected, _ = imgui.selectable(
                            candidate or ctx.tr("none"), candidate == texture
                        )
                        if selected:
                            texture = candidate
                            texture_changed = True
                    imgui.end_combo()
                _property_control_row(ctx, "texture repeat")
                repeat_changed, tex_repeat = imgui.drag_float2(
                    "##material_texture_repeat",
                    material.tex_repeat,
                    0.05,
                    0.01,
                    1000.0,
                    "%.2f",
                )
                _property_control_row(ctx, "uniform texture scale")
                uniform_changed, tex_uniform = imgui.checkbox(
                    "##material_texture_uniform", material.tex_uniform
                )
            else:
                _property_control_row(ctx, "material properties")
                imgui.text_disabled(
                    ctx.tr("Create or assign a shared material to edit its properties")
                )
            imgui.end_table()

        if color_changed and node_id >= 0:
            self._submit_edit(ctx, cmd.SetGeometryColor(node_id, np.asarray(rgba, np.float32)))
        if assignment_choice is not None:
            self._submit_edit(ctx, cmd.SetGeometryMaterial(node_id, assignment_choice))
            imgui.pop_id()
            return
        if open_assets and ctx.panels is not None:
            panel = ctx.panels.get("Assets")
            focus = getattr(panel, "focus", None)
            if focus is not None:
                focus(model_id, "material", material.name.removeprefix(prefix))
        if create or duplicate:
            ctx.submit(
                cmd.AddModelMaterial(
                    node_id,
                    new_name,
                    material_index if duplicate and assigned else -1,
                )
            )
            imgui.pop_id()
            return
        can_import = bool(assigned and asset_editable and ctx.request_texture_import is not None)
        if import_texture and can_import:
            ctx.request_texture_import(model_id, material_index)
        if import_cube and asset_editable and ctx.request_texture_import is not None:
            ctx.request_texture_import(model_id, -1, "cube")
        if import_skybox and asset_editable and ctx.request_texture_import is not None:
            ctx.request_texture_import(model_id, -1, "skybox")

        if not assigned:
            imgui.pop_id()
            return
        if any(
            (
                emission_changed,
                specular_changed,
                shininess_changed,
                reflectance_changed,
                metallic_toggle_changed,
                metallic_changed,
                roughness_toggle_changed,
                roughness_changed,
                rgba_changed,
                preset_changed,
                texture_changed,
                repeat_changed,
                uniform_changed,
            )
        ):
            self._submit_edit(
                ctx,
                cmd.SetMaterial(
                    material_index,
                    replace(
                        material,
                        rgba=np.asarray(material_rgba, np.float32),
                        emission=float(emission),
                        specular=float(specular),
                        shininess=float(shininess),
                        reflectance=float(reflectance),
                        metallic=float(metallic),
                        roughness=float(roughness),
                        texture=texture,
                        tex_repeat=np.asarray(tex_repeat, np.float32),
                        tex_uniform=bool(tex_uniform),
                    ),
                ),
            )
        imgui.pop_id()

    def _geometry_shape_properties(self, ctx: PanelContext, node_id: int) -> None:
        current = ctx.session.geometry_shape_properties(node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._geometry_shape_node != node_id
            or self._geometry_shape_generation != generation
            or self._geometry_shape_edit is None
        ):
            self._geometry_shape_node = node_id
            self._geometry_shape_generation = generation
            self._geometry_shape_edit = current
            self._geometry_shape_error = ""
        properties = self._geometry_shape_edit
        if properties is None or not imgui.collapsing_header(ctx.tr("geometry shape and resource")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
            and (scene_node := ctx.session.node(node_id)) is not None
            and scene_node.source_editable
        )
        types = (
            "plane",
            "hfield",
            "sphere",
            "capsule",
            "ellipsoid",
            "cylinder",
            "box",
            "mesh",
        )
        type_index = types.index(properties.type)
        type_changed = False
        resource_changed = False
        edited = properties
        if not editable:
            imgui.begin_disabled()
        if _begin_property_table("insp_geometry_shape"):
            _property_control_row(ctx, "geometry type")
            type_changed, type_index = imgui.combo(
                "##geometry_type",
                type_index,
                tuple(value.title() for value in types),
            )
            geom_type = types[type_index]
            if type_changed:
                choices = (
                    properties.mesh_names
                    if geom_type == "mesh"
                    else properties.height_field_names
                    if geom_type == "hfield"
                    else ()
                )
                edited = replace(
                    edited,
                    type=geom_type,
                    resource_name=choices[0] if choices else "",
                )
            resources = (
                properties.mesh_names
                if geom_type == "mesh"
                else properties.height_field_names
                if geom_type == "hfield"
                else ()
            )
            if geom_type in ("mesh", "hfield"):
                _property_control_row(ctx, "resource")
                current_resource = edited.resource_name
                resource_index = (
                    resources.index(current_resource) if current_resource in resources else 0
                )
                if resources:
                    resource_changed, resource_index = imgui.combo(
                        "##geometry_resource", resource_index, resources
                    )
                    if resource_changed or not current_resource:
                        edited = replace(edited, resource_name=resources[resource_index])
                else:
                    imgui.text_disabled(ctx.tr(f"no {geom_type} resources in this model"))
            imgui.end_table()
        else:
            geom_type = types[type_index]
            resources = (
                properties.mesh_names
                if geom_type == "mesh"
                else properties.height_field_names
                if geom_type == "hfield"
                else ()
            )
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model geometry shape"))

        self._geometry_shape_edit = edited
        dirty = edited.type != current.type or edited.resource_name != current.resource_name
        ready = geom_type not in ("mesh", "hfield") or edited.resource_name in resources
        can_import = bool(
            editable
            and ctx.session.adapter.caps.model_assets
            and ctx.request_geometry_resource_import is not None
        )
        if _begin_property_table("insp_geometry_shape_actions"):
            if geom_type in ("mesh", "hfield") and edited.resource_name and ctx.panels is not None:
                (open_assets,) = _property_button_row(ctx, "resource actions", ("Open in Assets",))
                if open_assets:
                    panel = ctx.panels.get("Assets")
                    focus = getattr(panel, "focus", None)
                    node = ctx.session.node(edited.node_id)
                    if focus is not None and node is not None:
                        focus(node.model_id, geom_type, edited.resource_name)

            if not editable or not dirty or not ready:
                imgui.begin_disabled()
            apply, revert = _property_button_row(ctx, "changes", ("Apply", "Revert"))
            if not editable or not dirty or not ready:
                imgui.end_disabled()
            if apply:
                result = ctx.submit(
                    cmd.SetGeometryShape(edited.node_id, edited.type, edited.resource_name)
                )
                if result.ok:
                    self._geometry_shape_generation = -1
                    self._geometry_shape_error = ""
                else:
                    self._geometry_shape_error = result.message
            if revert:
                self._geometry_shape_edit = current
                self._geometry_shape_error = ""

            if not can_import:
                imgui.begin_disabled()
            import_mesh, import_hfield = _property_button_row(
                ctx,
                "import",
                ("Import and assign mesh", "Import and assign height field"),
            )
            if not can_import:
                imgui.end_disabled()
            if import_mesh and can_import:
                ctx.request_geometry_resource_import(node_id, "mesh")
            if import_hfield and can_import:
                ctx.request_geometry_resource_import(node_id, "hfield")
            imgui.end_table()
        if self._geometry_shape_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._geometry_shape_error)
            if imgui.button(f"{ctx.tr('Copy error')}##geometry-shape"):
                imgui.set_clipboard_text(self._geometry_shape_error)

    def _geometry_contact_properties(self, ctx: PanelContext, node_id: int) -> None:
        properties = ctx.session.geometry_properties(node_id)
        if properties is None or not imgui.collapsing_header(ctx.tr("contact properties")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
            and (scene_node := ctx.session.node(node_id)) is not None
            and scene_node.source_editable
        )
        friction = np.asarray(properties.friction, np.float32)
        dimension = 1
        type_mask = properties.collision_type_mask
        affinity_mask = properties.collision_affinity_mask
        priority = properties.contact_priority
        margin = properties.margin
        gap = properties.gap
        solver_mix = properties.solver_mix
        solver_reference = np.asarray(properties.solver_reference, np.float32)
        impedance_first = np.asarray(properties.solver_impedance[:3], np.float32)
        impedance_shape = np.asarray(properties.solver_impedance[3:], np.float32)
        adhesion = properties.adhesion
        friction_changed = dimension_changed = type_changed = affinity_changed = False
        priority_changed = margin_changed = gap_changed = mix_changed = False
        reference_changed = impedance_first_changed = impedance_shape_changed = False
        adhesion_changed = linear_velocity_changed = angular_velocity_changed = False
        linear_velocity = np.asarray(properties.surface_velocity[:3], np.float32)
        angular_velocity = np.asarray(properties.surface_velocity[3:], np.float32)
        if not editable:
            imgui.begin_disabled()
        dimensions = (1, 3, 4, 6)
        dimension_labels = (
            "1 · frictionless",
            "3 · sliding",
            "4 · sliding + torsional",
            "6 · sliding + torsional + rolling",
        )
        dimension = (
            dimensions.index(properties.contact_dimension)
            if properties.contact_dimension in dimensions
            else 1
        )
        scene_node = ctx.session.node(node_id)
        if _begin_property_table("insp_geometry_contact"):
            _property_control_row(ctx, "friction (slide spin roll)")
            friction_changed, friction = imgui.drag_float3(
                "##contact_friction", friction, 0.005, 0.0, 1000000.0, "%.5f"
            )
            _property_control_row(ctx, "contact dimension")
            dimension_changed, dimension = imgui.combo(
                "##contact_dimension",
                dimension,
                tuple(ctx.tr(label) for label in dimension_labels),
            )
            _property_control_row(ctx, "collision type mask")
            type_changed, type_mask = imgui.input_int(
                "##collision_type_mask", properties.collision_type_mask, 1, 16
            )
            imgui.set_item_tooltip(ctx.tr("Decimal MuJoCo contype bitmask"))
            _property_control_row(ctx, "collision affinity mask")
            affinity_changed, affinity_mask = imgui.input_int(
                "##collision_affinity_mask", properties.collision_affinity_mask, 1, 16
            )
            imgui.set_item_tooltip(ctx.tr("Decimal MuJoCo conaffinity bitmask"))
            _property_control_row(ctx, "contact priority")
            priority_changed, priority = imgui.drag_int(
                "##contact_priority", properties.contact_priority, 1.0, 0, 2147483647, "%d"
            )
            _property_control_row(ctx, "contact margin")
            margin_changed, margin = imgui.drag_float(
                "##contact_margin", properties.margin, 0.001, 0.0, 1000000.0, "%.5f m"
            )
            _property_control_row(ctx, "contact gap")
            gap_changed, gap = imgui.drag_float(
                "##contact_gap", properties.gap, 0.001, 0.0, 1000000.0, "%.5f m"
            )
            _property_control_row(ctx, "solver mix")
            mix_changed, solver_mix = imgui.drag_float(
                "##contact_solver_mix", properties.solver_mix, 0.01, 0.0, 1.0, "%.3f"
            )
            _property_control_row(ctx, "solver reference")
            reference_changed, solver_reference = imgui.drag_float2(
                "##contact_solver_reference",
                solver_reference,
                0.001,
                -1000000.0,
                1000000.0,
                "%.5g",
            )
            imgui.set_item_tooltip(
                ctx.tr(
                    "Positive values use time-constant/damping-ratio format; non-positive values "
                    "use direct stiffness/damping format"
                )
            )
            _property_control_row(ctx, "impedance min / max / width")
            impedance_first_changed, impedance_first = imgui.drag_float3(
                "##contact_impedance_first", impedance_first, 0.001, 0.0, 1.0, "%.5g"
            )
            _property_control_row(ctx, "impedance midpoint / power")
            impedance_shape_changed, impedance_shape = imgui.drag_float2(
                "##contact_impedance_shape", impedance_shape, 0.01, 0.0, 1000.0, "%.4g"
            )
            _property_control_row(ctx, "adhesion")
            adhesion_changed, adhesion = imgui.drag_float(
                "##contact_adhesion", properties.adhesion, 0.01, 0.0, 1000000000.0, "%.5g"
            )
            if scene_node is not None:
                linear_velocity_changed, linear_velocity = _property_vector_row(
                    ctx,
                    scene_node,
                    "surface linear velocity",
                    "contact_surface_linear_velocity",
                    properties.surface_velocity[:3],
                    editable=editable,
                    speed=0.01,
                    lo=0.0,
                    hi=0.0,
                    fmt="%.4g",
                    reset_values=properties.surface_velocity[:3],
                )
                angular_velocity_changed, angular_velocity = _property_vector_row(
                    ctx,
                    scene_node,
                    "surface angular velocity",
                    "contact_surface_angular_velocity",
                    properties.surface_velocity[3:],
                    editable=editable,
                    speed=0.01,
                    lo=0.0,
                    hi=0.0,
                    fmt="%.4g",
                    reset_values=properties.surface_velocity[3:],
                )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model contact properties"))

        invalid_masks = type_mask < 0 or affinity_mask < 0
        if invalid_masks:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning), "Collision masks cannot be negative"
            )
        changed = any(
            (
                friction_changed,
                dimension_changed,
                type_changed,
                affinity_changed,
                priority_changed,
                margin_changed,
                gap_changed,
                mix_changed,
                reference_changed,
                impedance_first_changed,
                impedance_shape_changed,
                adhesion_changed,
                linear_velocity_changed,
                angular_velocity_changed,
            )
        )
        if changed and editable and not invalid_masks:
            self._submit_edit(
                ctx,
                cmd.SetGeometryProperties(
                    node_id,
                    tuple(float(value) for value in friction),
                    int(type_mask),
                    int(affinity_mask),
                    dimensions[dimension],
                    int(priority),
                    float(margin),
                    float(gap),
                    float(solver_mix),
                    tuple(float(value) for value in solver_reference),
                    tuple(float(value) for value in (*impedance_first, *impedance_shape)),
                    float(adhesion),
                    tuple(float(value) for value in (*linear_velocity, *angular_velocity)),
                ),
            )

    def _geometry_advanced_properties(self, ctx: PanelContext, node_id: int) -> None:
        current = ctx.session.geometry_advanced_properties(node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._geometry_advanced_node != node_id
            or self._geometry_advanced_generation != generation
            or self._geometry_advanced_edit is None
        ):
            self._geometry_advanced_node = node_id
            self._geometry_advanced_generation = generation
            self._geometry_advanced_edit = current
            self._geometry_advanced_error = ""
        properties = self._geometry_advanced_edit
        if properties is None or not imgui.collapsing_header(ctx.tr("mass, group, and fluid")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        visual_group = int(properties.visual_group)
        mass_values = ("density", "mass")
        mass_mode = mass_values.index(properties.mass_mode)
        mass_mode_value = properties.mass_mode
        density = properties.density
        mass = properties.mass
        inertia_values = ("volume", "shell")
        inertia_mode = inertia_values.index(properties.inertia_mode)
        fluid_ellipsoid = properties.fluid_ellipsoid
        first_fluid = np.asarray(properties.fluid_coefficients[:3], np.float32)
        last_fluid = np.asarray(properties.fluid_coefficients[3:], np.float32)
        group_changed = mass_mode_changed = mass_value_changed = False
        inertia_changed = fluid_changed = first_fluid_changed = last_fluid_changed = False
        if not editable:
            imgui.begin_disabled()
        if _begin_property_table("insp_geometry_advanced"):
            _property_control_row(ctx, "visual group")
            group_changed, visual_group = imgui.combo(
                "##geometry_visual_group",
                int(properties.visual_group),
                tuple(str(value) for value in range(6)),
            )
            imgui.set_item_tooltip(ctx.tr("MuJoCo geom group used by visibility filters"))
            _property_control_row(ctx, "mass source")
            mass_mode_changed, mass_mode = imgui.combo(
                "##geometry_mass_source",
                mass_mode,
                (ctx.tr("density"), ctx.tr("explicit mass")),
            )
            mass_mode_value = mass_values[mass_mode]
            if mass_mode_value == "density":
                _property_control_row(ctx, "density")
                mass_value_changed, density = imgui.drag_float(
                    "##geometry_density",
                    properties.density,
                    1.0,
                    0.000001,
                    1000000000000.0,
                    "%.6g kg/m³",
                )
            else:
                _property_control_row(ctx, "mass")
                mass_value_changed, mass = imgui.drag_float(
                    "##geometry_mass",
                    properties.mass,
                    0.01,
                    0.000001,
                    1000000000000.0,
                    "%.6g kg",
                )
            _property_control_row(ctx, "inertia distribution")
            inertia_changed, inertia_mode = imgui.combo(
                "##geometry_inertia_distribution",
                inertia_mode,
                tuple(ctx.tr(value) for value in inertia_values),
            )
            _property_control_row(ctx, "ellipsoid fluid interaction")
            fluid_changed, fluid_ellipsoid = imgui.checkbox(
                "##geometry_fluid_ellipsoid", properties.fluid_ellipsoid
            )
            _property_control_row(ctx, "fluid blunt / slender / angular")
            first_fluid_changed, first_fluid = imgui.drag_float3(
                "##geometry_fluid_first", first_fluid, 0.01, 0.0, 1000000.0, "%.4g"
            )
            _property_control_row(ctx, "fluid Kutta / Magnus")
            last_fluid_changed, last_fluid = imgui.drag_float2(
                "##geometry_fluid_last", last_fluid, 0.01, 0.0, 1000000.0, "%.4g"
            )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model geometry properties"))

        edited = properties
        if group_changed:
            edited = replace(edited, visual_group=int(visual_group))
        if mass_mode_changed:
            edited = replace(edited, mass_mode=mass_mode_value)
        if mass_value_changed:
            edited = (
                replace(edited, density=float(density))
                if mass_mode_value == "density"
                else replace(edited, mass=float(mass))
            )
        if inertia_changed:
            edited = replace(edited, inertia_mode=inertia_values[inertia_mode])
        if fluid_changed:
            edited = replace(edited, fluid_ellipsoid=bool(fluid_ellipsoid))
        if first_fluid_changed or last_fluid_changed:
            edited = replace(
                edited,
                fluid_coefficients=tuple(float(value) for value in (*first_fluid, *last_fluid)),
            )
        self._geometry_advanced_edit = edited
        dirty = edited != current
        if _begin_property_table("insp_geometry_advanced_actions"):
            if not editable or not dirty:
                imgui.begin_disabled()
            apply, revert = _property_button_row(ctx, "changes", ("Apply", "Revert"))
            if not editable or not dirty:
                imgui.end_disabled()
            imgui.end_table()
            if apply:
                result = ctx.submit(
                    cmd.SetGeometryAdvancedProperties(
                        node_id=edited.node_id,
                        visual_group=edited.visual_group,
                        mass_mode=edited.mass_mode,
                        mass=edited.mass,
                        density=edited.density,
                        inertia_mode=edited.inertia_mode,
                        fluid_ellipsoid=edited.fluid_ellipsoid,
                        fluid_coefficients=edited.fluid_coefficients,
                    )
                )
                if result.ok:
                    self._geometry_advanced_generation = -1
                    self._geometry_advanced_error = ""
                else:
                    self._geometry_advanced_error = result.message
            if revert:
                self._geometry_advanced_edit = current
                self._geometry_advanced_error = ""
        if self._geometry_advanced_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._geometry_advanced_error)
            if imgui.button(f"{ctx.tr('Copy error')}##geometry-advanced"):
                imgui.set_clipboard_text(self._geometry_advanced_error)
