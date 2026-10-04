"""Inspector: geometry."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    NodeType,
    SceneNode,
    SceneSource,
)
from mojive.scene.geometry import geometry_dimensions, geometry_size_from_dimensions
from mojive.ui.panels import (
    PanelContext,
)

from .fields import (
    _begin_property_table,
    _edited_float_components,
    _property_button_row,
    _property_color_edit4,
    _property_control_row,
    _property_section,
    _property_solver_rows,
    _property_vector_row,
)
from .materials import _material_asset_rows, _material_surface_rows, _material_texture_rows


def _shape_resources(properties: GeometryShapeProperties) -> tuple[str, ...]:
    if properties.type == "mesh":
        return properties.mesh_names
    if properties.type == "hfield":
        return properties.height_field_names
    return ()


def _geometry_shape_rows(
    ctx: PanelContext, properties: GeometryShapeProperties
) -> GeometryShapeProperties:
    """Edit a shape/resource value; the panel owns its draft and commit lifecycle."""
    types = ("plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh")
    _property_control_row(ctx, "geometry type")
    changed, index = imgui.combo(
        "##geometry_type", types.index(properties.type), tuple(value.title() for value in types)
    )
    edited = replace(properties, type=types[index]) if changed else properties
    resources = _shape_resources(edited)
    if changed:
        edited = replace(edited, resource_name=resources[0] if resources else "")
    if edited.type in ("mesh", "hfield"):
        _property_control_row(ctx, "resource")
        if resources:
            index = (
                resources.index(edited.resource_name) if edited.resource_name in resources else 0
            )
            resource_changed, index = imgui.combo("##geometry_resource", index, resources)
            if resource_changed or not edited.resource_name:
                edited = replace(edited, resource_name=resources[index])
        else:
            imgui.text_disabled(ctx.tr(f"no {edited.type} resources in this model"))
    return edited


def _contact_collision_rows(
    ctx: PanelContext, properties: GeometryProperties
) -> GeometryProperties:
    edited = properties
    _property_control_row(ctx, "friction (slide spin roll)")
    changed, friction = imgui.drag_float3(
        "##contact_friction", properties.friction, 0.005, 0.0, 1000000.0, "%.5f"
    )
    if changed:
        edited = replace(edited, friction=_edited_float_components(properties.friction, friction))
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
    _property_control_row(ctx, "contact dimension")
    changed, dimension = imgui.combo(
        "##contact_dimension", dimension, tuple(ctx.tr(label) for label in dimension_labels)
    )
    if changed:
        edited = replace(edited, contact_dimension=dimensions[dimension])
    _property_control_row(ctx, "collision type mask")
    changed, type_mask = imgui.input_int(
        "##collision_type_mask", properties.collision_type_mask, 1, 16
    )
    imgui.set_item_tooltip(ctx.tr("Decimal MuJoCo contype bitmask"))
    if changed:
        edited = replace(edited, collision_type_mask=int(type_mask))
    _property_control_row(ctx, "collision affinity mask")
    changed, affinity_mask = imgui.input_int(
        "##collision_affinity_mask", properties.collision_affinity_mask, 1, 16
    )
    imgui.set_item_tooltip(ctx.tr("Decimal MuJoCo conaffinity bitmask"))
    if changed:
        edited = replace(edited, collision_affinity_mask=int(affinity_mask))
    _property_control_row(ctx, "contact priority")
    changed, priority = imgui.drag_int(
        "##contact_priority", properties.contact_priority, 1.0, 0, 2147483647, "%d"
    )
    if changed:
        edited = replace(edited, contact_priority=int(priority))
    return edited


def _contact_solver_rows(ctx: PanelContext, properties: GeometryProperties) -> GeometryProperties:
    edited = properties
    _property_control_row(ctx, "contact margin")
    changed, margin = imgui.drag_float(
        "##contact_margin", properties.margin, 0.001, 0.0, 1000000.0, "%.5f m"
    )
    if changed:
        edited = replace(edited, margin=float(margin))
    _property_control_row(ctx, "contact gap")
    changed, gap = imgui.drag_float(
        "##contact_gap", properties.gap, 0.001, 0.0, 1000000.0, "%.5f m"
    )
    if changed:
        edited = replace(edited, gap=float(gap))
    _property_control_row(ctx, "solver mix")
    changed, solver_mix = imgui.drag_float(
        "##contact_solver_mix", properties.solver_mix, 0.01, 0.0, 1.0, "%.3f"
    )
    if changed:
        edited = replace(edited, solver_mix=float(solver_mix))
    changed, reference, impedance = _property_solver_rows(
        ctx,
        "contact",
        properties.solver_reference,
        properties.solver_impedance,
        reference_tooltip=(
            "Positive values use time-constant/damping-ratio format; non-positive values "
            "use direct stiffness/damping format"
        ),
    )
    if changed:
        edited = replace(edited, solver_reference=reference, solver_impedance=impedance)
    return edited


def _contact_surface_rows(
    ctx: PanelContext, node: SceneNode | None, properties: GeometryProperties, *, editable: bool
) -> GeometryProperties:
    edited = properties
    _property_control_row(ctx, "adhesion")
    changed, adhesion = imgui.drag_float(
        "##contact_adhesion", properties.adhesion, 0.01, 0.0, 1000000000.0, "%.5g"
    )
    if changed:
        edited = replace(edited, adhesion=float(adhesion))
    if node is None:
        return edited
    linear_changed, linear = _property_vector_row(
        ctx,
        node,
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
    angular_changed, angular = _property_vector_row(
        ctx,
        node,
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
    if linear_changed or angular_changed:
        edited = replace(
            edited,
            surface_velocity=(
                *(
                    tuple(float(value) for value in linear)
                    if linear_changed
                    else properties.surface_velocity[:3]
                ),
                *(
                    tuple(float(value) for value in angular)
                    if angular_changed
                    else properties.surface_velocity[3:]
                ),
            ),
        )
    return edited


def _geometry_mass_rows(
    ctx: PanelContext, properties: GeometryAdvancedProperties
) -> GeometryAdvancedProperties:
    edited = properties
    _property_control_row(ctx, "visual group")
    changed, group = imgui.combo(
        "##geometry_visual_group", properties.visual_group, tuple(str(value) for value in range(6))
    )
    imgui.set_item_tooltip(ctx.tr("MuJoCo geom group used by visibility filters"))
    if changed:
        edited = replace(edited, visual_group=int(group))
    mass_modes = ("density", "mass")
    _property_control_row(ctx, "mass source")
    changed, mode = imgui.combo(
        "##geometry_mass_source",
        mass_modes.index(properties.mass_mode),
        (ctx.tr("density"), ctx.tr("explicit mass")),
    )
    if changed:
        edited = replace(edited, mass_mode=mass_modes[mode])
    if edited.mass_mode == "density":
        _property_control_row(ctx, "density")
        changed, density = imgui.drag_float(
            "##geometry_density", properties.density, 1.0, 0.000001, 1000000000000.0, "%.6g kg/m³"
        )
        if changed:
            edited = replace(edited, density=float(density))
    else:
        _property_control_row(ctx, "mass")
        changed, mass = imgui.drag_float(
            "##geometry_mass", properties.mass, 0.01, 0.000001, 1000000000000.0, "%.6g kg"
        )
        if changed:
            edited = replace(edited, mass=float(mass))
    inertia_modes = ("volume", "shell")
    _property_control_row(ctx, "inertia distribution")
    changed, mode = imgui.combo(
        "##geometry_inertia_distribution",
        inertia_modes.index(properties.inertia_mode),
        tuple(ctx.tr(value) for value in inertia_modes),
    )
    if changed:
        edited = replace(edited, inertia_mode=inertia_modes[mode])
    return edited


def _geometry_fluid_rows(
    ctx: PanelContext, properties: GeometryAdvancedProperties
) -> GeometryAdvancedProperties:
    edited = properties
    _property_control_row(ctx, "ellipsoid fluid interaction")
    changed, ellipsoid = imgui.checkbox("##geometry_fluid_ellipsoid", properties.fluid_ellipsoid)
    if changed:
        edited = replace(edited, fluid_ellipsoid=bool(ellipsoid))
    coefficients = properties.fluid_coefficients
    _property_control_row(ctx, "fluid blunt / slender / angular")
    first_changed, first = imgui.drag_float3(
        "##geometry_fluid_first", coefficients[:3], 0.01, 0.0, 1000000.0, "%.4g"
    )
    _property_control_row(ctx, "fluid Kutta / Magnus")
    last_changed, last = imgui.drag_float2(
        "##geometry_fluid_last", coefficients[3:], 0.01, 0.0, 1000000.0, "%.4g"
    )
    if first_changed or last_changed:
        edited = replace(
            edited,
            fluid_coefficients=(
                *(
                    _edited_float_components(coefficients[:3], first)
                    if first_changed
                    else coefficients[:3]
                ),
                *(
                    _edited_float_components(coefficients[3:], last)
                    if last_changed
                    else coefficients[3:]
                ),
            ),
        )
    return edited


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
        scene_node = ctx.session.node(node_id)
        label = scene_node.name if scene_node is not None else f"geometry {node_id}"
        imgui.push_id(node_id)
        try:
            if not imgui.collapsing_header(
                f"{label}##component",
                imgui.TreeNodeFlags_.default_open if len(instances) == 1 else 0,
            ):
                return
            self._geometry_shape_properties(ctx, node_id)
            self._geometry_dimensions(ctx, src, scene_node, node_id, first)
            self._geometry_contact_properties(ctx, node_id)
            self._geometry_advanced_properties(ctx, node_id)
            self._geometry_appearance(ctx, src, scene_node, node_id, first, material_index)
        finally:
            imgui.pop_id()

    def _geometry_dimensions(
        self,
        ctx: PanelContext,
        src: SceneSource,
        scene_node: SceneNode | None,
        node_id: int,
        first: int,
    ) -> None:
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
                "Full primitive dimensions"
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

    def _geometry_appearance(
        self,
        ctx: PanelContext,
        src: SceneSource,
        scene_node: SceneNode | None,
        node_id: int,
        first: int,
        material_index: int,
    ) -> None:
        if not _property_section(ctx, "appearance") or not _begin_property_table(
            "insp_geometry_appearance"
        ):
            return
        material = src.materials[material_index]
        model_id = scene_node.model_id if scene_node is not None else -1
        model_assets = bool(model_id >= 0 and ctx.session.adapter.caps.model_assets)
        compatible_materials = ctx.session.model_material_indices(model_id) if model_assets else ()
        asset_editable = bool(
            not model_assets or not ctx.session.adapter.caps.simulation or ctx.session.paused
        )
        assigned = not model_assets or material_index in compatible_materials
        _property_control_row(ctx, "instance color")
        color_changed, rgba = _property_color_edit4(
            ctx, "##geometry_instance_color", np.asarray(src.geom_rgba[first], np.float32)
        )
        actions = None
        if model_assets:
            actions = _material_asset_rows(
                ctx,
                src,
                node_id,
                material_index,
                compatible_materials,
                model_id=model_id,
                assigned=assigned,
                editable=asset_editable,
                assignment_editable=bool(scene_node is not None and scene_node.source_editable),
            )
        edited = material
        if assigned:
            edited = _material_surface_rows(ctx, edited, material_index)
            edited = _material_texture_rows(
                ctx, edited, src, model_id=model_id, model_assets=model_assets
            )
        else:
            _property_control_row(ctx, "material properties")
            imgui.text_disabled(ctx.tr("Create or assign a shared material to edit its properties"))
        imgui.end_table()

        # Submit after drawing: assignment and creation rebuild the material catalog and
        # take precedence over property edits made against the old assignment this frame.
        if color_changed and node_id >= 0:
            self._submit_edit(ctx, cmd.SetGeometryColor(node_id, np.asarray(rgba, np.float32)))
        if actions is not None:
            if actions.assignment is not None:
                self._submit_edit(ctx, actions.assignment)
                return
            if actions.open_assets and ctx.panels is not None:
                panel = ctx.panels.load("Assets")
                focus = getattr(panel, "focus", None)
                if focus is not None:
                    focus(model_id, "material", material.name.removeprefix(f"opengl_{model_id}_"))
            if actions.creation is not None:
                ctx.submit(actions.creation)
                return
            if asset_editable and ctx.request_texture_import is not None:
                if actions.import_texture and assigned:
                    ctx.request_texture_import(model_id, material_index)
                if actions.import_cube:
                    ctx.request_texture_import(model_id, -1, "cube")
                if actions.import_skybox:
                    ctx.request_texture_import(model_id, -1, "skybox")
        if edited is not material:
            self._submit_edit(ctx, cmd.SetMaterial(material_index, edited))

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
        if not editable:
            imgui.begin_disabled()
        if _begin_property_table("insp_geometry_shape"):
            self._geometry_shape_edit = _geometry_shape_rows(ctx, properties)
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model geometry shape"))
        self._geometry_shape_actions(ctx, current, self._geometry_shape_edit, editable=editable)

    def _geometry_shape_actions(
        self,
        ctx: PanelContext,
        current: GeometryShapeProperties,
        edited: GeometryShapeProperties,
        *,
        editable: bool,
    ) -> None:
        dirty = edited.type != current.type or edited.resource_name != current.resource_name
        ready = edited.type not in ("mesh", "hfield") or edited.resource_name in _shape_resources(
            edited
        )
        can_import = bool(
            editable
            and ctx.session.adapter.caps.model_assets
            and ctx.request_geometry_resource_import is not None
        )
        open_assets = apply = revert = import_mesh = import_hfield = False
        if _begin_property_table("insp_geometry_shape_actions"):
            if (
                edited.type in ("mesh", "hfield")
                and edited.resource_name
                and ctx.panels is not None
            ):
                (open_assets,) = _property_button_row(ctx, "resource actions", ("Open in Assets",))
            apply, revert = _property_button_row(
                ctx,
                "changes",
                ("Apply", "Revert"),
                enabled=(editable and dirty and ready, dirty),
            )
            import_mesh, import_hfield = _property_button_row(
                ctx,
                "import",
                ("Import and assign mesh", "Import and assign height field"),
                enabled=(can_import, can_import),
            )
            imgui.end_table()
        # Resource navigation and model rebuilds run after all actions have been drawn.
        if open_assets:
            panel = ctx.panels.load("Assets")
            focus = getattr(panel, "focus", None)
            node = ctx.session.node(edited.node_id)
            if focus is not None and node is not None:
                focus(node.model_id, edited.type, edited.resource_name)
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
        if import_mesh:
            ctx.request_geometry_resource_import(edited.node_id, "mesh")
        if import_hfield:
            ctx.request_geometry_resource_import(edited.node_id, "hfield")
        if self._geometry_shape_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._geometry_shape_error)
            if imgui.button(f"{ctx.tr('Copy error')}##geometry-shape"):
                imgui.set_clipboard_text(self._geometry_shape_error)

    def _geometry_contact_properties(self, ctx: PanelContext, node_id: int) -> None:
        properties = ctx.session.geometry_properties(node_id)
        if properties is None or not imgui.collapsing_header(ctx.tr("contact properties")):
            return
        node = ctx.session.node(node_id)
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
            and node is not None
            and node.source_editable
        )
        edited = properties
        if not editable:
            imgui.begin_disabled()
        if _begin_property_table("insp_geometry_contact"):
            edited = _contact_collision_rows(ctx, edited)
            edited = _contact_solver_rows(ctx, edited)
            edited = _contact_surface_rows(ctx, node, edited, editable=editable)
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model contact properties"))
        invalid_masks = edited.collision_type_mask < 0 or edited.collision_affinity_mask < 0
        if invalid_masks:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning), "Collision masks cannot be negative"
            )
        if edited != properties and editable and not invalid_masks:
            self._submit_edit(
                ctx,
                cmd.SetGeometryProperties(
                    node_id=node_id,
                    friction=edited.friction,
                    collision_type_mask=edited.collision_type_mask,
                    collision_affinity_mask=edited.collision_affinity_mask,
                    contact_dimension=edited.contact_dimension,
                    contact_priority=edited.contact_priority,
                    margin=edited.margin,
                    gap=edited.gap,
                    solver_mix=edited.solver_mix,
                    solver_reference=edited.solver_reference,
                    solver_impedance=edited.solver_impedance,
                    adhesion=edited.adhesion,
                    surface_velocity=edited.surface_velocity,
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
        if not editable:
            imgui.begin_disabled()
        if _begin_property_table("insp_geometry_advanced"):
            edited = _geometry_mass_rows(ctx, properties)
            self._geometry_advanced_edit = _geometry_fluid_rows(ctx, edited)
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model geometry properties"))
        self._geometry_advanced_actions(
            ctx, current, self._geometry_advanced_edit, editable=editable
        )

    def _geometry_advanced_actions(
        self,
        ctx: PanelContext,
        current: GeometryAdvancedProperties,
        edited: GeometryAdvancedProperties,
        *,
        editable: bool,
    ) -> None:
        dirty = edited != current
        apply = revert = False
        if _begin_property_table("insp_geometry_advanced_actions"):
            apply, revert = _property_button_row(
                ctx, "changes", ("Apply", "Revert"), enabled=(editable and dirty, dirty)
            )
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
