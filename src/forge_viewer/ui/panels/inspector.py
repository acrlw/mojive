"""Selected entity properties and transform editing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ... import math3d
from ...adapters.base import (
    BodyProperties,
    FrameNeeds,
    GeometryAdvancedProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    ModelComponentInfo,
    ModelComponentPathItem,
    NodeType,
    SceneModelInfo,
    SceneNode,
    SiteProperties,
)
from ...render.backend import RenderFlag
from ...types import DEFAULT_HEADLIGHT, Environment, LightType, MeshShape, TextureType
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_row_layout,
    button_width,
    labeled,
    segmented_control,
)

GIZMO_REFUSAL_RUNNING = "Physics is running; pause to move things"
GIZMO_REFUSAL_DRIVEN = "This link is joint-driven; use its joint gizmo or the Joints panel"
_MATERIAL_PRESETS = {
    "Matte": (0.0, 0.05, 0.10, 0.0),
    "Plastic": (0.0, 0.40, 0.50, 0.05),
    "Metal": (0.0, 0.90, 0.90, 0.65),
    "Rubber": (0.0, 0.08, 0.20, 0.0),
    "Emissive": (1.0, 0.10, 0.20, 0.0),
}


def _geometry_dimensions(shape: MeshShape, size) -> tuple[str, np.ndarray] | None:
    """Return user-facing full dimensions for one editable primitive."""
    value = np.asarray(size, np.float32).reshape(3)
    if shape is MeshShape.PLANE:
        return "width / length", value[:2] * 2.0
    if shape is MeshShape.BOX:
        return "width / depth / height", value * 2.0
    if shape is MeshShape.SPHERE:
        if np.allclose(value, value[0], rtol=1e-5, atol=1e-7):
            return "diameter", value[:1] * 2.0
        return "width / depth / height", value * 2.0
    if shape is MeshShape.CYLINDER:
        return "diameter / height", np.array((value[0] * 2.0, value[2] * 2.0))
    if shape is MeshShape.CAPSULE_SHAFT:
        return "diameter / shaft length", np.array((value[0] * 2.0, value[2] * 2.0))
    return None


def _geometry_size_from_dimensions(shape: MeshShape, size, dimensions) -> np.ndarray:
    """Convert full UI dimensions back to the render-size convention."""
    value = np.asarray(size, np.float32).reshape(3).copy()
    dimensions = np.maximum(np.asarray(dimensions, np.float32).reshape(-1), 0.002)
    half = dimensions * 0.5
    if shape is MeshShape.PLANE:
        value[:2] = half[:2]
    elif shape is MeshShape.BOX or (shape is MeshShape.SPHERE and len(half) == 3):
        value[:] = half[:3]
    elif shape is MeshShape.SPHERE:
        value[:] = half[0]
    elif shape in (MeshShape.CYLINDER, MeshShape.CAPSULE_SHAFT):
        value[:2] = half[0]
        value[2] = half[1]
    return value


def _unique_component_name(category: str, existing: set[str]) -> str:
    if category not in existing:
        return category
    index = 2
    while f"{category}{index}" in existing:
        index += 1
    return f"{category}{index}"


_MULTILINE_COMPONENT_FIELDS = {
    "act",
    "body",
    "cellcount",
    "ctrl",
    "data",
    "element",
    "elemtexcoord",
    "face",
    "mpos",
    "mquat",
    "node",
    "point",
    "qpos",
    "qvel",
    "texcoord",
    "vertex",
    "vertid",
    "vertweight",
}


def _component_value_editor(
    label: str,
    value: str,
    choices: tuple[str, ...],
    *,
    multiline: bool = False,
) -> str:
    if choices:
        if imgui.begin_combo(label, value or "select"):
            for index, choice in enumerate(choices):
                selected, _ = imgui.selectable(
                    f"{choice or '<default>'}##choice-{index}", choice == value
                )
                if selected:
                    value = choice
            imgui.end_combo()
        return value
    if multiline:
        _changed, value = imgui.input_text_multiline(
            label,
            value,
            imgui.ImVec2(-1.0, 68.0),
        )
        return value
    _changed, value = imgui.input_text(label, value)
    return value


_MODEL_COMPONENT_CATEGORIES = (
    "contact",
    "actuator",
    "sensor",
    "tendon",
    "equality",
)


def _path_preset_label(preset: ModelComponentPathItem) -> str:
    object_type = next((field.value for field in preset.fields if field.name == "objtype"), "")
    return f"{preset.type} · {object_type}" if object_type else preset.type


def _matching_path_preset(
    presets: tuple[ModelComponentPathItem, ...],
    element_type: str,
    fields: list[list[str]],
) -> ModelComponentPathItem | None:
    object_type = next((value for name, value in fields if name == "objtype"), "")
    if not object_type:
        return None
    return next(
        (
            preset
            for preset in presets
            if preset.type == element_type
            and any(
                field.name == "objtype" and field.value == object_type for field in preset.fields
            )
        ),
        None,
    )


def gizmo_refusal_reason(
    paused: bool,
    posable: bool,
) -> str | None:
    if not paused:
        return GIZMO_REFUSAL_RUNNING
    if not posable:
        return GIZMO_REFUSAL_DRIVEN
    return None


class InspectorPanel(Panel):
    name = "Inspector"
    default_open = True
    shortcut = "F4"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self.show_transform = True
        self._transform_velocity = False
        self.show_velocity = False
        self._rotation_node = -1
        self._rotation_euler = np.zeros(3, np.float64)
        self._rotation_matrix = np.eye(3, dtype=np.float64)
        self._edit_transaction = False
        self._model_name_node = -1
        self._model_name = ""
        self._source_model_id = -1
        self._source_text = ""
        self._source_error = ""
        self._open_source_popup = False
        self._component_cache_generation = -1
        self._component_cache_model = -1
        self._component_cache: dict[str, tuple[ModelComponentInfo, ...]] = {}
        self._component_presets: dict[str, tuple[str, ...]] = {}
        self._component_edit: ModelComponentInfo | None = None
        self._component_name = ""
        self._component_fields: list[list[str]] = []
        self._component_path: list[tuple[str, list[list[str]]]] = []
        self._component_path_choices: list[dict[str, tuple[str, ...]]] = []
        self._component_path_presets = ()
        self._component_error = ""
        self._open_component_popup = False
        self._model_transform_model = -1
        self._model_transform_generation = -1
        self._model_transform_position = np.zeros(3, np.float32)
        self._model_transform_euler = np.zeros(3, np.float64)
        self._body_property_node = -1
        self._body_property_generation = -1
        self._body_property_edit: BodyProperties | None = None
        self._body_inertial_euler = np.zeros(3, np.float64)
        self._body_property_error = ""
        self._geometry_advanced_node = -1
        self._geometry_advanced_generation = -1
        self._geometry_advanced_edit: GeometryAdvancedProperties | None = None
        self._geometry_advanced_error = ""
        self._geometry_shape_node = -1
        self._geometry_shape_generation = -1
        self._geometry_shape_edit: GeometryShapeProperties | None = None
        self._geometry_shape_error = ""
        self._joint_advanced_id = -1
        self._joint_advanced_generation = -1
        self._joint_advanced_edit: JointAdvancedProperties | None = None
        self._joint_advanced_error = ""
        self._site_property_node = -1
        self._site_property_generation = -1
        self._site_property_edit: SiteProperties | None = None
        self._site_property_error = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(
            poses=True,
            qvel=(self.show_transform and self._transform_velocity) or self.show_velocity,
        )

    def finish_frame(self, ctx: PanelContext) -> None:
        gizmo = ctx.gizmo
        if gizmo is not None and gizmo.model_placement_model_id >= 0:
            node = ctx.session.selected_node
            model_id = gizmo.model_placement_model_id
            invalid_placement = (
                not ctx.session.paused
                or node is None
                or node.type is not NodeType.MODEL
                or node.model_id != model_id
                or not gizmo.model_placement_active(ctx.session, model_id)
            )
            if invalid_placement and gizmo.cancel_model_placement(ctx.session).ok:
                self._model_transform_model = -1
        if self._edit_transaction and not imgui.is_any_item_active():
            ctx.submit(cmd.EndEditTransaction())
            self._edit_transaction = False

    def _submit_edit(self, ctx: PanelContext, command) -> None:
        if imgui.is_any_item_active() and not self._edit_transaction and not ctx.session.editing:
            result = ctx.submit(cmd.BeginEditTransaction("Inspector edit"))
            self._edit_transaction = result.ok
        ctx.submit(command)

    def draw(self, ctx: PanelContext) -> None:
        self._draw_model_source(ctx)
        self._draw_component_editor(ctx)
        s = ctx.session
        node = s.selected_node
        if node is None:
            self._transform_velocity = False
            imgui.text_disabled(ctx.tr("nothing selected"))
            imgui.text_disabled(ctx.tr("click an object in the viewport or the Hierarchy panel"))
            return

        color = ctx.theme.node_color(node.type)
        imgui.text_colored(imgui.ImVec4(*color), node.name or "?")
        imgui.same_line()
        imgui.text_disabled(f"({node.type})")

        self._identity(ctx, node)
        self._name_editor(ctx, node)
        if node.type is NodeType.MODEL:
            self._model(ctx, node)
            return
        if node.type is NodeType.LIGHT:
            self._light(ctx, node)
            return
        if node.type is NodeType.CAMERA:
            self._camera(ctx, node)
            return
        if node.type is NodeType.ENVIRONMENT:
            self._environment(ctx)
            return
        if node.type is NodeType.JOINT:
            self._joint(ctx, node)
            return
        self._transform(ctx, node)
        self._gizmo_reason(ctx, node)
        self._velocity(ctx, node)
        if node.type in (NodeType.LINK, NodeType.ROBOT):
            self._body_properties(ctx, node)
        if node.type is NodeType.SITE:
            self._site_properties(ctx, node)
        self._material(ctx, node)

    def _name_editor(self, ctx: PanelContext, node: SceneNode) -> None:
        model_element = node.model_id >= 0 and node.type not in (NodeType.WORLD, NodeType.MODEL)
        scene_entity = (
            node.model_id < 0
            and node.object_id > 0
            and node.type in (NodeType.LINK, NodeType.LIGHT, NodeType.CAMERA)
            and ctx.session.adapter.caps.scene_authoring
        )
        if not model_element and not scene_entity:
            return
        if self._model_name_node != node.node_id:
            self._model_name_node = node.node_id
            prefix = f"forge_{node.model_id}_"
            self._model_name = node.name.removeprefix(prefix)
        imgui.set_next_item_width(-80.0 * ctx.style_scale)
        entered, self._model_name = imgui.input_text(
            "##entity_name",
            self._model_name,
            imgui.InputTextFlags_.enter_returns_true.value,
        )
        imgui.same_line()
        apply = imgui.button(f"{ctx.tr('Apply')}##name")
        value = self._model_name.strip()
        if (entered or apply) and value:
            command = (
                cmd.RenameModelElement(node.node_id, value)
                if model_element
                else cmd.RenameSceneEntity(node.object_id, value)
            )
            ctx.submit(command)
        imgui.separator()

    def _model(self, ctx: PanelContext, node: SceneNode) -> None:
        info = next(
            (item for item in ctx.session.scene_models if item.model_id == node.model_id), None
        )
        if info is None:
            return
        imgui.text_disabled(str(info.path))
        gizmo = ctx.gizmo
        placement_active = gizmo is not None and gizmo.model_placement_active(
            ctx.session, info.model_id
        )
        if self._model_transform_model != info.model_id or (
            self._model_transform_generation != ctx.session.structure_generation
        ):
            self._sync_model_transform(ctx, node, info)
        if placement_active:
            transform = gizmo.model_placement_transform(ctx.session, info.model_id)
            if transform is not None:
                self._model_transform_position = np.asarray(transform[0], np.float32).copy()
                self._model_transform_euler = self._continuous_euler(node.node_id, transform[1])

        imgui.separator()
        imgui.text("Model placement")
        can_edit = ctx.session.paused and info.removable and gizmo is not None
        if placement_active:
            imgui.text_wrapped("Preview only; applying rebuilds the composed model once.")
        else:
            imgui.text_wrapped("Locked to avoid accidental model rebuilds.")
            if not can_edit:
                imgui.begin_disabled()
            begin_placement = imgui.button("Edit Placement")
            if not can_edit:
                imgui.end_disabled()
            if begin_placement and gizmo is not None:
                result = gizmo.begin_model_placement(ctx.session, info.model_id)
                if result.ok:
                    placement_active = True
                    ctx.report(result.message, level="info")
                else:
                    ctx.report(result.message, level="error")

        (pos_changed, position), (rot_changed, euler) = _vector_fields(
            ctx,
            node,
            "insp_model_transform",
            (
                ("position", self._model_transform_position, 0.01, "%.3f", None),
                ("rotation", self._model_transform_euler, 0.5, "%.1f°", None),
            ),
            editable=placement_active,
        )
        if placement_active and (pos_changed or rot_changed) and gizmo is not None:
            next_position = np.asarray(position, np.float32).copy()
            next_euler = np.asarray(euler, np.float64).copy()
            result = gizmo.preview_model_placement(
                ctx.session,
                info.model_id,
                next_position,
                math3d.euler_xyz_to_mat3(np.radians(next_euler)),
            )
            if result.ok:
                self._model_transform_position = next_position
                self._model_transform_euler = next_euler
            else:
                ctx.report(result.message, level="error")

        if placement_active and gizmo is not None:
            apply = imgui.button("Apply Placement")
            imgui.same_line()
            cancel = imgui.button("Cancel##model-placement")
            if apply:
                result = gizmo.apply_model_placement(ctx.session)
                if not result.ok:
                    ctx.report(result.message, level="error")
            elif cancel:
                result = gizmo.cancel_model_placement(ctx.session)
                if result.ok:
                    self._model_transform_model = -1
                    ctx.report(result.message, level="info")
                else:
                    ctx.report(result.message, level="error")
            return

        if info.removable and imgui.button("Remove Model"):
            ctx.submit(cmd.RemoveSceneModel(info.model_id))
        if ctx.session.adapter.caps.topology_editing:
            imgui.same_line()
            if not ctx.session.paused:
                imgui.begin_disabled()
            edit_source = imgui.button("Edit MJCF Source...")
            if not ctx.session.paused:
                imgui.end_disabled()
            if edit_source:
                source = ctx.session.adapter.scene_model_source(info.model_id)
                if source is not None:
                    self._source_model_id = info.model_id
                    self._source_text = source
                    self._source_error = ""
                    self._open_source_popup = True
            self._model_components(ctx, info.model_id)

    def _sync_model_transform(
        self, ctx: PanelContext, node: SceneNode, info: SceneModelInfo
    ) -> None:
        self._model_transform_model = info.model_id
        self._model_transform_generation = ctx.session.structure_generation
        self._model_transform_position = np.asarray(info.position, np.float32).copy()
        self._model_transform_euler = self._continuous_euler(
            node.node_id, np.asarray(info.rotation, np.float64).reshape(3, 3)
        )

    def _model_components(self, ctx: PanelContext, model_id: int) -> None:
        self._refresh_component_cache(ctx, model_id)
        imgui.separator()
        imgui.text_disabled("Model Components")
        editable = ctx.session.paused
        populated = tuple(
            (category, self._component_cache[category])
            for category in _MODEL_COMPONENT_CATEGORIES
            if self._component_cache[category]
        )
        if not populated:
            imgui.text_disabled("no authored components")
        for category, components in populated:
            label = f"{category.capitalize()} ({len(components)})"
            if not imgui.collapsing_header(label):
                continue
            for component in components:
                imgui.push_id(f"{category}-{component.component_id}")
                imgui.text(f"{component.name}  ({component.subtype})")
                imgui.same_line()
                if not editable:
                    imgui.begin_disabled()
                if imgui.small_button("Edit"):
                    self._begin_component_edit(component)
                imgui.same_line()
                if imgui.small_button("Delete"):
                    ctx.submit(cmd.RemoveModelComponent(model_id, category, component.component_id))
                if not editable:
                    imgui.end_disabled()
                imgui.pop_id()

        add_options = tuple(
            (category, subtype)
            for category in _MODEL_COMPONENT_CATEGORIES
            for subtype in self._component_presets[category]
        )
        if not editable or not add_options:
            imgui.begin_disabled()
        if imgui.begin_combo("Add Component...##model-component", "select type"):
            for category, subtype in add_options:
                selected, _ = imgui.selectable(f"{category.capitalize()} / {subtype}", False)
                if selected:
                    names = {component.name for component in self._component_cache[category]}
                    name = _unique_component_name(category, names)
                    ctx.submit(cmd.AddModelComponent(model_id, category, subtype, name))
            imgui.end_combo()
        if not editable or not add_options:
            imgui.end_disabled()
        if not editable:
            imgui.set_item_tooltip("Pause the simulation before editing model components")
        elif not add_options:
            imgui.set_item_tooltip("Add the referenced model elements first")

    def _refresh_component_cache(self, ctx: PanelContext, model_id: int) -> None:
        generation = ctx.session.structure_generation
        if (
            generation == self._component_cache_generation
            and model_id == self._component_cache_model
        ):
            return
        self._component_cache_generation = generation
        self._component_cache_model = model_id
        self._component_cache = {
            category: ctx.session.model_components(model_id, category)
            for category in _MODEL_COMPONENT_CATEGORIES
        }
        self._component_presets = {
            category: ctx.session.model_component_presets(model_id, category)
            for category in self._component_cache
        }

    def _begin_component_edit(self, component: ModelComponentInfo) -> None:
        self._component_edit = component
        self._component_name = component.name
        self._component_fields = [[field.name, field.value] for field in component.fields]
        self._component_path = [
            (item.type, [[field.name, field.value] for field in item.fields])
            for item in component.path
        ]
        self._component_path_choices = [
            {field.name: field.choices for field in item.fields} for item in component.path
        ]
        self._component_path_presets = component.path_presets
        self._component_error = ""
        self._open_component_popup = True

    def _draw_component_editor(self, ctx: PanelContext) -> None:
        component = self._component_edit
        if self._open_component_popup and component is not None:
            imgui.open_popup("Model Component")
            self._open_component_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(560.0 * ctx.style_scale, 520.0 * ctx.style_scale),
            imgui.Cond_.appearing.value,
        )
        visible, _ = imgui.begin_popup_modal("Model Component")
        if not visible:
            return
        if component is None:
            imgui.close_current_popup()
            imgui.end_popup()
            return
        imgui.text_disabled(f"{component.category} / {component.subtype}")
        choices = {field.name: field.choices for field in component.fields}
        if begin_kv_table("component_fields"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("name")
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._component_name = imgui.input_text(
                "##component-name", self._component_name
            )
            for index, field in enumerate(self._component_fields):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(field[0])
                imgui.table_next_column()
                imgui.set_next_item_width(-1.0)
                field[1] = _component_value_editor(
                    f"##component-field-{index}",
                    field[1],
                    choices.get(field[0], ()),
                    multiline=field[0] in _MULTILINE_COMPONENT_FIELDS or len(field[1]) > 72,
                )
            imgui.end_table()
        if self._component_path:
            imgui.separator()
            imgui.text_disabled("Path")
        for path_index, (element_type, fields) in enumerate(tuple(self._component_path)):
            imgui.push_id(f"path-{path_index}")
            object_type = next((value for name, value in fields if name == "objtype"), "")
            suffix = f" · {object_type}" if object_type else ""
            imgui.text_disabled(f"{path_index + 1}. {element_type}{suffix}")
            imgui.same_line()
            if path_index == 0:
                imgui.begin_disabled()
            move_up = imgui.small_button("Up")
            if path_index == 0:
                imgui.end_disabled()
            imgui.same_line()
            if path_index + 1 == len(self._component_path):
                imgui.begin_disabled()
            move_down = imgui.small_button("Down")
            if path_index + 1 == len(self._component_path):
                imgui.end_disabled()
            imgui.same_line()
            remove = imgui.small_button("Remove")
            if begin_kv_table(f"component_path_fields_{path_index}"):
                imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
                imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
                for field_index, field in enumerate(fields):
                    preset = _matching_path_preset(
                        self._component_path_presets, element_type, fields
                    )
                    preset_fields = (
                        {item.name: item for item in preset.fields} if preset is not None else {}
                    )
                    choices = self._component_path_choices[path_index].get(field[0], ())
                    if field[0] in preset_fields:
                        choices = preset_fields[field[0]].choices
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_disabled(field[0])
                    imgui.table_next_column()
                    imgui.set_next_item_width(-1.0)
                    previous = field[1]
                    field[1] = _component_value_editor(
                        f"##path-field-{field_index}",
                        field[1],
                        choices,
                        multiline=field[0] in _MULTILINE_COMPONENT_FIELDS or len(field[1]) > 72,
                    )
                    if field[0] == "objtype" and field[1] != previous:
                        next_preset = _matching_path_preset(
                            self._component_path_presets, element_type, fields
                        )
                        if next_preset is not None:
                            default_name = next(
                                (
                                    item.value
                                    for item in next_preset.fields
                                    if item.name == "objname"
                                ),
                                "",
                            )
                            for candidate in fields:
                                if candidate[0] == "objname":
                                    candidate[1] = default_name
                                    break
                imgui.end_table()
            imgui.pop_id()
            if move_up:
                self._component_path[path_index - 1 : path_index + 1] = reversed(
                    self._component_path[path_index - 1 : path_index + 1]
                )
                self._component_path_choices[path_index - 1 : path_index + 1] = reversed(
                    self._component_path_choices[path_index - 1 : path_index + 1]
                )
                break
            if move_down:
                self._component_path[path_index : path_index + 2] = reversed(
                    self._component_path[path_index : path_index + 2]
                )
                self._component_path_choices[path_index : path_index + 2] = reversed(
                    self._component_path_choices[path_index : path_index + 2]
                )
                break
            if remove:
                self._component_path.pop(path_index)
                self._component_path_choices.pop(path_index)
                break
        if self._component_path_presets and imgui.begin_combo("Add path item", "select type"):
            for preset in self._component_path_presets:
                selected, _ = imgui.selectable(_path_preset_label(preset), False)
                if selected:
                    self._component_path.append(
                        (
                            preset.type,
                            [[field.name, field.value] for field in preset.fields],
                        )
                    )
                    self._component_path_choices.append(
                        {field.name: field.choices for field in preset.fields}
                    )
            imgui.end_combo()
        if self._component_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.3, 1.0), self._component_error)
            if imgui.small_button("Copy error##component"):
                imgui.set_clipboard_text(self._component_error)
        if imgui.button("Apply", imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            result = ctx.submit(
                cmd.UpdateModelComponent(
                    component.model_id,
                    component.category,
                    component.component_id,
                    self._component_name,
                    tuple((name, value) for name, value in self._component_fields),
                    tuple(
                        (element_type, tuple((name, value) for name, value in fields))
                        for element_type, fields in self._component_path
                    ),
                )
            )
            if result.ok:
                self._component_edit = None
                imgui.close_current_popup()
            else:
                self._component_error = result.message
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            self._component_edit = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_model_source(self, ctx: PanelContext) -> None:
        if self._open_source_popup:
            imgui.open_popup("MJCF Source")
            self._open_source_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(820.0 * ctx.style_scale, 620.0 * ctx.style_scale),
            imgui.Cond_.appearing.value,
        )
        visible, _ = imgui.begin_popup_modal("MJCF Source")
        if not visible:
            return
        imgui.text_disabled("MjSpec validates and recompiles the model when changes are applied.")
        _changed, self._source_text = imgui.input_text_multiline(
            "##mjcf_source",
            self._source_text,
            imgui.ImVec2(-1.0, -70.0 * ctx.style_scale),
            imgui.InputTextFlags_.allow_tab_input.value,
        )
        if self._source_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.3, 1.0), self._source_error)
            if imgui.small_button("Copy error##source"):
                imgui.set_clipboard_text(self._source_error)
        if imgui.button("Apply", imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            result = ctx.submit(cmd.SetModelSource(self._source_model_id, self._source_text))
            if result.ok:
                imgui.close_current_popup()
            else:
                self._source_error = result.message
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            imgui.close_current_popup()
        imgui.end_popup()

    def _identity(self, ctx: PanelContext, node: SceneNode) -> None:
        if begin_kv_table("insp_id"):
            labeled(ctx.tr("node id"), str(node.node_id))
            labeled(
                ctx.tr("object id"),
                str(node.object_id) if node.object_id else f"— ({ctx.tr('not pickable')})",
            )
            labeled(ctx.tr("body"), str(node.body_index) if node.body_index >= 0 else "—")
            labeled(ctx.tr("posable"), ctx.tr("yes") if node.posable else ctx.tr("no"))
            imgui.end_table()

    def _transform(self, ctx: PanelContext, node: SceneNode) -> None:
        style = imgui.get_style()
        imgui.push_style_var(
            imgui.StyleVar_.item_spacing,
            imgui.ImVec2(style.item_spacing.x, 0.0),
        )
        self.show_transform = imgui.collapsing_header(
            ctx.tr("transform"), imgui.TreeNodeFlags_.default_open
        )
        imgui.pop_style_var()
        if not self.show_transform:
            return
        frame = ctx.session.frame
        pos, mat = _node_pose(frame, node)
        if pos is None:
            imgui.text_disabled(ctx.tr("no pose this frame"))
            return
        mat = np.eye(3, dtype=np.float32) if mat is None else np.asarray(mat).reshape(3, 3)
        euler = self._continuous_euler(node.node_id, mat)
        editable = _pose_editable(
            ctx.session.adapter.caps.write_pose, ctx.session.paused, node.posable
        )
        self._transform_velocity = _has_free_velocity(ctx.session.joints, node.body_index)
        velocity = _free_velocity(ctx.session.frame.qvel, ctx.session.joints, node.body_index)

        pos_changed = rot_changed = False
        new_pos = np.asarray(pos, np.float64).copy()
        new_euler = euler.copy()
        compact = _compact_transform(
            imgui.get_content_region_avail().x - 12.0 * ctx.style_scale,
            ctx.style_scale,
        )
        body_height = (96.0 if compact else 52.0) + (104.0 if self._transform_velocity else 0.0)

        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*ctx.theme.bg_popup))
        imgui.push_style_var(
            imgui.StyleVar_.window_padding,
            imgui.ImVec2(6.0 * ctx.style_scale, 4.0 * ctx.style_scale),
        )
        child_flags = imgui.ChildFlags_.always_use_window_padding
        window_flags = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
        child_visible = imgui.begin_child(
            f"##transform_body_{node.node_id}",
            imgui.ImVec2(0.0, body_height * ctx.style_scale),
            child_flags.value,
            window_flags.value,
        )
        if child_visible:
            imgui.push_font(None, 12.0 * ctx.style_scale)
            flags = (
                imgui.TableFlags_.sizing_stretch_same
                | imgui.TableFlags_.no_saved_settings
                | imgui.TableFlags_.no_pad_inner_x
                | imgui.TableFlags_.no_pad_outer_x
            )
            columns = 1 if compact else 4
            if imgui.begin_table("insp_transform", columns, flags):
                if compact:
                    imgui.table_setup_column("transform", imgui.TableColumnFlags_.width_stretch)
                else:
                    imgui.table_setup_column(
                        "value",
                        imgui.TableColumnFlags_.width_fixed,
                        60.0 * ctx.style_scale,
                    )
                    for axis in "xyz":
                        imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)

                pos_changed, new_pos = _vector_row(
                    ctx,
                    node,
                    ctx.tr("position"),
                    np.asarray(pos, np.float64),
                    editable=editable,
                    speed=0.01,
                    fmt="%.3f",
                    compact=compact,
                )
                rot_changed, new_euler = _vector_row(
                    ctx,
                    node,
                    ctx.tr("rotation"),
                    euler,
                    editable=editable,
                    speed=0.5,
                    fmt="%.1f",
                    compact=compact,
                )

                imgui.end_table()

            if velocity is not None and imgui.begin_table("insp_transform_velocity", 1, flags):
                imgui.table_setup_column("velocity", imgui.TableColumnFlags_.width_stretch)
                for name, values in (
                    (ctx.tr("linear velocity"), velocity[0]),
                    (ctx.tr("angular velocity"), velocity[1]),
                ):
                    _vector_row(
                        ctx,
                        node,
                        name,
                        values,
                        editable=False,
                        speed=0.0,
                        fmt="%.3f",
                        compact=True,
                    )
                imgui.end_table()
            imgui.pop_font()
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()

        if pos_changed or rot_changed:
            rotation = math3d.euler_xyz_to_mat3(np.radians(new_euler))
            if rot_changed:
                self._rotation_euler[:] = new_euler
                self._rotation_matrix[:] = rotation
            self._submit_edit(
                ctx,
                cmd.SetPose(
                    node.node_id,
                    np.asarray(new_pos, np.float32),
                    rotation,
                ),
            )

    def _continuous_euler(self, node_id: int, matrix) -> np.ndarray:
        matrix = np.asarray(matrix, np.float64).reshape(3, 3)
        same_node = node_id == self._rotation_node
        if same_node and np.allclose(matrix, self._rotation_matrix, atol=2e-5):
            return self._rotation_euler.copy()
        reference = self._rotation_euler if same_node else None
        self._rotation_node = node_id
        self._rotation_euler[:] = _nearest_euler_degrees(matrix, reference)
        self._rotation_matrix[:] = matrix
        return self._rotation_euler.copy()

    def _gizmo_reason(self, ctx: PanelContext, node: SceneNode) -> None:
        caps = ctx.session.adapter.caps
        availability = ctx.gizmo.evaluate(ctx.session, node) if ctx.gizmo is not None else None
        reason = (
            availability.reason
            if availability is not None
            else gizmo_refusal_reason(ctx.session.paused, node.posable)
        )
        if availability is None and not caps.write_pose:
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{caps.name} cannot edit this transform",
            )
            return
        active = availability.ok if availability is not None else reason is None
        if active:
            return
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), ctx.tr("gizmo hidden"))
        imgui.text_wrapped(reason)

    def _velocity(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_velocity = imgui.collapsing_header(ctx.tr("velocity"))
        if not self.show_velocity:
            return
        qvel = ctx.session.frame.qvel
        if qvel is None:
            imgui.text_disabled("waiting for the next frame (qvel is produced on demand)")
            return
        dofs = ctx.session.joints_for_body(node.body_index)
        if not dofs:
            imgui.text_disabled("no joint on this body")
            return
        if begin_kv_table("insp_vel"):
            for j in dofs:
                lo = j.qvel_adr
                hi = min(lo + max(1, j.dof), len(qvel))
                labeled(j.name or f"dof{lo}", "  ".join(f"{v:+.4f}" for v in qvel[lo:hi]))
            imgui.end_table()

    def _body_properties(self, ctx: PanelContext, node: SceneNode) -> None:
        current = ctx.session.body_properties(node.node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._body_property_node != node.node_id
            or self._body_property_generation != generation
            or self._body_property_edit is None
        ):
            self._body_property_node = node.node_id
            self._body_property_generation = generation
            self._body_property_edit = current
            self._body_inertial_euler = _nearest_euler_degrees(
                math3d.quat_to_mat3(current.inertial_quaternion), None
            )
            self._body_property_error = ""
        properties = self._body_property_edit
        if properties is None or not imgui.collapsing_header(ctx.tr("body inertial and dynamics")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        inertia_modes = ("auto from geoms", "diagonal", "full tensor")
        mode_values = ("auto", "diagonal", "full")
        mode = mode_values.index(properties.inertia_mode)
        inertia_mode = properties.inertia_mode
        edited = properties
        mode_changed = mass_changed = position_changed = rotation_changed = False
        diagonal_changed = full_diagonal_changed = full_cross_changed = False
        gravity_changed = mocap_changed = sleep_changed = False
        mass = float(edited.mass)
        inertial_position = np.asarray(edited.inertial_position, np.float64)
        inertial_euler = self._body_inertial_euler.copy()
        diagonal_inertia = np.asarray(edited.diagonal_inertia, np.float64)
        full_diagonal = np.asarray(edited.full_inertia[:3], np.float64)
        full_cross = np.asarray(edited.full_inertia[3:], np.float64)
        gravity_compensation = float(edited.gravity_compensation)
        mocap = bool(edited.mocap)
        parent = ctx.session.node(node.parent)
        root_body = parent is not None and parent.type in (NodeType.WORLD, NodeType.MODEL)
        movable_root = root_body and bool(ctx.session.joints_for_body(node.body_index))
        sleep_values = ("auto", "never", "allowed", "init")
        sleep_labels = ("automatic", "never", "allowed", "initially asleep")
        sleep_policy = (
            sleep_values.index(edited.sleep_policy) if edited.sleep_policy in sleep_values else 0
        )
        if _begin_property_table("body_inertial_properties"):
            _property_control_row(ctx, "inertia mode")
            mode_changed, mode = imgui.combo(
                "##body-inertia-mode", mode, tuple(ctx.tr(label) for label in inertia_modes)
            )
            inertia_mode = mode_values[mode]
            edited = replace(properties, inertia_mode=inertia_mode) if mode_changed else properties

            derived = inertia_mode == "auto"
            if derived:
                imgui.begin_disabled()
            _property_control_row(ctx, "mass")
            mass_changed, mass = imgui.drag_float(
                "##body-mass", float(edited.mass), 0.01, 0.000001, 1000000000.0, "%.6g kg"
            )
            position_changed, inertial_position = _property_vector_row(
                ctx,
                node,
                "inertial position",
                "body_inertial_position",
                edited.inertial_position,
                editable=not derived,
                speed=0.001,
                lo=-1000000.0,
                hi=1000000.0,
                fmt="%.5f m",
                reset_values=current.inertial_position,
            )
            rotation_disabled = derived or inertia_mode == "full"
            rotation_changed, inertial_euler = _property_vector_row(
                ctx,
                node,
                "inertial rotation",
                "body_inertial_rotation",
                self._body_inertial_euler,
                editable=not rotation_disabled,
                speed=0.25,
                lo=-360000.0,
                hi=360000.0,
                fmt="%.2f°",
                reset_values=_nearest_euler_degrees(
                    math3d.quat_to_mat3(current.inertial_quaternion), None
                ),
            )
            imgui.set_item_tooltip(
                ctx.tr(
                    "Body-frame rotation of diagonal principal axes; full tensors derive this rotation"
                )
            )
            if inertia_mode in ("auto", "diagonal"):
                diagonal_changed, diagonal_inertia = _property_vector_row(
                    ctx,
                    node,
                    "diagonal inertia",
                    "body_diagonal_inertia",
                    edited.diagonal_inertia,
                    editable=not derived,
                    speed=0.001,
                    lo=0.000000001,
                    hi=1000000000000.0,
                    fmt="%.6g",
                    reset_values=current.diagonal_inertia,
                )
            else:
                _property_control_row(ctx, "tensor xx / yy / zz")
                full_diagonal_changed, full_diagonal = imgui.drag_float3(
                    "##body-full-diagonal",
                    np.asarray(edited.full_inertia[:3], np.float32),
                    0.001,
                    -1000000000000.0,
                    1000000000000.0,
                    "%.6g",
                )
                _property_control_row(ctx, "tensor xy / xz / yz")
                full_cross_changed, full_cross = imgui.drag_float3(
                    "##body-full-cross",
                    np.asarray(edited.full_inertia[3:], np.float32),
                    0.001,
                    -1000000000000.0,
                    1000000000000.0,
                    "%.6g",
                )
            if derived:
                imgui.end_disabled()

            _property_control_row(ctx, "gravity compensation")
            gravity_changed, gravity_compensation = imgui.drag_float(
                "##body-gravity-compensation",
                float(edited.gravity_compensation),
                0.01,
                -1000000.0,
                1000000.0,
                "%.4f",
            )
            _property_control_row(ctx, "mocap body")
            mocap_changed, mocap = imgui.checkbox("##body-mocap", bool(edited.mocap))
            _property_control_row(ctx, "sleep policy")
            if not movable_root:
                imgui.begin_disabled()
            sleep_changed, sleep_policy = imgui.combo(
                "##body-sleep-policy",
                sleep_policy,
                tuple(ctx.tr(label) for label in sleep_labels),
            )
            if not movable_root:
                imgui.end_disabled()
                imgui.set_item_tooltip(
                    ctx.tr("MuJoCo sleep policies apply only to movable root bodies")
                )
            imgui.end_table()
        derived = inertia_mode == "auto"
        rotation_disabled = derived or inertia_mode == "full"
        if derived:
            imgui.text_disabled(ctx.tr("Mass and inertia are derived from attached geoms"))
        sleep_policy_value = sleep_values[sleep_policy]

        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model body properties"))

        if mass_changed:
            edited = replace(edited, mass=float(mass))
        if position_changed:
            edited = replace(
                edited,
                inertial_position=tuple(float(value) for value in inertial_position),
            )
        if rotation_changed and not rotation_disabled:
            self._body_inertial_euler = np.asarray(inertial_euler, np.float64)
            quaternion = math3d.mat3_to_quat(
                math3d.euler_xyz_to_mat3(np.radians(self._body_inertial_euler))
            )
            edited = replace(
                edited, inertial_quaternion=tuple(float(value) for value in quaternion)
            )
        if diagonal_changed:
            edited = replace(
                edited,
                diagonal_inertia=tuple(float(value) for value in diagonal_inertia),
            )
        if full_diagonal_changed or full_cross_changed:
            edited = replace(
                edited,
                full_inertia=tuple(float(value) for value in (*full_diagonal, *full_cross)),
            )
        if gravity_changed:
            edited = replace(edited, gravity_compensation=float(gravity_compensation))
        if mocap_changed:
            edited = replace(edited, mocap=bool(mocap))
        if sleep_changed and movable_root:
            edited = replace(edited, sleep_policy=sleep_policy_value)
        self._body_property_edit = edited

        dirty = edited != current
        if not editable or not dirty:
            imgui.begin_disabled()
        if imgui.button(f"{ctx.tr('Apply')}##body-properties"):
            result = ctx.submit(
                cmd.SetBodyProperties(
                    node_id=edited.node_id,
                    inertia_mode=edited.inertia_mode,
                    mass=edited.mass,
                    inertial_position=edited.inertial_position,
                    inertial_quaternion=edited.inertial_quaternion,
                    diagonal_inertia=edited.diagonal_inertia,
                    full_inertia=edited.full_inertia,
                    gravity_compensation=edited.gravity_compensation,
                    mocap=edited.mocap,
                    sleep_policy=edited.sleep_policy,
                )
            )
            if result.ok:
                self._body_property_generation = -1
                self._body_property_error = ""
            else:
                self._body_property_error = result.message
        if not editable or not dirty:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button(f"{ctx.tr('Revert')}##body-properties"):
            self._body_property_edit = current
            self._body_inertial_euler = _nearest_euler_degrees(
                math3d.quat_to_mat3(current.inertial_quaternion), None
            )
            self._body_property_error = ""
        imgui.same_line()
        imgui.text_disabled(ctx.tr("Apply rebuilds the model once"))
        if self._body_property_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._body_property_error)
            if imgui.small_button("Copy error##body-properties"):
                imgui.set_clipboard_text(self._body_property_error)

    def _joint(self, ctx: PanelContext, node: SceneNode) -> None:
        joint = next(
            (item for item in ctx.session.joints if item.joint_id == node.joint_index), None
        )
        if joint is None:
            imgui.text_disabled("joint metadata is unavailable")
            return
        if _begin_property_table("joint_identity"):
            for label, value in (
                ("type", joint.type),
                ("qpos address", str(joint.qpos_adr)),
                ("dof", str(joint.dof)),
            ):
                _property_control_row(ctx, label)
                imgui.align_text_to_frame_padding()
                imgui.text(value)
            imgui.end_table()
        if _property_section(ctx, "joint properties"):
            self._joint_properties(ctx, node, joint)
        self._joint_advanced_properties(ctx, joint)

    def _joint_properties(self, ctx: PanelContext, node: SceneNode, joint: JointInfo) -> None:
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and ctx.session.paused
            and joint.type != "free"
        )
        if not editable:
            imgui.begin_disabled()
        axis = np.asarray(joint.axis, np.float32)
        axis_changed = False
        limited = bool(joint.limited)
        limited_changed = False
        range_changed = False
        damping_changed = False
        damping = float(joint.damping)
        stiffness_changed = False
        stiffness = float(joint.stiffness)
        value_range = np.asarray(joint.range, np.float64).copy()
        range_valid = (
            value_range[1] > 0.0 if joint.type == "ball" else value_range[1] > value_range[0]
        )
        default_range = np.array(
            (0.0, np.pi)
            if joint.type == "ball"
            else ((-np.pi, np.pi) if joint.type == "hinge" else (-1.0, 1.0)),
            np.float64,
        )
        displayed_range = value_range.copy() if range_valid else default_range
        if _begin_property_table("joint_properties_table"):
            if joint.type in ("hinge", "slide"):
                axis_changed, axis = _property_vector_row(
                    ctx,
                    node,
                    "axis",
                    "joint_axis",
                    axis,
                    editable=editable,
                    speed=0.01,
                    lo=-1000000.0,
                    hi=1000000.0,
                    fmt="%.4f",
                    reset_values=joint.axis,
                    label_tooltip="Axis is expressed in the body frame",
                )

            if joint.type in ("hinge", "slide", "ball"):
                _property_control_row(ctx, "limited")
                limited_changed, limited = imgui.checkbox("##joint_limited", limited)
                if joint.type == "ball":
                    upper_deg = float(np.degrees(displayed_range[1]))
                    _property_control_row(ctx, "limit angle")
                    range_changed, upper_deg = imgui.drag_float(
                        "##joint_limit_angle", upper_deg, 0.5, 0.001, 360.0, "%.2f deg"
                    )
                    displayed_range[:] = (0.0, np.radians(upper_deg))
                elif joint.type == "hinge":
                    degrees = np.degrees(displayed_range)
                    _property_control_row(ctx, "range")
                    range_changed, degrees = imgui.drag_float2(
                        "##joint_range", degrees, 0.5, -36000.0, 36000.0, "%.2f deg"
                    )
                    displayed_range[:] = np.radians(degrees)
                else:
                    _property_control_row(ctx, "range")
                    range_changed, displayed_range = imgui.drag_float2(
                        "##joint_range",
                        displayed_range,
                        0.01,
                        -1000000.0,
                        1000000.0,
                        "%.4f m",
                    )
                    displayed_range = np.asarray(displayed_range, np.float64)
                if range_changed or (limited_changed and limited and not range_valid):
                    value_range = displayed_range

            _property_control_row(ctx, "damping")
            damping_changed, damping = imgui.drag_float(
                "##joint_damping", damping, 0.01, 0.0, 1000000.0, "%.4f"
            )
            _property_control_row(ctx, "stiffness")
            stiffness_changed, stiffness = imgui.drag_float(
                "##joint_stiffness", stiffness, 0.01, 0.0, 1000000.0, "%.4f"
            )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            reason = (
                "Free-joint properties stay defined by the free body"
                if joint.type == "free"
                else (
                    "Pause the simulation to edit model properties"
                    if not ctx.session.paused
                    else "This adapter cannot write model properties"
                )
            )
            imgui.text_disabled(reason)

        changed = any(
            (
                axis_changed,
                limited_changed,
                range_changed,
                damping_changed,
                stiffness_changed,
            )
        )
        invalid_axis = joint.type in ("hinge", "slide") and np.linalg.norm(axis) <= 1e-6
        invalid_range = bool(limited) and value_range[1] <= value_range[0]
        if invalid_axis:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "Axis must be non-zero")
        if invalid_range:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning), "Range upper bound must exceed lower bound"
            )
        if changed and editable and not invalid_axis and not invalid_range:
            self._submit_edit(
                ctx,
                cmd.SetJointProperties(
                    joint.joint_id,
                    np.asarray(axis, np.float64),
                    bool(limited),
                    (float(value_range[0]), float(value_range[1])),
                    float(damping),
                    float(stiffness),
                ),
            )

    def _joint_advanced_properties(self, ctx: PanelContext, joint: JointInfo) -> None:
        current = ctx.session.joint_advanced_properties(joint.joint_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._joint_advanced_id != joint.joint_id
            or self._joint_advanced_generation != generation
            or self._joint_advanced_edit is None
        ):
            self._joint_advanced_id = joint.joint_id
            self._joint_advanced_generation = generation
            self._joint_advanced_edit = current
            self._joint_advanced_error = ""
        properties = self._joint_advanced_edit
        if properties is None or not _property_section(ctx, "advanced joint properties"):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        rotational = joint.type in ("hinge", "ball")
        group_changed, group = False, int(properties.group)
        armature_changed, armature = False, float(properties.armature)
        friction_changed, friction_loss = False, float(properties.friction_loss)
        reference = (
            float(np.degrees(properties.reference)) if rotational else float(properties.reference)
        )
        spring_reference = (
            float(np.degrees(properties.spring_reference))
            if rotational
            else float(properties.spring_reference)
        )
        reference_changed = False
        spring_reference_changed = False
        margin_changed, margin = False, float(properties.margin)
        limit_reference_changed = False
        limit_reference = np.asarray(properties.limit_solver_reference, np.float32)
        limit_impedance_first_changed = False
        limit_impedance_first = np.asarray(properties.limit_solver_impedance[:3], np.float32)
        limit_impedance_shape_changed = False
        limit_impedance_shape = np.asarray(properties.limit_solver_impedance[3:], np.float32)
        friction_reference_changed = False
        friction_reference = np.asarray(properties.friction_solver_reference, np.float32)
        friction_impedance_first_changed = False
        friction_impedance_first = np.asarray(properties.friction_solver_impedance[:3], np.float32)
        friction_impedance_shape_changed = False
        friction_impedance_shape = np.asarray(properties.friction_solver_impedance[3:], np.float32)
        force_modes = ("auto", "unlimited", "limited")
        force_mode = force_modes.index(properties.actuator_force_limit_mode)
        force_mode_changed = False
        force_range_changed = False
        force_range = np.asarray(properties.actuator_force_range, np.float32)
        gravity_changed = False
        gravity_compensation = properties.actuator_gravity_compensation

        if _begin_property_table("joint_advanced_properties_table"):
            _property_control_row(ctx, "group")
            group_changed, group = imgui.combo(
                "##joint_advanced_group", group, tuple(str(value) for value in range(6))
            )
            _property_control_row(ctx, "armature")
            armature_changed, armature = imgui.drag_float(
                "##joint_armature", armature, 0.001, 0.0, 1000000000.0, "%.6g"
            )
            _property_control_row(ctx, "friction loss")
            friction_changed, friction_loss = imgui.drag_float(
                "##joint_friction_loss",
                friction_loss,
                0.001,
                0.0,
                1000000000.0,
                "%.6g",
            )
            _property_control_row(ctx, "reference")
            reference_changed, reference = imgui.drag_float(
                "##joint_reference",
                reference,
                0.25 if rotational else 0.001,
                -360000.0 if rotational else -1000000.0,
                360000.0 if rotational else 1000000.0,
                "%.3f deg" if rotational else "%.6g m",
            )
            _property_control_row(ctx, "spring reference")
            spring_reference_changed, spring_reference = imgui.drag_float(
                "##joint_spring_reference",
                spring_reference,
                0.25 if rotational else 0.001,
                -360000.0 if rotational else -1000000.0,
                360000.0 if rotational else 1000000.0,
                "%.3f deg" if rotational else "%.6g m",
            )
            _property_control_row(ctx, "limit margin")
            margin_changed, margin = imgui.drag_float(
                "##joint_limit_margin", margin, 0.001, 0.0, 1000000.0, "%.6g"
            )
            _property_control_row(ctx, "limit solver reference")
            limit_reference_changed, limit_reference = imgui.drag_float2(
                "##joint_limit_solver_reference",
                limit_reference,
                0.001,
                -1000000.0,
                1000000.0,
                "%.5g",
            )
            _property_control_row(ctx, "limit impedance min / max / width")
            limit_impedance_first_changed, limit_impedance_first = imgui.drag_float3(
                "##joint_limit_impedance_first",
                limit_impedance_first,
                0.001,
                0.0,
                1.0,
                "%.5g",
            )
            _property_control_row(ctx, "limit impedance midpoint / power")
            limit_impedance_shape_changed, limit_impedance_shape = imgui.drag_float2(
                "##joint_limit_impedance_shape",
                limit_impedance_shape,
                0.01,
                0.0,
                1000.0,
                "%.4g",
            )
            _property_control_row(ctx, "friction solver reference")
            friction_reference_changed, friction_reference = imgui.drag_float2(
                "##joint_friction_solver_reference",
                friction_reference,
                0.001,
                -1000000.0,
                1000000.0,
                "%.5g",
            )
            _property_control_row(ctx, "friction impedance min / max / width")
            friction_impedance_first_changed, friction_impedance_first = imgui.drag_float3(
                "##joint_friction_impedance_first",
                friction_impedance_first,
                0.001,
                0.0,
                1.0,
                "%.5g",
            )
            _property_control_row(ctx, "friction impedance midpoint / power")
            friction_impedance_shape_changed, friction_impedance_shape = imgui.drag_float2(
                "##joint_friction_impedance_shape",
                friction_impedance_shape,
                0.01,
                0.0,
                1000.0,
                "%.4g",
            )
            _property_control_row(ctx, "actuator force limit")
            force_mode_changed, force_mode = imgui.combo(
                "##joint_actuator_force_limit", force_mode, force_modes
            )
            _property_control_row(ctx, "actuator force range")
            force_range_changed, force_range = imgui.drag_float2(
                "##joint_actuator_force_range",
                force_range,
                0.01,
                -1000000000.0,
                1000000000.0,
                "%.6g N",
            )
            imgui.set_item_tooltip(
                ctx.tr(
                    "Auto enables the limit when a valid range is authored; "
                    "unlimited ignores the range"
                )
            )
            _property_control_row(ctx, "actuator gravity compensation")
            gravity_changed, gravity_compensation = imgui.checkbox(
                "##joint_actuator_gravity_compensation", gravity_compensation
            )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled("Pause the simulation to edit advanced joint properties")

        edited = properties
        if group_changed:
            edited = replace(edited, group=int(group))
        if armature_changed:
            edited = replace(edited, armature=float(armature))
        if friction_changed:
            edited = replace(edited, friction_loss=float(friction_loss))
        if reference_changed:
            edited = replace(
                edited,
                reference=float(np.radians(reference) if rotational else reference),
            )
        if spring_reference_changed:
            edited = replace(
                edited,
                spring_reference=float(
                    np.radians(spring_reference) if rotational else spring_reference
                ),
            )
        if margin_changed:
            edited = replace(edited, margin=float(margin))
        if limit_reference_changed:
            edited = replace(
                edited,
                limit_solver_reference=tuple(float(value) for value in limit_reference),
            )
        if limit_impedance_first_changed or limit_impedance_shape_changed:
            edited = replace(
                edited,
                limit_solver_impedance=tuple(
                    float(value) for value in (*limit_impedance_first, *limit_impedance_shape)
                ),
            )
        if friction_reference_changed:
            edited = replace(
                edited,
                friction_solver_reference=tuple(float(value) for value in friction_reference),
            )
        if friction_impedance_first_changed or friction_impedance_shape_changed:
            edited = replace(
                edited,
                friction_solver_impedance=tuple(
                    float(value) for value in (*friction_impedance_first, *friction_impedance_shape)
                ),
            )
        if force_mode_changed:
            edited = replace(edited, actuator_force_limit_mode=force_modes[force_mode])
        if force_range_changed:
            edited = replace(
                edited,
                actuator_force_range=tuple(float(value) for value in force_range),
            )
        if gravity_changed:
            edited = replace(edited, actuator_gravity_compensation=bool(gravity_compensation))
        self._joint_advanced_edit = edited

        dirty = edited != current
        invalid_force_range = (
            edited.actuator_force_limit_mode == "limited"
            and edited.actuator_force_range[1] <= edited.actuator_force_range[0]
        )
        if invalid_force_range:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                "Actuator force upper bound must exceed its lower bound",
            )
        apply_clicked = False
        revert_clicked = False
        if _begin_property_table("joint_advanced_actions"):
            _property_control_row(ctx, "changes")
            if not editable or not dirty or invalid_force_range:
                imgui.begin_disabled()
            apply_clicked = imgui.small_button(f"{ctx.tr('Apply')}##joint-advanced")
            if not editable or not dirty or invalid_force_range:
                imgui.end_disabled()
            imgui.same_line()
            revert_clicked = imgui.small_button(f"{ctx.tr('Revert')}##joint-advanced")
            imgui.same_line()
            imgui.text_disabled(ctx.tr("Apply rebuilds the model once"))
            imgui.end_table()
        if apply_clicked:
            result = ctx.submit(
                cmd.SetJointAdvancedProperties(
                    joint_id=edited.joint_id,
                    group=edited.group,
                    armature=edited.armature,
                    friction_loss=edited.friction_loss,
                    reference=edited.reference,
                    spring_reference=edited.spring_reference,
                    margin=edited.margin,
                    limit_solver_reference=edited.limit_solver_reference,
                    limit_solver_impedance=edited.limit_solver_impedance,
                    friction_solver_reference=edited.friction_solver_reference,
                    friction_solver_impedance=edited.friction_solver_impedance,
                    actuator_force_limit_mode=edited.actuator_force_limit_mode,
                    actuator_force_range=edited.actuator_force_range,
                    actuator_gravity_compensation=edited.actuator_gravity_compensation,
                )
            )
            if result.ok:
                self._joint_advanced_generation = -1
                self._joint_advanced_error = ""
            else:
                self._joint_advanced_error = result.message
        if revert_clicked:
            self._joint_advanced_edit = current
            self._joint_advanced_error = ""
        if self._joint_advanced_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._joint_advanced_error)
            if imgui.small_button("Copy error##joint-advanced"):
                imgui.set_clipboard_text(self._joint_advanced_error)

    def _site_properties(self, ctx: PanelContext, node: SceneNode) -> None:
        current = ctx.session.site_properties(node.node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._site_property_node != node.node_id
            or self._site_property_generation != generation
            or self._site_property_edit is None
        ):
            self._site_property_node = node.node_id
            self._site_property_generation = generation
            self._site_property_edit = current
            self._site_property_error = ""
        properties = self._site_property_edit
        if properties is None or not imgui.collapsing_header("site shape and endpoints"):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        site_types = ("sphere", "ellipsoid", "capsule", "cylinder", "box")
        site_type = site_types.index(properties.type)
        type_changed, site_type = imgui.combo("type", site_type, site_types)
        group_changed, group = imgui.combo(
            "visual group", int(properties.group), tuple(str(value) for value in range(6))
        )
        type_value = site_types[site_type]
        supports_endpoints = type_value in ("capsule", "cylinder")
        use_from_to = properties.use_from_to if supports_endpoints else False
        if not supports_endpoints:
            imgui.begin_disabled()
        endpoints_changed, use_from_to = imgui.checkbox("define with endpoints", use_from_to)
        if not supports_endpoints:
            imgui.end_disabled()
            imgui.set_item_tooltip("Endpoints apply only to capsule and cylinder sites")
        from_to = np.asarray(properties.from_to, np.float32)
        first_changed = second_changed = False
        if use_from_to:
            (first_changed, first), (second_changed, second) = _vector_fields(
                ctx,
                node,
                "site_endpoints",
                (
                    ("endpoint A (body frame)", from_to[:3], 0.001, "%.5f m", from_to[:3]),
                    ("endpoint B (body frame)", from_to[3:], 0.001, "%.5f m", from_to[3:]),
                ),
                editable=editable,
            )
            from_to = np.asarray((*first, *second), np.float32)
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled("Pause the simulation to edit site properties")

        edited = properties
        if type_changed:
            edited = replace(
                edited,
                type=type_value,
                use_from_to=bool(use_from_to),
            )
        if group_changed:
            edited = replace(edited, group=int(group))
        if endpoints_changed:
            edited = replace(edited, use_from_to=bool(use_from_to))
        if first_changed or second_changed:
            edited = replace(edited, from_to=tuple(float(value) for value in from_to))
        self._site_property_edit = edited

        invalid_endpoints = (
            edited.use_from_to
            and np.linalg.norm(np.asarray(edited.from_to[3:]) - np.asarray(edited.from_to[:3]))
            <= 1e-9
        )
        if invalid_endpoints:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "Site endpoints must be distinct")
        dirty = edited != current
        if not editable or not dirty or invalid_endpoints:
            imgui.begin_disabled()
        if imgui.button("Apply##site-properties"):
            result = ctx.submit(
                cmd.SetSiteProperties(
                    node_id=edited.node_id,
                    type=edited.type,
                    group=edited.group,
                    use_from_to=edited.use_from_to,
                    from_to=edited.from_to,
                )
            )
            if result.ok:
                self._site_property_generation = -1
                self._site_property_error = ""
            else:
                self._site_property_error = result.message
        if not editable or not dirty or invalid_endpoints:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Revert##site-properties"):
            self._site_property_edit = current
            self._site_property_error = ""
        imgui.same_line()
        imgui.text_disabled("Apply rebuilds the model once")
        if self._site_property_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._site_property_error)
            if imgui.small_button("Copy error##site-properties"):
                imgui.set_clipboard_text(self._site_property_error)

    def _material(self, ctx: PanelContext, node: SceneNode) -> None:
        if not imgui.collapsing_header(ctx.tr("material")):
            return
        src = ctx.session.source
        if src is None or node.body_index < 0 or len(src.geom_body) == 0:
            imgui.text_disabled("no geometry")
            return
        instances = (
            np.flatnonzero(np.asarray(src.geom_node) == node.node_id)
            if node.type in (NodeType.GEOM, NodeType.SITE)
            else np.flatnonzero(np.asarray(src.geom_body) == node.body_index)
        )
        if len(instances) == 0:
            imgui.text_disabled("no geometry on this body")
            return
        groups: dict[int, list[int]] = {}
        for instance in instances:
            node_id = int(src.geom_node[instance]) if instance < len(src.geom_node) else -1
            groups.setdefault(node_id, []).append(int(instance))
        imgui.text_disabled(f"{len(groups)} geometry component(s)")
        for node_id, group in list(groups.items())[:8]:
            self._geometry_material(ctx, node_id, group)

    def _geometry_material(self, ctx: PanelContext, node_id: int, instances: list[int]) -> None:
        src = ctx.session.source
        assert src is not None
        first = instances[0]
        material_index = src.geom_material[first] if first < len(src.geom_material) else -1
        if not 0 <= material_index < len(src.materials):
            imgui.text_disabled("material data is unavailable")
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
        size_editor = _geometry_dimensions(shape, src.geom_size[first])
        editable_size = (
            size_editor is not None
            and not infinite_plane
            and scene_node is not None
            and (
                ctx.session.adapter.caps.topology_editing
                or (scene_node.model_id < 0 and ctx.session.adapter.caps.scene_authoring)
            )
        )
        if infinite_plane:
            imgui.text_disabled("infinite plane")
        elif size_editor is not None and _property_section(ctx, "geometry dimensions"):
            if not editable_size:
                imgui.begin_disabled()
            dimension_label, dimensions = size_editor
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
            imgui.set_item_tooltip(hint)
            if size_changed and editable_size:
                self._submit_edit(
                    ctx,
                    cmd.SetGeometrySize(
                        node_id,
                        _geometry_size_from_dimensions(shape, src.geom_size[first], dimensions),
                    ),
                )

        self._geometry_contact_properties(ctx, node_id)
        self._geometry_advanced_properties(ctx, node_id)

        model_id = scene_node.model_id if scene_node is not None else -1
        model_assets = bool(model_id >= 0 and ctx.session.adapter.caps.model_assets)
        compatible_materials = ctx.session.model_material_indices(model_id) if model_assets else ()
        asset_editable = bool(
            not model_assets or not ctx.session.adapter.caps.simulation or ctx.session.paused
        )
        assigned = not model_assets or material_index in compatible_materials
        prefix = f"forge_{model_id}_" if model_assets else ""
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
                if imgui.begin_combo("##assigned_material", assignment_label):
                    selected, _ = imgui.selectable(ctx.tr("inline appearance"), not assigned)
                    if selected and assigned:
                        assignment_choice = -1
                    for candidate in compatible_materials:
                        candidate_material = src.materials[candidate]
                        selected, _ = imgui.selectable(
                            candidate_material.name or f"material {candidate}",
                            candidate == material_index,
                        )
                        if selected and candidate != material_index:
                            assignment_choice = candidate
                    imgui.end_combo()

                if assigned and ctx.panels is not None:
                    (open_assets,) = _property_button_row(ctx, "asset browser", ("Open in Assets",))
                existing_materials = {
                    src.materials[index].name.removeprefix(prefix) for index in compatible_materials
                }
                new_name = _unique_component_name("material", existing_materials)
                (create,) = _property_button_row(ctx, "create material", ("New material",))
                if not assigned:
                    imgui.begin_disabled()
                (duplicate,) = _property_button_row(
                    ctx, "material actions", ("Duplicate material",)
                )
                if not assigned:
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
            type_changed, type_index = imgui.combo("##geometry_type", type_index, types)
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
            if imgui.small_button("Copy error##geometry-shape"):
                imgui.set_clipboard_text(self._geometry_shape_error)

    def _geometry_contact_properties(self, ctx: PanelContext, node_id: int) -> None:
        properties = ctx.session.geometry_properties(node_id)
        if properties is None or not imgui.collapsing_header(ctx.tr("contact properties")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
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
                "##contact_dimension", dimension, dimension_labels
            )
            _property_control_row(ctx, "collision type mask")
            type_changed, type_mask = imgui.input_int(
                "##collision_type_mask", properties.collision_type_mask, 1, 16
            )
            imgui.set_item_tooltip("Decimal MuJoCo contype bitmask")
            _property_control_row(ctx, "collision affinity mask")
            affinity_changed, affinity_mask = imgui.input_int(
                "##collision_affinity_mask", properties.collision_affinity_mask, 1, 16
            )
            imgui.set_item_tooltip("Decimal MuJoCo conaffinity bitmask")
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
                "Positive values use time-constant/damping-ratio format; non-positive values use "
                "direct stiffness/damping format"
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
            imgui.text_disabled("Pause the simulation to edit model contact properties")

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
            imgui.set_item_tooltip("MuJoCo geom group used by visibility filters")
            _property_control_row(ctx, "mass source")
            mass_mode_changed, mass_mode = imgui.combo(
                "##geometry_mass_source", mass_mode, ("density", "explicit mass")
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
                "##geometry_inertia_distribution", inertia_mode, inertia_values
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
            imgui.text_disabled("Pause the simulation to edit model geometry properties")

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
        imgui.text_disabled(ctx.tr("Apply rebuilds the model once"))
        if self._geometry_advanced_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._geometry_advanced_error)
            if imgui.small_button("Copy error##geometry-advanced"):
                imgui.set_clipboard_text(self._geometry_advanced_error)

    def _light(self, ctx: PanelContext, node: SceneNode) -> None:
        source = ctx.session.source
        index = node.light_index
        if source is None or not 0 <= index < len(source.lights.lights):
            imgui.text_disabled("light data is unavailable")
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
                ["directional", "point", "spot", "area", "image"],
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
            imgui.text_disabled("environment data is unavailable")
            return
        environment = source.lights.environment()
        changed = False

        imgui.text_disabled("skybox")
        self._render_flag(ctx, RenderFlag.SKYBOX, "enabled##skybox")
        cube_textures = [
            name
            for name, item in source.textures.items()
            if item.type in (TextureType.CUBE, TextureType.SKYBOX)
        ]
        skyboxes = [None, *cube_textures]
        skybox_index = skyboxes.index(source.skybox) if source.skybox in skyboxes else 0
        skybox_changed, skybox_index = imgui.combo(
            "texture##skybox",
            skybox_index,
            [name or "none" for name in skyboxes],
        )
        if skybox_changed:
            self._submit_edit(ctx, cmd.SetSkybox(skyboxes[skybox_index]))

        imgui.separator()
        imgui.text_disabled("ambient light")
        ambient_changed, ambient = imgui.color_edit3("color##ambient", environment.ambient)
        changed |= ambient_changed

        imgui.separator()
        imgui.text_disabled("headlight")
        headlight_enabled = environment.headlight is not None
        enabled_changed, headlight_enabled = imgui.checkbox("enabled##headlight", headlight_enabled)
        changed |= enabled_changed
        headlight = environment.headlight or DEFAULT_HEADLIGHT
        intensity = float(np.max(headlight.diffuse))
        color = headlight.diffuse / intensity if intensity > 0.0 else np.ones(3, np.float32)
        color_changed, color = imgui.color_edit3("color##headlight", color)
        intensity_changed, intensity = imgui.drag_float(
            "intensity##headlight", intensity, 0.01, 0.0, 10.0, "%.2f"
        )
        specular_changed, specular = imgui.color_edit3("specular##headlight", headlight.specular)
        headlight_ambient_changed, headlight_ambient = imgui.color_edit3(
            "ambient##headlight", headlight.ambient
        )
        changed |= (
            color_changed or intensity_changed or specular_changed or headlight_ambient_changed
        )

        imgui.separator()
        imgui.text_disabled("fog")
        self._render_flag(ctx, RenderFlag.FOG, "enabled##fog")
        fog_color_changed, fog_color = imgui.color_edit3("color##fog", environment.fog_color)
        fog_start_changed, fog_start = imgui.drag_float(
            "start", environment.fog_start, 0.05, 0.0, 1e6, "%.2f m"
        )
        fog_end_changed, fog_end = imgui.drag_float(
            "end", environment.fog_end, 0.05, 0.0, 1e6, "%.2f m"
        )
        changed |= fog_color_changed or fog_start_changed or fog_end_changed

        imgui.separator()
        imgui.text_disabled("haze")
        self._render_flag(ctx, RenderFlag.HAZE, "enabled##haze")
        mode_changed, haze_mode = imgui.combo(
            "mode##haze",
            int(environment.horizon_haze),
            ["volumetric", "horizon"],
        )
        horizon_haze = bool(haze_mode)
        haze_color_changed, haze_color = imgui.color_edit3("color##haze", environment.haze_color)
        haze_label = "radius" if horizon_haze else "density"
        haze_format = "%.4f" if horizon_haze else "%.4f / m"
        haze_density_changed, haze_density = imgui.drag_float(
            haze_label, environment.haze_density, 0.001, 0.0, 100.0, haze_format
        )
        slices_changed = False
        haze_slices = environment.horizon_haze_slices
        if horizon_haze:
            slices_changed, haze_slices = imgui.drag_int("slices", haze_slices, 1.0, 3, 512, "%d")
        changed |= haze_color_changed or haze_density_changed or mode_changed or slices_changed

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
            imgui.text_disabled("camera data is unavailable")
            return
        info = ctx.session.cameras[index]
        view = ctx.session.camera_view(info.camera_id)
        if view is None:
            imgui.text_disabled("camera view is unavailable")
            return

        eye = np.asarray(view.eye, np.float64).copy()
        target = np.asarray(view.target, np.float64).copy()
        up = np.asarray(view.up, np.float64).copy()
        eye_changed = target_changed = up_changed = False
        fov_changed = near_changed = far_changed = ortho_changed = False
        fov = float(np.degrees(view.fov_y))
        near = float(view.near)
        far = float(view.far)
        orthographic = view.orthographic
        height_changed = False
        ortho_height = view.ortho_height

        if _property_section(ctx, "camera transform") and _begin_property_table(
            "insp_camera_transform"
        ):
            eye_changed, eye = _property_vector_row(
                ctx,
                node,
                "position",
                "camera_position",
                eye,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=0.0,
                fmt="%.4f",
            )
            target_changed, target = _property_vector_row(
                ctx,
                node,
                "target",
                "camera_target",
                target,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=0.0,
                fmt="%.4f",
            )
            up_changed, up = _property_vector_row(
                ctx,
                node,
                "up",
                "camera_up",
                up,
                editable=True,
                speed=0.01,
                lo=0.0,
                hi=0.0,
                fmt="%.4f",
                reset_values=np.array((0.0, 0.0, 1.0)),
            )
            imgui.end_table()

        if _property_section(ctx, "camera projection") and _begin_property_table(
            "insp_camera_projection"
        ):
            _property_control_row(ctx, "vertical fov")
            fov_changed, fov = imgui.drag_float("##camera_fov", fov, 0.1, 1.0, 179.0, "%.2f deg")
            _property_control_row(ctx, "near")
            near_changed, near = imgui.drag_float(
                "##camera_near", near, 0.001, 1e-5, float(view.far), "%.6f"
            )
            _property_control_row(ctx, "far")
            far_changed, far = imgui.drag_float("##camera_far", far, 0.1, float(near), 1e7, "%.3f")
            _property_control_row(ctx, "projection")
            projection = 1 if orthographic else 0
            supported = ctx.backend.caps.orthographic
            imgui.begin_disabled(not supported)
            selected_projection = segmented_control(
                f"camera-inspector-projection-{node.node_id}",
                (ctx.tr("persp"), ctx.tr("ortho")),
                projection,
                theme=ctx.theme,
            )
            imgui.end_disabled()
            if not supported:
                imgui.set_item_tooltip(f"{ctx.backend.caps.name} has no orthographic projection")
            elif selected_projection != projection:
                orthographic = selected_projection == 1
                ortho_changed = True
            if orthographic:
                _property_control_row(ctx, "ortho height")
                height_changed, ortho_height = imgui.drag_float(
                    "##camera_ortho_height",
                    float(view.ortho_height),
                    0.01,
                    1e-4,
                    1e6,
                    "%.4f",
                )
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
                        fov_y=float(np.radians(fov)),
                        near=float(near),
                        far=max(float(far), float(near) + 1e-5),
                        orthographic=orthographic,
                        ortho_height=float(ortho_height),
                    ),
                ),
            )

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


def _body_pose(xpos, xmat, body_index: int):
    if xpos is None or body_index < 0 or body_index >= len(xpos):
        return None, None
    mat = xmat[body_index] if xmat is not None and body_index < len(xmat) else None
    return xpos[body_index], mat


def _node_pose(frame, node: SceneNode):
    if node.type is NodeType.GEOM:
        return _body_pose(frame.geom_xpos, frame.geom_xmat, node.geom_index)
    if node.type is NodeType.SITE:
        return _body_pose(frame.site_xpos, frame.site_xmat, node.site_index)
    return _body_pose(frame.body_xpos, frame.body_xmat, node.body_index)


def _pose_editable(write_pose: bool, paused: bool, posable: bool) -> bool:
    return bool(write_pose and paused and posable)


def _nearest_euler_degrees(matrix, reference=None) -> np.ndarray:
    base = np.degrees(math3d.mat3_to_euler_xyz(matrix)).astype(np.float64)
    if reference is None:
        return base
    reference = np.asarray(reference, np.float64)
    alternate = np.array((base[0] + 180.0, 180.0 - base[1], base[2] + 180.0))
    candidates = []
    for value in (base, alternate):
        candidates.append(value + 360.0 * np.round((reference - value) / 360.0))
    return min(candidates, key=lambda value: float(np.linalg.norm(value - reference)))


def _free_velocity(qvel, joints, body_index: int):
    if qvel is None:
        return None
    joint = next(
        (j for j in joints if j.body == body_index and j.type == "free" and j.dof >= 6), None
    )
    if joint is None or joint.qvel_adr + 6 > len(qvel):
        return None
    values = np.asarray(qvel[joint.qvel_adr : joint.qvel_adr + 6], np.float64)
    return values[:3], values[3:]


def _has_free_velocity(joints, body_index: int) -> bool:
    return any(j.body == body_index and j.type == "free" and j.dof >= 6 for j in joints)


def _compact_transform(width: float, style_scale: float) -> bool:
    return float(width) < 210.0 * float(style_scale)


def _begin_property_table(table_id: str) -> bool:
    flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
    if not imgui.begin_table(table_id, 2, flags):
        return False
    imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.38)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch, 0.62)
    return True


def _property_section(ctx: PanelContext, label: str) -> bool:
    """Draw one translated, initially open Inspector property group."""

    return imgui.collapsing_header(
        ctx.tr(label),
        imgui.TreeNodeFlags_.default_open,
    )


def _property_control_row(
    ctx: PanelContext,
    label: str,
    *,
    tooltip: str = "",
) -> None:
    """Advance a property table to one left-label/right-control row."""

    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    translated = ctx.tr(label)
    available = imgui.get_content_region_avail().x
    shown = translated
    width = imgui.calc_text_size(shown).x
    truncated = width > available
    if truncated:
        ellipsis = "…"
        while shown and imgui.calc_text_size(f"{shown}{ellipsis}").x > available:
            shown = shown[:-1]
        shown = f"{shown.rstrip()}{ellipsis}" if shown else ellipsis
        width = imgui.calc_text_size(shown).x
    # Keep the label's right edge attached to the control boundary. On narrow
    # HiDPI panels a long label is ellipsized inside the label cell rather than
    # being clipped at the panel's left edge or drifting into the control.
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, available - width))
    imgui.text_disabled(shown)
    if (tooltip or truncated) and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(ctx.tr(tooltip) if tooltip else translated)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)


def _property_button_row(
    ctx: PanelContext,
    label: str,
    buttons: tuple[str, ...],
) -> tuple[bool, ...]:
    """Draw a wrapping row of compact property actions."""

    _property_control_row(ctx, label)
    translated = tuple(ctx.tr(button) for button in buttons)
    widths = tuple(button_width(button) for button in translated)
    layout = button_row_layout(
        widths,
        imgui.get_content_region_avail().x,
        imgui.get_style().item_spacing.x,
    )
    pressed: list[bool] = []
    for index, button in enumerate(translated):
        if layout[index]:
            imgui.same_line()
        pressed.append(imgui.small_button(f"{button}##{label}-{index}"))
    return tuple(pressed)


def _property_color_edit3(ctx: PanelContext, item_id: str, value):
    """Keep RGB editors usable when a scaled control column becomes narrow."""

    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 150.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit3(item_id, value, flags)


def _property_color_edit4(ctx: PanelContext, item_id: str, value):
    """Keep RGBA editors usable when a scaled control column becomes narrow."""

    flags = imgui.ColorEditFlags_.none
    if imgui.get_content_region_avail().x < 190.0 * ctx.style_scale:
        flags |= imgui.ColorEditFlags_.no_inputs
    return imgui.color_edit4(item_id, value, flags)


def _property_vector_row(
    ctx: PanelContext,
    node: SceneNode,
    label: str,
    name: str,
    values,
    *,
    editable: bool,
    speed: float,
    lo: float,
    hi: float,
    fmt: str,
    reset_values=None,
    label_tooltip: str = "",
) -> tuple[bool, np.ndarray]:
    """Draw one XYZ triplet as a property-table control with joined axis fields."""

    out = np.asarray(values, np.float64).copy()
    resets = (
        np.zeros(3, np.float64)
        if reset_values is None
        else np.asarray(reset_values, np.float64).reshape(3)
    )
    _property_control_row(ctx, label, tooltip=label_tooltip)
    compact = _compact_transform(imgui.get_content_region_avail().x, ctx.style_scale)
    flags = (
        imgui.TableFlags_.sizing_stretch_same
        | imgui.TableFlags_.no_saved_settings
        | imgui.TableFlags_.no_pad_inner_x
        | imgui.TableFlags_.no_pad_outer_x
    )
    columns = 1 if compact else 3
    if not imgui.begin_table(f"##{name}_property_axes_{node.node_id}", columns, flags):
        return False, out
    for axis in "xyz"[:columns]:
        imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)
    changed = False
    for axis, axis_label in enumerate("XYZ"):
        if compact:
            imgui.table_next_row()
        imgui.table_next_column()
        reset, edited, value = _axis_field(
            ctx,
            node,
            name,
            axis,
            axis_label,
            float(out[axis]),
            editable=editable,
            speed=speed,
            lo=lo,
            hi=hi,
            fmt=fmt,
            grouped=not compact,
        )
        if reset:
            out[axis] = resets[axis]
            changed = True
        if edited:
            out[axis] = value
            changed = True
    imgui.end_table()
    return changed, out


def _vector_fields(
    ctx: PanelContext,
    node: SceneNode,
    table_id: str,
    rows,
    *,
    editable: bool = True,
) -> tuple[tuple[bool, np.ndarray], ...]:
    compact = _compact_transform(imgui.get_content_region_avail().x, ctx.style_scale)
    flags = (
        imgui.TableFlags_.sizing_stretch_same
        | imgui.TableFlags_.no_saved_settings
        | imgui.TableFlags_.no_pad_inner_x
        | imgui.TableFlags_.no_pad_outer_x
    )
    columns = 1 if compact else 4
    if not imgui.begin_table(table_id, columns, flags):
        return tuple((False, np.asarray(row[1], np.float64).copy()) for row in rows)
    if compact:
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
    else:
        imgui.table_setup_column(
            "value", imgui.TableColumnFlags_.width_fixed, 96.0 * ctx.style_scale
        )
        for axis in "xyz":
            imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)
    result = tuple(
        _vector_row(
            ctx,
            node,
            name,
            values,
            editable=editable,
            speed=speed,
            fmt=fmt,
            compact=compact,
            reset_values=reset_values,
        )
        for name, values, speed, fmt, reset_values in rows
    )
    imgui.end_table()
    return result


def _vector_row(
    ctx: PanelContext,
    node: SceneNode,
    name: str,
    values,
    *,
    editable: bool,
    speed: float,
    fmt: str,
    compact: bool,
    reset_values=None,
) -> tuple[bool, np.ndarray]:
    out = np.asarray(values, np.float64).copy()
    resets = (
        np.zeros(3, np.float64)
        if reset_values is None
        else np.asarray(reset_values, np.float64).reshape(3)
    )
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ctx.theme.text_disabled))
    imgui.text(name)
    imgui.pop_style_color()
    group_hovered = imgui.is_item_hovered()
    if group_hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
        imgui.set_clipboard_text(_format_vector(out))
    if group_hovered:
        imgui.set_tooltip(ctx.tr("Right-click to copy XYZ"))

    if compact:
        row_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        if not imgui.begin_table(f"##{name}_axes_{node.node_id}", 3, row_flags):
            return False, out
        for axis in "xyz":
            imgui.table_setup_column(axis, imgui.TableColumnFlags_.width_stretch, 1.0)

    changed = False
    for axis, label in enumerate("XYZ"):
        imgui.table_next_column()
        reset, edited, value = _axis_field(
            ctx,
            node,
            name,
            axis,
            label,
            float(out[axis]),
            editable=editable,
            speed=speed,
            fmt=fmt,
        )
        if reset:
            out[axis] = resets[axis]
            changed = True
        if edited:
            out[axis] = value
            changed = True
    if compact:
        imgui.end_table()
    return changed, out


def _axis_field(
    ctx: PanelContext,
    node: SceneNode,
    name: str,
    axis: int,
    label: str,
    value: float,
    *,
    editable: bool,
    speed: float,
    fmt: str,
    lo: float = 0.0,
    hi: float = 0.0,
    grouped: bool = True,
) -> tuple[bool, bool, float]:
    axis_color = ctx.theme.axis_color(axis)
    color = _mix_color(ctx.theme.bg_frame, axis_color, 0.56)
    hovered_color = _mix_color(ctx.theme.bg_frame_hovered, axis_color, 0.72)
    active_color = _mix_color(ctx.theme.bg_frame_active, axis_color, 0.88)
    axis_width = min(
        22.0 * ctx.style_scale,
        max(1.0, imgui.get_content_region_avail().x * 0.4),
    )

    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*color))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hovered_color))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*active_color))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    if not editable:
        imgui.push_style_var(imgui.StyleVar_.disabled_alpha, 1.0)
    imgui.begin_disabled(not editable)
    reset = imgui.button(
        f"{label}##{name}_{axis}_{node.node_id}",
        imgui.ImVec2(axis_width, 0.0),
    )
    imgui.end_disabled()
    if not editable:
        imgui.pop_style_var()
    button_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    button_active = imgui.is_item_active()
    button_lo, button_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    imgui.pop_style_color(4)
    if button_hovered:
        imgui.set_tooltip(ctx.tr("Click to reset to 0") if editable else ctx.tr("Read only"))

    group_gap = 5.0 * ctx.style_scale if grouped and axis < 2 else 0.0
    imgui.same_line(0.0, 0.0)
    imgui.set_next_item_width(max(1.0, imgui.get_content_region_avail().x - group_gap))
    imgui.begin_disabled(not editable)
    edited, next_value = imgui.drag_float(
        f"##{name}_{axis}_{node.node_id}", value, speed, lo, hi, fmt
    )
    imgui.end_disabled()
    field_hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
    field_active = imgui.is_item_active()
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    _fill_axis_field_seam(
        ctx,
        button_lo,
        button_hi,
        field_lo,
        field_hi,
        badge_color=(
            active_color
            if button_active and editable
            else hovered_color
            if button_hovered and editable
            else color
        ),
        field_color=(
            ctx.theme.bg_frame_active
            if field_active
            else ctx.theme.bg_frame_hovered
            if field_hovered and editable
            else ctx.theme.bg_frame
        ),
        editable=editable,
    )
    if field_hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
        imgui.set_clipboard_text(fmt % value)
    if field_hovered:
        hint = (
            ctx.tr("Drag to edit · right-click to copy")
            if editable
            else ctx.tr("Read only · right-click to copy")
        )
        imgui.set_tooltip(hint)

    return reset, edited, next_value


def _fill_axis_field_seam(
    ctx: PanelContext,
    button_lo,
    button_hi,
    field_lo,
    field_hi,
    *,
    badge_color,
    field_color,
    editable: bool,
) -> None:
    """Square the joined edges while preserving only the pair's outer corners."""

    rounding = min(
        float(imgui.get_style().frame_rounding),
        max(0.0, (float(button_hi.y) - float(button_lo.y)) * 0.5),
    )
    if rounding <= 0.0:
        return
    style = imgui.get_style()

    def packed(value, opacity: float) -> int:
        background = ctx.theme.bg_popup
        mixed = tuple(
            background[index] + (value[index] - background[index]) * opacity for index in range(3)
        )
        return imgui.color_convert_float4_to_u32(imgui.ImVec4(*mixed, 1.0))

    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        imgui.ImVec2(button_hi.x - rounding, button_lo.y),
        button_hi,
        packed(badge_color, float(style.alpha)),
    )
    field_opacity = float(style.alpha) * (1.0 if editable else float(style.disabled_alpha))
    for y0, y1 in (
        (field_lo.y, field_lo.y + rounding),
        (field_hi.y - rounding, field_hi.y),
    ):
        draw_list.add_rect_filled(
            imgui.ImVec2(field_lo.x, y0),
            imgui.ImVec2(field_lo.x + rounding, y1),
            packed(field_color, field_opacity),
        )


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)


def _mix_color(background, foreground, amount: float):
    weight = min(1.0, max(0.0, float(amount)))
    return (
        *(
            background[index] + (foreground[index] - background[index]) * weight
            for index in range(3)
        ),
        foreground[3],
    )
