"""Selected entity properties and transform editing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ... import math3d
from ...adapters.base import FrameNeeds, ModelComponentInfo, NodeType, SceneNode
from ...render.backend import RenderFlag
from ...types import DEFAULT_HEADLIGHT, Environment, LightType, MeshShape, TextureType
from . import Panel, PanelContext, begin_kv_table, labeled

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


def _component_value_editor(label: str, value: str, choices: tuple[str, ...]) -> str:
    if choices:
        if imgui.begin_combo(label, value or "select"):
            for choice in choices:
                selected, _ = imgui.selectable(choice, choice == value)
                if selected:
                    value = choice
            imgui.end_combo()
        return value
    _changed, value = imgui.input_text(label, value)
    return value


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
        self._component_error = ""
        self._open_component_popup = False
        self._model_transform_model = -1
        self._model_transform_generation = -1
        self._model_transform_position = np.zeros(3, np.float32)
        self._model_transform_euler = np.zeros(3, np.float64)
        self._model_transform_dirty = False

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(
            poses=True,
            qvel=(self.show_transform and self._transform_velocity) or self.show_velocity,
        )

    def finish_frame(self, ctx: PanelContext) -> None:
        if self._edit_transaction and not imgui.is_any_item_active():
            ctx.submit(cmd.EndEditTransaction())
            self._edit_transaction = False
        if self._model_transform_dirty and not imgui.is_any_item_active():
            result = ctx.submit(
                cmd.SetSceneModelTransform(
                    self._model_transform_model,
                    self._model_transform_position.copy(),
                    math3d.euler_xyz_to_mat3(np.radians(self._model_transform_euler)),
                )
            )
            self._model_transform_dirty = False
            if not result.ok:
                self._model_transform_model = -1

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
            imgui.text_disabled("nothing selected")
            imgui.text_disabled("click an object in the viewport or the Hierarchy panel")
            return

        color = ctx.theme.node_color(node.type)
        imgui.text_colored(imgui.ImVec4(*color), node.name or "?")
        imgui.same_line()
        imgui.text_disabled(f"({node.type})")

        self._identity(node)
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
        self._transform(ctx, node)
        self._gizmo_reason(ctx, node)
        self._velocity(ctx, node)
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
        apply = imgui.button("Apply##name")
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
        if self._model_transform_model != info.model_id or (
            self._model_transform_generation != ctx.session.structure_generation
            and not self._model_transform_dirty
        ):
            self._model_transform_model = info.model_id
            self._model_transform_generation = ctx.session.structure_generation
            self._model_transform_position = np.asarray(info.position, np.float32).copy()
            self._model_transform_euler = self._continuous_euler(
                node.node_id, np.asarray(info.rotation, np.float64).reshape(3, 3)
            )
        editable = ctx.session.paused and info.removable
        (pos_changed, position), (rot_changed, euler) = _vector_fields(
            ctx,
            node,
            "insp_model_transform",
            (
                ("position", self._model_transform_position, 0.01, "%.3f", None),
                ("rotation", self._model_transform_euler, 0.5, "%.1f°", None),
            ),
            editable=editable,
        )
        if editable and (pos_changed or rot_changed):
            self._model_transform_position = np.asarray(position, np.float32).copy()
            self._model_transform_euler = np.asarray(euler, np.float64).copy()
            self._model_transform_dirty = True
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

    def _model_components(self, ctx: PanelContext, model_id: int) -> None:
        self._refresh_component_cache(ctx, model_id)
        imgui.separator()
        imgui.text_disabled("Model Components")
        editable = ctx.session.paused
        for category in ("actuator", "sensor", "tendon", "equality"):
            components = self._component_cache[category]
            label = f"{category.capitalize()} ({len(components)})"
            if not imgui.collapsing_header(label):
                continue
            if not components:
                imgui.text_disabled(f"no {category} components")
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
            presets = self._component_presets[category]
            if not editable or not presets:
                imgui.begin_disabled()
            if imgui.begin_combo(f"Add {category}...##add-{category}", "select type"):
                names = {component.name for component in components}
                for subtype in presets:
                    selected, _ = imgui.selectable(subtype, False)
                    if selected:
                        name = _unique_component_name(category, names)
                        ctx.submit(cmd.AddModelComponent(model_id, category, subtype, name))
                imgui.end_combo()
            if not editable or not presets:
                imgui.end_disabled()
            if not presets:
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
            for category in ("actuator", "sensor", "tendon", "equality")
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
        _changed, self._component_name = imgui.input_text("name", self._component_name)
        choices = {field.name: field.choices for field in component.fields}
        for index, field in enumerate(self._component_fields):
            field[1] = _component_value_editor(
                f"{field[0]}##component-field-{index}", field[1], choices.get(field[0], ())
            )
        if self._component_path:
            imgui.separator()
            imgui.text_disabled("Path")
        path_choices = [
            {field.name: field.choices for field in item.fields} for item in component.path
        ]
        for path_index, (element_type, fields) in enumerate(self._component_path):
            imgui.push_id(f"path-{path_index}")
            imgui.text_disabled(f"{path_index + 1}. {element_type}")
            for field_index, field in enumerate(fields):
                field[1] = _component_value_editor(
                    f"{field[0]}##path-field-{field_index}",
                    field[1],
                    path_choices[path_index].get(field[0], ()),
                )
            imgui.pop_id()
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

    def _identity(self, node: SceneNode) -> None:
        if begin_kv_table("insp_id"):
            labeled("node id", str(node.node_id))
            labeled("object id", str(node.object_id) if node.object_id else "— (not pickable)")
            labeled("body", str(node.body_index) if node.body_index >= 0 else "—")
            labeled("posable", "yes" if node.posable else "no")
            imgui.end_table()

    def _transform(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_transform = imgui.collapsing_header(
            "transform", imgui.TreeNodeFlags_.default_open
        )
        if not self.show_transform:
            return
        frame = ctx.session.frame
        pos, mat = _node_pose(frame, node)
        if pos is None:
            imgui.text_disabled("no pose this frame")
            return
        mat = np.eye(3, dtype=np.float32) if mat is None else np.asarray(mat).reshape(3, 3)
        euler = self._continuous_euler(node.node_id, mat)
        editable = _pose_editable(
            ctx.session.adapter.caps.write_pose, ctx.session.paused, node.posable
        )

        compact = _compact_transform(imgui.get_content_region_avail().x, ctx.style_scale)
        flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        columns = 1 if compact else 7
        if not imgui.begin_table("insp_transform", columns, flags):
            return
        if compact:
            imgui.table_setup_column("transform", imgui.TableColumnFlags_.width_stretch)
        else:
            imgui.table_setup_column(
                "value", imgui.TableColumnFlags_.width_fixed, 96.0 * ctx.style_scale
            )
            for axis in "xyz":
                imgui.table_setup_column(
                    axis, imgui.TableColumnFlags_.width_fixed, 20.0 * ctx.style_scale
                )
                imgui.table_setup_column(
                    f"{axis} value", imgui.TableColumnFlags_.width_stretch, 1.0
                )

        pos_changed, new_pos = _vector_row(
            ctx,
            node,
            "position",
            np.asarray(pos, np.float64),
            editable=editable,
            speed=0.01,
            fmt="%.3f",
            compact=compact,
        )
        rot_changed, new_euler = _vector_row(
            ctx,
            node,
            "rotation",
            euler,
            editable=editable,
            speed=0.5,
            fmt="%.1f",
            compact=compact,
        )

        self._transform_velocity = _has_free_velocity(ctx.session.joints, node.body_index)
        velocity = _free_velocity(ctx.session.frame.qvel, ctx.session.joints, node.body_index)
        if velocity is not None:
            _vector_row(
                ctx,
                node,
                "linear velocity",
                velocity[0],
                editable=False,
                speed=0.0,
                fmt="%.3f",
                compact=compact,
            )
            _vector_row(
                ctx,
                node,
                "angular velocity",
                velocity[1],
                editable=False,
                speed=0.0,
                fmt="%.3f",
                compact=compact,
            )
        imgui.end_table()

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
        imgui.separator()
        if availability is None and not caps.write_pose:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{caps.name} cannot edit this transform",
            )
            return
        active = availability.ok if availability is not None else reason is None
        if active:
            direct_joints = ctx.session.joints_for_body(node.body_index)
            joint = None
            if not node.posable and direct_joints:
                selected = ctx.gizmo.selected_joint_id(node.body_index)
                joint = next(
                    (item for item in direct_joints if item.joint_id == selected),
                    direct_joints[0] if len(direct_joints) == 1 else None,
                )
            detail = (
                "hinge / revolute joint; rotate about its axis"
                if joint is not None and joint.type == "hinge"
                else (
                    f"{joint.type} joint; joint-axis control"
                    if joint is not None
                    else f"{ctx.gizmo.space if ctx.gizmo is not None else 'body'} frame; "
                    "g/r mode, t frame"
                )
            )
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.primary),
                f"gizmo: active ({detail})",
            )
            return
        imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "gizmo hidden")
        imgui.text_wrapped(reason)

    def _velocity(self, ctx: PanelContext, node: SceneNode) -> None:
        self.show_velocity = imgui.collapsing_header("velocity")
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

    def _material(self, ctx: PanelContext, node: SceneNode) -> None:
        if not imgui.collapsing_header("material"):
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
        elif size_editor is not None:
            if not editable_size:
                imgui.begin_disabled()
            dimension_label, dimensions = size_editor
            if len(dimensions) == 1:
                size_changed, scalar = imgui.drag_float(
                    dimension_label,
                    float(dimensions[0]),
                    0.05,
                    0.002,
                    1000000.0,
                    "%.3f",
                )
                dimensions = np.array((scalar,), np.float32)
            elif len(dimensions) == 2:
                size_changed, dimensions = imgui.drag_float2(
                    dimension_label, dimensions, 0.05, 0.002, 1000000.0, "%.3f"
                )
            else:
                size_changed, dimensions = imgui.drag_float3(
                    dimension_label, dimensions, 0.05, 0.002, 1000000.0, "%.3f"
                )
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

        color_changed, rgba = imgui.color_edit4("instance color", src.geom_rgba[first])
        if color_changed and node_id >= 0:
            self._submit_edit(ctx, cmd.SetGeometryColor(node_id, np.asarray(rgba, np.float32)))

        imgui.text_disabled(f"shared material: {material.name or material_index}")
        rgba_changed, material_rgba = imgui.color_edit4("base color", material.rgba)
        emission = material.emission
        specular = material.specular
        shininess = material.shininess
        reflectance = material.reflectance
        preset_changed = False
        if imgui.begin_combo("preset", "Presets..."):
            for preset, values in _MATERIAL_PRESETS.items():
                selected, _ = imgui.selectable(preset, False)
                if selected:
                    emission, specular, shininess, reflectance = values
                    preset_changed = True
            imgui.end_combo()
        emission_changed, emission = imgui.drag_float("emission", emission, 0.01, 0.0, 10.0, "%.2f")
        specular_changed, specular = imgui.drag_float("specular", specular, 0.01, 0.0, 1.0, "%.2f")
        shininess_changed, shininess = imgui.drag_float(
            "shininess", shininess, 0.01, 0.0, 1.0, "%.2f"
        )
        reflectance_changed, reflectance = imgui.drag_float(
            "reflectance", reflectance, 0.01, 0.0, 1.0, "%.2f"
        )
        texture = material.texture
        texture_changed = False
        if imgui.begin_combo("texture", texture or "none"):
            for candidate in (None, *src.textures):
                selected, _ = imgui.selectable(candidate or "none", candidate == texture)
                if selected:
                    texture = candidate
                    texture_changed = True
            imgui.end_combo()
        repeat_changed, tex_repeat = imgui.drag_float2(
            "texture repeat", material.tex_repeat, 0.05, 0.01, 1000.0, "%.2f"
        )
        uniform_changed, tex_uniform = imgui.checkbox("uniform texture scale", material.tex_uniform)
        if any(
            (
                emission_changed,
                specular_changed,
                shininess_changed,
                reflectance_changed,
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
                        texture=texture,
                        tex_repeat=np.asarray(tex_repeat, np.float32),
                        tex_uniform=bool(tex_uniform),
                    ),
                ),
            )
        imgui.pop_id()

    def _light(self, ctx: PanelContext, node: SceneNode) -> None:
        source = ctx.session.source
        index = node.light_index
        if source is None or not 0 <= index < len(source.lights.lights):
            imgui.text_disabled("light data is unavailable")
            return
        light = source.lights.lights[index]
        self._entity_gizmo_lock(ctx, node)

        changed, active = imgui.checkbox("enabled", light.active)
        kind_changed, kind_index = imgui.combo(
            "type", int(light.type), ["directional", "point", "spot", "area", "image"]
        )
        changed |= kind_changed
        light_type = LightType(kind_index)

        diffuse = light.diffuse
        specular = light.specular
        ambient = light.ambient
        position = light.position
        direction = light.direction
        image_intensity = light.intensity
        texture = light.texture

        if light_type is LightType.IMAGE:
            intensity_changed, image_intensity = imgui.drag_float(
                "intensity", light.intensity, 50.0, 0.0, 100000.0, "%.0f"
            )
            textures = [
                name
                for name, item in source.textures.items()
                if item.type in (TextureType.CUBE, TextureType.SKYBOX)
            ]
            texture_index = textures.index(light.texture) if light.texture in textures else 0
            texture_changed = False
            if textures:
                texture_changed, texture_index = imgui.combo("texture", texture_index, textures)
                texture = textures[texture_index]
            else:
                imgui.text_disabled("add a cube texture to illuminate the scene")
            changed |= intensity_changed or texture_changed
        else:
            intensity = float(np.max(light.diffuse))
            color = light.diffuse / intensity if intensity > 0.0 else np.ones(3, np.float32)
            color_changed, color = imgui.color_edit3("color", color)
            intensity_changed, intensity = imgui.drag_float(
                "intensity", intensity, 0.01, 0.0, 10.0, "%.2f"
            )
            changed |= color_changed or intensity_changed
            diffuse = np.asarray(color, np.float32) * intensity

            specular_changed, specular = imgui.color_edit3("specular", light.specular)
            ambient_changed, ambient = imgui.color_edit3("ambient", light.ambient)
            changed |= specular_changed or ambient_changed

        (pos_changed, position), (dir_changed, direction) = _vector_fields(
            ctx,
            node,
            "insp_light_pose",
            (
                ("position (local)", position, 0.01, "%.3f", None),
                (
                    "direction (local)",
                    direction,
                    0.01,
                    "%.3f",
                    np.array((0.0, 0.0, -1.0)),
                ),
            ),
        )
        changed |= pos_changed or dir_changed

        range_changed = cutoff_changed = exponent_changed = attenuation_changed = False
        area_changed = False
        light_range, cutoff, exponent = light.range, light.cutoff, light.exponent
        area_radius = light.area_radius
        attenuation = light.attenuation
        if light_type in (LightType.POINT, LightType.SPOT, LightType.AREA):
            range_changed, light_range = imgui.drag_float(
                "range (0 = unlimited)", light.range, 0.05, 0.0, 10000.0, "%.2f"
            )
            ((attenuation_changed, attenuation),) = _vector_fields(
                ctx,
                node,
                "insp_light_attenuation",
                (
                    (
                        "attenuation",
                        light.attenuation,
                        0.01,
                        "%.3f",
                        np.array((1.0, 0.0, 0.0)),
                    ),
                ),
            )
        if light_type is LightType.SPOT:
            cutoff_changed, cutoff = imgui.drag_float(
                "cutoff", light.cutoff, 0.25, 0.1, 89.9, "%.1f deg"
            )
            exponent_changed, exponent = imgui.drag_float(
                "falloff exponent", light.exponent, 0.1, 0.0, 100.0, "%.1f"
            )
        if light_type is LightType.AREA:
            area_changed, area_radius = imgui.drag_float(
                "source radius", light.area_radius, 0.01, 0.0, 1000.0, "%.3f"
            )
        shadow_changed = False
        cast_shadow = light.cast_shadow
        if light_type is not LightType.IMAGE:
            shadow_changed, cast_shadow = imgui.checkbox("cast shadow", light.cast_shadow)
        changed |= (
            range_changed
            or cutoff_changed
            or exponent_changed
            or attenuation_changed
            or area_changed
            or shadow_changed
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

        self._entity_gizmo_lock(ctx, node)

        if ctx.select_model_camera is not None:
            active = ctx.model_camera_id == info.camera_id
            label = "Return to Editor Camera" if active else "View Through Camera"
            if imgui.button(ctx.tr(label)):
                ctx.select_model_camera(-1 if active else info.camera_id)

        (eye_changed, eye), (target_changed, target), (up_changed, up) = _vector_fields(
            ctx,
            node,
            "insp_camera_pose",
            (
                (ctx.tr("position"), view.eye, 0.01, "%.4f", None),
                (ctx.tr("target"), view.target, 0.01, "%.4f", None),
                (
                    ctx.tr("up"),
                    view.up,
                    0.01,
                    "%.4f",
                    np.array((0.0, 0.0, 1.0)),
                ),
            ),
        )
        fov_changed, fov = imgui.drag_float(
            f"{ctx.tr('vertical fov')}##camera_fov",
            float(np.degrees(view.fov_y)),
            0.1,
            1.0,
            179.0,
            "%.2f deg",
        )
        near_changed, near = imgui.drag_float(
            f"{ctx.tr('near')}##camera_near",
            float(view.near),
            0.001,
            1e-5,
            float(view.far),
            "%.6f",
        )
        far_changed, far = imgui.drag_float(
            f"{ctx.tr('far')}##camera_far",
            float(view.far),
            0.1,
            float(near),
            1e7,
            "%.3f",
        )
        ortho_changed, orthographic = imgui.checkbox(
            f"{ctx.tr('orthographic')}##camera_orthographic", view.orthographic
        )
        height_changed = False
        ortho_height = view.ortho_height
        if orthographic:
            height_changed, ortho_height = imgui.drag_float(
                f"{ctx.tr('ortho height')}##camera_ortho_height",
                float(view.ortho_height),
                0.01,
                1e-4,
                1e6,
                "%.4f",
            )

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
    return float(width) < 420.0 * float(style_scale)


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
    columns = 1 if compact else 7
    if not imgui.begin_table(table_id, columns, flags):
        return tuple((False, np.asarray(row[1], np.float64).copy()) for row in rows)
    if compact:
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
    else:
        imgui.table_setup_column(
            "value", imgui.TableColumnFlags_.width_fixed, 96.0 * ctx.style_scale
        )
        for axis in "xyz":
            imgui.table_setup_column(
                axis, imgui.TableColumnFlags_.width_fixed, 20.0 * ctx.style_scale
            )
            imgui.table_setup_column(f"{axis} value", imgui.TableColumnFlags_.width_stretch, 1.0)
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
    imgui.text(name)
    group_hovered = imgui.is_item_hovered()
    if group_hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
        imgui.set_clipboard_text(_format_vector(out))
    if group_hovered:
        imgui.set_tooltip("right-click: copy XYZ")

    if compact:
        row_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.no_saved_settings
            | imgui.TableFlags_.no_pad_inner_x
            | imgui.TableFlags_.no_pad_outer_x
        )
        if not imgui.begin_table(f"##{name}_axes_{node.node_id}", 6, row_flags):
            return False, out
        for axis in "xyz":
            imgui.table_setup_column(
                axis, imgui.TableColumnFlags_.width_fixed, 20.0 * ctx.style_scale
            )
            imgui.table_setup_column(f"{axis} value", imgui.TableColumnFlags_.width_stretch, 1.0)

    changed = False
    for axis, label in enumerate("XYZ"):
        imgui.table_next_column()
        color = ctx.theme.axis_color(axis)
        hovered_color = _lift_color(color, 0.20)
        active_color = _lift_color(color, 0.32)
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*color))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hovered_color))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*active_color))
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ctx.theme.bg_child))
        imgui.begin_disabled(not editable)
        reset = imgui.button(
            f"{label}##{name}_{axis}_{node.node_id}",
            imgui.ImVec2(imgui.get_content_region_avail().x, 0.0),
        )
        imgui.end_disabled()
        imgui.pop_style_color(4)
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip("click: reset to 0" if editable else "read only")
        if reset:
            out[axis] = resets[axis]
            changed = True

        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        imgui.begin_disabled(not editable)
        edited, value = imgui.drag_float(
            f"##{name}_{axis}_{node.node_id}", float(out[axis]), speed, 0.0, 0.0, fmt
        )
        imgui.end_disabled()
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        if hovered and imgui.is_mouse_clicked(imgui.MouseButton_.right):
            imgui.set_clipboard_text(fmt % out[axis])
        if hovered:
            hint = "drag: edit · right-click: copy" if editable else "read only · right-click: copy"
            imgui.set_tooltip(hint)
        if edited:
            out[axis] = value
            changed = True
    if compact:
        imgui.end_table()
    return changed, out


def _format_vector(values) -> str:
    return ", ".join(f"{float(value):.6g}" for value in values)


def _lift_color(color, amount: float):
    return (*(c + (1.0 - c) * amount for c in color[:3]), color[3])
