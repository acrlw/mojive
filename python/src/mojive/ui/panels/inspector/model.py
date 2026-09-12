"""Inspector: model."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import (
    ModelComponentInfo,
    SceneModelInfo,
    SceneNode,
)
from mojive.ui.panels import (
    PanelContext,
    begin_kv_table,
    button_width,
    copyable_name_item,
)

from .fields import (
    _vector_fields,
)
from .support import (
    _MODEL_COMPONENT_CATEGORIES,
    _MULTILINE_COMPONENT_FIELDS,
    _component_value_editor,
    _matching_path_preset,
    _path_preset_label,
    _unique_component_name,
)


class _Model:
    """Private model methods of InspectorPanel; state belongs to its owner."""

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
        imgui.text(ctx.tr("Model placement"))
        can_edit = ctx.session.paused and info.removable and gizmo is not None
        if not placement_active:
            if not can_edit:
                imgui.begin_disabled()
            begin_placement = imgui.button(ctx.tr("Edit Placement"))
            if not can_edit:
                imgui.end_disabled()
                imgui.set_item_tooltip(ctx.tr("Locked to avoid accidental model rebuilds."))
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
            if not ctx.live_model_updates:
                return
            apply = imgui.button(ctx.tr("Apply Placement"))
            imgui.set_item_tooltip(
                ctx.tr("Preview only; applying rebuilds the composed model once.")
            )
            imgui.same_line()
            cancel = imgui.button(f"{ctx.tr('Cancel')}##model-placement")
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

        if info.removable and imgui.button(ctx.tr("Remove Model")):
            ctx.submit(cmd.RemoveSceneModel(info.model_id))
        if ctx.session.adapter.caps.supports("mujoco.mjcf"):
            imgui.same_line()
            if not ctx.session.paused:
                imgui.begin_disabled()
            edit_source = imgui.button(ctx.tr("Edit MJCF Source..."))
            if not ctx.session.paused:
                imgui.end_disabled()
            if edit_source:
                source = ctx.session.adapter.scene_model_source(info.model_id)
                if source is not None:
                    self._source_model_id = info.model_id
                    self._source_text = source
                    self._source_error = ""
                    self._open_source_popup = True
        if ctx.session.adapter.caps.supports("model.components"):
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
        imgui.text_disabled(ctx.tr("Model Components"))
        editable = ctx.session.paused
        if not any(self._component_counts.values()):
            imgui.text_disabled(ctx.tr("no authored components"))
        for category, count in self._component_counts.items():
            if not count or not imgui.collapsing_header(f"{category.capitalize()} ({count})"):
                continue
            if category not in self._component_cache:
                self._component_cache[category] = ctx.session.model_components(model_id, category)
            components = self._component_cache[category]
            flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_saved_settings
            if not imgui.begin_table(f"components-{category}", 3, flags):
                continue
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column(
                "Edit", imgui.TableColumnFlags_.width_fixed, button_width(ctx.tr("Edit"))
            )
            imgui.table_setup_column(
                "Delete", imgui.TableColumnFlags_.width_fixed, button_width(ctx.tr("Delete"))
            )
            clipper = imgui.ListClipper()
            clipper.begin(len(components))
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    component = components[index]
                    imgui.push_id(f"{category}-{component.component_id}")
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.align_text_to_frame_padding()
                    label = f"{component.name}  ({component.subtype})"
                    imgui.text(label)
                    imgui.set_item_tooltip(label)
                    imgui.table_next_column()
                    if not editable:
                        imgui.begin_disabled()
                    if imgui.button(ctx.tr("Edit")):
                        self._begin_component_edit(component)
                    imgui.table_next_column()
                    if imgui.button(ctx.tr("Delete")):
                        ctx.submit_model_edit(
                            cmd.RemoveModelComponent(model_id, category, component.component_id)
                        )
                    if not editable:
                        imgui.end_disabled()
                    imgui.pop_id()
            clipper.end()
            imgui.end_table()

        if not editable:
            imgui.begin_disabled()
        if imgui.begin_combo(
            f"{ctx.tr('Add Component...')}##model-component", ctx.tr("select type")
        ):
            if not self._component_presets:
                self._component_presets = {
                    category: ctx.session.model_component_presets(model_id, category)
                    for category in _MODEL_COMPONENT_CATEGORIES
                }
            if not any(self._component_presets.values()):
                imgui.text_disabled(ctx.tr("Add the referenced model elements first"))
            for category, subtypes in self._component_presets.items():
                for subtype in subtypes:
                    selected, _ = imgui.selectable(f"{category.capitalize()} / {subtype}", False)
                    if selected:
                        if category not in self._component_cache:
                            self._component_cache[category] = ctx.session.model_components(
                                model_id, category
                            )
                        names = {component.name for component in self._component_cache[category]}
                        name = _unique_component_name(category, names)
                        ctx.submit_model_edit(
                            cmd.AddModelComponent(model_id, category, subtype, name)
                        )
            imgui.end_combo()
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip(ctx.tr("Pause the simulation before editing model components"))

    def _refresh_component_cache(self, ctx: PanelContext, model_id: int) -> None:
        generation = ctx.session.structure_generation
        if (
            generation == self._component_cache_generation
            and model_id == self._component_cache_model
        ):
            return
        self._component_cache_generation = generation
        self._component_cache_model = model_id
        self._component_cache.clear()
        self._component_presets.clear()
        self._component_counts = {
            category: ctx.session.model_component_count(model_id, category)
            for category in _MODEL_COMPONENT_CATEGORIES
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
        popup_title = f"{ctx.tr('Model Component')}###Model Component"
        if self._open_component_popup and component is not None:
            imgui.open_popup(popup_title)
            self._open_component_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(560.0 * ctx.style_scale, 520.0 * ctx.style_scale),
            imgui.Cond_.appearing.value,
        )
        visible, _ = imgui.begin_popup_modal(popup_title)
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
            imgui.text_disabled(ctx.tr("name"))
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
                    ctx,
                    f"##component-field-{index}",
                    field[1],
                    choices.get(field[0], ()),
                    multiline=field[0] in _MULTILINE_COMPONENT_FIELDS or len(field[1]) > 72,
                )
            imgui.end_table()
        if self._component_path:
            imgui.separator()
            imgui.text_disabled(ctx.tr("Path"))
        for path_index, (element_type, fields) in enumerate(tuple(self._component_path)):
            imgui.push_id(f"path-{path_index}")
            object_type = next((value for name, value in fields if name == "objtype"), "")
            suffix = f" · {object_type}" if object_type else ""
            imgui.text_disabled(f"{path_index + 1}. {element_type}{suffix}")
            imgui.same_line()
            if path_index == 0:
                imgui.begin_disabled()
            move_up = imgui.button(ctx.tr("Up"))
            if path_index == 0:
                imgui.end_disabled()
            imgui.same_line()
            if path_index + 1 == len(self._component_path):
                imgui.begin_disabled()
            move_down = imgui.button(ctx.tr("Down"))
            if path_index + 1 == len(self._component_path):
                imgui.end_disabled()
            imgui.same_line()
            remove = imgui.button(ctx.tr("Remove"))
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
                        ctx,
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
        if self._component_path_presets and imgui.begin_combo(
            ctx.tr("Add path item"), ctx.tr("select type")
        ):
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
            if imgui.button(f"{ctx.tr('Copy error')}##component"):
                imgui.set_clipboard_text(self._component_error)
        if imgui.button(ctx.tr("Apply"), imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            ctx.submit_model_edit(
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
                ),
                self._component_edit_completed,
            )
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button(ctx.tr("Cancel"), imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            self._component_edit = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _component_edit_completed(self, result) -> None:
        if result.ok:
            self._component_edit = None
        else:
            self._component_error = result.message
            self._open_component_popup = True

    def _source_edit_completed(self, result) -> None:
        if not result.ok:
            self._source_error = result.message
            self._open_source_popup = True

    def _draw_model_source(self, ctx: PanelContext) -> None:
        popup_title = f"{ctx.tr('MJCF Source')}###MJCF Source"
        if self._open_source_popup:
            imgui.open_popup(popup_title)
            self._open_source_popup = False
        imgui.set_next_window_size(
            imgui.ImVec2(820.0 * ctx.style_scale, 620.0 * ctx.style_scale),
            imgui.Cond_.appearing.value,
        )
        visible, _ = imgui.begin_popup_modal(popup_title)
        if not visible:
            return
        _changed, self._source_text = imgui.input_text_multiline(
            "##mjcf_source",
            self._source_text,
            imgui.ImVec2(-1.0, -70.0 * ctx.style_scale),
            imgui.InputTextFlags_.allow_tab_input.value,
        )
        imgui.set_item_tooltip(
            ctx.tr("MjSpec validates and recompiles the model when changes are applied.")
        )
        if self._source_error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.35, 0.3, 1.0), self._source_error)
            if imgui.button(f"{ctx.tr('Copy error')}##source"):
                imgui.set_clipboard_text(self._source_error)
        if imgui.button(ctx.tr("Apply"), imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            ctx.submit_model_edit(
                cmd.SetModelSource(self._source_model_id, self._source_text),
                self._source_edit_completed,
            )
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button(ctx.tr("Cancel"), imgui.ImVec2(100.0 * ctx.style_scale, 0.0)):
            imgui.close_current_popup()
        imgui.end_popup()

    def _identity(self, ctx: PanelContext, node: SceneNode) -> None:
        text = f"{ctx.tr('node id')} {node.node_id} · {ctx.tr('object id')} {node.object_id}"
        if node.body_index >= 0:
            text += f" · {ctx.tr('body')} {node.body_index}"
        if node.posable:
            text += f" · {ctx.tr('posable')}"
        imgui.push_font(None, imgui.get_font_size() * 0.85)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ctx.theme.text_disabled))
        imgui.text_wrapped(text)
        imgui.pop_style_color()
        imgui.pop_font()
        copyable_name_item(ctx, text, imgui.get_content_region_avail().x)
        imgui.separator()
