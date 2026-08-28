"""Model-local MJCF asset inventory and lifecycle controls."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, ModelAssetInfo, NodeType
from . import Panel, PanelContext, begin_kv_table, button_row_layout, button_width, labeled


def unique_asset_name(base: str, assets: tuple[ModelAssetInfo, ...], asset_type: str) -> str:
    """Return a model-local name that is unique within one MJCF asset namespace."""

    existing = {item.name for item in assets if item.type == asset_type}
    if base not in existing:
        return base
    index = 2
    while f"{base}{index}" in existing:
        index += 1
    return f"{base}{index}"


def filter_assets(
    assets: tuple[ModelAssetInfo, ...], asset_type: str, query: str
) -> tuple[ModelAssetInfo, ...]:
    """Filter a cached model asset inventory without touching the adapter."""

    needle = query.strip().casefold()
    return tuple(
        item
        for item in assets
        if (asset_type == "all" or item.type == asset_type)
        and (
            not needle
            or needle in item.name.casefold()
            or needle in item.type.casefold()
            or needle in item.file.casefold()
        )
    )


def _parse_vector(value: str, length: int, default: tuple[float, ...]) -> np.ndarray:
    try:
        parsed = np.asarray(tuple(float(item) for item in value.split()), np.float32).reshape(
            length
        )
    except (TypeError, ValueError):
        parsed = np.asarray(default, np.float32)
    return parsed


def _format_vector(value) -> str:
    return " ".join(f"{float(item):.9g}" for item in value)


def _hfield_size_editor(str_id: str, value) -> tuple[bool, np.ndarray]:
    edited = np.asarray(value, np.float32).reshape(4).copy()
    changed = False
    if begin_kv_table(str_id):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
        for index, label in enumerate(
            ("X half-size", "Y half-size", "elevation scale", "base depth")
        ):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(label)
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            item_changed, edited[index] = imgui.drag_float(
                f"##{str_id}-{index}",
                float(edited[index]),
                0.01,
                0.001,
                1000000.0,
                "%.3f",
            )
            changed |= item_changed
        imgui.end_table()
    return changed, edited


class AssetsPanel(Panel):
    """Browse model assets separately from scene-object bindings in Inspector."""

    name = "Assets"
    default_open = True
    shortcut = ""
    closable = False
    dock_with = "Hierarchy"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = -1
        self._cache_generation = -1
        self._assets: tuple[ModelAssetInfo, ...] = ()
        self._asset_type = "all"
        self._filter = ""
        self._selected: tuple[int, str, str] | None = None
        self._rename = ""
        self._selection_key: tuple[int, str, str] | None = None
        self._hfield_import_size = np.array((1.0, 1.0, 1.0, 0.1), np.float32)
        self._hfield_size = self._hfield_import_size.copy()
        self._hfield_source_size = self._hfield_import_size.copy()

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        models = ctx.session.scene_models
        if not models:
            imgui.text_disabled("no editable model assets")
            return
        model_ids = tuple(item.model_id for item in models)
        if self._model_id not in model_ids:
            selected = ctx.session.selected_node
            preferred = selected.model_id if selected is not None else model_ids[0]
            self._model_id = preferred if preferred in model_ids else model_ids[0]
            self._cache_generation = -1

        if begin_kv_table("asset_model"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("model")
            imgui.table_next_column()
            slot = model_ids.index(self._model_id)
            imgui.set_next_item_width(-1.0)
            changed, slot = imgui.combo("##asset-model", slot, tuple(item.name for item in models))
            if changed:
                self._model_id = model_ids[slot]
                self._cache_generation = -1
                self._selected = None
            imgui.end_table()

        self._refresh(ctx)
        self._import_controls(ctx)
        imgui.separator()

        imgui.set_next_item_width(-1.0)
        _changed, self._filter = imgui.input_text_with_hint(
            "##asset-filter", "Filter assets...", self._filter
        )
        types = ("all", *tuple(dict.fromkeys(item.type for item in self._assets)))
        if self._asset_type not in types:
            self._asset_type = "all"
        slot = types.index(self._asset_type)
        imgui.set_next_item_width(-1.0)
        changed, slot = imgui.combo("##asset-type", slot, tuple(item.title() for item in types))
        if changed:
            self._asset_type = types[slot]

        visible = filter_assets(self._assets, self._asset_type, self._filter)
        list_height = min(230.0, max(92.0, 25.0 * max(2, min(len(visible), 8))))
        if imgui.begin_child(
            "asset-list",
            imgui.ImVec2(0.0, list_height * ctx.style_scale),
            imgui.ChildFlags_.borders.value,
        ):
            for item in visible:
                key = (item.model_id, item.type, item.name)
                selected, _ = imgui.selectable(
                    f"{item.name}##{item.type}:{item.index}", self._selected == key
                )
                if selected:
                    self._selected = key
                imgui.same_line()
                imgui.text_disabled(f"{item.type} · used {len(item.references)}")
        imgui.end_child()
        if not visible:
            imgui.text_disabled("no matching assets")

        item = self._selected_asset()
        if item is not None:
            self._details(ctx, item)

    def _refresh(self, ctx: PanelContext) -> None:
        generation = ctx.session.structure_generation
        if generation == self._cache_generation:
            return
        self._cache_generation = generation
        self._assets = ctx.session.model_assets(self._model_id)
        if self._selected is not None and not any(
            (item.model_id, item.type, item.name) == self._selected for item in self._assets
        ):
            self._selected = None

    def _selected_asset(self) -> ModelAssetInfo | None:
        if self._selected is None:
            return None
        return next(
            (
                item
                for item in self._assets
                if (item.model_id, item.type, item.name) == self._selected
            ),
            None,
        )

    def _import_controls(self, ctx: PanelContext) -> None:
        editable = ctx.session.paused and ctx.session.adapter.caps.model_assets
        mesh_label = "Import Mesh..."
        hfield_label = "Import Height Field..."
        widths = (
            button_width(mesh_label),
            button_width(hfield_label),
        )
        inline = button_row_layout(
            widths,
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        if not editable:
            imgui.begin_disabled()
        if imgui.small_button(mesh_label) and ctx.request_model_asset_import is not None:
            ctx.request_model_asset_import(self._model_id, "mesh", ())
        if inline[1]:
            imgui.same_line()
        if imgui.small_button(hfield_label) and ctx.request_model_asset_import is not None:
            ctx.request_model_asset_import(
                self._model_id,
                "hfield",
                (("size", _format_vector(self._hfield_import_size)),),
            )
        if not editable:
            imgui.end_disabled()
        if imgui.tree_node("Height-field import size"):
            _changed, self._hfield_import_size = _hfield_size_editor(
                "hfield-import-size", self._hfield_import_size
            )
            imgui.text_disabled("Set physical dimensions before importing the PNG.")
            imgui.tree_pop()
        if not ctx.session.paused:
            imgui.text_disabled("Pause the simulation to edit model assets")

    def _details(self, ctx: PanelContext, item: ModelAssetInfo) -> None:
        key = (item.model_id, item.type, item.name)
        if key != self._selection_key:
            self._selection_key = key
            self._rename = item.name
            fields = {field.name: field.value for field in item.fields}
            self._hfield_size = _parse_vector(fields.get("size", ""), 4, (1.0, 1.0, 1.0, 0.1))
            self._hfield_source_size = self._hfield_size.copy()
        imgui.separator()
        imgui.text(f"{item.type}: {item.name}")
        if begin_kv_table("asset_details"):
            labeled("source", item.file or "inline")
            labeled("references", str(len(item.references)))
            for field in item.fields:
                if field.value:
                    labeled(field.name, field.value)
            imgui.end_table()
        for reference in item.references:
            imgui.bullet_text(reference)

        editable = ctx.session.paused and ctx.session.adapter.caps.model_assets
        if not editable:
            imgui.begin_disabled()
        imgui.set_next_item_width(-1.0)
        _changed, self._rename = imgui.input_text("##asset-name", self._rename)
        rename = self._rename.strip()
        action_labels = ["Rename", "Duplicate"]
        if item.type in ("mesh", "hfield"):
            action_labels.append("Replace File...")
        action_labels.append("Delete")
        inline = button_row_layout(
            tuple(button_width(label) for label in action_labels),
            imgui.get_content_region_avail().x,
            imgui.get_style().item_spacing.x,
        )
        action_index = 0
        if not rename or rename == item.name:
            imgui.begin_disabled()
        if imgui.small_button(action_labels[action_index]):
            result = ctx.submit(cmd.RenameModelAsset(item.model_id, item.type, item.name, rename))
            if result.ok:
                self._selected = (item.model_id, item.type, rename)
                self._selection_key = None
        if not rename or rename == item.name:
            imgui.end_disabled()
        action_index += 1
        if inline[action_index]:
            imgui.same_line()
        if imgui.small_button(action_labels[action_index]):
            duplicate = unique_asset_name(f"{item.name}_copy", self._assets, item.type)
            result = ctx.submit(
                cmd.DuplicateModelAsset(item.model_id, item.type, item.name, duplicate)
            )
            if result.ok:
                self._selected = (item.model_id, item.type, duplicate)
                self._selection_key = None
        action_index += 1
        if item.type in ("mesh", "hfield"):
            if inline[action_index]:
                imgui.same_line()
            if (
                imgui.small_button(action_labels[action_index])
                and ctx.request_model_asset_replace is not None
            ):
                ctx.request_model_asset_replace(item.model_id, item.type, item.name)
            action_index += 1
        if item.references:
            imgui.begin_disabled()
        if inline[action_index]:
            imgui.same_line()
        if imgui.small_button(action_labels[action_index]):
            ctx.submit(cmd.RemoveModelAsset(item.model_id, item.type, item.name))
        if item.references:
            imgui.end_disabled()
            imgui.set_item_tooltip("Remove or replace every reference before deleting this asset")

        if item.type == "hfield":
            self._hfield_controls(ctx, item)
        if item.type in ("mesh", "hfield"):
            self._assignment_control(ctx, item)
        if not editable:
            imgui.end_disabled()

    def _hfield_controls(self, ctx: PanelContext, item: ModelAssetInfo) -> None:
        imgui.separator()
        imgui.text_disabled("height-field dimensions")
        _changed, self._hfield_size = _hfield_size_editor("hfield-size", self._hfield_size)
        dirty = not np.allclose(self._hfield_size, self._hfield_source_size)
        if not dirty:
            imgui.begin_disabled()
        if imgui.small_button("Apply Dimensions"):
            result = ctx.submit(
                cmd.SetModelPropertyGroups(
                    item.model_id,
                    (
                        (
                            f"asset:hfield:{item.index}",
                            (("size", _format_vector(self._hfield_size)),),
                        ),
                    ),
                )
            )
            if result.ok:
                self._hfield_source_size = self._hfield_size.copy()
        if not dirty:
            imgui.end_disabled()

    @staticmethod
    def _assignment_control(ctx: PanelContext, item: ModelAssetInfo) -> None:
        node = ctx.session.selected_node
        if node is None or node.type is not NodeType.GEOM:
            imgui.text_disabled("Select a geometry to assign this asset")
            return
        properties = ctx.session.geometry_shape_properties(node.node_id)
        choices = (
            properties.mesh_names
            if properties is not None and item.type == "mesh"
            else properties.height_field_names
            if properties is not None and item.type == "hfield"
            else ()
        )
        if item.name not in choices:
            imgui.text_disabled("The selected geometry belongs to a different model")
            return
        if imgui.button("Assign to Selected Geometry"):
            ctx.submit(cmd.SetGeometryShape(node.node_id, item.type, item.name))


__all__ = ["AssetsPanel", "filter_assets", "unique_asset_name"]
