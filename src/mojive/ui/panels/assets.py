"""Model-local MJCF asset inventory and lifecycle controls."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, ModelAssetInfo, NodeType
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_row_layout,
    button_width,
    labeled,
    search_input,
)


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


def height_field_preview_color(value: float) -> tuple[float, float, float, float]:
    """Map normalized elevation to a legible terrain-style preview color."""

    t = min(1.0, max(0.0, float(value)))
    low = (0.08, 0.18, 0.38)
    middle = (0.12, 0.58, 0.46)
    high = (0.96, 0.82, 0.28)
    if t <= 0.5:
        amount = t * 2.0
        start, end = low, middle
    else:
        amount = (t - 0.5) * 2.0
        start, end = middle, high
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
        start[2] + (end[2] - start[2]) * amount,
        1.0,
    )


def _hfield_size_editor(ctx: PanelContext, str_id: str, value) -> tuple[bool, np.ndarray]:
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
            imgui.text_disabled(ctx.tr(label))
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

    id = "assets"
    name = "Assets"
    default_open = False
    shortcut = ""
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
        self._hfield_preview_item: ModelAssetInfo | None = None
        self._hfield_preview_grid: tuple[int, int, tuple[int, ...]] = (0, 0, ())
        self._texture_import_type = "2d"
        self._material_create_name = "material"

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def focus(self, model_id: int, asset_type: str, name: str) -> None:
        """Open this panel with one model-local asset selected."""

        self.open = True
        self._model_id = int(model_id)
        self._asset_type = str(asset_type)
        self._filter = ""
        self._selected = (self._model_id, self._asset_type, str(name))
        self._cache_generation = -1

    def draw(self, ctx: PanelContext) -> None:
        models = ctx.session.scene_models
        if not models:
            imgui.text_disabled(ctx.tr("no editable model assets"))
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
            imgui.text_disabled(ctx.tr("model"))
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
        _changed, self._filter = search_input(
            "##asset-filter",
            self._filter,
            hint=ctx.tr("Filter assets..."),
            search_tooltip=ctx.tr("Search assets"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        types = ("all", *tuple(dict.fromkeys(item.type for item in self._assets)))
        if self._asset_type not in types:
            self._asset_type = "all"
        slot = types.index(self._asset_type)
        type_labels = tuple(ctx.tr("All") if item == "all" else item for item in types)
        imgui.set_next_item_width(-1.0)
        changed, slot = imgui.combo("##asset-type", slot, type_labels)
        if changed:
            self._asset_type = types[slot]

        visible = filter_assets(self._assets, self._asset_type, self._filter)
        list_height = min(230.0, max(92.0, 25.0 * max(2, min(len(visible), 8))))
        if imgui.begin_child(
            "asset-list",
            imgui.ImVec2(0.0, list_height * ctx.style_scale),
            imgui.ChildFlags_.borders.value,
        ):
            flags = (
                imgui.TableFlags_.sizing_stretch_prop
                | imgui.TableFlags_.row_bg
                | imgui.TableFlags_.no_saved_settings
            )
            if imgui.begin_table("asset-list-table", 3, flags):
                imgui.table_setup_column(ctx.tr("Name"), imgui.TableColumnFlags_.width_stretch, 1.0)
                imgui.table_setup_column(ctx.tr("Type"), imgui.TableColumnFlags_.width_fixed)
                imgui.table_setup_column(ctx.tr("Used"), imgui.TableColumnFlags_.width_fixed)
                imgui.table_headers_row()
                clipper = imgui.ListClipper()
                clipper.begin(len(visible))
                while clipper.step():
                    for index in range(clipper.display_start, clipper.display_end):
                        item = visible[index]
                        key = (item.model_id, item.type, item.name)
                        imgui.table_next_row()
                        imgui.table_next_column()
                        selected, _ = imgui.selectable(
                            f"{item.name}##{item.type}:{item.index}",
                            self._selected == key,
                            imgui.SelectableFlags_.span_all_columns.value,
                        )
                        if selected:
                            self._selected = key
                        imgui.table_next_column()
                        imgui.text_disabled(item.type)
                        imgui.table_next_column()
                        imgui.text_disabled(str(len(item.references)))
                imgui.end_table()
        imgui.end_child()
        if not visible:
            imgui.text_disabled(ctx.tr("no matching assets"))

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
        mesh_label = ctx.tr("Import Mesh...")
        hfield_label = ctx.tr("Import Height Field...")
        texture_label = ctx.tr("Import Texture...")
        widths = (
            button_width(mesh_label),
            button_width(hfield_label),
            button_width(texture_label),
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
        if inline[2]:
            imgui.same_line()
        if imgui.small_button(texture_label) and ctx.request_texture_import is not None:
            ctx.request_texture_import(self._model_id, -1, self._texture_import_type)
        if not editable:
            imgui.end_disabled()
        if begin_kv_table("texture_import_type"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("texture type"))
            imgui.table_next_column()
            texture_types = ("2d", "cube", "skybox")
            slot = texture_types.index(self._texture_import_type)
            imgui.set_next_item_width(-1.0)
            changed, slot = imgui.combo(
                "##texture-import-type",
                slot,
                (ctx.tr("2D"), ctx.tr("Cube"), ctx.tr("Skybox")),
            )
            if changed:
                self._texture_import_type = texture_types[slot]
            imgui.end_table()
        hfield_open = imgui.tree_node(
            f"{ctx.tr('Height-field import size')}##height-field-import-size"
        )
        imgui.set_item_tooltip(ctx.tr("Set physical dimensions before importing the PNG."))
        if hfield_open:
            _changed, self._hfield_import_size = _hfield_size_editor(
                ctx, "hfield-import-size", self._hfield_import_size
            )
            imgui.tree_pop()
        material_open = imgui.tree_node(f"{ctx.tr('New material')}##new-material")
        imgui.set_item_tooltip(ctx.tr("Creates an unbound material asset for later assignment."))
        if material_open:
            imgui.set_next_item_width(-1.0)
            _changed, self._material_create_name = imgui.input_text(
                f"{ctx.tr('name')}##new-material-name", self._material_create_name
            )
            material_name = self._material_create_name.strip()
            if not material_name or not editable:
                imgui.begin_disabled()
            if imgui.small_button(ctx.tr("Create Material")):
                result = ctx.submit(cmd.CreateModelMaterial(self._model_id, material_name))
                if result.ok:
                    self._selected = (self._model_id, "material", material_name)
                    self._selection_key = None
                    self._material_create_name = unique_asset_name(
                        "material", ctx.session.model_assets(self._model_id), "material"
                    )
            if not material_name or not editable:
                imgui.end_disabled()
            imgui.tree_pop()
        if not ctx.session.paused:
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model assets"))

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
            labeled(ctx.tr("source"), item.file or ctx.tr("inline"))
            labeled(ctx.tr("references"), str(len(item.references)))
            for field in item.fields:
                if item.type == "hfield" and field.name == "elevation":
                    labeled(
                        ctx.tr("elevation"),
                        f"{len(field.value.split())} {ctx.tr('inline samples')}",
                    )
                elif field.value:
                    labeled(field.name, field.value)
            imgui.end_table()
        for reference in item.references:
            imgui.bullet_text(reference)
        if item.type == "hfield":
            self._height_field_preview(ctx, item)

        editable = ctx.session.paused and ctx.session.adapter.caps.model_assets
        if not editable:
            imgui.begin_disabled()
        imgui.set_next_item_width(-1.0)
        _changed, self._rename = imgui.input_text("##asset-name", self._rename)
        rename = self._rename.strip()
        action_labels = [ctx.tr("Rename"), ctx.tr("Duplicate")]
        if item.type in ("mesh", "hfield", "texture"):
            action_labels.append(ctx.tr("Replace File..."))
        action_labels.append(ctx.tr("Delete"))
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
        if item.type in ("mesh", "hfield", "texture"):
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
            imgui.set_item_tooltip(
                ctx.tr("Remove or replace every reference before deleting this asset")
            )

        if item.type == "hfield":
            self._hfield_controls(ctx, item)
        if item.type == "material":
            self._material_controls(ctx, item)
        if item.type in ("mesh", "hfield"):
            self._assignment_control(ctx, item)
        if not editable:
            imgui.end_disabled()

    def _height_field_preview(self, ctx: PanelContext, item: ModelAssetInfo) -> None:
        rows, columns = item.preview_shape
        source_rows, source_columns = item.data_shape
        if rows <= 0 or columns <= 0 or len(item.preview_values) != rows * columns:
            imgui.text_disabled(ctx.tr("height-field preview unavailable"))
            return

        available = max(1.0, imgui.get_content_region_avail().x)
        height = float(
            np.clip(
                available * source_rows / max(source_columns, 1),
                72.0 * ctx.style_scale,
                180.0 * ctx.style_scale,
            )
        )
        imgui.invisible_button("##height-field-preview", imgui.ImVec2(available, height))
        lo = imgui.get_item_rect_min()
        hi = imgui.get_item_rect_max()
        low, high = item.preview_range
        display_rows, display_columns, colors = self._height_field_preview_data(item)
        cell_width = (hi.x - lo.x) / display_columns
        cell_height = (hi.y - lo.y) / display_rows
        draw_list = imgui.get_window_draw_list()
        for row in range(display_rows):
            for column in range(display_columns):
                rgba = colors[row * display_columns + column]
                draw_list.add_rect_filled(
                    imgui.ImVec2(lo.x + column * cell_width, lo.y + row * cell_height),
                    imgui.ImVec2(
                        lo.x + (column + 1) * cell_width,
                        lo.y + (row + 1) * cell_height,
                    ),
                    rgba,
                )
        border = imgui.color_convert_float4_to_u32(imgui.ImVec4(*ctx.theme.border))
        draw_list.add_rect(lo, hi, border)
        imgui.text_disabled(
            f"{source_columns} × {source_rows} {ctx.tr('samples')} · "
            f"{ctx.tr('normalized')} {low:.4g} .. {high:.4g}"
        )

    def _height_field_preview_data(
        self,
        item: ModelAssetInfo,
    ) -> tuple[int, int, tuple[int, ...]]:
        """Cache a bounded display grid and its packed colors for one stable asset."""

        if self._hfield_preview_item is item:
            return self._hfield_preview_grid
        rows, columns = item.preview_shape
        values = np.asarray(item.preview_values, np.float64).reshape(rows, columns)
        display_rows = min(rows, 24)
        display_columns = min(columns, 24)
        row_indices = np.rint(np.linspace(0, rows - 1, display_rows)).astype(np.intp)
        column_indices = np.rint(np.linspace(0, columns - 1, display_columns)).astype(np.intp)
        display = values[np.ix_(row_indices, column_indices)].reshape(-1)
        low, high = item.preview_range
        span = high - low
        normalized = (display - low) / span if span > 1e-12 else np.full_like(display, 0.5)
        colors = tuple(
            imgui.color_convert_float4_to_u32(
                imgui.ImVec4(*height_field_preview_color(float(value)))
            )
            for value in normalized
        )
        self._hfield_preview_item = item
        self._hfield_preview_grid = (display_rows, display_columns, colors)
        return self._hfield_preview_grid

    def _hfield_controls(self, ctx: PanelContext, item: ModelAssetInfo) -> None:
        imgui.separator()
        imgui.text_disabled(ctx.tr("height-field dimensions"))
        _changed, self._hfield_size = _hfield_size_editor(ctx, "hfield-size", self._hfield_size)
        dirty = not np.allclose(self._hfield_size, self._hfield_source_size)
        if not dirty:
            imgui.begin_disabled()
        if imgui.small_button(ctx.tr("Apply Dimensions")):
            result = ctx.submit(
                cmd.SetHeightFieldSize(
                    item.model_id,
                    item.name,
                    tuple(float(value) for value in self._hfield_size),
                )
            )
            if result.ok:
                self._hfield_source_size = self._hfield_size.copy()
        if not dirty:
            imgui.end_disabled()

    def _material_controls(self, ctx: PanelContext, item: ModelAssetInfo) -> None:
        source = ctx.session.source
        material_index = item.runtime_index
        if source is None or not 0 <= material_index < len(source.materials):
            imgui.text_disabled(ctx.tr("compiled material unavailable"))
            return
        material = source.materials[material_index]
        rgba = material.rgba
        emission = material.emission
        specular = material.specular
        shininess = material.shininess
        reflectance = material.reflectance
        metallic = material.metallic
        roughness = material.roughness
        texture = material.texture
        tex_repeat = material.tex_repeat
        tex_uniform = material.tex_uniform
        changed = False

        imgui.separator()
        imgui.text_disabled(ctx.tr("material appearance"))
        if begin_kv_table("asset_material"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("base color"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            item_changed, rgba = imgui.color_edit4("##asset-material-rgba", rgba)
            changed |= item_changed

            scalar_fields = (
                ("emission", emission, 10.0),
                ("specular", specular, 1.0),
                ("shininess", shininess, 1.0),
                ("reflectance", reflectance, 1.0),
            )
            edited_scalars: list[float] = []
            for label, value, maximum in scalar_fields:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(ctx.tr(label))
                imgui.table_next_column()
                imgui.set_next_item_width(-1.0)
                item_changed, value = imgui.drag_float(
                    f"##asset-material-{label}", value, 0.01, 0.0, maximum, "%.2f"
                )
                changed |= item_changed
                edited_scalars.append(float(value))
            emission, specular, shininess, reflectance = edited_scalars

            for label, value, default in (
                ("metallic", metallic, 0.0),
                ("roughness", roughness, 0.5),
            ):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(ctx.tr(label))
                imgui.table_next_column()
                enabled = value >= 0.0
                item_changed, enabled = imgui.checkbox(f"##asset-material-{label}-enabled", enabled)
                if item_changed:
                    value = default if enabled else -1.0
                    changed = True
                imgui.set_item_tooltip(
                    ctx.tr("Toggle the MuJoCo PBR override; disabled uses classic material shading")
                )
                imgui.same_line()
                if not enabled:
                    imgui.begin_disabled()
                imgui.set_next_item_width(-1.0)
                item_changed, value = imgui.drag_float(
                    f"##asset-material-{label}", value, 0.01, 0.0, 1.0, "%.2f"
                )
                changed |= item_changed
                if not enabled:
                    imgui.end_disabled()
                if label == "metallic":
                    metallic = float(value)
                else:
                    roughness = float(value)

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("texture"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            if imgui.begin_combo("##asset-material-texture", texture or ctx.tr("none")):
                for candidate in (None, *ctx.session.model_texture_names(item.model_id)):
                    selected, _ = imgui.selectable(
                        candidate or ctx.tr("none"), candidate == texture
                    )
                    if selected:
                        texture = candidate
                        changed = True
                imgui.end_combo()

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("texture repeat"))
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            item_changed, tex_repeat = imgui.drag_float2(
                "##asset-material-repeat", tex_repeat, 0.05, 0.01, 1000.0, "%.2f"
            )
            changed |= item_changed

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(ctx.tr("uniform scale"))
            imgui.table_next_column()
            item_changed, tex_uniform = imgui.checkbox("##asset-material-uniform", tex_uniform)
            changed |= item_changed
            imgui.end_table()

        if changed:
            result = ctx.submit(
                cmd.SetMaterial(
                    material_index,
                    replace(
                        material,
                        rgba=np.asarray(rgba, np.float32),
                        emission=emission,
                        specular=specular,
                        shininess=shininess,
                        reflectance=reflectance,
                        metallic=metallic,
                        roughness=roughness,
                        texture=texture,
                        tex_repeat=np.asarray(tex_repeat, np.float32),
                        tex_uniform=bool(tex_uniform),
                    ),
                )
            )
            if result.ok:
                self._selection_key = None

    @staticmethod
    def _assignment_control(ctx: PanelContext, item: ModelAssetInfo) -> None:
        node = ctx.session.selected_node
        if node is None or node.type is not NodeType.GEOM:
            imgui.text_disabled(ctx.tr("Select a geometry to assign this asset"))
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
            imgui.text_disabled(ctx.tr("The selected geometry belongs to a different model"))
            return
        if imgui.button(ctx.tr("Assign to Selected Geometry")):
            ctx.submit(cmd.SetGeometryShape(node.node_id, item.type, item.name))


__all__ = [
    "AssetsPanel",
    "filter_assets",
    "height_field_preview_color",
    "unique_asset_name",
]
