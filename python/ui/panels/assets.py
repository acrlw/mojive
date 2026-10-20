"""Model-local MJCF asset inventory and lifecycle controls."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, ModelAssetInfo, NodeType
from mojive.ui.panels import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_row_layout,
    button_width,
    labeled,
    padded_selectable,
    search_input,
)
from mojive.ui.panels.presentation import PropertyPresentation


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


def _hfield_size_editor(
    ctx: PanelContext, str_id: str, value, presentation=None
) -> tuple[bool, np.ndarray]:
    edited = np.asarray(value, np.float32).reshape(4).copy()
    changed = False
    if presentation:
        if presentation.begin_table(str_id):
            labels = tuple(
                ctx.tr(label)
                for label in ("X half-size", "Y half-size", "elevation scale", "base depth")
            )
            # Four visible fields cost less than starting a native clipper. Use
            # clipping when the group scrolls out of view, retaining active edits.
            rows = (
                range(len(labels))
                if imgui.is_rect_visible((1, imgui.get_frame_height()))
                else presentation.clipped_rows(labels)
            )
            for index in rows:
                presentation.row(labels[index])
                item_changed, edited[index] = presentation.numeric(
                    f"##{str_id}-{index}", float(edited[index]), 0.01, 0.001, 1000000, "%.3f"
                )
                changed |= item_changed
            presentation.end_table()
    else:
        if begin_kv_table(str_id):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            for index, label in enumerate(
                ("X half-size", "Y half-size", "elevation scale", "base depth")
            ):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.align_text_to_frame_padding()
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
        self.presentation: PropertyPresentation | None = None
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

        if self._begin_properties(ctx, "asset_model"):
            self._property(ctx.tr("model"), control_part="select")
            slot = model_ids.index(self._model_id)
            imgui.set_next_item_width(-1.0)
            changed, slot = self._combo("##asset-model", slot, tuple(item.name for item in models))
            if changed:
                self._model_id = model_ids[slot]
                self._cache_generation = -1
                self._selected = None
            self._end_properties()

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
        changed, slot = self._combo("##asset-type", slot, type_labels)
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
            style = imgui.get_style()
            # The inventory needs column gutters even when surrounding forms use zero padding.
            imgui.push_style_var(
                imgui.StyleVar_.cell_padding,
                (max(style.cell_padding.x, style.item_spacing.x / 2), style.cell_padding.y),
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
                        selected, _ = padded_selectable(
                            f"{item.name}##{item.type}:{item.index}",
                            self._selected == key,
                            imgui.SelectableFlags_.span_all_columns.value,
                        )
                        if selected:
                            self._selected = key
                        imgui.table_next_column()
                        imgui.align_text_to_frame_padding()
                        imgui.text_disabled(item.type)
                        imgui.table_next_column()
                        imgui.align_text_to_frame_padding()
                        imgui.text_disabled(str(len(item.references)))
                self._end_properties()
            imgui.pop_style_var()
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
        actions = (
            (ctx.tr("Import Mesh..."), editable and ctx.request_model_asset_import is not None),
            (
                ctx.tr("Import Height Field..."),
                editable and ctx.request_model_asset_import is not None,
            ),
            (ctx.tr("Import Texture..."), editable and ctx.request_texture_import is not None),
        )
        imgui.set_next_item_width(-1)
        selected = None
        if self.presentation:
            selected = self.presentation.action_menu("asset-import", ctx.tr("Import"), actions)
        else:
            imgui.begin_disabled(not any(enabled for _, enabled in actions))
            opened = imgui.begin_combo("##asset-import", ctx.tr("Import"))
            if opened:
                for index, (label, enabled) in enumerate(actions):
                    if imgui.menu_item(label, "", False, enabled)[0]:
                        selected = index
                imgui.end_combo()
            imgui.end_disabled()
        if selected == 0:
            ctx.request_model_asset_import(self._model_id, "mesh", ())
        elif selected == 1:
            ctx.request_model_asset_import(
                self._model_id,
                "hfield",
                (("size", _format_vector(self._hfield_import_size)),),
            )
        elif selected == 2:
            ctx.request_texture_import(self._model_id, -1, self._texture_import_type)
        if self._section(f"{ctx.tr('Import options')}##asset-import-options"):
            self._import_options(ctx)
        self._material_creation(ctx, editable)
        if not ctx.session.paused:
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model assets"))

    def _import_options(self, ctx):
        if self._begin_properties(ctx, "texture_import_type"):
            self._property(ctx.tr("texture type"), control_part="select")
            texture_types = ("2d", "cube", "skybox")
            slot = texture_types.index(self._texture_import_type)
            imgui.set_next_item_width(-1.0)
            changed, slot = self._combo(
                "##texture-import-type",
                slot,
                (ctx.tr("2D"), ctx.tr("Cube"), ctx.tr("Skybox")),
            )
            if changed:
                self._texture_import_type = texture_types[slot]
            self._end_properties()
        hfield_open = self._section(
            f"{ctx.tr('Height-field import size')}##height-field-import-size"
        )
        self._tooltip(ctx.tr("Set physical dimensions before importing the PNG."))
        if hfield_open:
            _changed, self._hfield_import_size = _hfield_size_editor(
                ctx, "hfield-import-size", self._hfield_import_size, self.presentation
            )

    def _material_creation(self, ctx, editable):
        material_open = self._section(f"{ctx.tr('New material')}##new-material")
        self._tooltip(ctx.tr("Creates an unbound material asset for later assignment."))
        if material_open:
            if self._begin_properties(ctx, "new_material"):
                self._property(ctx.tr("name"))
                imgui.set_next_item_width(-1.0)
                _changed, self._material_create_name = imgui.input_text(
                    "##new-material-name", self._material_create_name
                )
                self._end_properties()
            material_name = self._material_create_name.strip()
            if not material_name or not editable:
                imgui.begin_disabled()
            if imgui.button(ctx.tr("Create Material")):
                result = ctx.submit(cmd.CreateModelMaterial(self._model_id, material_name))
                if result.ok:
                    self._selected = (self._model_id, "material", material_name)
                    self._selection_key = None
                    self._material_create_name = unique_asset_name(
                        "material", ctx.session.model_assets(self._model_id), "material"
                    )
            if not material_name or not editable:
                imgui.end_disabled()

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
        opened = (
            self.presentation.begin_table("asset_details")
            if self.presentation
            else begin_kv_table("asset_details")
        )
        if opened:
            display = self.presentation.labeled if self.presentation else labeled
            display(ctx.tr("source"), item.file or ctx.tr("inline"))
            display(ctx.tr("references"), str(len(item.references)))
            for field in item.fields:
                if item.type == "hfield" and field.name == "elevation":
                    display(
                        ctx.tr("elevation"),
                        f"{len(field.value.split())} {ctx.tr('inline samples')}",
                    )
                elif field.value:
                    display(field.name, field.value)
            self._end_properties()
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
        if imgui.button(action_labels[action_index]):
            result = ctx.submit(cmd.RenameModelAsset(item.model_id, item.type, item.name, rename))
            if result.ok:
                self._selected = (item.model_id, item.type, rename)
                self._selection_key = None
        if not rename or rename == item.name:
            imgui.end_disabled()
        action_index += 1
        if inline[action_index]:
            imgui.same_line()
        if imgui.button(action_labels[action_index]):
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
                imgui.button(action_labels[action_index])
                and ctx.request_model_asset_replace is not None
            ):
                ctx.request_model_asset_replace(item.model_id, item.type, item.name)
            action_index += 1
        if item.references:
            imgui.begin_disabled()
        if inline[action_index]:
            imgui.same_line()
        if imgui.button(action_labels[action_index]):
            ctx.submit(cmd.RemoveModelAsset(item.model_id, item.type, item.name))
        if item.references:
            imgui.end_disabled()
            self._tooltip(ctx.tr("Remove or replace every reference before deleting this asset"))

        if item.type == "hfield":
            self._hfield_controls(ctx, item)
        if item.type == "material":
            self._material_controls(ctx, item)
        if item.type in ("mesh", "hfield", "material"):
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
        if self.presentation:
            if not self.presentation.section(
                ctx.tr("height-field dimensions"), imgui.TreeNodeFlags_.default_open
            ):
                return
        else:
            imgui.separator()
            imgui.text_disabled(ctx.tr("height-field dimensions"))
        _changed, self._hfield_size = _hfield_size_editor(
            ctx, "hfield-size", self._hfield_size, self.presentation
        )
        dirty = not np.allclose(self._hfield_size, self._hfield_source_size)
        if not dirty:
            imgui.begin_disabled()
        if imgui.button(ctx.tr("Apply Dimensions")):
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

        if self.presentation:
            if not self.presentation.section(
                ctx.tr("material appearance"), imgui.TreeNodeFlags_.default_open
            ):
                return
        else:
            imgui.separator()
            imgui.text_disabled(ctx.tr("material appearance"))
        if self._begin_properties(ctx, "asset_material"):
            self._property(ctx.tr("base color"))
            imgui.set_next_item_width(-1.0)
            item_changed, rgba = self._color("##asset-material-rgba", rgba)
            changed |= item_changed

            scalar_fields = (
                ("emission", emission, 10.0),
                ("specular", specular, 1.0),
                ("shininess", shininess, 1.0),
                ("reflectance", reflectance, 1.0),
            )
            edited_scalars: list[float] = []
            for label, value, maximum in scalar_fields:
                self._property(ctx.tr(label))
                imgui.set_next_item_width(-1.0)
                item_changed, value = self._numeric(
                    f"##asset-material-{label}", value, 0.01, 0.0, maximum, "%.2f"
                )
                changed |= item_changed
                edited_scalars.append(float(value))
            emission, specular, shininess, reflectance = edited_scalars

            for label, value, default in (
                ("metallic", metallic, 0.0),
                ("roughness", roughness, 0.5),
            ):
                self._property(ctx.tr(label))
                enabled = value >= 0.0
                item_changed, enabled = self._checkbox(f"##asset-material-{label}-enabled", enabled)
                if item_changed:
                    value = default if enabled else -1.0
                    changed = True
                self._tooltip(
                    ctx.tr("Toggle the MuJoCo PBR override; disabled uses classic material shading")
                )
                imgui.same_line()
                if not enabled:
                    imgui.begin_disabled()
                imgui.set_next_item_width(-1.0)
                item_changed, value = self._numeric(
                    f"##asset-material-{label}", value, 0.01, 0.0, 1.0, "%.2f"
                )
                changed |= item_changed
                if not enabled:
                    imgui.end_disabled()
                if label == "metallic":
                    metallic = float(value)
                else:
                    roughness = float(value)

            self._property(ctx.tr("texture"), control_part="select")
            imgui.set_next_item_width(-1.0)
            textures = (None, *ctx.session.model_texture_names(item.model_id))
            slot = textures.index(texture) if texture in textures else -1
            texture_changed, selected = self._combo(
                "##asset-material-texture",
                slot,
                tuple(candidate or ctx.tr("none") for candidate in textures),
                preview=texture or ctx.tr("none"),
            )
            if texture_changed:
                texture = textures[selected]
                changed = True

            self._property(ctx.tr("texture repeat"))
            imgui.set_next_item_width(-1.0)
            item_changed, tex_repeat = self._numbers(
                "##asset-material-repeat", tex_repeat, 0.05, 0.01, 1000.0, "%.2f"
            )
            changed |= item_changed

            self._property(ctx.tr("uniform scale"))
            item_changed, tex_uniform = self._checkbox("##asset-material-uniform", tex_uniform)
            changed |= item_changed
            self._end_properties()

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
        allowed = (NodeType.GEOM, NodeType.SITE) if item.type == "material" else (NodeType.GEOM,)
        if node is None or node.type not in allowed:
            imgui.text_disabled(ctx.tr("Select a geometry to assign this asset"))
            return
        if item.type == "material":
            if node.model_id != item.model_id:
                imgui.text_disabled(ctx.tr("The selected geometry belongs to a different model"))
                return
            if not node.source_editable or item.runtime_index < 0:
                imgui.text_disabled(
                    ctx.tr("The selected geometry has no editable material binding")
                )
                return
            if imgui.button(ctx.tr("Assign to Selected Geometry")):
                ctx.submit(cmd.SetGeometryMaterial(node.node_id, item.runtime_index))
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

    def _section(self, label, flags=0):
        return (
            self.presentation.section(label, flags)
            if self.presentation
            else imgui.collapsing_header(label, flags)
        )

    def _end_properties(self):
        if self.presentation:
            self.presentation.end_table()
        else:
            imgui.end_table()

    def _begin_properties(self, ctx, table_id):
        if self.presentation:
            return self.presentation.begin_table(table_id)
        if not begin_kv_table(table_id):
            return False
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, 96 * ctx.style_scale)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
        return True

    def _property(self, label, *, control_part="input_number"):
        if self.presentation:
            self.presentation.row(label, control_part=control_part)
        else:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(label)
            imgui.table_next_column()

    def _combo(self, key, index, labels, *, preview=None):
        if self.presentation:
            return self.presentation.combo(key, index, labels, preview=preview)
        if preview is None:
            return imgui.combo(key, index, labels)
        selected = index
        if imgui.begin_combo(key, preview):
            for option, label in enumerate(labels):
                if imgui.selectable(label, option == index)[0]:
                    selected = option
            imgui.end_combo()
        return selected != index, selected

    def _numeric(self, key, value, speed, low, high, fmt):
        return (
            self.presentation.numeric(key, value, speed, low, high, fmt)
            if self.presentation
            else imgui.drag_float(key, value, speed, low, high, fmt)
        )

    def _numbers(self, key, values, speed, low, high, fmt):
        return (
            self.presentation.numbers(key, values, speed, low, high, fmt)
            if self.presentation
            else imgui.drag_float2(key, values, speed, low, high, fmt)
        )

    def _color(self, key, values):
        return (
            self.presentation.color(key, values)
            if self.presentation
            else imgui.color_edit4(key, values)
        )

    def _checkbox(self, key, value):
        return (
            self.presentation.checkbox(key, value)
            if self.presentation
            else imgui.checkbox(key, value)
        )


__all__ = [
    "AssetsPanel",
    "filter_assets",
    "height_field_preview_color",
    "unique_asset_name",
]
