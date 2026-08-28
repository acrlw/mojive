"""Model-local MuJoCo keyframe capture and metadata editing."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, KeyframeProperties
from . import Panel, PanelContext, begin_kv_table


def unique_keyframe_name(existing: set[str]) -> str:
    index = 1
    name = f"key{index}"
    while name in existing:
        index += 1
        name = f"key{index}"
    return name


class KeyframesPanel(Panel):
    """Capture and recall model states without exposing raw state arrays."""

    name = "Keyframes"
    default_open = False
    shortcut = ""
    dock_with = "Output"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = -1
        self._selected_id = -1
        self._selection_generation = -1
        self._properties: KeyframeProperties | None = None
        self._name = ""
        self._time = 0.0
        self._error = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        models = tuple(ctx.session.scene_models)
        if not models:
            imgui.text_disabled("no editable model keyframes")
            return

        model_ids = tuple(model.model_id for model in models)
        selected = ctx.session.selected_node
        if self._model_id not in model_ids:
            preferred = selected.model_id if selected is not None else model_ids[0]
            self._model_id = preferred if preferred in model_ids else model_ids[0]
            self._clear_selection()

        if begin_kv_table("keyframe_model"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("model")
            imgui.table_next_column()
            slot = model_ids.index(self._model_id)
            imgui.set_next_item_width(-1.0)
            changed, slot = imgui.combo(
                "##keyframe-model", slot, tuple(model.name for model in models)
            )
            if changed:
                self._model_id = model_ids[slot]
                self._clear_selection()
            imgui.end_table()

        keyframes = tuple(key for key in ctx.session.keyframes if key.model_id == self._model_id)
        editable = bool(ctx.session.paused)
        if not editable:
            imgui.begin_disabled()
        if imgui.button("Capture Current State"):
            name = unique_keyframe_name({key.name for key in keyframes})
            result = ctx.submit(cmd.AddModelKeyframe(self._model_id, name))
            if result.ok:
                self._selected_id = result.entity_id
                self._selection_generation = -1
                self._error = ""
            else:
                self._error = result.message
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip("Pause the simulation before capturing a keyframe")
        imgui.same_line()
        imgui.text_disabled(f"{len(keyframes)} saved state(s)")

        imgui.separator()
        table_flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
        )
        if imgui.begin_table("keyframe_list", 3, table_flags):
            imgui.table_setup_column("name")
            imgui.table_setup_column("time", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("state", imgui.TableColumnFlags_.width_fixed)
            imgui.table_headers_row()
            for keyframe in keyframes:
                imgui.table_next_row()
                imgui.table_next_column()
                label = keyframe.name or f"key {keyframe.keyframe_id}"
                selected_row, _ = imgui.selectable(
                    f"{label}##keyframe-{keyframe.keyframe_id}",
                    self._selected_id == keyframe.keyframe_id,
                    imgui.SelectableFlags_.span_all_columns,
                )
                if selected_row:
                    self._selected_id = keyframe.keyframe_id
                    self._selection_generation = -1
                imgui.table_next_column()
                imgui.text(f"{keyframe.time:g} s")
                imgui.table_next_column()
                if keyframe.keyframe_id == ctx.session.active_keyframe:
                    imgui.text_colored(imgui.ImVec4(*ctx.theme.success), "loaded")
                else:
                    imgui.text_disabled("saved")
            imgui.end_table()

        if not keyframes:
            imgui.text_disabled("Capture the current state to create the first keyframe.")
            self._clear_selection()
        elif self._selected_id not in {key.keyframe_id for key in keyframes}:
            self._selected_id = keyframes[0].keyframe_id
            self._selection_generation = -1

        self._draw_selected(ctx, editable)

    def _draw_selected(self, ctx: PanelContext, editable: bool) -> None:
        if self._selected_id < 0:
            return
        generation = ctx.session.structure_generation
        if self._selection_generation != generation:
            self._selection_generation = generation
            self._properties = ctx.session.keyframe_properties(self._selected_id)
            if self._properties is not None:
                self._name = self._properties.name
                self._time = self._properties.time
        properties = self._properties
        if properties is None:
            return

        imgui.separator()
        imgui.text_disabled("selected keyframe")
        if begin_kv_table("keyframe_properties"):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("name")
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._name = imgui.input_text("##keyframe-name", self._name)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled("time")
            imgui.table_next_column()
            imgui.set_next_item_width(-1.0)
            _changed, self._time = imgui.input_double(
                "##keyframe-time", self._time, 0.0, 0.0, "%.9g"
            )
            imgui.end_table()

        dirty = self._name.strip() != properties.name or self._time != properties.time
        if not editable or not dirty or not self._name.strip():
            imgui.begin_disabled()
        if imgui.button("Apply Metadata"):
            result = ctx.submit(
                cmd.SetModelKeyframe(
                    properties.keyframe_id,
                    properties.model_id,
                    self._name.strip(),
                    float(self._time),
                    properties.qpos,
                    properties.qvel,
                    properties.act,
                    properties.ctrl,
                    properties.mocap_position,
                    properties.mocap_quaternion,
                )
            )
            if result.ok:
                self._selection_generation = -1
                self._error = ""
            else:
                self._error = result.message
        if not editable or not dirty or not self._name.strip():
            imgui.end_disabled()
        imgui.same_line()
        if not editable:
            imgui.begin_disabled()
        if imgui.button("Load"):
            result = ctx.submit(cmd.LoadKeyframe(properties.keyframe_id))
            self._error = "" if result.ok else result.message
        imgui.same_line()
        if imgui.button("Delete"):
            result = ctx.submit(cmd.RemoveModelKeyframe(properties.keyframe_id))
            if result.ok:
                self._clear_selection()
            else:
                self._error = result.message
        if not editable:
            imgui.end_disabled()
            imgui.set_item_tooltip("Pause the simulation before editing keyframes")

        imgui.text_disabled(
            "A MuJoCo keyframe stores the complete model state; raw arrays remain in MJCF source."
        )
        if self._error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.danger), self._error)
            if imgui.small_button("Copy error##keyframes"):
                imgui.set_clipboard_text(self._error)

    def _clear_selection(self) -> None:
        self._selected_id = -1
        self._selection_generation = -1
        self._properties = None
        self._name = ""
        self._time = 0.0
        self._error = ""


__all__ = ["KeyframesPanel", "unique_keyframe_name"]
