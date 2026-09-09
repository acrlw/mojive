"""Actuator and equality-constraint controls."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import ActuatorInfo, FrameNeeds, NodeType
from . import (
    Panel,
    PanelContext,
    activate_edit_gizmo,
    copy_state_vector,
    copyable_name_item,
    publish_focus_item_hint,
    searchable_ordered_list_header,
    themed_checkbox,
)
from .value_cards import value_card, value_rail


class ControlPanel(Panel):
    id = "control"
    name = "Control"
    default_open = True
    shortcut = ""
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._angular_degrees = False
        self._initial_ctrl = np.zeros(0, np.float64)
        self._snapshot_generation = -1
        self._search = ""
        self._sort_by_name = False
        self._row_cache_key: tuple[int, str, bool] | None = None
        self._row_cache: tuple[tuple[ActuatorInfo, int], ...] = ()
        self._selected_address = -1
        self._selection_revision = -1
        self._joint_nodes = {}

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, actuator=True)

    def draw(self, ctx: PanelContext) -> None:
        self._snapshot(ctx)
        if imgui.collapsing_header(ctx.tr("actuators"), imgui.TreeNodeFlags_.default_open):
            self._actuators(ctx)
        if ctx.session.equality_constraints and imgui.collapsing_header(
            ctx.tr("equality"), imgui.TreeNodeFlags_.default_open
        ):
            self._equality(ctx)

    def _actuators(self, ctx: PanelContext) -> None:
        session = ctx.session
        if self._selection_revision != session.selection_revision:
            self._selected_address = -1
            self._selection_revision = session.selection_revision
        if not session.actuators:
            imgui.text_disabled(ctx.tr("no actuators"))
            return
        if session.frame.ctrl is None:
            imgui.text_disabled(ctx.tr("ctrl not produced this frame"))
            return
        _changed, self._search, _sort_changed, self._sort_by_name = searchable_ordered_list_header(
            "##actuator_search",
            self._search,
            self._sort_by_name,
            hint=ctx.tr("Search actuators"),
            search_tooltip=ctx.tr("Search actuators"),
            clear_tooltip=ctx.tr("Clear search"),
            state_order="ctrl / act",
            translate=ctx.tr,
            bindings=ctx.input_bindings,
        )
        self._state_copy_buttons(ctx)
        cache_key = (session.structure_generation, self._search, self._sort_by_name)
        if cache_key != self._row_cache_key:
            actuators = sort_actuators(
                filter_actuators(session.actuators, self._search),
                by_name=self._sort_by_name,
            )
            self._row_cache = tuple(
                (actuator, component)
                for actuator in actuators
                for component in range(actuator.ctrl_count)
            )
            self._row_cache_key = cache_key
            self._joint_nodes = {
                n.joint_index: n for n in session.nodes if n.type is NodeType.JOINT
            }
        rows = self._row_cache
        if not rows:
            imgui.text_disabled(ctx.tr("No matching actuators"))
            return
        if any(a.target_node_id >= 0 or a.joint in self._joint_nodes for a, _ in rows):
            publish_focus_item_hint(ctx)
        clipper = imgui.ListClipper()
        clipper.begin(len(rows))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                actuator, component = rows[index]
                self._actuator_row(ctx, actuator, component)
        clipper.end()

    @staticmethod
    def _state_copy_buttons(ctx: PanelContext) -> None:
        frame = ctx.session.frame
        flags = imgui.TableFlags_.sizing_stretch_same | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("actuator_state_copy", 2, flags):
            return
        imgui.table_next_column()
        if imgui.button(ctx.tr("Copy ctrl"), imgui.ImVec2(-1.0, 0.0)):
            copy_state_vector(frame.ctrl)
        imgui.table_next_column()
        imgui.begin_disabled(not ctx.session.adapter.caps.state_snapshots)
        if imgui.button(ctx.tr("Copy act"), imgui.ImVec2(-1.0, 0.0)):
            state = ctx.session.adapter.capture_state()
            copy_state_vector(None if state is None else state.act)
        imgui.end_disabled()
        imgui.end_table()

    def _actuator_row(self, ctx: PanelContext, actuator: ActuatorInfo, component: int) -> None:
        ctrl = ctx.session.frame.ctrl
        if ctrl is None:
            return
        lo, hi = actuator.ctrl_range if actuator.ctrl_limited else (-1.0, 1.0)
        if hi <= lo:
            lo, hi = -1.0, 1.0
        name = actuator.name or f"act{actuator.actuator_id}"
        address = actuator.ctrl_address + component
        if address >= len(ctrl):
            return
        suffix = f"[{component}]" if actuator.ctrl_count > 1 else ""
        with value_card(
            ctx,
            f"##actuator-select-{address}",
            f"{name}{suffix}",
            "",
            selected=self._selected_address == address,
        ) as (clicked, focused):
            target = ctx.session.node(actuator.target_node_id) or self._joint_nodes.get(
                actuator.joint
            )
            if clicked or focused:
                if target is not None:
                    ctx.submit(cmd.SelectNode(target.node_id))
                self._selection_revision = ctx.session.selection_revision
                self._selected_address = address
            if focused and target is not None:
                joint = next(
                    (j for j in ctx.session.joints if j.joint_id == target.joint_index), None
                )
                activate_edit_gizmo(ctx, target, joint)
                if target.type is NodeType.JOINT and ctx.focus_joint is not None:
                    ctx.focus_joint(target.joint_index)
                elif ctx.focus_node is not None:
                    ctx.focus_node(target.node_id)
            value = float(ctrl[address])
            initial = (
                float(self._initial_ctrl[address]) if address < len(self._initial_ctrl) else value
            )
            imgui.begin_disabled(not ctx.session.adapter.caps.write_ctrl)
            edit = value_rail(
                ctx,
                f"##control-actuator-{address}",
                value,
                (lo, hi) if actuator.ctrl_limited and hi > lo else None,
                initial=initial,
                fmt="%+.3f",
                show_reset=False,
                unit=actuator.unit,
                angular_degrees=self._angular_degrees,
                toggle_unit=self._toggle_angle_unit,
            )
            imgui.end_disabled()
            if edit.changed:
                ctx.submit(cmd.SetCtrl(address, edit.value))

    def _toggle_angle_unit(self):
        self._angular_degrees = not self._angular_degrees

    @staticmethod
    def _equality(ctx: PanelContext) -> None:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.pad_outer_x
        if not imgui.begin_table("control_equality", 2, flags):
            return
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 1.0)
        imgui.table_setup_column(
            "enabled",
            imgui.TableColumnFlags_.width_fixed,
            imgui.get_frame_height() + imgui.get_style().cell_padding.x * 2.0,
        )
        for constraint in ctx.session.equality_constraints:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.align_text_to_frame_padding()
            label_width = max(1.0, imgui.get_content_region_avail().x)
            imgui.text(constraint.name)
            imgui.set_item_tooltip(constraint.type)
            copyable_name_item(ctx, constraint.name, label_width)
            imgui.table_next_column()
            changed, enabled = themed_checkbox(
                f"##equality-{constraint.constraint_id}",
                constraint.enabled,
                ctx.theme,
            )
            if changed:
                ctx.submit(cmd.SetEqualityEnabled(constraint.constraint_id, enabled))
        imgui.end_table()

    def _snapshot(self, ctx: PanelContext) -> None:
        generation = ctx.session.structure_generation
        ctrl = ctx.session.frame.ctrl
        if generation != self._snapshot_generation:
            self._snapshot_generation = generation
            self._initial_ctrl = np.zeros(0, np.float64)
            self._search = ""
            self._row_cache_key = None
            self._row_cache = ()
            self._selected_address = -1
        if ctrl is not None and len(self._initial_ctrl) != len(ctrl):
            self._initial_ctrl = np.asarray(ctrl, np.float64).copy()


def filter_actuators(actuators, query: str) -> tuple[ActuatorInfo, ...]:
    """Return actuators whose stable display name contains ``query``."""

    needle = str(query).strip().casefold()
    if not needle:
        return tuple(actuators)
    return tuple(
        actuator
        for actuator in actuators
        if needle in (actuator.name or f"act{actuator.actuator_id}").casefold()
    )


def sort_actuators(actuators, *, by_name: bool) -> tuple[ActuatorInfo, ...]:
    """Order actuators by control/activation address or stable display name."""

    def state_key(actuator: ActuatorInfo) -> tuple[int, int, int]:
        act_address = int(actuator.act_address)
        return int(actuator.ctrl_address), act_address, int(actuator.actuator_id)

    if by_name:
        return tuple(
            sorted(
                actuators,
                key=lambda actuator: (
                    (actuator.name or f"act{actuator.actuator_id}").casefold(),
                    *state_key(actuator),
                ),
            )
        )
    return tuple(sorted(actuators, key=state_key))
