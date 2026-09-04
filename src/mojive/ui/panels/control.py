"""Actuator and equality-constraint controls."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import ActuatorInfo, FrameNeeds
from . import (
    Panel,
    PanelContext,
    copy_state_vector,
    copyable_name_item,
    searchable_ordered_list_header,
    themed_checkbox,
    value_slider,
)


class ControlPanel(Panel):
    id = "control"
    name = "Control"
    default_open = True
    shortcut = ""
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._initial_ctrl = np.zeros(0, np.float64)
        self._snapshot_generation = -1
        self._search = ""
        self._sort_by_name = False
        self._row_cache_key: tuple[int, str, bool] | None = None
        self._row_cache: tuple[tuple[ActuatorInfo, int], ...] = ()

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
        rows = self._row_cache
        if not rows:
            imgui.text_disabled(ctx.tr("No matching actuators"))
            return
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("control_actuators", 2, flags):
            return
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.36)
        imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch, 0.64)
        clipper = imgui.ListClipper()
        clipper.begin(len(rows))
        while clipper.step():
            for index in range(clipper.display_start, clipper.display_end):
                actuator, component = rows[index]
                self._actuator_row(ctx, actuator, component)
        clipper.end()
        imgui.end_table()

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
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        label_width = max(1.0, imgui.get_content_region_avail().x)
        imgui.text_disabled(f"{name}{suffix}")
        copyable_name_item(ctx, f"{name}{suffix}", label_width)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        value = float(ctrl[address])
        initial = float(self._initial_ctrl[address]) if address < len(self._initial_ctrl) else value
        edit = value_slider(
            f"##control-actuator-{address}",
            value,
            lo,
            hi,
            initial=initial,
            fmt="%+.3f",
            more_hint="",
        )
        if edit.changed:
            ctx.submit(cmd.SetCtrl(address, edit.value))

    @staticmethod
    def _equality(ctx: PanelContext) -> None:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
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
