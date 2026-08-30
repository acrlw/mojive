"""MuJoCo joint controls and range display."""

from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds, JointInfo
from . import Panel, PanelContext, search_input, value_slider

_SCALAR_KINDS = ("hinge", "slide")
_BROWSE_THRESHOLD = 256
_PAGE_SIZE = 128


class JointsPanel(Panel):
    name = "Joints"
    default_open = True
    shortcut = "F5"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._initial_qpos = np.zeros(0, np.float64)
        self._snapshot_generation = -1
        self._joint_page = 0
        self._search = ""

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, qpos=True)

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        frame = s.frame
        self._snapshot(ctx)

        caps = s.adapter.caps
        if not caps.write_qpos:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                f"{caps.name} {ctx.tr('cannot write joint positions')}",
            )
        elif not s.paused:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                ctx.tr("Pause simulation to edit joint positions"),
            )

        if frame.qpos is None:
            imgui.text_disabled(ctx.tr("qpos not produced this frame"))
            return
        if not s.joints:
            imgui.text_disabled(ctx.tr("no joints"))
            return

        imgui.set_next_item_width(-1.0)
        changed, self._search = search_input(
            "##joint_search",
            self._search,
            hint=ctx.tr("Search joints"),
            search_tooltip=ctx.tr("Search joints"),
            clear_tooltip=ctx.tr("Clear search"),
        )
        if changed:
            self._joint_page = 0

        filtered = filter_joints(s.joints, self._search)
        if not filtered:
            imgui.text_disabled(ctx.tr("No matching joints"))
            return

        selected = s.selected_node
        selected_ids = {
            joint.joint_id
            for joint in (s.joints_for_body(selected.body_index) if selected is not None else ())
        }
        ordered = tuple(joint for joint in filtered if joint.joint_id in selected_ids) + tuple(
            joint for joint in filtered if joint.joint_id not in selected_ids
        )
        if selected_ids and any(joint.joint_id in selected_ids for joint in filtered):
            imgui.text_disabled(f"{ctx.tr('Selected link')} · {selected.name}")
        if len(ordered) > _BROWSE_THRESHOLD:
            imgui.text_disabled(f"{len(ordered)} {ctx.tr('joints total')}")
            self._joint_page, start, stop = _page_controls(
                ctx, "joints", len(ordered), self._joint_page
            )
            ordered = ordered[start:stop]

        imgui.begin_disabled(not caps.write_qpos or not s.paused)
        self._joint_table(ctx, ordered)
        imgui.end_disabled()

    def _joint_table(self, ctx: PanelContext, joints) -> None:
        flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
        if not imgui.begin_table("joint_values", 2, flags):
            return
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.36)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch, 0.64)
        for joint in joints:
            self._joint_row(ctx, joint)
        imgui.end_table()

    def _joint_row(self, ctx: PanelContext, j: JointInfo) -> None:
        qpos = ctx.session.frame.qpos
        if qpos is None or j.qpos_adr >= len(qpos):
            return
        name = j.name or f"joint{j.joint_id}"
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(name)
        direct_joints = ctx.session.joints_for_body(j.body)
        if ctx.gizmo is not None and len(direct_joints) > 1 and j.type in (*_SCALAR_KINDS, "ball"):
            selected = ctx.gizmo.selected_joint_id(j.body) == j.joint_id
            label = ctx.tr("Using gizmo") if selected else ctx.tr("Use gizmo")
            imgui.same_line()
            if imgui.small_button(f"{label}##joint-gizmo-{j.joint_id}"):
                ctx.gizmo.select_joint(j.body, j.joint_id)
            imgui.set_item_tooltip(ctx.tr("Choose which direct joint the viewport gizmo edits"))
        imgui.table_next_column()
        if j.type not in _SCALAR_KINDS:
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(f"{ctx.tr(f'{j.type.title()} joint')} · {j.dof} {ctx.tr('dof')}")
            return

        value = float(qpos[j.qpos_adr])
        lo, hi = _joint_range(j)
        imgui.set_next_item_width(-1.0)
        edit = value_slider(
            f"##joint-qpos-{j.qpos_adr}",
            value,
            lo,
            hi,
            initial=_initial_value(self._initial_qpos, j.qpos_adr, value),
            fmt="%.4f",
            more_hint="",
        )
        if edit.changed:
            ctx.submit(cmd.SetQpos(j.qpos_adr, edit.value))

    def _snapshot(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        frame = ctx.session.frame
        if gen != self._snapshot_generation:
            self._snapshot_generation = gen
            self._initial_qpos = np.zeros(0, np.float64)
            self._joint_page = 0
            self._search = ""
        if frame.qpos is not None and len(self._initial_qpos) != len(frame.qpos):
            self._initial_qpos = np.asarray(frame.qpos, np.float64).copy()


def _page_controls(ctx: PanelContext, label: str, count: int, page: int) -> tuple[int, int, int]:
    page, pages, start, stop = page_span(count, page)
    if pages <= 1:
        return page, start, stop
    if imgui.button(f"{ctx.tr('Previous')}##{label}"):
        page -= 1
    imgui.same_line()
    if imgui.button(f"{ctx.tr('Next')}##{label}"):
        page += 1
    page, pages, start, stop = page_span(count, page)
    imgui.same_line()
    imgui.text_disabled(f"{ctx.tr('page')} {page + 1}/{pages} · {start + 1}-{stop}")
    return page, start, stop


def filter_joints(joints, query: str) -> tuple[JointInfo, ...]:
    """Return joints whose stable display name contains ``query``."""

    needle = str(query).strip().casefold()
    if not needle:
        return tuple(joints)
    return tuple(
        joint for joint in joints if needle in (joint.name or f"joint{joint.joint_id}").casefold()
    )


def page_span(count: int, page: int, page_size: int = _PAGE_SIZE) -> tuple[int, int, int, int]:
    """Return a bounded page and half-open item span for a large editor list."""
    size = max(1, int(page_size))
    total = max(0, int(count))
    pages = max(1, (total + size - 1) // size)
    current = min(max(0, int(page)), pages - 1)
    start = current * size
    return current, pages, start, min(total, start + size)


def _initial_value(values: np.ndarray, index: int, fallback: float) -> float:
    return float(values[index]) if 0 <= index < len(values) else float(fallback)


def _joint_range(j: JointInfo) -> tuple[float, float]:
    if j.limited and j.range[1] > j.range[0]:
        return float(j.range[0]), float(j.range[1])
    return (-math.pi, math.pi) if j.type == "hinge" else (-1.0, 1.0)
