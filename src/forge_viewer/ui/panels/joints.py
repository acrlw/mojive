"""MuJoCo joint controls and range display."""

from __future__ import annotations

import math

import numpy as np
from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import ActuatorInfo, FrameNeeds, JointInfo
from . import Panel, PanelContext, value_slider

_SCALAR_KINDS = ("hinge", "slide")
_BROWSE_THRESHOLD = 256
_PAGE_SIZE = 128


class JointsPanel(Panel):
    name = "Joints"
    default_open = True
    shortcut = "F5"

    def __init__(self) -> None:
        super().__init__()
        self._initial_qpos = np.zeros(0, np.float64)
        self._initial_ctrl = np.zeros(0, np.float64)
        self._snapshot_generation = -1
        self._joint_page = 0
        self._actuator_page = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(poses=False, qpos=True, actuator=True)

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        frame = s.frame
        self._snapshot(ctx)

        caps = s.adapter.caps
        if not caps.write_qpos:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning), f"{caps.name} cannot write joint positions"
            )
        elif not s.paused:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                "Pause simulation to edit joint positions",
            )

        if imgui.collapsing_header("joints", imgui.TreeNodeFlags_.default_open):
            if frame.qpos is None:
                imgui.text_disabled("qpos not produced this frame")
            elif not s.joints:
                imgui.text_disabled("no joints")
            else:
                imgui.begin_disabled(not caps.write_qpos or not s.paused)
                selected = s.selected_node
                selected_joints = (
                    s.joints_for_body(selected.body_index) if selected is not None else ()
                )
                if selected_joints:
                    imgui.text_disabled(f"{selected.name} · direct joints")
                    for joint in selected_joints:
                        self._joint_row(ctx, joint)
                    if len(s.joints) <= _BROWSE_THRESHOLD and len(selected_joints) != len(s.joints):
                        imgui.separator()
                        imgui.text_disabled("all joints")
                if len(s.joints) > _BROWSE_THRESHOLD:
                    if not selected_joints:
                        imgui.text_disabled("select a link to show its direct joints")
                    self._browse_joints(ctx)
                else:
                    selected_ids = {joint.joint_id for joint in selected_joints}
                    for joint in s.joints:
                        if joint.joint_id not in selected_ids:
                            self._joint_row(ctx, joint)
                imgui.end_disabled()

        if imgui.collapsing_header("actuators", imgui.TreeNodeFlags_.default_open):
            if not s.actuators:
                imgui.text_disabled("no actuators")
            elif frame.ctrl is None:
                imgui.text_disabled("ctrl not produced this frame")
            elif len(s.actuators) > _BROWSE_THRESHOLD:
                self._browse_actuators(ctx)
            else:
                for a in s.actuators:
                    self._actuator_row(ctx, a)

    def _browse_joints(self, ctx: PanelContext) -> None:
        joints = ctx.session.joints
        imgui.text_disabled(f"{len(joints)} joints total")
        if not imgui.collapsing_header("browse all joints"):
            return
        self._joint_page, start, stop = _page_controls("joints", len(joints), self._joint_page)
        for joint in joints[start:stop]:
            self._joint_row(ctx, joint)

    def _browse_actuators(self, ctx: PanelContext) -> None:
        actuators = ctx.session.actuators
        imgui.text_disabled(f"{len(actuators)} actuators total")
        if not imgui.collapsing_header("browse all actuators"):
            return
        self._actuator_page, start, stop = _page_controls(
            "actuators", len(actuators), self._actuator_page
        )
        for actuator in actuators[start:stop]:
            self._actuator_row(ctx, actuator)

    def _joint_row(self, ctx: PanelContext, j: JointInfo) -> None:
        qpos = ctx.session.frame.qpos
        if qpos is None or j.qpos_adr >= len(qpos):
            return
        if j.type not in _SCALAR_KINDS:
            imgui.text_disabled(f"{j.name or f'joint{j.joint_id}'}  ({j.type}, {j.dof} dof)")
            return

        value = float(qpos[j.qpos_adr])
        lo, hi = _joint_range(j)
        label = f"{j.name or f'joint{j.joint_id}'}##q{j.qpos_adr}"
        edit = value_slider(
            label,
            value,
            lo,
            hi,
            initial=_initial_value(self._initial_qpos, j.qpos_adr, value),
            fmt="%.4f",
            more_hint="show actuator gain",
        )
        if edit.changed:
            ctx.submit(cmd.SetQpos(j.qpos_adr, edit.value))
        if edit.expanded:
            self._drive_detail(ctx, j)

    def _drive_detail(self, ctx: PanelContext, j: JointInfo) -> None:
        drivers = ctx.session.actuators_for_joint(j.joint_id)
        imgui.indent()
        if not drivers:
            imgui.text_disabled("no actuator drives this joint")
        for a in drivers:
            imgui.text_disabled(f"{a.name or f'act{a.actuator_id}'}  gain {a.gain:g}")

            self._actuator_row(ctx, a, prefix="  ctrl ")
        imgui.unindent()

    def _actuator_row(self, ctx: PanelContext, a: ActuatorInfo, prefix: str = "") -> None:
        ctrl = ctx.session.frame.ctrl
        if ctrl is None:
            return
        lo, hi = a.ctrl_range if a.ctrl_limited else (-1.0, 1.0)
        if hi <= lo:
            lo, hi = -1.0, 1.0
        name = a.name or f"act{a.actuator_id}"
        for component in range(a.ctrl_count):
            address = a.ctrl_address + component
            if address >= len(ctrl):
                continue
            suffix = f"[{component}]" if a.ctrl_count > 1 else ""
            edit = value_slider(
                f"{prefix}{name}{suffix}##c{address}",
                float(ctrl[address]),
                lo,
                hi,
                initial=_initial_value(self._initial_ctrl, address, float(ctrl[address])),
                fmt="%.4f",
                more_hint="none",
            )
            if edit.changed:
                ctx.submit(cmd.SetCtrl(address, edit.value))

    def _snapshot(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        frame = ctx.session.frame
        if gen != self._snapshot_generation:
            self._snapshot_generation = gen
            self._initial_qpos = np.zeros(0, np.float64)
            self._initial_ctrl = np.zeros(0, np.float64)
            self._joint_page = 0
            self._actuator_page = 0
        if frame.qpos is not None and len(self._initial_qpos) != len(frame.qpos):
            self._initial_qpos = np.asarray(frame.qpos, np.float64).copy()
        if frame.ctrl is not None and len(self._initial_ctrl) != len(frame.ctrl):
            self._initial_ctrl = np.asarray(frame.ctrl, np.float64).copy()


def _page_controls(label: str, count: int, page: int) -> tuple[int, int, int]:
    page, pages, start, stop = page_span(count, page)
    if pages <= 1:
        return page, start, stop
    if imgui.button(f"Previous##{label}"):
        page -= 1
    imgui.same_line()
    if imgui.button(f"Next##{label}"):
        page += 1
    page, pages, start, stop = page_span(count, page)
    imgui.same_line()
    imgui.text_disabled(f"page {page + 1}/{pages} · {start + 1}-{stop}")
    return page, start, stop


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
