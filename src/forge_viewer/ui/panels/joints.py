"""MuJoCo joint controls and range display."""

from __future__ import annotations

import math

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import ActuatorInfo, FrameNeeds, JointInfo
from . import Panel, PanelContext, value_slider

_SCALAR_KINDS = ("hinge", "slide")


class JointsPanel(Panel):
    name = "Joints"
    default_open = True
    shortcut = "F5"

    def __init__(self) -> None:
        super().__init__()
        self._initial_qpos: dict[int, float] = {}
        self._initial_ctrl: dict[int, float] = {}
        self._snapshot_generation = -1

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

        if imgui.collapsing_header("joints", imgui.TreeNodeFlags_.default_open):
            if frame.qpos is None:
                imgui.text_disabled("qpos not produced this frame")
            elif not s.joints:
                imgui.text_disabled("no joints")
            else:
                imgui.begin_disabled(not caps.write_qpos)
                for j in s.joints:
                    self._joint_row(ctx, j)
                imgui.end_disabled()

        if imgui.collapsing_header("actuators", imgui.TreeNodeFlags_.default_open):
            if not s.actuators:
                imgui.text_disabled("no actuators")
            elif frame.ctrl is None:
                imgui.text_disabled("ctrl not produced this frame")
            else:
                for a in s.actuators:
                    self._actuator_row(ctx, a)

    def _joint_row(self, ctx: PanelContext, j: JointInfo) -> None:
        qpos = ctx.session.frame.qpos
        if qpos is None or j.qpos_adr >= len(qpos):
            return
        if j.kind not in _SCALAR_KINDS:
            imgui.text_disabled(f"{j.name or f'joint{j.joint_id}'}  ({j.kind}, {j.dof} dof)")
            return

        value = float(qpos[j.qpos_adr])
        lo, hi = _joint_range(j)
        label = f"{j.name or f'joint{j.joint_id}'}##q{j.qpos_adr}"
        edit = value_slider(
            label,
            value,
            lo,
            hi,
            initial=self._initial_qpos.get(j.qpos_adr, 0.0),
            fmt="%.4f",
            more_hint="show actuator gain",
        )
        if edit.changed:
            ctx.submit(cmd.SetQpos(j.qpos_adr, edit.value))
        if edit.expanded:
            self._drive_detail(ctx, j)

    def _drive_detail(self, ctx: PanelContext, j: JointInfo) -> None:
        drivers = [a for a in ctx.session.actuators if a.joint == j.joint_id]
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
                initial=self._initial_ctrl.get(address, 0.0),
                fmt="%.4f",
                more_hint="none",
            )
            if edit.changed:
                ctx.submit(cmd.SetCtrl(address, edit.value))

    def _snapshot(self, ctx: PanelContext) -> None:
        gen = ctx.session.structure_generation
        frame = ctx.session.frame
        if gen != self._snapshot_generation:
            self._initial_qpos.clear()
            self._initial_ctrl.clear()
        if frame.qpos is not None and not self._initial_qpos:
            self._initial_qpos = {
                j.qpos_adr: float(frame.qpos[j.qpos_adr])
                for j in ctx.session.joints
                if j.qpos_adr < len(frame.qpos)
            }
            self._snapshot_generation = gen
        if frame.ctrl is not None and not self._initial_ctrl:
            self._initial_ctrl = {
                address: float(frame.ctrl[address])
                for a in ctx.session.actuators
                for address in range(a.ctrl_address, a.ctrl_address + a.ctrl_count)
                if address < len(frame.ctrl)
            }


def _joint_range(j: JointInfo) -> tuple[float, float]:
    if j.limited and j.range[1] > j.range[0]:
        return float(j.range[0]), float(j.range[1])
    return (-math.pi, math.pi) if j.kind == "hinge" else (-1.0, 1.0)
