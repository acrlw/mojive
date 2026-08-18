"""Scrolling telemetry plot panel."""

from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext

CAPACITY = 512


class Ring:
    __slots__ = ("data", "filled", "index")

    def __init__(self, capacity: int = CAPACITY) -> None:
        self.data = np.zeros(capacity, np.float32)
        self.index = 0
        self.filled = 0

    def push(self, value: float) -> None:
        self.data[self.index] = value
        self.index = (self.index + 1) % len(self.data)
        self.filled = min(self.filled + 1, len(self.data))

    def clear(self) -> None:
        self.data[:] = 0.0
        self.index = 0
        self.filled = 0

    @property
    def offset(self) -> int:
        return self.index if self.filled == len(self.data) else 0

    def span(self) -> tuple[float, float]:
        if self.filled == 0:
            return -1.0, 1.0
        view = self.data if self.filled == len(self.data) else self.data[: self.filled]
        lo, hi = float(view.min()), float(view.max())
        if hi - lo < 1e-6:
            mid = (lo + hi) * 0.5
            return mid - 0.5, mid + 0.5
        pad = (hi - lo) * 0.08
        return lo - pad, hi + pad


class PlotPanel(Panel):
    name = "Plot"
    default_open = False
    shortcut = "F7"

    def __init__(self) -> None:
        super().__init__()
        self.show_angle = True
        self.show_velocity = True
        self.show_contact = False

        self.joint_index = 0
        self._angle = Ring()
        self._velocity = Ring()
        self._contact = Ring()
        self._tracked = -1

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds(
            poses=False,
            qpos=self.show_angle,
            qvel=self.show_velocity,
            contacts=self.show_contact,
        )

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session
        joints = s.joints
        if not joints:
            imgui.text_disabled("no joints to plot")
            return

        self.joint_index = max(0, min(self.joint_index, len(joints) - 1))
        names = [j.name or f"joint{j.joint_id}" for j in joints]
        imgui.set_next_item_width(-1)
        changed, self.joint_index = imgui.combo("##joint", self.joint_index, names)
        if changed or self._tracked != self.joint_index:
            self._angle.clear()
            self._velocity.clear()
            self._tracked = self.joint_index

        _c, self.show_angle = imgui.checkbox("angle", self.show_angle)
        imgui.same_line()
        _c, self.show_velocity = imgui.checkbox("velocity", self.show_velocity)
        imgui.same_line()
        _c, self.show_contact = imgui.checkbox("contact force", self.show_contact)

        self._sample(ctx)

        height = 90.0 * ctx.style_scale
        j = joints[self.joint_index]
        if self.show_angle:
            self._curve("qpos", self._angle, height, f"{names[self.joint_index]} [{j.qpos_adr}]")
        if self.show_velocity:
            self._curve("qvel", self._velocity, height, f"{names[self.joint_index]} [{j.qvel_adr}]")
        if self.show_contact:
            self._curve("contact", self._contact, height, "sum |force|")

    def _sample(self, ctx: PanelContext) -> None:
        frame = ctx.session.frame
        joints = ctx.session.joints
        j = joints[self.joint_index]

        if self.show_angle and frame.qpos is not None and j.qpos_adr < len(frame.qpos):
            self._angle.push(float(frame.qpos[j.qpos_adr]))
        if self.show_velocity and frame.qvel is not None and j.qvel_adr < len(frame.qvel):
            self._velocity.push(float(frame.qvel[j.qvel_adr]))
        if self.show_contact:
            c = frame.contacts
            self._contact.push(float(np.abs(c[:, 6]).sum()) if c is not None and len(c) else 0.0)

    def _curve(self, label: str, ring: Ring, height: float, overlay: str) -> None:
        lo, hi = ring.span()
        imgui.plot_lines(
            f"##{label}",
            ring.data,
            values_offset=ring.offset,
            overlay_text=f"{label}  {overlay}",
            scale_min=lo,
            scale_max=hi,
            graph_size=imgui.ImVec2(-1, height),
        )
