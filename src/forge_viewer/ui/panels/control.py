"""Simulation playback and speed controls."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    labeled,
    value_slider,
)


class ControlPanel(Panel):
    name = "Control"
    default_open = True
    shortcut = ""
    closable = False

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session

        paused = s.paused
        caps = s.adapter.caps
        imgui.begin_disabled(not caps.simulation)
        edit = value_slider("speed", s.speed, 0.05, 8.0, initial=1.0, fmt="x%.2f", more_hint="none")
        if edit.changed:
            ctx.submit(cmd.SetSpeed(edit.value))
        imgui.end_disabled()

        if s.equality_constraints and imgui.collapsing_header("equality constraints"):
            for constraint in s.equality_constraints:
                changed, enabled = imgui.checkbox(
                    f"{constraint.name}##equality-{constraint.constraint_id}",
                    constraint.enabled,
                )
                if changed:
                    ctx.submit(cmd.SetEqualityEnabled(constraint.constraint_id, enabled))
                imgui.same_line()
                imgui.text_disabled(constraint.type)

        frame = s.frame
        if begin_kv_table("control_kv"):
            labeled("sim time", f"{frame.time:.3f} s")
            labeled("steps", str(frame.step))
            state = (
                "recording take"
                if s.state_take_recording
                else "replaying take"
                if s.state_take_playing
                else ("paused" if paused else "running")
                if caps.simulation
                else "static"
            )
            labeled("state", state)
            labeled(
                "timestep",
                f"{s.adapter.timestep() * 1000.0:.2f} ms" if caps.simulation else "—",
            )
            imgui.end_table()

        if s.last_message:
            imgui.separator()
            imgui.text_wrapped(s.last_message)
            if imgui.small_button("Copy message"):
                imgui.set_clipboard_text(s.last_message)
