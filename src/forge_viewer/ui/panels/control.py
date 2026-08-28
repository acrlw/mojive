"""Simulation playback and speed controls."""

from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from . import (
    Panel,
    PanelContext,
    begin_kv_table,
    button_width,
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

        if s.keyframes:
            imgui.separator()
            selected = s.active_keyframe if s.active_keyframe >= 0 else 0
            imgui.begin_disabled(not paused)
            available = imgui.get_content_region_avail().x
            spacing = imgui.get_style().item_spacing.x
            load_width = button_width("Load", 60.0 * ctx.style_scale)
            slider_min = 80.0 * ctx.style_scale
            inline = available >= slider_min + spacing + load_width
            imgui.set_next_item_width(available - spacing - load_width if inline else -1.0)
            changed, selected = imgui.slider_int(
                "##keyframe", selected, 0, len(s.keyframes) - 1, "%d"
            )
            if changed:
                ctx.submit(cmd.LoadKeyframe(selected))
            if inline:
                imgui.same_line()
            if imgui.button("Load", imgui.ImVec2(load_width, 0)):
                ctx.submit(cmd.LoadKeyframe(selected))
            imgui.end_disabled()
            if not paused:
                imgui.set_item_tooltip("physics is running; pause to load a keyframe")
            key = s.keyframes[selected]
            imgui.text_disabled(
                f"{key.name} · {selected + 1}/{len(s.keyframes)} · t={key.time:g} s"
            )

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
            labeled("state", ("paused" if paused else "running") if caps.simulation else "static")
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
