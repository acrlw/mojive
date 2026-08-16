from __future__ import annotations

from imgui_bundle import imgui

from ... import commands as cmd
from ...adapters.base import FrameNeeds
from . import Panel, PanelContext, begin_kv_table, labeled, value_slider


class ControlPanel(Panel):
    name = "Control"
    default_open = True
    shortcut = "F2"

    def frame_needs(self) -> FrameNeeds:

        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        s = ctx.session

        paused = s.paused
        caps = s.adapter.caps

        imgui.begin_disabled(not caps.simulation)
        if imgui.button("Resume" if paused else "Pause", imgui.ImVec2(78, 0)):
            ctx.submit(cmd.Play() if paused else cmd.Pause())
        imgui.set_item_tooltip("Space")

        imgui.same_line()

        imgui.begin_disabled(not paused)
        if imgui.button("Step", imgui.ImVec2(60, 0)):
            ctx.submit(cmd.Step(1))
        imgui.end_disabled()
        if not paused:
            imgui.set_item_tooltip("physics is running; pause to step")
        imgui.end_disabled()

        imgui.same_line()
        if imgui.button("Reset", imgui.ImVec2(60, 0)):
            ctx.submit(cmd.Reset())

        imgui.same_line()
        imgui.begin_disabled(not caps.reload)
        if imgui.button("Reload", imgui.ImVec2(70, 0)):
            ctx.submit(cmd.Reload())
        imgui.end_disabled()
        if not caps.reload:
            imgui.set_item_tooltip(f"{caps.name} does not support reload")

        imgui.separator()

        imgui.begin_disabled(not caps.simulation)
        edit = value_slider("speed", s.speed, 0.05, 8.0, initial=1.0, fmt="x%.2f", more_hint="none")
        if edit.changed:
            ctx.submit(cmd.SetSpeed(edit.value))
        imgui.end_disabled()

        if s.keyframes:
            imgui.separator()
            selected = s.active_keyframe if s.active_keyframe >= 0 else 0
            imgui.begin_disabled(not paused)
            imgui.set_next_item_width(max(80.0, imgui.get_content_region_avail().x - 68.0))
            changed, selected = imgui.slider_int(
                "##keyframe", selected, 0, len(s.keyframes) - 1, "%d"
            )
            if changed:
                ctx.submit(cmd.LoadKeyframe(selected))
            imgui.same_line()
            if imgui.button("Load", imgui.ImVec2(60, 0)):
                ctx.submit(cmd.LoadKeyframe(selected))
            imgui.end_disabled()
            if not paused:
                imgui.set_item_tooltip("physics is running; pause to load a keyframe")
            key = s.keyframes[selected]
            imgui.text_disabled(
                f"{key.name} · {selected + 1}/{len(s.keyframes)} · t={key.time:g} s"
            )

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
