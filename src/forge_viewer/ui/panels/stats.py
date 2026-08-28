"""Frame timing and render-pass statistics."""

from __future__ import annotations

from imgui_bundle import imgui

from ...adapters.base import FrameNeeds
from . import Panel, PanelContext, begin_kv_table, labeled
from .plot import Ring

_FRAME_SCALE_STEPS = (16.7, 25.0, 33.4, 50.0, 66.7, 100.0, 133.4, 200.0)


def _scale_ceiling(value: float) -> float:
    for step in _FRAME_SCALE_STEPS:
        if value <= step:
            return step
    return max(_FRAME_SCALE_STEPS[-1], float(value))


class StatsPanel(Panel):
    name = "Stats"
    default_open = True
    shortcut = "F8"
    closable = False

    def __init__(self) -> None:
        super().__init__()
        self._frame_ms = Ring(240)
        self._scale_ms = _FRAME_SCALE_STEPS[0]
        self._scale_hold = 0

    def frame_needs(self) -> FrameNeeds:
        return FrameNeeds.none()

    def draw(self, ctx: PanelContext) -> None:
        stats = ctx.backend.stats
        caps = ctx.backend.caps

        self._frame_ms.push(ctx.dt * 1000.0)
        _lo, hi = self._frame_ms.span()
        self._update_scale(hi)
        recent = ctx.dt * 1000.0
        imgui.plot_lines(
            "##frame",
            self._frame_ms.data,
            values_offset=self._frame_ms.offset,
            overlay_text=f"{recent:6.2f} ms   {1000.0 / recent if recent > 0.01 else 0.0:5.1f} fps",
            scale_min=0.0,
            scale_max=self._scale_ms,
            graph_size=imgui.ImVec2(-1, 60.0 * ctx.style_scale),
        )
        imgui.text_disabled(f"scale 0 .. {self._scale_ms:.1f} ms   (60 fps = 16.7 ms)")
        count_flags = (
            imgui.TableFlags_.sizing_stretch_same
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.no_saved_settings
        )
        if imgui.begin_table("stats_counts", 2, count_flags):
            imgui.table_setup_column("metric", imgui.TableColumnFlags_.width_stretch, 1.0)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch, 1.0)
            labeled("draw calls", str(stats.draw_calls))
            labeled("instances", str(stats.instances))
            labeled("triangles", f"{stats.triangles:,}")
            labeled("buckets", str(stats.buckets))
            labeled("frame cpu", f"{stats.frame_cpu_ms:.3f} ms")
            imgui.end_table()

        imgui.separator()
        self._passes(ctx, stats, caps)

        if stats.notes:
            imgui.separator()
            if begin_kv_table("stats_notes"):
                for k, v in stats.notes.items():
                    labeled(k, v)
                imgui.end_table()

    def _update_scale(self, peak: float) -> None:
        target = _scale_ceiling(peak)
        if target > self._scale_ms:
            self._scale_ms = target
            self._scale_hold = len(self._frame_ms.data)
        elif target < self._scale_ms and self._scale_hold == 0:
            self._scale_ms = target
        elif self._scale_hold:
            self._scale_hold -= 1

    def _passes(self, ctx: PanelContext, stats, caps) -> None:
        if not stats.cpu_ms and not stats.gpu_ms:
            imgui.text_disabled("no per-pass timing from this backend")
            return

        names = list(stats.cpu_ms)
        names += [k for k in stats.gpu_ms if k not in stats.cpu_ms]

        flags = (
            imgui.TableFlags_.sizing_stretch_prop
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
        )
        if imgui.begin_table("stats_passes", 3, flags):
            imgui.table_setup_column("pass")
            imgui.table_setup_column("cpu ms")
            imgui.table_setup_column("gpu ms")
            imgui.table_headers_row()
            for name in names:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(name)
                imgui.table_next_column()
                cpu = stats.cpu_ms.get(name)
                imgui.text(f"{cpu:.3f}" if cpu is not None else "—")
                imgui.table_next_column()
                gpu = stats.gpu_ms.get(name)
                imgui.text(f"{gpu:.3f}" if gpu is not None else "—")
            imgui.end_table()

        if not caps.gpu_timing:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.text_disabled),
                f"{caps.name}: no GPU timer queries on this machine; cpu column only",
            )
