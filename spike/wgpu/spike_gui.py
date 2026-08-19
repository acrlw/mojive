"""Spike C: rendercanvas (glfw + offscreen) and imgui-bundle over wgpu.

  offscreen  - render scene + imgui overlay into an OffscreenRenderCanvas, save PNG
  glfw       - real window, 120 frames, report pixel ratio / physical size / fps
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge_viewer.render.mesh import all_builtin
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer

from backend import WgpuSceneBackend
from spike_offscreen import OUT, build_scene


def _setup():
    adapter, source, scene = build_scene()
    backend = WgpuSceneBackend(2, 2)  # own targets unused on the canvas path
    backend.set_meshes({**all_builtin(), **source.meshes})
    backend._scene = scene
    backend.upload_instances(scene)
    backend.set_camera_uniforms(scene.camera, scene.lights)
    return adapter, backend, scene


def _gui(frame_index: int) -> None:
    imgui.set_next_window_size((320, 120), imgui.Cond_.first_use_ever)
    imgui.begin("forge-wgpu spike")
    imgui.text(f"frame {frame_index}")
    imgui.text("imgui-bundle over wgpu render pass")
    if imgui.button("a button"):
        pass
    imgui.end()


def cmd_offscreen() -> None:
    from rendercanvas.offscreen import OffscreenRenderCanvas

    adapter, backend, scene = _setup()
    canvas = OffscreenRenderCanvas(size=(640, 480), pixel_ratio=2.0)
    context = canvas.get_context("wgpu")
    format = context.get_preferred_format(backend.device.adapter)
    print(f"offscreen canvas format={format} physical={canvas.get_physical_size()} "
          f"pixel_ratio={canvas.get_pixel_ratio()}")
    context.configure(device=backend.device, format=format)

    imgui_renderer = ImguiRenderer(backend.device, canvas, render_target_format=format)
    imgui_renderer.set_gui(lambda: _gui(0))

    def draw():
        view = context.get_current_texture().create_view()
        backend.draw_to_view(view, *canvas.get_physical_size(), format)
        imgui_renderer.render()

    canvas.request_draw(draw)
    pixels = canvas.draw()  # (H, W, 4) uint8, physical resolution
    img = np.asarray(pixels)
    print(f"draw() returned {img.shape} {img.dtype}")
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img[..., :3]).save(OUT / "wgpu_gui_offscreen.png")
    print(f"saved {OUT / 'wgpu_gui_offscreen.png'}")
    adapter.release()


def cmd_glfw() -> None:
    from rendercanvas.glfw import RenderCanvas, loop

    adapter, backend, scene = _setup()
    canvas = RenderCanvas(title="forge-wgpu spike", size=(640, 480), update_mode="continuous")
    context = canvas.get_context("wgpu")
    format = context.get_preferred_format(backend.device.adapter)
    context.configure(device=backend.device, format=format)
    imgui_renderer = ImguiRenderer(backend.device, canvas, render_target_format=format)
    frames = [0]

    def draw():
        frames[0] += 1
        imgui_renderer.set_gui(lambda: _gui(frames[0]))
        view = context.get_current_texture().create_view()
        backend.draw_to_view(view, *canvas.get_physical_size(), format)
        imgui_renderer.render()

    canvas.request_draw(draw)  # continuous mode re-schedules from here
    t0 = time.perf_counter()
    loop.call_later(3.0, loop.stop)
    loop.run()
    dt = time.perf_counter() - t0
    print(f"glfw: {frames[0]} frames in {dt:.2f}s -> {frames[0] / dt:.0f} fps")
    print(f"physical size={canvas.get_physical_size()} pixel_ratio={canvas.get_pixel_ratio():.2f}")
    adapter.release()


if __name__ == "__main__":
    {"offscreen": cmd_offscreen, "glfw": cmd_glfw}[sys.argv[1]]()
