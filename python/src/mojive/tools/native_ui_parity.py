"""Compare real ImGui filtering and view-axis label motion across window backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.ui.camera import OrbitCamera
from mojive.ui.draw2d import ImguiDraw2D
from mojive.ui.viewcube import ViewCube
from mojive.ui.window import Window, WindowConfig


class LabelProbe(ImguiDraw2D):
    def __init__(self, draw_list):
        super().__init__(draw_list)
        self.errors = []

    def centered_label(self, text, center, color, max_width):
        start = len(self._dl.vtx_buffer)
        super().centered_label(text, center, color, max_width)
        vertices = np.array(
            [
                (self._dl.vtx_buffer[i].pos.x, self._dl.vtx_buffer[i].pos.y)
                for i in range(start, len(self._dl.vtx_buffer))
            ]
        )
        if len(vertices):
            actual = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
            self.errors.append(float(np.max(np.abs(actual - center))))


class CameraSink:
    def set_camera(self, _camera):
        pass


def capture(backend, scale):
    config = WindowConfig(
        width=600,
        height=320,
        ui_scale=scale,
        vsync=False,
        ini_path=None,
        docking=False,
        show_on_start=False,
    )
    if backend == "bgfx":
        from mojive.render.native.device import acquire_device
        from mojive.ui.window_native import NativeWindow

        window = NativeWindow(config, device=acquire_device())
    else:
        window = Window(config)
    cube, sink, frames, errors = ViewCube(), CameraSink(), [], []
    try:
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            camera = OrbitCamera(yaw=145, pitch=35)
            cube.update(camera.view(), (10, 40, 160, 180), (-100, -100), scale)
            ball = next(b for b in cube.balls if b.axis == axis and b.sign == sign)
            cube.click(camera, ball, sink)
            for index in range(40):
                camera.advance(1 / 120, sink)
                window.begin_frame()
                dl = imgui.get_foreground_draw_list()
                probe = LabelProbe(dl)
                probe.rect_filled((0, 0), window.size_points, (0.10, 0.12, 0.14, 1))
                cube.update(camera.view(), (10, 40, 160, 180), (-100, -100), scale)
                cube.draw(probe, scale)
                probe.text((220.25, 30.25), (1, 1, 1, 1), "+X  -X  +Y  -Y  +Z", pixel_snap=False)
                probe.text((220.25, 65.25), (1, 1, 1, 1), "All levels  全部级别", pixel_snap=False)
                # Native ImGui's textured AA path is used by navigation outlines.
                dl.add_rect(
                    (210.25, 110.25),
                    (570.25, 170.25),
                    imgui.get_color_u32((0.65, 0.85, 0.6, 1)),
                    12.0,
                    2.0,
                    0,
                )
                for i in range(7):
                    dl.add_line(
                        (220.25, 210.25 + i * 9),
                        (560.25, 228.75 + i * 9),
                        imgui.get_color_u32((1, 1, 1, 1)),
                        1 + i * 0.5,
                    )
                errors.extend(probe.errors)
                frame = window.end_frame(readback=True)
                if axis == 0 and sign == 1 and index == 0:
                    # The first frame can upload newly baked font glyphs.
                    continue
                frames.append(frame[::-1].copy())
        return frames, max(errors), window.pixel_scale
    finally:
        window.close()


def run(output, check=True):
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for scale in (1.0, 1.5):
        gl, a, ratio = capture("opengl", scale)
        native, b, _ = capture("bgfx", scale)
        errors = [np.abs(x.astype(float) - y) for x, y in zip(gl, native, strict=True)]
        row = {
            "label_center_error_points": {"opengl": a, "bgfx": b},
            "color_mean_max": max(float(e.mean()) for e in errors),
            "color_p99_max": max(float(np.percentile(e, 99)) for e in errors),
        }
        report[str(scale)] = row
        preview = [
            Image.fromarray(np.concatenate((x, y), axis=1)) for x, y in zip(gl, native, strict=True)
        ]
        preview[0].save(output / f"ui-{scale}.png")
        rect = (int(10 * ratio), int(40 * ratio), int(200 * ratio), int(220 * ratio))
        movie = [
            Image.fromarray(
                np.concatenate(
                    (
                        x[rect[1] : rect[3], rect[0] : rect[2]],
                        y[rect[1] : rect[3], rect[0] : rect[2]],
                    ),
                    axis=1,
                )
            )
            for x, y in zip(gl, native, strict=True)
        ]
        movie[0].save(
            output / f"axis-clicks-{scale}.webp",
            save_all=True,
            append_images=movie[1:],
            duration=8,
            loop=0,
            lossless=True,
        )
        print(scale, row, flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if check:
        for row in report.values():
            assert max(row["label_center_error_points"].values()) < 3e-5, row
            assert row["color_mean_max"] < 0.2 and row["color_p99_max"] <= 2, row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/native-ui-parity"))
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    run(args.output, not args.record_only)


if __name__ == "__main__":
    main()
