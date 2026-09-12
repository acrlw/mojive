"""Exercise live Layers controls, recording countdown, and compact hinge interaction."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imageio_ffmpeg import read_frames
from imgui_bundle import imgui
from PIL import Image

from mojive.application.composition import build
from mojive.capture import CaptureSurface, RecordingPhase
from mojive.drawing.canvas import Canvas2D
from mojive.interaction.gizmo import SIZE_PT, GizmoHandle
from mojive.scene.assets import resolve

from .. import commands as cmd
from ..config import PanelConfig, RecordingConfig, ViewerConfig, ViewportLayers
from ..render.debugdraw import Occlusion
from ..types import CameraView
from ..ui.gizmo import JOINT_RANGE_RADIUS, _RotationDialProjector
from ..ui.panels import layers as layers_panel


def _save(viewer, path: Path, surface=CaptureSurface.WINDOW) -> np.ndarray:
    image = viewer.capture_array(surface=surface)
    Image.fromarray(image).save(path)
    return image


def _click(viewer, point) -> None:
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.sync()


def _run(viewer, output: Path) -> dict:
    app = viewer.app
    viewer.sync()
    node = next(node for node in viewer.session.nodes if node.name == "06_precision_hinge")
    viewer.session.submit(cmd.Select(node.object_id))
    viewer.sync()
    joint, reason = app.gizmo._joint_target(viewer.session, node)
    assert joint is not None, reason
    pos, basis = app.gizmo._target_pose(viewer.session, node, joint)
    app.camera.adopt(CameraView(eye=pos - basis[:, 2] * 2.5, target=pos, up=basis[:, 1]))
    app.camera.publish(app.camera_out)
    debug = viewer.backend.debug
    debug.layer("trajectory", Occlusion.ALWAYS).line(
        "reference",
        pos + basis[:, 0] * -0.5 + basis[:, 1] * 0.35,
        pos + basis[:, 0] * 0.5 + basis[:, 1] * 0.35,
        (1, 0.2, 0.4, 1),
        7,
    )
    canvas = Canvas2D(debug, origin=pos, x_axis=basis[:, 0], y_axis=basis[:, 1])
    canvas.layer("annotations", occlusion=Occlusion.ALWAYS).line(
        "reference",
        (-0.5, -0.35),
        (0.5, -0.35),
        (0.15, 0.9, 1, 1),
        7,
    )
    controls = {}
    native_checkbox = layers_panel.themed_checkbox

    def checkbox(label, *args, **kwargs):
        result = native_checkbox(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        controls[label.split("##")[0]] = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2)
        return result

    with patch.object(layers_panel, "themed_checkbox", checkbox):
        for _ in range(5):
            viewer.sync()
        _save(viewer, output / "layers-panel.png")
        baseline = _save(viewer, output / "all-layers.png", CaptureSurface.VIEWPORT)
        for field, label in (
            ("debug_3d", "3D debug drawings"),
            ("debug_2d", "2D canvas drawings"),
            ("viewport_ui", "Viewport controls"),
            ("gizmos", "Transform and joint gizmos"),
        ):
            _click(viewer, controls[app.localizer.text(label)])
            assert not getattr(app.viewport_layers, field), field
            hidden = _save(viewer, output / f"hidden-{field}.png", CaptureSurface.VIEWPORT)
            assert (
                np.count_nonzero(np.max(np.abs(hidden.astype(int) - baseline), axis=2) > 5) > 25
            ), field
            _click(viewer, controls[app.localizer.text(label)])
            assert getattr(app.viewport_layers, field), field
        assert debug.layer("trajectory", Occlusion.ALWAYS).visible
        _click(viewer, controls["trajectory"])
        assert "trajectory" in app.viewport_layers.hidden_debug_layers
        _save(viewer, output / "hidden-named-layer.png", CaptureSurface.VIEWPORT)
        _click(viewer, controls["trajectory"])

        # Hover the middle of the complementary arc, far from the four-degree allowed range.
        view, rect = app._camera_view(), app._viewport_rect
        dial = _RotationDialProjector(
            view, rect, pos, basis[:, 2], basis[:, 0], SIZE_PT * viewer.window.style_scale
        )
        cursor = dial.points(JOINT_RANGE_RADIUS, (np.pi,))[0, :2]
        imgui.get_io().add_mouse_pos_event(*cursor)
        viewer.sync()
        assert app.gizmo.hovered_handle is GizmoHandle.ROTATE_Z
        _save(viewer, output / "joint-complement-hover.png", CaptureSurface.VIEWPORT)
        imgui.get_io().add_mouse_pos_event(4, 4)
        viewer.sync()

        canceled = output / "canceled.mp4"
        viewer.start_recording(canceled)
        viewer.sync()
        assert viewer.recording.phase is RecordingPhase.COUNTDOWN
        assert viewer.stop_recording() is None
        assert not canceled.exists()

        path = output / "viewport.mp4"
        path.unlink(missing_ok=True)
        viewer.start_recording(path, countdown=0.75)
        viewer.sync()
        _save(viewer, output / "countdown.png")
        assert not path.exists()
        deadline = time.monotonic() + 10
        while viewer.recording.phase is RecordingPhase.COUNTDOWN:
            assert time.monotonic() < deadline
            viewer.sync()
            time.sleep(0.005)
        first = _save(viewer, output / "recording-first-frame.png", CaptureSurface.VIEWPORT)
        for _ in range(40):
            viewer.sync()
            time.sleep(1 / 60)
        _click(viewer, controls[app.localizer.text("3D debug drawings")])
        _click(viewer, controls[app.localizer.text("Viewport controls")])
        for _ in range(40):
            viewer.sync()
            time.sleep(1 / 60)
        last = _save(viewer, output / "recording-layers-hidden.png", CaptureSurface.VIEWPORT)
        frames = viewer.recording.frames
        assert viewer.stop_recording() == path
        reader = read_frames(str(path))
        metadata = next(reader)
        first_frame = next(reader)
        last_frame = first_frame
        count = 1
        for frame in reader:
            last_frame = frame
            count += 1
        assert count == frames
        assert metadata["fps"] == 60
        shape = (metadata["size"][1], metadata["size"][0], 3)
        start_image = np.frombuffer(first_frame, np.uint8).reshape(shape)
        end_image = np.frombuffer(last_frame, np.uint8).reshape(shape)

        # FFmpeg pads odd viewport dimensions to an even encoder size.
        def difference(encoded, reference):
            height = min(encoded.shape[0], reference.shape[0])
            width = min(encoded.shape[1], reference.shape[1])
            return float(
                np.mean(np.abs(encoded[:height, :width].astype(float) - reference[:height, :width]))
            )

        first_error = difference(start_image, first)
        last_error = difference(end_image, last)
        assert first_error < 4, first_error
        assert last_error < 4, last_error
        Image.fromarray(start_image).save(output / "video-first-frame.png")
        Image.fromarray(end_image).save(output / "video-last-frame.png")
        viewer.configure_layers(ViewportLayers())
        viewer.panels.open_panel("Settings")
        viewer.panels.get("Settings").show_category("Recording")
        for _ in range(3):
            viewer.sync()
        _save(viewer, output / "recording-settings.png")
        return {
            "backend": viewer.backend.caps.name,
            "frames": frames,
            "fps": metadata["fps"],
            "first_frame_mean_error": first_error,
            "last_frame_mean_error": last_error,
            "checks": [
                "live layer checkboxes",
                "named layer visibility",
                "complement hover",
                "countdown cancellation",
                "clean first encoded frame",
                "live toggles in video",
            ],
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/recording-layers"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    config = ViewerConfig(
        panels={"layers": PanelConfig(open=True), "inspector": PanelConfig(open=False)},
        recording=RecordingConfig(),
    )
    with patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")}):
        scale = max(1.0, float(os.environ.get("MOJIVE_UI_SCALE", "1")))
        viewer = build(
            resolve("joint_gizmo"),
            config=config,
            vsync=False,
            width=round(1600 * scale),
            height=round(1000 * scale),
        )
        try:
            report = _run(viewer, args.output)
        finally:
            viewer.release()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
