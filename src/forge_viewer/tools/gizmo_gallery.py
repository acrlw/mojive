"""Generate close-up acceptance images for the native 2D/3D gizmos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .. import commands as cmd
from ..assets import resolve
from ..composition import build
from ..gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, GizmoMode, hit_test, project, world_scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/gizmo-gallery"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(resolve("gizmo"), paused=True, vsync=False, width=1600, height=1000)
    try:
        node = next(node for node in viewer.session.nodes if node.posable)
        viewer.session.submit(cmd.Select(node.object_id))
        viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
        for _ in range(4):
            viewer.sync()

        for style in ("2d", "3d"):
            _position(viewer, node, style, args.output)
            _rotation(viewer, node, style, args.output)
    finally:
        viewer.release()
    print(args.output.resolve())
    return 0


def _position(viewer, node, style: str, output: Path) -> None:
    io = imgui.get_io()
    viewer.app.gizmo.set_mode("translate")
    viewer.app.gizmo.set_style(style)
    viewer.app.gizmo.set_space("body")
    for _ in range(3):
        viewer.sync()
    _save(viewer, node, output / f"position-{style}.png")

    camera, rect, origin, rotation, scale = _state(viewer, node)
    cursor = _axis_cursor(viewer, camera, rect, origin, rotation, scale)
    io.add_mouse_pos_event(*cursor)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    axis = project(camera, (origin, origin + rotation[:, 2] * scale), rect)[:, :2]
    direction = axis[1] - axis[0]
    direction /= np.linalg.norm(direction)
    io.add_mouse_pos_event(*(cursor + direction * 42.0))
    viewer.sync()
    _save(viewer, node, output / f"position-drag-{style}.png")
    viewer.app.gizmo.translation_snap_m = 0.5
    io.add_key_event(imgui.Key.mod_shift, True)
    io.add_mouse_pos_event(*(cursor + direction * 57.0))
    viewer.sync()
    if not viewer.app.gizmo.snapping:
        raise RuntimeError("position snap input was not applied")
    _save(viewer, node, output / f"position-snap-{style}.png")
    io.add_key_event(imgui.Key.mod_shift, False)
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.session.submit(cmd.Reset())
    viewer.sync()


def _rotation(viewer, node, style: str, output: Path) -> None:
    io = imgui.get_io()
    viewer.app.gizmo.set_mode("rotate")
    viewer.app.gizmo.set_style(style)
    viewer.app.gizmo.set_space("body")
    for _ in range(3):
        viewer.sync()
    camera, rect, origin, rotation, scale = _state(viewer, node)

    def ring_point(angle: float) -> np.ndarray:
        return origin + scale * RING_RADIUS * (
            np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1]
        )

    start_angle = next(
        angle
        for angle in np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        if hit_test(
            camera,
            origin,
            rotation,
            rect,
            tuple(np.floor(project(camera, (ring_point(angle),), rect)[0, :2])),
            GizmoMode.ROTATE,
            viewer.window.style_scale,
        )[0]
        is GizmoHandle.ROTATE_Z
    )
    start = np.floor(project(camera, (ring_point(start_angle),), rect)[0, :2])
    io.add_mouse_pos_event(*start)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    end = project(camera, (ring_point(start_angle + np.radians(42.0)),), rect)[0, :2]
    io.add_mouse_pos_event(*end)
    viewer.sync()
    _save(viewer, node, output / f"rotation-drag-{style}.png")
    viewer.app.gizmo.rotation_snap_deg = 5.0
    io.add_key_event(imgui.Key.mod_shift, True)
    snapped = project(camera, (ring_point(start_angle + np.radians(49.0)),), rect)[0, :2]
    io.add_mouse_pos_event(*snapped)
    viewer.sync()
    if not viewer.app.gizmo.snapping:
        raise RuntimeError(
            "rotation snap input was not applied: "
            f"using={viewer.app.gizmo.using}, "
            f"active={viewer.app.gizmo.active_handle.name}, "
            f"label={viewer.app.gizmo.value_label!r}, "
            f"cursor=({io.mouse_pos.x:.1f}, {io.mouse_pos.y:.1f}), "
            f"target=({snapped[0]:.1f}, {snapped[1]:.1f})"
        )
    _save(viewer, node, output / f"rotation-snap-{style}.png")
    viewer.app.camera.look_from(-135.0, 0.0, viewer.app.camera_out, animate=False)
    for _ in range(3):
        viewer.sync()
    _save(viewer, node, output / f"rotation-snap-edge-{style}.png")
    viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
    for _ in range(3):
        viewer.sync()
    io.add_key_event(imgui.Key.mod_shift, False)
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.session.submit(cmd.Reset())
    viewer.sync()


def _state(viewer, node):
    camera = viewer.app.camera.view()
    rect = viewer.app._viewport_rect
    origin = np.asarray(viewer.session.frame.body_xpos[node.body_index], np.float64)
    rotation = np.asarray(viewer.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3)
    return (
        camera,
        rect,
        origin,
        rotation,
        world_scale(camera, origin, rect[3], SIZE_PT * viewer.window.style_scale),
    )


def _axis_cursor(viewer, camera, rect, origin, rotation, scale) -> np.ndarray:
    for fraction in np.linspace(0.4, 0.75, 15):
        point = np.floor(
            project(camera, (origin + rotation[:, 2] * scale * fraction,), rect)[0, :2]
        )
        handle, _axes, _planes = hit_test(
            camera,
            origin,
            rotation,
            rect,
            tuple(point),
            GizmoMode.TRANSLATE,
            viewer.window.style_scale,
        )
        if handle is GizmoHandle.Z:
            return point
    raise RuntimeError("Z-axis gizmo handle is not visible")


def _save(viewer, node, path: Path) -> None:
    viewer.sync()
    pixels = viewer.window.read_frame()[::-1, :, :3]
    camera, rect, origin, _rotation, _scale = _state(viewer, node)
    center = project(camera, (origin,), rect)[0, :2]
    display = imgui.get_io().display_size
    sx = pixels.shape[1] / display.x
    sy = pixels.shape[0] / display.y
    crop_points = 180.0 if viewer.window.style_scale <= 1.0 else 240.0 * viewer.window.style_scale
    half = int(crop_points * max(sx, sy))
    x0 = max(0, int(center[0] * sx) - half)
    y0 = max(0, int(center[1] * sy) - half)
    crop = pixels[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
    Image.fromarray(crop, "RGB").resize((1080, 1080), Image.Resampling.LANCZOS).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
