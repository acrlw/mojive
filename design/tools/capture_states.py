"""Capture full-window UI state screenshots for design review.

Run one state per process so each capture starts from a fresh viewer:

    .venv/bin/python design/tools/capture_states.py selected-gizmo
    .venv/bin/python design/tools/capture_states.py all

Images are written to output/ui-design-baseline/ at 1920x1080, including the imgui UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import commands as cmd
from mojive.composition import build

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "ui-design-baseline"
WIDTH, HEIGHT = 1920, 1080
SETTLE = 30


def _node_by_name(session, name: str):
    return next((n for n in session.nodes if n.name == name), None)


def _save_window(viewer, path: Path) -> None:
    px = viewer.window.read_frame()
    if px is None:
        raise RuntimeError("read_frame returned no pixels")
    arr = np.asarray(px)[::-1][..., :3]
    Image.fromarray(arr, "RGB").save(path)
    print(f"saved {path}")


def _settle(viewer, frames: int = SETTLE) -> None:
    for _ in range(frames):
        viewer.sync()


def _focus_panel(name: str) -> None:
    """Select a docked panel tab by registry name, e.g. "###Inspector".

    Uses imgui.internal between frames; imgui.set_window_focus crashes here.
    """
    from imgui_bundle import imgui

    win = imgui.internal.find_window_by_name(f"###{name}")
    if win is None:
        print(f"warning: panel window {name!r} not found; tab left unchanged")
        return
    imgui.internal.focus_window(win)


def _build(asset: str, *, paused: bool = True):
    return build(
        ROOT / "assets" / asset,
        "mujoco",
        paused=paused,
        vsync=False,
        width=WIDTH,
        height=HEIGHT,
    )


def capture_selected_gizmo() -> None:
    viewer = _build("showcase.xml")
    try:
        _settle(viewer, 5)
        node = _node_by_name(viewer.session, "a_sphere")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 3)
        _focus_panel("Inspector")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-selected-gizmo.png")
    finally:
        viewer.release()


def capture_rotate_gizmo() -> None:
    viewer = _build("showcase.xml")
    try:
        _settle(viewer, 5)
        node = _node_by_name(viewer.session, "a_sphere")
        viewer.session.submit(cmd.Select(node.object_id))
        viewer.app.gizmo.set_mode("rotate")
        _settle(viewer, 3)
        _focus_panel("Inspector")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-rotate-gizmo.png")
    finally:
        viewer.release()


def capture_settings_modal() -> None:
    viewer = _build("showcase.xml")
    try:
        _settle(viewer, 5)
        viewer.app.panels.open_panel("Settings")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-settings-modal.png")
    finally:
        viewer.release()


def capture_help() -> None:
    viewer = _build("showcase.xml")
    try:
        _settle(viewer, 5)
        viewer.app.panels.open_panel("Help")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-help.png")
    finally:
        viewer.release()


def capture_joints_plot() -> None:
    viewer = _build("joint_types.xml")
    try:
        _settle(viewer, 5)
        viewer.app.panels.open_panel("Joints")
        viewer.app.panels.open_panel("Plot")
        node = _node_by_name(viewer.session, "hinge_body")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 3)
        _focus_panel("Joints")
        # A focus change only takes effect when the panel draws while focused,
        # so settle between focusing tabs from different dock nodes.
        _settle(viewer, 2)
        _focus_panel("Plot")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-joints-plot.png")
    finally:
        viewer.release()


def capture_camera_preview() -> None:
    viewer = _build("showcase.xml")
    try:
        _settle(viewer, 5)
        node = _node_by_name(viewer.session, "showcase")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 3)
        _focus_panel("Inspector")
        _settle(viewer)
        _save_window(viewer, OUT_DIR / "state-camera-preview.png")
    finally:
        viewer.release()


def capture_running() -> None:
    viewer = _build("showcase.xml", paused=False)
    try:
        _settle(viewer, 40)
        _save_window(viewer, OUT_DIR / "state-running.png")
    finally:
        viewer.release()


STATES = {
    "selected-gizmo": capture_selected_gizmo,
    "rotate-gizmo": capture_rotate_gizmo,
    "settings-modal": capture_settings_modal,
    "help": capture_help,
    "joints-plot": capture_joints_plot,
    "camera-preview": capture_camera_preview,
    "running": capture_running,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=[*STATES, "all"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(STATES) if args.state == "all" else [args.state]
    for name in names:
        print(f"--- capturing {name} ---")
        STATES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
