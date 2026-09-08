"""Generate camera bookmark and scene snapshot acceptance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import commands as cmd
from ..adapters.mujoco_adapter import MuJoCoAdapter
from ..assets import resolve
from ..render.backend import RenderFlag
from ..scene_state import camera_bookmark, capture_scene, restore_scene, save_named_snapshot
from ..session import Session
from ..ui.camera import OrbitCamera


class _Options:
    def __init__(self) -> None:
        self.flags = {RenderFlag.SHADOW: True, RenderFlag.TENDON: True}

    def render_options(self):
        return tuple(self.flags)

    def get_flag(self, flag):
        return self.flags[flag]

    def set_flag(self, flag, enabled):
        self.flags[flag] = bool(enabled)
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate scene state acceptance artifacts")
    parser.add_argument("-o", "--out", type=Path, default=Path("output/snapshots"))
    args = parser.parse_args(argv)
    asset = resolve("gizmo")
    adapter = MuJoCoAdapter(asset)
    try:
        session = Session(adapter, asset)
        session.submit(cmd.Pause())
        camera = OrbitCamera(pivot=[0.2, -0.1, 0.5], yaw=-125.0, pitch=28.0, distance=4.5)
        backend = _Options()
        bookmark = camera_bookmark(camera, camera.view())
        scene = capture_scene(session, backend, camera)
        camera_path = save_named_snapshot("camera-demo", bookmark, args.out / "cameras")
        scene_path = save_named_snapshot("scene-demo", scene, args.out / "scenes")
        adapter.data.qpos[:] = 0.0
        restore_scene(scene, session, backend, camera)
        report = {
            "camera": str(camera_path),
            "scene": str(scene_path),
            "qpos_count": int(adapter.model.nq),
            "restored": True,
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(args.out.resolve())
        return 0
    finally:
        adapter.release()


if __name__ == "__main__":
    raise SystemExit(main())
