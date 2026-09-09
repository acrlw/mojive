"""Measure editor operations on a composed MS-Human-700 workspace.

Use one backend per process and run without competing GPU workloads. Command
and first updated-window timings include Python editing, resource publication,
and native queue waits; they do not measure physical input-to-photon latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.composition import build_editor
from mojive.config import LayoutConfig, ViewerConfig
from mojive.types import MeshShape


def run(root: Path, output: Path, backend: str, *, show_window: bool = True, gallery: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with build_editor(
        renderer=backend,
        vsync=False,
        width=1200,
        height=800,
        show_window=show_window,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        scene_uploads = 0
        set_scene = viewer.backend.set_scene

        def record_scene(source):
            nonlocal scene_uploads
            scene_uploads += 1
            return set_scene(source)

        viewer.backend.set_scene = record_scene

        def resource_stats():
            device = getattr(viewer.backend, "device", None)
            if device is None:
                return {}
            stats = device.runtime.resource_stats()
            return {
                key: int(getattr(stats, key))
                for key in ("mesh_uploads", "texture_uploads", "upload_bytes")
            }

        def resource_delta(before):
            return {key: value - before[key] for key, value in resource_stats().items()}

        def measure_deferred(name, command):
            initial_resources = resource_stats()
            completed = []
            start = time.perf_counter()
            viewer.app._queue_model_edit(command, completed.append)
            durations = []
            while not completed:
                tick = time.perf_counter()
                viewer.sync()
                durations.append(time.perf_counter() - tick)
                if time.perf_counter() - start > 120:
                    raise TimeoutError(name)
            if not completed[0].ok:
                raise RuntimeError(completed[0].message)
            row = {
                "operation": name,
                "completed_ms": (time.perf_counter() - start) * 1000,
                "longest_ui_frame_ms": max(durations) * 1000,
                "loading_frames": len(durations),
                "resource_uploads": resource_delta(initial_resources),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            return completed[0]

        def measure(name, action, *, repeats=1):
            values = []
            initial_uploads = scene_uploads
            initial_resources = resource_stats()
            for _ in range(repeats):
                start = time.perf_counter()
                result = action()
                command_done = time.perf_counter()
                if result is not None and hasattr(result, "ok") and not result.ok:
                    raise RuntimeError(result.message)
                viewer.sync()
                submitted = time.perf_counter()
                values.append(
                    {
                        "command_ms": (command_done - start) * 1000,
                        "updated_window_ms": (submitted - start) * 1000,
                    }
                )
            uploads = scene_uploads - initial_uploads
            if name.startswith("select_"):
                assert uploads == 0, "Selection must not re-upload scene resources"
            row = {
                "operation": name,
                "samples": values,
                "scene_uploads": uploads,
                "resource_uploads": resource_delta(initial_resources),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            return result

        def add_model(relative, position):
            initial_resources = resource_stats()
            viewer.app._queue_model_load("add", root / relative, position)
            durations = []
            start = time.perf_counter()
            while True:
                tick = time.perf_counter()
                viewer.sync()
                durations.append(time.perf_counter() - tick)
                if not viewer.app._model_load_queue and viewer.app._model_load_future is None:
                    break
                if time.perf_counter() - start > 120:
                    raise TimeoutError(relative)
            if viewer.app._model_load_error:
                raise RuntimeError(viewer.app._model_load_error)
            return {
                "path": relative,
                "resource_uploads": resource_delta(initial_resources),
                "submitted_ms": (time.perf_counter() - start) * 1000,
                "longest_ui_frame_ms": max(durations) * 1000,
                "loading_frames": len(durations),
            }

        for _ in range(15):
            viewer.sync()
        measure(
            "add_floor",
            lambda: viewer.session.submit(
                cmd.AddSceneObject(MeshShape.PLANE, "floor", (2.0, 2.0, 0.1))
            ),
        )
        loads = [add_model("ms_human_700/MS-Human-700.xml", (0.0, 0.0, 0.0))]
        human = next(model for model in viewer.session.scene_models if model.model_id > 0)
        human_id = human.model_id
        original_models = {m.model_id for m in viewer.session.scene_models}

        def select_model(model_id):
            node = next(
                node
                for node in viewer.session.nodes
                if node.type is NodeType.MODEL and node.model_id == model_id
            )
            return viewer.session.submit(cmd.SelectNode(node.node_id))

        measure("select_human_first", lambda: select_model(human_id))
        measure("select_human_repeat", lambda: select_model(human_id), repeats=10)
        viewer.capture(output / "selected-human.png", surface="window")
        measure(
            "pan_human",
            lambda: viewer.app.camera.pan(0.1, 0, viewer.backend.target.height),
            repeats=60,
        )
        measure(
            "list_all_human_components",
            lambda: tuple(
                viewer.session.model_components(human_id, category)
                for category in ("contact", "actuator", "sensor", "tendon", "equality")
            ),
        )
        measure(
            "preview_model_placement",
            lambda: viewer.session.submit(
                cmd.PreviewSceneModelTransform(
                    human_id, np.array((0.01, 0, 0), np.float32), np.eye(3, dtype=np.float32)
                )
            ),
            repeats=10,
        )
        measure(
            "cancel_model_placement",
            lambda: viewer.session.submit(cmd.ClearSceneModelTransformPreview(human_id)),
        )
        snapshot = measure(
            "capture_snapshot",
            lambda: viewer.session.submit(cmd.AddModelKeyframe(human_id, "benchmark")),
        )
        measure(
            "load_snapshot",
            lambda: viewer.session.submit(cmd.LoadKeyframe(snapshot.entity_id)),
            repeats=5,
        )
        measure(
            "remove_snapshot",
            lambda: viewer.session.submit(cmd.RemoveModelKeyframe(snapshot.entity_id)),
        )
        loads.append(add_model("unitree_go2/scene.xml", (1.5, 0.0, 0.0)))
        other_id = max(model.model_id for model in viewer.session.scene_models)
        measure("select_other_model", lambda: select_model(other_id))
        measure("select_human_in_composition", lambda: select_model(human_id))
        source = measure(
            "read_mjcf_source", lambda: viewer.session.adapter.scene_model_source(human_id)
        )
        measure(
            "apply_mjcf_source", lambda: viewer.session.submit(cmd.SetModelSource(human_id, source))
        )
        deferred = measure_deferred(
            "capture_snapshot_deferred", cmd.AddModelKeyframe(human_id, "deferred")
        )
        measure_deferred("remove_snapshot_deferred", cmd.RemoveModelKeyframe(deferred.entity_id))
        measure_deferred("apply_mjcf_source_deferred", cmd.SetModelSource(human_id, source))
        measure("select_human_after_edit", lambda: select_model(human_id))
        viewer.capture(output / "composed-models.png", surface="window")
        measure("remove_other_model", lambda: viewer.app.remove_model(other_id))
        measure("undo_remove", lambda: viewer.session.submit(cmd.Undo()))
        measure("redo_remove", lambda: viewer.session.submit(cmd.Redo()))
        assert {m.model_id for m in viewer.session.scene_models} == original_models
        result = {
            "backend": backend,
            "visible_window": show_window,
            "loads": loads,
            "rows": rows,
            "viewport_pixels": [viewer.backend.target.width, viewer.backend.target.height],
        }
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        if gallery:
            select_model(human_id)
            capture_review(viewer, output)
        return result


def capture_review(viewer, output):
    from imgui_bundle import imgui

    from mojive.tools.ui_runtime import _open_collapsing_header

    _open_collapsing_header(viewer, "Actuator (700)")
    original = imgui.collapsing_header

    def scroll_header(label, *args, **kwargs):
        result = original(label, *args, **kwargs)
        if label == "Actuator (700)":
            imgui.set_scroll_here_y(0.1)
        return result

    imgui.collapsing_header = scroll_header
    try:
        viewer.sync()
    finally:
        imgui.collapsing_header = original
    viewer.capture(output / "component-table.png", surface="window")
    settings = viewer.panels.get("Settings")
    settings._category = "Interaction"
    viewer.panels.open_panel("Settings")
    original_heading = settings._group_heading

    def scroll_heading(title):
        original_heading(title)
        if title == "Mouse gestures":
            imgui.set_scroll_here_y(0.05)

    settings._group_heading = scroll_heading
    try:
        viewer.sync()
    finally:
        settings._group_heading = original_heading
    viewer.capture(output / "mouse-settings.png", surface="window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=("bgfx", "opengl"), default="bgfx")
    parser.add_argument("--output", type=Path, default=Path("output/native-editor-benchmark"))
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    run(args.root, args.output, args.backend, show_window=not args.hidden, gallery=args.gallery)


if __name__ == "__main__":
    main()
