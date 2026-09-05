"""Compile, adapt, and render collections of MuJoCo XML models."""

from __future__ import annotations

import argparse
import concurrent.futures
import enum
import json
import multiprocessing
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


class ModelAuditStatus(enum.StrEnum):
    """Final outcome of one model audit."""

    PASSED = "passed"
    SKIPPED_DEPENDENCY = "skipped_dependency"
    SKIPPED_FRAGMENT = "skipped_fragment"
    SKIPPED_UNSUPPORTED_ASSET = "skipped_unsupported_asset"
    LOAD_FAILED = "load_failed"
    ADAPTER_FAILED = "adapter_failed"
    WORKSPACE_FAILED = "workspace_failed"
    COMPOSITION_FAILED = "composition_failed"
    RENDER_FAILED = "render_failed"
    EMPTY_RENDER = "empty_render"


@dataclass(frozen=True)
class ModelAuditRequest:
    """Serializable input consumed by one isolated audit worker."""

    path: str
    backend: str
    camera_count: int
    width: int
    height: int
    dynamic_frames: int
    load_only: bool
    model_fragment: bool = False


@dataclass(frozen=True)
class ModelAuditResult:
    """Serializable compile, adapter, and rendering result for one XML file."""

    path: str
    status: str
    message: str = ""
    plugins: tuple[str, ...] = ()
    geom_count: int = 0
    flex_count: int = 0
    skin_count: int = 0
    source_instances: int = 0
    workspace_instances: int = 0
    composed_instances: int = 0
    rendered_views: int = 0
    visible_views: int = 0
    visible_objects: int = 0
    rgb_range: int = 0
    rgb_std: float = 0.0


_CAMERA_POSES = (
    (0.0, -20.0),
    (60.0, -20.0),
    (120.0, -20.0),
    (180.0, -20.0),
    (240.0, -20.0),
    (300.0, -20.0),
    (45.0, -55.0),
    (225.0, -55.0),
)


def discover_models(roots: list[Path]) -> list[Path]:
    """Return MuJoCo MJCF and URDF documents below the existing roots."""

    models: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file() and _is_model_document(resolved):
            models.add(resolved)
        elif resolved.is_dir():
            models.update(
                path.resolve()
                for path in resolved.rglob("*")
                if path.is_file()
                and path.suffix.lower() in (".xml", ".mjcf", ".urdf")
                and _is_model_document(path)
            )
    return sorted(models)


def _is_model_document(path: Path) -> bool:
    """Reject unrelated XML metadata while retaining malformed model candidates."""

    if path.suffix.lower() not in (".xml", ".mjcf", ".urdf"):
        return False
    try:
        tag = ET.parse(path).getroot().tag.rsplit("}", 1)[-1].lower()
    except (ET.ParseError, OSError):
        return True
    return tag in ("mujoco", "robot")


def inspect_fragment_files(models: list[Path]) -> set[Path]:
    """Return MJCF files that are composition fragments rather than standalone models."""

    included: set[Path] = set()
    candidates = set(models)
    for path in models:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        includes = tuple(root.iter("include"))
        worldbody = root.find("worldbody")
        has_standalone_geometry = worldbody is not None and any(
            worldbody.find(f".//{tag}") is not None for tag in ("body", "geom")
        )
        if (
            root.tag.rsplit("}", 1)[-1].lower() == "mujoco"
            and not includes
            and not has_standalone_geometry
        ):
            included.add(path)
        for element in includes:
            filename = element.attrib.get("file", "")
            if not filename:
                continue
            target = (path.parent / filename).resolve()
            if target in candidates:
                included.add(target)
    return included


def inspect_plugin_references(path: Path) -> tuple[str, ...]:
    """Return plugin identifiers declared by one MJCF document."""

    plugins: set[str] = set()
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return ()
    for element in root.iter():
        plugin = element.attrib.get("plugin")
        if plugin:
            plugins.add(plugin)
        if element.tag == "geom" and element.attrib.get("type", "").lower() == "sdf":
            plugins.add("mujoco.sdf")
    return tuple(sorted(plugins))


def inspect_unsupported_assets(path: Path) -> tuple[str, ...]:
    """Return mesh formats that the installed MuJoCo compiler cannot decode."""

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return ()
    formats = {
        Path(element.attrib.get("filename", element.attrib.get("file", ""))).suffix.lower()
        for element in root.iter("mesh")
    }
    return tuple(sorted(formats & {".dae"}))


def classify_load_failure(
    plugins: tuple[str, ...], message: str, *, model_fragment: bool = False
) -> ModelAuditStatus:
    """Distinguish unavailable runtime dependencies from model compilation failures."""

    lowered = message.lower()
    unavailable_plugin = any(
        marker in lowered
        for marker in (
            "unrecognized plugin",
            "plugin not found",
            "plugin library",
            "unknown element 'extension'",
        )
    )
    if plugins and unavailable_plugin:
        return ModelAuditStatus.SKIPPED_DEPENDENCY
    if model_fragment:
        return ModelAuditStatus.SKIPPED_FRAGMENT
    return ModelAuditStatus.LOAD_FAILED


def _camera(model, center: np.ndarray, extent: float, azimuth: float, elevation: float):
    """Create a deterministic free camera framing the supplied scene bounds."""

    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = max(float(extent) * 2.8, float(model.stat.extent) * 2.5, 0.1)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def _visible_object_ids(segmentation: np.ndarray) -> set[tuple[int, int]]:
    """Return visible MuJoCo object and object-type ID pairs."""

    pixels = np.asarray(segmentation, np.int32).reshape(-1, 2)
    foreground = pixels[pixels[:, 0] >= 0]
    return {tuple(map(int, pair)) for pair in foreground}


def audit_model(request: ModelAuditRequest) -> ModelAuditResult:
    """Compile, adapt, and render one MuJoCo model in an isolated process."""

    path = Path(request.path)
    plugins = inspect_plugin_references(path)
    unsupported_assets = inspect_unsupported_assets(path)
    try:
        import mujoco

        if path.suffix.lower() == ".urdf":
            from ..adapters.mujoco_adapter import _load_editable_spec

            model = _load_editable_spec(path).compile()
        else:
            model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        message = str(exc)
        unsupported_failure = bool(unsupported_assets) and any(
            marker in message.lower()
            for marker in ("no decoder found", "repeated element: 'material'")
        )
        return ModelAuditResult(
            path=str(path),
            status=(
                ModelAuditStatus.SKIPPED_UNSUPPORTED_ASSET
                if unsupported_failure
                else classify_load_failure(plugins, message, model_fragment=request.model_fragment)
            ),
            message=(
                f"MuJoCo cannot compile {', '.join(unsupported_assets)} mesh assets: {message}"
                if unsupported_failure
                else message
            ),
            plugins=plugins,
        )

    try:
        from ..adapters.base import FrameNeeds
        from ..adapters.mujoco_adapter import MuJoCoAdapter

        adapter = MuJoCoAdapter(path)
        try:
            source = adapter.scene_source()
            adapter.frame(FrameNeeds(poses=True))
            center = np.asarray(source.scene_center, np.float64)
            extent = max(float(source.scene_extent), 1e-3)
            source_instances = int(source.instance_count)
        finally:
            adapter.release()
    except Exception as exc:
        return ModelAuditResult(
            path=str(path),
            status=ModelAuditStatus.ADAPTER_FAILED,
            message=str(exc),
            plugins=plugins,
            geom_count=int(model.ngeom),
            flex_count=int(model.nflex),
            skin_count=int(model.nskin),
        )

    try:
        from ..adapters.workspace import WorkspaceAdapter

        workspace_primary = MuJoCoAdapter()
        workspace_primary.new_scene()
        workspace = WorkspaceAdapter(workspace_primary)
        try:
            workspace.load(path)
            workspace_source = workspace.scene_source()
            workspace.frame(FrameNeeds(poses=True))
            workspace_instances = int(workspace_source.instance_count)
        finally:
            workspace.release()
    except Exception as exc:
        return ModelAuditResult(
            path=str(path),
            status=ModelAuditStatus.WORKSPACE_FAILED,
            message=str(exc),
            plugins=plugins,
            geom_count=int(model.ngeom),
            flex_count=int(model.nflex),
            skin_count=int(model.nskin),
            source_instances=source_instances,
        )

    if workspace_instances != source_instances:
        return ModelAuditResult(
            path=str(path),
            status=ModelAuditStatus.WORKSPACE_FAILED,
            message=(
                f"Workspace produced {workspace_instances} instances; "
                f"direct adapter produced {source_instances}"
            ),
            plugins=plugins,
            geom_count=int(model.ngeom),
            flex_count=int(model.nflex),
            skin_count=int(model.nskin),
            source_instances=source_instances,
            workspace_instances=workspace_instances,
        )

    try:
        primary = MuJoCoAdapter()
        primary.new_scene()
        try:
            primary.add_scene_model(path, np.zeros(3, np.float32), np.eye(3, dtype=np.float32))
            composed_source = primary.scene_source()
            primary.frame(FrameNeeds(poses=True))
            composed_instances = int(composed_source.instance_count)
            composed_counts = (primary.model.ngeom, primary.model.nflex, primary.model.nskin)
        finally:
            primary.release()
    except Exception as exc:
        return ModelAuditResult(
            path=str(path),
            status=ModelAuditStatus.COMPOSITION_FAILED,
            message=str(exc),
            plugins=plugins,
            geom_count=int(model.ngeom),
            flex_count=int(model.nflex),
            skin_count=int(model.nskin),
            source_instances=source_instances,
            workspace_instances=workspace_instances,
        )

    direct_counts = (int(model.ngeom), int(model.nflex), int(model.nskin))
    if composed_instances != source_instances or composed_counts != direct_counts:
        return ModelAuditResult(
            path=str(path),
            status=ModelAuditStatus.COMPOSITION_FAILED,
            message=(
                f"Composition produced geom/flex/skin={composed_counts} and "
                f"{composed_instances} instances; direct loading produced "
                f"geom/flex/skin={direct_counts} and {source_instances} instances"
            ),
            plugins=plugins,
            geom_count=int(model.ngeom),
            flex_count=int(model.nflex),
            skin_count=int(model.nskin),
            source_instances=source_instances,
            workspace_instances=workspace_instances,
            composed_instances=composed_instances,
        )

    common = {
        "path": str(path),
        "plugins": plugins,
        "geom_count": int(model.ngeom),
        "flex_count": int(model.nflex),
        "skin_count": int(model.nskin),
        "source_instances": source_instances,
        "workspace_instances": workspace_instances,
        "composed_instances": composed_instances,
    }
    if request.load_only:
        return ModelAuditResult(status=ModelAuditStatus.PASSED, **common)

    os.environ["MOJIVE_RENDERER"] = request.backend
    try:
        from ..renderer import Renderer

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        width = min(max(1, int(request.width)), max(1, int(model.vis.global_.offwidth)))
        height = min(max(1, int(request.height)), max(1, int(model.vis.global_.offheight)))
        view_count = min(max(1, int(request.camera_count)), len(_CAMERA_POSES))
        visible_objects: set[tuple[int, int]] = set()
        visible_views = 0
        rgb_min = 255
        rgb_max = 0
        rgb_stds: list[float] = []
        with Renderer(model, height=height, width=width) as renderer:
            for index, (azimuth, elevation) in enumerate(_CAMERA_POSES[:view_count]):
                if index == view_count - 1:
                    for _ in range(max(0, int(request.dynamic_frames))):
                        mujoco.mj_step(model, data)
                camera = _camera(model, center, extent, azimuth, elevation)
                renderer.update_scene(data, camera)
                rgb = renderer.render()
                rgb_min = min(rgb_min, int(rgb.min(initial=255)))
                rgb_max = max(rgb_max, int(rgb.max(initial=0)))
                rgb_stds.append(float(rgb.std()))
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                objects = _visible_object_ids(segmentation)
                if objects:
                    visible_views += 1
                    visible_objects.update(objects)
    except Exception as exc:
        return ModelAuditResult(
            status=ModelAuditStatus.RENDER_FAILED,
            message=str(exc),
            **common,
        )

    renderable_content = model.ngeom > 0 or model.nflex > 0 or model.nskin > 0
    status = (
        ModelAuditStatus.PASSED
        if visible_views > 0 or not renderable_content
        else ModelAuditStatus.EMPTY_RENDER
    )
    message = "" if status == ModelAuditStatus.PASSED else "No scene object was visible in any view"
    return ModelAuditResult(
        status=status,
        message=message,
        rendered_views=view_count,
        visible_views=visible_views,
        visible_objects=len(visible_objects),
        rgb_range=max(0, rgb_max - rgb_min),
        rgb_std=max(rgb_stds, default=0.0),
        **common,
    )


def run_suite(
    models: list[Path],
    *,
    backend: str,
    jobs: int,
    camera_count: int,
    width: int,
    height: int,
    dynamic_frames: int,
    load_only: bool,
) -> list[ModelAuditResult]:
    """Audit model paths in parallel and return results in path order."""

    fragment_files = inspect_fragment_files(models)
    requests = [
        ModelAuditRequest(
            path=str(path),
            backend=backend,
            camera_count=camera_count,
            width=width,
            height=height,
            dynamic_frames=dynamic_frames,
            load_only=load_only,
            model_fragment=path in fragment_files,
        )
        for path in models
    ]
    results: list[ModelAuditResult] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(jobs)), mp_context=context
    ) as executor:
        futures = {executor.submit(audit_model, request): request.path for request in requests}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed:>3}/{len(futures)}] {result.status:<15} {result.path}",
                flush=True,
            )
    return sorted(results, key=lambda result: result.path)


def build_report(roots: list[Path], results: list[ModelAuditResult]) -> dict[str, object]:
    """Build the compact JSON report written by the command-line tool."""

    counts = {status.value: 0 for status in ModelAuditStatus}
    for result in results:
        counts[result.status] += 1
    return {
        "schema": 2,
        "roots": [str(root.expanduser().resolve()) for root in roots],
        "counts": counts,
        "results": [asdict(result) for result in results],
    }


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the model suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="model directories or MJCF/URDF files")
    parser.add_argument("--backend", choices=("opengl", "wgpu"), default="opengl")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--camera-count", type=int, default=len(_CAMERA_POSES))
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--dynamic-frames", type=int, default=1)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MuJoCo model suite and return a shell status code."""

    args = _parser().parse_args(argv)
    models = discover_models(args.roots)
    if not models:
        raise SystemExit("No MuJoCo MJCF or URDF models found")
    results = run_suite(
        models,
        backend=args.backend,
        jobs=args.jobs,
        camera_count=args.camera_count,
        width=args.width,
        height=args.height,
        dynamic_frames=args.dynamic_frames,
        load_only=args.load_only,
    )
    report = build_report(args.roots, results)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {args.report}")
    print(json.dumps(report["counts"], indent=2))
    failures = {
        ModelAuditStatus.LOAD_FAILED,
        ModelAuditStatus.ADAPTER_FAILED,
        ModelAuditStatus.WORKSPACE_FAILED,
        ModelAuditStatus.COMPOSITION_FAILED,
        ModelAuditStatus.RENDER_FAILED,
        ModelAuditStatus.EMPTY_RENDER,
    }
    return int(any(result.status in failures for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
