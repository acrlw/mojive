"""Persistent camera bookmarks and complete scene snapshots."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import PhysicsState
from .types import CameraView, Environment, Light, LightType, Material

SCENE_SNAPSHOT_FORMAT = "mojive.scene-snapshot"
LEGACY_SCENE_SNAPSHOT_FORMATS = frozenset({"forge.scene-snapshot"})
FORMAT_VERSION = 2
CAMERA_BOOKMARK_FORMAT = "mojive.camera-bookmark"
LEGACY_CAMERA_BOOKMARK_FORMATS = frozenset({"forge.camera-bookmark"})
CAMERA_BOOKMARK_VERSION = 1
DEFAULT_DIRECTORY = Path("output/snapshots")


def camera_bookmark(camera, view: CameraView, source: int = -1) -> dict[str, Any]:
    return {
        "format": CAMERA_BOOKMARK_FORMAT,
        "version": CAMERA_BOOKMARK_VERSION,
        "source": int(source),
        "eye": _array(view.eye),
        "target": _array(view.target),
        "up": _array(view.up),
        "yaw": float(getattr(camera, "yaw", 0.0)),
        "pitch": float(getattr(camera, "pitch", 0.0)),
        "distance": float(getattr(camera, "distance", view.distance())),
        "fov_y": float(view.fov_y),
        "near": float(view.near),
        "far": float(view.far),
        "aspect": float(view.aspect),
        "orthographic": bool(view.orthographic),
        "ortho_height": float(view.ortho_height),
        "focal_length": _array(view.focal_length),
        "sensor_size": _array(view.sensor_size),
        "principal_offset": _array(view.principal_offset),
    }


def apply_camera_bookmark(bookmark: dict[str, Any], camera, select_source=None) -> CameraView:
    format_name = bookmark.get("format")
    if format_name != CAMERA_BOOKMARK_FORMAT and format_name not in LEGACY_CAMERA_BOOKMARK_FORMATS:
        raise ValueError(f"Unsupported camera bookmark format: {format_name}")
    version = bookmark.get("version", -1)
    if int(version) != CAMERA_BOOKMARK_VERSION:
        raise ValueError(f"Unsupported camera bookmark version: {version}")
    view = CameraView(
        eye=np.asarray(bookmark["eye"], np.float32),
        target=np.asarray(bookmark["target"], np.float32),
        up=np.asarray(bookmark["up"], np.float32),
        fov_y=float(bookmark["fov_y"]),
        near=float(bookmark["near"]),
        far=float(bookmark["far"]),
        aspect=float(bookmark["aspect"]),
        orthographic=bool(bookmark["orthographic"]),
        ortho_height=float(bookmark["ortho_height"]),
        focal_length=np.asarray(bookmark.get("focal_length", (0.0, 0.0)), np.float32),
        sensor_size=np.asarray(bookmark.get("sensor_size", (0.0, 0.0)), np.float32),
        principal_offset=np.asarray(bookmark.get("principal_offset", (0.0, 0.0)), np.float32),
    )
    source = int(bookmark.get("source", -1))
    if select_source is not None:
        select_source(source)
    if source < 0 and hasattr(camera, "adopt"):
        camera.adopt(view)
        camera.yaw = float(bookmark.get("yaw", camera.yaw))
        camera.pitch = float(bookmark.get("pitch", camera.pitch))
        camera.distance = float(bookmark.get("distance", camera.distance))
    return view


def capture_scene(
    session, backend, camera, *, camera_source: int = -1, camera_view: CameraView | None = None
) -> dict[str, Any]:
    state = session.adapter.capture_state()
    if state is None:
        raise RuntimeError(f"{session.adapter.caps.name} does not expose scene state")
    source = session.source
    return {
        "format": SCENE_SNAPSHOT_FORMAT,
        "version": FORMAT_VERSION,
        "asset": str(session.asset_path) if session.asset_path is not None else "",
        "backend": session.adapter.caps.name,
        "physics": physics_state_to_dict(state),
        "active_keyframe": int(session.active_keyframe),
        "selection": int(session.selected),
        "selected_node": int(session.selected_node.node_id) if session.selected_node else -1,
        "visual_groups": {
            group.category: list(group.visible) for group in session.adapter.visual_groups()
        },
        "render_flags": {
            flag.value: bool(backend.get_flag(flag)) for flag in backend.render_options()
        },
        "camera": camera_bookmark(camera, camera_view or camera.view(), camera_source),
        "lights": [_light_to_json(light) for light in source.lights.lights] if source else [],
        "environment": _environment_to_json(source.lights.environment()) if source else None,
        "materials": [_material_to_json(material) for material in source.materials]
        if source
        else [],
    }


def restore_scene(snapshot, session, backend, camera, *, select_source=None):
    version = int(snapshot.get("version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported scene snapshot version: {version}")
    format_name = snapshot.get("format")
    if format_name != SCENE_SNAPSHOT_FORMAT and format_name not in LEGACY_SCENE_SNAPSHOT_FORMATS:
        raise ValueError(f"Unsupported scene snapshot format: {snapshot.get('format')}")
    expected = str(session.asset_path) if session.asset_path is not None else ""
    if snapshot.get("asset", "") != expected:
        raise ValueError("Scene snapshot belongs to a different model")
    result = session.restore_physics_state(
        physics_state_from_dict(snapshot["physics"]),
        active_keyframe=int(snapshot.get("active_keyframe", -1)),
    )
    if not result.ok:
        raise ValueError(result.message)
    for category, values in snapshot.get("visual_groups", {}).items():
        for index, visible in enumerate(values):
            session.submit(cmd.SetVisualGroup(category, index, bool(visible)))
    supported = {flag.value: flag for flag in backend.render_options()}
    for name, enabled in snapshot.get("render_flags", {}).items():
        if name in supported:
            backend.set_flag(supported[name], bool(enabled))
    for index, payload in enumerate(snapshot.get("lights", [])):
        session.submit(cmd.SetLight(index, _light_from_json(payload)))
    environment = snapshot.get("environment")
    if environment is not None:
        session.submit(cmd.SetEnvironment(_environment_from_json(environment)))
    for index, payload in enumerate(snapshot.get("materials", [])):
        session.submit(cmd.SetMaterial(index, _material_from_json(payload)))
    node_id = int(snapshot.get("selected_node", -1))
    if node_id >= 0:
        session.submit(cmd.SelectNode(node_id))
    else:
        session.submit(cmd.Select(int(snapshot.get("selection", 0))))
    return apply_camera_bookmark(snapshot["camera"], camera, select_source)


def save_named_snapshot(name: str, snapshot: dict[str, Any], directory=DEFAULT_DIRECTORY) -> Path:
    path = Path(directory) / f"{_safe_name(name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return path


def load_named_snapshot(name: str, directory=DEFAULT_DIRECTORY) -> dict[str, Any]:
    return json.loads((Path(directory) / f"{_safe_name(name)}.json").read_text())


def delete_named_snapshot(name: str, directory=DEFAULT_DIRECTORY) -> None:
    (Path(directory) / f"{_safe_name(name)}.json").unlink(missing_ok=True)


def list_named_snapshots(directory=DEFAULT_DIRECTORY) -> list[str]:
    root = Path(directory)
    return sorted(path.stem for path in root.glob("*.json")) if root.is_dir() else []


def next_available_snapshot_name(name: str, directory=DEFAULT_DIRECTORY) -> str:
    """Return a sanitized non-overwriting name for one UI snapshot save."""

    base = _safe_name(name)
    existing = set(list_named_snapshots(directory))
    if base not in existing:
        return base
    numbered = re.fullmatch(r"(.*?)-(\d+)", base)
    if numbered is None:
        prefix, index = base, 2
    else:
        prefix, index = numbered.group(1), int(numbered.group(2)) + 1
    candidate = f"{prefix}-{index}"
    while candidate in existing:
        index += 1
        candidate = f"{prefix}-{index}"
    return candidate


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")
    if not value:
        raise ValueError("Snapshot name is empty")
    return value


def _array(value) -> list:
    return np.asarray(value).tolist()


def physics_state_to_dict(state: PhysicsState) -> dict[str, Any]:
    arrays = {}
    for field in fields(state):
        if field.name == "time":
            continue
        value = np.asarray(getattr(state, field.name))
        arrays[field.name] = {"shape": list(value.shape), "values": value.reshape(-1).tolist()}
    arrays["time"] = state.time
    return arrays


def physics_state_from_dict(value: dict[str, Any]) -> PhysicsState:
    def array(name: str) -> np.ndarray:
        payload = value[name]
        return np.asarray(payload["values"], np.float64).reshape(payload["shape"])

    return PhysicsState(
        qpos=array("qpos"),
        qvel=array("qvel"),
        act=array("act"),
        ctrl=array("ctrl"),
        time=float(value["time"]),
        mocap_pos=array("mocap_pos"),
        mocap_quat=array("mocap_quat"),
    )


def _light_to_json(light: Light) -> dict[str, Any]:
    return {
        field.name: (
            int(value)
            if field.name == "type"
            else _array(value)
            if isinstance(value, np.ndarray)
            else value
        )
        for field in fields(light)
        if (value := getattr(light, field.name)) is not None
    }


def _light_from_json(value: dict[str, Any]) -> Light:
    arrays = {"position", "direction", "diffuse", "specular", "ambient", "attenuation"}
    return Light(
        **{
            name: LightType(item)
            if name == "type"
            else np.asarray(item, np.float32)
            if name in arrays
            else item
            for name, item in value.items()
        }
    )


def _environment_to_json(environment: Environment) -> dict[str, Any]:
    return {
        "headlight": _light_to_json(environment.headlight) if environment.headlight else None,
        "ambient": _array(environment.ambient),
        "fog_color": _array(environment.fog_color),
        "fog_start": environment.fog_start,
        "fog_end": environment.fog_end,
        "haze_color": _array(environment.haze_color),
        "haze_density": environment.haze_density,
        "horizon_haze": environment.horizon_haze,
        "horizon_haze_slices": environment.horizon_haze_slices,
    }


def _environment_from_json(value: dict[str, Any]) -> Environment:
    return Environment(
        headlight=_light_from_json(value["headlight"]) if value.get("headlight") else None,
        ambient=np.asarray(value["ambient"], np.float32),
        fog_color=np.asarray(value["fog_color"], np.float32),
        fog_start=float(value["fog_start"]),
        fog_end=float(value["fog_end"]),
        haze_color=np.asarray(value["haze_color"], np.float32),
        haze_density=float(value["haze_density"]),
        horizon_haze=bool(value.get("horizon_haze", False)),
        horizon_haze_slices=max(3, int(value.get("horizon_haze_slices", 64))),
    )


def _material_to_json(material: Material) -> dict[str, Any]:
    return {
        "name": material.name,
        "rgba": _array(material.rgba),
        "emission": material.emission,
        "specular": material.specular,
        "shininess": material.shininess,
        "reflectance": material.reflectance,
        "metallic": material.metallic,
        "roughness": material.roughness,
        "texture": material.texture,
        "tex_repeat": _array(material.tex_repeat),
        "tex_uniform": material.tex_uniform,
    }


def _material_from_json(value: dict[str, Any]) -> Material:
    value = value | {
        "metallic": float(value.get("metallic", -1.0)),
        "roughness": float(value.get("roughness", -1.0)),
    }
    return Material(
        **(
            value
            | {
                "rgba": np.asarray(value["rgba"], np.float32),
                "tex_repeat": np.asarray(value["tex_repeat"], np.float32),
            }
        )
    )
