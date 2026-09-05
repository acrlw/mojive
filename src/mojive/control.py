"""Transport-independent scene operations and application resource ownership."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import FrameNeeds, PhysicsState
from .camera_control import CameraState, update_camera
from .config import InteractionConfig, SelectionStyle
from .control_errors import ControlError
from .control_schema import json_value
from .geometry import geometry_dimensions
from .operations import (
    CAPTURE_PRODUCTS,
    CAPTURE_RENDER_FLAGS,
    CAPTURE_VISUAL_FLAGS,
    COMMAND_ID_FIELDS,
    OPERATIONS,
    document_state,
    prepare_operation,
)
from .render.backend import DebugView, RenderFlag
from .scene_queries import node_geometry_indices, node_hierarchy_visible, node_world_pose
from .scene_state import (
    DEFAULT_DIRECTORY,
    apply_camera_bookmark,
    camera_bookmark,
    load_named_snapshot,
    physics_state_to_dict,
)
from .session import Session
from .session_capture import SessionCapture
from .types import DEFAULT_MATERIAL, CameraView

CAMERA_DIRECTORY = DEFAULT_DIRECTORY / "cameras"


class ControlApplication:
    """Route control requests through one Session and its scene adapter."""

    def __init__(
        self,
        adapter=None,
        asset_path: Path | None = None,
        *,
        session: Session | None = None,
        camera=None,
        app: Any | None = None,
    ) -> None:
        if session is None and adapter is None:
            raise TypeError("adapter or session is required")
        self._owns_session = session is None
        self.session = session or Session(adapter, asset_path)
        if self._owns_session and app is None and self.session.adapter.caps.simulation:
            self.session.submit(cmd.Pause())
        hint = self.session.adapter.camera_hint() or CameraView()
        self.camera = CameraState(camera.view() if camera is not None else hint)
        self.app = app
        self.camera_source = -1
        self._capture_document_id = self.session.document_id
        self._capture_service = SessionCapture(self.session, threaded=app is None)
        self._capture_debug_flags: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._closed = False

    @property
    def lock(self):
        """Return the session serialization lock shared with protocol deadlines."""
        return self._lock

    def close(self) -> None:
        """Release the cached renderer and adapter resources."""
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=2.0)
            self._scheduler_thread = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._capture_service.close()
            if self._owns_session:
                self.session.release()

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Invoke one typed control operation by protocol method name."""
        with self._lock:
            if self._closed:
                raise ControlError("unavailable", "The application service is closed")
            self._sync_capture_document()
            operation, values = prepare_operation(
                self.session, method, params, viewer_attached=self.app is not None
            )
            if operation.handler:
                result = getattr(self, operation.handler)(values)
            else:
                result = self._command(operation.command(values))
            if isinstance(result, dict) and "ok" in result and "entity_id" in result:
                result["document"] = document_state(self.session)
            return json_value(result)

    def _sync_capture_document(self):
        if self.session.document_id != self._capture_document_id:
            self._capture_document_id = self.session.document_id
            self.camera_source = -1
            self.camera.adopt(self.session.adapter.camera_hint() or CameraView())
            self._capture_service.reset()

    def _capture_camera(self):
        if self.camera_source >= 0:
            view = self.session.camera_view(self.camera_source)
            if view is None:
                self.camera_source = -1
            else:
                self.camera.adopt(view)
        return camera_bookmark(self.camera, self.camera.view(), self.camera_source)

    def _capabilities(self, _params=None) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "service": "mojive.control",
            "viewer_attached": self.app is not None,
            "deadline_clock": "monotonic",
            "methods": tuple(sorted(OPERATIONS)),
            "available_methods": tuple(
                sorted(
                    name
                    for name, operation in OPERATIONS.items()
                    if operation.unavailable_reason(
                        self.session, viewer_attached=self.app is not None
                    )
                    is None
                )
            ),
            "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
            "document": document_state(self.session),
            "capture_modes": tuple(CAPTURE_PRODUCTS),
            "capture_scope": "session_scene",
            "adapter": asdict(self.session.adapter.caps),
        }

    def _describe_operations(self, params):
        selected = list(OPERATIONS.values())
        if "name" in params:
            if params["name"] not in OPERATIONS:
                raise ControlError("unknown_method", f"Unknown control method: {params['name']}")
            selected = [OPERATIONS[params["name"]]]
        if "scope" in params:
            selected = [item for item in selected if item.scope == params["scope"]]
        descriptions = [
            item.describe(self.session, viewer_attached=self.app is not None) for item in selected
        ]
        if params.get("available_only"):
            descriptions = [item for item in descriptions if item["available"]]
        return {
            "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
            "document": document_state(self.session),
            "operations": descriptions,
        }

    def _resume(self, _params=None) -> dict[str, Any]:
        result = self._command(cmd.Play())
        if self.app is None:
            self._start_scheduler()
        return result

    def _start_scheduler(self) -> None:
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            return
        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="mojive-rpc-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        previous = time.perf_counter()
        while not self._scheduler_stop.wait(1.0 / 240.0):
            now = time.perf_counter()
            elapsed = min(0.1, now - previous)
            previous = now
            with self._lock:
                self.session.tick(FrameNeeds.none(), wall_dt=elapsed)

    def _command(self, command) -> dict[str, Any]:
        result = self.session.submit(command)
        if not result.ok:
            raise ControlError("command_failed", result.message)
        payload = {**asdict(result), "document": document_state(self.session)}
        if identity := COMMAND_ID_FIELDS.get(type(command)):
            payload[identity] = result.entity_id
        return payload

    def _load(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(params["path"]).expanduser().resolve()
        if self.app is not None:
            result = self.app.load_model(path)
            if not result.ok:
                raise ControlError("command_failed", result.message)
            self.camera_source = -1
            self._drop_renderer()
            return asdict(result)
        result = self._command(cmd.LoadAsset(path))
        self.camera_source = -1
        hint = self.session.adapter.camera_hint()
        if hint is not None:
            self.camera.adopt(hint)
        self._drop_renderer()
        return result

    def _reload(self, _params=None) -> dict[str, Any]:
        if self.app is None:
            result = self._command(cmd.Reload())
            hint = self.session.adapter.camera_hint()
            if hint is not None:
                self.camera.adopt(hint)
            self._drop_renderer()
            return result
        result = self.app.reload_model()
        if not result.ok:
            raise ControlError("command_failed", result.message)
        self.camera_source = -1
        self._drop_renderer()
        return asdict(result)

    def _step(self, params: dict[str, Any]) -> dict[str, Any]:
        if "ctrl" in params:
            self._set_ctrl({"values": params["ctrl"]})
        result = self._command(OPERATIONS["step"].command(params))
        self.session.tick(FrameNeeds(), wall_dt=0.0)
        if bool(params.get("observe", False)):
            result["state"] = self._state()
        return result

    def _set_qpos(self, params: dict[str, Any]) -> dict[str, Any]:
        if "values" not in params:
            return self._command(OPERATIONS["set_qpos"].command(params))
        state = self._require_state()
        values = np.asarray(params["values"], np.float64)
        if values.shape != state.qpos.shape:
            raise ControlError(
                "invalid_params",
                f"Expected qpos shape {state.qpos.shape}; received {values.shape}",
            )
        updated = PhysicsState(
            qpos=values,
            qvel=state.qvel,
            act=state.act,
            ctrl=state.ctrl,
            time=state.time,
            mocap_pos=state.mocap_pos,
            mocap_quat=state.mocap_quat,
        )
        result = self.session.restore_physics_state(
            updated, active_keyframe=self.session.active_keyframe
        )
        if not result.ok:
            raise ControlError("command_failed", result.message)
        return asdict(result)

    def _set_ctrl(self, params: dict[str, Any]) -> dict[str, Any]:
        if "values" not in params:
            return self._command(OPERATIONS["set_ctrl"].command(params))
        state = self._require_state()
        values = self._vector("ctrl", params["values"], state.ctrl)
        results = [self._command(cmd.SetCtrl(index, value)) for index, value in enumerate(values)]
        return {
            "ok": True,
            "message": "",
            "entity_id": -1,
            "count": len(results),
        }

    def _set_qvel(self, params):
        return self._set_state_fields({"qvel": params["values"]})

    def _set_render_flag(self, params):
        return self._set_option_flag(params, "render")

    def _set_visualization_flag(self, params):
        return self._set_option_flag(params, "visualization")

    def _edit_scene(self, params):
        operations = []
        for index, item in enumerate(params["operations"]):
            operation, values = prepare_operation(self.session, item["method"], item["params"])
            if not operation.transactional or operation.command is None:
                raise ControlError(
                    "invalid_params",
                    f"{item['method']} cannot be part of an edit transaction",
                    details={"index": index, "method": item["method"]},
                )
            operations.append((operation, operation.command(values)))
        self._command(cmd.BeginEditTransaction(params.get("label", "Edit scene")))
        results = []
        try:
            for index, (operation, command) in enumerate(operations):
                try:
                    results.append(self._command(command))
                except ControlError as exc:
                    raise ControlError(
                        exc.code, str(exc), details={"index": index, "method": operation.name}
                    ) from exc
            result = self._command(cmd.EndEditTransaction())
        except Exception:
            if self.session.editing:
                rollback = self.session.submit(cmd.CancelEditTransaction())
                if not rollback.ok:
                    raise ControlError("rollback_failed", rollback.message) from None
            raise
        result["results"] = results
        return result

    def _set_mocap(self, params: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if "position" in params:
            updates["mocap_pos"] = params["position"]
        if "quaternion" in params:
            updates["mocap_quat"] = params["quaternion"]
        if not updates:
            raise ControlError("invalid_params", "position or quaternion is required")
        return self._set_state_fields(updates)

    def _set_state_fields(self, params: dict[str, Any]) -> dict[str, Any]:
        state = self._require_state()
        aliases = {
            "qpos": state.qpos,
            "qvel": state.qvel,
            "act": state.act,
            "ctrl": state.ctrl,
            "mocap_pos": state.mocap_pos,
            "mocap_quat": state.mocap_quat,
        }
        unknown = set(params) - {*aliases, "time", "active_keyframe"}
        if unknown:
            raise ControlError(
                "invalid_params", f"Unknown state field(s): {', '.join(sorted(unknown))}"
            )
        values = {
            name: self._vector(name, params[name], current) if name in params else current
            for name, current in aliases.items()
        }
        state_time = float(params.get("time", state.time))
        if not np.isfinite(state_time):
            raise ControlError("invalid_params", "time must be finite")
        updated = PhysicsState(time=state_time, **values)
        result = self.session.restore_physics_state(
            updated,
            active_keyframe=int(params.get("active_keyframe", self.session.active_keyframe)),
        )
        if not result.ok:
            raise ControlError("command_failed", result.message)
        return asdict(result)

    @staticmethod
    def _vector(name: str, value: object, expected: np.ndarray) -> np.ndarray:
        values = np.asarray(value, np.float64)
        if values.shape != expected.shape:
            raise ControlError(
                "invalid_params",
                f"Expected {name} shape {expected.shape}; received {values.shape}",
            )
        if not np.all(np.isfinite(values)):
            raise ControlError("invalid_params", f"{name} values must be finite")
        return values

    def _state(self, _params=None) -> dict[str, Any]:
        self.session.tick(FrameNeeds(qpos=True, qvel=True, actuator=True), wall_dt=0.0)
        state = self.session.adapter.capture_state()
        return {
            "document": document_state(self.session),
            "asset": str(self.session.asset_path or ""),
            "paused": self.session.paused,
            "active_keyframe": self.session.active_keyframe,
            "selected_object_id": self.session.selected,
            "speed": self.session.speed,
            "step": self.session.frame.step,
            "time": self.session.frame.time,
            "physics": physics_state_to_dict(state) if state is not None else None,
            "camera": self._capture_camera(),
        }

    def _require_state(self) -> PhysicsState:
        state = self.session.adapter.capture_state()
        if state is None:
            raise ControlError("unsupported", "The scene adapter does not expose physics state")
        return state

    def _set_camera(self, params: dict[str, Any]) -> dict[str, Any]:
        view, source = update_camera(self.camera.view(), params, self.session)
        self.camera.adopt(view)
        self.camera_source = source
        return camera_bookmark(self.camera, self.camera.view(), self.camera_source)

    def _load_camera_bookmark(self, params: dict[str, Any]) -> dict[str, Any]:
        bookmark = load_named_snapshot(params["name"], params.get("directory", CAMERA_DIRECTORY))
        view = apply_camera_bookmark(bookmark, object())
        values = {
            name: json_value(getattr(view, name))
            for name in OPERATIONS["set_capture_camera"].input_schema["properties"]
            if hasattr(view, name)
        }
        values["source"] = int(bookmark.get("source", -1))
        return self._set_camera(values)

    def _capture_settings(self, _params=None):
        self.session.tick(FrameNeeds.none(), wall_dt=0.0)
        return {
            "scope": "capture",
            "camera": self._capture_camera(),
            "supported_render_flags": sorted(flag.value for flag in CAPTURE_RENDER_FLAGS),
            "supported_visualization_flags": sorted(flag.value for flag in CAPTURE_VISUAL_FLAGS),
            "render_flag_overrides": {
                flag.value: enabled for flag, enabled in self._capture_service.flags.items()
            },
            "debug_view": (self._capture_service.debug_view or DebugView.SHADED).value,
            "dynamic_opacity": self._capture_service.dynamic_opacity,
            "modes": list(CAPTURE_PRODUCTS),
        }

    def _viewport_camera(self, _params=None):
        app = self._require_viewer()
        return app.viewport_camera_state()

    def _set_viewport_camera(self, params):
        app = self._require_viewer()
        view, source = update_camera(self.session.camera, params, self.session)
        app.set_viewport_camera(view, camera_id=source)
        return app.viewport_camera_state()

    def _capture_viewport(self, params):
        app = self._require_viewer()
        return app.request_capture_async(
            params.get("output"), surface=params.get("surface", "viewport")
        )

    def _set_option_flag(self, params: dict[str, Any], family: str) -> dict[str, Any]:
        prefix = "mjRND_" if family == "render" else "mjVIS_"
        name = str(params.get("name", "")).removeprefix(prefix)
        if family == "visualization" and name.upper() == "TRANSPARENT":
            enabled = params["enabled"]
            self._capture_service.dynamic_opacity = 0.3 if enabled else 1.0
            return {"name": prefix + "TRANSPARENT", "enabled": enabled}
        debug_views = {
            "DEPTH": DebugView.DEPTH,
            "IDCOLOR": DebugView.IDCOLOR,
            "SEGMENT": DebugView.SEGMENT,
        }
        if family == "render" and name.upper() in debug_views:
            enabled = params["enabled"]
            self._capture_debug_flags[name.upper()] = enabled
            self._capture_service.debug_view = next(
                (view for key, view in debug_views.items() if self._capture_debug_flags.get(key)),
                DebugView.SHADED,
            )
            return {"name": prefix + name.upper(), "enabled": enabled}
        flag = RenderFlag.__members__.get(name.upper())
        if flag is None:
            flag = next((item for item in RenderFlag if item.value == name.lower()), None)
        if flag is not None and (flag in CAPTURE_RENDER_FLAGS) != (family == "render"):
            flag = None
        if flag in {RenderFlag.OUTLINE, RenderFlag.TONEMAP, RenderFlag.MSAA}:
            flag = None
        if flag is None:
            raise ControlError("invalid_params", f"Unknown {family} flag: {name}")
        enabled = params["enabled"]
        self._capture_service.flags[flag] = enabled
        return {"name": prefix + flag.name, "enabled": enabled}

    def _capture(self, params: dict[str, Any]) -> dict[str, Any]:
        self._capture_camera()
        from PIL import Image

        mode = params.get("mode", "rgb")
        width = params.get("width", 640)
        height = params.get("height", 480)
        try:
            image = self._capture_service.render(
                self.camera.view(),
                width=width,
                height=height,
                product=CAPTURE_PRODUCTS[mode],
                camera_id=self.camera_source,
            )
        except NotImplementedError as exc:
            raise ControlError("unsupported", str(exc)) from exc
        default_suffix = ".png" if mode == "rgb" else ".npy"
        output = Path(params.get("output", f"output/rpc/{mode}{default_suffix}"))
        output.parent.mkdir(parents=True, exist_ok=True)
        if mode == "rgb":
            Image.fromarray(image, "RGB").save(output)
        else:
            np.save(output, image)
            if output.suffix != ".npy":
                output = output.with_suffix(output.suffix + ".npy")
        return {
            "path": str(output.resolve()),
            "mode": mode,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "orientation": "top_left",
            "scope": "session_scene",
            "document": document_state(self.session),
            "structure_generation": self.session.structure_generation,
            "step": self.session.frame.step,
            "time": self.session.frame.time,
        }

    def _drop_renderer(self) -> None:
        self._capture_service.reset()

    def _list_objects(self, _params=None) -> list[dict[str, Any]]:
        self.session.tick(FrameNeeds.none(), wall_dt=0.0)
        return [_node_payload(node) for node in self.session.nodes]

    def _scene(self, _params=None) -> dict[str, Any]:
        objects = self._list_objects()
        lo, hi = self.session.bounds()
        return {
            "document": document_state(self.session),
            "asset": str(self.session.asset_path or ""),
            "structure_generation": self.session.structure_generation,
            "bounds": {"minimum": np.asarray(lo).tolist(), "maximum": np.asarray(hi).tolist()},
            "objects": objects,
            "cameras": [
                {
                    "camera_id": int(camera.camera_id),
                    "object_id": int(camera.object_id),
                    "name": camera.name,
                }
                for camera in self.session.cameras
            ],
        }

    def _bounds(self, _params=None) -> dict[str, list[float]]:
        self.session.tick(FrameNeeds.none(), wall_dt=0.0)
        lo, hi = self.session.bounds()
        return {"minimum": np.asarray(lo).tolist(), "maximum": np.asarray(hi).tolist()}

    def _select_object(self, params: dict[str, Any]) -> dict[str, Any]:
        object_id = int(params["object_id"])
        result = self._command(cmd.Select(object_id))
        result["object"] = (
            _node_payload(self.session.selected_node) if self.session.selected_node else None
        )
        return result

    def _select_node(self, params: dict[str, Any]) -> dict[str, Any]:
        result = self._command(cmd.SelectNode(int(params["node_id"])))
        result["object"] = (
            _node_payload(self.session.selected_node) if self.session.selected_node else None
        )
        return result

    def _require_viewer(self):
        if self.app is None:
            raise ControlError(
                "unsupported", "This method requires RPC attached to an active viewer"
            )
        return self.app

    def _viewer_settings(self, _params=None) -> dict[str, Any]:
        app = self._require_viewer()
        return {
            "interactions": asdict(app.interactions),
            "selection_style": asdict(app.selection_style),
            "shadow_quality": app.backend.get_shadow_quality().value,
            "layout": {
                "persistence": bool(app.window.config.ini_path),
                "path": app.window.config.ini_path or None,
            },
            "panels": self._panels(),
        }

    def _reset_layout(self, _params=None) -> dict[str, bool]:
        app = self._require_viewer()
        app.reset_layout()
        return {"reset": True}

    def _set_interactions(self, params: dict[str, Any]) -> dict[str, Any]:
        app = self._require_viewer()
        changes = {key: value for key, value in params.items() if key != "persist"}
        merged = _deep_merge(asdict(app.interactions), changes)
        app.set_interactions(
            InteractionConfig.from_mapping(merged),
            persist=bool(params.get("persist", False)),
        )
        return asdict(app.interactions)

    def _set_selection_style(self, params: dict[str, Any]) -> dict[str, Any]:
        app = self._require_viewer()
        changes = {key: value for key, value in params.items() if key != "persist"}
        merged = _deep_merge(asdict(app.selection_style), changes)
        app.set_selection_style(
            SelectionStyle.from_mapping(merged),
            persist=bool(params.get("persist", False)),
        )
        return asdict(app.selection_style)

    def _set_shadow_quality(self, params: dict[str, Any]) -> dict[str, str]:
        app = self._require_viewer()
        if not app.set_shadow_quality(
            params["quality"],
            persist=bool(params.get("persist", False)),
        ):
            raise ControlError(
                "invalid_params", f"Unsupported shadow quality: {params['quality']!r}"
            )
        return {"quality": app.backend.get_shadow_quality().value}

    def _panels(self, _params=None) -> list[dict[str, Any]]:
        app = self._require_viewer()
        return [
            {
                "id": panel.id,
                "name": panel.name,
                "enabled": bool(panel.enabled),
                "open": bool(panel.open),
            }
            for panel in app.panels
        ]

    def _set_panel(self, params: dict[str, Any]) -> dict[str, Any]:
        app = self._require_viewer()
        panel_id = str(params["id"])
        panel = app.panels.get(panel_id)
        if panel is None:
            raise ControlError("not_found", f"Panel {panel_id!r} is unavailable")
        if "enabled" in params:
            app.panels.set_enabled(panel_id, bool(params["enabled"]))
        opening = bool(params.get("open", False))
        if "open" in params and not app.panels.set_open(panel_id, opening) and opening:
            raise ControlError("command_failed", f"Panel {panel_id!r} is disabled")
        state = app.panels.state(panel_id)
        return {
            "id": panel.id,
            "name": panel.name,
            "enabled": state.enabled,
            "open": state.open,
        }

    def _inspect_object(self, params: dict[str, Any]) -> dict[str, Any]:
        self.session.tick(FrameNeeds(), wall_dt=0.0)
        if "object_id" in params:
            node = self.session.node_by_object_id(int(params["object_id"]))
        else:
            node = self.session.node(int(params["node_id"]))
        if node is None:
            raise ControlError("not_found", "Scene object is unavailable")
        payload = _node_payload(node)
        position, rotation = node_world_pose(self.session, node)
        payload.update(
            position=position.tolist(),
            rotation=rotation.tolist(),
            document=document_state(self.session),
            structure_generation=self.session.structure_generation,
            hierarchy_visible=node_hierarchy_visible(self.session, node),
            geometries=[
                self._geometry_payload(int(index))
                for index in node_geometry_indices(self.session, node)
            ],
        )
        payload["children"] = [
            _node_payload(child)
            for child_id in node.children
            if (child := self.session.node(child_id)) is not None
        ]
        return payload

    def _geometry_payload(self, index: int) -> dict:
        source = self.session.source
        material_index = source.geom_material[index] if index < len(source.geom_material) else 0
        if 0 <= material_index < len(source.materials):
            material = source.materials[material_index]
        else:
            material_index, material = -1, DEFAULT_MATERIAL
        size = source.geom_size[index]
        dimensions = geometry_dimensions(source.geom_mesh[index].shape, size)
        return {
            "instance_index": index,
            "node_id": int(source.geom_node[index]),
            "object_id": int(source.geom_object_id[index]),
            "mesh": source.geom_mesh[index],
            "size": size.copy(),
            "rgba": source.geom_rgba[index].copy(),
            "material_index": int(material_index),
            "material": material,
            "dimensions": {"label": dimensions.label, "values": dimensions.values}
            if dimensions
            else None,
        }


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge nested settings dictionaries without discarding unspecified fields."""

    result = dict(base)
    for key, value in updates.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _node_payload(node) -> dict[str, Any]:
    return {
        "node_id": int(node.node_id),
        "object_id": int(node.object_id),
        "name": node.name,
        "type": node.type.value,
        "parent": int(node.parent),
        "visible": bool(node.visible),
        "posable": bool(node.posable),
        "source_editable": bool(node.source_editable),
        "body_index": int(node.body_index),
        "site_index": int(node.site_index),
    }
