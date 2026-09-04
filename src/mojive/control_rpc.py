"""Versioned local RPC for headless scene control and capture."""

from __future__ import annotations

import json
import queue
import socket
import socketserver
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import FrameNeeds, PhysicsState
from .config import InteractionConfig, SelectionStyle
from .scene_state import (
    CAMERA_BOOKMARK_FORMAT,
    CAMERA_BOOKMARK_VERSION,
    DEFAULT_DIRECTORY,
    apply_camera_bookmark,
    camera_bookmark,
    load_named_snapshot,
    physics_state_to_dict,
)
from .session import Session
from .types import CameraView
from .ui.camera import OrbitCamera

PROTOCOL_VERSION = 1
DEFAULT_SOCKET = Path("output/mojive.sock")
CAMERA_DIRECTORY = DEFAULT_DIRECTORY / "cameras"


class RpcError(RuntimeError):
    """A structured RPC failure returned by the control service."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ControlService:
    """Route control requests through one Session and its scene adapter."""

    def __init__(
        self,
        adapter=None,
        asset_path: Path | None = None,
        *,
        session: Session | None = None,
        camera: OrbitCamera | None = None,
        app: Any | None = None,
    ) -> None:
        if session is None and adapter is None:
            raise TypeError("adapter or session is required")
        self._owns_session = session is None
        self.session = session or Session(adapter, asset_path)
        hint = self.session.adapter.camera_hint() or CameraView()
        self.camera = camera or OrbitCamera()
        if camera is None:
            self.camera.adopt(hint)
        self.app = app
        self.camera_source = -1
        self._renderer = None
        self._renderer_size = (0, 0)
        self._scene_option = None
        self._render_flags: dict[str, bool] = {}
        self._lock = threading.RLock()
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None

    def close(self) -> None:
        """Release the cached renderer and adapter resources."""
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=2.0)
            self._scheduler_thread = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._owns_session:
            self.session.release()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate one protocol request and return a serializable response."""
        request_id = request.get("id")
        version = request.get("version")
        if version != PROTOCOL_VERSION:
            return _response(
                request_id,
                error={
                    "code": "version_mismatch",
                    "message": f"Expected protocol version {PROTOCOL_VERSION}; received {version}",
                },
            )
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str) or not isinstance(params, dict):
            return _response(
                request_id,
                error={"code": "invalid_request", "message": "method and params are required"},
            )
        try:
            with self._lock:
                result = self.dispatch(method, params)
        except RpcError as exc:
            return _response(request_id, error={"code": exc.code, "message": str(exc)})
        except Exception as exc:
            return _response(
                request_id,
                error={"code": "internal_error", "message": str(exc)},
            )
        return _response(request_id, result=result)

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Invoke one typed control operation by protocol method name."""
        handlers = {
            "hello": lambda _: self._capabilities(),
            "get_capabilities": lambda _: self._capabilities(),
            "load": self._load,
            "reload": lambda _: self._reload(),
            "reset_layout": lambda _: self._reset_layout(),
            "pause": lambda _: self._command(cmd.Pause()),
            "resume": lambda _: self._resume(),
            "step": lambda p: self._step(p),
            "reset": lambda _: self._command(cmd.Reset()),
            "set_speed": lambda p: self._command(cmd.SetSpeed(float(p["factor"]))),
            "set_keyframe": lambda p: self._command(cmd.LoadKeyframe(int(p["keyframe_id"]))),
            "set_qpos": self._set_qpos,
            "set_qvel": lambda p: self._set_state_fields({"qvel": p["values"]}),
            "set_ctrl": self._set_ctrl,
            "set_mocap": self._set_mocap,
            "set_state": self._set_state_fields,
            "get_state": lambda _: self._state(),
            "get_scene": lambda _: self._scene(),
            "get_bounds": lambda _: self._bounds(),
            "set_camera": self._set_camera,
            "load_camera_bookmark": self._load_camera_bookmark,
            "set_visual_group": self._set_visual_group,
            "set_render_flag": lambda p: self._set_option_flag(p, "render"),
            "set_shadow_quality": self._set_shadow_quality,
            "set_visualization_flag": lambda p: self._set_option_flag(p, "visualization"),
            "capture": self._capture,
            "list_objects": lambda _: self._list_objects(),
            "select_object": self._select_object,
            "select_node": self._select_node,
            "inspect_object": self._inspect_object,
            "set_visible": self._set_visible,
            "get_viewer_settings": lambda _: self._viewer_settings(),
            "set_interactions": self._set_interactions,
            "set_selection_style": self._set_selection_style,
            "get_panels": lambda _: self._panels(),
            "set_panel": self._set_panel,
        }
        handler = handlers.get(method)
        if handler is None:
            raise RpcError("unknown_method", f"Unknown control method: {method}")
        return handler(params)

    def _capabilities(self) -> dict[str, Any]:
        methods = (
            "capture",
            "get_bounds",
            "get_capabilities",
            "get_panels",
            "get_scene",
            "get_state",
            "get_viewer_settings",
            "hello",
            "inspect_object",
            "list_objects",
            "load",
            "load_camera_bookmark",
            "pause",
            "reload",
            "reset_layout",
            "reset",
            "resume",
            "select_node",
            "select_object",
            "set_camera",
            "set_ctrl",
            "set_interactions",
            "set_keyframe",
            "set_mocap",
            "set_panel",
            "set_qpos",
            "set_qvel",
            "set_render_flag",
            "set_shadow_quality",
            "set_selection_style",
            "set_speed",
            "set_state",
            "set_visible",
            "set_visual_group",
            "set_visualization_flag",
            "step",
        )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "service": "mojive.control",
            "viewer_attached": self.app is not None,
            "methods": methods,
            "adapter": asdict(self.session.adapter.caps),
        }

    def _resume(self) -> dict[str, Any]:
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
            raise RpcError("command_failed", result.message)
        return asdict(result)

    def _load(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(params["path"]).expanduser().resolve()
        if self.app is not None:
            result = self.app.load_model(path)
            if not result.ok:
                raise RpcError("command_failed", result.message)
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

    def _reload(self) -> dict[str, Any]:
        if self.app is None:
            result = self._command(cmd.Reload())
            hint = self.session.adapter.camera_hint()
            if hint is not None:
                self.camera.adopt(hint)
            self._drop_renderer()
            return result
        result = self.app.reload_model()
        if not result.ok:
            raise RpcError("command_failed", result.message)
        self.camera_source = -1
        self._drop_renderer()
        return asdict(result)

    def _step(self, params: dict[str, Any]) -> dict[str, Any]:
        if "ctrl" in params:
            self._set_ctrl({"values": params["ctrl"]})
        result = self._command(cmd.Step(int(params.get("count", 1))))
        self.session.tick(FrameNeeds(), wall_dt=0.0)
        if bool(params.get("observe", False)):
            result["state"] = self._state()
        return result

    def _set_qpos(self, params: dict[str, Any]) -> dict[str, Any]:
        if "values" not in params:
            return self._command(cmd.SetQpos(int(params["index"]), float(params["value"])))
        state = self._require_state()
        values = np.asarray(params["values"], np.float64)
        if values.shape != state.qpos.shape:
            raise RpcError(
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
            raise RpcError("command_failed", result.message)
        return asdict(result)

    def _set_ctrl(self, params: dict[str, Any]) -> dict[str, Any]:
        if "values" not in params:
            return self._command(cmd.SetCtrl(int(params["index"]), float(params["value"])))
        state = self._require_state()
        values = self._vector("ctrl", params["values"], state.ctrl)
        results = [self._command(cmd.SetCtrl(index, value)) for index, value in enumerate(values)]
        return {
            "ok": True,
            "message": "",
            "entity_id": -1,
            "count": len(results),
        }

    def _set_mocap(self, params: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if "position" in params:
            updates["mocap_pos"] = params["position"]
        if "quaternion" in params:
            updates["mocap_quat"] = params["quaternion"]
        if not updates:
            raise RpcError("invalid_params", "position or quaternion is required")
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
            raise RpcError(
                "invalid_params", f"Unknown state field(s): {', '.join(sorted(unknown))}"
            )
        values = {
            name: self._vector(name, params[name], current) if name in params else current
            for name, current in aliases.items()
        }
        state_time = float(params.get("time", state.time))
        if not np.isfinite(state_time):
            raise RpcError("invalid_params", "time must be finite")
        updated = PhysicsState(time=state_time, **values)
        result = self.session.restore_physics_state(
            updated,
            active_keyframe=int(params.get("active_keyframe", self.session.active_keyframe)),
        )
        if not result.ok:
            raise RpcError("command_failed", result.message)
        return asdict(result)

    @staticmethod
    def _vector(name: str, value: object, expected: np.ndarray) -> np.ndarray:
        values = np.asarray(value, np.float64)
        if values.shape != expected.shape:
            raise RpcError(
                "invalid_params",
                f"Expected {name} shape {expected.shape}; received {values.shape}",
            )
        if not np.all(np.isfinite(values)):
            raise RpcError("invalid_params", f"{name} values must be finite")
        return values

    def _state(self) -> dict[str, Any]:
        self.session.tick(FrameNeeds(qpos=True, qvel=True, actuator=True), wall_dt=0.0)
        state = self.session.adapter.capture_state()
        return {
            "asset": str(self.session.asset_path or ""),
            "paused": self.session.paused,
            "active_keyframe": self.session.active_keyframe,
            "selected_object_id": self.session.selected,
            "speed": self.session.speed,
            "step": self.session.frame.step,
            "time": self.session.frame.time,
            "physics": physics_state_to_dict(state) if state is not None else None,
            "camera": camera_bookmark(self.camera, self.camera.view(), self.camera_source),
        }

    def _require_state(self) -> PhysicsState:
        state = self.session.adapter.capture_state()
        if state is None:
            raise RpcError("unsupported", "The scene adapter does not expose physics state")
        return state

    def _set_camera(self, params: dict[str, Any]) -> dict[str, Any]:
        if "camera_id" in params:
            camera_id = int(params["camera_id"])
            view = self.session.adapter.camera_view(camera_id)
            if view is None:
                raise RpcError("invalid_params", f"Camera {camera_id} is unavailable")
            self.camera_source = camera_id
            self.camera.adopt(view)
        else:
            self.camera_source = int(params.get("source", -1))
            bookmark = camera_bookmark(self.camera, self.camera.view(), self.camera_source)
            bookmark.update(params)
            bookmark["format"] = CAMERA_BOOKMARK_FORMAT
            bookmark["version"] = CAMERA_BOOKMARK_VERSION
            apply_camera_bookmark(bookmark, self.camera)
        return camera_bookmark(self.camera, self.camera.view(), self.camera_source)

    def _load_camera_bookmark(self, params: dict[str, Any]) -> dict[str, Any]:
        bookmark = load_named_snapshot(params["name"], params.get("directory", CAMERA_DIRECTORY))
        view = apply_camera_bookmark(bookmark, self.camera)
        self.camera_source = int(bookmark.get("source", -1))
        return camera_bookmark(self.camera, view, self.camera_source)

    def _set_visual_group(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._command(
            cmd.SetVisualGroup(
                str(params["category"]), int(params["group"]), bool(params["visible"])
            )
        )

    def _set_option_flag(self, params: dict[str, Any], family: str) -> dict[str, Any]:
        import mujoco

        enum_type = mujoco.mjtRndFlag if family == "render" else mujoco.mjtVisFlag
        prefix = "mjRND_" if family == "render" else "mjVIS_"
        name = str(params["name"])
        member = getattr(
            enum_type, name if name.startswith(prefix) else prefix + name.upper(), None
        )
        if member is None:
            raise RpcError("invalid_params", f"Unknown {family} flag: {name}")
        enabled = bool(params["enabled"])
        if family == "render":
            self._render_flags[member.name] = enabled
        else:
            self._option().flags[int(member)] = enabled
        return {"name": member.name, "enabled": enabled}

    def _option(self):
        if self._scene_option is None:
            import mujoco

            self._scene_option = mujoco.MjvOption()
        return self._scene_option

    def _capture(self, params: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        mode = str(params.get("mode", "rgb")).lower()
        if mode not in {"rgb", "depth", "segmentation"}:
            raise RpcError("invalid_params", f"Unknown capture mode: {mode}")
        width = int(params.get("width", 640))
        height = int(params.get("height", 480))
        renderer = self._capture_renderer(width, height)
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        if mode == "depth":
            renderer.enable_depth_rendering()
        elif mode == "segmentation":
            renderer.enable_segmentation_rendering()
        renderer.update_scene(
            self.session.adapter.data,
            camera=_mujoco_camera(self.camera.view()),
            scene_option=self._option(),
        )
        for name, enabled in self._render_flags.items():
            renderer.set_render_flag(name, enabled)
        image = renderer.render()
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
        }

    def _capture_renderer(self, width: int, height: int):
        if self._renderer is not None and self._renderer_size == (width, height):
            return self._renderer
        self._drop_renderer()
        from .renderer import Renderer

        model = self.session.adapter.model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self._renderer = Renderer(model, width=width, height=height)
        self._renderer_size = (width, height)
        return self._renderer

    def _drop_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._renderer_size = (0, 0)

    def _list_objects(self) -> list[dict[str, Any]]:
        return [_node_payload(node) for node in self.session.nodes]

    def _scene(self) -> dict[str, Any]:
        lo, hi = self.session.bounds()
        return {
            "asset": str(self.session.asset_path or ""),
            "structure_generation": self.session.structure_generation,
            "bounds": {"minimum": np.asarray(lo).tolist(), "maximum": np.asarray(hi).tolist()},
            "objects": self._list_objects(),
            "cameras": [
                {
                    "camera_id": int(camera.camera_id),
                    "name": camera.name,
                }
                for camera in self.session.cameras
            ],
        }

    def _bounds(self) -> dict[str, list[float]]:
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

    def _set_visible(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._command(cmd.SetVisible(int(params["node_id"]), bool(params["visible"])))

    def _require_viewer(self):
        if self.app is None:
            raise RpcError("unsupported", "This method requires RPC attached to an active viewer")
        return self.app

    def _viewer_settings(self) -> dict[str, Any]:
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

    def _reset_layout(self) -> dict[str, bool]:
        app = self._require_viewer()
        app.window.reset_layout()
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
            raise RpcError("invalid_params", f"Unsupported shadow quality: {params['quality']!r}")
        return {"quality": app.backend.get_shadow_quality().value}

    def _panels(self) -> list[dict[str, Any]]:
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
            raise RpcError("not_found", f"Panel {panel_id!r} is unavailable")
        if "enabled" in params:
            app.panels.set_enabled(panel_id, bool(params["enabled"]))
        opening = bool(params.get("open", False))
        if "open" in params and not app.panels.set_open(panel_id, opening) and opening:
            raise RpcError("command_failed", f"Panel {panel_id!r} is disabled")
        state = app.panels.state(panel_id)
        return {
            "id": panel.id,
            "name": panel.name,
            "enabled": state.enabled,
            "open": state.open,
        }

    def _inspect_object(self, params: dict[str, Any]) -> dict[str, Any]:
        if "object_id" in params:
            node = self.session.node_by_object_id(int(params["object_id"]))
        else:
            node = self.session.node(int(params["node_id"]))
        if node is None:
            raise RpcError("not_found", "Scene object is unavailable")
        payload = _node_payload(node)
        payload["children"] = [
            _node_payload(child)
            for child_id in node.children
            if (child := self.session.node(child_id)) is not None
        ]
        return payload


@dataclass
class _PendingRequest:
    request: dict[str, Any]
    ready: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class ViewerControlService:
    """Marshal RPC requests onto an interactive viewer's UI thread."""

    def __init__(self, viewer) -> None:
        self._core = ControlService(
            session=viewer.session,
            camera=viewer.app.camera,
            app=viewer.app,
        )
        self._pending: queue.Queue[_PendingRequest] = queue.Queue()
        self._closed = False

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            return _response(
                request.get("id"),
                error={"code": "unavailable", "message": "Viewer RPC service is closed"},
            )
        pending = _PendingRequest(request)
        self._pending.put(pending)
        pending.ready.wait()
        assert pending.response is not None
        return pending.response

    def pump(self, limit: int = 64) -> int:
        """Execute queued requests on the caller's viewer thread."""

        handled = 0
        while handled < max(1, int(limit)):
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                break
            pending.response = self._core.handle(pending.request)
            pending.ready.set()
            handled += 1
        return handled

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                break
            pending.response = _response(
                pending.request.get("id"),
                error={"code": "unavailable", "message": "Viewer RPC service is closed"},
            )
            pending.ready.set()
        self._core.close()


class ViewerRpcServer:
    """Background socket transport attached to one interactive viewer."""

    def __init__(self, viewer, socket_path: Path = DEFAULT_SOCKET) -> None:
        self.service = ViewerControlService(viewer)
        self.server = ControlServer(socket_path, self.service)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="mojive-viewer-rpc",
            daemon=True,
        )
        viewer.app._rpc_service = self.service
        self._viewer = viewer
        self._closed = False
        self.thread.start()

    @property
    def socket_path(self) -> Path:
        return self.server.socket_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self.service.close()
        if getattr(self._viewer.app, "_rpc_service", None) is self.service:
            self._viewer.app._rpc_service = None
        self.thread.join(timeout=2.0)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                line = self.rfile.readline()
            except OSError:
                return
            if not line:
                return
            try:
                request = json.loads(line)
                response = self.server.service.handle(request)
            except Exception as exc:
                response = _response(None, error={"code": "invalid_request", "message": str(exc)})
            try:
                self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                self.wfile.flush()
            except OSError:
                return


class ControlServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Concurrent newline-delimited JSON server over an AF_UNIX socket."""

    daemon_threads = True

    def __init__(self, socket_path: Path, service: ControlService) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.service = service
        super().__init__(str(self.socket_path), _RequestHandler)
        self.socket_path.chmod(0o600)

    def server_close(self) -> None:
        super().server_close()
        self.socket_path.unlink(missing_ok=True)


class RpcClient:
    """Persistent local control client with correlation, timeouts, and recovery."""

    def __init__(self, socket_path: Path = DEFAULT_SOCKET, timeout: float = 5.0) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.timeout = float(timeout)
        self._next_id = 1
        self._client: socket.socket | None = None
        self._lock = threading.Lock()

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send one request, validate correlation metadata, and return its result."""
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            request = {
                "version": PROTOCOL_VERSION,
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            try:
                client = self._connect()
                client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                response = _read_response(client)
            except TimeoutError as exc:
                self.close()
                raise RpcError(
                    "timeout", f"RPC request timed out after {self.timeout:g} seconds"
                ) from exc
            except RpcError:
                self.close()
                raise
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.close()
                raise RpcError("connection_failed", str(exc)) from exc
            if response.get("version") != PROTOCOL_VERSION:
                self.close()
                raise RpcError("invalid_response", "RPC response version is incompatible")
            if response.get("id") != request_id:
                self.close()
                raise RpcError("invalid_response", "RPC response ID does not match the request")
            if response.get("error"):
                error = response["error"]
                raise RpcError(error["code"], error["message"])
            return response.get("result")

    def hello(self) -> dict[str, Any]:
        return self.call("hello")

    def get_state(self) -> dict[str, Any]:
        return self.call("get_state")

    def set_ctrl(self, values) -> dict[str, Any]:
        return self.call("set_ctrl", {"values": list(values)})

    def step(self, count: int = 1, *, ctrl=None, observe: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"count": int(count), "observe": bool(observe)}
        if ctrl is not None:
            params["ctrl"] = list(ctrl)
        return self.call("step", params)

    def _connect(self) -> socket.socket:
        if self._client is None:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(self.timeout)
            try:
                client.connect(str(self.socket_path))
            except Exception:
                client.close()
                raise
            self._client = client
        else:
            self._client.settimeout(self.timeout)
        return self._client

    def close(self) -> None:
        """Close the persistent connection; the next call reconnects."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> RpcClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _read_response(client: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = client.recv(65536)
        if not chunk:
            break
        data.extend(chunk)
    if not data:
        raise RpcError("invalid_response", "RPC server closed without a response")
    return json.loads(data)


def _response(request_id, *, result=None, error=None) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "id": request_id,
        "result": result if error is None else None,
        "error": error,
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


def _mujoco_camera(view: CameraView):
    import mujoco

    delta = np.asarray(view.eye, np.float64) - np.asarray(view.target, np.float64)
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    direction = delta / distance
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = view.target
    camera.distance = distance
    camera.azimuth = float(np.degrees(np.arctan2(direction[1], direction[0]))) + 180.0
    camera.elevation = float(-np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    camera.orthographic = int(view.orthographic)
    return camera
