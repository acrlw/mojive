"""Versioned local RPC for headless scene control and capture."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import FrameNeeds, PhysicsState
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

    def __init__(self, adapter, asset_path: Path | None = None) -> None:
        self.session = Session(adapter, asset_path)
        hint = adapter.camera_hint() or CameraView()
        self.camera = OrbitCamera()
        self.camera.adopt(hint)
        self.camera_source = -1
        self._renderer = None
        self._renderer_size = (0, 0)
        self._scene_option = None
        self._render_flags: dict[str, bool] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        """Release the cached renderer and adapter resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.session.adapter.release()

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
            "load": self._load,
            "reload": lambda _: self._command(cmd.Reload()),
            "pause": lambda _: self._command(cmd.Pause()),
            "resume": lambda _: self._command(cmd.Play()),
            "step": lambda p: self._step(p),
            "reset": lambda _: self._command(cmd.Reset()),
            "set_keyframe": lambda p: self._command(cmd.LoadKeyframe(int(p["keyframe_id"]))),
            "set_qpos": self._set_qpos,
            "get_state": lambda _: self._state(),
            "set_camera": self._set_camera,
            "load_camera_bookmark": self._load_camera_bookmark,
            "set_visual_group": self._set_visual_group,
            "set_render_flag": lambda p: self._set_option_flag(p, "render"),
            "set_visualization_flag": lambda p: self._set_option_flag(p, "visualization"),
            "capture": self._capture,
            "list_objects": lambda _: self._list_objects(),
            "select_object": self._select_object,
            "inspect_object": self._inspect_object,
        }
        handler = handlers.get(method)
        if handler is None:
            raise RpcError("unknown_method", f"Unknown control method: {method}")
        return handler(params)

    def _command(self, command) -> dict[str, Any]:
        result = self.session.submit(command)
        if not result.ok:
            raise RpcError("command_failed", result.message)
        return asdict(result)

    def _load(self, params: dict[str, Any]) -> dict[str, Any]:
        path = Path(params["path"]).expanduser().resolve()
        result = self._command(cmd.LoadAsset(path))
        self.camera_source = -1
        hint = self.session.adapter.camera_hint()
        if hint is not None:
            self.camera.adopt(hint)
        self._drop_renderer()
        return result

    def _step(self, params: dict[str, Any]) -> dict[str, Any]:
        result = self._command(cmd.Step(int(params.get("count", 1))))
        self.session.tick(FrameNeeds(), wall_dt=0.0)
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

    def _state(self) -> dict[str, Any]:
        state = self._require_state()
        self.session.tick(FrameNeeds(qpos=True, qvel=True), wall_dt=0.0)
        return {
            "asset": str(self.session.asset_path or ""),
            "paused": self.session.paused,
            "active_keyframe": self.session.active_keyframe,
            "selected_object_id": self.session.selected,
            "physics": physics_state_to_dict(state),
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

    def _select_object(self, params: dict[str, Any]) -> dict[str, Any]:
        object_id = int(params["object_id"])
        result = self._command(cmd.Select(object_id))
        result["object"] = (
            _node_payload(self.session.selected_node) if self.session.selected_node else None
        )
        return result

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


def _node_payload(node) -> dict[str, Any]:
    return {
        "node_id": int(node.node_id),
        "object_id": int(node.object_id),
        "name": node.name,
        "type": node.type.value,
        "parent": int(node.parent),
        "visible": bool(node.visible),
        "posable": bool(node.posable),
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
