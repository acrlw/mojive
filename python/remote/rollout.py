"""Optional pull-only rollout windows using stdlib HTTP and compact float32 qpos.

A producer publishes complete immutable windows. Clients request a bounded subset
only when instructed; publication never pushes frames or waits for a viewer.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    import mujoco

_MAGIC = b"MJVROL01"
_HEADER = struct.Struct("<8sI")
MAX_HEADER_BYTES = 64 * 1024
MAX_MODEL_BYTES = 128 * 1024 * 1024
DEFAULT_WINDOW_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RolloutWindow:
    """Owned compact poses and exact source identities for one complete rollout."""

    metadata: dict
    qpos: np.ndarray
    received_bytes: int = 0


class RolloutStore:
    """Retain just the latest completed window; meshes/model bytes are published once.

    ``publish`` copies CPU qpos once at a rollout boundary. Select/downsample on
    the training side before GPU-to-CPU transfer when monitoring fewer worlds.
    Pass original ``world_ids`` when publishing a preselected preview batch.
    """

    def __init__(
        self,
        model: bytes,
        metadata: dict,
        *,
        max_worlds=64,
        max_frames=512,
        max_window_bytes=DEFAULT_WINDOW_BYTES,
        max_source_bytes=256 * 1024 * 1024,
    ):
        self.metadata = json.loads(json.dumps(metadata))
        if (
            len(model) > MAX_MODEL_BYTES
            or hashlib.sha256(model).hexdigest() != metadata["model_sha256"]
        ):
            raise ValueError("Invalid rollout model size or checksum")
        if any(
            type(n) is not int or n < 1
            for n in (max_worlds, max_frames, max_window_bytes, max_source_bytes)
        ):
            raise ValueError("Rollout limits must be positive integers")
        if max_frames < 2:
            raise ValueError("A rollout window needs at least two frames")
        self.model = bytes(model)
        self.max_worlds, self.max_frames = max_worlds, max_frames
        self.max_window_bytes, self.max_source_bytes = max_window_bytes, max_source_bytes
        self._lock = threading.Lock()
        self._latest: RolloutWindow | None = None
        self.window_requests = 0
        self.sent_pose_bytes = 0

    @classmethod
    def from_archive(cls, archive: Path, **limits) -> RolloutStore:
        """Serve a stable read-only archive without reading/copying all recorded worlds."""
        archive = Path(archive)
        metadata = json.loads((archive / "manifest.json").read_text())
        store = cls((archive / "model.mjb").read_bytes(), metadata, **limits)
        qpos = np.load(archive / "qpos.npy", mmap_mode="r", allow_pickle=False)
        store._validate(qpos, 0)
        store._install(qpos, 0)
        return store

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        total_worlds: int,
        hz: float,
        display_spacing: float = 7.0,
        **limits,
    ) -> RolloutStore:
        """Prepare a MuJoCo/MJWarp producer once, without a diagnostic archive.

        Pass the CPU model used to initialize simulation. Publish selected CPU
        float32 qpos windows after collecting them at the declared cadence; this
        method neither imports Warp nor steps or reads back a simulation.
        """
        import mujoco

        if (
            type(total_worlds) is not int
            or total_worlds < 1
            or model.nq < 1
            or not math.isfinite(hz)
            or hz <= 0
            or not math.isfinite(display_spacing)
            or display_spacing <= 0
        ):
            raise ValueError("Invalid world count, joint layout, cadence, or display spacing")
        size = mujoco.mj_sizeModel(model)
        if size > MAX_MODEL_BYTES:
            raise ValueError("Rollout model exceeds the byte budget")
        buffer = np.empty(size, dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        data = buffer.tobytes()
        return cls(
            data,
            {
                "version": 1,
                "dtype": "float32",
                "shape": [2, total_worlds, model.nq],
                "hz": float(hz),
                "display_spacing": float(display_spacing),
                "model_sha256": hashlib.sha256(data).hexdigest(),
                "mujoco_version": mujoco.__version__,
                "joint_names": [model.joint(i).name for i in range(model.njnt)],
                "coordinates": "Z-up, MuJoCo qpos, root quaternion wxyz; origins for display only",
            },
            **limits,
        )

    def _validate(self, qpos, start_step, world_ids=None):
        if (
            qpos.ndim != 3
            or min(qpos.shape) < 1
            or qpos.shape[0] < 2
            or qpos.shape[2] != self.metadata["shape"][2]
            or qpos.dtype != np.float32
            or type(start_step) is not int
            or start_step < 0
        ):
            raise ValueError("Invalid rollout shape, dtype, or start step")
        ids = range(self.metadata["shape"][1]) if world_ids is None else world_ids
        if (
            len(ids) != qpos.shape[1]
            or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < self.metadata["shape"][1] for i in ids)
        ):
            raise ValueError("Invalid published world identities")
        if not math.isfinite(self.metadata["hz"]) or self.metadata["hz"] <= 0:
            raise ValueError("Invalid rollout cadence")

    def _install(self, qpos, start_step, world_ids=None):
        metadata = {
            **deepcopy(self.metadata),
            "shape": list(qpos.shape),
            "revision": uuid4().hex,
            "start_step": start_step,
            "total_worlds": self.metadata["shape"][1],
        }
        if world_ids is not None:
            metadata["world_ids"] = list(world_ids)
        with self._lock:
            self._latest = RolloutWindow(metadata, qpos)
        return metadata["revision"]

    def publish(self, qpos: np.ndarray, *, start_step: int = 0, world_ids=None) -> str:
        """Publish an owned CPU window atomically; failed writes keep the old window."""
        values = np.asarray(qpos)
        if values.nbytes > self.max_source_bytes:
            raise ValueError("Published rollout exceeds the source byte budget")
        world_ids = None if world_ids is None else tuple(world_ids)
        self._validate(values, start_step, world_ids)
        owned = np.array(values, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(owned).all():
            raise ValueError("Rollout contains non-finite qpos")
        owned.flags.writeable = False
        return self._install(owned, start_step, world_ids)

    def _snapshot(self):
        with self._lock:
            if self._latest is None:
                raise ValueError("No completed rollout has been published")
            return self._latest

    def info(self) -> dict:
        """Describe the latest complete window and bounded request limits."""
        latest = self._snapshot()
        return {
            **deepcopy(latest.metadata),
            "max_worlds": self.max_worlds,
            "max_frames": self.max_frames,
            "max_window_bytes": self.max_window_bytes,
        }

    def window(self, world_ids, frame_count: int, *, after: str = "") -> RolloutWindow | None:
        """Slice before serialization; unchanged revisions return no pose bytes."""
        ids = tuple(world_ids)
        latest = self._snapshot()
        if (
            not 1 <= len(ids) <= self.max_worlds
            or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < latest.metadata["total_worlds"] for i in ids)
        ):
            raise ValueError("Choose unique world IDs within the server preview limit")
        if type(frame_count) is not int or not 2 <= frame_count <= self.max_frames:
            raise ValueError(f"Window frames must be between 2 and {self.max_frames}")
        available = latest.metadata.get("world_ids")
        if available is None:
            columns = ids
        else:
            lookup = {world: column for column, world in enumerate(available)}
            if any(i not in lookup for i in ids):
                raise ValueError("Requested worlds are absent from the published preview window")
            columns = tuple(lookup[i] for i in ids)
        count = min(frame_count, len(latest.qpos))
        size = count * len(ids) * latest.qpos.shape[2] * 4
        if size > self.max_window_bytes:
            raise ValueError("Requested rollout exceeds the window byte budget")
        with self._lock:
            self.window_requests += 1
        if after == latest.metadata["revision"]:
            return None
        first = len(latest.qpos) - count
        # Slice time first, then gather worlds. Work scales with the requested window.
        values = np.ascontiguousarray(latest.qpos[first:, columns, :], dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("Selected rollout contains non-finite qpos")
        metadata = {
            **deepcopy(latest.metadata),
            "shape": list(values.shape),
            "world_ids": list(ids),
            "start_step": latest.metadata["start_step"] + first,
            "qpos_bytes_per_frame": values[0].nbytes,
        }
        with self._lock:
            self.sent_pose_bytes += values.nbytes
        return RolloutWindow(metadata, values)


class RolloutServer:
    """Serve manual preview requests on one bounded worker, loopback by default."""

    def __init__(self, store: RolloutStore, host: str = "127.0.0.1", port: int = 47651):
        self.store = store

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3.0)

            def log_message(self, *_args):
                pass

            def send(self, code, payload=b"", content_type="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                try:
                    if len(self.path) > 8192:
                        raise ValueError("Request path is too long")
                    parsed = urlparse(self.path)
                    if parsed.path == "/info":
                        self.send(200, json.dumps(store.info()).encode())
                    elif parsed.path == "/model":
                        self.send(200, store.model, "application/octet-stream")
                    elif parsed.path == "/window":
                        params = parse_qs(parsed.query, strict_parsing=True)
                        if set(params) - {"world_ids", "frames", "after"} or any(
                            len(v) != 1 for v in params.values()
                        ):
                            raise ValueError("Invalid rollout parameters")
                        ids = tuple(int(i) for i in params["world_ids"][0].split(","))
                        window = store.window(
                            ids, int(params["frames"][0]), after=params.get("after", [""])[0]
                        )
                        if window is None:
                            self.send(304)
                            return
                        header = json.dumps(window.metadata).encode()
                        if len(header) > MAX_HEADER_BYTES:
                            raise ValueError("Rollout metadata exceeds the header budget")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/vnd.mojive.rollout")
                        self.send_header(
                            "Content-Length", str(_HEADER.size + len(header) + window.qpos.nbytes)
                        )
                        self.end_headers()
                        self.wfile.write(_HEADER.pack(_MAGIC, len(header)))
                        self.wfile.write(header)
                        self.wfile.write(memoryview(window.qpos).cast("B"))
                    else:
                        self.send(404, b'{"error":"Unknown rollout resource"}')
                except (ValueError, KeyError) as error:
                    self.send(400, json.dumps({"error": str(error)}).encode())
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

        self._server = HTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound address, including an OS-selected ephemeral port."""
        return self._server.server_address

    def start(self) -> None:
        """Start the optional service; no polling or copies occur without requests."""
        if self._thread is not None:
            raise RuntimeError("Rollout server is already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the owned service and close its socket."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=4)
            self._thread = None
        self._server.server_close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.close()


class RolloutClient:
    """Fetch one bounded window on demand, with no timers or background polling."""

    def __init__(self, url: str, *, timeout: float = 3.0, max_bytes=DEFAULT_WINDOW_BYTES):
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Rollout URL must be an HTTP(S) server URL without query or fragment")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or type(max_bytes) is not int
            or max_bytes < 1
        ):
            raise ValueError("Rollout timeout and byte budget must be positive")
        self.url, self.timeout, self.max_bytes = url.rstrip("/"), timeout, max_bytes

    def _get(self, path, limit):
        started = time.monotonic()
        try:
            response = urlopen(self.url + path, timeout=self.timeout)
        except HTTPError as error:
            with error:
                if error.code == 304:
                    return None
                message = self._read(error, MAX_HEADER_BYTES, started).decode(errors="replace")
                raise ValueError(f"Rollout server returned {error.code}: {message}") from error
        with response:
            return self._read(response, limit, started)

    def _read(self, response, limit, started):
        size = int(response.headers.get("Content-Length", "-1"))
        if not 0 <= size <= limit:
            raise ValueError("Rollout response exceeds the byte budget or has no length")
        result = bytearray()
        while len(result) < size:
            # read() waits to fill its requested size; a trickling peer can keep
            # that call alive indefinitely. read1() returns after one socket read.
            chunk = response.read1(min(64 * 1024, size - len(result)))
            if time.monotonic() - started > self.timeout:
                raise TimeoutError("Rollout download exceeded its deadline")
            if not chunk:
                raise ValueError("Incomplete rollout response")
            result.extend(chunk)
        return result

    def info(self) -> dict:
        """Read source metadata without transferring a pose window."""
        return json.loads(self._get("/info", MAX_HEADER_BYTES))

    def model(self, expected_sha256: str) -> bytes:
        """Download and verify the model once when opening a source."""
        data = self._get("/model", MAX_MODEL_BYTES)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Rollout model checksum mismatch")
        return bytes(data)

    def fetch(self, world_ids, frame_count: int, *, after: str = "") -> RolloutWindow | None:
        """Return a complete validated float32 window, or None if it is unchanged."""
        ids = tuple(world_ids)
        query = urlencode(
            {
                "world_ids": ",".join(map(str, ids)),
                "frames": frame_count,
                **({"after": after} if after else {}),
            }
        )
        payload = self._get("/window?" + query, self.max_bytes + MAX_HEADER_BYTES + _HEADER.size)
        if payload is None:
            return None
        if len(payload) < _HEADER.size:
            raise ValueError("Invalid rollout envelope")
        magic, size = _HEADER.unpack_from(payload)
        if magic != _MAGIC or size > MAX_HEADER_BYTES or len(payload) < _HEADER.size + size:
            raise ValueError("Invalid rollout header")
        metadata = json.loads(payload[_HEADER.size : _HEADER.size + size])
        shape = metadata["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape)
            or not 2 <= shape[0] <= frame_count
            or shape[1] != len(ids)
            or metadata.get("world_ids") != list(ids)
            or metadata.get("dtype") != "float32"
            or not isinstance(metadata.get("revision"), str)
            or not metadata["revision"]
            or type(metadata.get("start_step")) is not int
            or metadata["start_step"] < 0
        ):
            raise ValueError("Rollout response does not match the requested window")
        data_size = math.prod(shape) * 4
        offset = _HEADER.size + size
        if data_size > self.max_bytes or len(payload) != offset + data_size:
            raise ValueError("Invalid rollout payload size")
        values = np.frombuffer(payload, dtype="<f4", offset=offset).reshape(shape)
        if not np.isfinite(values).all():
            raise ValueError("Rollout contains non-finite qpos")
        values.flags.writeable = False
        return RolloutWindow(metadata, values, len(payload))
