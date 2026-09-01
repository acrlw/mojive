"""Local debug drawing bridge and command transport."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import selectors
import socket
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log import get_logger
from .render.debugdraw import NEVER, DebugDraw, Layer, Occlusion

log = get_logger("bridge")

APP = "mojive"
DEFAULT_BUDGET = 4096


_WHITE = (1.0, 1.0, 1.0, 1.0)
_ACTIVE_PATHS: set[Path] = set()


def socket_dir(app: str = APP) -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return Path(base) / app


def socket_path(pid: int | None = None, app: str = APP) -> Path:
    return socket_dir(app) / f"{os.getpid() if pid is None else int(pid)}.sock"


def live_sockets(app: str = APP) -> list[Path]:
    out: list[Path] = []
    d = socket_dir(app)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.sock")):
        try:
            pid = int(p.stem)
        except ValueError:
            continue
        if _alive(pid):
            out.append(p)
        else:
            p.unlink(missing_ok=True)
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class BridgeStats:
    applied: int = 0
    dropped: int = 0
    invalid: int = 0
    queued: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and (not self.notes or self.notes[-1] != message):
            self.notes.append(message)


def _apply_line(layer: Layer, m: dict) -> None:
    layer.line(
        m["id"],
        m["a"],
        m["b"],
        m.get("color", _WHITE),
        float(m.get("width_px", 1.5)),
        float(m.get("duration", NEVER)),
    )


def _apply_lines(layer: Layer, m: dict) -> None:
    layer.lines(
        m["id"],
        m["pts_a"],
        m["pts_b"],
        m.get("color", _WHITE),
        float(m.get("width_px", 1.5)),
        float(m.get("duration", NEVER)),
    )


def _apply_arrow(layer: Layer, m: dict) -> None:
    layer.arrow(
        m["id"],
        m["a"],
        m["b"],
        m.get("color", _WHITE),
        float(m.get("width_px", 2.0)),
        float(m.get("duration", NEVER)),
    )


def _apply_arrows(layer: Layer, m: dict) -> None:
    layer.arrows(
        m["id"],
        m["pts_a"],
        m["pts_b"],
        m.get("color", _WHITE),
        float(m.get("width_px", 2.0)),
        float(m.get("duration", NEVER)),
    )


def _apply_point(layer: Layer, m: dict) -> None:
    layer.point(
        m["id"],
        m["p"],
        m.get("color", _WHITE),
        float(m.get("radius_px", 4.0)),
        float(m.get("duration", NEVER)),
    )


def _apply_points(layer: Layer, m: dict) -> None:
    layer.points(
        m["id"],
        m["positions"],
        m.get("color", _WHITE),
        float(m.get("radius_px", 4.0)),
        float(m.get("duration", NEVER)),
    )


def _apply_frame(layer: Layer, m: dict) -> None:
    layer.frame(
        m["id"], m["transform"], float(m.get("axis_len", 0.1)), float(m.get("duration", NEVER))
    )


def _apply_frames(layer: Layer, m: dict) -> None:
    layer.frames(
        m["id"],
        m["positions"],
        m["rotations"],
        float(m.get("axis_len", 0.1)),
        float(m.get("duration", NEVER)),
    )


def _apply_box(layer: Layer, m: dict) -> None:
    layer.box(m["id"], m["transform"], m.get("color", _WHITE), float(m.get("duration", NEVER)))


def _apply_sphere(layer: Layer, m: dict) -> None:
    layer.sphere(m["id"], m["transform"], m.get("color", _WHITE), float(m.get("duration", NEVER)))


def _apply_solid_arrow(layer: Layer, m: dict) -> None:
    layer.solid_arrow(
        m["id"], m["transform"], m.get("color", _WHITE), float(m.get("duration", NEVER))
    )


def _apply_solid_double_arrow(layer: Layer, m: dict) -> None:
    layer.solid_double_arrow(
        m["id"], m["transform"], m.get("color", _WHITE), float(m.get("duration", NEVER))
    )


def _apply_sector(layer: Layer, m: dict) -> None:
    layer.sector(
        m["id"],
        m["center"],
        m["rotvec_end"],
        m["ref_end"],
        m.get("color", _WHITE),
        float(m.get("duration", NEVER)),
        radius_px=float(m.get("radius_px", 0.0)),
    )


def _apply_text(layer: Layer, m: dict) -> None:
    layer.text(
        m["id"],
        m["anchor"],
        str(m["text"]),
        m.get("color", _WHITE),
        m.get("offset_px", (0.0, 0.0)),
        m.get("align", (0.0, 0.5)),
        float(m.get("duration", NEVER)),
    )


def _apply_clear(layer: Layer, m: dict) -> None:
    ident = m.get("id")
    if ident:
        layer.erase(str(ident))
    else:
        layer.clear()


OPS = {
    "line": _apply_line,
    "lines": _apply_lines,
    "arrow": _apply_arrow,
    "arrows": _apply_arrows,
    "point": _apply_point,
    "points": _apply_points,
    "frame": _apply_frame,
    "frames": _apply_frames,
    "box": _apply_box,
    "sphere": _apply_sphere,
    "solid_arrow": _apply_solid_arrow,
    "solid_double_arrow": _apply_solid_double_arrow,
    "sector": _apply_sector,
    "text": _apply_text,
    "clear": _apply_clear,
}


class DebugBridge:
    def __init__(self, backend: Any | None = None, app: str = APP) -> None:
        self.app = app
        self.stats = BridgeStats()
        self._backend: Any | None = None
        self._queue: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self._server: _SocketServer | None = None
        self._last_note = ""
        if backend is not None:
            self.bind(backend)

    def bind(self, backend: Any | None) -> None:
        self._backend = backend
        self._last_note = ""

    @property
    def draw(self) -> DebugDraw | None:
        backend = self._backend
        if backend is None:
            return None
        return backend.debug

    @property
    def available(self) -> bool:
        backend = self._backend
        return bool(backend is not None and backend.caps.debug_draw and backend.debug is not None)

    def _unavailable(self, count: int) -> None:
        self.stats.dropped += max(1, int(count))
        backend = self._backend
        name = backend.caps.name if backend is not None else "no backend"
        note = f"{name} does not support debug draw; annotations were dropped"
        self.stats.note(note)
        if note != self._last_note:
            self._last_note = note
            log.warning("{}", note)

    def layer(self, name: str, occlusion: Occlusion = Occlusion.DEPTH) -> Layer | None:
        dd = self.draw
        if dd is None or not self.available:
            self._unavailable(1)
            return None
        return dd.layer(name, occlusion)

    def publish(self, name: str, occlusion: Occlusion, fn) -> bool:
        layer = self.layer(name, occlusion)
        if layer is None:
            return False
        fn(layer)
        return True

    def clear(self, name: str | None = None) -> bool:
        dd = self.draw
        if dd is None or not self.available:
            self._unavailable(1)
            return False
        if name is None:
            dd.clear()
        else:
            dd.layer(name).clear()
        return True

    def serve(self, path: Path | None = None) -> Path | None:
        if self._server is not None:
            return self._server.path
        target = Path(path) if path is not None else socket_path(app=self.app)
        if path is None and target in _ACTIVE_PATHS:
            index = 2
            while target.with_name(f"{target.stem}-{index}.sock") in _ACTIVE_PATHS:
                index += 1
            target = target.with_name(f"{target.stem}-{index}.sock")
        try:
            self._server = _SocketServer(target, self._queue)
        except OSError as e:
            self.stats.note(f"Debug draw endpoint failed at {target}: {e}")
            log.warning("Debug draw endpoint failed at {}: {}", target, e)
            self._server = None
            return None
        _ACTIVE_PATHS.add(target)
        log.info("Debug draw endpoint: {}", target)
        return target

    def pump(self, budget: int = DEFAULT_BUDGET) -> int:
        applied = 0
        for _ in range(max(0, int(budget))):
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break
            if self._apply(msg):
                applied += 1
        self.stats.applied += applied

        self.stats.queued = self._queue.qsize()
        return applied

    def apply_batch(self, messages, budget: int = DEFAULT_BUDGET) -> int:
        limit = max(0, int(budget))
        applied = 0
        for message in messages[:limit]:
            applied += int(self._apply(message))
        self.stats.applied += applied
        dropped = max(0, len(messages) - limit)
        if dropped:
            self.stats.dropped += dropped
            self.stats.note(
                f"Debug command budget {limit} exceeded; dropped {dropped} commands. "
                "Use lines, arrows, points, or frames batches for high-cardinality data."
            )
        return applied

    def _apply(self, msg: dict) -> bool:
        op = msg.get("op")
        if op == "__invalid__":
            self.stats.invalid += 1
            self.stats.note(f"Invalid external command JSON: {msg.get('error')}")
            return False
        fn = OPS.get(str(op))
        if fn is None:
            self.stats.invalid += 1
            self.stats.note(f"Unknown operation {op!r}. Available operations: {sorted(OPS)}")
            return False
        occ = _occlusion_of(msg.get("occlusion", "depth"))
        layer = self.layer(str(msg.get("layer", "external")), occ)
        if layer is None:
            return False
        try:
            fn(layer, msg)
        except (KeyError, TypeError, ValueError) as e:
            self.stats.invalid += 1
            self.stats.note(f"Invalid arguments for operation {op!r}: {e}")
            return False
        return True

    def close(self) -> None:
        if self._server is not None:
            _ACTIVE_PATHS.discard(self._server.path)
            self._server.close()
            self._server = None

    @property
    def socket(self) -> Path | None:
        return self._server.path if self._server is not None else None


def _occlusion_of(value: Any) -> Occlusion:
    try:
        return Occlusion(str(value).lower())
    except ValueError:
        return Occlusion.DEPTH


class _SocketServer:
    def __init__(self, path: Path, sink: queue.SimpleQueue) -> None:
        self.path = path
        self._sink = sink
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(path))
        self._sock.listen(8)
        path.chmod(0o600)
        self._sock.setblocking(False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._sock, selectors.EVENT_READ, None)
        self._buffers: dict[int, bytearray] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mojive-debugdraw", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                events = self._sel.select(timeout=0.2)
            except OSError:
                break
            for key, _mask in events:
                if key.data is None:
                    self._accept()
                else:
                    self._read(key.fileobj)

    def _accept(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except (BlockingIOError, OSError):
            return
        conn.setblocking(False)
        self._sel.register(conn, selectors.EVENT_READ, data=b"")
        self._buffers[conn.fileno()] = bytearray()

    def _read(self, conn) -> None:
        try:
            chunk = conn.recv(65536)
        except (BlockingIOError, ConnectionResetError, OSError):
            chunk = b""
        if not chunk:
            self._drop(conn)
            return
        buf = self._buffers.get(conn.fileno())
        if buf is None:
            return
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[: nl + 1]
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                self._sink.put({"op": "__invalid__", "error": str(e)})
                continue
            if isinstance(msg, dict):
                self._sink.put(msg)

    def _drop(self, conn) -> None:
        with contextlib.suppress(KeyError, ValueError):
            self._sel.unregister(conn)
        self._buffers.pop(conn.fileno(), None)
        conn.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        for key in list(self._sel.get_map().values()):
            with contextlib.suppress(KeyError, ValueError):
                self._sel.unregister(key.fileobj)
            if key.fileobj is not self._sock:
                key.fileobj.close()  # type: ignore[union-attr]
        self._sel.close()
        self._sock.close()
        Path(self.path).unlink(missing_ok=True)


class DebugClient:
    def __init__(self, pid: int | None = None, app: str = APP, timeout: float = 1.0) -> None:
        if pid is not None:
            path = socket_path(pid, app)
        else:
            live = live_sockets(app)
            if not live:
                raise FileNotFoundError(f"No viewer socket found in {socket_dir(app)}")
            path = live[0]
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(str(path))

    def send(self, **message: Any) -> None:
        self._sock.sendall(json.dumps(message).encode() + b"\n")

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> DebugClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_BRIDGE: DebugBridge | None = None


def bridge(backend: Any | None = None) -> DebugBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = DebugBridge(backend)
    elif backend is not None:
        _BRIDGE.bind(backend)
    return _BRIDGE
