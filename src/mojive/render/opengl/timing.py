"""CPU and GPU render-pass timing."""

from __future__ import annotations

import ctypes
import time
from collections import deque

import moderngl

from . import gl_native as G

LATENCY = 2


_POOL_LIMIT = 8


class PassTimer:
    __slots__ = ("_cpu_start", "active", "cpu_ms", "free", "gpu_ms", "name", "pending")

    def __init__(self, name: str) -> None:
        self.name = name
        self.cpu_ms = 0.0
        self.gpu_ms = 0.0
        self._cpu_start = 0.0
        self.pending: deque[int] = deque()
        self.free: list[int] = []
        self.active: int | None = None


class FrameTiming:
    def __init__(self, ctx: moderngl.Context, enabled: bool = True) -> None:
        self.ctx = ctx
        self._gl = G.native()
        self._timers: dict[str, PassTimer] = {}
        self._order: list[str] = []
        self._depth = 0
        self.enabled = enabled

        self._u32 = (ctypes.c_uint * 1)()
        self._u64 = (ctypes.c_uint64 * 1)()
        self.gpu_available = bool(enabled and self._gl.has_query and self._probe())

    def _probe(self) -> bool:
        try:
            qid = self._gl.gen_query()
            self._gl.begin_time_query(qid)
            self.ctx.clear()
            self._gl.end_time_query()
            self.ctx.finish()
            ready = self._gl.query_ready(qid, self._u32)
            ns = self._gl.query_result_ns(qid, self._u64) if ready else 0
            self._gl.delete_query(qid)
            self._gl.drain_errors()
            return bool(ready) and ns >= 0
        except Exception:
            self._gl.drain_errors()
            return False

    def _timer(self, name: str) -> PassTimer:
        t = self._timers.get(name)
        if t is None:
            t = PassTimer(name)
            self._timers[name] = t
            self._order.append(name)
        return t

    def begin(self, name: str) -> None:
        t = self._timer(name)
        t._cpu_start = time.perf_counter()
        self._gl.push_debug_group(name)

        if self.gpu_available and self._depth == 0:
            qid = t.free.pop() if t.free else self._gl.gen_query()
            self._gl.begin_time_query(qid)
            t.active = qid
        self._depth += 1

    def end(self, name: str) -> None:
        t = self._timer(name)
        self._depth = max(0, self._depth - 1)
        if t.active is not None:
            self._gl.end_time_query()
            t.pending.append(t.active)
            t.active = None
        self._gl.pop_debug_group()
        t.cpu_ms = (time.perf_counter() - t._cpu_start) * 1000.0

    def collect(self) -> None:
        if not self.gpu_available:
            return
        for t in self._timers.values():
            while len(t.pending) > LATENCY:
                qid = t.pending[0]
                if not self._gl.query_ready(qid, self._u32):
                    break
                t.gpu_ms = self._gl.query_result_ns(qid, self._u64) / 1e6
                t.pending.popleft()
                if len(t.free) < _POOL_LIMIT:
                    t.free.append(qid)
                else:
                    self._gl.delete_query(qid)

    def cpu_table(self) -> dict[str, float]:
        return {n: self._timers[n].cpu_ms for n in self._order}

    def gpu_table(self) -> dict[str, float]:
        if not self.gpu_available:
            return {}
        return {n: self._timers[n].gpu_ms for n in self._order}

    def scope(self, name: str) -> _Scope:
        return _Scope(self, name)

    def release(self) -> None:
        for t in self._timers.values():
            for qid in (*t.pending, *t.free):
                self._gl.delete_query(qid)
            t.pending.clear()
            t.free.clear()


class _Scope:
    __slots__ = ("name", "timing")

    def __init__(self, timing: FrameTiming, name: str) -> None:
        self.timing = timing
        self.name = name

    def __enter__(self) -> _Scope:
        self.timing.begin(self.name)
        return self

    def __exit__(self, *exc) -> None:
        self.timing.end(self.name)
