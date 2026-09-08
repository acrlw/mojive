"""OpenGL context attachment and capability probing."""

from __future__ import annotations

from dataclasses import dataclass, field

import moderngl

from ...log import get_logger
from . import gl_native as G

log = get_logger("context")


@dataclass(frozen=True)
class ContextCaps:
    version_code: int = 0
    version: str = ""
    renderer: str = ""
    vendor: str = ""
    core_profile: bool = False
    max_samples: int = 0
    max_texture_units: int = 0
    max_vertex_attribs: int = 0
    max_texture_size: int = 0
    khr_debug: bool = False
    timer_query: str = "none"

    line_width_max: float = 1.0

    native_state: bool = False
    native_clear_uint: bool = False
    native_instance_offset: bool = False
    extensions: frozenset[str] = field(default_factory=frozenset)
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.version_code >= 330


def attach(ctx: moderngl.Context | None = None) -> tuple[moderngl.Context, ContextCaps]:
    gl_ctx = ctx if ctx is not None else moderngl.create_context()
    return gl_ctx, probe(gl_ctx)


def probe(ctx: moderngl.Context) -> ContextCaps:
    info = ctx.info
    exts = frozenset(ctx.extensions)
    native = G.native()
    notes: list[str] = []

    timer = "none"
    try:
        q = ctx.query(time=True)
        with q:
            ctx.clear()
        ctx.finish()
        _ = q.elapsed
        timer = "elapsed"
    except Exception:
        notes.append("GPU timer queries are unavailable; per-pass GPU timings are disabled")

    if not native.has_debug_group:
        notes.append("KHR_debug is unavailable; RenderDoc pass groups are disabled")
    if not native.has_attrib_pointer:
        notes.append(
            "Native glVertexAttribPointer is unavailable; using per-bucket instance buffers"
        )
    if not native.has_state:
        notes.append("Native glGet* is unavailable; GL state restoration is limited")

    _lo, hi = native.line_width_range()
    if hi <= 1.0:
        notes.append(f"glLineWidth is limited to {hi:g}; wide lines use triangle strips")

    caps = ContextCaps(
        version_code=int(ctx.version_code),
        version=str(info.get("GL_VERSION", "")),
        renderer=str(info.get("GL_RENDERER", "")),
        vendor=str(info.get("GL_VENDOR", "")),
        core_profile=bool(int(info.get("GL_CONTEXT_PROFILE_MASK", 0)) & 1),
        max_samples=int(info.get("GL_MAX_SAMPLES", 0) or 0),
        max_texture_units=int(info.get("GL_MAX_TEXTURE_IMAGE_UNITS", 0) or 0),
        max_vertex_attribs=int(info.get("GL_MAX_VERTEX_ATTRIBS", 0) or 0),
        max_texture_size=int(info.get("GL_MAX_TEXTURE_SIZE", 0) or 0),
        khr_debug=native.has_debug_group,
        timer_query=timer,
        line_width_max=hi,
        native_state=native.has_state,
        native_clear_uint=native.has_clear_buffer_uiv,
        native_instance_offset=native.has_attrib_pointer,
        extensions=exts,
        notes=tuple(notes),
    )
    log.info(
        "OpenGL attached: {} / {} / core={} / max_samples={} / timer={}",
        caps.version,
        caps.renderer,
        caps.core_profile,
        caps.max_samples,
        caps.timer_query,
    )
    for n in notes:
        log.info("{}", n)
    native.drain_errors()
    return caps
