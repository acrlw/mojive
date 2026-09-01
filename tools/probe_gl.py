"""Report OpenGL context and framebuffer capabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import glfw
import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mojive.render.opengl import gl_native as G
from mojive.render.opengl.context import probe
from mojive.render.opengl.state_guard import GLStateGuard
from mojive.render.opengl.targets import IdLayout, RenderTarget, probe_id_layout

OK, BAD = "✓", "✗"


def row(name: str, value, note: str = "") -> None:
    print(f"  {name:34} {value}" + (f"   ← {note}" if note else ""))


def main() -> int:
    if not glfw.init():
        print("GLFW initialization failed")
        return 1
    for k, v in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, True),
        (glfw.VISIBLE, False),
    ):
        glfw.window_hint(k, v)
    win = glfw.create_window(320, 240, "opengl probe", None, None)
    if not win:
        print("OpenGL 3.3 core context creation failed")
        glfw.terminate()
        return 1
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    caps = probe(ctx)
    native = G.native()

    print("\n=== Context ===")
    row("GL_VERSION", caps.version)
    row("GL_RENDERER", caps.renderer)
    row("GL_VENDOR", caps.vendor)
    row("core profile", caps.core_profile)
    row("version_code", caps.version_code, "minimum 330")
    row("GL_MAX_SAMPLES", caps.max_samples)
    row("GL_MAX_VERTEX_ATTRIBS", caps.max_vertex_attribs, "mesh 3 + instance 7 = 10")
    row("GL_MAX_TEXTURE_IMAGE_UNITS", caps.max_texture_units)
    row("GL_MAX_TEXTURE_SIZE", caps.max_texture_size)

    print("\n=== Required capabilities ===")
    row(
        "BaseInstance (GL 4.2)",
        f"{BAD} unavailable" if caps.version_code < 420 else "available",
        "one VAO and byte offset per bucket" if caps.version_code < 420 else "",
    )
    row(
        "SSBO (GL 4.3)",
        f"{BAD} unavailable" if caps.version_code < 430 else "available",
        "instance data uses vertex attributes" if caps.version_code < 430 else "",
    )
    row("compute shader", f"{BAD} unavailable" if caps.version_code < 430 else "available")
    _lo, hi = native.line_width_range()
    row(
        "glLineWidth maximum",
        f"{hi:g}",
        "wide lines use triangle strips" if hi <= 1.0 else "driver supports wide lines",
    )

    print("\n=== Extensions and native entry points ===")
    row("GL_ARB_timer_query", OK if "GL_ARB_timer_query" in caps.extensions else BAD)
    row(
        "GPU timing",
        f"{OK} {caps.timer_query}" if caps.timer_query != "none" else f"{BAD} none",
        "verified with a draw call",
    )
    row("GL_KHR_debug", OK if caps.khr_debug else BAD)
    row("native glGet*", OK if native.has_state else BAD)
    row("native glClearBufferuiv", OK if native.has_clear_buffer_uiv else BAD)
    row("native glVertexAttribPointer", OK if native.has_attrib_pointer else BAD)
    row("native depth clear", OK if native.has_clear_depth else BAD)

    print("\n=== Geometry shader ===")
    try:
        p = ctx.program(
            vertex_shader="#version 330 core\nvoid main(){gl_Position=vec4(0);}",
            geometry_shader=(
                "#version 330 core\nlayout(triangles) in;"
                "layout(triangle_strip,max_vertices=3) out;"
                "void main(){for(int i=0;i<3;++i){gl_Position=gl_in[i].gl_Position;EmitVertex();}"
                "EndPrimitive();}"
            ),
            fragment_shader="#version 330 core\nout vec4 c;void main(){c=vec4(1);}",
        )
        p.release()
        row("geometry shader", f"{OK} available", "wireframe barycentric path")
    except Exception as e:
        row("geometry shader", f"{BAD} {e}")

    print("\n=== ID buffer layout ===")
    for s in (1, 2, 4, 8):
        if s > max(caps.max_samples, 1):
            continue
        layout = probe_id_layout(ctx, s)
        note = (
            "integer attachment uses a separate FBO" if layout is IdLayout.SPLIT else "shared FBO"
        )
        row(f"samples={s}", layout, note)

    print("\n=== ModernGL compatibility checks ===")
    fbo_i = ctx.framebuffer([ctx.texture((16, 16), 1, dtype="u4")])
    fbo_i.use()
    fbo_i.clear(0.13, 0.13, 0.13, 1.0)
    got = int(np.frombuffer(fbo_i.read(components=1, dtype="u4"), np.uint32)[0])
    bits = int(np.float32(0.13).view(np.uint32))
    row(
        "integer attachment clear()",
        f"read {got}",
        f"{'reproduced' if got == bits else 'not reproduced'}; float bits={bits}",
    )
    native.drain_errors()

    tgt = RenderTarget(ctx, 64, 64, samples=min(4, max(caps.max_samples, 1)))
    ok_clear = True
    for v in (0, 77, 4_000_000_000):
        tgt.clear_main((0.1, 0.1, 0.1, 1.0))
        tgt.clear_id(v)
        ok_clear &= int(np.unique(tgt.read_ids())[0]) == v
    row(
        "opengl integer clear",
        f"{OK} exact" if ok_clear else f"{BAD} inexact",
        "verified with 0, 77, and 4e9",
    )

    col = ctx.texture((32, 32), 4, dtype="f1")
    dep = ctx.depth_texture((32, 32))
    f2 = ctx.framebuffer([col], dep)
    f2.use()
    prog = ctx.program(
        vertex_shader="#version 330 core\nin vec3 p;void main(){gl_Position=vec4(p,1);}",
        fragment_shader="#version 330 core\nout vec4 c;uniform vec4 k;void main(){c=k;}",
    )
    vbo = ctx.buffer(np.array([[-1, -1, 0], [3, -1, 0], [-1, 3, 0]], "f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f", "p")])
    ctx.enable(moderngl.DEPTH_TEST)
    f2.clear(0, 0, 0, 1)
    prog["k"].value = (1, 0, 0, 1)
    vao.render()
    f2.depth_mask = False
    f2.clear(0, 0, 0, 1)
    f2.depth_mask = True
    prog["k"].value = (0, 1, 0, 1)
    vao.render()
    px = np.frombuffer(f2.read(components=3), np.uint8)[:3]
    row(
        "depth-mask replay",
        f"{'reproduced' if px[1] < 100 else 'not reproduced'}",
        "clear must enable depth writes before clearing",
    )

    print("\n=== GLStateGuard ===")
    guard = GLStateGuard()
    ctx.enable(moderngl.BLEND)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.wireframe = True
    ctx.viewport = (3, 5, 77, 55)
    native.set_enabled(G.GL_MULTISAMPLE, False)
    before = guard.snapshot()
    with guard:
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.wireframe = False
        ctx.viewport = (0, 0, 320, 240)
        ctx.multisample = True
    after = guard.snapshot()
    diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    row("state entries", len(before))
    row("restored", f"{OK} yes" if not diff else f"{BAD} {diff}")

    import timeit

    t = timeit.timeit(lambda: (guard.capture(), guard.restore()), number=3000) / 3000 * 1000
    row("capture + restore", f"{t:.4f} ms/frame", "reference 0.036")

    print("\n=== HiDPI ===")
    w, h = glfw.get_window_size(win)
    fw, fh = glfw.get_framebuffer_size(win)
    sx, _ = glfw.get_window_content_scale(win)
    row("window / framebuffer", f"{w}×{h} / {fw}×{fh}")
    row("content_scale", f"{sx:g}", "UI points to framebuffer pixels")

    err = native.drain_errors()
    print(f"\nfinal glGetError = {err} {'clean' if err == 0 else 'pending errors'}")
    for n in caps.notes:
        print(f"  · {n}")
    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
