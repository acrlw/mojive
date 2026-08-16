from __future__ import annotations

import sys
from pathlib import Path

import glfw
import moderngl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge_viewer.render.forge import gl_native as G
from forge_viewer.render.forge.context import probe
from forge_viewer.render.forge.state_guard import GLStateGuard
from forge_viewer.render.forge.targets import IdLayout, RenderTarget, probe_id_layout

OK, BAD = "✓", "✗"


def row(name: str, value, note: str = "") -> None:
    print(f"  {name:34} {value}" + (f"   ← {note}" if note else ""))


def main() -> int:
    if not glfw.init():
        print("glfw 起不来")
        return 1
    for k, v in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, True),
        (glfw.VISIBLE, False),
    ):
        glfw.window_hint(k, v)
    win = glfw.create_window(320, 240, "forge probe", None, None)
    if not win:
        print("建不出 3.3 core 上下文——这台机器达不到本项目的下限（02 §2.2）")
        glfw.terminate()
        return 1
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    caps = probe(ctx)
    native = G.native()

    print("\n=== 上下文 ===")
    row("GL_VERSION", caps.version)
    row("GL_RENDERER", caps.renderer)
    row("GL_VENDOR", caps.vendor)
    row("core profile", caps.core_profile)
    row("version_code", caps.version_code, "下限 330（02 §2.2）")
    row("GL_MAX_SAMPLES", caps.max_samples)
    row("GL_MAX_VERTEX_ATTRIBS", caps.max_vertex_attribs, "网格 3 + 实例 7 = 10")
    row("GL_MAX_TEXTURE_IMAGE_UNITS", caps.max_texture_units)
    row("GL_MAX_TEXTURE_SIZE", caps.max_texture_size)

    print("\n=== 规格点名的四条硬约束（02 §2.2） ===")
    row(
        "① BaseInstance（GL 4.2）",
        f"{BAD} 不可用" if caps.version_code < 420 else "可用",
        "每桶一个 VAO + 字节偏移" if caps.version_code < 420 else "",
    )
    row(
        "② SSBO（GL 4.3）",
        f"{BAD} 不可用" if caps.version_code < 430 else "可用",
        "逐实例数据只能走顶点属性" if caps.version_code < 430 else "",
    )
    row("③ 计算着色器", f"{BAD} 不可用" if caps.version_code < 430 else "可用")
    _lo, hi = native.line_width_range()
    row(
        "④ glLineWidth 上限",
        f"{hi:g}",
        "粗线只能展开成三角形带" if hi <= 1.0 else "驱动允许粗线，但规格仍走三角形带",
    )

    print("\n=== 扩展与原生符号 ===")
    row("GL_ARB_timer_query", OK if "GL_ARB_timer_query" in caps.extensions else BAD)
    row(
        "GPU 计时实际可用",
        f"{OK} {caps.timer_query}" if caps.timer_query != "none" else f"{BAD} none",
        "报了扩展不等于真能用，探测要包一次真的绘制",
    )
    row("GL_KHR_debug", OK if caps.khr_debug else f"{BAD}（pass 标签退化成空操作）")
    row("原生 glGet*（状态守卫）", OK if native.has_state else BAD)
    row("原生 glClearBufferuiv", OK if native.has_clear_buffer_uiv else BAD)
    row("原生 glVertexAttribPointer", OK if native.has_attrib_pointer else BAD)
    row("原生 glClear（只清深度）", OK if native.has_clear_depth else BAD)

    print("\n=== 几何着色器 ===")
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
        row("几何着色器", f"{OK} 可用", "WIREFRAME 的重心坐标路径")
    except Exception as e:
        row("几何着色器", f"{BAD} {e}")

    print("\n=== ID buffer 布局（docs/PLATFORM.md §1） ===")
    for s in (1, 2, 4, 8):
        if s > max(caps.max_samples, 1):
            continue
        layout = probe_id_layout(ctx, s)
        note = "整数附件进不了 MSAA FBO" if layout is IdLayout.SPLIT else "规格原样"
        row(f"samples={s}", layout, note)

    print("\n=== moderngl 的已知缺陷（02 §2.4） ===")
    fbo_i = ctx.framebuffer([ctx.texture((16, 16), 1, dtype="u4")])
    fbo_i.use()
    fbo_i.clear(0.13, 0.13, 0.13, 1.0)
    got = int(np.frombuffer(fbo_i.read(components=1, dtype="u4"), np.uint32)[0])
    bits = int(np.float32(0.13).view(np.uint32))
    row(
        "clear() 打整数附件",
        f"读回 {got}",
        f"{'复现' if got == bits else '本机不复现'}：0.13f 的位模式是 {bits}",
    )
    native.drain_errors()

    tgt = RenderTarget(ctx, 64, 64, samples=min(4, max(caps.max_samples, 1)))
    ok_clear = True
    for v in (0, 77, 4_000_000_000):
        tgt.clear_main((0.1, 0.1, 0.1, 1.0))
        tgt.clear_id(v)
        ok_clear &= int(np.unique(tgt.read_ids())[0]) == v
    row(
        "forge 的整数清屏",
        f"{OK} 三档值精确" if ok_clear else f"{BAD} 不精确",
        "含 4e9，证明 R32UI 不像 RGB8 那样 16M 溢出",
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
        "深度写掩码重放",
        f"{'复现' if px[1] < 100 else '本机不复现'}",
        "上一帧关深度写 → 这一帧 clear 不清深度 → 整批几何一个片元都不写",
    )

    print("\n=== GLStateGuard（02 §2.3） ===")
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
    row("守住的项数", len(before))
    row("逐项不变", f"{OK} 是" if not diff else f"{BAD} {diff}")

    import timeit

    t = timeit.timeit(lambda: (guard.capture(), guard.restore()), number=3000) / 3000 * 1000
    row("capture + restore", f"{t:.4f} ms/帧", "规格参照 0.036")

    print("\n=== HiDPI（10 §10.4） ===")
    w, h = glfw.get_window_size(win)
    fw, fh = glfw.get_framebuffer_size(win)
    sx, _ = glfw.get_window_content_scale(win)
    row("窗口 / 帧缓冲", f"{w}×{h} / {fw}×{fh}")
    row("content_scale", f"{sx:g}", "视口矩形是 UI 点、渲染目标是物理像素，两者不是 1:1")

    err = native.drain_errors()
    print(f"\n收尾 glGetError = {err} {'（干净）' if err == 0 else '（有残留，要查）'}")
    for n in caps.notes:
        print(f"  · {n}")
    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
