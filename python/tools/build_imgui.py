"""Build ImGui Bundle with bulk drawing bindings and slider/focus geometry fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.92.900"
URL = "https://files.pythonhosted.org/packages/18/47/ff477f7258baddf0c02bbcf6406099c7ed78ebd63f3b9cebce899e6cf79a/imgui_bundle-1.92.900.tar.gz"
SHA256 = "1476409b76c7f600d0da472aed87e96382444df50695280b3ed9acd5065cb6c2"
FREETYPE_URL = "https://codeload.github.com/freetype/freetype/tar.gz/refs/tags/VER-2-13-3"
FREETYPE_SHA256 = "bc5c898e4756d373e0d991bab053036c5eb2aa7c0d5c67e8662ddc6da40c4103"
# Keep HelloImGui for GLFW/OpenGL and FreeType support, plus the
# always-built portable file dialogs. Unused bindings otherwise enlarge every GC scan.
UNUSED_BUNDLE_MODULES = (
    "IMMAPP",
    "IMGUI_MD",
    "NANOVG",
    "IMPLOT",
    "IMPLOT3D",
    "IMGUI_NODE_EDITOR",
    "IMGUIZMO",
    "IMGUI_TEX_INSPECT",
    "IMFILEDIALOG",
    "IMMVISION",
    "IMANIM",
    "IMGUI_EXPLORER_LIB",
    "MICROTEX",
)
# These upstream bindings are unconditional or coupled to required native support.
# Keep HelloImGui's FreeType/GLFW build, but omit its unused Python application API.
UNUSED_BINDINGS = (
    ("hello_imgui", "himgui", "hello_imgui", "hello_imgui"),
    (
        "imgui_color_text_edit",
        "imgui_color_text_edit",
        "imgui_color_text_edit",
        "ImGuiColorTextEdit",
    ),
    ("imgui_knobs", "imgui_knobs", "imgui_knobs", "imgui-knobs"),
    ("imspinner", "imspinner", "imspinner", "imspinner"),
    ("imgui_toggle", "imgui_toggle", "imgui_toggle", "imgui_toggle"),
    (
        "imgui_command_palette",
        "imgui_command_palette",
        "imgui_command_palette",
        "imgui-command-palette",
    ),
    ("imcoolbar", "imcooolbar", "imcoolbar", "ImCoolBar"),
)
# Restore every historically patched file before applying a changed recipe.
# This also removes obsolete patches from an existing incremental build.
RESTORED_FILES = (
    "external/bindings_generation/cpp/pybind_imgui_bundle.cpp",
    "external/bindings_generation/cpp/all_pybind_files.cmake",
    "pyproject.toml",
    "external/imgui/imgui/imgui.h",
    "external/imgui/imgui/imgui_internal.h",
    "external/imgui/imgui/imgui.cpp",
    "external/imgui/imgui/imgui_draw.cpp",
    "external/imgui/imgui/imgui_widgets.cpp",
    "external/imgui/bindings/pybind_imgui.cpp",
    "external/imgui/bindings/pybind_imgui_internal.cpp",
    "bindings/imgui_bundle/imgui/__init__.pyi",
    "bindings/imgui_bundle/imgui/internal.pyi",
    "imgui_bundle_cmake/internal/add_glfw_submodule.cmake",
    "external/glfw/glfw/src/wl_init.c",
    "external/glfw/glfw/src/wl_window.c",
    "bindings/imgui_bundle/_glfw_set_search_path.py",
    "external/hello_imgui/hello_imgui/hello_imgui_cmake/hello_imgui_build_lib.cmake",
)


def _archive_matches(archive: Path) -> bool:
    if not archive.is_file():
        return False
    with archive.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == SHA256


def ensure_source_archive(archive: Path) -> None:
    """Reuse verified sources and atomically replace incomplete downloads."""
    if _archive_matches(archive):
        return
    if archive.exists():
        print(f"Cached source archive is incomplete or corrupt; downloading again: {archive}")
    else:
        print(f"Downloading imgui-bundle {VERSION} source archive")
    # A failed transfer must never become the reusable archive. Keep an existing
    # cache untouched until its replacement passes the pinned upstream checksum.
    with tempfile.TemporaryDirectory(prefix=".imgui-download-", dir=archive.parent) as directory:
        candidate = Path(directory) / archive.name
        with urllib.request.urlopen(URL, timeout=60) as response, candidate.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if not _archive_matches(candidate):
            raise RuntimeError(
                "Downloaded imgui-bundle source archive checksum mismatch; cache was not replaced"
            )
        candidate.replace(archive)


def replace_source(path: Path, old: str, new: str, *, count: int = 1) -> None:
    """Patch upstream UTF-8 sources independently of the host locale."""
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    if content.count(old) != count:
        raise RuntimeError(f"unexpected upstream source at {path}: {old[:80]}")
    path.write_text(content.replace(old, new, count), encoding="utf-8", newline="\n")


def trim_unused_bindings(source: Path) -> None:
    """Exclude unused binding creation without changing upstream Python import semantics."""
    import re

    root = source / "external/bindings_generation/cpp"
    entry = root / "pybind_imgui_bundle.cpp"
    content = entry.read_text(encoding="utf-8")
    for name, variable, initializer, directory in UNUSED_BINDINGS:
        replacement = f'    _register_submodule("{name}", false); // Mojive: native support only'
        if replacement not in content:
            pattern = (
                rf'    _register_submodule\("{name}"\);\n'
                rf'    auto module_{variable} = +m.def_submodule\("[^"]+"\);\n'
                rf"    py_init_module_{initializer}\(module_{variable}\);"
            )
            content, count = re.subn(pattern, replacement, content)
            if count != 1:
                raise RuntimeError(f"unexpected upstream binding registration: {name}")
        replace_source(
            root / "all_pybind_files.cmake",
            f"    ${{external_dir}}/{directory}/bindings/pybind_{initializer}.cpp",
            f"    # Mojive: omit unused {name} Python bindings",
        )
    if entry.read_text(encoding="utf-8") != content:
        entry.write_text(content, encoding="utf-8", newline="\n")


def patch(source: Path) -> None:
    def change(relative, old, new, *, count=1):
        replace_source(source / relative, old, new, count=count)

    imgui = "external/imgui/imgui/"
    trim_unused_bindings(source)
    # Preserve Bundle's FreeType version without fetching the complete Git history.
    change(
        "external/hello_imgui/hello_imgui/hello_imgui_cmake/hello_imgui_build_lib.cmake",
        "                GIT_REPOSITORY https://github.com/freetype/freetype.git\n"
        "                GIT_TAG        VER-2-13-3\n"
        "                GIT_PROGRESS TRUE",
        f"                URL {FREETYPE_URL}\n                URL_HASH SHA256={FREETYPE_SHA256}",
    )
    change(
        "external/glfw/glfw/src/wl_init.c",
        "if (wl_seat_get_version(_glfw.wl.seat) >= WL_KEYBOARD_REPEAT_INFO_SINCE_VERSION)",
        "if (_glfw.wl.seat && wl_seat_get_version(_glfw.wl.seat) >= WL_KEYBOARD_REPEAT_INFO_SINCE_VERSION)",
    )
    change(
        "external/glfw/glfw/src/wl_window.c",
        "        window->wl.hovered = GLFW_TRUE;\n        _glfwSetCursorWayland",
        "        window->wl.hovered = GLFW_TRUE;\n"
        "        window->wl.cursorPosX = wl_fixed_to_double(sx);\n"
        "        window->wl.cursorPosY = wl_fixed_to_double(sy);\n"
        "        _glfwInputCursorPos(window, window->wl.cursorPosX, window->wl.cursorPosY);\n"
        "        _glfwSetCursorWayland",
    )
    change(
        "imgui_bundle_cmake/internal/add_glfw_submodule.cmake",
        "    set(GLFW_BUILD_WAYLAND OFF PARENT_SCOPE)",
        "    if (UNIX AND NOT APPLE)\n        set(GLFW_BUILD_WAYLAND ON PARENT_SCOPE)\n    endif()",
    )
    # pyGLFW may already have loaded a platform-only libglfw.so.3 (for example
    # through MuJoCo). A distinct SONAME keeps it from satisfying ImGui's richer
    # native-symbol dependency accidentally.
    change(
        "imgui_bundle_cmake/internal/add_glfw_submodule.cmake",
        "    install(TARGETS glfw DESTINATION imgui_bundle)",
        "    if (UNIX AND NOT APPLE)\n"
        "        set_target_properties(glfw PROPERTIES OUTPUT_NAME mojive_glfw)\n"
        "    endif()\n    install(TARGETS glfw DESTINATION imgui_bundle)",
    )
    change(
        "bindings/imgui_bundle/_glfw_set_search_path.py",
        'lib_filenames = ["libglfw.so.3",',
        'lib_filenames = ["libmojive_glfw.so.3", "libglfw.so.3",',
    )
    change("pyproject.toml", 'version = "1.92.900"', 'version = "1.92.900+mojive.1"')
    # The bundled ndarray casters use nanobind's four-argument export API.
    change("pyproject.toml", '"nanobind>=2.4.0"', '"nanobind==2.9.2"')
    binding = "external/imgui/bindings/pybind_imgui.cpp"
    change(
        binding,
        '#include "imgui_pywrappers.h"',
        '#include "imgui_pywrappers.h"\n#include "ImguiCallbacks.hpp"\n#include "ImguiDrawMesh.hpp"',
    )
    change(
        binding,
        "    auto pyClassImDrawCmd =",
        """    nb::enum_<mojive::imguiBridge::CallbackKind>(m, "DrawCallbackKind")
        .value("none", mojive::imguiBridge::CallbackKind::None)
        .value("reset_render_state", mojive::imguiBridge::CallbackKind::ResetRenderState)
        .value("unsupported", mojive::imguiBridge::CallbackKind::Unsupported);
    auto pyClassImDrawCmd =""",
    )
    change(
        binding,
        '        .def_rw("clip_rect", &ImDrawCmd::ClipRect,',
        """        .def_prop_ro("callback_kind", &mojive::imguiBridge::callbackKind)
        .def_rw("clip_rect", &ImDrawCmd::ClipRect,""",
    )
    change(
        binding,
        "    auto pyClassImDrawList =",
        """    nb::class_<mojive::imguiBridge::DrawMesh>(m, "DrawMesh")
        .def(nb::init<std::vector<ImVec2>, std::vector<unsigned int>, std::vector<bool>>(),
             nb::arg("positions"), nb::arg("indices"), nb::arg("opaque"))
        .def("draw", &mojive::imguiBridge::DrawMesh::draw,
             nb::arg("draw_list"), nb::arg("origin"), nb::arg("color"));
    auto pyClassImDrawList =""",
    )
    stubs = "bindings/imgui_bundle/imgui/__init__.pyi"
    change(
        stubs,
        "class ImDrawList:",
        """class DrawMesh:
    def __init__(self, positions: List[ImVec2Like], indices: List[int], opaque: List[bool]) -> None: ...
    def draw(self, draw_list: ImDrawList, origin: ImVec2Like, color: ImU32) -> None: ...

class ImDrawList:""",
    )
    change(
        stubs,
        "class ImDrawCmd:",
        """class DrawCallbackKind(enum.IntEnum):
    none = 0
    reset_render_state = 1
    unsupported = 3

class ImDrawCmd:
    @property
    def callback_kind(self) -> DrawCallbackKind: ...
""",
    )
    change(
        binding,
        '        .def("add_convex_poly_filled",',
        """        .def("add_indexed_fill",
            [](ImDrawList& self, const std::vector<ImVec2>& points, const std::vector<unsigned int>& indices, ImU32 col)
            {
                if (indices.size() % 3) throw nb::value_error("triangle indices must be grouped in threes");
                for (unsigned int i : indices)
                    if (i >= points.size()) throw nb::value_error("triangle index outside vertex buffer");
                if (points.empty() || indices.empty()) return;
                self.PrimReserve(static_cast<int>(indices.size()),static_cast<int>(points.size()));
                const unsigned int base = self._VtxCurrentIdx;
                const ImVec2 uv = ImGui::GetIO().Fonts->TexUvWhitePixel;
                for (const ImVec2& point : points) self.PrimWriteVtx(point,uv,col);
                for (unsigned int i : indices) self.PrimWriteIdx(static_cast<ImDrawIdx>(base+i));
            }, nb::arg("points"), nb::arg("indices"), nb::arg("col"))
        .def("add_convex_poly_filled",""",
    )
    change(
        "bindings/imgui_bundle/imgui/__init__.pyi",
        "    def add_convex_poly_filled(self, points: List[ImVec2Like], col: ImU32) -> None:",
        "    def add_indexed_fill(self, points: List[ImVec2Like], indices: List[int], col: ImU32) -> None:\n"
        '        """Append a triangle mesh without an antialias fringe."""\n'
        "        pass\n"
        "    def add_convex_poly_filled(self, points: List[ImVec2Like], col: ImU32) -> None:",
    )
    change(
        binding,
        '        .def("add_concave_poly_filled",',
        """        .def("add_poly_fringe",
            [](ImDrawList& self, const std::vector<ImVec2>& inner, const std::vector<ImVec2>& outer, ImU32 col)
            {
                if (inner.size() != outer.size()) throw nb::value_error("fringe boundaries must have equal lengths");
                const int count = static_cast<int>(inner.size());
                if (count < 3 || !(self.Flags & ImDrawListFlags_AntiAliasedFill)) return;
                self.PrimReserve(count*6,count*2);
                const unsigned int base = self._VtxCurrentIdx;
                const ImVec2 uv = ImGui::GetIO().Fonts->TexUvWhitePixel;
                for (int i = 0; i < count; ++i)
                {
                    self.PrimWriteVtx(inner[i],uv,col);
                    self.PrimWriteVtx(outer[i],uv,col & ~IM_COL32_A_MASK);
                }
                for (int i = 0; i < count; ++i)
                {
                    const int j = (i+1)%count;
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+i*2));
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+j*2));
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+j*2+1));
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+i*2));
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+j*2+1));
                    self.PrimWriteIdx(static_cast<ImDrawIdx>(base+i*2+1));
                }
            }, nb::arg("inner"), nb::arg("outer"), nb::arg("col"))
        .def("add_concave_poly_filled",""",
    )
    change(
        "bindings/imgui_bundle/imgui/__init__.pyi",
        "    def add_concave_poly_filled(self, points: List[ImVec2Like], col: ImU32) -> None:",
        "    def add_poly_fringe(self, inner: List[ImVec2Like], outer: List[ImVec2Like], col: ImU32) -> None:\n"
        '        """Append an outward AA ring around an already solid polygon."""\n'
        "        pass\n"
        "    def add_concave_poly_filled(self, points: List[ImVec2Like], col: ImU32) -> None:",
    )
    # Keep native resize input and dock splitter colors, without painting a
    # filled corner wedge across the window's nested content inset.
    change(
        imgui + "imgui.cpp",
        """        // Render resize grips (after their input handling so we don't have a frame of latency)
        if (handle_borders_and_resize_grips && !(flags & ImGuiWindowFlags_NoResize))
        {
            for (int resize_grip_n = 0; resize_grip_n < resize_grip_count; resize_grip_n++)
            {
                const ImU32 col = resize_grip_col[resize_grip_n];
                if ((col & IM_COL32_A_MASK) == 0)
                    continue;
                const ImGuiResizeGripDef& grip = resize_grip_def[resize_grip_n];
                const ImVec2 corner = ImLerp(window->Pos, window->Pos + window->Size, grip.CornerPosN);
                const float border_inner = IM_ROUND(window_border_size * 0.5f);
                window->DrawList->PathLineTo(corner + grip.InnerDir * ((resize_grip_n & 1) ? ImVec2(border_inner, resize_grip_draw_size) : ImVec2(resize_grip_draw_size, border_inner)));
                window->DrawList->PathLineTo(corner + grip.InnerDir * ((resize_grip_n & 1) ? ImVec2(resize_grip_draw_size, border_inner) : ImVec2(border_inner, resize_grip_draw_size)));
                window->DrawList->PathArcToFast(ImVec2(corner.x + grip.InnerDir.x * (window_rounding + border_inner), corner.y + grip.InnerDir.y * (window_rounding + border_inner)), window_rounding, grip.AngleMin12, grip.AngleMax12);
                window->DrawList->PathFillConvex(col);
            }
        }

""",
        """        // Native corner hit targets and resize cursors remain active. A filled
        // grip wedge cuts across the nested content inset and obscures its contour.
        IM_UNUSED(resize_grip_count);
        IM_UNUSED(resize_grip_col);
        IM_UNUSED(resize_grip_draw_size);

""",
    )
    # Preserve the original contour when an item intersects a scrolling clip.
    # The scissor clips pixels; changing the rectangle creates false rounded ends.
    change(
        imgui + "imgui.cpp",
        "    display_rect.ClipWith(window->ClipRect);",
        "    // Retain the item contour; clipping belongs to the draw command.",
    )
    change(
        imgui + "imgui.cpp",
        "        bool fully_visible = window->ClipRect.Contains(display_rect);\n        if (!fully_visible)",
        "        const bool extend_clip = window->ClipRect.Contains(bb) && !window->ClipRect.Contains(display_rect);\n        if (extend_clip)",
    )
    change(
        imgui + "imgui.cpp",
        "        if (!fully_visible)\n            window->DrawList->PopClipRect();",
        "        if (extend_clip)\n            window->DrawList->PopClipRect();",
    )
    # Keep the focus stroke against the frame rather than floating beyond it.
    change(
        imgui + "imgui.cpp",
        "        const float distance = (float)(int)(3.0f + thickness * 0.5f);",
        "        const float distance = thickness * 0.5f + 0.5f;",
    )
    change(
        imgui + "imgui.cpp",
        "        display_rect.Expand(ImVec2(distance, distance));",
        """        if (rounding > 0.0f)
            rounding = ImMin(rounding, ImMin(bb.GetWidth(), bb.GetHeight()) * 0.5f) + distance - 0.5f;
        display_rect.Expand(ImVec2(distance, distance));""",
    )
    change(
        imgui + "imgui.cpp",
        "            RenderNavCursor(bb, child_window->ChildId);",
        "            RenderNavCursor(bb, child_window->ChildId, ImGuiNavRenderCursorFlags_None, child_window->WindowRounding);",
    )
    change(
        imgui + "imgui.cpp",
        "                RenderNavCursor(ImRect(bb.Min - ImVec2(2, 2), bb.Max + ImVec2(2, 2)), g.NavId, ImGuiNavRenderCursorFlags_Compact);",
        "                RenderNavCursor(ImRect(bb.Min - ImVec2(2, 2), bb.Max + ImVec2(2, 2)), g.NavId, ImGuiNavRenderCursorFlags_Compact, child_window->WindowRounding > 0.0f ? child_window->WindowRounding + 2.0f : 0.0f);",
    )
    # Dock tabs can render during NewFrame, before the application suppresses
    # their cursor. The selected tab already identifies the focused dock window.
    change(
        imgui + "imgui_widgets.cpp",
        "        RenderNavCursor(bb, id);\n\n        // Select with right mouse button.",
        "        if (docked_window == NULL)\n            RenderNavCursor(bb, id);\n\n        // Select with right mouse button.",
    )

    # Size the actual grab in SliderBehaviorT, so rendering, click offsets, and
    # value mapping all use the same bounds. Keep its size stable during radius edits.
    change(
        imgui + "imgui_widgets.cpp",
        "    grab_sz = ImMin(grab_sz, slider_sz);",
        """    const ImGuiAxis cross_axis = axis == ImGuiAxis_X ? ImGuiAxis_Y : ImGuiAxis_X;
    grab_sz = ImMax(grab_sz, bb.Max[cross_axis] - bb.Min[cross_axis] - grab_padding * 2.0f);
    grab_sz = ImMin(grab_sz, slider_sz);""",
    )
    helper = """static float MojiveSliderGrabRounding(const ImGuiStyle& style, const ImRect& frame, const ImRect& grab)
{
    const float frame_half = ImMin(frame.GetWidth(), frame.GetHeight()) * 0.5f;
    const float grab_half = ImMin(grab.GetWidth(), grab.GetHeight()) * 0.5f;
    const float inner_radius = ImMax(0.0f, ImMin(style.FrameRounding, frame_half) - 2.0f);
    return ImMin(grab_half, ImMin(style.GrabRounding, inner_radius));
}"""
    change(
        imgui + "imgui_widgets.cpp",
        '#include "imgui_internal.h"',
        '#include "imgui_internal.h"' + "\n\n" + helper,
    )
    change(
        imgui + "imgui_widgets.cpp",
        "GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), style.GrabRounding);",
        "GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), MojiveSliderGrabRounding(style, frame_bb, grab_bb));",
        count=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build/imgui",
        help="Separate source and compiler cache directory for this platform",
    )
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    archive = build / f"imgui_bundle-{VERSION}.tar.gz"
    ensure_source_archive(archive)
    source = build / f"imgui_bundle-{VERSION}"
    if not source.exists():
        with tarfile.open(archive) as tf:
            tf.extractall(build, filter="data")
    headers = {
        name: (ROOT / "cpp/bindings" / name).read_bytes()
        for name in ("ImguiCallbacks.hpp", "ImguiDrawMesh.hpp")
    }
    recipe_bytes = Path(__file__).read_bytes() + b"".join(headers.values())
    recipe = hashlib.sha256(recipe_bytes).hexdigest()
    prepared = build / "source-patch.sha256"
    unchanged = {}
    if not prepared.exists() or prepared.read_text(encoding="utf-8").strip() != recipe:
        # Reapply changed recipes to upstream sources, including removal of old
        # patches. Keep the CMake tree and untouched sources for incremental builds.
        with tarfile.open(archive) as tf:
            for relative in RESTORED_FILES:
                path = source / relative
                if path.exists():
                    unchanged[path] = (path.read_bytes(), path.stat().st_mtime_ns)
                member = tf.extractfile(f"imgui_bundle-{VERSION}/{relative}")
                path.write_bytes(member.read())
    (source / "external/imgui/imgui/mojive_g3.h").unlink(missing_ok=True)
    patch(source)
    for name, content in headers.items():
        header = source / "external/imgui/bindings" / name
        if not header.exists() or header.read_bytes() != content:
            header.write_bytes(content)
    # A recipe change must not rebuild every translation unit when the final
    # patched headers are unchanged.
    for path, (content, modified) in unchanged.items():
        if path.read_bytes() == content:
            os.utime(path, ns=(path.stat().st_atime_ns, modified))
    prepared.write_text(recipe + "\n", encoding="utf-8")
    if args.prepare_only:
        print(source)
        return
    signature = hashlib.sha256(
        recipe_bytes
        + sys.implementation.cache_tag.encode()
        + sys.platform.encode()
        + platform.machine().encode()
    ).hexdigest()
    wheels = build / "wheels"
    record = build / "wheel-build.json"
    wheel = None
    if record.exists():
        previous = json.loads(record.read_text(encoding="utf-8"))
        candidate = Path(previous["wheel"])
        if previous["signature"] == signature and candidate.is_file():
            wheel = candidate
    if wheel is None:
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", str(build / "uv-cache"))
        env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "6")
        if sys.platform == "win32":
            # MSVC otherwise decodes upstream UTF-8 comments using the ANSI code page.
            env["CL"] = (env.get("CL", "") + " /utf-8").strip()
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--python",
                sys.executable,
                "--out-dir",
                str(wheels),
                "-C",
                f"build-dir={build / 'cmake'}",
                *(
                    option
                    for module in UNUSED_BUNDLE_MODULES
                    for option in ("-C", f"cmake.define.IMGUI_BUNDLE_WITH_{module}=OFF")
                ),
                "-C",
                "cmake.define.IMGUI_BUNDLE_BUILD_DEMOS=OFF",
                "-C",
                "cmake.define.HELLOIMGUI_WITH_TEST_ENGINE=OFF",
                str(source),
            ],
            check=True,
            env=env,
        )
        wheel = max(wheels.glob("imgui_bundle-*.whl"), key=lambda p: p.stat().st_mtime)
        record.write_text(
            json.dumps({"signature": signature, "wheel": str(wheel)}, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.install:
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "--reinstall", str(wheel)],
            check=True,
        )
    print(wheel)


if __name__ == "__main__":
    main()
