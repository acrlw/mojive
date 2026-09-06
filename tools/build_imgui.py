"""Build ImGui Bundle with bulk drawing bindings and slider/focus geometry fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.92.900"
URL = "https://files.pythonhosted.org/packages/18/47/ff477f7258baddf0c02bbcf6406099c7ed78ebd63f3b9cebce899e6cf79a/imgui_bundle-1.92.900.tar.gz"
SHA256 = "1476409b76c7f600d0da472aed87e96382444df50695280b3ed9acd5065cb6c2"
# Restore every historically patched file before applying a changed recipe.
# This also removes obsolete patches from an existing incremental build.
RESTORED_FILES = (
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
)


def patch(source: Path) -> None:
    def change(relative, old, new, *, count=1):
        path = source / relative
        content = path.read_text()
        if new in content:
            return
        if content.count(old) != count:
            raise RuntimeError(f"unexpected upstream source at {relative}: {old[:80]}")
        path.write_text(content.replace(old, new, count))

    imgui = "external/imgui/imgui/"
    change("pyproject.toml", 'version = "1.92.900"', 'version = "1.92.900+mojive.1"')
    # The bundled ndarray casters use nanobind's four-argument export API.
    change("pyproject.toml", '"nanobind>=2.4.0"', '"nanobind==2.9.2"')
    binding = "external/imgui/bindings/pybind_imgui.cpp"
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
    # Focus rings expand the control's contour. Use its effective radius before
    # adding the outset; otherwise saturated controls get a different silhouette.
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
    args = parser.parse_args()
    build = ROOT / "output/g3-ui/build"
    build.mkdir(parents=True, exist_ok=True)
    archive = build / f"imgui_bundle-{VERSION}.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("imgui-bundle source archive checksum mismatch")
    source = build / f"imgui_bundle-{VERSION}"
    if not source.exists():
        with tarfile.open(archive) as tf:
            tf.extractall(build, filter="data")
    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared = build / "source-patch.sha256"
    if not prepared.exists() or prepared.read_text().strip() != recipe:
        # Reapply changed recipes to upstream sources, including removal of old
        # patches. Keep the CMake tree and untouched sources for incremental builds.
        with tarfile.open(archive) as tf:
            for relative in RESTORED_FILES:
                member = tf.extractfile(f"imgui_bundle-{VERSION}/{relative}")
                (source / relative).write_bytes(member.read())
    (source / "external/imgui/imgui/mojive_g3.h").unlink(missing_ok=True)
    patch(source)
    prepared.write_text(recipe + "\n")
    if args.prepare_only:
        print(source)
        return
    signature = hashlib.sha256(
        Path(__file__).read_bytes()
        + sys.implementation.cache_tag.encode()
        + sys.platform.encode()
        + platform.machine().encode()
    ).hexdigest()
    wheels = build / "wheels"
    record = build / "wheel-build.json"
    wheel = None
    if record.exists():
        previous = json.loads(record.read_text())
        candidate = Path(previous["wheel"])
        if previous["signature"] == signature and candidate.is_file():
            wheel = candidate
    if wheel is None:
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", str(build / "uv-cache"))
        env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "6")
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
                str(source),
            ],
            check=True,
            env=env,
        )
        wheel = max(wheels.glob("imgui_bundle-*.whl"), key=lambda p: p.stat().st_mtime)
        record.write_text(
            json.dumps({"signature": signature, "wheel": str(wheel)}, indent=2) + "\n"
        )
    if args.install:
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "--reinstall", str(wheel)],
            check=True,
        )
    print(wheel)


if __name__ == "__main__":
    main()
