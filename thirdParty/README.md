# Repository-managed dependencies

`dependencies.json` is the source-of-truth revision lock. Production dependencies are available
as repository-managed source; optional comparison dependencies are checksum-locked downloads.
The C++ build never replaces an edited ImGui tree with a fetched copy.

| Source | Management | Purpose |
|---|---|---|
| `imgui/` | Tracked source, v1.92.9b-docking | Dear ImGui core and platform integration; owned customization history |
| `bgfx/`, `bx/`, `bimg/`, `bgfxCmake/` | Git submodules at the tested commits | Renderer, offline shader compilation, and stb image resizing |
| `glfw/` | Git submodule | Native platform windows and input |
| `glm/`, `spdlog/` | Git submodules | Private graphics math and native logging |
| `nanobind/`, `robinMap/` | Git submodules | Python binding support |
| SDL, pybind11, separate shader tools | Optional locked downloads | Retained comparison experiments, disabled by default |

After cloning, initialize only the top-level submodules:

```bash
git submodule update --init --depth 1
python tools/check_dependencies.py
```

Nested upstream submodules are not required: bgfx, bx, bimg and robin-map are supplied through
the top-level paths above. Do not use `--remote` for routine setup. A dependency update changes
the submodule commit and the matching lock entry together, then runs the applicable native gates.
Upstream licenses remain in every source directory; Mojive's license does not replace them.

## ImGui customization

The initial import is byte-identical to the locked upstream archive. `imguiBaseline.json` records
all imported files. Check it with `python tools/check_dependencies.py --imgui-baseline`; that
optional check intentionally fails after a local customization. It is not a requirement to keep
ImGui unmodified during development.

Record local behavior changes in `imguiChanges.md` and normal, focused Git commits. Keep upstream
version updates separate from Mojive behavior patches. Retain the upstream API and formatting;
do not apply Mojive's C++ naming rules across vendor source. During an upstream update, reconcile
local patches against the recorded baseline and regenerate that baseline from the new unmodified
upstream archive, not from the customized working tree.

UI spacing, typography and common widgets belong in Mojive's UI layer. Changes to ImGui's
rounding geometry or internal draw paths belong here when its public customization points are
insufficient. The baseline import does not implement continuous-curvature corners. Existing
Python `imgui-bundle` remains independent until the native UI bridge is integrated.

## bgfx Metal offscreen sampling

`cpp/cmake/MetalHeadless.cmake` compiles a narrowly modified build-tree copy of the pinned
Metal renderer. Upstream sampler anisotropy reads the main window swap chain, which is absent
for an offscreen-only runtime. The patch preserves reset-selected anisotropy independently of
a swap chain so Viewer and offscreen textures use the same filtering. The submodule remains
unchanged; CMake fails if the pinned source context no longer matches. Reconcile this patch
explicitly during a bgfx update and rerun textured tendon and image-light parity.

## Native texture preprocessing

`cpp/src/Texture.cpp` uses `bimg/3rdparty/stb/stb_image_resize2.h` from the pinned bimg
revision. Its upstream license remains in that header. The box filter preserves the shared
linear-light RGB and independent-alpha mip contract, including non-power-of-two extents.
Mojive does not introduce a separate stb download or modify the vendored implementation.

## Optional mesh preparation

`cpp/src/MeshProcessing.cpp` uses meshoptimizer's allocator and simplifier from
`bgfx/3rdparty/meshoptimizer`, pinned by the bgfx submodule revision. The optional Python
extension builds those CPU-only sources without a graphics dependency and releases the GIL
while simplifying. Initialize the bgfx submodule even for a renderer-free Python extension
build. The MIT license is retained upstream and included in platform wheels. The helper returns
ordinary Mojive mesh data and does not couple consumer renderers to bgfx.
