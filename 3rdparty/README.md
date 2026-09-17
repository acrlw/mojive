# Repository-managed dependencies

Python package dependencies are declared in `pyproject.toml` and resolved in `uv.lock`;
they are not copied into this source tree. Font files retain their own licenses and use the
existing system/cache discovery policy.

`dependencies.json` is the source-of-truth revision lock. Production dependencies are available
as repository-managed source; optional comparison dependencies are checksum-locked downloads.
The C++ build never replaces an edited ImGui tree with a fetched copy.

| Source | Management | Purpose |
|---|---|---|
| `imgui/` | Tracked source, v1.92.9b-docking | Dear ImGui core for the native gallery; owned customization history |
| `bgfx/`, `bx/`, `bimg/`, `bgfx.cmake/` | Git submodules at the tested commits | Renderer, offline shader compilation, and stb image resizing |
| `glfw/` | Git submodule | Native platform windows and input |
| `glm/`, `spdlog/` | Git submodules | Private graphics math and native logging |
| `nanobind/`, `robin-map/` | Git submodules | Python binding support |
| `libtess2/` | Git submodule | CPU-only contour triangulation for Canvas2D |
| SDL, pybind11, separate shader tools | Optional locked downloads | Retained comparison experiments, disabled by default |

After cloning, initialize only the top-level submodules:

```bash
git submodule update --init --depth 1
python -m mojive.tools.check_dependencies
```

Nested upstream submodules are not required: bgfx, bx, bimg and robin-map are supplied through
the top-level paths above. Do not use `--remote` for routine setup. A dependency update changes
the submodule commit and the matching lock entry together, then runs the applicable native gates.
Upstream licenses remain in every source directory; Mojive's license does not replace them.

Upstream examples, documentation and unused platform implementations remain in the pinned
source trees. They are not part of the Python wheel. The build disables dependency examples,
tests and installation; `make native-python-build` builds only `_native` and its dependencies,
including the required shader compiler and shaders. `make native-build` additionally builds
Mojive's native acceptance tools and their ImGui/GLFW dependencies. Do not delete files inside
a submodule to reduce runtime size: this dirties the upstream checkout without changing the
product. Optional comparison downloads remain disabled until explicitly requested.

## ImGui customization

The initial import is byte-identical to the locked upstream archive. `imguiBaseline.json` records
all imported files. Check it with `python -m mojive.tools.check_dependencies --imgui-baseline`; that
optional check intentionally fails after a local customization. It is not a requirement to keep
ImGui unmodified during development.

Record local behavior changes in `imguiChanges.md` and normal, focused Git commits. Keep upstream
version updates separate from Mojive behavior patches. Retain the upstream API and formatting;
do not apply Mojive's C++ naming rules across vendor source. During an upstream update, reconcile
local patches against the recorded baseline and regenerate that baseline from the new unmodified
upstream archive, not from the customized working tree.

UI spacing, typography and common widgets belong in Mojive's UI layer. Changes to ImGui's
rounding geometry or internal draw paths belong here when its public customization points are
insufficient. The baseline import does not implement continuous-curvature corners.

### Which ImGui the Viewer uses

The Python Viewer imports `imgui_bundle.imgui` on OpenGL, wgpu and bgfx. `ui/window_native.py`
passes its generated vertices, indices, textures and clip rectangles to the native renderer;
bgfx does not generate widget geometry. The tracked `imgui/` tree is currently compiled into
`mojive_imgui` for the standalone native gallery, not the Python Viewer's ImGui extension.

`make setup-imgui` runs `python/tools/build_imgui.py`, which downloads a pinned ImGui Bundle source
archive and builds the Mojive wheel. The recipe contains checked patches for bulk geometry
submission, slider/focus geometry and platform support. Its source/build cache lives under
`build/imgui`; it is not an additional dependency to commit. Downloads enter the reusable cache
only after checksum verification; an incomplete cached archive is downloaded again automatically.
Bundle's FreeType dependency remains `VER-2-13-3`; the recipe downloads its checksum-verified
source archive instead of cloning the full Git history. The package version and Dear ImGui
core version are distinct: the current recipe uses Bundle `1.92.900`, with core
`1.92.9`, whereas this tracked core baseline is `1.92.9b-docking`.

`cpp/bindings/ImguiCallbacks.hpp` supplies read-only callback classification for the Python
ImGui backends. The build recipe copies it into Bundle and includes its content in the build
signature. Backends distinguish ordinary draws, render-state resets and unsupported callbacks;
they do not add external renderer insertion markers. Rebuild with `make setup-imgui` after
changing the header or recipe.

Shared custom widgets use continuous-curvature geometry from `geometry2d/curves.py` through
the `ui/paint_protocol.py` contract; `ui/imgui_draw.py` adapts that contract to ImGui. To extend it
to standard ImGui widgets globally, change the core draw paths
actually compiled into ImGui Bundle and rebuild the wheel. Reuse a reviewed core implementation
for the native gallery after reconciling the core versions and bindings; changing only this
tracked tree does not affect the Viewer. Fill, border and focus paths must agree, and real
circles must retain circular geometry. This work does not require switching rendering backends
or maintaining another independent ImGui checkout.

## bgfx Metal offscreen sampling

`cpp/cmake/MetalHeadless.cmake` compiles a narrowly modified build-tree copy of the pinned
Metal renderer. Upstream sampler anisotropy reads the main window swap chain, which is absent
for an offscreen-only runtime. The patch preserves reset-selected anisotropy independently of
a swap chain so Viewer and offscreen textures use the same filtering. The submodule remains
unchanged; CMake fails if the pinned source context no longer matches. Reconcile this patch
explicitly during a bgfx update and rerun textured tendon and image-light parity.
The patch also restricts `synchronizeResource` to Managed storage; Metal Shared readback buffers
are already coherent and reject that operation under API validation.

`cpp/cmake/MetalRasterization.cmake` converts offscreen viewport, scissor and winding to
bottom-left rasterization while keeping swap-chain presentation top-left. Negative viewport
height aligns triangle edge ownership as well as the standard MSAA sample pattern with OpenGL.
Texture-origin metadata drives sampling and public readback conversion. Reconcile checked
replacements during upgrades; verify exact 2/4/8x MSAA coverage, depth, ROI readback, asymmetric
UI composition, and lifecycle with Metal API validation enabled.

## bgfx Vulkan peer presentation and rasterization

`cpp/cmake/VulkanHeadless.cmake` preserves reset flags in the no-main-swapchain path
and updates peer swap chains when VSync changes. Without this correction, the native
runtime's peer windows ignore VSync requests. Both swap-chain recreation and the
maintenance-extension path receive the same policy.

`cpp/cmake/VulkanRasterization.cmake` uses bottom-left offscreen coordinates to match
OpenGL's sample pattern and triangle boundary ownership. Mirroring sample locations
alone leaves different coverage at exact edge ties. Swap chains retain top-left
presentation; viewport, scissor and clear rectangles retain their public top-left
coordinates. Pipeline caching includes the target orientation. This uses the standard
Vulkan sample pattern and does not require programmable sample locations.

Mojive converts render-target sampling at UI composition and RGB readback boundaries,
and places directional-shadow atlas tiles according to the reported texture origin.
Reconcile the checked replacements when updating bgfx; verify MSAA 2/4/8 color, preserved
depth, cropped readback, asymmetric UI composition, and peer VSync on/off.

## GLFW X11 visibility waits

`cpp/cmake/GlfwX11.cmake` compiles a checked build-tree copy of GLFW's X11 window
implementation. Its visibility wait polls for new socket data after checking for the
requested event, so unrelated queued events cannot bypass the timeout indefinitely.
The pinned submodule stays unchanged. Reconcile the patch when updating GLFW and run
`make native-windows` on X11 to verify repeated minimize/restore and resize cycles.

## GLFW Wayland initialization and Python coexistence

`cpp/cmake/GlfwWayland.cmake` guards GLFW 3.4's keyboard-repeat initialization when a Wayland
compositor exposes no input seat. The source replacement is checked and compiled from the
build tree; the submodule stays unchanged. Run `make native-windows` inside a headless Wayland
compositor when reconciling this patch.

`python/tools/build_imgui.py` applies the same guard to ImGui Bundle's GLFW and enables both X11 and
Wayland. Its Linux library uses `libmojive_glfw.so.3`, with the package's pyGLFW search path
updated accordingly. The distinct SONAME prevents a platform-only GLFW already loaded by
MuJoCo/pyGLFW from satisfying ImGui's dependency while missing its X11 native symbols.

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

## Canvas2D path topology

`cpp/src/geometry2d/Tessellator.cpp` uses the pinned libtess2 revision for nonzero/evenodd
fills, holes, self-intersections and external boundary edges. Its SGI Free Software
License B 2.0 remains in `libtess2/LICENSE.txt` and is included in native wheels.
Only its C sources are compiled; upstream examples and graphics libraries are not used.
Coordinates are translated and normalized before conversion to the upstream float type.
Scratch allocation has an explicit byte limit. A C-only jump boundary handles exhaustion
because some upstream allocation sites do not check null; all outstanding allocations
are owned and released by the wrapper, without modifying the pinned dependency.
