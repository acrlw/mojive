# Python and C++ development

Mojive remains a Python package. `mojive.Renderer`, `SceneRenderer`, Viewer entry points and
existing snake_case methods remain the user-facing API. C++ supplies selected implementation
workloads through the private `mojive._native` extension. Select `renderer="bgfx"` in the
existing Python API or run `make native-viewer`; the C++ tools remain acceptance fixtures,
not a replacement standalone product.

## Source layout

```text
python/
  src/mojive/        Python package and compatibility facade
  tests/            Python tests, GPU tests and reviewed golden fixtures
  bindingTests/     Optional native binding comparison tests
cpp/
  include/mojive/   Backend-neutral C++ contracts
  src/             Common code and private renderer adapters
  bindings/        C++ binding translation units
  tools/           Native acceptance fixtures and benchmarks
  tests/           C++ contract tests
  cmake/           Build integration
thirdParty/        Vendored ImGui and pinned upstream submodules
examples/          Python usage examples
tools/             Repository maintenance and verification scripts
output/            Generated builds, captures and reports
```

The root `pyproject.toml` remains the Python build entry point. Imports never contain `python`,
`cpp`, `thirdParty`, or a backend library name. C++ headers describe Mojive's contracts rather
than exposing bgfx handles, ImGui types, or MuJoCo-owned mutable state.

## Naming

Mojive-owned C++ uses a Qt-like style without adopting Qt or its `Q` type prefix:

| Item | Style | Example |
|---|---|---|
| Types, enums and enum values | PascalCase | `SceneFrame`, `ReadbackState::Pending` |
| Functions and methods | camelCase | `setScene()`, `createTarget()`, `waitForReadback()` |
| Parameters, locals and public fields | camelCase | `objectId`, `cameraRevision` |
| Private class members | `m` + PascalCase | `mDevice`, `mReadbacks` |
| Namespaces | camelCase | `mojive`, `mojive::bindingProbe` |
| Source filenames | PascalCase | `SceneStream.hpp`, `SceneCapture.cpp` |

Use `.hpp` / `.cpp`, four spaces and the committed `cpp/.clang-format`. The optional
`cpp/.clang-tidy` names the same identifier rules. Upstream APIs, standard-library names,
platform/compiler macros, existing build target names and Python ABI names keep their required
spelling; they are not mechanically renamed to match Mojive's C++ style.

Python keeps its existing snake_case function/module names and PascalCase classes. Do not add
a second public spelling merely because C++ uses camelCase. The binding layer translates
between conventions, preserving argument names, return shapes, dtypes, context-manager behavior
and error semantics. For example, Python `update_scene()` stays unchanged even if the native
implementation uses `setScene()`.

## Local preparation

```bash
uv sync --extra dev --extra docs --extra mujoco --extra wgpu --extra native
make setup-imgui
make cpp-deps
make check
make cpp-test
make cpp-probe
```

The `native` extra installs CMake 3.24 or newer and Ninja into `.venv`. Make prefers
`.venv/bin` for build tools, including recursive native Viewer builds, so an older
system CMake does not require a manual `PATH` override.

Linux native window builds need the X11, Wayland and xkbcommon development libraries and
`wayland-scanner`. On Ubuntu 22.04, the GLFW dependencies include `libxrandr-dev`, `libxinerama-dev`,
`libxcursor-dev`, `libxi-dev`, `libwayland-dev`, and `libxkbcommon-dev`.

`make setup-imgui` installs the project's ImGui Bundle wheel with bulk drawing and
slider/focus geometry fixes. Run it again after `uv sync` or a synchronizing
`uv run` command (including `make docs-check`) replaces the wheel with the upstream
release; the GPU UI tests require these fixes. The command reuses its
cached platform wheel when the build recipe has not changed.

The default Hatch wheel remains pure Python. `make native-wheel-test` explicitly builds a
platform wheel with the native extension, compiled shaders and dependency licenses, then
installs it in isolation and renders through the public Python API. Local native builds do not
replace the installed Python package; Make selects the extension with `MOJIVE_NATIVE_BUILD`.

`cpp-*` targets are convenient entry points; the earlier `native-*` acceptance commands remain
available. C++ builds use fresh `output/cpp-build`, `output/cpp-core-build` and
`output/cpp-bindings-build` directories so previous evaluation artifacts stay readable.
The renderer is selected at the composition root; SDL comparison remains opt-in.

The user deferred workflow CI. No evaluation workflow is installed by this branch. Linux native
windows support X11 and Wayland; the runtime selects the protocol reported by GLFW. All windows
sharing a native device must use the same protocol. Offscreen-first applications infer it from
the session environment. Local validation uses Ubuntu 22.04 x86_64 and NVIDIA Vulkan, with a
private X11 display and a headless Weston compositor. Physical Wayland input, compositor-specific
desktop integration, other GPU vendors, and Windows still need platform acceptance. Local commands and coverage are in the
[native verification guide](native-probe.md) and [verification matrix](testing.md).

For headless Weston acceptance, set `PYGLFW_LIBRARY` to the installed ImGui Bundle's
`libmojive_glfw.so.3` before importing MuJoCo or pyGLFW. The project's build handles compositors
without an input seat; an independently installed GLFW can still contain the upstream crash.
OpenGL comparisons on Wayland also require `PYOPENGL_PLATFORM=egl` before importing PyOpenGL.
Use `make native-windows` and `make native-viewer-test` inside the compositor's environment.

## Native render diagnostics

Native color targets accept 1x, 2x, 4x, and 8x MSAA. Object IDs, metric depth and segmentation
remain single-sampled. `RenderStats.cpu_ms` and `gpu_ms` expose named native stages such as
`shadow`, `reflection`, `color`, `scene data`, `outline`, and `debug`. CPU values measure bgfx
submission work; GPU timestamps arrive later. The statistics notes identify their independent
submission numbers. A cached render that issues no new draws can report an earlier submission's
measurements. These boundaries differ from OpenGL's finer pass decomposition.

Use `viewer.backend.enable_hot_reload(True)` to opt into native shader reload. Development
builds watch `cpp/shaders` and rebuild the shader target after edits; packaged builds watch
their `.bin` files. A reload replaces the complete program set and invalidates scene caches.
Compile or binary-load failures log an error and preserve the last working programs. Reload is
synchronous on the next render and can briefly pause the viewer while CMake compiles shaders.

## Dependency changes

Read the repository's `thirdParty/README.md` before editing an upstream source tree.
Dear ImGui is tracked directly to make custom drawing changes reviewable alongside Mojive.
Its first import has no local behavior patches. The native library and Python `imgui-bundle`
are separate until integration; editing the former does not change the latter's installed wheel.

The [native dependency plan (Chinese)](../plans/cpp-dependencies.zh.md) recommends GLM for
graphics math and spdlog for runtime-owned native output and bounded log history. Python can
publish records, configure output and subscribe through optional Loguru/logging bridges; native
output must not depend on a Python consumer. C++ provides rendering and runtime infrastructure;
business algorithms and extension policy remain in Python. Eigen and native business solvers are
outside the current roadmap. EnTT requires a demonstrated infrastructure need. GLM and spdlog are pinned build dependencies; Eigen and EnTT are not included.

## Private native extension

The optional `mojive._native` module is built separately behind the public bgfx backend.
GLM camera calculations preserve the existing row-major contract. The native
log supports independent cursors and optional rotating file output without changing host logging.
`RenderRuntime` owns one native backend thread, bounded dispatch and joined teardown.
bgfx processes commands on that owner instead of adding a second CPU frame queue; GPU execution
remains asynchronous, with at most two frames in flight. Metal uses two presentation images
with VSync and a third spare image without VSync to avoid compositor-owned drawable stalls.
Python calls release the GIL while waiting. Window events and ImGui stay on
the UI thread, independently of physics and resource workers.

Synchronous readback submits the copy and waits in one owner job. Asynchronous requests capture
the submitted frame before later draws can replace it. Readback uses aligned bgfx buffers, with
a GPU-only resolved color copy for MSAA targets. Row padding is removed at the owned-image
boundary; returned NumPy arrays remain valid after reuse, resize and runtime teardown.

```bash
make cpp-python-test
make native-fixture HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make cpp-python-gpu
```

`cpp-python-test` builds a CPU-only extension. `cpp-python-gpu` builds bgfx in `NATIVE_BUILD`,
loads that exact module and uses the existing `NATIVE_SCENE` fixture. These opt-in tests cover
four image products, array ownership, cross-thread callers, cancellation, initialization failure,
interpreter shutdown and the official 100-humanoid scene. Captures are written under
`output/cpp-python/bgfx/`. No extension is copied into an existing Python installation.

Independent public Renderer instances and windows share a process-wide device while owning
separate scenes, targets and readback lifetimes. Static output reuse invalidates on geometry,
lighting, style, camera and target changes; separately updated overlays disable color reuse.
Native texture preparation releases the GIL and transfers immutable upload storage to bgfx.

OpenGL, WebGPU and bgfx use the same bounded anisotropic footprint in their albedo shaders
and the same linear-light area filter for 2D mip levels, including odd texture dimensions.
Hardware anisotropy stays disabled for those 2D samplers to avoid applying the filter twice.
On Vulkan devices with `VK_EXT_sample_locations`, native MSAA uses the reflected OpenGL
sample positions and preserves those positions through depth transitions. Devices without
that extension retain their hardware sample pattern.

`make gpu-bgfx` exercises the shared rendering, picking, gizmo, debug drawing, physics and
UI tests with the native backend. Tests that inspect private OpenGL or WebGPU pass objects
remain specific to those implementations. Automated composition tests hide their windows;
run explicit shown-window lifecycle tools on a separate display when desktop focus must
remain uninterrupted.

See the [Viewer guide (Chinese)](../how-to/native-viewer.zh.md) for public usage and the
[acceptance record (Chinese)](../plans/native-renderer-parity.zh.md) for measured coverage and limits.
