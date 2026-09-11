# Python and C++ development

Mojive remains a Python package. `mojive.Renderer`, `SceneRenderer`, Viewer entry points and
existing snake_case methods remain the user-facing API. C++ supplies selected implementation
workloads through the private `mojive._native` extension. Select `renderer="bgfx"` in the
existing Python API or run `make native-viewer`; the C++ tools remain acceptance fixtures,
not a replacement standalone product.

## Source layout

```text
python/
  src/mojive/       Python package and compatibility facade
  tests/           Python tests, GPU tests and reviewed golden fixtures
  binding_tests/   Optional native binding comparison tests
cpp/
  include/mojive/   Backend-neutral C++ contracts
  src/             Common code and private renderer adapters
  bindings/        C++ binding translation units
  tools/           Native acceptance fixtures and benchmarks
  tests/           C++ contract tests
  cmake/           Build integration
3rdparty/          Vendored ImGui and pinned upstream submodules
examples/          Python usage examples
tools/             Repository maintenance and verification scripts
output/            Generated builds, captures and reports
```

The root `pyproject.toml` remains the Python build entry point. Imports never contain `python`,
`cpp`, `3rdparty`, or a backend library name. C++ headers describe Mojive's contracts rather
than exposing bgfx handles, ImGui types, or MuJoCo-owned mutable state.

## Naming

Owned directories use lowercase names, with underscores between words (`python/binding_tests`).
Repository-managed dependencies live in `3rdparty`, with upstream names such as `bgfx.cmake`
and `robin-map`. Directory names do not follow C++ identifier casing. Existing submodule section
names in `.gitmodules` are stable Git identifiers; the `path` entries define their locations.

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

Python constants use `UPPER_SNAKE_CASE`; internal helpers use a leading underscore. Prefer
domain names over implementation abbreviations, and annotate shared interfaces with concrete
types or protocols. Use `Callable` for callbacks rather than `object`. Keep adapters' public
contracts compatible; runtime attribute probes are for an intentional compatibility boundary,
not a substitute for defining the contract. Apply the repository formatter to changed code only.

## Implementation principles

Use the existing command, adapter and composition boundaries. Add an abstraction when it has
a concrete consumer or removes duplicated policy; keep its inputs, outputs and owner explicit.
A small function or dataclass is preferable to a new service, registry or inheritance hierarchy
when the problem only needs value transformation or a short sequence of commands.

| Concern | Do | Do not |
|---|---|---|
| State ownership | Keep document state and command routing in `Session`, physics state in adapters, and transient gestures in the UI. | Reach into a physics implementation from a panel or store the same authoritative state in multiple layers. |
| Capabilities | Gate the operation that needs a capability; keep selection, navigation and local timeline gestures usable with an empty or minimal adapter. | Disable or reset an entire panel because no model, take or physics feature is available. |
| Entity identity | Use returned command identities and rebind across structure changes using the owning contract. | Predict adapter-generated names, reuse old node indices after rebuilding, or silently skip an unresolved target. |
| Transactions | Group one user edit into one undoable transaction; restore the document on failure and retain the original error. | Commit successful fragments after a dependent command fails or rebase an arbitrary stale draft because its document ID matches. |
| Recovery | Reuse pending edits only after a known successful rollback or another explicitly verified compatible transition. | Broaden guards, catch and ignore errors, or install an empty fallback that hides the root cause. |
| UI geometry | Share production painters, hit bounds, layout and focus geometry with prototypes and panels; follow the [icon grid and optical-size rules](../how-to/ui-icons.md) for custom glyphs. | Copy drawing formulas into another caller, maintain a visually similar second implementation, or clamp one icon dimension independently. |
| Frame cost | Cache stable geometry and measurements using their real invalidation inputs; batch dependent model writes. | Rebuild, reset or allocate repeatedly for unchanged state, or keep an unbounded cache. |
| Regression checks | Exercise the real entry point, held input across frames, failures, and relevant capability combinations. | Replace the failing path with a test-only handler, change a baseline without inspecting it, or assert only implementation details. |

For a bug fix, first identify the command or state transition that fails. A screenshot or final
rollback message describes the symptom; trace the first failed operation before changing guards.
Retain a focused regression that fails on the previous implementation. Check the successful
path and the affected failure/recovery path, including undo/redo for document transactions.
Use the [verification matrix](testing.md#change-mapping) for the applicable completion gates.

## Collaboration and commits

`AGENTS.md` is the entry point for repository instructions; this guide owns implementation detail,
and the testing guide owns the verification matrix. Extend those sources instead of copying
rules into new skills, agent notes or task-specific checklists. Add a skill only for a distinct,
reusable workflow; historical plans and generated reports do not override current guidance.

Before editing, inspect the branch, status and relevant diff, then read the source around the
change. In shared worktrees, identify the files and contracts being changed, serialize edits to
the same code, and re-read a file if another contributor changed it. Preserve unrelated work;
do not use blanket restore, reset, formatting or staging. An isolated worktree can help when
independent changes would otherwise compete for the same files or mutable test state.

Handoffs state the concrete behavior, affected entry points, changed contracts, actual check
commands and outcomes, and unresolved work. Distinguish code that was inspected, code that was
exercised, and platform behavior that remains unverified. Keep captures and temporary diagnostics
under `output/`; put reusable acceptance scenarios in the existing tools and Make targets.

When a commit is authorized, group code, regression coverage and relevant documentation by
coherent behavior. Use a concise imperative English subject, such as `Preserve timeline drags
without model capabilities`. Explain the trigger, resulting behavior, validation and material
limits in the body when needed. Inspect the staged diff and whitespace before committing;
include no incidental dependency, generated-output or unrelated working-tree changes. A local
fix does not imply authorization to push, merge or publish it.

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
desktop integration and other GPU vendors still need platform acceptance. Current development targets macOS and Linux; Windows is deferred. Local commands and coverage are in the
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
Compilation runs on an owned worker while rendering retains the last working program set.
A completed generation is published on a subsequent render; newer edits supersede unfinished
generations. Compile or binary-load failures preserve working programs. Runtime teardown stops
and joins the owned compiler process group. GPU program replacement still runs on the render owner.

## Dependency changes

Read the repository's `3rdparty/README.md` before editing an upstream source tree.
Dear ImGui is tracked directly to make custom drawing changes reviewable alongside Mojive.
Its first import has no local behavior patches. The standalone native gallery builds this core;
the Python Viewer currently uses `imgui-bundle` for UI generation on all rendering backends.
Editing the tracked core does not change the installed Bundle wheel. See the dependency README
for the source/build relationship and the requirements for global corner customization.

The [native dependency plan (Chinese)](../plans/cpp-dependencies.zh.md) recommends GLM for
graphics math and spdlog for runtime-owned native output and bounded log history. Python can
publish records, configure output and subscribe through optional Loguru/logging bridges; native
output must not depend on a Python consumer. C++ provides rendering and runtime infrastructure;
business algorithms and extension policy remain in Python. Eigen and native business solvers are
outside the current roadmap. EnTT requires a demonstrated infrastructure need. GLM and spdlog are pinned build dependencies; Eigen and EnTT are not included.

## Shared UI controls

Keep reusable ImGui controls in `python/src/mojive/ui/controls.py`, independent of panel
and session state. Use `segmented_control` for N-way exclusive choices; it owns connected
outer corners, equal segment widths, narrow-layout reflow, disabled styling and native
keyboard input. `clearable_combo` shares its inset clear action with `search_input`.
`begin_property_table` and `property_row` provide left-label/right-control layouts;
`pill_label` draws passive capsule badges. Existing panel imports re-export common controls.

Reuse `compound_fields.py` for joined numeric fields and inset focus contours, `panels/value_cards.py` for value
rails, `panels/filters.py` for filter pills and severity icons, and `viewport_widgets.py`
for overlay glyphs. Keep domain commands in callers. Cache stable measurements and local
geometry by their actual font, scale, text or size inputs, rather than duplicating painters
inside each panel. Compound controls suppress the native rectangular focus cursor and draw
focus with the same corner flags as their own surface. Verify shared controls through their affected consumers and `make ui-layout-audit`.

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
the submitted frame before later draws can replace it. Pending asynchronous waits yield to other
accepted runtime jobs. A shared eight-slot reservation bounds public readbacks; a full asynchronous
queue reports backpressure immediately. Canceling a public Future still drains its native ticket
and preserves the canceled output buffer. Completion callbacks never wait for their own cleanup
queue. Readback uses aligned bgfx buffers, with
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
Scene sources hold immutable shared meshes and pixel storage. Published resources must not be
mutated; adapters replace an asset or send scene-local `MeshUpdate` data for deformation. Identity
hits avoid content hashing; recompilers producing fresh equivalent objects use SHA-256 keys.
Weak CPU/GPU caches plus per-scene leases reuse unchanged assets without retaining an unbounded
history. First deformation detaches shared GPU geometry. Private `runtime.resource_stats()` counts
actual mesh/texture uploads and bytes, separately from scene publication and dynamic instance data.

The optional backend `prepare_scene(source)` hook performs CPU-only preparation on the existing
model-loader worker; its leases survive until UI publication. Native mesh/texture preparation
releases the GIL. GPU resource creation and atomic scene replacement remain render-owner jobs;
large initial uploads are not yet split across a per-frame budget.

Output targets lease view IDs per GPU submission instead of retaining a fixed twelve-target pool.
Views are reused only after advancing the submitted frame. Allocation is still bounded by bgfx
resource handles and GPU memory; it is not an unlimited-target guarantee.

OpenGL, WebGPU and bgfx use the same bounded anisotropic footprint in their albedo shaders
and the same linear-light area filter for 2D mip levels, including odd texture dimensions.
Hardware anisotropy stays disabled for those 2D samplers to avoid applying the filter twice.
Vulkan and Metal use bottom-left rasterization for offscreen targets, including viewport, scissor
and winding conversion. This matches OpenGL edge ownership with standard MSAA sample patterns;
programmable sample locations are not required. Window surfaces retain presentation coordinates.

`make gpu-bgfx` exercises the shared rendering, picking, gizmo, debug drawing, physics and
UI tests with the native backend. Tests that inspect private OpenGL or WebGPU pass objects
remain specific to those implementations. Automated composition tests hide their windows;
run explicit shown-window lifecycle tools on a separate display when desktop focus must
remain uninterrupted.

See the [Viewer guide (Chinese)](../how-to/native-viewer.zh.md) for public usage and the
[acceptance record (Chinese)](../plans/native-renderer-parity.zh.md) for measured coverage and limits.
