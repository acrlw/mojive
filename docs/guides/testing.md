# Testing

The suite is split by runtime and dependency boundary. Start with the smallest meaningful check
for the changed behavior, then complete the applicable gates below. This page owns the verification
matrix; other agent guidance links here.

| Layer | Marker or target | Coverage |
|---|---|---|
| Fast | `make test-fast` | pure CPU behavior and module contracts |
| Examples | `make examples-check` | example syntax and import-independent entry points |
| Documentation | `make docs-check` | public docstrings, CLI/config reference, both example catalogs, asset paths, snippets, strict site build |
| Integration | `make test-integration` | files, serialization, protocols, processes, composition |
| Physics | `make test-physics` | model compilation and live physics worlds |
| OpenGL GPU | `make gpu` | real OpenGL contexts and rendered output |
| WebGPU | `make gpu-wgpu` | Metal, Vulkan, or DX12 backend behavior |
| Native bgfx GPU | `make gpu-bgfx` | shared renderer, interaction, capture and UI contracts with the native backend |
| Native probe | `make native-test`, `make native-probe` | optional C++ contracts and real GPU output/lifetime checks; choose `NATIVE_BACKEND=sdl` for SDL GPU |
| Native composition and lifecycle | `make native-composition`, `make native-windows` | texture dependencies, alpha/scissor output, repeated native surface lifecycle |
| Native live runtime | `make native-runtime` | native MuJoCo serial/parallel comparison with bounded output queues |
| Private native runtime | `make cpp-python-test`, `make cpp-python-gpu` | native logging, GLM compatibility, NumPy ownership, real GPU products and runtime teardown |
| Native Viewer | `make native-viewer-test HUMANOIDS_MODEL=/path/to/100_humanoids.xml` | public Python products, independent scenes, asynchronous readback, window lifecycle and advancing dense physics |
| Native Wayland input | `make native-wayland-test` | isolated compositor presentation, native pointer/keyboard events, focus, clipboard transfers, and external file drops across bgfx, OpenGL, and wgpu |
| Native render parity | `make native-parity`, `make native-model-parity HUMANOIDS_MODEL=/path/to/100_humanoids.xml` | matched OpenGL/wgpu/bgfx color, depth, identity, feature effects, deformable updates and pose restoration |
| Native motion and corpus | `make native-motion-parity`, `make native-corpus-parity MENAGERIE_ROOT=/path/to/mujoco_menagerie` | continuous close panning/orbit/zoom, object color checks, all local model loads and eight-view comparisons |
| Native UI parity | `make native-ui-parity` | fractional axis-label motion, CJK/Latin atlas filtering and ImGui antialiasing at normal and 150% scale |
| Native load latency | `make native-load-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` | isolated model processes, source/resource/first-frame timings and repeated loads against OpenGL |
| Native window cadence | `make native-window-benchmark` | real-window pan/orbit/dolly timing with VSync on/off and same-frame camera publication; run separately from other checks |
| Native feature lifecycle | `make native-features-test` | cache invalidation, frame-owned depth, light fallback, 8x color MSAA, delayed pass timing, shader reload recovery, pending readbacks and resource reuse |
| Native distribution | `make native-spirv`, `make native-wheel-test` | Vulkan shader compilation and installed platform wheel rendering without development paths |
| Native bindings | `make native-bindings-test` | isolated pybind11/nanobind behavior, MuJoCo coexistence, ownership, and GIL release |
| Golden | `make golden` | reviewed image baselines |
| Full | `make test-all` | CPU, physics, OpenGL, and WebGPU layers |

## Change mapping

MuJoCo viewer compatibility is covered by `tests/test_mujoco_viewer.py` (real MuJoCo state with
isolated window ownership) and `tests/gpu/test_mujoco_viewer.py` (user geometry on both renderers).
`tests/test_mujoco_model_updates.py` covers every `mjOption` field and enable/disable flag,
physics-only model edits, control metadata, diagnostic refresh, and visual/resource invalidation.
The tests distinguish retained model/data objects, retained scene sources, and updated values.
Run `make viewer-compat ARGS='--seconds 10'` for the actual window, callback, camera and user-scene
workflow. Space pauses/resumes the caller's simulation; the example closes after ten seconds.
See the [migration guide](../tutorials/mujoco-viewer.md) for supported signatures and limitations.
The minimal comparison programs are `make viewer-managed` and `make viewer-passive`.

`make geometry-views ARGS='--renderer opengl'` (or `wgpu` / `bgfx`) captures the four geometry
views using `assets/geometry_views.xml`. Inspect the hidden transparent capsule proxy and the
shared concave mesh/hull comparison. `tests/test_geometry_views.py` checks classification and
state preservation; `tests/gpu/test_geometry_views.py` verifies real colors, depth, IDs, resource
reuse, the MuJoCo-compatible renderer's segmentation, and scene screenshot/video frames in
each view. The repeated-instance regression ensures Both keeps diagnostic overlays batched.
With `MOJIVE_RENDERER=bgfx`, the geometry-view tests also count native mesh/texture uploads
across repeated view switches and verify that replacing the source clears retained mesh slots.
`make geometry-ui` clicks the production Hierarchy and Inspector controls in English and
Chinese at narrow, standard and HiDPI sizes, checks that Help remains visible in the real menu
bar, and captures the resulting panels. Inspect the captures under `output/geometry-ui`.

`make physics-options-check` exercises **Window > Physics Options...** and the environment
Inspector in English and Chinese at normal and 150% scale. It checks numeric commit and
validation, enum and flag writes, Undo/Redo, live worker timestep changes, and workspace/MJCF
round trips. Inspect `output/physics-options/` for algorithm, physical, and flag sections.

Viewer capture sharing is covered by `tests/test_capture_sharing.py`: saved image/file formats,
save failures, filename uniqueness, queued requests, cancellation, and asynchronous clipboard
ownership. `make keyframe-timeline ARGS='--recording-settings --language zh_CN'` captures the
settings controls. Desktop acceptance must also paste a screenshot and a finished video into an
application accepting images/file attachments; offscreen rendering does not validate that transfer.

Code, executable example, test, and build behavior changes finish with `make check`. The table adds
checks for each affected behavior; combine applicable rows without rerunning shared prerequisites.
Pure prose, link, and metadata edits use their own rows instead of the CPU or GPU suites.

| Change | Iteration target | Acceptance target |
|---|---|---|
| Shared types or commands | focused CPU test | `make check` |
| File format or remote protocol | focused integration test | `make check` |
| MuJoCo adapter or MJCF authoring | focused physics test | `make test-physics`, `make mujoco-audit`, and `make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables` |
| Public renderer performance | isolated quick matrix | `make renderer-benchmark` |
| MuJoCo model loading | one XML path with the model-suite module | `make mujoco-model-suite` |
| Rendering behavior, render pass, or shader | one GPU test file | `make gpu`, `make gpu-wgpu`, and relevant visual output |
| Visual interaction or settings layout | focused UI GPU test | relevant scripted gallery or interactive Make target with captured evidence |
| Custom UI icon geometry | family-focused CPU test and production multi-size gallery | `make check`, focused UI GPU test, and inspected capture following the [icon design guide](../how-to/ui-icons.md#verify-geometry-and-raster-output) |
| Native renderer probe or its shader | `make native-test`, `make native-probe` | `make check`, relevant `make native-gallery` or `make native-benchmark` evidence; Python rendering changes also use the rendering row above |
| Native binding boundary | `make native-bindings-test`; `make cpp-python-test` and `make cpp-python-gpu` for the private runtime | `make check`; relevant `make native-bindings-benchmark` evidence for performance claims |
| Registered regression invariant | its focused regression test | `make reverse` |
| Documentation or executable examples | relevant document/example checks | `make docs-check` |
| Instruction or Skill wording, links, or metadata | scope and reference review | [Skill validation](../how-to/agent-workflows.md#skill-maintenance) when applicable |
| Scene-control task decisions, operation behavior, or acceptance examples | relevant operation tests, such as `tests/test_operations.py` | `make agent-control`; also `make agent-viewer` when viewer attachment or presented capture is affected |

Markers may be combined. A file-format test that compiles MuJoCo uses both `integration` and
`physics`; it runs in the physics layer. A Skill behavior change still uses the scene-control row
even if it changes only Markdown. Backend-specific capture examples use `MOJIVE_RENDERER` as
described in [agent workflows](../how-to/agent-workflows.md#executable-acceptance-example).

Once applicable checks pass, broaden or repeat them only for a new change, failure, or unresolved
concern. Test observable behavior and meaningful invariants; a wording edit does not need a test
that merely matches the new wording. Extended release gates such as `make p1` are for release
acceptance or changes that affect that breadth of behavior.

Keep `make reverse` exclusive in its checkout: it temporarily mutates source files and restores
them. Run GPU/window checks sequentially when they compete for the same device or desktop.
Independent CPU checks can run together when they do not share mutable files or services.
On macOS, `TMPDIR=/private/tmp make check` keeps temporary Unix socket paths within the platform
length limit when the system's default temporary directory is too long.

If a dependency, display, or device blocks a gate, attempt recovery within the authorized scope
and complete independent checks. Report the blocked command, reason, and remaining coverage;
an unavailable check is not a passed check.

## Visual review and baselines

The agent inspects relevant captures and golden comparisons before delivery. A golden image is a
reviewed regression reference, not a visual quality score. Compare the intended change and areas
that should remain stable; fix unexpected differences before updating a reference. User sign-off
is required only when explicitly requested.

Use the existing target that demonstrates the behavior: examples include `make gizmo-gallery`,
`make ui-runtime`, `make lighting`, `make deformables`, and `make showcase`. Targets such as
`make outline`, `make gizmo`, `make perturb`, and `make settings` open interactive viewers. Exercise
the relevant behavior, capture the result under `output/`, and close only the viewer you started.
A scripted gallery can supply the evidence when it covers the same behavior.

`make ui-layout-audit` captures Camera, Settings, Inspector, Keyframes, and menus in English
and Chinese at normal and 150% UI scale. It checks horizontal containment, complete Shadow
quality labels, and Transform height after reflow. Images and measured bounds are written to
`output/ui-layout-audit/`; inspect the images as well as the assertions. The native input
regressions in `tests/gpu/test_ui_layout_input.py` exercise dock splitters, popup dismissal,
checkbox label/keyboard activation, and the timeline's distinct wheel and right-drag gestures.

Golden comparison and baseline updates are separate actions. Scope them to the affected cases:

```bash
make golden ARGS='showcase'
```

After the agent reviews an intentional, in-scope visual difference, accept and compare that case:

```bash
make golden-accept ARGS='showcase'
make golden ARGS='showcase'
```

Omit `ARGS` for the full set. The agent can update a baseline for an intended change within the
task after inspecting the difference; do not refresh unrelated references or relax thresholds
merely to make checks pass. Reviewed references remain in `tests/golden/`; generated comparisons
and reports stay under `output/`.

When a task produces visual results, include clickable absolute paths to representative images,
galleries, or videos in the final response and briefly explain what they demonstrate. Prefer a
useful before/after view or final result over a list of every diagnostic file. Show an image inline
when useful. Sharing these results lets the user inspect the work without making their review a
completion gate.

## Isolated Wayland acceptance on Linux

For a visible desktop you can operate with your own mouse and keyboard, run:

```bash
make native-wayland-viewer
make native-wayland-viewer ARGS='--renderer opengl'
MOJIVE_UI_SCALE=2.5 make native-wayland-viewer SCENE=joint_gizmo
```

This opens a nested Weston desktop window on the current X11 or Wayland desktop. Mojive inside
it uses native Wayland. Move, resize, or maximize the Mojive window using its title bar, then
exercise its viewport and controls normally. The default renderer is bgfx; `--renderer wgpu`
is also available. Desktop dimensions can be set with `ARGS='--width 1920 --height 1200'`;
flags after `--` are forwarded to `mojive view`. Close Mojive, close Weston, or press Ctrl+C
in the launching terminal to stop both. Settings are isolated for each launch; logs remain in
`output/native-wayland-viewer/`. This target intentionally opens a visible window only when
invoked; automated agent checks should continue using the headless target below.

On Ubuntu 22.04, `make native-wayland-test` prepares a private Weston 9 environment under
`build/wayland/`. It downloads and extracts distribution packages without installing
them system-wide. Preparation requires `curl`; compilation requires a C compiler, `pkg-config`, and the existing Wayland,
Pixman, and xkbcommon development headers. A headless EGL compositor owns a virtual input seat;
the viewer still receives real `wl_pointer`, `wl_keyboard`, and `wl_data_device` events.
No window, input handle, or display connection is attached to the user's desktop.

The target runs bgfx, OpenGL, and wgpu in separate processes. It checks picking, camera drag and
zoom, pause shortcuts, paired modifier keys, focus loss, input ownership across ImGui contexts,
simultaneous windows and peer closure,
external Unicode clipboard transfers, and dragging a model from a path containing spaces and
Chinese characters. Compositor captures are compared with the application's window image, so
a valid offscreen frame with a blank presented window does not pass. Logs, JUnit results, and
paired captures are saved to `output/native-wayland/`.

Use `ARGS="--renderer bgfx"` for focused iteration, or provide `--weston-prefix /path/to/usr`
and `--weston-source /path/to/weston-9.0.0` to reuse a matching installation. The test module uses
Weston 9's internal backend ABI and is loaded only into the owned compositor. Input control uses
inherited private sockets; compositor and helper processes are stopped when the run finishes.

This environment validates the actual GPU when its EGL/Wayland driver supports the compositor.
It does not measure monitor scanout latency or validate GNOME/KDE integration, input methods,
mixed-DPI monitors, or desktop decorations. Weston 9's kiosk shell and libdecor 0.1 offset client
content by the shadow margin; the input driver measures that offset and compares the overlapping
visible image. A nested Weston running inside Xvfb instead uses software composition and may
require Mesa's software Vulkan ICD; it is not a substitute for GPU performance measurements.

## Renderer performance

`make model-loading` captures the empty scene, file-drop preview, and MJCF/URDF loading UI.
For resource-loading measurements on an existing model, use the same target without a window:

```bash
make model-loading ARGS='--offscreen --asset /path/to/scene.xml --renderer opengl --profile'
```

This initializes an empty renderer before timing source preparation, resource replacement, and
the first readable image. Device startup is excluded; image readback waits for GPU completion.
The output directory contains `model.png`, `report.json`, and, with `--profile`,
`resources.prof` / `resources.txt`. Omit profiling for repeated latency measurements, and run
them separately from other tests. This measures the resource path used by runtime loading;
window presentation and UI responsiveness still require the windowed target.
`tests/gpu/test_backend_parity.py` checks OpenGL power-of-two texture filtering without redundant
CPU mipmap generation. `tests/gpu/test_texture_mipmaps.py` reads back WebGPU mip levels for 2D
and cube textures, checking sRGB, alpha, narrow extents, resource reuse, and GPU-only generation
for power-of-two sizes. Non-power-of-two sizes retain shared area filtering; CPU tests compare
its sparse reduction against exact source-pixel overlap.

`make timeline-profile` measures production Keyframes panel CPU draw-data generation at 100,
10,000 and 100,000 keys, including zoom, pan and full selection. It requires no display server.
On Linux, `make timeline-profile ARGS='--capture'` also renders the panel through offscreen EGL
and saves images under `output/timeline-profile/`. The JSON report records CPU samples, vertex
counts and the capture device; CPU timings exclude GPU submission and are not viewer frame rates.

The production physics/render concurrency comparison includes the official MuJoCo 100-humanoid
model (1,600 moving bodies, 2,700 degrees of freedom), deformable stress, and a lightweight rigid
scene:

```bash
make physics-concurrency
make physics-concurrency HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
```

This target compares `ViewerConfig(threaded_physics=False)` and the default concurrent driver
through the real Viewer and Session. It alternates modes over three repeats, uses identical
render quality, verifies physics against serial replay, and saves captures, frame samples, and
thread timelines under `output/physics-concurrency/`. The default model path follows the first
`MUJOCO_MODEL_ROOTS` checkout. No models are downloaded by the target.

Report both rendering rate and simulation progress. `display_time_lag_ms` measures displayed
simulation time against elapsed target time; a recently published snapshot can still represent
a late simulation. Frame work and `sync()` return timings are CPU/application metrics, not
display scanout latency. Each case owns a hidden window with VSync disabled and drains the GPU
before ending its throughput measurement. Run performance measurements separately from tests.

`make physics-render-benchmark` retains a fixed-workload experiment for isolating snapshot-copy
cost. Its ordered publication mode preserves every displayed state and applies backpressure;
it is distinct from the production latest-state policy. Pass `--production` to measure the real
Session runtime, and `--renderer wgpu` or `--renderer bgfx` to select another backend. The native backend requires
`MOJIVE_NATIVE_BUILD` when it has not been installed as a platform wheel.

Thread ownership, pause/step/history, command fences, model replacement, controls, replay, and
failure recovery are covered by `tests/test_threaded_physics.py`. Default viewer startup and
model-loading interaction additionally run through the GPU suites.

The quick renderer benchmark compares `mujoco.Renderer`, Mojive OpenGL, and Mojive wgpu through
their public `update_scene()` and `render()` APIs:

```bash
make renderer-benchmark
```

Each renderer/workload/output/resolution case owns an isolated process. The default matrix measures
RGB output at 640×480 for primitive, many-object, and dense-mesh scenes. The full matrix also covers
64- and 1,024-object dynamic transforms, textured/transparent materials, and 256 logical material
variants sharing a small mesh/texture-binding set. It records constructor and first-frame time,
update and render median/p95 latency, FPS, instance-stream upload bytes, close time,
shadow/reflection cache reuse, and peak RSS growth. Mojive cases additionally report backend-only
command-graph CPU time, optional aggregate GPU time, draw calls, and buckets separately from
synchronous readback in
`output/renderer-benchmark/report.json`. Ratios below `1.0x` are faster than MuJoCo for the same case.

Run the larger resolution and RGB/depth/segmentation matrix explicitly:

```bash
make renderer-benchmark ARGS="--renderers mujoco,mojive-opengl,mojive-wgpu,mojive-bgfx"
make renderer-benchmark-full
make renderer-benchmark ARGS="--workloads dynamic --modes rgb,depth --frames 200"
make renderer-benchmark ARGS="--workloads dynamic_large --modes rgb --resolutions 1920x1080"
```

The complete target writes `output/renderer-benchmark/full-report.json`, keeping the quick report
available for routine before/after comparisons.

RGB and depth reuse destination arrays. MuJoCo 3.11 does not accept its segmentation ID-pair shape
as an `out` array, so both implementations use allocating `render()` for segmentation. Internal
Mojive GPU pass timers are deliberately excluded from cross-renderer ratios. Baselines are recorded
with host and dependency versions and are not a cross-machine hard gate.

For visible native-window comparisons, record actual window pixels, viewport pixels, focus,
occlusion and Metal display-sync state. Use the same window size for both backends. To hold
scene work constant across display scales, use:

```bash
make native-window-benchmark ARGS="--render-size 1306 1036"
```

This fixes scene resolution only; the window framebuffer still follows the attached display.
Do not compare a 1× presentation directly with a prior 2× presentation or treat CPU submission
rate as physical input-to-photon latency.

The independent-world benchmark retains full-grid overview as its default. Use
`make g1-worlds-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie ARGS="--camera detail"`
to keep all worlds loaded while inspecting the center of the grid. This mode exercises camera
visibility rejection and runs its own image parity checks before timing. It does not reduce mesh
quality, remove offscreen shadow casters, or represent the overview workload.

## Documentation

Build the user guide and generated API reference with:

```bash
make docs-check
```

The focused checker also compares the CLI reference with the live parser, verifies render/debug
enum values and core `MOJIVE_*` variables, requires every example in both catalogs, and rejects
missing snippet or asset paths. The strict site build rejects broken links, unresolved API modules,
and documentation warnings.

## Native loading and UI diagnostics

`make native-load-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` starts a fresh
process for each model/backend, then performs three queued loads through an initialized Viewer.
The first process load and same-process reloads are reported separately. OS file caches are not
purged. Source preparation, resource preparation, first-frame submission, readback readiness and
the longest UI frame are distinct metrics; readback bounds GPU readiness but does not measure
physical scanout. The benchmark reads the already submitted window image rather than rendering
a second frame. Run it separately from performance and GPU checks.

`make native-ui-parity` captures OpenGL and bgfx through actual windows. It clicks all four
horizontal axis endpoints, checks that moving labels retain fractional positions relative to the
axis balls, and compares English/CJK glyphs, rounded outlines and multiple line widths at 100%
and 150% UI scale. Review the PNG comparisons and axis animation under `output/native-ui-parity/`.

`make viewcube-transitions BACKEND=bgfx` captures an orbit, all axis depth ties, and the shrinking
shaft's circle limit at 65%, 100%, 125%, and 150% UI scale. Inspect the paired PNGs and animation under
`output/viewcube-transitions/`. Its CPU timings cover widget update and draw submission, excluding
readback and presentation; run it separately from other tests or benchmarks. The GPU regression
also checks 65% and 250% UI scale and bounds pixel changes at depth ties and the shrinking shaft's
circle limit. It verifies that the white origin's surrounding shell reveals the scene background.
The `origin-hover-*.png` pairs show idle and hovered origins; only the white disk should change.
The same GPU test file exercises origin clicks, held input, drag/release cancellation, axis
occlusion, input gates, and projection switching from scene cameras through the composed viewer
with a static scene adapter.

## MuJoCo model corpus

`make mujoco-model-suite` compiles, adapts, and renders the XML files under the configured model
roots. Each model uses multiple azimuth and elevation views, RGB and segmentation output, and one
dynamic simulation step. Workers run in isolated processes so a native model failure has a stable
file-level result.

```bash
make mujoco-model-suite
make mujoco-model-suite ARGS="--backend wgpu"
make mujoco-model-suite \
  MUJOCO_MODEL_ROOTS="/path/to/model /path/to/another/model" \
  MUJOCO_MODEL_JOBS=8
```

The JSON report defaults to `output/mujoco-model-suite.json`. Unavailable plugin runtimes are
reported as `skipped_dependency`; compilation, adapter, render, and empty-output failures remain
test failures.

## Direct backend product comparisons

`tests/gpu/test_backend_parity.py` renders the same textured and transparent scene with both
backends, in linear and MuJoCo classic modes. RGB allows a mean difference below one display
level and p99 at most five levels; object-ID and segmentation disagreement must stay below 0.1%
of pixels; metric depth p99 on shared visible pixels must stay below 1e-4 world units. The test
also checks generated texture orientation and classic lighting saturation independently. These
are functional image checks, with reports and captures under `output/quality-improvements/`.

MuJoCo reference comparisons use approximate renderer-parity thresholds; they do not promise
pixel-identical shading. Review changed baselines using the workflow above.

## Joint gizmo edge views

`make joint-gizmo-profile BACKEND=bgfx` captures a limited hinge from both sides of
its edge-on fade and measures stationary and continuously changing cameras separately.
Use the reported link name to reproduce a particular model, for example:

```bash
make joint-gizmo-profile BACKEND=bgfx ARGS="--asset /path/to/unitree_g1/scene.xml --link left_wrist_roll_link --distance 0.42"
```

Run this separately from other benchmarks and GPU tests. The default window is hidden,
VSync is disabled, and layout persistence is disabled. `--visible` enables visible-window
measurements, which must be reported separately. Frame and overlay times measure CPU/application
work, not display scanout. Inspect the captures and JSON under `output/joint-gizmo-profile/`.
The CPU regression also exercises the production ImGui painter and bounds exact contour
predicates during fading, without relying on machine-dependent timing thresholds.

## Composed-model editor latency

`make hierarchy-benchmark` checks browsing and filtering of 1,500 authored objects (3,002
hierarchy nodes), scrolls to the last object, clicks it, and captures the resulting Inspector.
It records panel CPU time, submitted rows, and whole `sync()` time under
`output/hierarchy-benchmark/`. The window is hidden with VSync disabled; run it separately from
other benchmarks and GPU tests. `ARGS="--objects 5000 --frames 240"` increases the workload.
Browse and filtered results must both retain the last object; clipping limits drawing work,
not the number of accessible nodes. These timings do not measure display scanout latency.

`make timeline-benchmark` exercises 20,000 model keyframes in the production panel, including
overview, zoomed, panning and hover cases. It records panel and whole-frame median/p95 timings,
projected marker counts, framebuffer scale and captures under `output/timeline-benchmark/`.
`ARGS="--markers 50000 --frames 240"` increases the workload. Compare runs with the same
framebuffer scale and workload; stationary caching results do not describe panning costs.

`make recording-benchmark` compares paced camera motion with and without full-window interactive
recording. Reports include frame percentiles, encoder write and buffer-submission timings,
late frames, initial capture and finalization cost, and the encoded dimensions. Video and JSON
are written under `output/recording-benchmark/`. Run both benchmarks separately from GPU tests
and other timed workloads. These are diagnostic measurements, not hardware-independent limits.

`make native-editor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` creates an empty
workspace, adds a floor and MS-Human-700, then selects, previews placement, captures/loads/removes
keyframes, composes Go2, applies MJCF, removes a model, and exercises Undo/Redo. Use
`ARGS="--backend opengl --output output/editor-opengl"` for the reference backend. Run each
backend separately from GPU tests and other benchmarks. `--hidden` isolates editor/render work
from visible presentation; report it separately from visible-window measurements.

Selection must cause zero scene resource uploads. Reports distinguish synchronous public command
latency from queued UI edit completion and longest UI frame; background compilation does not make
the underlying compile instantaneous. Images capture the selected human and composed workspace.

## Independent-world and input acceptance

`make g1-worlds-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie` compares small, detailed
G1 captures before timing 1024/2048/4096 independent worlds. Geometry is shared and motion
phases are seeded. Default runs retain original meshes; explicit `--mesh-ratio` / `--mesh-error`
runs use the same LOD in both renderers and must keep separate output directories. Render timing
includes completed RGB readback. Backend draw-call statistics have different scopes and must not
be compared as if they counted identical passes. Unavailable GPU timers are reported as null.

`make g1-worlds` is the interactive acceptance entry point; `make g1-worlds-transport` measures
separate publisher and receiver processes over loopback without rendering.
`make g1-worlds-monitor-benchmark` adds completed GPU output and reports both receive-time
and completed-image age. Run the two backend variants serially with identical mesh quality. See the
[native Viewer guide](../how-to/native-viewer.md#independent-world-replay) for input data and limitations.

`make native-editor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie ARGS="--gallery"`
also captures the expanded 700-actuator component table and mouse mapping settings after
completing its timed operations. `tests/gpu/test_input_mapping.py` exercises changed
navigation, multi-button acquisition/release, unsupported perturbation, and panel/slider remaps
through actual windows. Run these with `MOJIVE_RENDERER=opengl`, `wgpu`, and `bgfx` as applicable.

## Startup and documentation captures

```bash
make startup-profile ARGS="--backend opengl wgpu bgfx --asset joint_gizmo --repeats 3"
make startup-profile ARGS="--backend opengl --profile --compare-icons"
make readme-media
```

Build each requested backend before profiling. Startup trials run in fresh processes and record
imports, fonts, Session setup, first frame and window presentation separately. Profiling adds
overhead; keep those samples separate from ordinary timing runs. `--compare-icons` compares
production presets with dynamic icon fitting. Inspect JSON and images under
`output/startup-profile/`; do not equate a hidden black window with a presented usable frame.

README media uses isolated settings, the real jointed scene and fixed-size captures. Inspect
all three files under `output/readme-media/` before delivery. The documentation gate checks
local links, including the README image paths; the capture command checks identical dimensions.

## Dense timeline and recording lifecycle

`make editor-profile ARGS="--asset /path/to/model.xml --frames 90"` profiles idle frames,
camera orbit, visibility and color edits in a production viewer. Use `BACKEND=wgpu` for WebGPU.
Each case separates uninstrumented wall/thread CPU timings, inclusive stage timers and cProfile
attribution. Nested stages must not be summed. OpenGL drain measures outstanding GPU work;
other backends include color readback, so those values are not directly comparable GPU times.
GPU pass timestamps are included when available. Reports, profiles and a final window capture
go under `output/editor-profile/`. Run timed workloads separately from other checks, and preserve
backend, framebuffer scale, asset and window dimensions when comparing versions.

`make timeline-benchmark ARGS="--editable --markers 20000"` uses real MuJoCo presets
and measures held drag, area selection, and release separately from static browsing. Edit completion
is measured after queued work finishes and the model's new keyframe time is verified. Marker-only
runs omit `--editable`. These timings measure application work, not display scanout latency.

`make recording-benchmark ARGS="--width 1920 --height 1080 --fps 60 --preset slow"` reports
first-frame submission, encoder writes, bounded-buffer waits, stop-request latency and UI frames
during finalization. Width and height are window points; use the reported `encoded_size` for
the actual video resolution, especially on HiDPI displays. Vary dimensions, FPS and preset in
separate output directories; run without competing GPU work. Queue and lifecycle tests include initial-write ordering, paused failures,
finalization failures, full-buffer ownership and stalled encoder shutdown.

## Material and optional-feature acceptance

`make material-workflow` captures box/floor material replacement, a skybox, and a retained
floor grid whose spacing, width and color change under the same ID. On headless Linux,
use `MOJIVE_GL=egl make material-workflow`; pass `ARGS='--renderer wgpu'` for WebGPU.
The tool checks the box's color separately from the floor, so unrelated pixel changes cannot
hide a broken material binding. Output is written under `output/material-workflow/`.

`tests/test_feature_composition.py` verifies selected panel imports in a fresh interpreter,
later activation, disabled-panel policy, minimal Viewer construction and socket suppression.
`tests/test_timeline_gestures.py` exercises the production panel with real ImGui input without
a native window. `make timeline-profile ARGS='--capture'` captures the same panel with EGL.
