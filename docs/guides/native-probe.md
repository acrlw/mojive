# Native renderer probe

The optional `cpp/` project evaluates a C++ renderer without changing the Python Viewer,
Session, renderer selection, dependencies, or public rendering APIs. It is an experimental
subset, not a replacement editor. See the [Chinese migration proposal](../plans/native-cpp-bgfx.zh.md)
for the broader plan. The [current backend decision](../plans/native-backend-decision.zh.md) selects bgfx for the native implementation and records the latest evidence and remaining platform gates.

## Boundaries

`mojive_render_contract` contains standard-library-only scene, camera, image, target, texture,
UI packet, and completion types. `mojive_render_core` validates inputs and implements a bounded
synchronous readback consumer. Neither target includes or links bgfx, SDL, GLFW, ImGui, or a Python binding library.

`mojive_backend_bgfx` implements the `Renderer` interface and privately owns bgfx handles,
view allocation, instance layouts, frame counters, staging textures, and device lifecycle.
`mojive_backend_sdl` implements the same interface with SDL3 GPU command buffers, transfer
buffers, fences, and swapchains. Only executable composition roots select a factory; switching
between `bgfx` and `sdl` does not change scene producers or the common readback consumer. The recording test double compiles and exercises that consumer with
both rendering backends disabled.

The ImGui adapter converts draw lists into generic `UiFrame` packets. The renderer itself does
not include ImGui or physics headers. Native OS window handles enter through `NativeWindow`;
no graphics-device handles cross the common boundary.

## Output and lifetime contracts

- Scene matrices are affine, row-major, and use a Z-up world. Camera projection uses canonical
  NDC depth in `[-1, 1]`; the backend converts it when necessary.
- CPU images are tightly packed with a top-left origin. Color is RGB `uint8`, metric depth is
  `float32`, object ID is `uint32`, and segmentation contains two signed `int32` values per pixel.
- Background ID is zero, segmentation is `(-1, -1)`, and metric depth equals the camera far plane.
- Readback tickets retain target generation, source revision, frame sequence, camera revision,
  and submission identity. A stale source cannot silently become a selection in a newer scene.
- A token can enqueue a readback only while its image is still current. Once queued, the
  backend preserves that image even if the same target is rendered again before `advance()`.
- Targets share one runtime. Closing a peer does not stop the others. Resize, scene replacement,
  and destruction cancel pending results; their staging storage survives until GPU completion.
- The queue has eight reusable staging slots. Exhaustion applies explicit backpressure. Callers
  must consume tickets, including canceled results. `poll()` does not advance GPU work.
- Only the owner thread calls the renderer. `advance()` submits work without exposing bgfx's
  completion frame numbers. Interactive applications poll between events; `waitForReadback()`
  is intended for offline consumers and validation.

The prototype uses separate color and data passes. Color supports 1x/4x MSAA. Data uses
single-sample attachments and lossless RGBA8 packing for each 32-bit identity word; it does not
convert a complete ID through a float. Integer render-target capability is reported separately
and does not claim that a native integer-fragment-output path has been tested.

bgfx's five-vector portable instance layout is contained inside the backend: three affine rows
are followed by either color or identity metadata for the corresponding pass. No upstream
source patch or vendor-specific scene layout is required.

## Build and verification

Install a C++20 compiler, CMake 3.24 or newer, and Ninja. Production sources come from vendored ImGui and initialized Git submodules; optional comparison
builds download checksum-locked dependencies described in `thirdParty/dependencies.json`. The bgfx/bx/bimg versions come
from one pinned bgfx.cmake revision. Build products stay under `output/`.

```bash
make native-test
make native-probe
```

`native-test` configures a separate build with both rendering backends disabled and checks common dependency
boundaries. `native-probe` runs real GPU checks for exact IDs above `2^24`, negative segmentation,
metric depth, background values, MSAA edge identities, one-pixel regions, dynamic mesh updates,
resize, peer destruction, source replacement, stale input, queue backpressure, owner-thread
violations, and runtime restart. It writes images and `conformance.json` to `output/native-probe/<backend>/`.

The tested host is macOS/Metal. Windows/D3D12 and Linux/Vulkan are selected by the build but
remain unverified on physical hardware. Linux currently uses X11; Wayland is not implemented
in the window adapter. A Metal device without an OS window is tested separately from the
visible gallery; this does not establish Linux server/headless support.

The build explicitly selects the platform renderer and disables WebGPU. Its shader compiler
uses upstream sources with `SHADERC_CONFIG_HAS_TINT=0`, without linking Tint/Dawn. Merely setting
`BGFX_CONFIG_RENDERER_WEBGPU=0` is insufficient: selecting any backend macro also disables
bgfx's automatic backend selection. No vendor sources are modified.

The SDL option and its offline portable shader pipeline are described below.

On the tested Clang host, `-DMOJIVE_ENABLE_SANITIZERS=ON` instruments Mojive's native code
with AddressSanitizer and UndefinedBehaviorSanitizer. Disable it for performance runs; vendor
libraries are not instrumented by this option. GPU checks used `ASAN_OPTIONS=detect_leaks=0`,
so they are not a leak-sanitizer certification.

## Visual gallery

```bash
make native-gallery HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
```

Set `NATIVE_FONT_LATIN` and `NATIVE_FONT_CJK` to existing JetBrains Mono and Noto Sans SC/CJK font
files when they are outside the normal macOS Mojive cache. The gallery shows real scene output,
ImGui docking, dynamically added CJK glyphs, a second native surface, and a main-window resize.
It captures `gallery.ppm` and records actual framebuffer scaling in `gallery.json`, then closes
only the windows it created. `MOJIVE_PROBE_UI_SCALE=1.5` exercises a larger UI scale; this is
reported separately from the operating system's actual framebuffer scale. The fixture uses
ImGui's default font rasterizer with the existing font assets; FreeType integration is deferred.

The native gallery is a rendering integration fixture. It does not implement the production
panels or automatic Dear ImGui detached-window callbacks. Secondary surfaces are explicitly
created and closed. Both adapters now resize a secondary surface in place while preserving its public identity.

## Fixed-trajectory benchmark

```bash
make native-benchmark HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make native-benchmark HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml \
  ARGS='--modes pick,color --resolutions 1920x1080 --fps-limit 120 --output output/native-probe/paced'
```

The exporter loads the official 100-humanoid model through the existing adapter and source
builder. On the tested model this produces 1,600 moving bodies, 2,700 degrees of freedom,
1,901 geoms, and 5,101 render instances across four shared meshes. Its private `.mjvp` stream
contains mesh data, exact identity metadata, and 120 recorded transform frames. It is not a
new public scene format. Export metadata includes the source path and a content checksum.

Runs alternate backend and mode order across repeats, warm up the selected path, reuse staging textures,
limit pending readbacks, and drain GPU work before recording total throughput. Modes distinguish
no readback, a one-pixel pick, and continuous RGB/depth/segmentation output. JSON retains each
run and aggregates the median of run statistics. CSV columns are separate sample sequences;
GPU and completion samples are asynchronous and must not be interpreted as the same frame's
end-to-end timeline. Empty columns mean no sample, not zero cost.

This renderer intentionally has simple diffuse shading and omits production shadows,
reflections, material textures, overlays, live physics, and the full editor. Its FPS must not
be compared as a speedup ratio against production Mojive. The benchmark isolates rendering and
readback design risks. A production comparison must use the current threaded-physics main,
match effects and physical viewport size, and measure simulation progress and displayed state
age as well as frame rate. GPU timing spans and readback completion are not display scanout
or input-to-photon latency.

## Initial measured result

On 2026-09-08, the M5/Metal probe passed conformance and runtime restart, the standalone core
passed without bgfx, and the visible gallery passed at 1x and 1.5x UI scale. The available display
reported a 1x framebuffer; actual 2x framebuffer and cross-display DPI behavior remain unverified.
The existing Python checks passed (1,625 fast and 129 integration cases), as did strict docs.

Three alternating 10-second runs per mode, after 90 warmup frames, produced the following
median throughput. These numbers describe only the simplified fixed-trajectory probe.

| Output per frame | 1920x1080 FPS | 2560x1440 FPS |
|---|---:|---:|
| None | 889.5 | 850.9 |
| One-pixel pick | 562.5 | 554.6 |
| RGB | 159.2 | 129.1 |
| Metric depth | 157.6 | 130.4 |
| Segmentation | 143.6 | 114.0 |

At a software-paced 120 FPS in 1080p, pick CPU frame P95 was 0.37 ms and completion P95 was
17.98 ms. RGB completion P95 was 21.96 ms. This supports continuing the renderer evaluation,
while retaining readback latency as an acceptance gate for real editing and recording.
It does not establish a production speedup or native physics/render concurrency.

The local Chinese report is `output/native-probe/report.zh.md`; raw records and separate sample
CSV files live under `output/native-probe/benchmark/` and `output/native-probe/paced/`.


## SDL3 GPU comparison

```bash
make native-probe NATIVE_BACKEND=sdl
make native-gallery NATIVE_BACKEND=sdl HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make native-benchmark NATIVE_BACKEND=sdl HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml \
  ARGS='--backends bgfx,sdl --output output/native-probe/sdl-comparison'
```

SDL GPU is a graphics API abstraction with Metal, D3D12, and Vulkan drivers. SDL's window/input
library and SDL's newer GPU API have different maturity histories. The evaluation currently
uses one set of GLSL sources, compiled to SPIR-V by glslang and translated to MSL/HLSL by
SPIRV-Cross. Windows additionally compiles HLSL to DXIL with DXC. These tools are build-time
programs in an isolated CMake project; they are not linked into the SDL runtime. The generated
MSL path has passed real Metal conformance. SPIR-V generation and HLSL translation do not
establish Vulkan or D3D12 runtime correctness. The evaluation workflow has been removed at the user's request; Linux will be validated locally
on the user's Linux system. Windows/Linux platform runtimes have not run in this preparation. The SDL-only build does not fetch
bgfx, bx, bimg, or their shader compiler; pass `NATIVE_CMAKE_ARGS=-DMOJIVE_BUILD_BGFX=OFF` to check it.

The gallery keeps GLFW's existing platform and ImGui integration and wraps native
windows through SDL's public native-window properties. Cocoa is exercised locally; Win32
and X11 branches are implemented but remain subject to platform validation on the corresponding systems. This isolates the GPU comparison from
an unrelated event-system rewrite. The wrapper does not own the GLFW window. Both windows
share a GPU device; the experiment does not require a central daemon or one process per window.

Both adapters use the same two passes, geometry, instance count, color/MSAA settings, and exact
identity encoding. Both now explicitly bound outstanding GPU frames to two; the earlier bgfx
result used its default queue setting and is historical, not the comparison baseline. SDL's
fence is backend-private and becomes a common `ReadbackTicket`. A zero-copy borrowed MuJoCo
array is never retained for asynchronous GPU access. Upload buffers and readback slots are reused.
SDL GPU exposes no portable timestamp-query API used by this adapter, so its GPU timing fields
are `null`. CPU timing and completed readback latency remain measured. Readback completion is
observed when the application polls, and a frame-rate cap changes that observation interval.

Benchmark captures always show the same recorded frame, independent of measured throughput.
Data conformance checks also compare exact raw arrays from both implementations. No production
rendering effect or live simulation is added by this experiment.

## Python binding comparison

```bash
make native-bindings-test
make native-bindings-benchmark HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
```

The build adds two isolated extension modules under `output/cpp-bindings-build/bindings`.
It does not install a package, modify the Python app's dependencies, or replace MuJoCo's module.
Versions and source checksums for pybind11, nanobind, and nanobind's robin-map dependency are
locked. Both wrappers call the same compiled C++20 implementation with the same optimization
level; automatic LTO, stripping, and size-oriented flags are disabled for this comparison.
The binding-only build does not fetch or link any renderer or UI dependency.

The tests cover all six import orders with MuJoCo, exact NumPy buffer addresses, explicit owned
snapshots, read-only views that survive deletion of the original owner, rejected dtype/stride
conversions, wrong-thread frame mutation, and Python execution while native computation runs.
Both bindings explicitly release the GIL for the batched C++ work. Neither makes arbitrary
Python code parallel. Copy a physics state at an adapter synchronization boundary before
starting asynchronous work; the GIL does not protect a buffer while another native thread is
writing it.

MuJoCo's official Python package uses pybind11. Mojive can use nanobind without rewrapping
MuJoCo's C++ object types: exchange NumPy arrays, ordinary values, and Mojive-owned snapshots.
Different binding libraries' registered C++ wrapper objects are intentionally rejected in the
comparison; an array boundary works. Sharing raw `mjData*` ownership across extension modules
is outside this contract. A future native physics adapter should call MuJoCo's C API directly
and own its model/data lifetime rather than routing every physics step through Python.

The benchmark measures small calls, strict ndarray borrowing, copying the official humanoid
model's `qpos`, and packing the same fixture's 5,101 transforms into two 80-byte records per
instance. It also measures a shared native computation after releasing the GIL. Seven alternating
runs retain raw values and median timings. These are boundary/kernel measurements, not claims
about production FPS, GIL-free Python, or a full C++ editor. Choosing nanobind does not change
the standard-library-only renderer contracts; the binding library stays in two module sources.

Official references: [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html),
[nanobind design](https://nanobind.readthedocs.io/en/latest/why.html),
[SDL GPU API](https://wiki.libsdl.org/SDL3/CategoryGPU), and
[SDL native-window properties](https://wiki.libsdl.org/SDL3/SDL_CreateWindowWithProperties).


## Initial comparison (historical, 2026-09-08)

The same M5/Metal host completed 60 alternating uncapped runs and 12 runs paced at 120 FPS.
Each combination used three 10-second runs after 90 warmup frames. The table reports median
throughput of the fixed-trajectory prototype with an explicit two-frame GPU queue budget.

| Output per frame | bgfx 1080p FPS | SDL 1080p FPS | bgfx 1440p FPS | SDL 1440p FPS |
|---|---:|---:|---:|---:|
| None | 824.8 | 887.2 | 793.7 | 832.3 |
| One-pixel pick | 587.1 | 894.3 | 522.8 | 844.6 |
| RGB | 182.7 | 211.3 | 141.6 | 129.1 |
| Metric depth | 161.1 | 183.5 | 130.8 | 110.0 |
| Segmentation | 142.6 | 177.4 | 113.4 | 107.1 |

At 120 FPS in 1080p, pick completion P95 was 19.11 ms with bgfx and
9.84 ms with SDL. RGB completion P95 was 22.89 ms and
13.36 ms respectively, but SDL's RGB CPU frame P95 was higher
(8.33 ms versus 6.40 ms). These early results motivated the live-runtime comparison below; the current decision selects
bgfx while retaining SDL as an experimental adapter. They do not establish a universal SDL
performance advantage.

The binding comparison used nanobind 3.0.1, pybind11 3.1.0, CPython 3.11.15, NumPy 2.4.6,
and MuJoCo 3.11.0. Seven alternating runs measured identical shared C++ kernels.

| Boundary or workload | pybind11 | nanobind |
|---|---:|---:|
| Scalar call, including Python loop | 39.4 ns | 24.0 ns |
| Strict ndarray borrowing | 176.5 ns | 134.9 ns |
| Copy 2,800 qpos values | 538.8 ns | 455.2 ns |
| Update 5,101 instance records | 10.67 us | 10.60 us |
| Shared native computation | 51.76 us | 52.15 us |
| Unstripped module | 281.4 KiB | 224.7 KiB |

Five alternating incremental wrapper rebuilds, with support libraries and the kernel already
built, took a median 1.788 s for pybind11 and 0.449 s for nanobind. This measures editing one
wrapper, not initial full-project compilation. Reproduce with
`python tools/benchmark_native_binding_builds.py` after `make native-bindings`.

Nanobind is the preferred thin-binding candidate for the next native core slice. Small boundary
costs and rebuild time improved; shared batch computation was effectively unchanged. Both
libraries passed 13 ownership, array, thread, and import-order checks with MuJoCo. Neither
library removes the need for explicit GIL release and a safe physics snapshot boundary.

SDL passed the shared conformance suite, including same-revision source replacement, both
UI scales, actual secondary surfaces, and renderer-only ASan/UBSan plus Metal API validation.
The three-object conformance outputs were bit-identical across backends. Full humanoid color
captures differed in less than 0.01% of pixels, with mean absolute channel error below 0.00035
on the 0-255 scale; these captures are not claimed to be bit-identical. SDL's Windows/Linux
shader runtime, real 2x framebuffers, and production effects remain untested. This paragraph
records the earlier comparison; live native physics and offline portable shader generation
were subsequently added by the decision acceptance below.
The Python regression and strict documentation gates passed. Ten locked dependency archives
and 8,900 extracted source files matched without vendor patches.

The full local Chinese report is `output/native-probe/sdl-nanobind-report.zh.md`. Independent
run records, CSV samples, and fixed-frame captures are under `sdl-comparison/` and `sdl-paced/`
within that output directory; binding data is under `bindings/`. Opaque renderer handles are
scoped to their renderer and must not be transferred between backend instances.


## Decision acceptance and live runtime

The final decision adds these reproducible targets:

```bash
make native-composition NATIVE_BACKEND=bgfx
make native-windows NATIVE_BACKEND=bgfx
make native-scene-capture NATIVE_BACKEND=bgfx HUMANOIDS_MODEL=/path/to/100_humanoids.xml
# Repeat with NATIVE_BACKEND=sdl to use the same tests against SDL GPU.
```

Composition checks render-to-texture dependencies with the consumer allocated before its
producer, plus alpha blending, clipping, texture orientation and 120 resource cycles. This
exposed a real bgfx adapter ordering bug: view allocation order previously dictated execution
order. The adapter now uses a complete view-order permutation for each submitted frame.
The patch belongs to Mojive; upstream bgfx files remain unchanged.

The window fixture creates, resizes, minimizes, restores and closes 24 peer windows while
checking output from the surviving device. `bgfx::updateSwapChain` and SDL's drawable-size
acquisition preserve surface identity on resize. macOS physical-footprint samples are retained,
including one after shutdown. Footprint growth is an observation, not a leak-sanitizer pass or
proof of a leak. `mojive_sdl_window_audit` isolates the SDL-only lifecycle without GLFW, ImGui
or Mojive rendering code. Actual Retina and cross-monitor behavior remain unverified.

For live physics, provide the MuJoCo Python distribution directory containing `include/` and
its native shared library. This links the C API directly; the executable does not embed Python.
The macOS wheel retains a framework install name, so the build supplies a local runtime symlink
without modifying the wheel.

```bash
make native-runtime HUMANOIDS_MODEL=/path/to/100_humanoids.xml \
  NATIVE_CMAKE_ARGS='-DMOJIVE_BUILD_SDL=ON -DMOJIVE_MUJOCO_ROOT=/path/to/site-packages/mujoco'
```

The runtime first compares all 120 exported poses to native MuJoCo output, including all split
capsule instances. It then measures 100-humanoid live physics serially and on an independent
thread. Three preallocated snapshots rotate through producer, completed mailbox and consumer;
only a completed buffer is exchanged under the lock. Renderer calls stay on their owner thread.
No `mjData` or writable NumPy view crosses the snapshot boundary.

The matrix alternates both adapters and execution modes across three 10-second repetitions at
1080p/4x MSAA/120 FPS pacing. Pick-only runs request one pixel each frame. Recording runs use
one pick every second frame and RGB/depth/segmentation every fourth frame; multi-camera runs
add a 640x480 camera and RGB every fourth frame. These are 60/30 Hz at 120 FPS; serial runs
below the cap produce fewer outputs, whose counts remain in the report.
The final workload waits at a bounded queue when necessary and records every requested output,
so reduced work cannot masquerade as lower frame cost. Results report frame CPU percentiles,
simulation steps, snapshot age, observed readback latency (starting before the request API
call), and completed request counts. This is offscreen throughput, not measured display scanout.
The earlier drop-on-full experiment is retained separately and is not the final performance
comparison. Neither test includes production shading or the complete editor.

Final raw reports are under `output/native-probe/runtime/`, `composition-*`, `windows-final-*`,
and `scene-final-*`. The Chinese delivery report is `output/native-probe/backend-decision.zh.md`.
