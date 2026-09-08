# Native renderer probe

The optional `native/` project evaluates a C++ renderer without changing the Python Viewer,
Session, renderer selection, dependencies, or public rendering APIs. It is an experimental
subset, not a replacement editor. See the [Chinese migration proposal](../plans/native-cpp-bgfx.zh.md)
for the broader plan.

## Boundaries

`mojive_render_contract` contains standard-library-only scene, camera, image, target, texture,
UI packet, and completion types. `mojive_render_core` validates inputs and implements a bounded
synchronous readback consumer. Neither target includes or links bgfx, GLFW, or ImGui.

`mojive_backend_bgfx` implements the `Renderer` interface and privately owns bgfx handles,
view allocation, instance layouts, frame counters, staging textures, and device lifecycle.
Only executable composition roots select `make_bgfx_renderer`. A future backend implements
`Renderer` and supplies another factory; it does not require changing scene producers or the
common readback consumer. The recording test double compiles and exercises that consumer with
`MOJIVE_BUILD_BGFX=OFF`.

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
  completion frame numbers. Interactive applications poll between events; `wait_for_readback()`
  is intended for offline consumers and validation.

The prototype uses separate color and data passes. Color supports 1x/4x MSAA. Data uses
single-sample attachments and lossless RGBA8 packing for each 32-bit identity word; it does not
convert a complete ID through a float. Integer render-target capability is reported separately
and does not claim that a native integer-fragment-output path has been tested.

bgfx's five-vector portable instance layout is contained inside the backend: three affine rows
are followed by either color or identity metadata for the corresponding pass. No upstream
source patch or vendor-specific scene layout is required.

## Build and verification

Install a C++20 compiler, CMake 3.24 or newer, and Ninja. The full build downloads checksum-locked
source archives described in `native/dependencies.lock.json`. The bgfx/bx/bimg versions come
from one pinned bgfx.cmake revision. Build products stay under `output/`.

```bash
make native-test
make native-probe
```

`native-test` configures a separate build with bgfx disabled and checks common dependency
boundaries. `native-probe` runs real GPU checks for exact IDs above `2^24`, negative segmentation,
metric depth, background values, MSAA edge identities, one-pixel regions, dynamic mesh updates,
resize, peer destruction, source replacement, stale input, queue backpressure, owner-thread
violations, and runtime restart. It writes images and `conformance.json` to `output/native-probe/`.

The tested host is macOS/Metal. Windows/D3D12 and Linux/Vulkan are selected by the build but
remain unverified on physical hardware. Linux currently uses X11; Wayland is not implemented
in the window adapter. A Metal device without an OS window is tested separately from the
visible gallery; this does not establish Linux server/headless support.

The build explicitly selects the platform renderer and disables WebGPU. Its shader compiler
uses upstream sources with `SHADERC_CONFIG_HAS_TINT=0`, without linking Tint/Dawn. Merely setting
`BGFX_CONFIG_RENDERER_WEBGPU=0` is insufficient: selecting any backend macro also disables
bgfx's automatic backend selection. No vendor sources are modified.

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
created and closed; resizing a secondary surface requires recreation in this probe.

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

Runs alternate mode order across repeats, warm up the selected path, reuse staging textures,
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
