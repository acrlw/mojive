# Native rendering fixtures

Native tools isolate C++ rendering, resource and window contracts. Use the
[production viewer](../how-to/native-viewer.md) to inspect actual editor behavior. The bgfx
implementation is shared with the product; the SDL GPU adapter remains an optional comparison
fixture, not a selectable Python renderer.

## Boundaries

`cpp/include/mojive/` defines scene, camera, image, resource and completion contracts.
Private adapters own backend handles and submission. Native ImGui adapters translate UI draw
data into packets without making the renderer depend on editor panels or MuJoCo state.

Matrices are row-major with Z-up world coordinates. CPU images use top-left origin and the
[public image types](../concepts/rendering.md#image-contracts). Readback tickets retain source,
frame, target and camera identity. Replaced inputs must produce an explicit cancellation rather
than a valid-looking result from another scene.

## Build and run

Use a C++20 compiler, CMake 3.24+, Ninja and initialized pinned submodules. Build products and
reports stay under `output/`; dependency provenance lives in `3rdparty/README.md` and
`3rdparty/dependencies.json`.

| Target | Purpose |
|---|---|
| `make native-test` | CPU contracts with graphics backends disabled |
| `make native-probe` | GPU image formats, IDs, depth, readback, resource lifetime and backpressure |
| `make native-composition` | UI texture dependencies, alpha and scissor behavior |
| `make native-windows` | Repeated surface creation, resize and teardown |
| `make native-gallery` | Native scene/UI integration captures |
| `make native-runtime` | Bounded simulation/render scheduling |
| `make native-bindings-test` | Binding ownership, NumPy behavior and GIL release |
| `make native-benchmark` | Fixed-trajectory renderer measurements |

`NATIVE_BACKEND=sdl` selects the optional comparison fixture where supported. It does not select
the product's renderer. Additional model fixtures take
`HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml`; gallery font paths can be set
through `NATIVE_FONT_LATIN` and `NATIVE_FONT_CJK`.

The fixture gallery creates its own surfaces and closes only those surfaces. It does not
implement the editor's panel collection. Inspect its captures alongside `conformance.json` and
other reports; a successful process exit alone does not establish visual parity.

## Measurement and acceptance

Use the [testing matrix](testing.md#change-mapping) for the required checks. Separate native
fixtures, public Python rendering and shown-window measurements: they cover different boundaries.
Compare fixed camera trajectories, resolution, MSAA, scene geometry and requested image products.
Count completed frames and readback where required, not just queued submissions.

Run benchmark variants serially without other GPU tests. Disable sanitizers for timing. Record
platform, GPU, driver, renderer, build mode, input paths and command arguments with results.
Do not reuse old host numbers as current performance claims. Optional SDL or binding comparisons
are engineering tools; retaining them does not imply that their backend is supported by the
production Viewer API.
