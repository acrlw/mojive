# Testing

The suite is split by runtime and dependency boundary. Use the smallest layer that covers the
change while iterating, then run the required acceptance targets before handoff.

| Layer | Marker or target | Coverage |
|---|---|---|
| Fast | `make test-fast` | pure CPU behavior and module contracts |
| Examples | `make examples-check` | example syntax and import-independent entry points |
| Documentation | `make docs-check` | public docstrings, CLI/config reference, both example catalogs, asset paths, snippets, strict site build |
| Integration | `make test-integration` | files, serialization, protocols, processes, composition |
| Physics | `make test-physics` | model compilation and live physics worlds |
| OpenGL GPU | `make gpu` | real OpenGL contexts and rendered output |
| WebGPU | `make gpu-wgpu` | Metal, Vulkan, or DX12 backend behavior |
| Golden | `make golden` | reviewed image baselines |
| Full | `make test-all` | CPU, physics, OpenGL, and WebGPU layers |

## Change mapping

| Change | Iteration target | Acceptance target |
|---|---|---|
| Shared types or commands | `make test-fast` | `make check` |
| File format or remote protocol | `make test-integration` | `make check` |
| MuJoCo adapter or MJCF authoring | focused physics test | `make test-physics` and `make mujoco-audit` |
| Public renderer performance | isolated quick matrix | `make renderer-benchmark` |
| MuJoCo model loading | one XML path with the model-suite module | `make mujoco-model-suite` |
| Render pass or shader | one GPU test file | `make gpu` and `make gpu-wgpu` |
| Visual interaction | focused GPU test | relevant Make gallery and `make reverse` |
| Settings layout | focused UI GPU test | `make settings` |
| Documentation or examples | `make examples-check` | `make docs-check` |

Markers may be combined. A file-format test that compiles MuJoCo uses both `integration` and
`physics`; it runs in the physics layer.

## Renderer performance

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

## Documentation

Build the user guide and generated API reference with:

```bash
make docs-check
```

The focused checker also compares the CLI reference with the live parser, verifies render/debug
enum values and core `MOJIVE_*` variables, requires every example in both catalogs, and rejects
missing snippet or asset paths. The strict site build rejects broken links, unresolved API modules,
and documentation warnings.

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

The scene and material goldens were refreshed on 2026-09-05 after reviewing the earlier classic-lighting
migration, correcting generated primitive texture coordinates and clamping lighting before
texture modulation. The corresponding MuJoCo reference comparison retains its separate
approximate renderer-parity thresholds; it does not promise pixel-identical MuJoCo shading.
