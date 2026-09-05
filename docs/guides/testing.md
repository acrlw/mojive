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
| Golden | `make golden` | reviewed image baselines |
| Full | `make test-all` | CPU, physics, OpenGL, and WebGPU layers |

## Change mapping

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
