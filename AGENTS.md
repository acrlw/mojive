# Repository guidance

## Project scope

forge-viewer is a backend-neutral 3D viewer for simulation and tooling. Forge is the rendering
backend. MuJoCo, custom physics engines, programmatic scenes, remote publishers, and snapshot
replay enter through scene adapters.

## Architecture

- Shared contracts live in `types.py`, `commands.py`, `math3d.py`, and `adapters/base.py`.
- Adapters emit `SceneSource`, `SceneFrame`, mesh updates, and debug commands.
- Render code imports shared contracts only.
- UI code imports session state and protocols.
- `Session` owns application state, selection, overrides, and command routing.
- Forge scene entities own cameras, lights, materials, and authoring metadata.
- Physics adapters own simulation state and capability-specific write-back.
- `tests/test_layering.py` enforces dependency boundaries.

## Coordinate conventions

- Python matrices are row-major; translation is `matrix[:3, 3]`.
- `math3d.to_gl()` performs the upload-boundary transpose.
- World coordinates use Z-up.
- Render target metadata defines vertical image orientation.

## Terminology

Use these names consistently:

- position and rotation for transform components
- body frame and world frame for gizmo orientation
- translation and rotation perturbation for physical mouse interaction
- scene source for stable structure and scene frame for dynamic data
- object ID for selection identity and body index for physics lookup
- render pass for one named renderer stage
- visual group for MuJoCo category filters
- render flag for renderer feature switches

## Implementation style

- Keep changes focused and compact.
- Prefer direct control flow and established invariants.
- Use professional English for code, comments, logs, UI copy, documentation, and commits.
- Comments explain architectural constraints, platform behavior, and non-obvious algorithms.
- Public names describe domain meaning and match existing terminology.
- Hot frame paths reuse buffers and avoid transient allocations.
- Output artifacts belong under `output/`.
- User-visible features include a Make target for visual acceptance.

## Verification

Run the smallest relevant test during iteration, then finish with:

```bash
make check
```

Rendering changes also run:

```bash
make gpu
```

Relevant visual targets include:

```bash
make outline
make gizmo
make gizmo-gallery
make perturb
make lighting
make deformables
make showcase
```

MuJoCo adapter changes also run:

```bash
.venv/bin/pytest -q -m physics
make mujoco-audit
make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables
```

Use `make reverse` when changing a registered regression invariant. Review generated gallery and
golden images before acceptance.

## Git

- Preserve unrelated working-tree changes.
- Use concise imperative English commit subjects.
- Group commits by coherent behavior.
- Keep generated captures and videos out of source directories.
