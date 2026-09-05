# Repository guidance

## Project scope

Mojive is a backend-neutral 3D viewer for simulation and tooling. OpenGL is the default rendering
backend; WebGPU is available through the optional wgpu backend. MuJoCo, custom physics engines,
programmatic scenes, remote publishers, and snapshot replay enter through scene adapters.

## Execution

- Carry the requested work through implementation and applicable verification. Resolve routine
  choices using repository conventions and available evidence; keep existing authorization for
  the same scope. Honor requests for review or proposals before edits.
- Ask when a consequential decision has no reliable default or an explicit approval requirement
  applies. Prepare the authorized, reviewable work before requesting approval for the gated action.
- Continue independent, authorized work around blockers. If guidance causes a pause, cite its
  file and exact clause, distinguishing a requirement from your interpretation. Report the concrete
  blocker and unmet requirement; reporting incomplete work does not replace attempting recovery.
- Parallelize independent reads and checks when useful. Serialize dependent mutations and checks
  that compete for shared state. Task authorization does not expand platform permissions.
- Use `skills/mojive/SKILL.md` for operating scenes; use the development guidance here for code
  changes. Read linked guides as needed. Historical plans provide context, not current requirements.

## Architecture

- Shared contracts live in `types.py`, `commands.py`, `math3d.py`, and `adapters/base.py`.
- Adapters emit `SceneSource`, `SceneFrame`, mesh updates, and debug commands.
- Render modules use shared contracts across application layers and remain independent of UI and
  physics implementations. Render internals and backend libraries are valid dependencies.
- UI code imports session state and protocols.
- `Session` owns application state, selection, overrides, and command routing.
- Mojive scene entities own cameras, lights, materials, and authoring metadata.
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
- Put temporary captures, recordings, and reports under `output/`. Maintained documentation media
  and reviewed golden fixtures use their existing tracked locations when their update is in scope.
- Reuse or extend a Make target for visual acceptance of visible behavior; add a target when no
  existing one demonstrates the change.

## Verification

Use the [verification matrix](docs/guides/testing.md#change-mapping) as the single source of
required checks. Start with focused verification, then complete the applicable gates. Once they
pass, repeat or broaden checks only for further changes, failures, or unresolved concerns.

The agent inspects relevant visual results and golden comparisons. User sign-off is needed only
when explicitly required. Baseline updates follow the [visual review workflow](docs/guides/testing.md#visual-review-and-baselines).

## Delivery

State what changed, how it was verified, and any unmet requirement. When the task produces visual
results, link representative files under `output/` using absolute filesystem paths and briefly
explain what to look at. Show an image inline when useful. Providing these results for the user to
view does not create an additional approval step.

## Git

- Preserve unrelated working-tree changes.
- Use concise imperative English commit subjects.
- Group commits by coherent behavior.
