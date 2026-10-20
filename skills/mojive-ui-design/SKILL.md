---
name: mojive-ui-design
description: Design, implement, and review Mojive editor UI using the project's shared components, visual rules, and production verification. Use for layouts, typography, icons, controls, interaction, the design guide, or workbench. For operating scene content through public APIs, use the mojive skill.
---

# Mojive UI Design

Keep Mojive's workspace and design tools consistent through shared production behavior.
For a review-only request, produce findings or a proposal without changing implementation.
For implementation, carry the authorized change through the applicable verification.

Resolve links from this source directory, `skills/mojive-ui-design/`, including when loaded
through the `.agents/skills/` symlink. All code paths below are repository-relative.

## Locate the contract and owner

Read the repository [guidance](../../AGENTS.md) and [UI design rules](../../docs/guides/ui-design.md).
Use the relevant sections of the [review checklist](../../docs/reference/ui-design-checklist.md)
to identify the invariant and focused checks, without treating every entry as a required test.

Before editing, identify:

- The production presenter/control and its semantic owner, including sibling consumers.
- The adopted specification, font, UI scale, framebuffer density, state, and viewport size.
- Whether the task changes a default, a local override, a reusable control, or an experiment.

The adopted configuration is `python/design/resources/design-spec.json`, with schemas,
component catalog, and group ownership beside it. Production presenters and controls live in
`python/ui/workspace/` and `python/design/native/`. Preserve these sources of truth; do not
create a parallel specification. The [design tool catalog](../../examples/ui_design/README.md)
explains the runnable entry points.

`make ui-native-design` edits the real workspace; `make ui-design-guide` uses production
presenters with a local specification copy. Browser references and `make ui-feasibility`
are comparison tools. A successful experiment does not silently adopt new production defaults.
OpenGL remains the default renderer.

## Read the relevant specialist guidance

| Change | Read when needed |
|---|---|
| Icon contour, weight, alignment, or semantic master | [Icon guide](../../docs/how-to/ui-icons.md) |
| Nested corners, capsules, fractional scale, native window edges | [Corner guide](../../docs/how-to/ui-corners.md) |
| Drawing, native items, clipping, measurement, caching | [Drawing guide](../../docs/how-to/ui-drawing.md) |
| Shared controls, command boundaries, transactions, capability ownership | [Implementation principles](../../docs/guides/development.md#implementation-principles) |
| Required gates, capture scope, golden baselines | [Testing guide](../../docs/guides/testing.md#change-mapping) |

## Implement at the shared owner

Correct the violated constraint at its owning layer and inspect affected consumers:

- Resolve measurement, painting, hit bounds, focus, reset, and export from the same style owner.
  Keep group-local edits local and restore scope between siblings.
- Keep insets distinct from item gaps. Derive nested corners from actual resolved insets;
  keep application zoom separate from native window points and framebuffer density.
- Choose numeric layouts from stable format budgets, not changing values. Preserve stored
  precision, complete vector/channel rows, and the actual width needs of Boolean controls.
- Use the adopted icon master and canonical grid. Scale all visible dimensions together;
  inspect inner openings and joins at actual sizes as well as enlarged sizes.
- Preserve one input owner through release/cancel and one undo transaction per completed edit.
  Gate physics writes at the command boundary; basic local editor behavior remains available.

Keep domain commands in their existing owner. Shared presentation must not remove actions,
merge Inspector responsibilities, or substitute a lookalike control for the production one.
For performance changes, profile the affected production workload and recovery to idle;
preserve measured geometry, appearance, hit bounds, and transactions.

## Verify and deliver

Choose focused checks from the checklist, then complete the applicable testing-matrix gates.
For visible changes, reuse or extend an existing Make acceptance target. Inspect the images
and exercise affected interactions; a geometry assertion alone does not establish visual
correctness. Match specifications, fonts, UI scale, framebuffer density, and scene state when
comparing captures. Native window-edge changes require inspection of the actual window.

Select states that challenge the changed constraint: fractional scale and narrow widths for
layout, small sizes and interaction colors for icons, or held input, cancellation, and the next
gesture for input fixes. Inspect relevant sibling consumers. Do not broaden verification
after the applicable gates pass unless further changes, failures, or unresolved concerns
justify it. Documentation-only work does not require launching the viewer.

Keep temporary evidence under `output/`. Report the changed contract, verification, useful
capture paths, and any unmet requirement. Follow the existing
[baseline policy](../../docs/guides/testing.md#visual-review-and-baselines); visual review does
not itself create an extra user-approval step.

If the change establishes a reusable rule, update its owning guide and checklist entry.
Describe a generic failure pattern, invariant, and reproducible check using repository or
public evidence. Exclude private conversations, personal identifiers, and local incident reports.
Keep detailed rules in their owning documents instead of copying them into this skill.
