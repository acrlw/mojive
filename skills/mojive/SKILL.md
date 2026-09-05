---
name: mojive
description: Inspect, edit, control, and capture 3D scenes with Mojive. Use for operating an existing Mojive viewer, authoring scenes, or controlling a simulation through its public APIs. For Mojive implementation changes, follow the repository development guidance.
---

# Mojive

Choose the state owner before choosing an API. Read the repository's
[agent workflows](../../docs/how-to/agent-workflows.md) for entry points and executable examples.

- Existing viewer: connect to its RPC socket. `hello` must report `viewer_attached: true`.
  A new standalone service owns a separate scene.
- New scene and images: use `Scene` and `SceneRenderer` in one Python process.
- Standalone simulation: use `mojive rpc-serve` and the same control client.

For RPC, start with `hello` and `get_scene`. Use `describe_operations` filtered by `name`
for the next operation's live input/output schema, defaults, scope, and availability reason.
Keep one `RpcClient` connection for a multi-step task. Shell calls use `mojive control METHOD
--params 'JSON_OBJECT' --json`; see the [RPC guide](../../docs/how-to/rpc-control.md).

Locate entities by metadata, then use returned IDs. `object_id` selects an entity and identifies
pixels; `node_id` addresses its hierarchy entry; `camera_id` addresses a camera. Read the named
ID field in creation results. `inspect_object.geometries` returns associated geometry edit IDs,
size, RGBA, and material values. Use those `node_id` values for geometry edits; a selected parent
can have a different ID. Read `dimensions` for conventional primitive measurements; box `size`
values are half extents. Use `hierarchy_visible` to account for hidden parents. Pass the latest
`document` as `expected_document` when editing retained IDs. Refresh after `stale_document` or
document replacement. A timeout can leave a mutation's outcome unknown: inspect before retrying.

Use `edit_scene` for related authored edits that must succeed together and produce one undo
record. It accepts operations marked `transactional` by discovery. Pause simulation when the
operation requires it. Use operation discovery to check authoring support; `source_editable`
refers to an adapter model-source element. Keep scene authoring in Session commands and physics
writes in adapters.

Read back changed properties through `inspect_object`; compare floating-point values with a
tolerance. For visual edits, also inspect an image: `capture` renders the composed scene with
independent capture settings; `capture_viewport` captures the presented
viewport or window. Use `set_capture_camera` or `set_viewport_camera` for the intended scope.
Use `object_id` captures to count selection pixels; segmentation IDs have different semantics.
After document replacement, rediscover scene cameras and select the intended capture camera again.
Keep artifacts under `output/` and report their paths and any unfinished requirement.

When changing this workflow, run `make agent-control` and the relevant operation tests. Maintain
parameter definitions in `src/mojive/operations.py` and shared schemas in `control_schema.py`;
keep this skill focused on task decisions. Update it when demonstrated usage exposes a gap.
