---
name: mojive
description: Inspect, author, control, and capture Mojive scenes through public APIs. Use for viewer operation, scene creation, or simulation control; follow repository development guidance for implementation changes.
---

# Mojive

Choose the state owner before choosing an API:

- Existing viewer: connect to its RPC socket. `hello` must report `viewer_attached: true`.
  A new standalone service owns a separate scene.
- New scene and images: use `Scene` and `SceneRenderer` in one Python process. Verify through
  public scene APIs and rendered output; RPC instructions apply only when using RPC.
- Standalone simulation: use `mojive rpc-serve` and the same control client.

Read [agent workflows](../../docs/how-to/agent-workflows.md) when you need startup or executable
examples, the [scene tutorial](../../docs/tutorials/programmatic-scene.md) for direct authoring,
and the [RPC guide](../../docs/how-to/rpc-control.md) for protocol details.

## RPC tasks

Start a connection with `hello` and `get_scene`. Use `describe_operations` filtered by `name`
for each needed operation's live schema and availability reason. Reuse schemas within the
connection and refresh availability after relevant state changes. Keep one `RpcClient` for a
multi-step task; individual shell calls use `mojive control METHOD --params 'JSON_OBJECT' --json`.

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
requested operation requires it, subject to user restrictions. Use discovery to check authoring
support; `source_editable` refers to an adapter model-source element. Keep scene authoring in
Session commands and physics writes in adapters. Viewer settings stay instance-local; use
`persist: true` only when the user intends to change persistent desktop preferences.

Read back object properties through `inspect_object`, and other state through the corresponding
query; compare floating-point values with a tolerance. For visual edits, also inspect an image:
`capture` renders the composed scene with independent capture settings; `capture_viewport`
captures the presented viewport or window. Use `set_capture_camera` or `set_viewport_camera`
for the intended scope.
Use `object_id` captures to count selection pixels; segmentation IDs have different semantics.
After document replacement, rediscover scene cameras and select the intended capture camera again.

## Completion

Resolve recoverable prerequisites within the requested scope and continue. If an operation remains
unavailable, report the concrete reason and complete independent requirements. For visual tasks,
inspect the resulting image yourself. Keep artifacts under `output/`; link representative visual
results with absolute paths and explain what they show. Report verification and any unmet
requirement. Run repository maintenance checks only when maintaining the workflow itself.
