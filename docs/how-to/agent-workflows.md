# Agent workflows

Choose the entry point by the state you need to operate on:

| Task | Entry point | State owner |
|---|---|---|
| Inspect or control an existing viewer | Start with `--rpc-socket` or attach `Viewer.start_rpc`, then use `RpcClient` or `mojive control` | The viewer's Session |
| Build a scene and produce images | `Scene` and `SceneRenderer` | Your Python process |
| Control a standalone simulation | `mojive rpc-serve` and the same RPC client | The service's Session |
| Display a remote publisher | Snapshot transport | The publisher owns simulation state |

Starting a standalone service creates a separate Session. To inspect the scene already visible
to a user, connect to its attached RPC endpoint. See [local RPC control](rpc-control.md) for
startup, methods, capabilities, and timeout behavior.

## Inspect and verify an existing scene

Begin with capabilities and the current scene. CLI commands below use `output/mojive.sock`:

```bash
uv run mojive control hello --json
uv run mojive control get_scene --json
uv run mojive control describe_operations --params '{"name":"edit_scene"}' --json
```

Use returned IDs for subsequent operations. `node_id` identifies a hierarchy entry and is used
by `set_visible`; `object_id` identifies a selection and pixels in an `object_id` capture;
`body_index` is an adapter-specific physics lookup. Names help locate candidates but may repeat.
Creation results expose named IDs. Pass the returned `document` as `expected_document` when
editing retained IDs. Refresh scene metadata after `stale_document` or replacing a document.
Use `describe_operations` for live parameter schemas and availability; filter by name to keep
the response small. Keep a single `RpcClient` connection for a multi-step workflow.

Read the mutation's result and then inspect the resulting state. `inspect_object.geometries`
returns geometry edit IDs and current size, RGBA, and material values. Use these to verify changes
without reconstructing the object from source code. `hierarchy_visible` includes hidden ancestors.
For visual changes, also capture RGB or object IDs; a successful command alone does not prove the requested image.
Capture results identify the frame and structure generation they rendered. Separate calls on a
running or externally clocked simulation may observe different frames.

Use `examples/control_client.py` as a small Python starting point.
For direct authored rendering, use the [programmatic scene tutorial](../tutorials/programmatic-scene.md)
and `examples/offscreen_scene.py`. Physics-specific `Renderer(model)` remains available for
MuJoCo compatibility; `SceneRenderer` consumes shared contracts.

## Executable acceptance example

Run from the repository checkout:

```bash
make agent-control
make agent-viewer ARGS='--output output/agent-viewer'
MOJIVE_RENDERER=wgpu make agent-control ARGS='--output output/agent-control-wgpu'
```

The example creates an isolated authored scene and service, discovers its object and camera IDs,
hides a box, verifies that its selection pixels disappear, then restores it. The plane and sphere
remain visible. It writes RGB images, object-ID arrays, and `report.json` under the output
directory. It then edits position, size, color, and name in one transaction, reads back the edited
and restored properties, verifies Undo/Redo and failure rollback, saves and reopens the document,
rejects stale IDs, and captures the edited
scene. `make agent-viewer` also verifies the actual viewport and window images. Both modes shut
down their service on completion.

```python
--8<-- "examples/agent_inspection.py"
```

## Maintained skill and responsibilities

The source skill lives in `skills/mojive/SKILL.md`. It routes tasks to the correct state owner,
operation discovery, document preconditions, and result verification. The operation catalog owns
parameter schemas, availability, and command construction. `ControlApplication` coordinates
Session commands and capture. `control_rpc` owns envelopes, sockets, deadlines, and UI-thread
queuing. Session owns mutations, transactions, and history; adapters own physics write-back.
Native remote scene-edit messages reuse the catalog's validation and command construction.

To make this checkout's skill available in your personal Codex installation, link it once:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$PWD/skills/mojive" "${CODEX_HOME:-$HOME/.codex}/skills/mojive"
```

Run from the repository root. An existing destination is preserved by `ln`; update an existing
installation deliberately. The symlink keeps the skill and its repository references current.
The skill can be invoked as `$mojive` in sessions where it has been discovered.

Validate skill edits with the skill creator's `quick_validate.py`, then run the acceptance
example. Extend the skill only when actual use exposes a missing task decision. Parameter and
protocol changes belong in code and tests, with matching documentation updates.

Batch rendering remains deferred. Capture and viewport have explicit independent settings;
transactional authoring applies only to the edits advertised by discovery. The native remote
transport retains its physics-specific operations; these are not all exposed through local RPC.
