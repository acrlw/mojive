# Agent workflows

This guide describes scene operation and Skill maintenance. Operating a scene does not require
running the repository test suites. Use the interface for the process that contains the scene:

| Task | Entry point | State owner |
|---|---|---|
| Inspect or control an existing viewer | Start with `--rpc-socket` or attach `Viewer.start_rpc`, then use `RpcClient` or `mojive control` | The viewer's Session |
| Build a scene and produce images | `Scene` and `SceneRenderer` | Your Python process |
| Connect an MCP agent to an existing scene | `mojive-mcp --socket PATH` | The existing RPC service or viewer's Session |
| Control a standalone simulation | `mojive rpc-serve` and the same RPC client | The service's Session |
| Display a caller-owned MuJoCo rollout | `launch_passive(model, data)` and `sync()` | The caller's physics loop |
| Display a remote publisher | Snapshot transport | The publisher owns simulation state |

For UI implementation changes, see the [drawing extension guide](ui-drawing.md) for shape
functions, drawing adapters, retained diagnostics, coordinate units, caching, and tests.

Starting a standalone service creates a separate Session. To inspect the scene already visible
to a user, connect to its attached RPC endpoint. See [local RPC control](rpc-control.md) for
startup, methods, capabilities, and timeout behavior.

## Inspect and verify an existing scene

Begin with capabilities and the current scene. CLI commands below use the default user-runtime socket:

```bash
uv run --no-sync mojive control hello --json
uv run --no-sync mojive control get_scene --params '{"include_objects":false}' --json
uv run --no-sync mojive control describe_operations --params '{"query":"edit","include_schemas":false}' --json
uv run --no-sync mojive control describe_operations --params '{"name":"edit_scene"}' --json
```

For repeated operations, use one `RpcClient`; `examples/control_client.py` is a small starting
point. Search summaries by `query` and optional `scope`, then read each needed operation's schema
by name. Reuse its schema and refresh availability after relevant state changes. Locate entities
with filtered, bounded `list_objects` queries; the scene summary does not serialize the hierarchy.
Compute and filter intermediate results in Python before returning them to the agent.
Use returned entity IDs and current document tokens as described in
[document editing](rpc-control.md#edit-a-document). Inspect the resulting state and, for visual
changes, the appropriate scene or presented-viewer capture. Capture metadata identifies the frame
and structure generation; separate calls on a running or externally clocked simulation may observe
different frames. Follow [deadline recovery](rpc-control.md#deadlines-and-recovery) before retrying
a mutation whose outcome is unknown.

## Connect an MCP client

Install the optional bridge with `uv pip install -e '.[mcp]'`. If this checkout already uses
the bgfx native extension, run `make native-editable` afterward to restore its explicit build
registration; see [native editable builds](../guides/development.md#source-layout).
Start the viewer or standalone RPC service that owns the scene, then configure the MCP client to launch:

```text
mojive-mcp --socket /absolute/path/to/control.sock
```

`mojive mcp --socket PATH` and `python -m mojive.control.mcp --socket PATH` are equivalent
stdio entry points. The bridge connects to that service; it does not create another scene or
start a renderer. A typical MCP client configuration is:

```json
{
  "mcpServers": {
    "mojive": {
      "command": "uv",
      "args": ["run", "--no-sync", "--directory", "/absolute/path/to/mojive",
               "mojive-mcp", "--socket", "/absolute/path/to/control.sock"]
    }
  }
}
```

Tools come from the connected service's operation catalog. Start with `hello`, `get_scene`,
and `describe_operations`; use the same discovery and document-token workflow described above.
Results are wrapped as `{"result": ...}`. Failures set MCP `isError` and preserve Mojive's
`error.code`, `message`, and `details`. A failed edit can be inspected and corrected in the same
connection. `--timeout SECONDS` defaults to 30 seconds; an unknown timeout outcome still requires
inspection before another mutation.

Call `capture` or `capture_viewport` with `transport: "base64"` and `encoding: "png"` to receive
an MCP image. Pixels appear once, in the image content block; `result` retains frame metadata
and `image.mime_type`. File captures return their server-side path without implicitly reading it.
The existing [renderer restrictions](rpc-control.md#start-a-service) still apply to the owner.

## Edit an existing model

Use `list_models` to obtain model IDs and `list_joints` to obtain the current joint and node IDs.
`mujoco.get_model_source` returns the current editable MJCF, including changes already applied
through the Inspector. `get_joint_properties`, `list_model_assets`, `list_model_keyframes`, and
`get_model_keyframe` provide readback for their corresponding edits. Source editing requires the
adapter's `mujoco.mjcf` capability; model, asset, topology, and keyframe availability is advertised
individually. An adapter without file-backed models returns an empty model list.

Discover the exact edit schema before writing. Pass the query's full `document` token, read back
the affected values, and capture visual changes. Use `edit_scene` when several supported commands
must commit together; a failed batch restores the prior authored state. Topology edits can rebuild
node and joint IDs, so rediscover IDs and document tokens before the next operation. Use `undo` to
recover a completed edit. See [model authoring](rpc-control.md#edit-model-properties) for a complete
joint edit and readback example.

## Build and render a new scene

Use `Scene` and `SceneRenderer` in one process. Follow the
[programmatic scene tutorial](../tutorials/programmatic-scene.md) and `examples/offscreen_scene.py`
when you need an example. Inspect public scene state and render the requested output; this route
does not require RPC or an existing viewer. Physics-specific `Renderer(model)` remains available
for MuJoCo compatibility; `SceneRenderer` consumes shared contracts.

## Deliver results

Verify the requested state and inspect relevant images yourself. Save captures under `output/`
and include clickable absolute paths to representative visual results in the final response,
with a short explanation of what they show. User review is optional unless explicitly required.
Report any unmet requirement and its concrete blocker after attempting in-scope recovery; finish
independent requirements while a dependency is blocked.

## Executable acceptance example

Use these isolated examples when changing scene-control behavior or the workflow's task decisions.
Run the relevant mode from the repository checkout:

```bash
make agent-control
make agent-viewer ARGS='--output output/agent-viewer'
MOJIVE_RENDERER=bgfx make agent-control ARGS='--output output/agent-control-bgfx'
```

The bgfx command requires the [native runtime and shaders](native-viewer.md). Use it for standalone
RPC capture on macOS, where the graphics worker cannot create an OpenGL context. An attached
OpenGL viewer renders on the UI thread. Select the relevant mode and renderer using the
[verification matrix](../guides/testing.md#change-mapping).

The example creates an isolated authored scene and service, searches operation summaries, reads
the selected schema, and discovers object and camera IDs through bounded queries. It
hides a box, verifies that its selection pixels disappear, then restores it. The plane and sphere
remain visible. It writes RGB images, object-ID arrays, and `report.json` under the output
directory. It then edits position, size, color, and name in one transaction, reads back the edited
and restored properties, verifies Undo/Redo and failure rollback, saves and reopens the document,
rejects stale IDs, and captures the edited scene. `make agent-viewer` also verifies the actual
viewport and window images. Both modes shut down their service on completion.

```python
--8<-- "examples/agent_inspection.py"
```

## Skill discovery

The source skill is `skills/mojive/SKILL.md`. The repository's `.agents/skills/mojive` symlink
points to that directory for repository discovery. Edit the source once; do not maintain a second
copy. Resolve Skill references from the source directory. The skill can be invoked as `$mojive`
once discovered, or selected automatically for matching scene tasks.

For optional use outside this repository, link the source into a user skill directory supported
by your Codex host. The shared user location is `~/.agents/skills`:

```bash
mkdir -p "$HOME/.agents/skills"
ln -s "$PWD/skills/mojive" "$HOME/.agents/skills/mojive"
```

Run that optional installation from the repository root. An existing destination is preserved by
`ln`; inspect an existing installation before deliberately updating it, and avoid duplicate
installations. Some Codex hosts also load `${CODEX_HOME:-$HOME/.codex}/skills`; existing installations
there need not be replaced. Repository operation requires no personal installation.

## Skill maintenance

Maintain instructions or examples when that work is part of the requested task. Ordinary scene
operation can report a demonstrated workflow gap without changing repository instructions.
Keep the Skill focused on task decisions; the operation catalog in `python/control/operations.py` owns
parameter definitions and `control/schema.py` owns shared schemas. Component responsibilities are
documented in [architecture](../concepts/architecture.md#ownership).

For Skill edits, validate frontmatter, naming, UI metadata, the discovery symlink, and referenced
paths. Use the installed skill-creator's `quick_validate.py` with the source Skill directory as its
argument when available; otherwise check the same format constraints directly and report that
validation method. The helper's absence does not block equivalent validation.

Check realistic task decisions as well: existing-viewer edits keep the correct Session, direct
scene creation works without RPC, unknown mutation outcomes are inspected before retry, and an
explicit request for review before edits still produces only a proposal. Format validation alone
does not establish these behaviors. Run the applicable [verification gates](../guides/testing.md#change-mapping)
for changed decisions or executable behavior. Pure wording changes do not require scene rendering.
Extend the Skill only for demonstrated gaps, keeping parameter and protocol details in code and
the relevant reference guide.
