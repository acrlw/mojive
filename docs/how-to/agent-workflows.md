# Agent workflows

Use this guide for entry points, runnable examples, and Skill maintenance. Read the section needed
for the task; ordinary scene operation does not require running repository acceptance suites.
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

For repeated operations, use one `RpcClient`; `examples/control_client.py` is a small starting
point. Discover each needed operation by name, reuse its schema, and refresh availability after
relevant state changes. Use returned entity IDs and current document tokens as described in
[document editing](rpc-control.md#edit-a-document). Inspect the resulting state and, for visual
changes, the appropriate scene or presented-viewer capture. Capture metadata identifies the frame
and structure generation; separate calls on a running or externally clocked simulation may observe
different frames. Follow [deadline recovery](rpc-control.md#deadlines-and-recovery) before retrying
a mutation whose outcome is unknown.

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
MOJIVE_RENDERER=wgpu make agent-control ARGS='--output output/agent-control-wgpu'
```

For standalone RPC capture on macOS, use the wgpu command: its graphics worker cannot own a macOS
OpenGL context. The attached-viewer mode renders on the UI thread. These are mode/backend choices,
not three mandatory runs for every edit; select them using the [verification matrix](../guides/testing.md#change-mapping).

The example creates an isolated authored scene and service, discovers its object and camera IDs,
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

The source skill lives in `skills/mojive/SKILL.md`. The repository's `.agents/skills/mojive` symlink
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
Keep the Skill focused on task decisions; the operation catalog in `src/mojive/operations.py` owns
parameter definitions and `control_schema.py` owns shared schemas. Component responsibilities are
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

This instruction workflow was reviewed on 2026-09-05 against the official
[GPT-6 Astra guidance](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices)
and [Skill authoring guidance](https://learn.chatgpt.com/docs/build-skills).

Batch rendering remains deferred. Capture and viewport have explicit independent settings;
transactional authoring applies only to the edits advertised by discovery. The native remote
transport retains its physics-specific operations; these are not all exposed through local RPC.
