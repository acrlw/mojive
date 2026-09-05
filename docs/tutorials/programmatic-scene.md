# Programmatic scene

Use `Scene` when geometry, cameras, lights, and materials originate in Python. The resulting scene
uses the same renderer, hierarchy, inspector, selection, and capture paths as physics-backed
models.

Objects and lights returned by a Scene remain attached to that Scene through editor Undo/Redo.
An undone or removed entity's handle raises `KeyError` until that entity is restored. IDs are not
reused for later entities, and handles from different scenes compare unequal. Opening a different
document creates a different owner; obtain handles from the new scene.

Mesh and texture registration copy caller arrays into read-only scene storage. Mutating the original
arrays afterward has no effect. Publish a replacement with `scene.add_texture(texture)` or
`scene.replace_mesh(object.mesh_key, mesh_data)`; the latter changes all objects sharing that mesh.
Undo snapshots share unchanged resource storage while keeping authored state independent.

Use an exception-safe Session transaction for a group of commands:

```python
from mojive import commands

with viewer.session.edit("Rename and move"):
    viewer.session.submit(commands.RenameSceneEntity(box.object_id, "workpiece"))
    node = viewer.session.node_by_object_id(box.object_id)
    viewer.session.submit(commands.SetPose(node.node_id, position, rotation))
```

The context creates one Undo record. An exception or a failed scene-edit command restores the
state from before the transaction. Low-level callers can use `BeginEditTransaction`,
`EndEditTransaction`, and `CancelEditTransaction`. Direct Scene or adapter mutations bypass
Session command tracking.

Session retains at most 100 document edits and an estimated 256 MiB of Python/NumPy history storage.
Configure these with `history_record_limit` and `history_byte_limit` when constructing Session;
`session.history_bytes` reports the retained estimate across both Undo and Redo. Shared resources
count once. If a single edit exceeds the byte budget, the edit is applied, history is cleared, and
the command result explains why Undo is unavailable. Native adapter allocations are outside the
Python byte estimate and remain bounded by the record count.

## Run the example

```bash
uv run python examples/programmatic_scene.py
```

The scene owns stable geometry and entity metadata. `build_scene()` wraps it with a static scene
adapter and composes the standard editor window.

## Source

```python
--8<-- "examples/programmatic_scene.py"
```

## Save and load

Programmatic scenes use the Mojive JSON scene format. The destination directory must already be
writable; the example below uses the repository's ignored `output/` tree:

```python
scene.save("output/examples/workcell.mojive.json")
restored = Scene.load("output/examples/workcell.mojive.json")
```

Use `WorkspaceAdapter` when one document combines authored entities with MJCF or URDF models.

`session.bounds()` returns `Bounds(minimum, maximum)` in world coordinates. Node-specific
`node_world_bounds()` and `node_local_bounds()` return `CenteredBounds(center, half_extent)` in
world and body coordinates, respectively. Both types expose named minimum, maximum, center, and
half-extent accessors; existing two-value tuple unpacking retains its original convention.

An input callback runs once per interactive frame before built-in shortcut polling.
`InputClaim(keyboard=True)` reserves editor commands as well as viewport tools and panel
shortcuts. For a chord, claiming either its letter or a held modifier prevents dispatch.
Explicit menu clicks remain available. Ctrl can be assigned to a viewport action and triggers
on its own key press; Ctrl/Cmd chords reserve their letter keys from viewport actions.
The framing action is named **Frame All**, and menu hints follow its current binding.
