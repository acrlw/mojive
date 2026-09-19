# Remote viewing and replay

Remote viewing separates simulation ownership from rendering. The publisher sends stable scene
structure reliably and replaces queued dynamic frames with the newest state. Each attached viewer
keeps an independent camera and render configuration.

## Publish a live scene

Start the publisher and attach viewers in separate terminals:

```bash
uv run --no-sync python examples/remote_publish.py
uv run --no-sync mojive attach --title effect
uv run --no-sync mojive attach --title debug --debug-view normal
```

```python
--8<-- "examples/remote_publish.py"
```

## Record and replay

Create a deterministic snapshot stream:

```bash
uv run --no-sync python examples/record_replay.py \
  --output output/examples/orbit.fvs --frames 300 --fps 60
uv run --no-sync mojive replay output/examples/orbit.fvs
uv run --no-sync mojive attach
```

```python
--8<-- "examples/record_replay.py"
```

A recording stores the same `RemoteStructure` and `RemoteFrame` packets consumed by a remote
viewer. `replay` republishes those packets; an `attach` process is still required to display them.
Replay timing uses frame timestamps and supports speed control and looping.

Replay is a read-only publisher: it exposes recorded camera and sensor metadata but disables
simulation and authoring capabilities. `--speed` and `--loop` are startup options; interactive
pause, seeking, frame stepping and a recording timeline are not implemented for `.fvs` playback.
The editor's state-take playback is a separate feature that requires native state snapshots.

An attached adapter refreshes its capabilities with each structure update. Stream disconnection
raises `ConnectionError` on the next frame request instead of silently returning the last frame.
Automatic reconnection is not implemented. Remote step, reset and reload failures are reported
to callers; reset failures also produce a failed Session command result. Adapter `timeout`
bounds connection handshakes and command send/receive, including partial replies. A command
timeout closes that command connection, so a late reply cannot be consumed by a later request.
Reattach after checking publisher state; timed-out commands may still complete remotely.

The publisher's `command_timeout` cancels commands that have not started. For a command already
executing, the failure message reports that completion is unknown. Check publisher state before
retrying a mutation. Commands are never automatically retried.

The publisher retains one bootstrap frame for viewers that connect later, but avoids repeatedly
serializing frames while no viewer is attached. Once a viewer is connected, each
`publish_frame()` call still snapshots its arrays with pickle before the latest-only sender can
drop an older packet. High-rate training loops should therefore publish at a deliberate viewing
cadence (commonly 20–30 Hz) instead of publishing every simulation step.

For dense diagnostics, publish plural debug operations such as one `arrows` command containing
NumPy start/end arrays. Thousands of scalar `arrow` commands carry separate dictionaries and
retained IDs and may exceed the viewer's per-frame command budget. Keep scalar commands for items
that need independent lifetime or erasure.

After authoring commands or topology changes, publish the new structure before the next frame.
The example compares `session.structure_generation` after ticking, so Undo/Redo and removal are
visible to attached viewers. Frames should never refer to geometry from an unpublished revision.

Snapshot streams and `.fvs` recordings use pickle and require trusted input. Use trusted peers
and files; this transport does not provide authentication or a safe untrusted-data decoder.

## Training data and local replay

For long training archives, store the model and asset manifest once, then timestamped root poses
and joint positions with shape `(time, world, coordinate)`. MuJoCo `qpos` is suitable when the
matching model and joint ordering are retained. For cross-engine data, describe joint names,
types, coordinate conventions, quaternion order, environment origins, and per-world model
variants explicitly. Include stable world IDs and episode/reset markers; never interpolate
across a reset or structure revision. Other moving objects need their own state tracks.

Pure rigid visual replay needs forward kinematics, without contact solving or integration.
MuJoCo's [`mj_kinematics`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-kinematics)
updates poses after assigning `qpos`; merely assigning state leaves derived geometry stale.
One compiled model can serve matching worlds, with one data workspace per worker. Exact physics
continuation needs the engine's complete
[integration state](https://mujoco.readthedocs.io/en/stable/computation/index.html#state),
not only positions. Velocities, actions, rewards, contacts, and training checkpoints serve
different analysis/resume needs and should be optional tracks.

Body/link poses are an alternative when a producer already has them or the consumer cannot
reconstruct its articulation. Shared body-to-geometry transforms can generate both visual and
collision geometry locally. Explicitly distinguish body frames and center-of-mass frames.
Sending one pose per geom is a compatible but larger fallback.
Choose the monitoring cadence and world subset before GPU readback and serialization.

For large archives, seekable array shards or a producer's existing chunked storage allow bounded
prefetch instead of loading the complete trajectory. NumPy memory-mapped arrays need no new
core dependency. Compressed chunks should match the time/world slices actually read. Mesh LOD
is a separate, optional rendering choice; it does not reduce recording or transfer volume.

The current `.fvs` stream does not implement compact joint-state storage or random seeking;
the optional joint replay and rollout-window paths below implement those preview operations.
`WorldInstances.set_poses()` currently consumes
complete batches of geometry positions and rotation matrices; it does not yet accept compact
body/quaternion packets or skip FK for invisible worlds. Rendering visibility culling occurs
after those poses are supplied.

## Selected-world joint replay

A recording can contain thousands of worlds while the local viewer evaluates only a chosen
subset. The optional `JointReplayAdapter` reads the diagnostic qpos archive directly, without
expanding geometry into a local TCP bridge. It keeps one model/data workspace, memory-maps the
trajectory, and gathers only selected worlds from the current sample before FK and instance
construction. Unselected worlds create no render instances and receive no FK work.

```bash
make g1-replay-generate MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-replay ARGS="--worlds 4 --rpc-socket"
# Choose original IDs from a larger existing archive:
mojive replay-joints output/g1-replay/archive --world-ids 7 42 --rpc-socket
mojive control get_world_selection --json
mojive control set_world_selection --params '{"world_ids":[42,7]}' --json
```

The default preview starts with up to 4 worlds, bounded by the archive and preview limit.
The limit defaults to 64; use `--world-limit` to set a different budget. The Hierarchy panel lets you enter
an initial ID and count or a comma-separated list of IDs. **Apply worlds** commits the change;
typing does no FK or rebuilding. IDs must be unique, in range and within the preview limit.
The displayed subset uses a compact grid while keeping original world labels and segmentation.
Switching subsets preserves the camera; use Frame All if a larger selection needs reframing.
LOD remains optional and off by default (`--enable-render mesh_lod` on bgfx).

Generation defaults to 64 recorded worlds (`G1_REPLAY_WORLDS`), writing a six-second synthetic
`(frame, world, qpos)` float32 NumPy archive, display origins, a compiled model and a manifest
containing its checksum, joint order and display spacing. It uses no downloaded motion or new
runtime dependency. The MJB archive requires the same MuJoCo version; this diagnostic format is
not a portable training storage specification. Playback uses the latest wall-clock sample and
reuses poses between samples. It skips missed deadlines rather than accumulating work; there is
no joint interpolation. Playback starts paused. The Keyframes panel provides play/pause,
previous/next frame, restart, seek-on-release, loop and speed controls; Space toggles playback.
Paused frames and repeated samples do no FK. Corrupt samples stop playback at the last valid
pose with a visible error. Use `--play` to opt into startup playback.

`make world-selection-check G1_REPLAY_ARCHIVE=path/to/archive` opens the real CLI viewer with
2 initial worlds and a limit of 4, switches between original IDs using persistent RPC and a
separate CLI process, and captures Visual/Collision/Both. It verifies the same document and
advancing playback, then closes the viewer. Inspect `output/world-selection/`. This is bounded
functional acceptance, not a 4096-world performance claim.

## Manual rollout windows

Training previews can pull a completed window and replay it locally without staying synchronized
with the training clock. The producer keeps only its latest complete window. A viewer requests
specific world IDs and a sample count, downloads compact float32 qpos on one background worker,
then installs the complete clip on the viewer thread and pauses at its first sample. The old
clip remains usable during download and survives a failed request. No timer requests new clips.

Start with a fixed archive in one terminal and a two-world preview in another:

```bash
uv run --no-sync mojive serve-rollout output/g1-replay/archive
uv run --no-sync mojive replay-joints http://127.0.0.1:47651 \
  --world-ids 0 1 --world-limit 4 --window-frames 64 --rpc-socket
```

Click **Sync latest rollout** in Keyframes to replace the clip. **Sync worlds** in Hierarchy
requests a different subset. Editing fields does no network work. Only one request can be
pending; conflicting world changes fail explicitly. Re-fetching an unchanged revision with
the same IDs/window returns no pose payload and leaves playback alone. New windows with the
same IDs reuse the scene and its geometry resources. A changed subset rebuilds instances and
preserves the camera, surviving selection, and visibility of retained world IDs.

The server limits a request to 64 worlds, 512 samples and 16 MiB of poses by default. The client
also enforces its preview limit, response budget, and download deadline. Model data is fetched
and checked once at startup. The binary wire format uses a bounded JSON header and float32
array, without pickle. This optional HTTP service binds to loopback by default; use an SSH tunnel
for a remote training machine. It provides no authentication or training job control.

To simulate successive completed rollouts, replace the fixed server with:

```bash
uv run --no-sync python examples/rollout_preview.py output/g1-replay/archive \
  --world-ids 0 1 --frames 64 --interval 5
```

```python
--8<-- "examples/rollout_preview.py"
```

Real training code calls `RolloutStore.publish(qpos, start_step=..., world_ids=...)` at an
appropriate rollout boundary. It copies and validates one CPU window synchronously, then swaps
the complete reference; it never waits for a viewer. Select worlds and decimate time samples
**before GPU readback** to keep this producer cost small. Set the manifest's `hz` to the stored
sampling cadence. Preserve original IDs when publishing a preselected batch. Missing IDs fail
explicitly; selecting arbitrary unpublished worlds requires a producer-side subscription/storage
integration that this example does not implement. The viewer does not retain a history of windows.

This is a MuJoCo qpos/MJB preview path requiring matching versions and joint layout.
Body/quaternion packets, variable-rate timestamps, and episode/reset annotations are not
implemented. Avoid interpolation across resets; current replay displays recorded samples exactly.
Full experiment archives and resuming training remain separate from this bounded preview cache.

### MuJoCo and MuJoCo Warp producers

Use the CPU `MjModel` that defines the training worlds to prepare the publisher directly:

```python
from mojive.remote.rollout import RolloutServer, RolloutStore

store = RolloutStore.from_model(model, total_worlds=4096, hz=30, max_worlds=16)
# preview_qpos is an owned CPU float32 array with shape (time, selected_world, nq).
store.publish(preview_qpos, world_ids=(7, 42), start_step=0)
with RolloutServer(store):
    run_training(store)  # At later rollout boundaries, publish another completed window.
```

`from_model` serializes the model once and records its checksum, MuJoCo version and joint order.
No diagnostic archive or G1-specific generator is required. Publish the first complete window
before connecting a viewer; there is deliberately no fabricated initial rollout.

[MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/) initializes device state from a
MuJoCo model and represents parallel worlds in batched arrays. Keep that original CPU model for
the preview publisher. Gather selected qpos samples on the producer's device, synchronize their
completion, then copy only that small window to CPU float32 for `publish`. Root positions must
be world-local: remove any training environment offsets before publication; the viewer applies
its own display grid. Model variants, mocap state and non-joint deformation need additional
tracks and are outside this qpos-only contract.

Mojive does not import or install Warp on a preview client, step training, or read back an entire
device batch. Device collection is owned by the training application. The model/CPU-window
contract is tested with MuJoCo; an actual NVIDIA GPU training job is a separate integration check.

`make rollout-preview-check G1_REPLAY_ARCHIVE=path/to/archive` exercises real HTTP, the CLI viewer,
RPC discovery, CLI seeking, local playback with zero additional requests, unchanged-window sync,
16 new windows, and Visual/Collision/Both with at most two worlds. It records window cadence,
RPC latency and process RSS under `output/rollout-preview/`, then closes both processes. Window
cadence includes repeated poses and does not imply a higher recorded motion sampling rate.

The older `make g1-replay-check ARGS="--seconds 10 --worlds 4"` separately exercises the full
geometry snapshot TCP bridge through `mojive attach`. Its `--geometry-switches` option records
switch RPC latency, first completed PNG and window cadence. It still transfers expanded geometry
poses. It does not use the compact rollout endpoint; `.fvs` and the live snapshot transport
retain their existing format.

## Negotiating write-back

Each `RemoteStructure` carries `protocol_version` and `command_versions`. A mismatched stream
revision fails the connection before its scene is used. Write-back capabilities are the
intersection of advertised adapter features and wire operations supported at the same exact
revision. A missing or unknown command revision withdraws that operation; legacy structures
without a command manifest remain readable but expose no unnegotiated write-back. Supported
capabilities refresh when a new structure arrives. Files, model imports and source editing are
not offered by this snapshot channel. Those operations require a separately authorized control
interface with its own discovery contract.

`get_capabilities` / `describe_operations` belong to the JSON control RPC interface; the binary
snapshot channel has its own manifest. Backend version strings are diagnostic metadata, not
permission to send a newer command. Unknown operations are rejected rather than silently mapped
to a similarly named engine function. The existing trusted-peer restriction still applies.

For a reproducible two-process transport stress test, run:

```bash
make g1-worlds-transport MENAGERIE_ROOT=/path/to/mujoco_menagerie
```

This downloads a checksum-pinned Unitree CSV, replays independent random phases, and measures
1024/2048/4096 worlds over localhost TCP. The publisher targets 120 Hz and the consumer samples
at 30 Hz. Reports include full pose bytes, snapshot age, publication time and coalesced frames.
These are transport measurements without rendering, not WAN latency or reinforcement-learning
throughput. The test owns its server and never connects to an existing viewer session.

To include the consumer's GPU work, run the separate end-to-end target:

```bash
make g1-worlds-monitor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-worlds-monitor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie \
  ARGS="--renderer opengl --output output/g1-worlds-monitor-opengl"
```

Both use original meshes unless an explicit LOD is requested. This path reports receive-time
snapshot age and completed-image age separately, including renderer update and RGB readback.
The publisher and renderer execute in separate processes; the consumer takes the latest frame
instead of accumulating a playback queue. These local monotonic-clock measurements do not
measure display scanout or unsynchronized clocks on separate hosts. The parent stops only its
own publisher after sampling finishes.
