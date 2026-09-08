# Remote viewing and replay

Remote viewing separates simulation ownership from rendering. The publisher sends stable scene
structure reliably and replaces queued dynamic frames with the newest state. Each attached viewer
keeps an independent camera and render configuration.

## Publish a live scene

Start the publisher and attach viewers in separate terminals:

```bash
uv run python examples/remote_publish.py
uv run mojive attach --title effect
uv run mojive attach --title debug --debug-view normal
```

```python
--8<-- "examples/remote_publish.py"
```

## Record and replay

Create a deterministic snapshot stream:

```bash
uv run python examples/record_replay.py \
  --output output/examples/orbit.fvs --frames 300 --fps 60
uv run mojive replay output/examples/orbit.fvs
uv run mojive attach
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
