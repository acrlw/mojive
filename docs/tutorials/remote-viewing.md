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
