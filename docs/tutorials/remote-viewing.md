# Remote viewing and replay

Remote viewing separates simulation ownership from rendering. The publisher sends stable scene
structure reliably and replaces queued dynamic frames with the newest state. Each attached viewer
keeps an independent camera and render configuration.

## Publish a live scene

Start the publisher and attach viewers in separate terminals:

```bash
.venv/bin/python examples/remote_publish.py
.venv/bin/forge-viewer attach --title effect
.venv/bin/forge-viewer attach --title debug --debug-view normal
```

```python
--8<-- "examples/remote_publish.py"
```

## Record and replay

Create a deterministic snapshot stream:

```bash
.venv/bin/python examples/record_replay.py \
  --output output/examples/orbit.fvs --frames 300 --fps 60
.venv/bin/forge-viewer replay output/examples/orbit.fvs
```

```python
--8<-- "examples/record_replay.py"
```

A recording stores the same `RemoteStructure` and `RemoteFrame` packets consumed by a remote
viewer. Replay timing uses frame timestamps and supports speed control and looping.
