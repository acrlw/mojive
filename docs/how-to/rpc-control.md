# Local RPC control

The local control service exposes typed simulation, state, selection, camera, and capture
operations over an AF_UNIX socket. `RpcClient` keeps one connection open across sequential calls
and reconnects after a transport failure.

## Start a service

```bash
.venv/bin/forge-viewer rpc-serve assets/test_scene.xml \
  --socket output/forge-viewer.sock
```

## Run the Python client

```bash
.venv/bin/python examples/control_client.py \
  --socket output/forge-viewer.sock \
  --steps 120 \
  --capture output/examples/rpc.png
```

```python
--8<-- "examples/control_client.py"
```

The command-line client provides the same protocol for scripts and shell automation:

```bash
.venv/bin/forge-viewer control get_state --socket output/forge-viewer.sock
```
