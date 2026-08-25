# Remote viewing and recording

## Remote transport

::: forge_viewer.remote

## Snapshot and video streams

::: forge_viewer.recording

`.fvs` version 2 begins with a format header and stores the packet sequence consumed by
`RemoteSceneAdapter`. The reader accepts version 1 streams and reports future versions and
truncated packets before replay.

## Local control RPC

::: forge_viewer.control_rpc

`RpcClient` reuses one AF_UNIX connection for sequential requests. `ControlServer` handles
multiple clients concurrently and continues after malformed requests. A timeout closes the client
connection; the next call establishes a new connection.
