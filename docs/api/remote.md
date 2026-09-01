# Remote viewing and recording

## Remote transport

::: mojive.remote

## Snapshot and video streams

::: mojive.recording

`.fvs` begins with a format header and stores the packet sequence consumed by
`RemoteSceneAdapter`. The reader accepts the current format and reports mismatched versions and
truncated packets before replay.

## Local control RPC

::: mojive.control_rpc

`RpcClient` reuses one AF_UNIX connection for sequential requests. `ControlServer` handles
multiple clients concurrently and continues after malformed requests. A timeout closes the client
connection; the next call establishes a new connection.
