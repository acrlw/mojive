# Testing

The suite is split by runtime and dependency boundary. Use the smallest layer that covers the
change while iterating, then run the required acceptance targets before handoff.

| Layer | Marker or target | Coverage |
|---|---|---|
| Fast | `make test-fast` | pure CPU behavior and module contracts |
| Integration | `make test-integration` | files, serialization, protocols, processes, composition |
| Physics | `make test-physics` | model compilation and live physics worlds |
| OpenGL GPU | `make gpu` | real OpenGL contexts and rendered output |
| WebGPU | `make gpu-wgpu` | Metal, Vulkan, or DX12 backend behavior |
| Golden | `make golden` | reviewed image baselines |
| Full | `make test-all` | CPU, physics, OpenGL, and WebGPU layers |

## Change mapping

| Change | Iteration target | Acceptance target |
|---|---|---|
| Shared types or commands | `make test-fast` | `make check` |
| File format or remote protocol | `make test-integration` | `make check` |
| MuJoCo adapter or MJCF authoring | focused physics test | `make test-physics` and `make mujoco-audit` |
| Render pass or shader | one GPU test file | `make gpu` and `make gpu-wgpu` |
| Visual interaction | focused GPU test | relevant Make gallery and `make reverse` |
| Settings layout | focused UI GPU test | `make settings` |

Markers may be combined. A file-format test that compiles MuJoCo uses both `integration` and
`physics`; it runs in the physics layer.

## Documentation

Build the user guide and generated API reference with:

```bash
make docs
```

The strict build rejects broken links, unresolved API modules, and documentation warnings.
