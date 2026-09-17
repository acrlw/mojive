# Mojive

![A jointed scene with joint controls, Inspector and Keyframes](images/readme/hero.png)

Mojive is a backend-neutral 3D viewer, editor, and renderer for robotics and simulation.
It supports MJCF and URDF models, scenes created in Python, custom simulations, remote streams,
and recordings. Rendering uses OpenGL or bgfx, independently of the physics adapter.

## User guide

| Topic | Documentation |
|---|---|
| Install and open a jointed scene | [Getting started](getting-started.md) |
| Edit geometry, joints, models and keyframes | [Editor and MJCF](guides/editor-and-mjcf.md) |
| Build a scene in Python | [Programmatic scenes](tutorials/programmatic-scene.md) |
| Render RGB, depth or segmentation | [MuJoCo rendering](tutorials/mujoco-rendering.md) |
| Embed a viewer beside existing simulation code | [Passive viewing](tutorials/passive-viewing.md) |
| Connect a custom simulation | [Adapter guide](how-to/custom-adapter.md) |
| Publish, record or replay scene data | [Remote viewing](tutorials/remote-viewing.md) |
| Control a local viewer from another process | [RPC control](how-to/rpc-control.md) |
| Add diagnostic shapes or paths | [DebugDraw](how-to/debug-draw.md), [Canvas2D](how-to/canvas2d.md) |

## Reference

- [Command-line reference](reference/cli.md)
- [Configuration](reference/configuration.md)
- [Python API](api/index.md)
- [Example programs](guides/examples.md)

## Development

- [Architecture](concepts/architecture.md)
- [Rendering](concepts/rendering.md)
- [Build and development instructions](guides/development.md)
- [Testing](guides/testing.md)
