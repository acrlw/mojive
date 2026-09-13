# Mojive

![A jointed scene with joint controls, Inspector and Keyframes](images/readme/hero.png)

Mojive is a backend-neutral 3D viewer, editor, and renderer for robotics and simulation.
MJCF, URDF, authored Python scenes, custom adapters, remote publishers and recordings share
scene contracts. Choose OpenGL, WebGPU (`wgpu`) or native bgfx independently from physics.

## Choose a workflow

| Goal | Start here |
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

The [CLI reference](reference/cli.md), [configuration reference](reference/configuration.md),
and [API map](api/index.md) describe the current interfaces. The
[architecture](concepts/architecture.md), [renderer](concepts/rendering.md),
[development](guides/development.md), and [testing](guides/testing.md) guides cover implementation
ownership and verification. Runnable recipes live in the [example catalog](guides/examples.md).
