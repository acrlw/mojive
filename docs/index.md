# Mojive

Mojive is a backend-neutral 3D viewer, editor, and renderer for robotics, simulation, and tooling.
The OpenGL and wgpu renderers consume stable scene structure and dynamic frame data from MuJoCo,
programmatic scenes, custom physics engines, remote publishers, and snapshot recordings.

The project provides three primary workflows:

- An interactive editor for model composition, entity authoring, inspection, and simulation.
- A MuJoCo-compatible `Renderer` API for RGB, depth, and segmentation output.
- A scene adapter protocol for custom physics engines, remote viewing, and replay.

Use the [getting started guide](getting-started.md) to run the editor. The
[command-line reference](reference/cli.md) distinguishes render backends from scene adapters, and
the [configuration reference](reference/configuration.md) lists runtime overrides and persistent
paths. The [architecture guide](concepts/architecture.md) explains ownership and data flow, while
the [batch rendering design](BATCH_RENDERING.md) evaluates the path from today's single-view API to
vectorized RL camera observations. The [API map](api/index.md) lists the public modules, contracts,
and entry points.

Current development priorities live in `plan/STATUS.md`. Completed implementation plans in
`plan/` are historical engineering records rather than user documentation.
