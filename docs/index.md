# Mojive

Mojive is a backend-neutral 3D viewer and renderer for robotics, simulation, and tooling.
The OpenGL renderers consume stable scene structure and dynamic frame data from MuJoCo,
programmatic scenes, custom physics engines, remote publishers, and snapshot recordings.

The project provides three primary workflows:

- An interactive editor for model composition, entity authoring, inspection, and simulation.
- A MuJoCo-compatible `Renderer` API for RGB, depth, and segmentation output.
- A scene adapter protocol for custom physics engines, remote viewing, and replay.

Use the [getting started guide](getting-started.md) to run the editor. The
[architecture guide](concepts/architecture.md) explains ownership and data flow. The
[API map](api/index.md) lists the public modules, contracts, and entry points.

Development plans and milestone status live in the repository `plan/` directory.
