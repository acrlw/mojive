# Command-line reference

Run commands as `uv run mojive ...` from a source checkout, or as `mojive ...` after installing
the package. `mojive -v ...` enables verbose logging. Commands with `--json` reserve stdout for the
JSON document and send logs to stderr.

## Render backend and scene adapter

Mojive has two independent backend choices:

- `MOJIVE_RENDERER=opengl|wgpu` selects the renderer. OpenGL is the default;
  `MOJIVE_BACKEND` remains a compatible fallback.
- `--adapter` on model commands selects the scene/physics adapter. The default is `mujoco`;
  `-b/--backend` remain compatible aliases.

For example, this loads a MuJoCo model through the MuJoCo adapter and renders it through wgpu:

```bash
MOJIVE_RENDERER=wgpu uv run mojive view test_scene --adapter mujoco
```

Asset arguments accept a filesystem path or a bundled asset name. The extension is optional for
bundled assets. Use `mojive assets --quick` and `mojive backends` to inspect the current install.

`--enable-render FLAG` is repeatable on `view`, `canvas`, `capture`, `record`, and `keyframes`.
Accepted values are `shadow`, `wireframe`, `reflection`, `additive`, `skybox`, `fog`, `haze`,
`cull_face`, `convexhull`, `texture`, `joint`, `actuator`, `activation`, `camera`, `light`,
`rangefinder`, `constraint`, `static`, `skin`, `flex_face`, `flex_skin`, `flex_vertex`,
`flex_edge`, `contactpoint`, `contactforce`, `contactsplit`, `island`, `autoconnect`, `tendon`,
`transparent`, `com`, `inertia`, `scaled_inertia`, `body_bvh`, `mesh_bvh`, `outline`, `tonemap`,
and `msaa`.

## Interactive commands

### `view`

Open one asset in the standard viewer. The model starts paused unless `--play` is passed.

```text
mojive view ASSET [-b ADAPTER] [--paused | --play] [--no-vsync] [--rpc-socket PATH]
                  [--enable-render FLAG ...]
```

### `editor`

Open an empty workspace or load one MJCF/URDF asset into the editor.

```text
mojive editor [ASSET] [--no-vsync] [--rpc-socket PATH]
```

Use the File menu for `.mojive.json` workspaces, additional models, resource directories, and
portable MJCF export.

### `canvas`

Open a programmatic scene using the standard viewer UI.

```text
mojive canvas [--demo empty|canvas|lighting|text] [--no-vsync]
              [--enable-render FLAG ...]
```

### `toy`

Open the dependency-free reference physics adapter.

```text
mojive toy [--no-vsync]
```

## Rendering and output

### `capture`

Save one PNG. Width and height must be provided together; omitting them uses the normal viewport
size. `--include-ui` includes panels and gizmos. `--camera` renders through a named model camera.

```text
mojive capture ASSET -o FILE [-b ADAPTER] [--width PX --height PX]
                           [--camera NAME] [--include-ui]
                           [--enable-render FLAG ...]
```

### `record`

Record a fixed-size viewport video without vsync.

```text
mojive record ASSET -o FILE [-b ADAPTER] [--frames N] [--fps FPS]
                          [--width PX] [--height PX]
                          [--enable-render FLAG ...]
```

Defaults: 300 frames, 30 fps, 1280×720.

### `keyframes`

Load every model keyframe in order and encode one video frame per keyframe. The command fails when
the model has no keyframes.

```text
mojive keyframes ASSET -o FILE [-b ADAPTER] [--fps FPS]
                              [--width PX] [--height PX]
                              [--camera NAME] [--camera-distance-scale FACTOR]
                              [--enable-render FLAG ...]
```

Defaults: 60 fps and 1920×1080.

## Inspection and validation

### `inspect`

Print node, joint, actuator, keyframe, sensor, mesh, texture, and instance metadata without
opening a window.

```text
mojive inspect ASSET [-b ADAPTER] [--json]
```

### `audit`

Audit MuJoCo visualization and schema coverage. `--strict` exits with status 1 when unsupported
features or runtime validation failures are present.

```text
mojive audit ASSET [-b ADAPTER] [--json] [--strict]
```

### `doctor`

Run a short window-path smoke test.

```text
mojive doctor ASSET [-b ADAPTER] [-n FRAMES] [--json]
```

The default is 90 frames.

### `conformance`

Validate one `SceneAdapter` contract without opening a window.

```text
mojive conformance [ADAPTER] [--asset ASSET] [--json]
```

The default adapter is `toy`.

### `backends`

List known physics/scene adapters, their renderer path, availability, and missing dependencies.

```text
mojive backends [--json]
```

### `assets`

List bundled assets. By default each asset is loaded to report movable-body support; `--quick`
only lists names.

```text
mojive assets [--quick] [-b ADAPTER] [--json]
```

### `probe`

Print OpenGL driver and capability measurements used by the renderer.

```text
mojive probe
```

## Remote viewing and replay

### `serve`

Run simulation in a headless process and publish scene snapshots. `--record-snapshot` also appends
the published structure and frames to a versioned `.fvs` recording.

```text
mojive serve ASSET [-b ADAPTER] [--host HOST] [--port PORT] [--hz HZ]
                   [--paused] [--record-snapshot FILE]
```

Defaults: `127.0.0.1:47650` at 120 Hz.

### `attach`

Open an independent viewer connected to `serve` or `replay`.

```text
mojive attach [--host HOST] [--port PORT] [--title TITLE]
              [--debug-view VIEW] [--no-vsync]
```

Debug views: `shaded`, `albedo`, `normal`, `depth`, `segment`, `idcolor`, `overdraw`, and
`wireframe`.

### `replay`

Republish a recorded `.fvs` stream through the normal attach protocol.

```text
mojive replay FILE [--host HOST] [--port PORT] [--speed FACTOR] [--loop]
```

Replay is read-only. Timing follows recorded frame timestamps.

## Local process control

### `rpc-serve`

Run a local AF_UNIX scene-control service.

```text
mojive rpc-serve ASSET [-b ADAPTER] [--socket PATH]
```

The default socket is `output/mojive.sock`.

On macOS, start the service with `MOJIVE_BACKEND=wgpu` when using `capture`; RPC requests run on
worker threads and the platform OpenGL context path is main-thread-only. Linux can use OpenGL.

### `control`

Send one typed RPC method. `--params` must be a JSON object.

```text
mojive control METHOD [--params JSON] [--socket PATH] [--timeout SECONDS] [--json]
```

`--json` preserves structured errors with exit status 2. Use `hello`, `get_scene`, and
`describe_operations` to discover the running service and its parameter schemas. `view` and
`editor` expose their own Session when started with `--rpc-socket`.

Use the [RPC control guide](../how-to/rpc-control.md) for a persistent Python client and capture
examples.

`--adapter` is the preferred spelling for the scene adapter selector; `--backend` and `-b`
remain compatible aliases. Renderer selection uses `MOJIVE_RENDERER` independently.
