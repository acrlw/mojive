# forge-viewer

forge-viewer is an interactive 3D viewer for robotics, simulation, and rendering tools. Its
Forge renderer consumes a backend-neutral scene protocol, so MuJoCo, custom physics engines,
static canvases, remote processes, and recorded snapshots share one rendering and UI stack.

Core workflows include scene inspection, object selection, transform gizmos, physical
perturbation, debug drawing, camera and light editing, render diagnostics, capture, and video.

The project is under active development. The scene adapter contracts are stable enough for
experimentation, while higher-level editor and renderer compatibility APIs continue to evolve.

## Requirements

- Python 3.11 or newer
- An OpenGL 4.1 core-profile driver
- A desktop session for interactive windows
- MuJoCo 3.1 or newer for MJCF, URDF, simulation, and physics tools

The current acceptance platform is macOS on Apple Silicon. Linux requires a working OpenGL 4.1
desktop driver. `uv` is the recommended environment and dependency manager.

## Quick start

Install `uv` on macOS, then clone and run the default MuJoCo scene:

```bash
brew install uv
git clone https://github.com/acrlw/forge-viewer.git
cd forge-viewer
uv sync --python 3.11 --extra mujoco
uv run forge-viewer view test_scene
```

The first sync creates `.venv`, installs forge-viewer in editable mode, and resolves the versions
recorded in `uv.lock`. The `mujoco` extra installs the physics backend. Forge scenes and the toy
physics adapter can use the core installation without that extra.

From a source checkout, `make viewer` opens the same default scene.

Open a local MJCF or URDF model directly:

```bash
uv run forge-viewer view path/to/model.xml --paused
uv run forge-viewer view path/to/model.urdf --paused
```

Start with an empty viewport and load a model from the File menu or by dropping it into the window:

```bash
make empty
```

### pip installation

`pyproject.toml` is the canonical dependency specification. A standard virtual environment works
without `uv`:

```bash
git clone https://github.com/acrlw/forge-viewer.git
cd forge-viewer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[mujoco]"
forge-viewer view test_scene
```

### Development installation

```bash
make setup
make check
make gpu
```

`make setup` installs the `dev` and `mujoco` extras. Generated captures, recordings, downloaded
robot models, caches, and local UI state stay under ignored paths and are not part of the source
distribution.

Useful discovery commands:

```bash
forge-viewer assets
forge-viewer backends
forge-viewer probe
```

## Command line

```text
forge-viewer view <asset> [--paused] [-b BACKEND]
forge-viewer canvas [--demo canvas|lighting|text]
forge-viewer toy
forge-viewer conformance [BACKEND] [--asset ASSET]
forge-viewer serve <asset> [--host HOST] [--port PORT]
forge-viewer attach [--host HOST] [--port PORT]
forge-viewer replay <snapshot> [--loop] [--speed FACTOR]
forge-viewer doctor <asset>
forge-viewer inspect <asset> [--json]
forge-viewer capture <asset> -o output/image.png
forge-viewer record <asset> -o output/video.mp4 [--frames N] [--fps FPS]
forge-viewer audit <asset> [--json] [--strict]
```

JSON commands reserve stdout for the JSON document and send logs to stderr.

The main menu and window file drop open MJCF, XML, and URDF models at runtime. On macOS the
viewport highlights when a model enters the window. `File > Reload Model` recompiles the current
file. Loading replaces the session structure, clears selection and interaction state, rebuilds GPU
scene resources, and frames the new model.

## Visual acceptance

Every user-facing feature has a reproducible Make target.

| Target | Purpose |
|---|---|
| `make viewer` | Open the default MuJoCo scene |
| `make empty` | Open an empty viewer and load MJCF or URDF from the File menu |
| `make model-loading` | Capture empty, MJCF, and URDF runtime-loading references |
| `make outline` | Selection, x-ray outline, and outline antialiasing |
| `make gizmo` | Native 2D and 3D transform gizmos |
| `make gizmo-gallery` | Enlarged position, rotation, and snap reference images |
| `make perturb` | MuJoCo translation and rotation perturbation |
| `make text-overlay` | GPU world-space text |
| `make lighting` | Editable lights and Environment controls for ambient light, fog, haze, and headlight |
| `make image-light` | MuJoCo cube-map environment light with editable intensity and texture |
| `make many-lights` | 16-light and 24-light MuJoCo reference images |
| `make scene-icons` | Camera and light scene icons |
| `make reflect` | Planar reflections |
| `make additive` | Standard and additive transparency reference images |
| `make cameras` | Free, named, and orthographic cameras |
| `make mujoco-visuals` | Height fields, sites, tendons, and contacts |
| `make mujoco-debug` | Joint, center-of-mass, and inertia overlays |
| `make mujoco-actuators` | Actuator and activation overlays |
| `make mujoco-islands` | Constraint-island material and color comparison |
| `make mujoco-bvh` | Body, mesh, and flex bounding-volume hierarchies |
| `make mujoco-convex-hull` | Original collision meshes and MuJoCo compiled convex hulls |
| `make mujoco-rangefinder` | Rangefinder rays, hits, and normals |
| `make mujoco-constraints` | Equality constraint markers |
| `make mujoco-editing` | Mocap pose and equality controls |
| `make deformables` | Flex and skin dynamic meshes |
| `make robot` | Download and open a MuJoCo Menagerie robot |
| `make canvas` | Standalone scene authoring with editable transforms, materials, and Forge entities |
| `make scene-io` | Save, reload, and capture a `.forge.json` scene |
| `make toy-physics` | Minimal physics backend independent of MuJoCo |
| `make live-view` | One publisher and two independent remote viewers |
| `make capture` | Write a PNG under `output/` |
| `make record` | Stream an MP4 under `output/` |

Examples:

```bash
make viewer SCENE=humanoid ARGS="--paused"
make empty
make robot ROBOT=unitree_g1
make capture SCENE=deformables SCREENSHOT=output/deformables.png
make record SCENE=humanoid OUTPUT=output/humanoid.mp4 ARGS="--frames 240"
```

## Verification

| Target | Coverage |
|---|---|
| `make check` | Lint, formatting, and CPU tests |
| `make gpu` | Isolated real-OpenGL test files |
| `make golden` | Golden-image comparison |
| `make reverse` | Mutation checks for regression assertions |
| `make doctor` | Window-path smoke test |
| `make mujoco-audit` | MuJoCo feature coverage for one model |
| `make adapter-conformance` | Backend-neutral scene contract |
| `make bench` | Median CPU and GPU pass timing |
| `make parity` | Forge and MuJoCo reference renders |
| `make calibrate` | Reference-lighting calibration |
| `make probe` | OpenGL capability report |

`make golden-accept` updates golden images for reviewed visual changes.

## Transform tools

Select a movable object and pause simulation to edit its pose.

- `G`: position gizmo
- `R`: rotation gizmo
- `T`: body or world frame
- `F9`: 2D or 3D gizmo setting
- `Shift`: snap while dragging
- `X`, `Y`, `Z`: constrain the active transform to one axis

Position snapping defaults to 0.5 m. Its axis ruler marks every snap interval and enlarges
whole-meter ticks. Rotation snapping defaults to 5 degrees. Its outer ring marks 5, 15, 45,
and 90 degree intervals. Both values are editable in Settings.

MuJoCo perturbation uses `Ctrl` + left drag for translation and `Ctrl` + right drag for
rotation.

## Backend-neutral scenes

`SceneSource` defines stable structure, meshes, materials, entities, and render metadata.
`SceneFrame` carries poses, dynamic meshes, lights, sensors, and debug commands. A backend
implements `scene_source()`, `frame()`, and `step()` through `SceneAdapterBase`.

```python
from forge_viewer import Light, Scene, build_scene

scene = Scene()
ball = scene.sphere(name="ball", position=(0.0, 0.0, 0.5))
key = scene.add_light("key", Light())
viewer = build_scene(scene)

for frame in range(300):
    ball.set_pose((frame * 0.01, 0.0, 0.5))
    viewer.sync()

key.remove()
viewer.release()
```

Forge scene entities own cameras and lights. Physics adapters provide dynamic transforms and
write-back capabilities. Hierarchy and Inspector present the same editing workflow across
programmatic scenes, MuJoCo, remote viewers, and snapshot replay.

`ToyPhysicsAdapter` exercises the public adapter contract with gravity, ground collision,
simulation controls, and pose editing. `check_adapter()` produces the same conformance report
used by MuJoCo.

## Remote viewing and replay

`make live-view` starts one headless simulation publisher and two viewer processes. Each viewer owns
its window, camera, layout, and rendering context. Structure revisions are delivered reliably;
frame transport keeps the latest state. Camera, light, environment, material, geometry color,
physics, perturbation, and scene-authoring commands use a separate request and response channel.

```bash
make serve LIVE_SCENE=deformables
make attach ARGS="--title effect"
make attach ARGS="--title normals --debug-view normal"
make remote-authoring
```

Runtime creation returns the stable object, light, or camera ID:

```python
from forge_viewer import MeshShape, commands as cmd

result = viewer.session.submit(
    cmd.AddSceneObject(MeshShape.BOX, "tool marker", position=(1.0, 0.0, 0.5))
)
viewer.session.submit(cmd.RemoveSceneObject(result.entity_id))
```

Snapshot recording stores structure, frames, and debug commands in `.fvs` files:

```bash
make snapshot-record LIVE_SCENE=gizmo SNAPSHOT=output/bug.fvs
make snapshot-replay SNAPSHOT=output/bug.fvs
```

## Architecture

```text
src/forge_viewer/
├── types.py, math3d.py, commands.py   shared contracts
├── session.py                         application state and command routing
├── scene.py                           programmatic Forge scenes
├── adapters/                          MuJoCo, static, toy, and remote sources
├── render/
│   ├── scene.py                       renderer scene representation
│   ├── backend.py                     rendering backend protocol
│   └── forge/                         OpenGL renderer, passes, and shaders
└── ui/                                window, panels, gestures, and gizmos
```

Renderer code depends on shared scene contracts. Physics integration lives in adapters. UI code
depends on protocols and session state. `tests/test_layering.py` enforces these boundaries.

Forge uses row-major matrices in Python and transposes at the OpenGL upload boundary. World
coordinates use Z-up.

## Documentation

- [Renderer design](docs/RENDERER.md)
- [Platform measurements](docs/PLATFORM.md)
- [Implementation decisions](docs/DECISIONS.md)
- [Roadmap](docs/ROADMAP.md)

## License

forge-viewer is available under the [MIT License](LICENSE).
