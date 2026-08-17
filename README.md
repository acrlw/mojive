# forge-viewer

forge-viewer is an interactive 3D viewer for robotics, simulation, and rendering tools. Its
Forge renderer consumes a backend-neutral scene protocol, so MuJoCo, custom physics engines,
static canvases, remote processes, and recorded snapshots share one rendering and UI stack.

Core workflows include scene inspection, object selection, transform gizmos, physical
perturbation, debug drawing, camera and light editing, render diagnostics, capture, and video.

## Setup

```bash
make setup
make viewer
```

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

## Visual acceptance

Every user-facing feature has a reproducible Make target.

| Target | Purpose |
|---|---|
| `make viewer` | Open the default MuJoCo scene |
| `make outline` | Selection, x-ray outline, and outline antialiasing |
| `make gizmo` | Native 2D and 3D transform gizmos |
| `make gizmo-gallery` | Enlarged position, rotation, and snap reference images |
| `make perturb` | MuJoCo translation and rotation perturbation |
| `make text-overlay` | GPU world-space text |
| `make lighting` | Editable point, spot, and area lights with fog and haze |
| `make scene-icons` | Camera and light scene icons |
| `make reflect` | Planar reflections |
| `make cameras` | Free, named, and orthographic cameras |
| `make mujoco-visuals` | Height fields, sites, tendons, and contacts |
| `make mujoco-debug` | Joint, center-of-mass, and inertia overlays |
| `make mujoco-actuators` | Actuator and activation overlays |
| `make mujoco-islands` | Constraint-island material and color comparison |
| `make mujoco-bvh` | Body, mesh, and flex bounding-volume hierarchies |
| `make mujoco-rangefinder` | Rangefinder rays, hits, and normals |
| `make mujoco-constraints` | Equality constraint markers |
| `make mujoco-editing` | Mocap pose and equality controls |
| `make deformables` | Flex and skin dynamic meshes |
| `make musculoskeletal` | Full musculoskeletal model with tendons and keyframes |
| `make robot` | Download and open a MuJoCo Menagerie robot |
| `make canvas` | Standalone scene authoring with Forge entities |
| `make toy-physics` | Minimal physics backend independent of MuJoCo |
| `make pvd` | One physics publisher and two independent viewers |
| `make capture` | Write a PNG under `output/` |
| `make record` | Stream an MP4 under `output/` |

Examples:

```bash
make viewer SCENE=humanoid ARGS="--paused"
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
from forge_viewer import Scene, build_scene

scene = Scene()
ball = scene.sphere(name="ball", position=(0.0, 0.0, 0.5))
viewer = build_scene(scene)

for frame in range(300):
    ball.set_pose((frame * 0.01, 0.0, 0.5))
    viewer.sync()

viewer.release()
```

Forge scene entities own cameras and lights. Physics adapters provide dynamic transforms and
write-back capabilities. Hierarchy and Inspector present the same editing workflow across
programmatic scenes, MuJoCo, remote viewers, and snapshot replay.

`ToyPhysicsAdapter` exercises the public adapter contract with gravity, ground collision,
simulation controls, and pose editing. `check_adapter()` produces the same conformance report
used by MuJoCo.

## Remote viewing and replay

`make pvd` starts one headless simulation publisher and two viewer processes. Each viewer owns
its window, camera, layout, and rendering context. Structure revisions are delivered reliably;
frame transport keeps the latest state. Commands use a separate request and response channel.

```bash
make serve PVD_SCENE=deformables
make attach ARGS="--title effect"
make attach ARGS="--title normals --debug-view normal"
```

Snapshot recording stores structure, frames, and debug commands in `.fvs` files:

```bash
make snapshot-record PVD_SCENE=gizmo SNAPSHOT=output/bug.fvs
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
- Original project specifications: `../prompt/`
