# forge-viewer

forge-viewer is an interactive 3D viewer for robotics, simulation, and rendering tools. Its
Forge renderer consumes a backend-neutral scene protocol, so MuJoCo, custom physics engines,
static canvases, remote processes, and recorded snapshots share one rendering and UI stack.

Core workflows include scene inspection, object selection, transform gizmos, physical
perturbation, debug drawing, camera and light editing, render diagnostics, capture, and video.

The project is under active development. The P0/P1 baseline includes the public Renderer API,
MuJoCo visualization semantics, persistent scene state, local control, and visual regression gates.

## Requirements

- Python 3.11 or newer
- An OpenGL 3.3 core-profile driver for Forge, or a Metal/Vulkan adapter for wgpu
- A desktop session for interactive windows
- MuJoCo 3.1 or newer for MJCF, URDF, simulation, and physics tools

Forge targets macOS on Apple Silicon and Linux with a desktop OpenGL 3.3 driver. The wgpu backend
is validated on macOS Metal and Linux Vulkan. `uv` is the recommended environment and dependency
manager.

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

### Display scaling

The viewer reads the GLFW content scale and keeps ImGui, fonts, 2D and 3D gizmos, perturbation
marks, picking, and world-space labels at the same physical size. Linux X11, Linux Wayland,
Windows, and macOS use their native framebuffer coordinate models.

Use an explicit scale when the desktop session reports an incorrect value:

```bash
FORGE_VIEWER_UI_SCALE=2 make viewer
make hidpi
make hidpi BACKEND=wgpu UI_SCALE=2
```

`make hidpi` opens the gizmo scene at a 200% UI scale. Set `UI_SCALE=1.5` to inspect fractional
scaling.

### Language and settings

Open `Edit > Settings...` or press `F9` to use the centered modal Settings panel. The editor stays
inactive until the panel closes. The language choice is stored in the platform application-settings
directory. Start directly in Simplified Chinese with:

```bash
make editor LANGUAGE=zh_CN
```

The UI atlas combines JetBrains Mono with Noto Sans SC. Noto is loaded from the system or downloaded
to the application cache with checksum verification. `FORGE_VIEWER_CJK_FONT=/path/to/font.otf`
selects a different CJK font file.

### Linux OpenGL contexts

Interactive windows use the native GLFW context API. On Wayland this is EGL; on X11 it is GLX.
Force GLFW EGL explicitly to verify that path:

```bash
make egl-viewer
make egl
```

The offscreen `Renderer` uses EGL by default on Linux. `FORGE_VIEWER_GL=native` selects a hidden
GLFW context. Both paths create desktop OpenGL 3.3 core contexts.

Open a local MJCF or URDF model directly:

```bash
uv run forge-viewer view path/to/model.xml --paused
uv run forge-viewer view path/to/model.urdf --paused
```

Start with an empty viewport and load a model from the File menu or by dropping it into the window:

```bash
make empty
```

Pause the simulation to compose models. `File > Add Model...` attaches another MJCF or URDF at
the current camera target. Dropping several model files opens the first and adds the rest. Attached
models appear as model groups in the Hierarchy and can be removed from the File menu, Hierarchy
context menu, or Inspector. Topology rebuilds preserve matching joint, actuator, mocap, equality,
and simulation-time state.

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
forge-viewer editor [--no-vsync]
forge-viewer canvas [--demo empty|canvas|lighting|text]
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
forge-viewer rpc-serve <asset> [--socket output/forge-viewer.sock]
forge-viewer control <method> [--params JSON] [--json]
```

JSON commands reserve stdout for the JSON document and send logs to stderr.

The main menu and window file drop open MJCF, XML, and URDF models at runtime. `File > Reload
Model` recompiles the current model and rebuilds GPU scene resources.

`make editor` starts an empty Forge workspace. A `.forge.json` workspace combines MJCF and URDF
models with Forge-authored geometry, materials, lights, cameras, and environment settings. Model
paths resolve from the workspace directory and its resource directories. Every model has an
editable root position and rotation. URDF enters as an import format and is stored as editable
MJCF when its topology changes.

The File menu creates, opens, and saves workspaces. `Add Model...` and file drop compose robots and
scene assets. `File > Resource Directories` manages reusable asset search paths. Opening a
workspace with missing models shows a repair dialog: locate one file directly or search a
directory to rewrite every unambiguous path before loading. The Hierarchy
context menu adds and removes MJCF bodies, geometry, joints, sites, cameras, and lights through
MjSpec. Selecting a model exposes structured actuator, sensor, tendon, and equality components in
Inspector; edits use model-local reference choices, MjSpec validation, undo/redo, and workspace
round trips. Model elements support rename and local transform editing. The Entity menu creates
backend-neutral primitives, lights, and cameras. Selected Forge entities support duplicate,
rename, and delete from the menu, keyboard shortcuts, and the Hierarchy context menu.

Cameras and lights are selectable in the viewport. Their position and rotation gizmos edit world
transforms; selected helpers show camera frustums and light influence volumes. Selecting a camera
also opens a draggable live preview in the lower-right corner of the viewport. **Pin** freezes the
current preview camera and widget position. **Lock** keeps the widget attached to that camera entity
and follows its live pose after selection changes. Camera and light gizmos lock by default while
simulation runs; Inspector can unlock an entity for runtime editing.
The Settings panel controls helper and influence visibility. `View Through Camera` switches to the
selected scene camera, and `Return to Editor Camera` restores the previous editor orbit view.
Unsaved workspaces display an asterisk in the title and prompt before replacement or exit.

Runnable integrations from basic scene construction through MuJoCo rendering, model composition,
and remote publishing live in [`examples/`](examples/README.md).

## Programmatic rendering

`forge_viewer.Renderer` provides the MuJoCo-style offscreen workflow through Forge. It supports
RGB, metric depth, segmentation, free and fixed cameras, named cameras, `MjvCamera`, `MjvOption`,
caller-owned output arrays, multiple contexts, and deterministic resource release.

```python
import mujoco
from forge_viewer import Renderer

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

with Renderer(model, height=480, width=640) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=-1)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    depth_m = renderer.render()

    renderer.enable_segmentation_rendering()
    object_id_and_type = renderer.render()
```

Run the public contract and real-OpenGL comparison gallery with `make renderer-api`.

### wgpu backend

The Renderer API can run on [wgpu](https://wgpu.rs/) (Vulkan/Metal/DX12) instead of OpenGL,
which removes the EGL/GLFW context requirement for offscreen rendering:

```bash
pip install forge-viewer[wgpu]
FORGE_VIEWER_BACKEND=wgpu python your_script.py
```

Validate the installation with `make renderer-api-wgpu` and the backend-parameterized GPU
suite with `make gpu-wgpu`. The full interactive viewer also runs on wgpu:

```bash
make viewer BACKEND=wgpu
```

The wgpu backend supports the interactive viewer and the public Renderer API on Metal and
Vulkan. It includes render flags, debug views, shadows, planar reflections, skybox/IBL,
tendons, debug draw, selection outlines, and the native gizmo. `backend.caps.notes` reports
backend-specific limits such as optional GPU timing and single-sample ID/depth export.

## Visual acceptance

Every user-facing feature has a reproducible Make target.

| Target | Purpose |
|---|---|
| `make viewer` | Open the default MuJoCo scene |
| `make egl-viewer` | Open the Linux viewer through GLFW EGL |
| `make hidpi` | Inspect UI, fonts, and gizmos at an explicit 200% scale |
| `make hidpi-gallery` | Capture enlarged 2D/3D gizmo references at 200% UI scale |
| `make empty` | Open an empty viewer and load MJCF or URDF from the File menu |
| `make model-loading` | Capture empty, MJCF, and URDF runtime-loading references |
| `make model-composition` | Validate MjSpec state migration and capture add/remove references |
| `make editor-performance` | Record large composition editing timings under `output/` |
| `make stability BACKEND=wgpu` | Run long-frame, large-model lifecycle, and multi-camera gates |
| `make outline` | Selection, x-ray outline, and outline antialiasing |
| `make gizmo` | Native 2D and 3D transform gizmos |
| `make gizmo-gallery` | Enlarged position, rotation, and snap reference images |
| `make perturb` | MuJoCo translation and rotation perturbation |
| `make text-overlay` | GPU world-space text |
| `make lighting` | Editable lights and Environment controls for ambient light, fog, haze, and headlight |
| `make image-light` | MuJoCo cube-map environment light with editable intensity and texture |
| `make many-lights` | 16-light and 24-light MuJoCo reference images |
| `make scene-icons` | Camera and light scene icons |
| `make scene-entities` | Selectable camera and light helpers, transforms, frustums, and influence volumes |
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
| `make editor` | Empty authored workspace with scene files and Entity editing |
| `make settings` | Open the editor with the centered modal Settings panel |
| `make workspace-edit` | Workspace composition, resource repair, structured MJCF, camera and light acceptance |
| `make editor-files` | Scene document workflow acceptance and capture |
| `make entity-edit` | Entity lifecycle CPU and GPU acceptance |
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
| `make p0` | Complete public Renderer compatibility gate |
| `make p1` | Complete P0 and P1 non-interactive acceptance gate |
| `make check` | Lint, formatting, and CPU tests |
| `make gpu` | Isolated real-OpenGL test files |
| `make egl` | Linux EGL Renderer and geometry-shader wireframe contract |
| `make renderer-api` | Public Renderer RGB, depth, segmentation, camera, option, and lifecycle contract |
| `make mujoco-physics` | Full MuJoCo adapter and simulation regression suite |
| `make camera-state` | Camera bookmark serialization and restore |
| `make scene-snapshot` | Complete physics, selection, option, light, environment, and material state |
| `make cli` | Typed local control command contract |
| `make rpc` | Versioned local RPC and RGB/depth/segmentation capture artifacts |
| `make material-parity` | Texture, transparency, tendon, deformable, and dense-scene baselines |
| `make shadow-scheduling` | Deterministic 100-light and 8-shadow-slot scheduling |
| `make golden` | Golden-image comparison |
| `make reverse` | Mutation checks for regression assertions |
| `make doctor` | Window-path smoke test |
| `make mujoco-audit` | MuJoCo feature coverage for one model |
| `make adapter-conformance` | Backend-neutral scene contract |
| `make bench` | Median CPU and GPU pass timing |
| `make editor-performance` | Composition compile and structured-edit timing baseline |
| `make stability BACKEND=wgpu` | Memory, lifecycle, and interleaved multi-camera stability |
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

Snapshot recording stores structure, frames, and debug commands in versioned `.fvs` files:

```bash
make snapshot-record LIVE_SCENE=gizmo SNAPSHOT=output/session.fvs
make snapshot-replay SNAPSHOT=output/bug.fvs
```

Camera bookmarks and complete scene snapshots use versioned JSON under `output/snapshots/`.
`make camera-state` and `make scene-snapshot` generate acceptance artifacts there.

Local automation uses a versioned AF_UNIX control service. Clients keep the connection open across
requests and reconnect after a timeout or transport failure:

```bash
forge-viewer rpc-serve humanoid
forge-viewer control get_state --json
forge-viewer control capture --params '{"mode":"depth","output":"output/depth.npy"}'
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
│   ├── forge/                         OpenGL renderer, passes, and shaders
│   └── webgpu/                        wgpu renderer, passes, and WGSL shaders
└── ui/                                window, panels, gestures, and gizmos
```

Renderer code depends on shared scene contracts. Physics integration lives in adapters. UI code
depends on protocols and session state. `tests/test_layering.py` enforces these boundaries.

Renderer contracts use row-major matrices in Python. Each backend applies its GPU upload
convention. World coordinates use Z-up.

## Documentation

- [Renderer design](docs/RENDERER.md)
- [wgpu backend report](docs/WGPU_BACKEND_REPORT.md)
- [Platform measurements](docs/PLATFORM.md)
- [Implementation decisions](docs/DECISIONS.md)

Development plans live in [plan/](plan/README.md).

## License

forge-viewer is available under the [MIT License](LICENSE).
