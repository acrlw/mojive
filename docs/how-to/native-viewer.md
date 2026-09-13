# Use the native bgfx backend

The native backend plugs into the existing Python Viewer, `Renderer`, and `SceneRenderer`.
Window interaction, panels, Session and adapters retain their public interfaces. C++ owns
GPU resources, submission and readback; bgfx selects the platform graphics API independently
from wgpu.

## Build and launch

Start with the [source installation](../getting-started.md). Then build the native renderer and
its shaders:

```bash
make native-viewer SCENE=joint_types
make native-editor
```

These targets configure the development build and open the production UI. Subsequent runs use
incremental compilation. `make native-python-build` builds without opening a viewer.
For scripts using that development build:

```bash
export MOJIVE_NATIVE_BUILD="$PWD/output/cpp-build"
MOJIVE_RENDERER=bgfx uv run --no-sync mojive view joint_types
```

An installed wheel locates its packaged extension and shaders without this environment variable.
A checkout's default native geometry build and its full bgfx build may be separate directories;
select the full renderer build for bgfx. See [development](../guides/development.md#private-native-extension).

## Python

```python
import mujoco
from mojive import Renderer

model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with Renderer(model, width=640, height=480, renderer="bgfx") as renderer:
    renderer.update_scene(data)
    rgb = renderer.render()
    next_rgb = renderer.render_async().result()
```

Use `renderer="bgfx"` with `SceneRenderer` and viewer builders as well. Image shapes, dtypes,
coordinate conventions and reusable output buffers follow the [renderer contract](../concepts/rendering.md).

## Resources and threading

- The Python UI thread owns window events, ImGui and Session operations.
- Native runtime work owns GPU API access. Waiting for native work and readback releases the GIL.
- Renderers and windows share the device while retaining independent scene and target resources.
- Presented scene textures stay on the GPU. Explicit capture returns owned CPU arrays.
- Readback queues are bounded. Finish or handle pending futures before replacing their inputs;
  resize and scene replacement can cancel pending results.
- Native logs use bounded history and are forwarded on the Python calling thread.

VSync is a shared bgfx device policy: an enabled window keeps shared presentation synchronized.
Use `--no-vsync` for explicit unsynchronized measurements. Caller-owned MuJoCo data must not be
mutated while a renderer reads it; Mojive-owned simulation uses the Session simulation driver.

## Platform checks

macOS uses Metal; Linux uses Vulkan with the native X11/Wayland window paths. Windows support
must be verified on its target machine before claiming compatibility. A hidden desktop window
still requires its window system. Offscreen GPU availability is a separate capability.

```bash
make gpu-bgfx
make native-parity
make native-ui-parity
make native-wayland-test       # Linux isolated compositor and input checks
make native-wheel-test         # Installed artifact without development paths
```

See [testing](../guides/testing.md) for model-path requirements, shader packaging, visual review,
and benchmark isolation. The [native fixtures](../guides/native-probe.md) test lower-level
contracts and do not replace production viewer acceptance.

## Independent-world replay

The G1 tools exercise independent pose streams with shared geometry:

```bash
make g1-worlds MENAGERIE_ROOT=/path/to/mujoco_menagerie G1_WORLDS=1024
make g1-worlds-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-worlds-transport MENAGERIE_ROOT=/path/to/mujoco_menagerie
make g1-worlds-monitor-benchmark MENAGERIE_ROOT=/path/to/mujoco_menagerie
```

The model comes from the local `unitree_g1/scene.xml`. The tool retrieves the pinned Unitree
motion CSV, verifies its checksum and never executes downloaded scripts. This is pose replay,
not independently stepped physics or training throughput. Original meshes are used by default;
LOD requires explicit `--mesh-ratio` / `--mesh-error` arguments and separate result directories.

Compare the same geometry in each backend. Report pose update, renderer update, completed RGB,
first-frame cost and memory separately. Transport-only timing does not include rendering;
monitor timing includes completed image age, which still differs from display scanout latency.
