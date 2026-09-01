# Getting started

## Install

Mojive requires Python 3.11 or newer. Install the core package with MuJoCo and WebGPU
support from a source checkout:

```bash
git clone https://github.com/acrlw/mojive.git
cd mojive
uv sync --python 3.11 --extra dev --extra mujoco --extra wgpu
```

Open the default MuJoCo scene:

```bash
make viewer
```

Open an empty editor, then load MJCF or URDF from the File menu or drop model files into the
viewport:

```bash
make editor
```

Use WebGPU through Metal, Vulkan, or DX12:

```bash
make editor BACKEND=wgpu
```

## Display scale and language

The application reads the native content scale. An explicit value is useful for HiDPI testing:

```bash
MOJIVE_UI_SCALE=2 make editor
make editor LANGUAGE=zh_CN
```

## Load a model directly

```bash
uv run mojive view path/to/model.xml --paused
uv run mojive view path/to/model.urdf --paused
```

## Render with the MuJoCo-compatible API

```python
import mujoco

from mojive import Renderer

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

with Renderer(model, width=640, height=480) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=-1)
    rgb = renderer.render()
```

The [renderer reference](api/rendering.md) covers RGB, metric depth, segmentation, cameras,
render flags, and lifecycle methods.
