# Materials, skyboxes and floor grids

Material editing changes rendered appearance. A shared material affects every geometry bound to
it; an unbound material has no visible effect until assigned. Materials control color, emission,
specular response, shininess, reflection, texture and repetition. They do not replace geometry or
change contact friction.

## In the editor

For an editable MuJoCo model, pause simulation and select a geometry in Hierarchy or Inspector.
The Inspector's **material > appearance** section offers **assigned material**, **New material**,
**Duplicate material**, texture import and texture repeat. A plane uses the same controls as a box.
**Assets > New material** creates an unbound asset. Select that asset and a geometry, then click
**Assign to Selected Geometry**. Materials are model-local; assignment across models is rejected.
Sites also accept material assignment. **Open in Assets** loads the panel when it was omitted at
startup.

An explicit instance color overrides a material's base color. This explains why assigning a material
can change its texture or reflectance without changing the object's color. MuJoCo's own geometry
color/default rules apply to model assets. For programmatic scenes, the API below lets an object
follow its material again.

Skyboxes use cube/skybox texture resources selected through the environment controls. A skybox is
an environment image, not a surface material. Image-based illumination uses a separate image light.

## Programmatic scenes

```python
import numpy as np
from mojive import Scene
from mojive.types import Material

scene = Scene()
box = scene.box(position=(0, 0, 0.5))
box.set_material(Material(name="blue", rgba=np.array((0.1, 0.3, 0.9, 1), np.float32)))
# An explicit color overrides the material; None restores inheritance.
box.set_color((1, 0.2, 0.1, 1))
box.set_color(None)
```

New objects follow their material when `color` is omitted. The default material preserves the
usual neutral object appearance. `scene.set_material(index, value)` replaces a shared material;
`box.set_material(value)` replaces only that object's binding. Both publish a structure revision
so `SceneRenderer.update_from(adapter)` sees the change. Scene save/load, duplication and Undo
preserve color inheritance. Older scene files with explicit color arrays keep their appearance.

Add or replace a texture with `scene.add_texture(TextureData(...))`, assign its name to
`Material.texture`, and use `Material.tex_repeat` to change surface tiling. Add a cube texture and
call `scene.set_skybox(name)` for the sky. Invalid skybox names/types are rejected.

## Add a grid without changing the floor material

```python
from mojive import Occlusion

grid = renderer.canvas2d.layer("floor-grid", depth=0.003, occlusion=Occlusion.DEPTH)
grid.grid("lines", (-5, -5, 5, 5), spacing=0.5, color=(0.2, 0.8, 1, 1), width_px=1.5)
# Reuse the ID to update the existing grid.
grid.grid("lines", (-5, -5, 5, 5), spacing=1.0, color=(1, 0.7, 0.1, 1), width_px=2)
grid.visible = False
```

This XY grid lies slightly above a Z=0 floor to avoid coincident depth. It has world-unit spacing
and screen-pixel line width. It is a retained debug overlay, so it is not saved as a material or
included in object-ID/depth products. A checker/grid image applied as a floor texture is instead
part of the surface material and is saved with the scene. There is currently no dedicated floor-grid
editor panel; use Canvas2D or import a texture.

Run `make material-workflow` for the complete offscreen example, or
`MOJIVE_GL=egl make material-workflow` on headless Linux. The implementation is
`python/tools/material_workflow.py`, and captures are written under `output/material-workflow/`.
