# Programmatic scene

Use `Scene` when geometry, cameras, lights, and materials originate in Python. The resulting scene
uses the same renderer, hierarchy, inspector, selection, and capture paths as physics-backed
models.

## Run the example

```bash
.venv/bin/python examples/programmatic_scene.py
```

The scene owns stable geometry and entity metadata. `build_scene()` wraps it with a static scene
adapter and composes the standard editor window.

## Source

```python
--8<-- "examples/programmatic_scene.py"
```

## Save and load

Programmatic scenes use the Forge JSON scene format:

```python
scene.save("output/examples/workcell.forge.json")
restored = Scene.load("output/examples/workcell.forge.json")
```

Use `WorkspaceAdapter` when one document combines authored entities with MJCF or URDF models.
