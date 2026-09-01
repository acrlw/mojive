# Custom scene adapter

A scene adapter connects stable structure and dynamic state to the viewer. This boundary supports
custom physics engines, procedural tools, replay sources, and remote publishers.

Implement these members first:

- `structure_revision`: increments after topology or resource changes;
- `scene_source()`: meshes, materials, hierarchy, cameras, and lights;
- `frame(needs)`: current transforms and requested dynamic diagnostics;
- `step()` and `reset()`: simulation control when `caps.simulation` is enabled.

Declare optional behavior through `AdapterCaps`. The session uses those capabilities to enable UI
and command paths.

## Minimal simulation adapter

```bash
.venv/bin/python examples/custom_adapter.py
```

```python
--8<-- "examples/custom_adapter.py"
```

Use `make adapter-conformance ADAPTER=<name>` after registering an adapter with
`mojive.backends.make_adapter`.
