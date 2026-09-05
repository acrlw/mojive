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
uv run python examples/custom_adapter.py
```

```python
--8<-- "examples/custom_adapter.py"
```

Derive from `SceneAdapterBase` for the full editor: it supplies defaults for unsupported
operations. `SceneAdapter` describes that complete editor interface; `SceneProvider` describes
only `structure_revision`, `scene_source()`, and `frame(needs)`. Read-only consumers such as
`SceneRenderer.update_from()` accept the smaller protocol without simulation or authoring stubs.
Capabilities describe supported write-back; do not advertise a capability without implementing
its operations. Frames may reuse arrays until the next frame request; consumers retaining them
must copy the required data.

Register an external adapter factory in the process that will use it:

```python
from mojive import build, check_adapter, register_adapter
from my_engine import MyAdapter

register_adapter("my-engine", MyAdapter, label="My engine")
adapter = MyAdapter()
try:
    report = check_adapter(adapter)
finally:
    adapter.release()

with build("model.custom", adapter_name="my-engine", renderer="opengl") as viewer:
    viewer.run()
```

The factory takes no arguments; a closure can supply engine configuration. `make_adapter()`
loads an optional asset and releases the newly created adapter if loading fails. Registration
rejects duplicate or built-in names. `unregister_adapter(name)` removes a custom registration;
existing adapter instances remain caller-owned. Registrations are process-local, with no
implicit package imports or discovery across processes. An application CLI can register its
factories before calling `mojive.cli.main()`.

Use `check_adapter(adapter)` for a custom instance, or
`make adapter-conformance ADAPTER=toy` for a built-in adapter. Importing shared contracts,
`Scene`, or `SceneProvider` does not initialize physics, UI, or graphics packages.

Legacy metadata names retain their constructor compatibility: `JointInfo.body` is a body index,
`qpos_adr` and `qvel_adr` are starting addresses in their state arrays, and `ActuatorInfo.joint`
is a joint index. These lookup values are distinct from selection object IDs. New extension
names should use explicit `*_index`, `*_address`, and `object_id` terminology.
