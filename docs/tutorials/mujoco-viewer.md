# Migrate from mujoco.viewer

Use `import mojive.viewer as viewer` for MuJoCo-style interactive calls. This entry point uses
Mojive's full desktop UI by default. `mojive.Renderer` remains a separate, UI-free offscreen API.
The compatibility target is MuJoCo 3.11 and its
[official viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#interactive-viewer).
The common patterns below are supported; the differences table lists incomplete coverage.

## Managed viewer

`launch()` blocks until the window closes. Mojive owns physics stepping in this mode.
Run `make viewer-managed` for a complete minimal example:

```python
--8<-- "examples/viewer_managed.py"
```

```python
import mujoco
import mojive.viewer as viewer

viewer.launch()                       # Empty editable workspace
m = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
viewer.launch(m)                      # Creates MjData internally
d = mujoco.MjData(m)
viewer.launch(m, d)                   # Steps this exact model/data pair
viewer.launch(loader=lambda: (m, d))
viewer.launch_from_path("assets/joint_types.xml")
```

The standalone spellings are `python -m mojive.viewer` and
`python -m mojive.viewer --mjcf=assets/joint_types.xml`.
Mojive-specific `renderer=` and `config=` arguments are keyword-only additions to `launch()`
and `launch_passive()`. Feature reduction requires an explicit configuration.

## Passive viewer

The caller owns stepping. `launch_passive()` returns a handle and keeps the UI responsive on
its own thread. Linux and Windows do not require a multiprocessing main guard for this API.
Run `make viewer-passive` (or `make viewer-passive ARGS='--seconds 10'`) for a complete example:

```python
--8<-- "examples/viewer_passive.py"
```

Both examples load the same scene and retain the existing default layout. In managed mode,
Space uses Mojive's playback control. In the passive example, Space calls the supplied callback
to pause the caller's stepping loop; camera navigation and synchronization continue while paused.
Passive playback belongs to the caller, so the viewer's internal physics controls are disabled.
Use `ARGS='--model /path/to/model.xml --renderer bgfx'` to select a model/backend in either example.

```python
import mujoco
import mojive.viewer as viewer

m = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
d = mujoco.MjData(m)
with viewer.launch_passive(m, d) as handle:
    with handle.lock():
        handle.cam.distance = 5
        handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    while handle.is_running():
        mujoco.mj_step(m, d)
        handle.sync()
```

The handle's `m` and `d` expose the original objects while running. Rendering uses private
copies and does not access caller model/data between sync calls. Every `sync()` performs an
exchange; it does not silently skip publication because the display has reached its frame limit.
Changed UI values apply once instead of restoring an entire stale state over the caller.

Runtime edits are classified by their effect. All `model.opt` fields (including the integrator,
solver, contact overrides and enable/disable flags) update in place. Friction, damping and other
physics-only arrays also avoid scene/resource reconstruction. Actuator-group toggles refresh the
existing visibility array; mass/inertia changes refresh diagnostic geometry. Actual appearance,
geometry or asset changes refresh the affected scene representation without recompiling `MjModel`.

As with MuJoCo, the caller remains responsible for derived constants when directly modifying
`MjModel`: apply `mujoco.mj_setConst(model, data)` where required before publishing the edit.
`mj_forward` recomputes state-dependent results; neither operation is model recompilation.
Changing topology, resource allocation or unsafe collision geometry requires a properly rebuilt
model and a new passive viewer. See MuJoCo's [runtime model-edit rules](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#mjmodel-changes).

- `sync()` synchronizes writable model fields, data, camera, options, perturbation and user geoms.
- `sync(state_only=True)` publishes integration state followed by `mj_forward` in the display.
  Caller model edits wait until a full sync; UI edits still return to the caller.
- `lock()` is a context manager protecting exchanges and visualization edits from the UI thread.
- `close()` ends the window and joins its thread. The handle supports context management.
- `key_callback(keycode)` receives GLFW integer codes on the UI thread, including while the caller
  is paused without calling `sync()`. Keep callbacks short.
- `cam`, `opt`, `perturb` and `user_scn` are mutable MuJoCo structures. `pert` aliases `perturb`.
- `viewport` is an `MjrRect` in framebuffer pixels, using a bottom-left origin.
- `update_hfield(id)`, `update_mesh(id)` and `update_texture(id)` publish only the specified
  resource's samples, mesh arrays or pixels and wait for a display frame to upload it. Other
  model edits wait for `sync()`. Compiled resource dimensions and offsets must remain unchanged.
  Call these methods outside `lock()`, as they lock internally.

Run `make viewer-compat ARGS='--seconds 10'` for Space-to-pause, camera/options and a custom sphere:

```python
--8<-- "examples/mujoco_viewer.py"
```

## UI visibility

### Visual and collision geometry

**Hierarchy > Geometry view** selects **Default / Visual / Collision / Both** for the scene.
Select a link and use **Inspector > Link geometry** to override just its own geometry;
child links keep their own settings. **Follow scene** clears that override. Narrow panels
use a dropdown. The scene selector is also available in **Settings > MuJoCo Visuals**.
Default preserves existing visual groups; the other views include hidden geom groups
without changing those group choices:

- **Visual** shows appearance geometry with its original materials.
- **Collision** shows collision geometry in opaque amber, including transparent or group-hidden
  proxies. Ordinary collision meshes use their compiled convex hulls; SDFs use their display
  surface. This view does not disable or enable physical collisions.
- **Both** compares the two: separate appearances use stippled coverage around their proxies;
  shared meshes display a stippled hull over the original mesh. This diagnostic transparency
  preserves instanced drawing instead of sorting one draw per translucent object. Identical
  shared primitives are drawn once; transparent shared primitives use opaque collision color.
  Shared hull overlays expand by 0.5% to avoid coplanar
  flicker; Collision mode uses the exact unexpanded geometry. Visual and Default retain
  ordinary material transparency.

MuJoCo does not carry an explicit appearance/proxy role. Mojive classifies collision candidates
using both masks and explicit contact pairs. A body with separate non-colliding appearance geoms
uses its collidable geoms as collision proxies; otherwise they serve both roles. Moving welded
bodies share that classification. World-attached geoms remain shared unless explicitly tagged.
Unusual models can override this presentation rule using named JSON arrays in MJCF:

```xml
<custom>
  <text name="mojive_visual_geoms" data='["shell"]'/>
  <text name="mojive_collision_geoms" data='["proxy"]'/>
  <text name="mojive_shared_geoms" data='["floor"]'/>
</custom>
```

Names must exist and may occur in only one list. These tags affect display classification only.
Custom scene providers can fill `SceneSource.geom_role` using `mojive.GeometryRole` bits;
an empty array preserves visual-only behavior. Skins are visual-only; collidable flex surfaces
serve both roles. Existing diagnostic overlays and manual node visibility remain independent.
The collision view does not add contact-margin envelopes. Flex reuses its existing display
shells/tubes, and SDF geometry uses its display mesh; neither gains a separate exact collision
boundary extractor. Use contact diagnostics to inspect generated contacts. The view includes
collision candidates, not just bodies currently touching.

For UI-free output, both `mojive.Renderer` and `mojive.SceneRenderer` expose
`set_geometry_view("visual" | "collision" | "both" | "default")`. With the MuJoCo-compatible
Renderer, call `update_scene(data)` after selecting the mode. Screenshots and recordings from
the Viewer use the current display mode. `make geometry-views` generates a four-view comparison.

### Panels and layout

Full functionality is available by default while preserving the existing default layout and
each panel's initial open state. This does not force every panel to be expanded.
Both `launch()` and `launch_passive()` accept the same `ViewerConfig`: use
`panels={"keyframes": PanelConfig(open=False)}` to initially close a panel, or
`builtin_panels=("inspector", "control", "joints", "camera")` to load only selected panels.
Import `ViewerConfig` and `PanelConfig` from `mojive` and pass the configuration as `config=`.

`show_left_ui=True` and `show_right_ui=True` are the defaults. `False` initially closes a dock
group without disabling features. The left group contains Hierarchy, Assets and bottom panels;
the right group contains Control, Joints, Camera, Inspector and related panels. `Tab` toggles
the left group; `Shift+Tab` toggles the right group. The Window menu exposes individual panels.
Menu/status bars and viewport tools remain available. The compatibility entry point selects the
MuJoCo navigation preset when no input preferences have been saved; deliberate user remaps are
preserved. Layout, selection, gizmos and editing gestures remain Mojive's.

## Current differences

| Area | Behavior |
|---|---|
| macOS passive UI thread | The compatibility API raises `NotImplementedError`: Mojive has no `mjpython` dispatcher. Use the process extension below from a guarded script, or a managed viewer on the main thread. |
| `user_scn` | Supports untextured sphere, ellipsoid, box, capsule, cylinder, finite plane, opaque arrow, line and label geometry plus supported render flags. Primitives use Mojive's debug pass. Unsupported geometry or material bindings raise `NotImplementedError` during sync. |
| MuJoCo 3.11 overlays | `set_figures`, `set_texts` and `set_images` raise `NotImplementedError`; use Mojive Plot/Canvas2D. Their clear methods are harmless when no such overlay exists. |
| Render options | Mapped to supported Mojive backend flags, label/frame modes and visual groups; images are not guaranteed to match MuJoCo pixels. |
| Model replacement | Passive model/data identities remain fixed; topology replacement and file loading are disabled. Close and relaunch after replacing the model. |
| Concurrency | A second compatibility viewer in the same process is rejected. Do not run another Mojive UI loop concurrently in that process. Separate processes retain independent UI ownership. |

The existing **`mojive.launch_passive`** / **`mojive.passive.launch_passive`** API is a different,
process-based extension described in [passive viewing](passive-viewing.md). It supports independent
display scheduling, captures, recording, RPC and declarative actions, requires a guarded script,
and limits state publication to `max_fps`. It has not silently changed to the new handle.
Choose **`mojive.viewer.launch_passive`** for the MuJoCo migration pattern above.
