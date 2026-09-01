# Editor and MJCF

## Scene documents

Use `.mojive.json` while composing multiple models and Mojive entities. The document stores:

- MJCF and URDF model references
- model root position and rotation
- editable MjSpec XML for changed models
- authored geometry, materials, lights, cameras, and environment
- resource search directories

Use **File > Save As** and select MuJoCo XML / MJCF to produce a standalone MuJoCo model. The
exporter writes formatted XML, copies file-backed assets into `<name>_assets/`, writes relative
resource paths, and recompiles the document before completing the save. The XML and its sibling
asset directory can be moved together.

The exporter preserves directional, point, spot, area, and image lights plus 2D, cube, and skybox
textures. MuJoCo has no native area-light enum, so OpenGL writes a point-light fallback with bulb
radius and private text metadata that restores the area semantic when the file is reopened through
OpenGL. Image lights require a cube or skybox texture; export reports an error instead of silently
dropping an invalid reference. Save `.mojive.json` when the OpenGL composition itself, including model
references and resource roots, must remain editable.

## Editing model topology

Select a model or model element in Hierarchy. Inspector exposes bodies, geometry, joints, sites,
cameras, lights, actuators, sensors, tendons, and equality constraints. Topology edits compile a
new MjSpec model and migrate named simulation state where dimensions remain compatible.
Hierarchy and **Entity > Duplicate** copy a selected body subtree or leaf topology element with
collision-free names. References between elements inside a copied body subtree follow the copies;
assets and model-level components remain shared. Duplicate and Delete shortcuts use the same
Undo/Redo-aware topology path and require a paused simulation.

`ModelEditBatch` groups dependent topology operations into one compile and can reference an element
created earlier in the same batch. Numeric edits that do not change derived constants use narrower
paths: joint properties and geometry contact/solver/surface properties update MjSpec and the
compiled model without rebuilding `SceneSource`. Body inertia and geometry mass/group/fluid edits
are buffered until **Apply**, then rebuild the model once.

## Structured model properties

The structured Inspector currently covers:

- fixed body and site transforms;
- finite plane, box, sphere/ellipsoid, capsule, cylinder, and site dimensions;
- joint axis, limits, damping, stiffness, armature, friction loss, reference/spring reference,
  limit/friction solver parameters, visual group, and actuator-force policy/range;
- site type, visual group, and optional capsule/cylinder endpoints, in addition to direct pose,
  dimensions, color, and material controls;
- body auto/diagonal/full inertia, mass, inertial frame, gravity compensation, mocap, and sleep
  policy;
- geometry friction, contact dimension, collision masks, priority, margin, gap, solver mix,
  `solref`, `solimp`, adhesion, and surface velocity;
- geometry visual group, density or explicit mass, volume/shell inertia, and ellipsoid fluid
  coefficients;
- geometry type switching plus model-local OBJ/STL/MSH/PLY mesh and PNG height-field import and
  assignment;
- model-local material creation, duplication, assignment, inline appearance, and PNG 2D texture
  import, plus cube and skybox PNG import;
- explicit contact pairs and body exclusions;
- non-plugin actuators, the MuJoCo sensor catalog, fixed/spatial tendon paths, and equality
  constraints through schema-driven reference fields;
- model-local keyframe capture and editing through the dedicated **Window > Keyframes** Dope Sheet:
  diamond-marker selection, exact-state loading, previous/next navigation, zoom, pan, drag retiming,
  naming, deletion, and Undo/Redo. MuJoCo keyframes are complete snapshots, so the editor does not
  present unsupported interpolation or property curves. The transport can record an in-memory
  whole-scene simulation take without recompiling MJCF, replay it with its recorded timing, seek its
  first/previous/next/last frames, and promote the current take frame to a persistent model keyframe
  with **Capture Snapshot**. A new recording replaces the previous transient take.

The **Assets** panel is the model-level inventory; Inspector remains responsible for binding an
asset to the selected scene element. It covers standalone mesh, PNG height-field and texture
import, material creation and replacement, basic appearance, height-field physical dimensions,
reference reporting, assignment, rename with reference repair, duplication, file replacement,
safe deletion, Undo/Redo, and portable MJCF export. Use the selected geometry's **assigned
material** control to replace a floor or another surface material. Skin assets remain visible in
the inventory.
Attached-model declarations and generator source forms must still be edited in their original
external MJCF source because MuJoCo expands them before `MjSpec` serialization.
OpenGL keeps file-backed assets from an expanded attached model resolvable during ordinary topology
edits by writing their child compiler directories into the normalized asset paths; this does not
reconstruct the original `asset/model` or `attach` declaration.

These controls validate values, participate in Undo/Redo, persist in workspace documents, and use
the remote typed-command boundary where the adapter exposes the capability. Pause a simulation
before editing model properties.

Compiler/option/visual blocks, default classes, custom arrays and tuples, deformable declarations,
PBR texture-role layers, and bulk mesh/height-field samples remain source-owned. OpenGL loads,
renders, composes, and preserves these core MuJoCo sections, but does not duplicate their rarely
used source-authoring surface in Inspector. **Edit MJCF Source...** edits normalized XML produced
by `MjSpec`; it is the escape hatch for one-off source changes, plugin-defined components, raw bulk
asset payloads, and unusual schema combinations, but it is not a source-preserving text editor.
`MjSpec` expands include structure and removes comments when it serializes the model. Keep editing
the original external files when their include layout, comments, or formatting must remain intact.
The source popup compiles before applying changes and keeps the last good model when validation
fails.

Core coverage treats `mj_printSchema()` as the linked-version attribute inventory, not as a reason
to reject plugin-bearing files. Explicit plugin branches are reported separately and excluded only
from the structured-authoring completion gate. Loading and runtime behavior still follow the
plugins registered with MuJoCo. Source/meta elements that are not completely represented by
`mj_printSchema()`, including include/frame/replicate behavior, are tracked separately so a clean
schema report cannot be mistaken for source-preserving MJCF coverage.

File-less MuJoCo root edits are stored inline as `root_mjcf` in `.mojive.json` workspaces. This
preserves topology and model-component edits created directly from an empty editor without
inventing a temporary external XML file.

In Hierarchy, Ctrl/Cmd+click selects multiple rows. **Delete model elements** removes the selected
top-level model elements through one `ModelEditBatch`; descendants of another selected row are
collapsed so the same subtree is never deleted twice.

Use these visual acceptance entries for the supported structured paths:

```bash
make primitive-authoring BACKEND=wgpu
make material-authoring BACKEND=wgpu
make contact-authoring BACKEND=wgpu
make body-authoring BACKEND=wgpu
make resource-authoring BACKEND=wgpu
make asset-browser BACKEND=wgpu
make joint-site-authoring BACKEND=wgpu
make model-component-authoring BACKEND=wgpu
make keyframe-authoring BACKEND=wgpu
make batch-editing BACKEND=wgpu
make joint-gizmo BACKEND=wgpu
```

## Current pose and key0

When qpos differs from the model default, MJCF Save and Save As show a pose prompt. Choose
**Save as key0** to add or replace the `key0` keyframe with the current qpos, actuator state,
controls, and mocap state. Choose **Save without keyframe** to preserve the model default.

## Round-trip acceptance

Run the focused export gate:

```bash
make mjcf-roundtrip
```

The gate exports a composed scene, recompiles the file with MuJoCo, reloads it through
`MuJoCoAdapter`, moves the export directory, and verifies resources, authored cameras, lights,
geometry, and current-pose keyframes.

## Runtime entity tools

Camera and light transform gizmos lock while simulation runs. Use **Lock gizmo while simulation
runs** in Inspector to enable runtime editing for one entity. This keeps physical perturbation on
nearby bodies unambiguous.

Selecting a camera opens its live preview in the viewport. **Pin** freezes the preview camera and
widget position while the scene continues to update. **Lock** keeps the widget attached to that
camera entity and follows its live pose after selection changes.

Open **Edit > Settings...**, press `F9`, or run `make settings`. Settings is a centered modal; scene
interaction resumes after the panel closes.
