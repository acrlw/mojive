# Editor and MJCF

## Scene documents

Use `.forge.json` while composing multiple models and Forge entities. The document stores:

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
textures. MuJoCo has no native area-light enum, so Forge writes a point-light fallback with bulb
radius and private text metadata that restores the area semantic when the file is reopened through
Forge. Image lights require a cube or skybox texture; export reports an error instead of silently
dropping an invalid reference. Save `.forge.json` when the Forge composition itself, including model
references and resource roots, must remain editable.

## Editing model topology

Select a model or model element in Hierarchy. Inspector exposes bodies, geometry, joints, sites,
cameras, lights, actuators, sensors, tendons, and equality constraints. Topology edits compile a
new MjSpec model and migrate named simulation state where dimensions remain compatible.

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
  import, plus cube and skybox PNG import.

These controls validate values, participate in Undo/Redo, persist in workspace documents, and use
the remote typed-command boundary where the adapter exposes the capability. Pause a simulation
before editing model properties.

This is not a complete form-based copy of the MJCF schema. Detailed mesh/height-field asset
parameters, many component subtypes, keyframe authoring, contact pair/exclude, default classes, and
global option/solver fields still use **Edit MJCF Source...**. The source popup compiles before
applying changes and keeps the last good model when validation fails.

Use these visual acceptance entries for the supported structured paths:

```bash
make primitive-authoring BACKEND=wgpu
make material-authoring BACKEND=wgpu
make contact-authoring BACKEND=wgpu
make body-authoring BACKEND=wgpu
make resource-authoring BACKEND=wgpu
make joint-site-authoring BACKEND=wgpu
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
