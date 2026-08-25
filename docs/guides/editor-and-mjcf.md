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

MJCF represents directional, point, and spot lights plus 2D textures. Save As reports Forge scene
features outside that format, such as area lights and image lights. Save `.forge.json` to preserve
the complete Forge scene.

## Editing model topology

Select a model or model element in Hierarchy. Inspector exposes bodies, geometry, joints, sites,
cameras, lights, actuators, sensors, tendons, and equality constraints. Topology edits compile a
new MjSpec model and migrate named simulation state where dimensions remain compatible.

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
