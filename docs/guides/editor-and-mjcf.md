# Editor and MJCF

## Scene documents

Use `.forge.json` while composing multiple models and Forge entities. The document stores:

- MJCF and URDF model references
- model root position and rotation
- editable MjSpec XML for changed models
- authored geometry, materials, lights, cameras, and environment
- resource search directories

Use **File > Save As** and select MuJoCo XML / MJCF to produce a standalone MuJoCo model. The
exporter writes formatted XML and recompiles the document before completing the save.

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
`MuJoCoAdapter`, and verifies the authored camera, light, geometry, and current-pose keyframe.
