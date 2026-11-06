import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import { createModelBundle } from '../src/resources/modelBundle.ts';
import { MujocoBackend } from '../src/physics/backends/mujoco/MujocoBackend.ts';
import { SceneBinding, ENVIRONMENT_ID } from '../src/physics/SceneBinding.ts';
import { writeQuaternion } from '../src/physics/rotation.ts';
import { selectionGeometry } from '../src/scene-selection.ts';

const mj = await loadMujoco();
const bundle = await createModelBundle('Scene mapping', 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(`
<mujoco><option gravity="0 0 0"/>
<worldbody>
 <geom name="floor" type="plane" size="3 3 .1"/>
 <body name="parent" pos="1 2 3" quat=".707106781 0 0 .707106781">
  <body name="moving" pos="0 0 1">
   <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
   <joint name="hinge" axis="0 0 1" range="-90 90"/>
   <geom name="capsule" type="capsule" size=".1 .4"/>
   <camera name="attached" pos="0 -2 1" euler="60 0 0" resolution="640 360"/>
   <light name="spot" pos="0 0 2" dir="0 0 -1"/>
   <light name="point" type="point" pos="1 0 2"/>
  </body>
 </body>
</worldbody></mujoco>`) }]);

test('camera matrix conversion handles every quaternion branch', () => {
  for (const axis of [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]) {
    for (const angle of [0, Math.PI / 3, Math.PI, -Math.PI]) {
      const expected = new THREE.Quaternion().setFromAxisAngle(axis, angle);
      const m = new THREE.Matrix4().makeRotationFromQuaternion(expected).elements;
      const rowMajor = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
      const result = new Float64Array(4);
      writeQuaternion(rowMajor, 0, result, 0);
      assert.ok(expected.angleTo(new THREE.Quaternion().fromArray(result)) < 1e-7);
    }
  }
});

test('real engine frames map multi-joint bodies, lights and cameras without invented joint parenting', async (t) => {
  const engine = await MujocoBackend.load(mj, bundle);
  t.after(() => engine.dispose());
  const binding = new SceneBinding(engine.source, await engine.frame());
  const doc = binding.document;
  const slide = doc.joints.find((joint) => joint.name === 'slide');
  const hinge = doc.joints.find((joint) => joint.name === 'hinge');
  assert.equal(slide.bodyId, hinge.bodyId);
  assert.equal(slide.object.parent, doc.root);
  assert.equal(hinge.object.parent, doc.root);
  await engine.execute({ type: 'jointPose', id: slide.id, position: [.3] });
  const frame = await engine.execute({ type: 'jointPose', id: hinge.id, position: [.5] });
  binding.apply(frame);
  for (const [index, body] of engine.source.bodies.entries()) {
    const object = doc.nodes.find((node) => node.id === body.id).object;
    const position = object.getWorldPosition(new THREE.Vector3()).toArray();
    position.forEach((value, axis) => assert.ok(Math.abs(value - frame.bodyTransforms[index * 7 + axis]) < 1e-12));
  }
  const camera = doc.nodes.find((node) => node.kind === 'camera').object;
  assert.equal(camera.aspect, 640 / 360);
  assert.deepEqual(camera.position.toArray(), Array.from(frame.cameraTransforms.slice(0, 3)));
  assert.ok(Math.abs(camera.quaternion.length() - 1) < 1e-12);
  assert.ok(doc.nodes.find((node) => node.name === 'point').object instanceof THREE.PointLight);
  const spot = doc.nodes.find((node) => node.name === 'spot').object;
  const direction = spot.target.position.clone().sub(spot.position).toArray();
  direction.forEach((value, axis) => assert.ok(Math.abs(value - frame.lightFrames[axis + 3]) < 1e-12));
  assert.equal(selectionGeometry(doc, hinge.id).size, 1);
  assert.equal(selectionGeometry(doc, ENVIRONMENT_ID).size, 0);
  assert.equal(doc.nodes.find((node) => node.name === 'floor').object.userData.viewportPickable, false);
  const geometry = doc.nodes.find((node) => node.name === 'capsule').object.geometry;
  geometry.computeBoundingBox();
  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  assert.ok(size.z > size.y * 4);
  const before = camera.position.toArray();
  const invalid = { ...frame, cameraTransforms: frame.cameraTransforms.slice() };
  invalid.cameraTransforms[0] = NaN;
  assert.throws(() => binding.apply(invalid), /incompatible/);
  assert.deepEqual(camera.position.toArray(), before);
});
