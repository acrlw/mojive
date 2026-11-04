import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import {
  elevatedFocusDirection,
  perpendicularFocusDirection,
  obliqueFocusDirections,
  leastOccludedFocusDirection,
  jointFocusPlan,
} from '../src/viewport/joint-focus.ts';
const fixture = JSON.parse(
  await readFile(new URL('./fixtures/joint-focus-native.json', import.meta.url)),
);
const vector = (values) => new THREE.Vector3().fromArray(values);
const close = (actual, expected, label) => {
  assert.equal(actual.length, expected.length, label);
  actual.forEach((value, index) =>
    assert.ok(
      Math.abs(value - expected[index]) < fixture.tolerance,
      `${label}[${index}]: ${value} != ${expected[index]}`,
    ),
  );
};
test('joint focus directions and marker readability match 40 live-native cases', () => {
  fixture.cases.forEach((f, index) => {
    const axis = vector(f.axis),
      offset = vector(f.offset),
      fallback = vector(f.fallback),
      candidates = obliqueFocusDirections(axis, offset, fallback, 35, f.markers.map(vector));
    close(elevatedFocusDirection(offset, fallback).toArray(), f.elevated, `${index} elevated`);
    close(
      perpendicularFocusDirection(axis, offset, fallback).toArray(),
      f.perpendicular,
      `${index} perpendicular`,
    );
    assert.ok(candidates.length >= 1 && candidates.length <= 9, `${index} bounded candidates`);
    // NumPy and JS can order exactly symmetric equal-score lattice points
    // differently at roundoff. Check cone/separation/order invariants for the
    // lattice and require the exact selected production result below.
    const current = (offset.length() > 1e-9 ? offset : fallback).clone().normalize();
    candidates.forEach((value, i) => {
      if (i)
        assert.ok(
          value.dot(current) <= candidates[i - 1].dot(current) + 1e-12,
          `${index} sorted turn`,
        );
      for (let j = 0; j < i; j++)
        assert.ok(
          value.dot(candidates[j]) <= Math.cos((10 * Math.PI) / 180) + 1e-12,
          `${index} separation`,
        );
      const angle = (Math.acos(Math.abs(value.dot(axis.clone().normalize()))) * 180) / Math.PI;
      assert.ok(angle >= 35 - 1e-6 && angle <= 55 + 1e-6, `${index} candidate ${i} readability`);
    });
    close(
      leastOccludedFocusDirection(candidates, offset, vector([1, 0, 0]), 0.2, 10).toArray(),
      f.chosen,
      `${index} chosen`,
    );
  });
});
test('occlusion selects nearby clear directions without jumping to a remote canonical quadrant', () => {
  const angle = (degrees) =>
      vector([
        Math.cos((degrees * Math.PI) / 180) * 0.8,
        Math.sin((degrees * Math.PI) / 180) * 0.8,
        0.6,
      ]),
    candidates = [angle(0), angle(15), angle(65)],
    current = angle(0).multiplyScalar(5),
    calls = [];
  const result = leastOccludedFocusDirection(
    candidates,
    current,
    vector([1, 0, 0]),
    0.2,
    8,
    (direction, distance) => {
      calls.push({ direction, distance });
      return direction.y < 0.1 ? { distance: 2, target: false } : null;
    },
  );
  close(result.toArray(), candidates[1].toArray(), 'near clear');
  assert.equal(calls.length, 2);
  assert.equal(calls[0].distance, 6);
  const own = leastOccludedFocusDirection(candidates, current, vector([1, 0, 0]), 0.2, 8, () => ({
    distance: 0.1,
    target: true,
  }));
  close(own.toArray(), candidates[0].toArray(), 'own body not obstruction');
});

test('joint plan uses current free position and full limited slide travel with local body extent', () => {
  const root = new THREE.Group(),
    parent = new THREE.Group(),
    object = new THREE.Group(),
    body = new THREE.Group(),
    mesh = new THREE.Mesh(new THREE.BoxGeometry(4, 0.4, 0.6));
  root.add(parent);
  parent.position.set(3, 2, 1);
  parent.rotation.z = 0.4;
  parent.add(object);
  object.add(body);
  body.add(mesh);
  const axis = vector([1, 0, 0]),
    origin = new THREE.Matrix4().makeTranslation(0.3, 0, 0.2),
    joint = {
      id: 'j',
      name: 'j',
      type: 'prismatic',
      object,
      origin,
      axis,
      lower: -0.5,
      upper: 1.5,
      value: 0.7,
      bodyId: 'b',
      child: 'b',
    },
    doc = {
      root,
      joints: [joint],
      nodes: [
        { id: 'j', kind: 'joint', object },
        { id: 'b', name: 'b', kind: 'link', object: body },
        { id: 'g', kind: 'geom', object: mesh, parentId: 'b' },
      ],
    },
    view = {
      position: [7, -8, 6],
      target: [0, 0, 0],
      up: [0, 0, 1],
      resolution: [800, 600],
      fov: 42,
    };
  mesh.userData.nodeId = 'g';
  object.position.set(1, 0, 0.2);
  root.updateMatrixWorld(true);
  const motionAxis = axis.clone().applyQuaternion(parent.quaternion),
    midpoint = body.getWorldPosition(new THREE.Vector3()).addScaledVector(motionAxis, -0.2),
    plan = jointFocusPlan(doc, 'j', view);
  close(plan.center, midpoint.toArray(), 'slide midpoint');
  assert.equal(plan.radius, 3);
  joint.type = 'free';
  object.position.set(2, -0.8, 1.3);
  root.updateMatrixWorld(true);
  const free = jointFocusPlan(doc, 'j', view);
  close(free.center, body.getWorldPosition(new THREE.Vector3()).toArray(), 'current free position');
  assert.equal(free.radius, 2);
});
