import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import {
  frontArc,
  arrow,
  dimensionAxis,
  affineCorners,
  axisAlpha,
  ringAlpha,
} from '../src/viewport/geometry.ts';
const reference = JSON.parse(
  await readFile(new URL('./gizmo-native-reference.json', import.meta.url), 'utf8'),
);
function same(actual, expected, message, tolerance = 1e-8) {
  assert.equal(actual.length, expected.length, message + ' vertex count');
  for (let i = 0; i < actual.length; i++)
    for (let j = 0; j < 2; j++)
      assert.ok(
        Math.abs(actual[i][j] - expected[i][j]) < tolerance,
        `${message}: ${i}/${j}: ${actual[i][j]} != ${expected[i][j]}`,
      );
}
test('position arrows match native G3 contours at normal and foreshortened lengths', () => {
  for (const s of reference.arrows) same(arrow(s.start, s.end), s.points, `arrow ${s.end}`);
});
test('dimension handles center on the endpoint and collapse into native rounded square', () => {
  for (const s of reference.dimensions)
    same(dimensionAxis(s.start, s.end), s.points, `dimension ${s.end}`);
});
test('plane corners preserve native affine G3 contour at narrow extreme projection', () => {
  for (const s of reference.planes) same(affineCorners(s.points), s.rounded, `plane ${s.points}`);
});
test('native projection fade endpoints and continuous range', () => {
  assert.equal(ringAlpha(0.08), 0);
  assert.equal(ringAlpha(-0.28), 1);
  assert.ok(Math.abs(ringAlpha(0.18) - 0.5) < 1e-12);
  assert.equal(axisAlpha(1), 0);
  assert.equal(axisAlpha(0), 1);
  assert.ok(Math.abs(axisAlpha(Math.sqrt(1 - 0.18 ** 2)) - 0.5) < 1e-12);
});

test('perspective half-ring extrema and orthographic direction match native projection', () => {
  for (const s of reference.arcs) {
    const cam = s.orthographic ? new THREE.OrthographicCamera() : new THREE.PerspectiveCamera();
    cam.position.fromArray(s.eye);
    cam.matrixWorldInverse.set(...s.view.flat());
    cam.matrixWorld.copy(cam.matrixWorldInverse).invert();
    cam.quaternion.setFromRotationMatrix(cam.matrixWorld);
    cam.projectionMatrix.set(...s.projection.flat());
    cam.matrixAutoUpdate = false;
    cam.matrix.copy(cam.matrixWorld);
    const normal = new THREE.Vector3().setComponent(s.axis, 1);
    same(
      frontArc(new THREE.Vector3(...s.origin), normal, s.radius, cam, 900, 600),
      s.points,
      `arc ${s.axis} ${s.eye} ortho=${s.orthographic}`,
      1e-5,
    );
  }
});

test('joint slide arrows use native rounded tails including short-length cap fitting', () => {
  for (const s of reference.roundedArrows)
    same(arrow(s.start, s.end, true), s.points, `rounded arrow ${s.end}`);
});
