import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import * as THREE from 'three';
import { CameraViewTransition, cameraForTransition } from '../src/viewport/camera-transition.ts';

const reference = JSON.parse(
  await readFile(new URL('./fixtures/camera-transition-native.json', import.meta.url), 'utf8'),
);
const close = (actual, expected, label, tolerance = reference.tolerance) => {
  const a = Array.isArray(actual) ? actual.flat(Infinity) : [actual],
    b = Array.isArray(expected) ? expected.flat(Infinity) : [expected];
  assert.equal(a.length, b.length, `${label}: length`);
  for (let index = 0; index < a.length; index++)
    assert.ok(
      Number.isFinite(a[index]) && Math.abs(a[index] - b[index]) <= tolerance,
      `${label}[${index}]: ${a[index]} != ${b[index]}`,
    );
};
const rows = (matrix) => matrix.clone().transpose().elements;

for (const fixture of reference.cases) {
  test(`native camera transition matrices: ${fixture.id}`, () => {
    let start = fixture.start;
    if (fixture.interruption) {
      const interrupted = fixture.interruption;
      start = new CameraViewTransition(interrupted.start).sample(
        interrupted.target,
        interrupted.elapsed,
      );
      for (const key of [
        'position',
        'target',
        'up',
        'projectionBlend',
        'normalizedFocal',
        'principalOffset',
      ])
        close(start[key], fixture.start[key], `interrupted ${key}`);
    }
    const transition = new CameraViewTransition(start, fixture.duration);
    let reused;
    for (const step of fixture.steps) {
      const actual = transition.sample(step.target, step.elapsed),
        expected = step.expected.view,
        label = `${fixture.id} at ${step.elapsed}s`;
      if (step.elapsed >= fixture.duration)
        assert.equal(actual, step.target, `${label}: exact endpoint identity`);
      for (const key of ['position', 'target', 'up', 'fov', 'near', 'far', 'orthoHeight'])
        close(actual[key], expected[key], `${label}: ${key}`);
      for (const key of ['normalizedFocal', 'principalOffset'])
        if (expected[key]) close(actual[key], expected[key], `${label}: ${key}`);
      assert.equal(actual.projection, expected.projection, `${label}: projection metadata`);
      close(
        actual.projectionBlend ?? (actual.projection === 'orthographic' ? 1 : 0),
        expected.projectionBlend ?? (expected.projection === 'orthographic' ? 1 : 0),
        `${label}: blend`,
      );
      reused = cameraForTransition(actual, actual.resolution[0] / actual.resolution[1], reused);
      close(rows(reused.matrixWorldInverse), step.expected.viewMatrix, `${label}: view matrix`);
      close(
        rows(reused.projectionMatrix),
        step.expected.projectionMatrix,
        `${label}: projection matrix`,
      );
    }
  });
}

test('absolute samples freeze their starting pose, support replay and follow the current target', () => {
  const fixture = reference.cases.find((value) => value.id === 'moving-target'),
    start = structuredClone(fixture.start),
    before = structuredClone(start),
    transition = new CameraViewTransition(start),
    middle = fixture.steps[4],
    first = transition.sample(middle.target, middle.elapsed);
  start.position[0] += 500;
  start.target[1] += 500;
  start.up[2] = -500;
  start.resolution[0] = 900;
  transition.sample(fixture.steps.at(-1).target, 1);
  assert.deepEqual(transition.sample(middle.target, middle.elapsed), first);
  close(transition.sample(middle.target, -1).position, before.position, 'clamped start');
  const moved = structuredClone(middle.target);
  moved.position[0] += 1;
  assert.notDeepEqual(transition.sample(moved, middle.elapsed).position, first.position);
  close(transition.start.position, before.position, 'frozen start');
});

test('zero-duration handoffs return the target exactly without degenerate matrices', () => {
  const fixture = reference.cases[0],
    target = fixture.steps[0].target,
    transition = new CameraViewTransition(fixture.start, 0);
  assert.equal(transition.sample(target, 0), target);
  assert.equal(transition.sample(target, 100), target);
});

test('render cameras reuse compatible instances and keep inverse projection synchronized through a mixed handoff', () => {
  const fixture = reference.cases.find(
      (value) => value.id === 'perspective-to-orthographic-opposing',
    ),
    transition = new CameraViewTransition(fixture.start),
    target = fixture.steps[0].target,
    start = transition.sample(target, 0),
    first = cameraForTransition(start, 1.25),
    middle = transition.sample(target, 0.14),
    second = cameraForTransition(middle, 1.25, first);
  assert.equal(second, first);
  assert.ok(second instanceof THREE.PerspectiveCamera);
  assert.ok(second.userData.projectionBlend > 0 && second.userData.projectionBlend < 1);
  const ndc = new THREE.Vector3(0.2, -0.3, 0.25),
    restored = ndc.clone().unproject(second).project(second);
  close(restored.toArray(), ndc.toArray(), 'mixed projection inverse', 1e-10);
  const final = cameraForTransition(transition.sample(target, 0.28), 1.25, second);
  assert.notEqual(final, second);
  assert.ok(final instanceof THREE.OrthographicCamera);
  assert.equal(cameraForTransition(target, 1.25, final), final);
  assert.equal(final.userData.projectionBlend, 1);
});
