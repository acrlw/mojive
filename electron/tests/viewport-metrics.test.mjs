import test from 'node:test';
import assert from 'node:assert/strict';
import { Object3D } from 'three';
import { ViewportMetrics } from '../src/viewport/metrics.ts';
import { objectGizmosEnabled, DEFAULT_VIEWPORT_LAYERS } from '../src/viewport/layers.ts';
test('render stats report actual trailing-window work and zero FPS after render-on-demand becomes idle', () => {
  const m = new ViewportMetrics();
  assert.equal(m.sample(0).idle, true);
  assert.equal(m.sample(0).fps, 0);
  m.record(0, 4, { calls: 2, triangles: 12, lines: 40, points: 0 });
  m.record(16, 22, { calls: 7, triangles: 48, lines: 60, points: 3 });
  m.record(32, 35, { calls: 9, triangles: 60, lines: 80, points: 4 });
  const active = m.sample(100);
  assert.equal(active.fps, 3);
  assert.equal(active.idle, false);
  assert.equal(active.frameMs, 13 / 3);
  assert.equal(active.drawCalls, 9);
  assert.equal(active.triangles, 60);
  assert.equal(active.lines, 80);
  assert.equal(active.points, 4);
  const idle = m.sample(536);
  assert.equal(idle.fps, 0);
  assert.equal(idle.idle, true);
  assert.equal(idle.drawCalls, 9);
  assert.equal(m.sample(1100).frameMs, 3);
});
test('per-object gizmo visibility inherits the nearest authored flag without altering other layers', () => {
  const root = new Object3D(),
    body = new Object3D(),
    geom = new Object3D();
  root.add(body);
  body.add(geom);
  assert.equal(objectGizmosEnabled(geom), true);
  root.userData.gizmoEnabled = false;
  assert.equal(objectGizmosEnabled(geom), false);
  body.userData.gizmoEnabled = true;
  assert.equal(objectGizmosEnabled(geom), true);
  geom.userData.gizmoEnabled = false;
  assert.equal(objectGizmosEnabled(geom), false);
  assert.equal(DEFAULT_VIEWPORT_LAYERS.selection, true);
  assert.equal(DEFAULT_VIEWPORT_LAYERS.geometry, true);
});
