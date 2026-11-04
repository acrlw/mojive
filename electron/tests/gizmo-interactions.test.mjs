import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: path.resolve('.'),
    server: { host: '127.0.0.1', port: 0 },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  server.middlewares.use('/__gizmo_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Gizmo gestures</title>');
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__gizmo_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function gesture(kind) {
  return page.evaluate(async (kind) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { GizmoOverlay } = await import('/src/viewport/gizmo.ts');
    const { worldScale, project, frontArc } = await import('/src/viewport/geometry.ts');
    const { createPrimitive } = await import('/src/scene.ts');
    const node = createPrimitive('box', 'Test box'),
      doc = {
        name: 'test',
        root: new THREE.Group(),
        nodes: [node],
        joints: [],
        files: [],
        warnings: [],
      };
    doc.root.add(node.object);
    node.object.position.set(0, 0, 0);
    doc.root.updateMatrixWorld(true);
    const canvas = document.createElement('canvas');
    document.body.append(canvas);
    let begins = 0,
      commits = 0,
      changes = 0;
    const gizmo = new GizmoOverlay(canvas, {
        onBegin: () => begins++,
        onCommit: () => commits++,
        onChange: () => changes++,
        invalidate: () => {},
      }),
      camera = new THREE.PerspectiveCamera(42, 900 / 600, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -4, 2.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const origin = new THREE.Vector3(),
      scale = worldScale(origin, camera, 600),
      center = project(origin, camera, 900, 600),
      tip = project(new THREE.Vector3(scale, 0, 0), camera, 900, 600),
      delta = [tip[0] - center[0], tip[1] - center[1]],
      length = Math.hypot(...delta);
    let start = [center[0] + delta[0] * 0.75, center[1] + delta[1] * 0.75],
      end = [start[0] + (delta[0] / length) * 25, start[1] + (delta[1] / length) * 25];
    gizmo.update(
      camera,
      doc,
      node.id,
      kind === 'rotate' ? 'rotation' : 'position',
      'world',
      900,
      600,
    );
    if (kind === 'rotate') {
      const path = frontArc(origin, new THREE.Vector3(0, 0, 1), scale * 0.78, camera, 900, 600);
      start = path[12];
      end = path[17];
    }
    const acquired = gizmo.begin(start);
    gizmo.move(end, false, kind === 'snap');
    const moved = node.object.position.toArray(),
      q = node.object.quaternion.toArray();
    gizmo.end(kind === 'cancel');
    const restored = node.object.position.toArray();
    gizmo.dispose();
    canvas.remove();
    return {
      acquired,
      begins,
      commits,
      changes,
      moved,
      restored,
      q,
      expected: (25 * scale) / length,
    };
  }, kind);
}
test('axis drag uses native projected travel and one command transaction', async () => {
  const r = await gesture('move');
  assert.equal(r.acquired, true);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
  assert.equal(r.changes, 1);
  assert.ok(Math.abs(r.moved[0] - r.expected) < 1e-8);
  assert.ok(Math.abs(r.moved[1]) + Math.abs(r.moved[2]) < 1e-9);
});
test('Shift snapping rounds axis travel to the native 0.1 m interval', async () => {
  const r = await gesture('snap');
  assert.ok(Math.abs(r.moved[0] * 10 - Math.round(r.moved[0] * 10)) < 1e-8);
});
test('cancel restores the pose after held input and closes the edit', async () => {
  const r = await gesture('cancel');
  assert.ok(r.moved[0] > 0.01);
  assert.deepEqual(r.restored, [0, 0, 0]);
  assert.equal(r.commits, 1);
});
test('oblique rotation follows the projected ring and rotates only the world axis', async () => {
  const r = await gesture('rotate');
  assert.equal(r.acquired, true);
  assert.ok(Math.abs(r.q[2]) > 0.02);
  assert.ok(Math.abs(r.q[0]) + Math.abs(r.q[1]) < 1e-7);
  assert.ok(Math.abs(r.q.reduce((s, v) => s + v * v, 0) - 1) < 1e-9);
});

test('position-axis conversion stays frozen across multiple projected motions', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { worldScale, project } = await import('/src/viewport/geometry.ts'),
      { createPrimitive } = await import('/src/scene.ts');
    const node = createPrimitive('box', 'Axis trace'),
      doc = { root: new T.Group(), nodes: [node], joints: [], files: [], warnings: [] };
    node.object.position.set(0, 0, 0);
    doc.root.add(node.object);
    const camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -4, 2.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const canvas = document.createElement('canvas'),
      g = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => {} });
    g.update(camera, doc, node.id, 'position', 'world', 900, 600);
    const origin = new T.Vector3(),
      scale = worldScale(origin, camera, 600),
      center = project(origin, camera, 900, 600),
      tip = project(new T.Vector3(scale, 0, 0), camera, 900, 600),
      dx = tip[0] - center[0],
      dy = tip[1] - center[1],
      length = Math.hypot(dx, dy),
      start = [center[0] + dx * 0.75, center[1] + dy * 0.75];
    g.begin(start);
    const actual = [];
    for (const travel of [12, 40, 85, 130, 55, -20]) {
      g.move([start[0] + (dx / length) * travel, start[1] + (dy / length) * travel]);
      actual.push(node.object.position.x);
    }
    g.end(true);
    g.dispose();
    return {
      actual,
      expected: [12, 40, 85, 130, 55, -20].map((travel) => (travel * scale) / length),
    };
  });
  for (let i = 0; i < r.actual.length; i++) assert.ok(Math.abs(r.actual[i] - r.expected[i]) < 1e-9);
});
