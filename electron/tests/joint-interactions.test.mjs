import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: path.resolve('.'),
    server: { host: '127.0.0.1', port: 5194 },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  server.middlewares.use('/__joint_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Joint gestures</title>');
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__joint_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('range, outward ticks and precision placement match independent Python production fixtures', async () => {
  const fixtures = JSON.parse(
    await readFile(new URL('./joint-native-reference.json', import.meta.url), 'utf8'),
  );
  const errors = await page.evaluate(async (fixtures) => {
    const THREE = await import('/node_modules/three/build/three.module.js'),
      { projectJoint } = await import('/src/viewport/joint_interaction.ts');
    const errors = [];
    for (const [i, s] of fixtures.entries()) {
      const camera = new THREE.PerspectiveCamera();
      camera.position.fromArray(s.eye);
      camera.matrixWorldInverse.set(...s.view.flat());
      camera.matrixWorld.copy(camera.matrixWorldInverse).invert();
      camera.matrix.copy(camera.matrixWorld);
      camera.matrixAutoUpdate = false;
      camera.projectionMatrix.set(...s.projection.flat());
      const j = {
          id: 'joint',
          name: 'joint',
          type: s.kind === 'hinge' ? 'revolute' : 'prismatic',
          lower: s.lower,
          upper: s.upper,
          value: s.value,
        },
        p = projectJoint(
          j,
          new THREE.Vector3(...s.origin),
          new THREE.Vector3(...s.axis),
          s.scale,
          camera,
          900,
          600,
        ),
        check = (a, b, name) => {
          if (a == null || b == null) {
            if (a !== b) errors.push(`${i} ${name}: null mismatch`);
            return;
          }
          const aa = Array.isArray(a) ? a.flat(4) : [a],
            bb = Array.isArray(b) ? b.flat(4) : [b];
          if (aa.length !== bb.length)
            errors.push(`${i} ${name}: length ${aa.length} != ${bb.length}`);
          else if (aa.some((v, k) => Math.abs(v - bb[k]) > 1e-5))
            errors.push(`${i} ${name}: max ${Math.max(...aa.map((v, k) => Math.abs(v - bb[k])))}`);
        };
      check(p.alpha, s.expected.alpha, 'alpha');
      if (s.kind === 'hinge') {
        check(p.allowed, s.expected.allowed, 'allowed');
        check(p.complement.length ? p.complement : null, s.expected.complement, 'complement');
        for (const kind of ['current', 'lower', 'upper']) {
          const tick = p.ticks.find((t) => t.kind === kind);
          check(tick ? [tick.start, tick.end] : null, s.expected[kind + '_tick'], kind);
        }
      } else {
        check(p.lower, s.expected.lower, 'lower');
        check(p.upper, s.expected.upper, 'upper');
        check(p.current, s.expected.current, 'current');
      }
      for (const key of ['start', 'end', 'current', 'source', 'rect'])
        check(p.rail?.[key] ?? null, s.rail?.[key] ?? null, 'rail ' + key);
    }
    return errors;
  }, fixtures);
  assert.deepEqual(errors, []);
});
test('native dwell waits0.50s, tolerates0.12s gaps and retains0.30s reveal grace', async () => {
  const values = await page.evaluate(async () => {
    const { JointDwell } = await import('/src/viewport/joint_interaction.ts');
    const d = new JointDwell();
    return [
      d.update('j', true, true, false, false, 1),
      d.update('j', true, true, false, false, 1.49),
      d.update('j', true, true, false, false, 1.5),
      d.update('j', true, false, false, false, 1.61),
      d.update('j', true, false, false, false, 1.74),
      d.update('j', true, false, false, false, 1.92),
    ];
  });
  assert.deepEqual(values, [false, false, true, true, true, false]);
});
async function interaction(kind) {
  return page.evaluate(async (kind) => {
    const THREE = await import('/node_modules/three/build/three.module.js'),
      { loadUrdf, setJoint, disposeDocument } = await import('/src/scene.ts'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { projectJoint } = await import('/src/viewport/joint_interaction.ts'),
      { worldScale } = await import('/src/viewport/geometry.ts');
    const doc = await loadUrdf(
        '<robot name="small"><link name="a"/><link name="b"/><joint name="hinge" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/><limit lower="-0.05" upper="0.05"/></joint></robot>',
      ),
      joint = doc.joints[0];
    setJoint(doc, joint.id, 0);
    doc.root.updateMatrixWorld(true);
    const camera = new THREE.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(2, -3, 2);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const origin = new THREE.Vector3(),
      scale = worldScale(origin, camera, 600),
      projection = projectJoint(joint, origin, new THREE.Vector3(0, 0, 1), scale, camera, 900, 600),
      canvas = document.createElement('canvas');
    document.body.append(canvas);
    let begins = 0,
      changes = 0,
      commits = 0;
    const gizmo = new GizmoOverlay(canvas, {
      onBegin: () => begins++,
      onChange: () => changes++,
      onCommit: () => commits++,
      invalidate: () => {},
    });
    gizmo.update(camera, doc, joint.id, 'joint', 'world', 900, 600);
    let beforeRelease, acquired;
    if (kind === 'limit' || kind === 'cancel-limit') {
      const tick = projection.ticks.find((t) => t.kind === 'upper');
      gizmo.hoverAt(tick.end);
      acquired = gizmo.begin(tick.end);
      beforeRelease = joint.value;
      if (kind === 'cancel-limit') gizmo.hoverAt([-100, -100]);
      gizmo.end();
    } else {
      gizmo.hoverAt(projection.current);
      await new Promise((r) => setTimeout(r, 530));
      gizmo.update(camera, doc, joint.id, 'joint', 'world', 900, 600);
      const rail = projection.rail,
        point = [rail.start[0] + (rail.end[0] - rail.start[0]) * 0.75, rail.start[1]];
      gizmo.hoverAt(point);
      acquired = gizmo.begin(point);
      gizmo.move([rail.end[0] + 100, rail.end[1]], false, false);
      beforeRelease = joint.value;
      gizmo.end(kind === 'cancel-rail');
    }
    const result = { acquired, beforeRelease, value: joint.value, begins, changes, commits };
    gizmo.dispose();
    canvas.remove();
    disposeDocument(doc);
    return result;
  }, kind);
}
test('outer endpoint tick applies the authored limit only on release as one transaction', async () => {
  const r = await interaction('limit');
  assert.equal(r.acquired, true);
  assert.equal(r.beforeRelease, 0);
  assert.equal(r.value, 0.05);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
});
test('endpoint release outside the same tick leaves joint unchanged without a transaction', async () => {
  const r = await interaction('cancel-limit');
  assert.equal(r.acquired, true);
  assert.equal(r.value, 0);
  assert.equal(r.begins, 0);
  assert.equal(r.commits, 0);
});
test('collapsed-range dwell rail maps the complete authored range and clamps outside', async () => {
  const r = await interaction('rail');
  assert.equal(r.acquired, true);
  assert.equal(r.value, 0.05);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
});
test('cancelled precision-rail drag restores the original joint value in one transaction', async () => {
  const r = await interaction('cancel-rail');
  assert.equal(r.acquired, true);
  assert.equal(r.beforeRelease, 0.05);
  assert.equal(r.value, 0);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
});
