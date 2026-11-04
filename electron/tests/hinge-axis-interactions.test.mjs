import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: process.cwd(),
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: { noDiscovery: true, include: [] },
    logLevel: 'error',
  });
  server.middlewares.use('/__hinge_axis', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><style>body{margin:0;background:#202529}canvas{display:block;width:900px;height:600px}</style>',
    );
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 900, height: 600 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__hinge_axis`);
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function run(eye, part, mode = 'commit', orthographic = false) {
  return page.evaluate(
    async ({ eye, part, mode, orthographic }) => {
      window.fixture?.dispose();
      const THREE = await import('/node_modules/three/build/three.module.js'),
        { loadUrdf, setJoint, disposeDocument } = await import('/src/scene.ts'),
        { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
        { rotationAxisGuide } = await import('/src/viewport/drag_guides.ts'),
        { projectJoint, jointHit } = await import('/src/viewport/joint_interaction.ts'),
        { project, worldScale, inside, pathDistance, ringAlpha, axisAlpha } = await import(
          '/src/viewport/geometry.ts'
        );
      const doc = await loadUrdf(
          '<robot name="hinge"><link name="a"/><link name="b"/><joint name="hinge" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5"/></joint></robot>',
        ),
        joint = doc.joints[0],
        origin = new THREE.Vector3(),
        axis = new THREE.Vector3(0, 0, 1);
      if (part === 'limit-crossing') joint.lower = Math.atan2(eye[1], eye[0]);
      setJoint(doc, joint.id, 0.2);
      doc.root.updateMatrixWorld(true);
      const camera = orthographic
        ? new THREE.OrthographicCamera(-3, 3, 2, -2, 0.01, 100)
        : new THREE.PerspectiveCamera(42, 1.5, 0.01, 100);
      camera.up.set(0, 0, 1);
      camera.position.fromArray(eye);
      if (Math.hypot(eye[0], eye[1]) < 1e-9) camera.up.set(0, 1, 0);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();
      const guide = rotationAxisGuide(camera, origin, axis, 900, 600, { hinge: true }),
        center = project(origin, camera, 900, 600),
        plus = project(axis, camera, 900, 600),
        length = Math.hypot(plus[0] - center[0], plus[1] - center[1]),
        direction = [(plus[0] - center[0]) / length, (plus[1] - center[1]) / length];
      const canvas = document.createElement('canvas');
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
      const update = () => gizmo.update(camera, doc, joint.id, 'joint', 'world', 900, 600);
      update();
      window.fixture = {
        dispose: () => {
          gizmo.dispose();
          canvas.remove();
          disposeDocument(doc);
        },
      };
      if (part === 'face-on')
        return {
          guideAbsent: guide === null,
          centerHit: gizmo.hoverAt(center),
          ringPresent:
            projectJoint(joint, origin, axis, worldScale(origin, camera, 600), camera, 900, 600)
              ?.alpha === 1,
        };
      const tip = guide.hitPolygon.reduce((best, p) =>
          p[0] * direction[0] + p[1] * direction[1] >
          best[0] * direction[0] + best[1] * direction[1]
            ? p
            : best,
        ),
        point =
          part === 'head'
            ? [tip[0] - direction[0] * 2, tip[1] - direction[1] * 2]
            : part === 'shaft'
              ? [center[0] * 0.65 + tip[0] * 0.35, center[1] * 0.65 + tip[1] * 0.35]
              : center;
      let unpainted = false,
        crossedLimit = false;
      if (part === 'limit-crossing') {
        const projection = projectJoint(
          joint,
          origin,
          axis,
          worldScale(origin, camera, 600),
          camera,
          900,
          600,
        );
        const endpoint = projection.ticks.find((tick) => tick.kind === 'lower').end;
        point.splice(0, 2, ...endpoint);
        crossedLimit =
          !!jointHit(point, projection).limit &&
          (inside(point, guide.hitPolygon) || pathDistance(point, guide.hitPolygon, true) <= 4);
      }
      if (part === 'gap') {
        const samples = Array.from({ length: 300 }, (_, i) => [
          center[0] + direction[0] * (i - 150),
          center[1] + direction[1] * (i - 150),
        ]);
        const gap = samples.find(
          (p) =>
            inside(p, guide.hitPolygon) &&
            !guide.polygons.some((poly) => inside(p, poly) || pathDistance(p, poly, true) < 0.5) &&
            !jointHit(
              p,
              projectJoint(joint, origin, axis, worldScale(origin, camera, 600), camera, 900, 600),
            ).hit,
        );
        if (!gap) throw Error('Native dashed interval missing from oblique test');
        point.splice(0, 2, ...gap);
        unpainted = true;
      }
      if (part === 'halo') {
        point[0] += direction[1] * 5;
        point[1] -= direction[0] * 5;
      }
      const hovered = gizmo.hoverAt(point),
        before = joint.value;
      if (mode === 'hover') return { hovered, point, unpainted };
      const acquired = gizmo.begin(point);
      gizmo.move(point);
      const stationary = joint.value;
      gizmo.move([point[0] + 28, point[1] + 16]);
      update();
      const moved = joint.value;
      gizmo.end(mode === 'cancel');
      const result = {
        hovered,
        acquired,
        before,
        stationary,
        moved,
        value: joint.value,
        begins,
        changes,
        commits,
        unpainted,
        crossedLimit,
        active: gizmo.active,
      };
      const view = origin.clone().sub(camera.position).normalize();
      result.ringAlpha = ringAlpha(axis.dot(view));
      result.axisAlpha = axisAlpha(axis.dot(view));
      return result;
    },
    { eye, part, mode, orthographic },
  );
}
for (const [name, eye, ortho] of [
  ['oblique', [2, -3, 2], false],
  ['edge-on', [2, -3, 0], false],
  ['fade-band', [2, -3, 0.002], false],
  ['orthographic', [2, -3, 2], true],
]) {
  test(`native hinge ${name} axis shaft, head and center activate rotation without a press jump`, async () => {
    for (const part of ['shaft', 'head', 'center']) {
      const result = await run(eye, part, 'commit', ortho);
      assert.equal(result.hovered, true, part);
      assert.equal(result.acquired, true, part);
      assert.equal(result.stationary, result.before, part);
      assert.ok(
        Number.isFinite(result.moved) && Math.abs(result.moved - result.before) > 0.01,
        `${part}: ${JSON.stringify(result)}`,
      );
      assert.equal(result.value, result.moved);
      assert.equal(result.begins, 1);
      assert.equal(result.commits, 1);
      assert.equal(result.active, false);
      if (name === 'edge-on') {
        assert.equal(result.ringAlpha, 0);
        assert.equal(result.axisAlpha, 1);
      }
    }
  });
}
test('native axis dashed gaps and four-point halo retain one hinge owner, with cancel restoring the joint', async () => {
  for (const part of ['gap', 'halo']) {
    const result = await run([2, -3, 2], part, 'cancel');
    assert.equal(result.hovered, true);
    assert.equal(result.acquired, true);
    assert.notEqual(result.moved, result.before);
    assert.equal(result.value, result.before);
    assert.equal(result.begins, 1);
    assert.equal(result.commits, 1);
    assert.equal(result.active, false);
    if (part === 'gap') assert.equal(result.unpainted, true);
  }
});
test('view-aligned axis disappears independently while its face-on hinge ring remains usable', async () => {
  const result = await run([0, 0, 5], 'face-on');
  assert.deepEqual(result, { guideAbsent: true, centerHit: false, ringPresent: true });
});
test('oblique and edge-on hovered hinge-axis production captures', async () => {
  for (const [name, eye, part] of [
    ['oblique', [2, -3, 2], 'gap'],
    ['edge-on', [2, -3, 0], 'center'],
  ]) {
    const result = await run(eye, part, 'hover');
    assert.equal(result.hovered, true);
    await page.screenshot({
      path: `../output/electron/verification/hinge-axis-hover-${name}.png`,
      clip: { x: 320, y: 170, width: 260, height: 260 },
    });
  }
});

test('hinge axis owns an overlapping outward limit tick instead of jumping to the endpoint', async () => {
  const result = await run([2, -3, 2], 'limit-crossing');
  assert.equal(
    result.crossedLimit,
    true,
    'Test starts in an outer limit tick and the original axis silhouette',
  );
  assert.equal(result.acquired, true);
  assert.equal(result.stationary, result.before);
  assert.ok(Math.abs(result.moved - result.before) > 0.01);
  assert.equal(result.value, result.moved);
  assert.equal(result.begins, 1);
  assert.equal(result.commits, 1);
});
