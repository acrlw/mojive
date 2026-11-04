import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, page, url;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-scene-camera-transitions',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  url = `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/camera-review.html`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1380, height: 900 } });
  page.on('pageerror', (e) => errors.push(e.message));
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  errors.length = 0;
  await page.goto(url);
  await page.waitForFunction(() => window.cameraReview?.api.current);
  await page.waitForTimeout(80);
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { createPrimitive } = await import('/src/scene.ts'),
      { updateCameraView } = await import('/src/scene-camera.ts');
    updateCameraView(f.doc, f.cameraId, {
      position: [4, -2, 3],
      target: [1, 1, 0.3],
      up: [0.7, 0.2, 1],
      fov: 74,
      resolution: [740, 900],
    });
    const b = createPrimitive('camera', 'Second camera');
    f.doc.root.add(b.object);
    f.doc.nodes.push(b);
    f.secondCameraId = b.id;
    updateCameraView(f.doc, b.id, {
      position: [-3, 6, 4],
      target: [-2, 1, 0.8],
      up: [1, 0, 0.4],
      projection: 'orthographic',
      orthoHeight: 8,
      resolution: [740, 900],
    });
    f.refresh();
    window.transitionClock = performance.now();
    Object.defineProperty(performance, 'now', {
      configurable: true,
      value: () => window.transitionClock,
    });
  });
});
const view = () => page.evaluate(() => window.cameraReview.api.current.getCameraView());
const advance = async (ms) => {
  await page.evaluate((ms) => (window.transitionClock += ms), ms);
  return view();
};
const choose = async (id, pilot = false) =>
  page.evaluate(
    ({ id, pilot }) => {
      const f = window.cameraReview;
      f.api.current.useSceneCamera(
        id === 'a' ? f.cameraId : id === 'b' ? f.secondCameraId : null,
        pilot,
      );
    },
    { id, pilot },
  );
function near(a, b, eps = 1e-7) {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < eps, `${a} != ${b}`));
}
function equalPose(a, b) {
  near(a.position, b.position);
  near(a.target, b.target);
  near(a.up, b.up);
  assert.ok(Math.abs(a.fov - b.fov) < 1e-8);
}
const goal = (id) =>
  page.evaluate(async (id) => {
    const f = window.cameraReview,
      { readCameraView } = await import('/src/scene-camera.ts');
    return readCameraView(
      f.doc.nodes.find((n) => n.id === (id === 'a' ? f.cameraId : f.secondCameraId)),
    );
  }, id);
function orbit(before, dx, dy) {
  const o = before.position.map((v, i) => v - before.target[i]),
    d = Math.hypot(...o),
    yaw = Math.atan2(o[1], o[0]) - (dx * 0.35 * Math.PI) / 180,
    pitch =
      (Math.max(-89.9, Math.min(89.9, (Math.asin(o[2] / d) * 180) / Math.PI + dy * 0.35)) *
        Math.PI) /
      180;
  return before.target.map(
    (v, i) =>
      v +
      d * [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)][i],
  );
}
async function pointer(dx, dy, release = true) {
  const b = await page.getByTestId('viewport-canvas').boundingBox(),
    x = b.x + b.width * 0.74,
    y = b.y + b.height * 0.6;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy);
  if (release) await page.mouse.up();
}

test('Editor to scene uses native 280ms cubic handoff with sampled pose and optics, not endpoint-only capture', async () => {
  const start = await view(),
    target = await goal('a');
  await choose('a');
  equalPose(await view(), start);
  for (const elapsed of [70, 140, 210, 280]) {
    const actual = await advance(70),
      t = 1 - (1 - elapsed / 280) ** 3;
    near(
      actual.position,
      start.position.map((v, i) => v + (target.position[i] - v) * t),
    );
    assert.ok(Math.abs(actual.fov - (start.fov + (target.fov - start.fov) * t)) < 1e-7);
    await page.waitForTimeout(20);
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/camera-handoff-${elapsed}ms.png` });
  }
  equalPose(await view(), target);
  assert.deepEqual(errors, []);
});

test('scene switching and returning to Editor interrupt at the rendered view while preserving the independent editor state', async () => {
  const editor = await view(),
    a = await goal('a'),
    b = await goal('b');
  await choose('a');
  const intermediate = await advance(100);
  await choose('b');
  equalPose(await view(), intermediate);
  const mixed = await advance(140);
  near(
    mixed.position,
    intermediate.position.map((v, i) => v + (b.position[i] - v) * 0.875),
  );
  assert.ok(mixed.projectionBlend > 0 && mixed.projectionBlend < 1);
  await choose(null);
  equalPose(await view(), mixed);
  await advance(280);
  assert.deepEqual(await view(), editor);
  assert.deepEqual(await goal('a'), a);
  assert.deepEqual(await goal('b'), b);
  assert.deepEqual(errors, []);
});

test('a moving target is sampled each frame and View/Pilot mode changes do not restart the same handoff', async () => {
  const start = await view();
  await choose('a');
  await advance(70);
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { updateCameraView } = await import('/src/scene-camera.ts');
    updateCameraView(f.doc, f.cameraId, { position: [6, -4, 5], target: [1, 0, 0.5] });
    f.refresh();
  });
  await choose('a', true);
  const actual = await advance(70),
    target = await goal('a');
  near(
    actual.position,
    start.position.map((v, i) => v + (target.position[i] - v) * 0.875),
  );
  await advance(140);
  equalPose(await view(), target);
  assert.deepEqual(errors, []);
});

test('orbit interrupts View from the displayed pivot and never resumes the abandoned handoff', async () => {
  await choose('a');
  const before = await advance(110);
  await pointer(24, 8);
  const after = await view();
  near(after.target, before.target);
  near(after.position, orbit(before, 24, 8));
  assert.equal(await page.evaluate(() => window.cameraReview.state.id), null);
  await advance(500);
  assert.deepEqual(await view(), after);
  assert.deepEqual(errors, []);
});

test('Pilot interrupts from the displayed view in one authoring transaction; Escape restores the authored camera', async () => {
  const original = await goal('a'),
    editor = await view();
  await choose('a', true);
  const before = await advance(100);
  await pointer(32, 12, false);
  const moved = await view();
  near(moved.target, before.target);
  near(moved.position, orbit(before, 32, 12));
  await page.keyboard.press('Escape');
  await page.mouse.up();
  equalPose(await view(), original);
  assert.equal(await page.evaluate(() => window.cameraReview.begins), 1);
  assert.equal(await page.evaluate(() => window.cameraReview.commits), 1);
  await choose(null);
  await advance(280);
  assert.deepEqual(await view(), editor);
  assert.deepEqual(errors, []);
});
