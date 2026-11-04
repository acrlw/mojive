import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page, url;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-camera-navigation-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  url = `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/camera-review.html`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1380, height: 900 }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => errors.push(error.message));
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  errors.length = 0;
  await page.goto(url);
  await page.waitForFunction(() => window.cameraReview?.api?.current);
  await page.waitForTimeout(100);
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { createPrimitive } = await import('/src/scene.ts'),
      { updateCameraView } = await import('/src/scene-camera.ts');
    updateCameraView(f.doc, f.cameraId, {
      position: [4, -2, 3],
      target: [1, 1, 0.3],
      up: [0.7, 0.2, 1],
      fov: 74,
    });
    const second = createPrimitive('camera', 'Second inspection camera');
    f.doc.nodes.push(second);
    f.doc.root.add(second.object);
    f.secondCameraId = second.id;
    updateCameraView(f.doc, second.id, {
      position: [-3, 6, 4],
      target: [-2, 1, 0.8],
      up: [1, 0, 0.4],
      projection: 'orthographic',
      orthoHeight: 8,
      resolution: [800, 600],
    });
    f.refresh();
  });
});
const view = () => page.evaluate(() => window.cameraReview.api.current.getCameraView());
async function scene(id, pilot = false) {
  await page.evaluate(
    ({ id, pilot }) => {
      const f = window.cameraReview;
      f.api.current.useSceneCamera(
        id === 'first' ? f.cameraId : id === 'second' ? f.secondCameraId : null,
        pilot,
      );
    },
    { id, pilot },
  );
  await page.waitForTimeout(320);
}
async function drag(dx, dy, button = 'left') {
  const box = await page.getByTestId('viewport-canvas').boundingBox(),
    x = box.x + box.width * 0.74,
    y = box.y + box.height * 0.6;
  await page.mouse.move(x, y);
  await page.mouse.down({ button });
  await page.mouse.move(x + dx, y + dy, { steps: 4 });
  await page.mouse.up({ button });
  await page.waitForTimeout(60);
}
function near(a, b, tolerance = 1e-9) {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < tolerance, `${a} != ${b}`);
}
function orbitExpected(before, dx, dy) {
  const delta = before.position.map((v, i) => v - before.target[i]),
    distance = Math.hypot(...delta),
    yaw = Math.atan2(delta[1], delta[0]) - (dx * 0.35 * Math.PI) / 180,
    pitch =
      (Math.max(
        -89.9,
        Math.min(89.9, (Math.asin(delta[2] / distance) * 180) / Math.PI + dy * 0.35),
      ) *
        Math.PI) /
      180;
  return before.target.map(
    (v, i) =>
      v +
      distance *
        [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)][i],
  );
}
for (const preset of ['front', 'top', 'bottom'])
  test(`editor ${preset} view restores its complete orbit after chained scene cameras and actual orbit drag`, async () => {
    await page.evaluate((preset) => window.cameraReview.api.current.view(preset), preset);
    await page.waitForTimeout(420);
    if (preset === 'top') {
      await page.evaluate(() => window.cameraReview.api.current.setProjection('orthographic'));
      await page.waitForTimeout(330);
    }
    const before = await view();
    if (preset === 'front')
      await page
        .getByTestId('viewport-canvas')
        .screenshot({ path: '../output/electron/verification/editor-camera-before-view.png' });
    const authored = await page.evaluate(async () => {
      const { readCameraView } = await import('/src/scene-camera.ts');
      const f = window.cameraReview;
      return [f.cameraId, f.secondCameraId].map((id) =>
        readCameraView(f.doc.nodes.find((n) => n.id === id)),
      );
    });
    await scene('first');
    await scene('second');
    await scene('first');
    await scene(null);
    const restored = await view();
    if (preset === 'front')
      await page
        .getByTestId('viewport-canvas')
        .screenshot({ path: '../output/electron/verification/editor-camera-after-return.png' });
    assert.deepEqual(
      restored,
      before,
      'Editor position, target, rendered up, projection and optics survive all scene-camera switches',
    );
    await drag(40, 12);
    const moved = await view();
    near(moved.target, before.target);
    near(moved.position, orbitExpected(before, 40, 12), 1e-8);
    assert.equal(await page.evaluate(() => window.cameraReview.state.id), null);
    assert.deepEqual(
      await page.evaluate(async () => {
        const { readCameraView } = await import('/src/scene-camera.ts');
        const f = window.cameraReview;
        return [f.cameraId, f.secondCameraId].map((id) =>
          readCameraView(f.doc.nodes.find((n) => n.id === id)),
        );
      }),
      authored,
      'View never authors the scene cameras',
    );
    assert.deepEqual(errors, []);
  });
test('first pan leaving a rolled scene camera adopts the displayed basis, film gate and pivot', async () => {
  const box = await page.getByTestId('viewport-canvas').boundingBox();
  await scene('second');
  const before = await view();
  const x = box.x + box.width * 0.74,
    y = box.y + box.height * 0.6;
  await page.mouse.move(x, y);
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(x + 32, y + 20);
  await page.mouse.up({ button: 'right' });
  const after = await view();
  const forward = before.target.map((v, i) => v - before.position[i]),
    length = Math.hypot(...forward);
  for (let i = 0; i < 3; i++) forward[i] /= length;
  const up = before.up,
    right = [
      forward[1] * up[2] - forward[2] * up[1],
      forward[2] * up[0] - forward[0] * up[2],
      forward[0] * up[1] - forward[1] * up[0],
    ];
  const gateY = Math.min(1, box.width / box.height / (before.resolution[0] / before.resolution[1]));
  const unit = before.orthoHeight / (box.height * gateY);
  near(
    after.target,
    before.target.map((v, i) => v - right[i] * 32 * unit + up[i] * 20 * unit),
    1e-8,
  );
  near(
    after.position.map((v, i) => v - after.target[i]),
    before.position.map((v, i) => v - before.target[i]),
    1e-8,
  );
  assert.deepEqual(errors, []);
});
test('pilot edits one scene camera while the original editor pose remains independent and restores on exit', async () => {
  const before = await view();
  await scene('first', true);
  const authored = await view(),
    counts = await page.evaluate(() => [window.cameraReview.begins, window.cameraReview.commits]);
  await drag(40, -12);
  const changed = await view();
  assert.notDeepEqual(changed.position, authored.position);
  assert.equal(await page.evaluate(() => window.cameraReview.state.pilot), true);
  const next = await page.evaluate(() => [window.cameraReview.begins, window.cameraReview.commits]);
  assert.deepEqual(
    next,
    counts.map((n) => n + 1),
  );
  await scene('second');
  await scene(null);
  assert.deepEqual(await view(), before);
  await drag(40, 12);
  near((await view()).position, orbitExpected(before, 40, 12), 1e-8);
  assert.deepEqual(errors, []);
});
test('removing the viewed camera restores the editor pose and projection without retaining scene-camera orbit state', async () => {
  const before = await view();
  await scene('second');
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { removeNode } = await import('/src/scene.ts');
    removeNode(f.doc, f.secondCameraId);
    f.refresh();
  });
  await page.waitForTimeout(60);
  assert.equal(await page.evaluate(() => window.cameraReview.state.id), null);
  assert.deepEqual(await view(), before);
  assert.deepEqual(errors, []);
});

for (const preset of ['oblique', 'top', 'bottom'])
  test(`Pilot and editor produce identical world-Z orbit poses from the same ${preset} view`, async () => {
    if (preset !== 'oblique') {
      await page.evaluate((preset) => window.cameraReview.api.current.view(preset), preset);
      await page.waitForTimeout(420);
    }
    if (preset === 'top') {
      await page.evaluate(() => window.cameraReview.api.current.setProjection('orthographic'));
      await page.waitForTimeout(330);
    }
    const initial = await view();
    await page.evaluate(() => {
      const f = window.cameraReview;
      f.api.current.alignCameraToView(f.cameraId);
    });
    const gestures = [
        [80, 0],
        [0, 32],
        [-28, -20],
        [44, 16],
      ],
      expected = [];
    for (const [dx, dy] of gestures) {
      await drag(dx, dy);
      expected.push(await view());
    }
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/editor-orbit-${preset}.png` });
    await scene('first', true);
    near((await view()).position, initial.position);
    near((await view()).up, initial.up);
    for (let i = 0; i < gestures.length; i++) {
      // Re-entering Pilot must not turn the previous screen-up into a new orbit pole.
      if (i === 2) {
        await scene('first');
        await scene('first', true);
      }
      await drag(...gestures[i]);
      const actual = await view();
      near(actual.position, expected[i].position, 1e-8);
      near(actual.target, expected[i].target, 1e-8);
      near(actual.up, expected[i].up, 1e-8);
      assert.equal(actual.fov, expected[i].fov);
    }
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/pilot-orbit-${preset}.png` });
    await scene(null);
    assert.deepEqual(
      await view(),
      expected.at(-1),
      'Pilot leaves the independent editor orbit intact',
    );
    assert.deepEqual(errors, []);
  });

test('Pilot preserves the authored rolled view until navigation and Escape restores it atomically', async () => {
  const before = await page.evaluate(async () => {
    const f = window.cameraReview,
      { readCameraView } = await import('/src/scene-camera.ts');
    return readCameraView(f.doc.nodes.find((n) => n.id === f.cameraId));
  });
  await scene('first', true);
  assert.deepEqual(await view(), before, 'Entering Pilot does not level an authored view');
  const initialCounts = await page.evaluate(() => [
    window.cameraReview.begins,
    window.cameraReview.commits,
  ]);
  const box = await page.getByTestId('viewport-canvas').boundingBox(),
    x = box.x + box.width * 0.74,
    y = box.y + box.height * 0.6;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 48, y + 20, { steps: 4 });
  assert.notDeepEqual((await view()).position, before.position);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await page.waitForTimeout(40);
  assert.deepEqual(
    await view(),
    before,
    'Cancel restores authored pose and roll, not the derived orbit frame',
  );
  assert.deepEqual(
    await page.evaluate(() => [window.cameraReview.begins, window.cameraReview.commits]),
    initialCounts.map((n) => n + 1),
  );
  await drag(40, -12);
  const after = await view();
  near(after.position, orbitExpected(before, 40, -12), 1e-8);
  const offset = after.position.map((v, i) => v - after.target[i]),
    horizontal = Math.hypot(offset[0], offset[1]),
    length = Math.hypot(...offset);
  near(
    after.up,
    [
      (-offset[2] * offset[0]) / (length * horizontal),
      (-offset[2] * offset[1]) / (length * horizontal),
      horizontal / length,
    ],
    1e-8,
  );
  assert.deepEqual(errors, []);
});

test('a new Pilot gesture adopts the current externally edited pose and pivot rather than stale entry coordinates', async () => {
  await scene('first', true);
  await drag(36, 12);
  const edited = await page.evaluate(async () => {
    const f = window.cameraReview,
      { updateCameraView, readCameraView } = await import('/src/scene-camera.ts');
    updateCameraView(f.doc, f.cameraId, {
      position: [-6, 3, 4],
      target: [-2, 1, 0.5],
      up: [0, 0, 1],
      fov: 63,
    });
    f.refresh();
    return readCameraView(f.doc.nodes.find((n) => n.id === f.cameraId));
  });
  await page.waitForTimeout(50);
  await drag(-44, 16);
  const after = await view();
  near(after.target, edited.target, 1e-8);
  near(after.position, orbitExpected(edited, -44, 16), 1e-8);
  assert.equal(after.fov, 63);
  assert.deepEqual(errors, []);
});

test('Pilot gestures match live-native roll and exact-pole adoption fixtures', async () => {
  const fixture = JSON.parse(
    await readFile(new URL('./fixtures/camera-orbit-native.json', import.meta.url), 'utf8'),
  );
  for (const sample of fixture.cases) {
    await scene(null);
    await page.evaluate(async (initial) => {
      const f = window.cameraReview,
        { updateCameraView } = await import('/src/scene-camera.ts');
      updateCameraView(f.doc, f.cameraId, {
        position: initial.eye,
        target: initial.target,
        up: initial.up,
        fov: initial.fovY,
        near: initial.near,
        far: initial.far,
        projection: initial.orthographic ? 'orthographic' : 'perspective',
        orthoHeight: initial.orthoHeight,
        resolution: [1700, 1000],
      });
      f.refresh();
    }, sample.initial);
    await scene('first', true);
    const box = await page.getByTestId('viewport-canvas').boundingBox(),
      x = box.x + box.width * 0.74,
      y = box.y + box.height * 0.6,
      { dx, dy } = sample.gesture;
    await page.mouse.move(x, y);
    await page.mouse.down();
    // Cross native click slop, then return to the requested accumulated delta.
    if (Math.hypot(dx, dy) <= 4) await page.mouse.move(x + 8, y);
    await page.mouse.move(x + dx, y + dy);
    await page.mouse.up();
    await page.waitForTimeout(20);
    const actual = await view();
    try {
      near(actual.position, sample.expected.eye, fixture.tolerance);
      near(actual.target, sample.expected.target, fixture.tolerance);
      near(actual.up, sample.expected.screenUp, fixture.tolerance);
      assert.ok(Math.abs(actual.fov - sample.expected.fovY) < fixture.tolerance);
    } catch (error) {
      error.message = `${sample.id}: ${error.message}`;
      throw error;
    }
  }
  assert.deepEqual(errors, []);
});

for (const projection of ['perspective', 'orthographic'])
  test(`Pilot ${projection} dolly preserves authored roll, cancels atomically, then orbits in world Z`, async () => {
    await page.evaluate(async (projection) => {
      const f = window.cameraReview,
        { updateCameraView } = await import('/src/scene-camera.ts');
      updateCameraView(f.doc, f.cameraId, { projection });
      f.refresh();
    }, projection);
    await scene('first', true);
    const before = await view();
    const box = await page.getByTestId('viewport-canvas').boundingBox();
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.6);
    await page.mouse.wheel(0, 80);
    await page.waitForTimeout(40);
    const moved = await view();
    near(moved.up, before.up, 1e-8);
    near(moved.target, before.target, 1e-8);
    const d = (v) => Math.hypot(...v.position.map((n, i) => n - v.target[i]));
    assert.ok(Math.abs(d(moved) / d(before) - Math.exp(0.096)) < 1e-8);
    if (projection === 'orthographic')
      assert.ok(Math.abs(moved.orthoHeight / before.orthoHeight - Math.exp(0.096)) < 1e-8);
    await page.getByTestId('viewport-canvas').focus();
    await page.keyboard.press('Escape');
    await page.waitForTimeout(20);
    assert.deepEqual(await view(), before);
    await drag(40, -12);
    const after = await view();
    near(after.position, orbitExpected(before, 40, -12), 1e-8);
    const o = after.position.map((v, i) => v - after.target[i]),
      h = Math.hypot(o[0], o[1]),
      length = Math.hypot(...o);
    near(
      after.up,
      [(-o[2] * o[0]) / (length * h), (-o[2] * o[1]) / (length * h), h / length],
      1e-8,
    );
    assert.deepEqual(errors, []);
  });
