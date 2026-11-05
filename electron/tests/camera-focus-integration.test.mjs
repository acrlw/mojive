import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, page, base;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-camera-focus-integration',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
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
  await page.goto(base + '/tests/harness/camera-review.html');
  await page.waitForFunction(() => window.cameraReview?.api?.current);
  await page.waitForTimeout(50);
  await page.evaluate(async () => {
    const f = window.cameraReview,
      T = await import('/node_modules/three/build/three.module.js'),
      { createPrimitive } = await import('/src/scene.ts');
    const body = new T.Group();
    body.position.set(2, 0, 0.6);
    f.doc.root.add(body);
    const own = createPrimitive('box', 'Body geometry');
    body.add(own.object);
    own.parentId = 'focus-body';
    own.object.position.set(0, 0, 0);
    own.object.scale.set(0.3, 0.2, 0.4);
    const child = new T.Group();
    child.position.set(12, 0, 0);
    body.add(child);
    const tip = createPrimitive('box', 'Distant child');
    child.add(tip.object);
    tip.parentId = 'focus-child';
    tip.object.position.set(0, 0, 0);
    f.doc.nodes.push(
      { id: 'focus-body', name: 'Focus body', kind: 'link', object: body, visible: true },
      own,
      {
        id: 'focus-child',
        name: 'Child body',
        kind: 'link',
        object: child,
        parentId: 'focus-body',
        visible: true,
      },
      tip,
    );
    f.refresh();
    window.focusClock = performance.now();
    Object.defineProperty(performance, 'now', {
      configurable: true,
      value: () => window.focusClock,
    });
  });
});
const view = () => page.evaluate(() => window.cameraReview.api.current.getCameraView());
const advance = async (ms) => {
  await page.evaluate((ms) => (window.focusClock += ms), ms);
  return view();
};
const close = (a, b, tolerance = 2e-6) => {
  assert.equal(a.length, b.length);
  a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < tolerance, `${a} != ${b}`));
};
const pose = (a, b) => {
  for (const k of ['position', 'target', 'up']) close(a[k], b[k]);
};
async function focus(id) {
  await page.evaluate((id) => window.cameraReview.api.current.focus(id), id);
}
async function drag(dx, dy) {
  const r = await page.getByTestId('viewport-canvas').boundingBox(),
    x = r.x + r.width * 0.8,
    y = r.y + r.height * 0.75;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy);
  await page.mouse.up();
}

test('body focus uses own finite geometry, elevated eye-to-center direction and native sampled travel', async () => {
  const start = await view();
  const expected = await page.evaluate(async (start) => {
    const { CameraFocusTransition } = await import('/src/viewport/camera-focus.ts'),
      { sceneFocusBounds } = await import('/src/viewport/focus-bounds.ts'),
      curve = new CameraFocusTransition(
        start,
        sceneFocusBounds(window.cameraReview.doc, 'focus-body'),
      );
    return [0, 0.09, 0.18, 0.27, 0.36].map((t) => curve.sample(t));
  }, start);
  await focus('focus-body');
  pose(await view(), start);
  for (let i = 1; i <= 4; i++) {
    const actual = await advance(90);
    close(actual.position, expected[i].position);
    close(actual.target, expected[i].target);
    await page.waitForTimeout(15);
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/focus-body-${i * 90}ms.png` });
  }
  const end = await view();
  close(end.target, [2, 0, 0.6]);
  const direction = end.position.map((v, i) => v - end.target[i]),
    length = Math.hypot(...direction);
  assert.ok(direction[2] / length >= 0.5 - 1e-6);
  assert.ok(length < 2, 'far child does not enlarge this body framing');
  assert.deepEqual(errors, []);
});

test('focus interruption starts from the visible sample and does not resume abandoned movement', async () => {
  await focus('focus-body');
  const before = await advance(120);
  await drag(28, -14);
  const after = await view();
  close(after.target, before.target);
  assert.notDeepEqual(after.position, before.position);
  pose(await advance(1000), after);
  await focus('focus-child');
  const secondStart = await view();
  pose(secondStart, after);
  await advance(360);
  close((await view()).target, [14, 0, 0.6]);
  assert.deepEqual(errors, []);
});

test('scene viewing pauses a focus animation and explicit return resumes the saved editor motion', async () => {
  await focus('focus-body');
  const saved = await advance(90);
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.useSceneCamera(f.cameraId);
  });
  await advance(280);
  await page.evaluate(() => window.cameraReview.api.current.useSceneCamera(null));
  const returned = await advance(280);
  pose(returned, saved);
  await advance(270);
  close((await view()).target, [2, 0, 0.6]);
  assert.deepEqual(errors, []);
});

test('joint focus frames scalar stroke and keeps a hinge axis away from axial and edge-on views', async () => {
  const result = await page.evaluate(async () => {
    const f = window.cameraReview,
      { createNativeJointScene } = await import('/src/SceneCatalog.ts');
    const sample = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text());
    f.doc.root.add(sample.root);
    f.doc.nodes.push(...sample.nodes);
    f.doc.joints.push(...sample.joints);
    f.refresh();
    return sample.joints
      .filter((j) => ['hinge', 'slide', 'revolute', 'prismatic'].includes(j.type))
      .map((j) => ({ id: j.id, type: j.type }));
  });
  const hinge = result.find((j) => ['hinge', 'revolute'].includes(j.type)),
    slide = result.find((j) => ['slide', 'prismatic'].includes(j.type));
  for (const joint of [hinge, slide]) {
    const planned = await page.evaluate(async (id) => {
      const { jointFocusPlan } = await import('/src/viewport/joint-focus.ts');
      return jointFocusPlan(
        window.cameraReview.doc,
        id,
        window.cameraReview.api.current.getCameraView(),
      );
    }, joint.id);
    await focus(joint.id);
    await advance(360);
    const actual = await view();
    close(actual.target, planned.center);
    const direction = actual.position.map((v, i) => v - actual.target[i]),
      length = Math.hypot(...direction);
    close(
      direction.map((v) => v / length),
      planned.direction,
    );
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/focus-${joint.type}-settled.png` });
  }
  assert.deepEqual(errors, []);
});

test('actual workspace tree double-click moves through intermediate framing and retains selected entity', async () => {
  const app = await browser.newPage({ viewport: { width: 1440, height: 960 } }),
    appErrors = [];
  app.on('pageerror', (e) => appErrors.push(e.message));
  await app.addInitScript(() =>
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    ),
  );
  await app.goto(base);
  const row = app.locator('[role=treeitem][data-tooltip="01_revolute_anchor"]');
  await row.waitFor();
  // Hold the animation clock while screenshots wait for compositor frames.
  await app.evaluate(() => {
    window.focusClock = performance.now();
    Object.defineProperty(performance, 'now', {
      configurable: true,
      value: () => window.focusClock,
    });
  });
  const canvas = app.getByTestId('viewport-canvas'),
    before = await canvas.screenshot();
  await row.dblclick();
  await app.evaluate(() => {
    window.focusClock += 90;
  });
  await app.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  const mid = await canvas.screenshot({
    path: '../output/electron/verification/tree-focus-intermediate.png',
  });
  await app.evaluate(() => {
    window.focusClock += 400;
  });
  await app.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  const end = await canvas.screenshot({
    path: '../output/electron/verification/tree-focus-settled.png',
  });
  assert.equal(await row.getAttribute('aria-selected'), 'true');
  assert.notDeepEqual(mid, before);
  assert.notDeepEqual(end, mid);
  assert.deepEqual(appErrors, []);
  await app.close();
});

test('resizing a focused free camera preserves pixel aspect and vertical magnification', async () => {
  await focus('focus-body');
  await advance(360);
  const before = await view();
  await page.setViewportSize({ width: 1680, height: 900 });
  await page.waitForTimeout(60);
  const after = await view();
  close(after.position, before.position);
  close(after.target, before.target);
  assert.ok(
    Math.abs(
      after.normalizedFocal[0] * after.resolution[0] -
        before.normalizedFocal[0] * before.resolution[0],
    ) < 1e-6,
  );
  assert.ok(Math.abs(after.normalizedFocal[1] - before.normalizedFocal[1]) < 1e-10);
  await page.setViewportSize({ width: 1380, height: 900 });
});

test('view cube uses selected sphere quartic travel and Frame All uses its separate scene framing path', async () => {
  await page.evaluate(() => window.cameraReview.select('focus-body'));
  await page.waitForTimeout(30);
  const start = await view(),
    data = await page.evaluate(async (start) => {
      const T = await import('/node_modules/three/build/three.module.js'),
        { cameraForTransition } = await import('/src/viewport/camera-transition.ts'),
        { CameraFocusTransition } = await import('/src/viewport/camera-focus.ts'),
        { sceneNodeBounds } = await import('/src/viewport/focus-bounds.ts'),
        { viewCubeLayout } = await import('/src/viewport/viewcube.ts'),
        camera = cameraForTransition(start, start.resolution[0] / start.resolution[1]),
        right = new T.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).toArray(),
        up = new T.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).toArray(),
        forward = camera.getWorldDirection(new T.Vector3()).toArray(),
        ball = viewCubeLayout(right, up, forward).find(
          (ball) => ball.axis === 2 && ball.sign === 1,
        ),
        bounds = sceneNodeBounds(window.cameraReview.doc, 'focus-body'),
        pitch = (89.9 * Math.PI) / 180,
        curve = new CameraFocusTransition(start, bounds, {
          duration: 0.35,
          easing: 'ease_out_quart',
          margin: 1.15,
          sphereMargin: 1.2,
          sphereRadius: Math.hypot(...bounds.half),
          eyeDirection: [0, -Math.cos(pitch), Math.sin(pitch)],
        });
      return { ball, middle: curve.sample(0.175), end: curve.end };
    }, start);
  const box = await page.getByTestId('view-cube').boundingBox();
  await page.mouse.click(box.x + data.ball.x, box.y + data.ball.y);
  close((await advance(175)).position, data.middle.position);
  close((await advance(175)).target, data.end.target);
  const sceneStart = await view(),
    expected = await page.evaluate(async (start) => {
      const { CameraFocusTransition } = await import('/src/viewport/camera-focus.ts'),
        { sceneFocusBounds } = await import('/src/viewport/focus-bounds.ts');
      return new CameraFocusTransition(start, sceneFocusBounds(window.cameraReview.doc), {
        frameScene: true,
      }).end;
    }, sceneStart);
  await focus(undefined);
  await advance(350);
  const all = await view();
  close(all.position, expected.position);
  close(all.target, expected.target);
  assert.deepEqual(errors, []);
});

for (const pilot of [false, true])
  test(`${pilot ? 'Pilot' : 'View'} rejects node and joint focus during entry and after settling without edits`, async () => {
    const start = await view(),
      ids = await page.evaluate(async (pilot) => {
        const f = window.cameraReview,
          { readCameraView } = await import('/src/scene-camera.ts'),
          authored = readCameraView(f.doc.nodes.find((node) => node.id === f.cameraId));
        f.api.current.useSceneCamera(f.cameraId, pilot);
        return {
          camera: f.cameraId,
          joint: f.doc.joints[0].id,
          authored,
          begins: f.begins,
          commits: f.commits,
        };
      }, pilot);
    await focus('focus-body');
    await focus(ids.joint);
    pose(await view(), start);
    const middle = await advance(90),
      eased = 1 - (1 - 90 / 280) ** 3;
    close(
      middle.position,
      start.position.map((value, index) => value + (ids.authored.position[index] - value) * eased),
    );
    await advance(190);
    const settled = await view();
    await focus('focus-body');
    await focus(ids.joint);
    await advance(500);
    assert.deepEqual(await view(), settled);
    const state = await page.evaluate(() => {
      const f = window.cameraReview;
      return { state: f.state, begins: f.begins, commits: f.commits };
    });
    assert.deepEqual(state, {
      state: { id: ids.camera, pilot },
      begins: ids.begins,
      commits: ids.commits,
    });
    await page.evaluate(() => window.cameraReview.api.current.useSceneCamera(null));
    await advance(280);
    pose(await view(), start);
    assert.deepEqual(errors, []);
  });

test('actual tree double-click preserves active View and Pilot without moving the scene camera', async () => {
  const app = await browser.newPage({ viewport: { width: 1440, height: 960 } }),
    appErrors = [];
  app.on('pageerror', (e) => appErrors.push(e.message));
  await app.addInitScript(() =>
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    ),
  );
  await app.goto(base);
  const row = app.locator('[role=treeitem][data-tooltip="01_revolute_anchor"]');
  await row.waitFor();
  await row.click();
  for (const label of ['View Camera', 'Pilot camera']) {
    await app
      .locator('.activity-rail')
      .getByRole('button', { name: 'Cameras', exact: true })
      .click();
    await app
      .locator('.camera-entry')
      .first()
      .getByRole('button', { name: label, exact: true })
      .click();
    await app.locator('.viewport-content.has-scene-camera').waitFor();
    await app.waitForTimeout(330);
    await app.locator('.activity-rail').getByRole('button', { name: 'Scene', exact: true }).click();
    const canvas = app.getByTestId('viewport-canvas'),
      before = await canvas.screenshot();
    await row.dblclick();
    await app.waitForTimeout(400);
    assert.equal(await app.locator('.viewport-content.has-scene-camera').count(), 1);
    assert.equal(await row.getAttribute('aria-selected'), 'true');
    assert.deepEqual(await canvas.screenshot(), before);
    await app.locator('.scene-camera-return').getByRole('button').click();
    await app.waitForTimeout(300);
    assert.equal(await app.locator('.viewport-content.has-scene-camera').count(), 0);
  }
  assert.deepEqual(appErrors, []);
  await app.close();
});

test('controlled projection changes notify once while interrupting an exact focus and reversing a transition', async () => {
  const changes = () => page.evaluate(() => window.cameraReview.projectionChanges);
  assert.deepEqual(await changes(), [], 'initial framing does not change projection');
  await focus('focus-body');
  await advance(90);
  assert.deepEqual(await changes(), [], 'ordinary focus does not echo the current mode');
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.setProjection('orthographic');
    f.api.current.setProjection('orthographic');
  });
  assert.deepEqual(await changes(), ['orthographic']);
  await advance(90);
  await page.waitForTimeout(25);
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.setProjection('perspective');
    f.api.current.setProjection('perspective');
  });
  assert.deepEqual(await changes(), ['orthographic', 'perspective']);
  await advance(300);
  await page.waitForTimeout(30);
  assert.equal((await view()).projection, 'perspective');
  assert.equal(await page.evaluate(() => window.cameraReview.projection), 'perspective');
  await focus('focus-body');
  await advance(360);
  await page.waitForTimeout(20);
  assert.deepEqual(await changes(), ['orthographic', 'perspective']);
  assert.deepEqual(errors, []);
});
