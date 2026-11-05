import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, page;
const errors = [];
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-camera-test',
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1380, height: 900 }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/camera-review.html`,
  );
  await page.waitForLoadState('networkidle');
  await page.waitForFunction(() => window.cameraReview?.api?.current);
  await page.locator('[data-camera-properties]').waitFor();
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('camera projection, world pose, sensor optics and resolution survive scene roundtrip with unchanged child identity', async () => {
  const r = await page.evaluate(async () => {
    const { createDemo, createPrimitive } = await import('/src/scene.ts'),
      { updateCameraView, readCameraView, cameraFromView } = await import('/src/scene-camera.ts'),
      { encodeDocument, decodeDocument, saveWorkspace, loadWorkspace } = await import(
        '/src/editor.ts'
      );
    const doc = await createDemo(),
      node = createPrimitive('camera'),
      child = createPrimitive('box');
    const parent = doc.nodes.find((n) => n.kind === 'link');
    parent.object.add(node.object);
    node.parentId = parent.id;
    node.object.add(child.object);
    child.parentId = node.id;
    doc.nodes.push(node, child);
    const uuid = node.object.uuid;
    updateCameraView(doc, node.id, {
      position: [2, -5, 3],
      target: [0, 0, 1],
      up: [0, 0, 1],
      fov: 60,
      resolution: [1600, 900],
      near: 0.0000025,
      far: 500,
    });
    const focal = readCameraView(node).focalLength;
    updateCameraView(doc, node.id, { sensorWidth: 50 });
    const wider = readCameraView(node).fov;
    updateCameraView(doc, node.id, { projection: 'orthographic', orthoHeight: 6 });
    const copied = decodeDocument(encodeDocument(doc));
    const saved = loadWorkspace(saveWorkspace(doc, [], 10));
    const restored = copied.nodes.find((n) => n.id === node.id);
    const view = readCameraView(restored);
    const render = cameraFromView(view);
    let rejected = 0;
    for (const patch of [
      { near: -1 },
      { far: 0 },
      { resolution: [8192, 8192] },
      { target: view.position },
      { up: [0, 0, 0] },
    ]) {
      const before = JSON.stringify(encodeDocument(copied));
      try {
        updateCameraView(copied, node.id, patch);
      } catch {
        rejected += JSON.stringify(encodeDocument(copied)) === before ? 1 : 0;
      }
    }
    return {
      uuid: restored.object.uuid === uuid,
      type: restored.object.type,
      parent: restored.parentId === parent.id,
      child: copied.nodes.find((n) => n.id === child.id).object.parent === restored.object,
      focal,
      wider,
      view,
      renderHeight: render.top - render.bottom,
      saved: saved.doc.nodes.find((n) => n.id === node.id).object.type,
      rejected,
    };
  });
  assert.equal(r.type, 'OrthographicCamera');
  assert.equal(r.saved, r.type);
  assert.ok(r.uuid && r.parent && r.child);
  assert.ok(r.focal > 0 && r.wider > 60);
  assert.equal(r.renderHeight, 6);
  assert.deepEqual(r.view.resolution, [1600, 900]);
  assert.equal(r.rejected, 5);
  for (const [i, v] of [2, -5, 3].entries()) assert.ok(Math.abs(r.view.position[i] - v) < 1e-10);
});
test('native influence creates actual frustum edges and bounded light volumes without modifying scene data', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/.vite-camera-test/deps/three.js'),
      { entityInfluence } = await import('/src/viewport/influence.ts'),
      { readCameraView, cameraFromView } = await import('/src/scene-camera.ts');
    const f = window.cameraReview,
      node = f.doc.nodes.find((n) => n.id === f.cameraId),
      camera = cameraFromView({ ...readCameraView(node), position: [5, -5, 4], target: [0, 0, 0] });
    const helper = (object) => ({
      id: 'helper',
      name: 'Helper',
      object,
      kind: 'light',
      visible: true,
    });
    const point = new T.PointLight(0xffffff, 1, 2),
      spot = new T.SpotLight(0xffffff, 1, 100, Math.PI / 4),
      area = new T.RectAreaLight(0xffffff, 1, 4, 2);
    spot.position.set(1, 0, 1);
    spot.target.position.set(0, 0, 0);
    const a = entityInfluence(node, camera, 800, 600),
      b = entityInfluence(helper(point), camera, 800, 600),
      c = entityInfluence(helper(spot), camera, 800, 600),
      d = entityInfluence(helper(area), camera, 800, 600);
    node.object.visible = false;
    const hidden = entityInfluence(node, camera, 800, 600).length;
    node.object.visible = true;
    return {
      frustum: a[0].positions.length,
      point: b[0].positions.length,
      spot: c.map((x) => x.positions.length),
      area: d.map((x) => x.positions.length),
      hidden,
    };
  });
  assert.equal(r.frustum, 72);
  assert.equal(r.point, 48 * 3 * 6);
  assert.deepEqual(r.spot, [18, 56 * 6]);
  assert.deepEqual(r.area, [18, 24]);
  assert.equal(r.hidden, 0);
});
test('camera panel activates real scene view; navigation exits without authoring and pilot is one undo transaction', async () => {
  await page.screenshot({ path: '../output/electron/verification/camera-panel.png' });
  await page.evaluate(() =>
    window.cameraReview.api.current.useSceneCamera(window.cameraReview.cameraId),
  );
  await page.waitForFunction(() => window.cameraReview.state.id);
  const before = await page.evaluate(async () => {
    const { readCameraView } = await import('/src/scene-camera.ts');
    return readCameraView(
      window.cameraReview.doc.nodes.find((n) => n.id === window.cameraReview.cameraId),
    );
  });
  const canvas = page.locator('[data-testid=viewport-canvas]'),
    b = await canvas.boundingBox();
  await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width * 0.5 + 45, b.y + b.height * 0.5 + 10, { steps: 6 });
  await page.mouse.up();
  assert.equal(await page.evaluate(() => window.cameraReview.state.id), null);
  const after = await page.evaluate(async () => {
    const { readCameraView } = await import('/src/scene-camera.ts');
    return readCameraView(
      window.cameraReview.doc.nodes.find((n) => n.id === window.cameraReview.cameraId),
    );
  });
  assert.deepEqual(after, before);
  await page.evaluate(() =>
    window.cameraReview.api.current.useSceneCamera(window.cameraReview.cameraId, true),
  );
  const counts = await page.evaluate(() => [
    window.cameraReview.begins,
    window.cameraReview.commits,
  ]);
  await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width * 0.5 + 50, b.y + b.height * 0.5 - 20, { steps: 6 });
  await page.mouse.up();
  const moved = await page.evaluate(async () => {
    const { readCameraView } = await import('/src/scene-camera.ts');
    const f = window.cameraReview;
    return {
      view: readCameraView(f.doc.nodes.find((n) => n.id === f.cameraId)),
      begins: f.begins,
      commits: f.commits,
      state: f.state,
    };
  });
  assert.notDeepEqual(moved.view.position, before.position);
  assert.equal(moved.begins, counts[0] + 1);
  assert.equal(moved.commits, counts[1] + 1);
  assert.equal(moved.state.pilot, true);
  await page.screenshot({ path: '../output/electron/verification/camera-pilot.png' });
});
test('capture uses authored dimensions and excludes helper overlays; extreme zoom stays finite with adaptive clipping', async () => {
  const capture = await page.evaluate(async () => {
    const f = window.cameraReview,
      result = await f.api.current.captureCameraView(f.cameraId);
    const image = new Image();
    image.src = result.dataUrl;
    await image.decode();
    return {
      width: result.width,
      height: result.height,
      actual: [image.width, image.height],
      png: result.dataUrl.startsWith('data:image/png;base64,'),
    };
  });
  assert.deepEqual(capture.actual, [640, 360]);
  assert.equal(capture.png, true);
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.useSceneCamera(null);
    f.api.current.focus();
  });
  await page.waitForTimeout(450);
  const canvas = page.locator('[data-testid=viewport-canvas]');
  await canvas.dispatchEvent('wheel', { deltaY: -1e9, deltaMode: 0 });
  await page.waitForTimeout(60);
  const near = await page.evaluate(() => window.cameraReview.api.current.getCameraView());
  const distance = Math.hypot(...near.position.map((v, i) => v - near.target[i]));
  assert.ok(distance > 0.00099 && distance < 0.00101);
  assert.ok(near.near < distance * 0.01);
  assert.ok(near.near > 0 && near.far > near.near);
  await page.screenshot({ path: '../output/electron/verification/camera-zoom-near.png' });
  await canvas.dispatchEvent('wheel', { deltaY: 1e9, deltaMode: 0 });
  await page.waitForTimeout(60);
  const far = await page.evaluate(() => window.cameraReview.api.current.getCameraView());
  assert.ok(far.position.every(Number.isFinite));
  assert.ok(far.far > Math.hypot(...far.position.map((v, i) => v - far.target[i])));
  await page.screenshot({ path: '../output/electron/verification/camera-zoom-far.png' });
  assert.deepEqual(errors, []);
});

test('light type edits retain identity and real direction through parent transforms, undo, save and invalid edits', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, createPrimitive } = await import('/src/scene.ts'),
      { readLight, updateLight } = await import('/src/scene-light.ts'),
      { encodeDocument, decodeDocument, saveWorkspace, loadWorkspace } = await import(
        '/src/editor.ts'
      );
    const doc = await createDemo(),
      node = createPrimitive('light'),
      parent = doc.nodes.find((n) => n.kind === 'link');
    parent.object.add(node.object);
    node.parentId = parent.id;
    doc.nodes.push(node);
    const uuid = node.object.uuid;
    const results = [];
    for (const type of ['spot', 'directional', 'area', 'hemisphere', 'point']) {
      updateLight(doc, node.id, {
        type,
        direction: [1, 2, -3],
        intensity: 7,
        range: 8,
        angle: 27,
        penumbra: 0.4,
        width: 3,
        height: 2,
      });
      const decoded = loadWorkspace(saveWorkspace(doc, [], 10)).doc,
        n = decoded.nodes.find((n) => n.id === node.id),
        state = readLight(n);
      results.push({
        type: state.type,
        id: n.object.uuid === uuid,
        parent: n.parentId === parent.id,
        intensity: state.intensity,
        range: state.range,
        angle: state.angle,
        width: state.width,
        direction: state.direction,
        target: !('target' in n.object) || n.object.target.parent === n.object,
      });
    }
    updateLight(doc, node.id, { type: 'spot', direction: [1, 0, -1] });
    const before = readLight(node).direction;
    updateLight(doc, node.id, { color: '#ff0000' });
    const after = readLight(node).direction;
    updateLight(doc, node.id, { range: 0.001 });
    const shadowValid =
      node.object.shadow.camera.near > 0 &&
      node.object.shadow.camera.near < node.object.shadow.camera.far;
    const original = JSON.stringify(encodeDocument(doc));
    let failures = 0;
    for (const patch of [
      { range: -1 },
      { angle: 90 },
      { direction: [0, 0, 0] },
      { width: 0 },
      { intensity: Infinity },
    ])
      try {
        updateLight(doc, node.id, patch);
      } catch {
        failures += original === JSON.stringify(encodeDocument(doc)) ? 1 : 0;
      }
    const undo = decodeDocument(encodeDocument(doc));
    node.object.rotation.y += 0.5;
    node.object.updateMatrixWorld(true);
    return {
      results,
      before,
      after,
      failures,
      shadowValid,
      changed: readLight(node).direction,
      restored: readLight(undo.nodes.find((n) => n.id === node.id)).direction,
    };
  });
  assert.equal(result.results.length, 5);
  for (const state of result.results) {
    assert.ok(state.id && state.parent && state.target);
    assert.equal(state.intensity, 7);
  }
  assert.equal(result.results[0].range, 8);
  assert.ok(Math.abs(result.results[0].angle - 27) < 1e-10);
  assert.equal(result.results[2].width, 3);
  assert.deepEqual(result.before, result.after);
  assert.equal(result.failures, 5);
  assert.equal(result.shadowValid, true);
  assert.notDeepEqual(result.changed, result.restored);
});

test('light properties drive rendered spot/area emitters and helpers with no renderer errors', async () => {
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { createPrimitive } = await import('/src/scene.ts'),
      { updateLight } = await import('/src/scene-light.ts');
    const light = createPrimitive('light', 'Studio light');
    light.object.position.set(0, -0.8, 1.5);
    f.doc.root.add(light.object);
    f.doc.nodes.push(light);
    f.lightId = light.id;
    updateLight(f.doc, light.id, { range: 3, intensity: 30 });
    f.api.current.focus();
    f.select(light.id);
    f.refresh();
  });
  await page.waitForTimeout(450);
  await page.getByRole('combobox', { name: 'Light type', exact: true }).click();
  await page.getByRole('option', { name: 'Spot', exact: true }).click();
  await page.getByLabel('Spot cutoff', { exact: true }).waitFor();
  await page.screenshot({ path: '../output/electron/verification/light-spot.png' });
  const spot = await page.evaluate(async () => {
    const f = window.cameraReview,
      { readLight } = await import('/src/scene-light.ts');
    return readLight(f.doc.nodes.find((n) => n.id === f.lightId));
  });
  assert.equal(spot.type, 'spot');
  assert.equal(spot.range, 3);
  await page.getByRole('combobox', { name: 'Light type', exact: true }).click();
  await page.getByRole('option', { name: 'Area', exact: true }).click();
  await page.getByLabel('Emitter width', { exact: true }).waitFor();
  await page.screenshot({ path: '../output/electron/verification/light-area.png' });
  assert.deepEqual(errors, []);
});

test('camera frustum and influence layer are visible in production rendering and toggles preserve source', async () => {
  await page.evaluate(async () => {
    const f = window.cameraReview,
      { updateCameraView } = await import('/src/scene-camera.ts');
    f.select(f.cameraId);
    updateCameraView(f.doc, f.cameraId, {
      position: [-1, -0.6, 1.3],
      target: [0, 0, 0.7],
      far: 2.5,
    });
    f.api.current.focus();
    f.refresh();
  });
  await page.waitForTimeout(450);
  await page.screenshot({ path: '../output/electron/verification/camera-frustum.png' });
  const visible = await page.locator('[data-testid=viewport-canvas]').screenshot();
  await page.evaluate(async () => {
    const { DEFAULT_VIEWPORT_LAYERS } = await import('/src/viewport/layers.ts');
    window.cameraReview.setLayers({ ...DEFAULT_VIEWPORT_LAYERS, influence: false });
  });
  await page.waitForTimeout(60);
  const hidden = await page.locator('[data-testid=viewport-canvas]').screenshot();
  assert.notDeepEqual(visible, hidden);
  assert.deepEqual(errors, []);
});

test('authored scene-camera projection updates controlled toolbar state once without leaving pilot view', async () => {
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.projectionChanges = [];
    f.select(f.cameraId);
    f.api.current.useSceneCamera(f.cameraId, true);
  });
  const select = page.getByRole('radiogroup', { name: 'Camera projection', exact: true });
  await select.getByRole('radio', { name: 'Orthographic', exact: true }).click();
  await page.waitForFunction(() => window.cameraReview.projection === 'orthographic');
  let state = await page.evaluate(() => {
    const f = window.cameraReview;
    return {
      events: f.projectionChanges,
      view: f.api.current.getCameraView(),
      active: f.state,
      expected: f.cameraId,
    };
  });
  assert.deepEqual(state.events, ['orthographic']);
  assert.equal(state.view.projection, 'orthographic');
  assert.equal(state.active.id, state.expected);
  assert.equal(state.active.pilot, true);
  await page.evaluate(() => {
    window.cameraReview.refresh();
    window.cameraReview.api.current.capture();
  });
  await page.waitForTimeout(60);
  assert.deepEqual(await page.evaluate(() => window.cameraReview.projectionChanges), [
    'orthographic',
  ]);
  await select.getByRole('radio', { name: 'Perspective', exact: true }).click();
  await page.waitForFunction(() => window.cameraReview.projection === 'perspective');
  state = await page.evaluate(() => {
    const f = window.cameraReview;
    return {
      events: f.projectionChanges,
      active: f.state,
      view: f.api.current.getCameraView(),
      expected: f.cameraId,
    };
  });
  assert.deepEqual(state.events, ['orthographic', 'perspective']);
  assert.equal(state.view.projection, 'perspective');
  assert.equal(state.active.id, state.expected);
  assert.equal(state.active.pilot, true);
  assert.deepEqual(errors, []);
});

test('camera actions stay in two rows while View/Pilot return in place and projection shares field edges', async () => {
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.select(f.cameraId);
    f.api.current.useSceneCamera(null);
  });
  const properties = page.locator('[data-camera-properties]'),
    actions = properties.locator('.camera-action-grid');
  const positions = async () =>
    actions.getByRole('button').evaluateAll((buttons) =>
      buttons.map((button) => {
        const r = button.getBoundingClientRect();
        return [r.x, r.y, r.width, r.height];
      }),
    );
  const original = await positions();
  assert.equal(original.length, 4);
  assert.equal(original[0][1], original[1][1]);
  assert.equal(original[2][1], original[3][1]);
  assert.ok(original[2][1] > original[0][1]);
  await actions.getByRole('button', { name: 'View Camera', exact: true }).click();
  await page.waitForFunction(
    () => window.cameraReview.state.id && !window.cameraReview.state.pilot,
  );
  assert.equal(
    await actions.getByRole('button').nth(0).getAttribute('aria-label'),
    'Return to editor',
  );
  assert.deepEqual(await positions(), original);
  await actions.getByRole('button', { name: 'Pilot camera', exact: true }).click();
  await page.waitForFunction(() => window.cameraReview.state.pilot);
  assert.equal(
    await actions.getByRole('button').nth(1).getAttribute('aria-label'),
    'Return to editor',
  );
  assert.deepEqual(await positions(), original);
  await actions.getByRole('button', { name: 'Return to editor', exact: true }).click();
  await page.waitForFunction(() => window.cameraReview.state.id === null);
  assert.deepEqual(await positions(), original);
  assert.equal(await properties.getByRole('switch').count(), 0);
  const fields = await properties.locator('.camera-property-row').evaluateAll((rows) =>
    rows.map((row) => {
      const r = row.lastElementChild.getBoundingClientRect();
      return [r.left, r.right];
    }),
  );
  for (const field of fields) assert.deepEqual(field, fields[0]);
  assert.equal(
    await properties.locator('.camera-property-row').first().locator('label').innerText(),
    'Projection type',
  );
  await page.evaluate(() => {
    document.querySelector('#root > div').style.gridTemplateColumns = '300px minmax(0, 1fr) 280px';
  });
  await actions.scrollIntoViewIfNeeded();
  const compact = await positions();
  await actions.getByRole('button', { name: 'View Camera', exact: true }).click();
  assert.deepEqual(await positions(), compact);
  const fit = await actions.locator('.camera-action-label').evaluateAll((labels) =>
    labels.map((label) => {
      const text = document.createRange();
      text.selectNodeContents(label);
      const t = text.getBoundingClientRect(),
        b = label.parentElement.getBoundingClientRect();
      return (
        t.left >= b.left + 8 && t.right <= b.right - 7 && t.top >= b.top && t.bottom <= b.bottom
      );
    }),
  );
  assert.ok(fit.every(Boolean));
  await properties.screenshot({
    path: '../output/electron/verification/camera-actions-narrow-insets.png',
  });
  await actions.getByRole('button', { name: 'Return to editor', exact: true }).click();
  await page.evaluate(() => {
    document.querySelector('#root > div').style.gridTemplateColumns = '300px minmax(0, 1fr) 340px';
  });
  assert.deepEqual(errors, []);
});

test('compact camera More menu supports keyboard dismissal and all four real secondary actions', async () => {
  const card = page.locator('.camera-entry').first(),
    trigger = card.getByRole('button', { name: 'More camera actions', exact: true });
  await trigger.focus();
  await page.keyboard.press('ArrowDown');
  const menu = page.getByRole('menu', { name: /More camera actions/ });
  assert.equal(await menu.getByRole('menuitem').count(), 4);
  assert.equal(
    await menu
      .getByRole('menuitem')
      .first()
      .evaluate((element) => element === document.activeElement),
    true,
  );
  await page.keyboard.press('End');
  assert.equal(
    await menu
      .getByRole('menuitem', { name: 'Delete camera', exact: true })
      .evaluate((element) => element === document.activeElement),
    true,
  );
  await page.keyboard.press('Escape');
  await menu.waitFor({ state: 'hidden' });
  assert.equal(await trigger.evaluate((element) => element === document.activeElement), true);
  for (const [label, method] of [
    ['Focus camera', 'focus'],
    ['Align to view', 'alignCameraToView'],
    ['Capture image', 'captureCameraView'],
  ]) {
    await page.evaluate((method) => {
      const f = window.cameraReview,
        original = f.api.current[method];
      f.cameraAction = null;
      f.api.current[method] = (...args) => {
        f.cameraAction = { method, args };
        return original(...args);
      };
    }, method);
    await trigger.click();
    if (label === 'Focus camera') {
      await page.evaluate(() => {
        window.cameraKeyLeaks = 0;
        window.cameraKeyListener = () => window.cameraKeyLeaks++;
        window.addEventListener('keydown', window.cameraKeyListener);
      });
      await page.keyboard.press('Space');
      assert.equal(await page.evaluate(() => window.cameraKeyLeaks), 0);
      await page.evaluate(() => window.removeEventListener('keydown', window.cameraKeyListener));
    } else await menu.getByRole('menuitem', { name: label, exact: true }).click();
    const action = await page.evaluate(() => window.cameraReview.cameraAction);
    assert.equal(action.method, method);
    assert.deepEqual(action.args, [await page.evaluate(() => window.cameraReview.cameraId)]);
    await menu.waitFor({ state: 'hidden' });
  }
  await page.waitForFunction(() =>
    window.cameraReview.captured?.dataUrl?.startsWith('data:image/png'),
  );
  const before = await page.evaluate(() => window.cameraReview.doc.nodes.length);
  await trigger.click();
  await menu.getByRole('menuitem', { name: 'Delete camera', exact: true }).click();
  await card.waitFor({ state: 'hidden' });
  assert.equal(await page.evaluate(() => window.cameraReview.doc.nodes.length), before - 1);
  assert.deepEqual(errors, []);
});
