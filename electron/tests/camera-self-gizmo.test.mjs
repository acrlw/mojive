import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

let server, browser, page;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    server: { host: '127.0.0.1', port: 0, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1380, height: 900 } });
  page.on('pageerror', (error) => errors.push(error.stack));
  await mkdir('../output/electron/verification', { recursive: true });
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/camera-review.html`,
  );
  await page.waitForFunction(() => window.cameraReview?.api?.current);
});
after(async () => {
  await browser?.close();
  await server?.close();
});

async function ink() {
  return page.getByTestId('viewport-canvas').evaluate((canvas) => {
    const overlay = [...canvas.parentElement.children].find(
      (node) => node.tagName === 'CANVAS' && !node.dataset.testid,
    );
    const pixels = overlay.getContext('2d').getImageData(0, 0, overlay.width, overlay.height).data;
    let count = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) count++;
    return count;
  });
}
async function settle() {
  await page.waitForTimeout(450);
}
test('View and Pilot yield the selected camera gizmo, while preview and other selected objects retain handles', async () => {
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.setTool('rotation');
    f.api.current.focus(f.cameraId);
  });
  await settle();
  const editorInk = await ink();
  assert.ok(editorInk > 300);
  await page.evaluate(() => window.cameraReview.setPreview(true));
  await page.locator('.camera-preview-image').waitFor();
  await settle();
  assert.equal(await ink(), editorInk, 'Preview is independent of the main editor camera');
  const previewImage = await page
    .locator('.camera-preview-image')
    .evaluate((canvas) => canvas.toDataURL());
  await page.screenshot({
    path: '../output/electron/verification/camera-self-preview-initial-workspace.png',
  });
  const previewPixels = await page.locator('.camera-preview-image').evaluate(async (canvas) => {
    const image = new Image();
    image.src = canvas.toDataURL();
    await image.decode();
    const copy = document.createElement('canvas');
    copy.width = image.width;
    copy.height = image.height;
    const context = copy.getContext('2d');
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, copy.width, copy.height).data,
      colors = new Set();
    for (let i = 0; i < rgba.length; i += 4)
      colors.add(`${rgba[i]},${rgba[i + 1]},${rgba[i + 2]},${rgba[i + 3]}`);
    return {
      width: copy.width,
      height: copy.height,
      colors: colors.size,
      renderCount: Number(canvas.dataset.renderCount),
    };
  });
  assert.ok(
    previewPixels.width > 100 &&
      previewPixels.height > 50 &&
      previewPixels.colors > 32 &&
      previewPixels.renderCount > 0,
    JSON.stringify(previewPixels),
  );
  for (const pilot of [false, true]) {
    await page.evaluate((pilot) => {
      const f = window.cameraReview;
      f.select(f.cameraId);
      f.api.current.useSceneCamera(f.cameraId, pilot);
    }, pilot);
    await settle();
    assert.equal(await ink(), 0, `self gizmo must not paint in ${pilot ? 'Pilot' : 'View'}`);
    assert.equal(
      await page.locator('.camera-preview-image').evaluate((canvas) => canvas.toDataURL()),
      previewImage,
      'Viewing the previewed camera preserves its separate preview pixels',
    );
    const withHelpers = await page
      .getByTestId('viewport-canvas')
      .evaluate((canvas) => canvas.toDataURL());
    await page.evaluate(async () => {
      const { DEFAULT_VIEWPORT_LAYERS } = await import('/src/viewport/layers.ts');
      window.cameraReview.setLayers({ ...DEFAULT_VIEWPORT_LAYERS, helpers: false });
    });
    await settle();
    assert.equal(
      await page.getByTestId('viewport-canvas').evaluate((canvas) => canvas.toDataURL()),
      withHelpers,
      'Scene cameras suppress entity helpers',
    );
    await page.getByTestId('viewport-canvas').screenshot({
      path: `../output/electron/verification/camera-self-${pilot ? 'pilot' : 'view'}.png`,
    });
    await page.screenshot({
      path: `../output/electron/verification/camera-self-preview-${pilot ? 'pilot' : 'view'}-workspace.png`,
    });
    await page.evaluate(() => {
      const f = window.cameraReview;
      f.select(f.doc.nodes.find((n) => n.kind === 'link').id);
    });
    await settle();
    assert.ok((await ink()) > 100, 'Other selected objects keep their gizmos');
    await page.evaluate(() => {
      const f = window.cameraReview;
      f.select(f.cameraId);
      f.api.current.useSceneCamera(null);
    });
    await settle();
    assert.equal(await ink(), editorInk, 'Editor return restores the camera handles');
  }
  // A drag where the projected self ring used to be must belong to navigation.
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.useSceneCamera(f.cameraId);
  });
  await settle();
  const before = await page.evaluate(async () => {
    const f = window.cameraReview,
      { readCameraView } = await import('/src/scene-camera.ts');
    return { view: readCameraView(f.doc.nodes.find((n) => n.id === f.cameraId)), begins: f.begins };
  });
  const rect = await page.getByTestId('viewport-canvas').boundingBox(),
    x = rect.x + rect.width / 2 + 80,
    y = rect.y + rect.height / 2;
  await page.mouse.move(x, y);
  assert.equal(
    await page.getByTestId('viewport-canvas').evaluate((c) => c.style.cursor),
    'default',
  );
  await page.mouse.down();
  await page.mouse.move(x + 35, y + 14, { steps: 4 });
  await page.mouse.up();
  await settle();
  const after = await page.evaluate(async () => {
    const f = window.cameraReview,
      { readCameraView } = await import('/src/scene-camera.ts');
    return {
      view: readCameraView(f.doc.nodes.find((n) => n.id === f.cameraId)),
      begins: f.begins,
      state: f.state,
    };
  });
  assert.equal(after.state.id, null);
  assert.equal(after.begins, before.begins);
  assert.deepEqual(after.view, before.view);
  assert.equal(await page.locator('.precise-gizmo-input').count(), 0);
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.select(f.cameraId);
    f.api.current.useSceneCamera(f.cameraId, true);
  });
  await settle();
  await page.mouse.dblclick(x, y);
  await settle();
  assert.equal(
    await page.locator('.precise-gizmo-input').count(),
    0,
    'A self-camera double click never opens precise transform input',
  );
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.select(f.cameraId);
  });
  await settle();
  const pilotBefore = await page.evaluate(() => {
    const f = window.cameraReview;
    return { view: f.api.current.getCameraView(), begins: f.begins, commits: f.commits };
  });
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 35, y + 14, { steps: 4 });
  await page.mouse.up();
  await settle();
  const pilotAfter = await page.evaluate(() => {
    const f = window.cameraReview;
    return {
      view: f.api.current.getCameraView(),
      begins: f.begins,
      commits: f.commits,
      state: f.state,
    };
  });
  assert.equal(pilotAfter.state.pilot, true);
  assert.equal(pilotAfter.begins, pilotBefore.begins + 1);
  assert.equal(pilotAfter.commits, pilotBefore.commits + 1);
  for (let axis = 0; axis < 3; axis++)
    assert.ok(
      Math.abs(pilotAfter.view.target[axis] - pilotBefore.view.target[axis]) < 1e-10,
      'Pilot uses camera orbit, not a self transform handle',
    );
  assert.notDeepEqual(pilotAfter.view.position, pilotBefore.view.position);
  assert.equal(await ink(), 0);
  assert.deepEqual(errors, []);
});
