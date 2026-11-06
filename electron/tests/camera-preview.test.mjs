import { openEditableScene } from './helpers/editable-scene.mjs';
import { installPaintStyle } from './helpers/surface-style.mjs';
import assert from 'node:assert/strict';
import { test, before, beforeEach, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, page;
const errors = [];
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-camera-preview-test',
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1380, height: 900 }, deviceScaleFactor: 1 });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.on('pageerror', (error) => errors.push(error.message));
  await mkdir('../output/electron/verification', { recursive: true });
});
beforeEach(async () => {
  errors.length = 0;
  await page.setViewportSize({ width: 1380, height: 900 });
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/camera-review.html`,
  );
  await page.waitForLoadState('networkidle');
  await page.waitForFunction(() => window.cameraReview?.api?.current);
  await page.evaluate(async () => {
    const { CameraPreviewRenderer } = await import('/src/viewport/camera-preview-renderer.ts');
    const render = CameraPreviewRenderer.prototype.render,
      dispose = CameraPreviewRenderer.prototype.dispose;
    window.previewTrace = { frames: [], disposed: 0, assetsDisposed: 0, peer: null };
    CameraPreviewRenderer.prototype.render = function (doc, view, geometry, width, height) {
      render.call(this, doc, view, geometry, width, height);
      window.previewTrace.peer = this;
      window.previewTrace.frames.push({
        view: structuredClone(view),
        geometry,
        source: doc.root.uuid,
      });
    };
    CameraPreviewRenderer.prototype.dispose = function () {
      window.previewTrace.disposed++;
      dispose.call(this);
    };
    const resources = new Set();
    window.cameraReview.doc.root.traverse((object) => {
      if (object.geometry) resources.add(object.geometry);
      for (const material of Array.isArray(object.material) ? object.material : [object.material])
        if (material) resources.add(material);
    });
    for (const resource of resources)
      resource.addEventListener('dispose', () => window.previewTrace.assetsDisposed++);
  });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
const preview = () => page.getByRole('region', { name: 'Camera Preview', exact: true });
const enable = async () => {
  await page.getByRole('switch', { name: 'Show Camera Preview', exact: true }).click();
  await page.waitForFunction(() => window.previewTrace.frames.length > 0);
};
const last = () => page.evaluate(() => window.previewTrace.frames.at(-1));
const waitFrame = async (action) => {
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  const count = await page.evaluate(() => window.previewTrace.frames.length);
  await action();
  await page.waitForFunction((before) => window.previewTrace.frames.length > before, count);
};

test('preview is explicit, live, follows geometry mode and leaves main view/source/assets intact', async () => {
  assert.equal(await preview().count(), 0);
  const source = await page.evaluate(async () => {
    const { encodeDocument } = await import('/src/editor.ts');
    const f = window.cameraReview;
    // Three lazily computes a bound the first time a hidden collision mesh is rendered.
    f.doc.root.traverse((object) => object.geometry?.computeBoundingSphere());
    f.originalRootParent = f.doc.root.parent;
    return { data: encodeDocument(f.doc), view: f.api.current.getCameraView() };
  });
  await enable();
  const initial = await preview().locator('canvas').screenshot();
  const first = await last();
  assert.equal(first.geometry, 'default');
  await waitFrame(() => page.evaluate(() => window.cameraReview.setGeometry('collision')));
  assert.equal((await last()).geometry, 'collision');
  const collision = await preview().locator('canvas').screenshot();
  assert.notDeepEqual(initial, collision);
  await waitFrame(() => page.evaluate(() => window.cameraReview.setGeometry('both')));
  assert.equal((await last()).geometry, 'both');
  await waitFrame(() => page.evaluate(() => window.cameraReview.setGeometry('default')));
  const stable = await page.evaluate(async () => {
    const { encodeDocument } = await import('/src/editor.ts');
    const f = window.cameraReview;
    return {
      data: encodeDocument(f.doc),
      view: f.api.current.getCameraView(),
      parent: f.doc.root.parent === f.originalRootParent,
    };
  });
  assert.deepEqual(stable.data, source.data);
  assert.deepEqual(stable.view, source.view);
  assert.equal(stable.parent, true);
  await waitFrame(() =>
    page.evaluate(async () => {
      const { setJoint } = await import('/src/scene.ts');
      const f = window.cameraReview,
        joint = f.doc.joints.find((joint) => joint.type === 'revolute');
      setJoint(f.doc, joint.id, 0.8);
      f.refresh();
    }),
  );
  assert.notDeepEqual(await preview().locator('canvas').screenshot(), initial);
  await page.screenshot({ path: '../output/electron/verification/camera-preview-live.png' });
  await preview().getByRole('button', { name: 'Close Camera Preview', exact: true }).click();
  await page.waitForFunction(() => window.previewTrace.disposed === 1);
  assert.equal(await page.evaluate(() => window.previewTrace.assetsDisposed), 0);
  assert.equal(await preview().count(), 0);
  assert.deepEqual(errors, []);
});

test('lock follows a camera across selection, while pin freezes only view and panel position', async () => {
  await enable();
  await preview().getByRole('button', { name: 'Lock', exact: true }).click();
  await page.evaluate(() =>
    window.cameraReview.select(window.cameraReview.doc.nodes.find((n) => n.kind === 'link').id),
  );
  assert.equal(await preview().getAttribute('data-locked'), 'true');
  const start = await last();
  await waitFrame(() =>
    page.evaluate(async () => {
      const { updateCameraView } = await import('/src/scene-camera.ts');
      const f = window.cameraReview;
      updateCameraView(f.doc, f.cameraId, { position: [2, -3, 2], target: [0, 0, 0.5] });
      f.refresh();
    }),
  );
  assert.notDeepEqual((await last()).view.position, start.view.position);
  await preview().getByRole('button', { name: 'Pin', exact: true }).click();
  assert.equal(await preview().getAttribute('data-pinned'), 'true');
  assert.equal(await preview().getAttribute('data-locked'), 'false');
  const pinnedView = (await last()).view;
  const pinnedPosition = await preview().boundingBox();
  const before = await preview().locator('canvas').screenshot();
  await waitFrame(() =>
    page.evaluate(async () => {
      const { updateCameraView } = await import('/src/scene-camera.ts'),
        { setJoint } = await import('/src/scene.ts');
      const f = window.cameraReview;
      updateCameraView(f.doc, f.cameraId, { position: [5, -4, 1] });
      setJoint(f.doc, f.doc.joints.find((joint) => joint.type === 'revolute').id, 0.9);
      f.refresh();
    }),
  );
  assert.deepEqual((await last()).view, pinnedView);
  assert.deepEqual(await preview().boundingBox(), pinnedPosition);
  assert.notDeepEqual(await preview().locator('canvas').screenshot(), before);
  assert.equal(await preview().locator('.camera-preview-title').isDisabled(), true);
  await preview().getByRole('button', { name: 'Pin', exact: true }).click();
  await preview().waitFor({ state: 'hidden' });
  assert.deepEqual(errors, []);
});

test('hidden previews stop drawing and release context; switching scene or camera rebinds cleanly', async () => {
  await enable();
  await page.evaluate(() => window.cameraReview.setPreviewVisible(false));
  await preview().waitFor({ state: 'hidden' });
  const hidden = await page.evaluate(() => ({
    frames: window.previewTrace.frames.length,
    disposed: window.previewTrace.disposed,
  }));
  await page.evaluate(() => window.cameraReview.refresh());
  await page.waitForTimeout(80);
  assert.equal(await page.evaluate(() => window.previewTrace.frames.length), hidden.frames);
  assert.equal(hidden.disposed, 1);
  await waitFrame(() => page.evaluate(() => window.cameraReview.setPreviewVisible(true)));
  const old = await last();
  await waitFrame(() =>
    page.evaluate(async () => {
      const { createPrimitive } = await import('/src/scene.ts'),
        { updateCameraView } = await import('/src/scene-camera.ts');
      const f = window.cameraReview,
        camera = createPrimitive('camera', 'Second camera');
      f.doc.root.add(camera.object);
      f.doc.nodes.push(camera);
      updateCameraView(f.doc, camera.id, {
        position: [-2, -3, 1.5],
        target: [0, 0, 0.7],
        projection: 'orthographic',
        orthoHeight: 2,
      });
      f.select(camera.id);
      f.refresh();
    }),
  );
  assert.equal((await last()).view.projection, 'orthographic');
  assert.equal(
    await preview().locator('.camera-preview-title').getAttribute('data-tooltip'),
    'Second camera',
  );
  await waitFrame(() =>
    page.evaluate(async () => {
      const { createDemo, createPrimitive } = await import('/src/scene.ts');
      const f = window.cameraReview,
        doc = await createDemo(),
        camera = createPrimitive('camera', 'Replacement camera');
      doc.root.add(camera.object);
      doc.nodes.push(camera);
      f.doc = doc;
      f.cameraId = camera.id;
      f.select(camera.id);
      f.refresh();
    }),
  );
  assert.notEqual((await last()).source, old.source);
  assert.equal(
    await preview().locator('.camera-preview-title').getAttribute('data-tooltip'),
    'Replacement camera',
  );
  assert.equal(await page.evaluate(() => window.previewTrace.assetsDisposed), 0);
  await page.evaluate(() => window.cameraReview.setPreview(false));
  await preview().waitFor({ state: 'hidden' });
  assert.equal(await page.evaluate(() => window.previewTrace.disposed), 2);
  assert.deepEqual(errors, []);
});

test('preview drag is bounded, Escape restores its anchor, and compact actions retain pin/lock', async () => {
  await enable();
  const title = preview().locator('.camera-preview-title');
  const box = await title.boundingBox(),
    original = await preview().boundingBox();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x - 180, box.y - 200, { steps: 4 });
  assert.notDeepEqual(await preview().boundingBox(), original);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.deepEqual(await preview().boundingBox(), original);
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(0, 0, { steps: 5 });
  await page.mouse.up();
  const bounds = await preview().evaluate((element) => {
    const a = element.getBoundingClientRect(),
      b = element.parentElement.getBoundingClientRect();
    return {
      left: a.left - b.left,
      top: a.top - b.top,
      right: b.right - a.right,
      bottom: b.bottom - a.bottom,
    };
  });
  assert.ok(bounds.left >= 7 && bounds.top >= 7 && bounds.right >= 7 && bounds.bottom >= 7);
  await page.setViewportSize({ width: 820, height: 700 });
  await preview().getByRole('button', { name: 'Camera preview actions', exact: true }).waitFor();
  await preview().getByRole('button', { name: 'Camera preview actions', exact: true }).click();
  await page.getByRole('menuitemcheckbox', { name: 'Lock', exact: true }).click();
  assert.equal(await preview().getAttribute('data-locked'), 'true');
  await page.screenshot({ path: '../output/electron/verification/camera-preview-compact.png' });
  await preview().getByRole('button', { name: 'Camera preview actions', exact: true }).click();
  await page.getByRole('menuitemcheckbox', { name: 'Pin', exact: true }).click();
  assert.equal(await preview().getAttribute('data-pinned'), 'true');
  const corners = await preview().evaluate((element) => ({
    outer: window.paintStyle(element).borderRadius,
    padding: window.paintStyle(element).padding,
    inner: window.paintStyle(element.querySelector('canvas')).borderRadius,
  }));
  assert.deepEqual(corners, { outer: '12px', padding: '8px', inner: '4px' });
  assert.deepEqual(errors, []);
});

test('projection is a keyboard accessible pair with one authoring transaction per actual change', async () => {
  const group = page.getByRole('radiogroup', { name: 'Camera projection', exact: true });
  assert.equal(await group.getByRole('radio').count(), 2);
  const perspective = group.getByRole('radio', { name: 'Perspective', exact: true });
  const orthographic = group.getByRole('radio', { name: 'Orthographic', exact: true });
  await perspective.focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await orthographic.getAttribute('aria-checked'), 'true');
  assert.equal(await orthographic.evaluate((element) => element === document.activeElement), true);
  await page.keyboard.press('End');
  assert.deepEqual(
    await page.evaluate(() => [window.cameraReview.begins, window.cameraReview.commits]),
    [1, 1],
  );
  await page.keyboard.press('Home');
  assert.equal(await perspective.getAttribute('aria-checked'), 'true');
  assert.deepEqual(
    await page.evaluate(() => [window.cameraReview.begins, window.cameraReview.commits]),
    [2, 2],
  );
  assert.deepEqual(errors, []);
});

test('preview resource cycles release private buffers, textures and shadows without retaining source listeners', async () => {
  await page.evaluate(async () => {
    const T = await import('/node_modules/.vite-camera-preview-test/deps/three.js'),
      { createPrimitive } = await import('/src/scene.ts'),
      { updateLight } = await import('/src/scene-light.ts');
    const f = window.cameraReview;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 2;
    const context = canvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, 2, 2);
    const texture = new T.CanvasTexture(canvas);
    texture.colorSpace = T.SRGBColorSpace;
    const materials = new Set();
    f.doc.root.traverse((object) => {
      if (object.isMesh)
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          materials.add(material);
    });
    for (const material of materials) {
      material.map = texture;
      material.needsUpdate = true;
    }
    const light = createPrimitive('light', 'Preview key');
    f.doc.root.add(light.object);
    f.doc.nodes.push(light);
    light.object.position.set(0, -3, 4);
    updateLight(f.doc, light.id, {
      type: 'spot',
      direction: [0, 3, -3],
      castShadow: true,
      range: 10,
      intensity: 10,
    });
    f.previewTexture = texture;
    f.previewMaterials = [...materials];
    f.previewLight = light;
    f.refresh();
  });
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
  const listeners = () =>
    page.evaluate(() => {
      const f = window.cameraReview,
        resources = new Set([f.previewTexture]);
      f.doc.root.traverse((object) => {
        if (object.geometry) resources.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          if (material) resources.add(material);
      });
      return [...resources].map((resource) => [
        resource.uuid,
        resource._listeners?.dispose?.length ?? 0,
      ]);
    });
  const baseline = await listeners();
  for (let iteration = 0; iteration < 3; iteration++) {
    await enable();
    const original = await preview().locator('canvas').screenshot();
    const beforeShadow = await page.evaluate(
      () => window.cameraReview.previewLight.object.shadow.map?.texture.uuid,
    );
    assert.ok(beforeShadow);
    await waitFrame(() =>
      page.evaluate(() => {
        const f = window.cameraReview;
        const context = f.previewTexture.image.getContext('2d');
        context.fillStyle = f.previewMaterials[0].color.r > 0.8 ? '#00aaff' : '#ff3300';
        context.fillRect(0, 0, 2, 2);
        f.previewTexture.needsUpdate = true;
        for (const material of f.previewMaterials)
          material.color.set(material.color.r > 0.8 ? '#669999' : '#ffffff');
        f.refresh();
      }),
    );
    assert.notDeepEqual(await preview().locator('canvas').screenshot(), original);
    await preview().getByRole('button', { name: 'Close Camera Preview', exact: true }).click();
    await preview().waitFor({ state: 'hidden' });
    assert.deepEqual(await listeners(), baseline);
    assert.equal(
      await page.evaluate(() => window.cameraReview.previewLight.object.shadow.map?.texture.uuid),
      beforeShadow,
    );
  }
  assert.equal(await page.evaluate(() => window.previewTrace.disposed), 3);
  assert.equal(await page.evaluate(() => window.previewTrace.assetsDisposed), 0);
  assert.deepEqual(errors, []);
});

test('production preview and direct projection controls fit dark/light workspaces at 100% and 125%', async () => {
  for (const { scale, language, appearance } of [
    { scale: 1, language: 'en', appearance: 'dark' },
    { scale: 1.25, language: 'zh', appearance: 'light' },
  ]) {
    await page.setViewportSize({ width: 1680, height: 1100 });
    await page.evaluate(
      ({ scale, language, appearance }) => {
        localStorage.setItem(
          'mojive.electron.settings.v1',
          JSON.stringify({ uiScale: scale, language, appearance }),
        );
        localStorage.removeItem('mojive.layout');
      },
      { scale, language, appearance },
    );
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/`);
    await page.waitForLoadState('networkidle');
    const chinese = language !== 'en';
    await openEditableScene(page, chinese);
    await page
      .locator('.activity-rail')
      .getByRole('button', { name: chinese ? '相机' : 'Cameras', exact: true })
      .click();
    await page.locator('.camera-entry-name').first().click();
    await page.getByRole('switch', { name: chinese ? '预览' : 'Preview', exact: true }).click();
    const panel = page.getByRole('region', {
      name: chinese ? '相机预览' : 'Camera Preview',
      exact: true,
    });
    await page.waitForFunction(
      () => Number(document.querySelector('.camera-preview canvas')?.dataset.renderCount) > 0,
    );
    const projection = page.getByRole('radiogroup', {
      name: chinese ? '相机投影' : 'Camera projection',
      exact: true,
    });
    await projection.scrollIntoViewIfNeeded();
    const segments = await projection.evaluate((element) => {
      const group = element.getBoundingClientRect();
      return [...element.children].map((child) => {
        const rect = child.getBoundingClientRect();
        return {
          width: rect.width,
          groupWidth: group.width,
          topInset: rect.top - group.top,
          bottomInset: group.bottom - rect.bottom,
          leftInset: rect.left - group.left,
          rightInset: group.right - rect.right,
          corners: window.paintStyle(child).borderRadius,
        };
      });
    });
    for (const segment of segments) {
      assert.ok(Math.abs(segment.width * 2 - segment.groupWidth) < 0.1);
      assert.equal(segment.topInset, 0);
      assert.equal(segment.bottomInset, 0);
    }
    assert.equal(segments[0].leftInset, 0);
    assert.equal(segments[1].rightInset, 0);
    assert.equal(segments[0].corners, '4px 0px 0px 4px');
    assert.equal(segments[1].corners, '0px 4px 4px 0px');
    const before = await panel.locator('canvas').screenshot();
    await projection
      .getByRole('radio', { name: chinese ? '正交' : 'Orthographic', exact: true })
      .click();
    assert.notDeepEqual(await panel.locator('canvas').screenshot(), before);
    await projection
      .getByRole('radio', { name: chinese ? '透视' : 'Perspective', exact: true })
      .click();
    await panel.locator('.camera-preview-title').hover();
    const titleInset = await panel.locator('.camera-preview-title').evaluate((element) => {
      const title = element.getBoundingClientRect(),
        icon = element.querySelector('.icon').getBoundingClientRect();
      return icon.left - title.left;
    });
    assert.ok(Math.abs(titleInset - 8 * scale) < 1);
    await panel.screenshot({
      path: `../output/electron/verification/camera-preview-title-hover-${appearance}-${scale}x.png`,
    });
    await page.mouse.move(20, 20);
    const layout = await panel.evaluate((element) => {
      const a = element.getBoundingClientRect(),
        parent = element.parentElement.getBoundingClientRect();
      const image = element.querySelector('canvas').getBoundingClientRect();
      return {
        inside:
          a.left >= parent.left &&
          a.right <= parent.right &&
          a.top >= parent.top &&
          a.bottom <= parent.bottom,
        width: a.width,
        imageHeight: image.height,
        height: a.height,
        imageInset: image.left - a.left,
        radius: window.paintStyle(element).borderRadius,
        overflow: element.scrollWidth - element.clientWidth,
      };
    });
    assert.equal(layout.inside, true);
    assert.ok(Math.abs(layout.width - 340 * scale) < 1);
    assert.ok(
      Math.abs(layout.height - layout.imageHeight - 52 * scale) < 1,
      JSON.stringify(layout),
    );
    assert.ok(Math.abs(layout.imageInset - 8 * scale) < 1);
    assert.equal(layout.radius, '12px');
    assert.equal(layout.overflow, 0);
    await page.screenshot({
      path: `../output/electron/verification/camera-preview-workspace-${appearance}-${scale}x.png`,
    });
    const properties = page.locator('[data-camera-properties]'),
      actions = properties.locator('.camera-action-grid');
    await actions.scrollIntoViewIfNeeded();
    const rows = () =>
      actions.getByRole('button').evaluateAll((buttons) =>
        buttons.map((button) => {
          const rect = button.getBoundingClientRect();
          return [rect.x, rect.y, rect.width, rect.height];
        }),
      );
    const positions = await rows();
    assert.equal(positions.length, 4);
    await actions
      .getByRole('button', { name: chinese ? '查看相机' : 'View Camera', exact: true })
      .click();
    await actions
      .getByRole('button', { name: chinese ? '返回编辑器' : 'Return to editor', exact: true })
      .waitFor();
    assert.deepEqual(await rows(), positions);
    await page.locator('.inspector').screenshot({
      path: `../output/electron/verification/camera-actions-view-${appearance}-${scale}x.png`,
    });
    await actions
      .getByRole('button', { name: chinese ? '操控相机' : 'Pilot camera', exact: true })
      .click();
    assert.equal(
      await actions.getByRole('button').nth(1).getAttribute('aria-label'),
      chinese ? '返回编辑器' : 'Return to editor',
    );
    assert.deepEqual(await rows(), positions);
    await page.locator('.inspector').screenshot({
      path: `../output/electron/verification/camera-actions-pilot-${appearance}-${scale}x.png`,
    });
    await actions
      .getByRole('button', { name: chinese ? '返回编辑器' : 'Return to editor', exact: true })
      .click();
    assert.deepEqual(await rows(), positions);
    const fields = await properties.locator('.camera-property-row').evaluateAll((elements) =>
      elements.map((element) => {
        const field = element.lastElementChild.getBoundingClientRect();
        return [field.left, field.right];
      }),
    );
    for (const field of fields) assert.deepEqual(field, fields[0]);
    await panel
      .getByRole('button', { name: chinese ? '关闭相机预览' : 'Close Camera Preview', exact: true })
      .click();
    await panel.waitFor({ state: 'hidden' });
    assert.deepEqual(errors, []);
  }
});

test('saved texture decoding invalidates an idle preview without an unrelated editor action', async () => {
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/.vite-camera-preview-test/deps/three.js'),
      { CameraPreviewRenderer } = await import('/src/viewport/camera-preview-renderer.ts'),
      { readCameraView } = await import('/src/scene-camera.ts');
    const f = window.cameraReview,
      image = new Image(),
      texture = new T.Texture(image);
    const decoded = document.createElement('canvas');
    decoded.width = decoded.height = 2;
    const context = decoded.getContext('2d');
    context.fillStyle = '#e51b23';
    context.fillRect(0, 0, 2, 2);
    texture.colorSpace = T.SRGBColorSpace;
    image.onload = () => {
      texture.needsUpdate = true;
    };
    f.doc.root.traverse((object) => {
      if (object.isMesh)
        for (const material of Array.isArray(object.material)
          ? object.material
          : [object.material]) {
          material.map = texture;
          material.needsUpdate = true;
        }
    });
    const canvas = document.createElement('canvas'),
      view = readCameraView(f.doc.nodes.find((node) => node.id === f.cameraId));
    let wake;
    const completed = new Promise((resolve) => {
      wake = resolve;
    });
    const peer = new CameraPreviewRenderer(canvas, () => {
      peer.render(f.doc, view, 'default', 320, 180);
      wake();
    });
    image.src = decoded.toDataURL('image/png');
    const pending = !image.complete;
    peer.render(f.doc, view, 'default', 320, 180);
    const before = canvas.toDataURL();
    await completed;
    const changed = before !== canvas.toDataURL();
    peer.dispose();
    return { pending, changed, listeners: texture._listeners?.dispose?.length ?? 0 };
  });
  assert.deepEqual(result, { pending: true, changed: true, listeners: 0 });
  assert.deepEqual(errors, []);
});
