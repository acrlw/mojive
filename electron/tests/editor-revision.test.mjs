import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
let server, browser, page, base;
before(async () => {
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-editor-revision',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime', 'three'],
    },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  await page.addInitScript(() =>
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'light' }),
    ),
  );
  await page.goto(base);
  await page.locator('[role=treeitem]').first().waitFor();
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('hidden tree state remains visible after selection and pointer leave, including flat search', async () => {
  const row = page.locator('[role=treeitem][data-tooltip="02_prismatic_anchor"]');
  await row.hover();
  await row.locator('.tree-visibility').click();
  await page.locator('[role=treeitem][data-tooltip="01_revolute_anchor"]').click();
  await page.mouse.move(800, 40);
  assert.equal(await row.locator('.tree-visibility').getAttribute('aria-pressed'), 'true');
  assert.equal(
    await row.locator('.tree-visibility').evaluate((e) => window.paintStyle(e).opacity),
    '1',
  );
  await page.locator('.left-panel').getByRole('button', { name: 'Search', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search Scene' }).fill('02_prismatic_anchor');
  assert.equal(
    await row.locator('.tree-visibility').evaluate((e) => window.paintStyle(e).opacity),
    '1',
  );
  await page
    .locator('.left-panel')
    .screenshot({ path: '../output/electron/verification/hidden-state-light.png' });
  await row.locator('.tree-visibility').click();
  assert.equal(await row.locator('.tree-visibility').getAttribute('aria-pressed'), 'false');
});
test('navigation exposes cameras and shared layers; inspector optics and layer controls work', async () => {
  await page
    .locator('.activity-rail')
    .getByRole('button', { name: 'Cameras', exact: true })
    .click();
  await page.locator('.cameras-panel').waitFor();
  assert.equal((await page.locator('.camera-entry').count()) > 0, true);
  await page.locator('.activity-rail').getByRole('button', { name: 'Layers', exact: true }).click();
  const grid = page
    .locator('.viewport-layers-panel')
    .getByRole('switch', { name: 'Grid', exact: true });
  await grid.click();
  assert.equal(await grid.getAttribute('aria-checked'), 'false');
  await page
    .locator('.viewport-navigation')
    .getByRole('button', { name: 'Layers', exact: true })
    .click();
  const popup = page.getByRole('dialog', { name: 'Viewport layers' });
  assert.equal(
    await popup.getByRole('switch', { name: 'Grid', exact: true }).getAttribute('aria-checked'),
    'false',
  );
  await popup.getByRole('switch', { name: 'Grid', exact: true }).click();
  assert.equal(await grid.getAttribute('aria-checked'), 'true');
  await popup.getByRole('switch', { name: 'Grid', exact: true }).press('Escape');
  await page.screenshot({ path: '../output/electron/verification/navigation-layers-light.png' });
});
test('WebM export contains timed changing frames and leaves original scene state intact', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { capturePose, encodeDocument } = await import('/src/editor.ts');
    const { disposeDocument } = await import('/src/scene.ts');
    const { exportTakeVideo } = await import('/src/export-video.ts');
    const doc = await createCatalogScene('joint_gizmo');
    const original = JSON.stringify(encodeDocument(doc));
    const world = doc.nodes.find((node) => node.kind === 'world').object;
    const first = capturePose(doc);
    world.position.x = 1;
    const second = capturePose(doc);
    if (JSON.stringify(first) === JSON.stringify(second))
      throw Error('The export fixture must contain distinct authored scene poses.');
    world.position.x = 0;
    const take = {
      id: 'take',
      name: 'Export test',
      frames: [
        { time: 0, pose: first },
        { time: 0.3, pose: second },
        { time: 0.6, pose: second },
      ],
    };
    const view = {
      projection: 'perspective',
      position: [4, -5, 3],
      target: [0, 0, 0.4],
      up: [0, 0, 1],
      fov: 45,
      near: 0.01,
      far: 100,
      orthoHeight: 4,
      sensorWidth: 35,
      focalLength: 40,
      resolution: [320, 240],
    };
    const blob = await exportTakeVideo(doc, take, view, { fps: 10, grid: false });
    const url = URL.createObjectURL(blob),
      video = document.createElement('video');
    video.muted = true;
    video.src = url;
    await new Promise((resolve, reject) => {
      video.onloadeddata = resolve;
      video.onerror = reject;
    });
    // Chromium MediaRecorder WebM may omit duration metadata; force its finite seekable end.
    if (!Number.isFinite(video.duration)) {
      video.currentTime = 1e6;
      await new Promise((resolve) => (video.onseeked = resolve));
    }
    const duration = video.duration;
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 240;
    const context = canvas.getContext('2d');
    async function frame(time) {
      video.currentTime = time;
      await new Promise((resolve) => (video.onseeked = resolve));
      context.drawImage(video, 0, 0);
      return Array.from(context.getImageData(0, 0, 320, 240).data);
    }
    const a = await frame(0.05),
      b = await frame(0.55);
    const changed = a.filter((v, i) => Math.abs(v - b[i]) > 12).length;
    const abort = new AbortController();
    abort.abort();
    let aborted = false;
    try {
      await exportTakeVideo(doc, take, view, { signal: abort.signal });
    } catch {
      aborted = true;
    }
    const untouched = JSON.stringify(encodeDocument(doc)) === original;
    const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
    URL.revokeObjectURL(url);
    disposeDocument(doc);
    return {
      duration,
      changed,
      untouched,
      aborted,
      width: video.videoWidth,
      height: video.videoHeight,
      bytes,
    };
  });
  assert.equal(result.width, 320);
  assert.equal(result.height, 240);
  assert.ok(result.duration >= 0.6 && result.duration < 2, String(result.duration));
  assert.ok(result.changed > 100, 'The recorded scene must move between frames.');
  assert.equal(result.untouched, true);
  assert.equal(result.aborted, true);
  await writeFile('../output/electron/verification/take-export.webm', Buffer.from(result.bytes));
});

test('projection menu has an unclipped shadow, equal insets and keyboard dismissal', async () => {
  await page
    .locator('.viewport-navigation')
    .getByRole('button', { name: 'Projection', exact: true })
    .click();
  const menu = page.getByRole('menu');
  await menu.waitFor();
  const geometry = await menu.evaluate((e) => {
    const s = window.paintStyle(e),
      r = e.getBoundingClientRect();
    return {
      shadow: s.boxShadow,
      clip: s.clipPath,
      pad: [s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft],
      radius: s.borderRadius,
      top: r.top,
      bottom: r.bottom,
      active: e.matches(':popover-open'),
    };
  });
  assert.equal(geometry.active, true);
  assert.equal(geometry.clip, 'none');
  assert.notEqual(geometry.shadow, 'none');
  assert.deepEqual(geometry.pad, ['4px', '4px', '4px', '4px']);
  assert.equal(geometry.radius, '8px');
  assert.ok(geometry.top >= 8 && geometry.bottom <= 992);
  await page.getByRole('menuitem', { name: 'Bottom', exact: true }).hover();
  await page.screenshot({ path: '../output/electron/verification/projection-menu-light.png' });
  await menu.press('End');
  assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), 'Bottom');
  await menu.press('Escape');
  await menu.waitFor({ state: 'detached' });
});

test('switching open application menus opens the next menu with one click and restores its opener', async () => {
  await page.goto(base);
  await page.getByRole('treeitem').first().waitFor();
  const nav = page.getByRole('navigation', { name: 'Application menu' });
  await nav.getByRole('button', { name: 'File', exact: true }).click();
  await page.getByRole('menuitem', { name: /^Open URDF/ }).waitFor();
  await nav.getByRole('button', { name: 'View', exact: true }).click();
  const menu = page.getByRole('menu');
  await page.getByRole('menuitem', { name: 'Frame all', exact: true }).waitFor();
  // A stale light-dismiss toggle must not close the replacement menu on the next task.
  await page.waitForTimeout(80);
  assert.equal(await menu.count(), 1);
  assert.equal(await menu.evaluate((node) => node.matches(':popover-open')), true);
  const anchor = await nav.getByRole('button', { name: 'View', exact: true }).boundingBox();
  const bounds = await menu.boundingBox();
  assert.ok(Math.abs(bounds.x - anchor.x) < 1);
  await menu.press('Escape');
  await menu.waitFor({ state: 'detached' });
  assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), 'View');
});

test('interaction guide owns modal focus and Backspace cannot delete the scene behind it', async () => {
  const nav = page.getByRole('navigation', { name: 'Application menu' });
  await nav.getByRole('button', { name: 'Add', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Box', exact: true }).click();
  await page.getByLabel('Position X', { exact: true }).waitFor();
  const count = await page.getByRole('treeitem').count();
  await nav.getByRole('button', { name: 'Help', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Interaction guide', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Interaction guide', exact: true });
  await dialog.waitFor();
  assert.equal(
    await dialog.evaluate(
      (node) => node.matches(':modal') && node.contains(document.activeElement),
    ),
    true,
  );
  await page.keyboard.press('Backspace');
  assert.equal(await page.locator('[role=treeitem]').count(), count);
  assert.equal(await dialog.isVisible(), true);
  await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'detached' });
  assert.equal(await page.getByRole('treeitem').count(), count);
});

test('export decodes cold embedded textures before the first frame and can cancel during decoding', async () => {
  const result = await page.evaluate(async () => {
    const { CanvasTexture } = await import('/node_modules/three/build/three.module.js');
    const { loadUrdf, disposeDocument } = await import('/src/scene.ts');
    const { capturePose } = await import('/src/editor.ts');
    const { exportTakeVideo } = await import('/src/export-video.ts');
    const { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const doc = await loadUrdf(
      '<robot name="Texture"><link name="body"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>',
      '',
    );
    // A canvas serializes to a cold embedded PNG, unlike an already-decoded data URL
    // whose browser image cache can conceal the ObjectLoader.parse race.
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 2048;
    const context = canvas.getContext('2d');
    const pixels = context.createImageData(2048, 2048);
    let seed = 123;
    for (let i = 0; i < pixels.data.length; i += 4) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      pixels.data[i] = seed >>> 24;
      pixels.data[i + 1] = (seed >>> 16) & 255;
      pixels.data[i + 2] = (seed >>> 8) & 255;
      pixels.data[i + 3] = 255;
    }
    context.putImageData(pixels, 0, 0);
    doc.root.traverse((object) => {
      if (object.isMesh) object.material.map = new CanvasTexture(canvas);
    });
    const take = { id: 'texture', name: 'Texture', frames: [{ time: 0, pose: capturePose(doc) }] };
    const view = {
      projection: 'perspective',
      position: [2, -3, 2],
      target: [0, 0, 0],
      up: [0, 0, 1],
      fov: 45,
      near: 0.01,
      far: 100,
      orthoHeight: 4,
      sensorWidth: 35,
      focalLength: 30,
      resolution: [64, 64],
    };
    const originalRender = ViewportPipeline.prototype.render;
    const originalDecode = HTMLImageElement.prototype.decode;
    const readiness = [];
    ViewportPipeline.prototype.render = function (scene, ...args) {
      scene.traverse((object) => {
        const material = object.material;
        for (const item of material ? (Array.isArray(material) ? material : [material]) : []) {
          const image = item.map?.image;
          if (image instanceof HTMLImageElement)
            readiness.push(image.complete && image.naturalWidth === 2048);
        }
      });
      return originalRender.call(this, scene, ...args);
    };
    try {
      const blob = await exportTakeVideo(doc, take, view, { fps: 10 });
      const abort = new AbortController();
      let decodeStarted = false;
      HTMLImageElement.prototype.decode = function () {
        decodeStarted = true;
        queueMicrotask(() => abort.abort());
        return new Promise(() => {});
      };
      let cancelled = false;
      try {
        await exportTakeVideo(doc, take, view, { signal: abort.signal });
      } catch (error) {
        cancelled = error.name === 'AbortError';
      }
      return { bytes: blob.size, readiness, decodeStarted, cancelled };
    } finally {
      ViewportPipeline.prototype.render = originalRender;
      HTMLImageElement.prototype.decode = originalDecode;
      disposeDocument(doc);
    }
  });
  assert.ok(result.bytes > 0);
  assert.ok(result.readiness.length >= 2);
  assert.ok(
    result.readiness.every(Boolean),
    'Every recorded frame must have its embedded texture.',
  );
  assert.equal(result.decodeStarted, true);
  assert.equal(result.cancelled, true);
});
