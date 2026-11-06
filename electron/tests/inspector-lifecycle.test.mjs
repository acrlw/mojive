import { openEditableJointScene } from './helpers/editable-scene.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, url;
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-inspector-lifecycle',
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  url = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('Inspector measurement subscriptions follow rapid selection, folding and removal commits', async () => {
  const page = await browser.newPage({ viewport: { width: 1700, height: 1050 } }),
    errors = [];
  page.on('pageerror', (error) => errors.push(error.stack));
  page.setDefaultTimeout(6000);
  try {
    await page.addInitScript(() => {
      localStorage.setItem(
        'mojive.electron.settings.v1',
        JSON.stringify({ language: 'en', appearance: 'dark', uiScale: 1.25 }),
      );
      const handlers = new Set();
      window.inspectorMenu = (action) => {
        for (const callback of [...handlers]) callback(action);
      };
      window.mojive = {
        hasNativeMenu: false,
        platform: 'darwin',
        onMenuAction: (callback) => {
          handlers.add(callback);
          return () => handlers.delete(callback);
        },
        updateMenuState: async () => {},
      };
    });
    await page.goto(url);
    await openEditableJointScene(page);
    await page.evaluate(() => document.fonts.ready);
    for (let i = 0; i < 6; i++) {
      for (const kind of ['camera', 'light', 'link', 'joint']) {
        await page.locator(`.tree-row.type-${kind}`).first().click();
        await page.evaluate(async () => {
          const section = document.querySelector('.inspector-content > .section');
          section.querySelector('.section-disclosure').click();
          await new Promise((resolve) => requestAnimationFrame(resolve));
          section.querySelector('.section-disclosure').click();
          await new Promise((resolve) => requestAnimationFrame(resolve));
        });
      }
      await page.locator('.tree-row.type-camera').first().click();
      // Exercise the real application's native menu callback outside React's
      // discrete event dispatch: adding a ground removes the camera sections.
      await page.evaluate(() => window.inspectorMenu('add.ground'));
      const preference = page.getByRole('switch', { name: 'Select in viewport', exact: true });
      await page.waitForFunction(() =>
        document.querySelector('.tree-row.selected')?.classList.contains('type-geom'),
      );
      assert.equal(await preference.getAttribute('aria-checked'), 'false');
      await preference.click();
      await page
        .locator('.operation-notice')
        .getByRole('button', { name: 'Undo', exact: true })
        .click();
      await page
        .locator('.operation-notice')
        .getByRole('button', { name: 'Redo', exact: true })
        .click();
      assert.equal(await preference.getAttribute('aria-checked'), 'true');
      await page.evaluate(() => window.inspectorMenu('edit.undo'));
      await page.evaluate(
        () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
      );
      assert.equal(await preference.getAttribute('aria-checked'), 'false');
      await page.evaluate(() => window.inspectorMenu('edit.undo'));
      await page.locator('.tree-row.type-camera').first().click();
    }
    await page.locator('.tree-row.type-camera').first().click();
    await page.evaluate(
      () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
    );
    assert.deepEqual(errors, []);
    const layouts = await page
      .locator('.inspector .section-content:has(> .vector-group)')
      .evaluateAll((sections) =>
        sections.map((section) => ({
          labels: section.dataset.vectorLabels,
          fields: section.dataset.vectorFields,
          finite: section.clientWidth > 0,
        })),
      );
    assert.ok(layouts.length >= 2);
    assert.ok(
      layouts.every(
        (layout) =>
          ['above', 'inline'].includes(layout.labels) &&
          ['row', 'stack'].includes(layout.fields) &&
          layout.finite,
      ),
      JSON.stringify(layouts),
    );
  } finally {
    await page.close();
  }
});
