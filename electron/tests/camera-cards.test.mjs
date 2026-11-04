import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
const errors = [];
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-camera-cards-test',
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1680, height: 1100 } });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'zh', appearance: 'light', uiScale: 1.25 }),
    );
    localStorage.removeItem('mojive.layout');
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/`);
  await page.waitForLoadState('networkidle');
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('actual light workspace keeps camera metadata, inset hover targets and localized actions at narrow and wide sizes', async () => {
  await page.locator('.activity-rail').getByTitle('相机', { exact: true }).click();
  for (const [width, label] of [
    [220, 'narrow'],
    [410, 'wide'],
  ]) {
    const splitter = page.getByRole('separator', { name: '调整左侧面板宽度', exact: true });
    const box = await splitter.boundingBox(),
      current = Number(await splitter.getAttribute('aria-valuenow'));
    await page.mouse.move(box.x + box.width / 2, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + (width - current) * 1.25, box.y + 100, {
      steps: 10,
    });
    await page.mouse.up();
    const card = page.locator('.camera-entry').first();
    await card.locator('.camera-entry-name').hover();
    const metrics = await card.evaluate((element) => {
      const r = element.getBoundingClientRect(),
        name = element.querySelector('.camera-entry-name'),
        n = name.getBoundingClientRect(),
        icon = name.querySelector('.icon').getBoundingClientRect();
      const texts = [
        ...element.querySelectorAll(
          '.camera-entry-projection>span:not(.icon),.camera-entry-resolution',
        ),
      ].map((text) => ({ text: text.textContent, overflow: text.scrollWidth - text.clientWidth }));
      return {
        height: r.height,
        nameInset: icon.left - n.left,
        outerInset: n.left - r.left,
        radius: getComputedStyle(element).borderRadius,
        texts,
        buttons: [
          ...element.querySelectorAll('.camera-entry-actions button,.camera-entry-more button'),
        ].map((button) => ({ title: button.title, label: button.getAttribute('aria-label') })),
      };
    });
    assert.ok(metrics.height <= 83 * 1.25);
    assert.equal(metrics.radius, '12px');
    assert.ok(Math.abs(metrics.nameInset - 10) < 1);
    assert.ok(Math.abs(metrics.outerInset - 10) < 1);
    assert.equal(metrics.texts.length, 2);
    assert.ok(metrics.texts.every((text) => text.overflow <= 1));
    assert.deepEqual(
      metrics.buttons.map((button) => button.title),
      ['更多相机操作', '查看相机', '操控相机'],
    );
    assert.ok(metrics.buttons.every((button) => button.title === button.label));
    await page.screenshot({ path: `../output/electron/verification/camera-cards-light-${label}.png` });
    await card.screenshot({ path: `../output/electron/verification/camera-card-hover-light-${label}.png` });
    await card.getByRole('button', { name: '更多相机操作', exact: true }).click();
    const menu = page.getByRole('menu', { name: /更多相机操作/ });
    await menu.waitFor();
    assert.equal(await menu.getByRole('menuitem').count(), 4);
    const m = await menu.boundingBox();
    assert.ok(m.x >= 0 && m.x + m.width <= 1680 && m.y >= 0 && m.y + m.height <= 1100);
    await page.screenshot({ path: `../output/electron/verification/camera-card-menu-light-${label}.png` });
    await page.locator('.camera-panel-header h2').click();
    await menu.waitFor({ state: 'hidden' });
    await page.waitForFunction(
      () =>
        document.querySelector('.camera-entry-more button')?.getAttribute('aria-expanded') ===
        'false',
    );
    assert.equal(
      await card
        .getByRole('button', { name: '更多相机操作', exact: true })
        .getAttribute('aria-expanded'),
      'false',
    );
  }
  assert.deepEqual(errors, []);
});

test('actual camera cards toggle View and Pilot back to the editor and stay synchronized with Inspector', async () => {
  await page.locator('.activity-rail').getByTitle('相机', { exact: true }).click();
  const card = page.locator('.camera-entry').first(),
    actions = card.locator('.camera-entry-actions'),
    view = actions.getByRole('button').nth(0),
    pilot = actions.getByRole('button').nth(1),
    inspector = page.locator('[data-camera-properties] .camera-action-grid');
  await card.locator('.camera-entry-name').click();
  await inspector.waitFor();
  const checkMode = async (mode) => {
    await page.waitForFunction((mode) => {
      const buttons = document.querySelectorAll(
        '.camera-entry:first-child .camera-entry-actions button',
      );
      return (
        buttons.length === 2 &&
        [...buttons].every(
          (button, index) => button.getAttribute('aria-pressed') === String(mode === index),
        )
      );
    }, mode);
    assert.equal(await view.getAttribute('title'), mode === 0 ? '返回编辑器相机' : '查看相机');
    assert.equal(await pilot.getAttribute('title'), mode === 1 ? '返回编辑器相机' : '操控相机');
    for (const button of [view, pilot])
      assert.equal(await button.getAttribute('aria-label'), await button.getAttribute('title'));
    const labels = await inspector
      .getByRole('button')
      .evaluateAll((buttons) =>
        buttons.slice(0, 2).map((button) => button.getAttribute('aria-label')),
      );
    assert.deepEqual(labels, [
      mode === 0 ? '返回编辑器' : '查看相机',
      mode === 1 ? '返回编辑器' : '操控相机',
    ]);
    assert.equal(await card.locator('.camera-entry-actions [data-icon=eye]').count(), 1);
    assert.equal(await card.locator('.camera-entry-actions [data-icon=move]').count(), 1);
  };
  await view.click();
  await checkMode(0);
  await view.click();
  await checkMode(-1);
  await pilot.click();
  await checkMode(1);
  await pilot.click();
  await checkMode(-1);
  await view.click();
  await pilot.click();
  await checkMode(1);
  await view.click();
  await checkMode(0);
  await view.hover();
  await card.screenshot({ path: '../output/electron/verification/camera-card-view-toggle-light.png' });
  await inspector.getByRole('button', { name: '返回编辑器', exact: true }).click();
  await checkMode(-1);
  await pilot.focus();
  await page.keyboard.press('Enter');
  await checkMode(1);
  await page.keyboard.press('Enter');
  await checkMode(-1);
  assert.deepEqual(errors, []);
});
