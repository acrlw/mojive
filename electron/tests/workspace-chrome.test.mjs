import { openEditableScene } from './helpers/editable-scene.mjs';
import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
let server, browser;
before(async () => {
  await mkdir('../output/electron/verification', { recursive: true });
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
for (const [language, appearance, scale] of [
  ['en', 'dark', 1],
  ['zh', 'light', 1.25],
]) {
  test(`workspace rail insets and geometry choices at ${language}/${appearance}/${scale}x`, async () => {
    const page = await browser.newPage({ viewport: { width: 2000, height: 1100 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.addInitScript(
      (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
      { language, appearance, uiScale: scale },
    );
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
    await page.waitForLoadState('networkidle');
    await openEditableScene(page, language === 'zh');
    await page.evaluate(() => document.fonts.ready);
    const insets = await page.locator('.activity-rail').evaluate((rail) => {
      const rect = rail.getBoundingClientRect(),
        buttons = [...rail.querySelectorAll(':scope > button')];
      const first = buttons[0].getBoundingClientRect(),
        last = buttons.at(-1).getBoundingClientRect();
      return [
        first.top - rect.top,
        rect.bottom - last.bottom,
        first.left - rect.left,
        rect.right - first.right,
      ];
    });
    for (const inset of insets) assert.ok(Math.abs(inset - 7 * scale) < 1, JSON.stringify(insets));
    const choices = await page.locator('.geometry-selector button').evaluateAll((buttons) =>
      buttons.map((button) => {
        const bounds = button.getBoundingClientRect(),
          icon = button.querySelector('.icon').getBoundingClientRect(),
          label = button.querySelector('span:not(.icon)').getBoundingClientRect();
        return {
          width: bounds.width,
          height: bounds.height,
          left: icon.left - bounds.left,
          right: bounds.right - label.right,
          overflow: button.scrollWidth - button.clientWidth,
        };
      }),
    );
    for (const c of choices) {
      assert.ok(c.width >= 84 * scale - 1, JSON.stringify(choices));
      assert.ok(Math.abs(c.width - choices[0].width) < 1, JSON.stringify(choices));
      assert.ok(c.left >= 12 * scale - 1 && c.right >= 12 * scale - 1, JSON.stringify(choices));
      assert.equal(c.overflow, 0);
      assert.ok(Math.abs(c.height - 28 * scale) < 1);
    }
    for (let i = 0; i < 4; i++) {
      await page.locator('.geometry-selector button').nth(i).click();
      assert.equal(
        await page.locator('.geometry-selector button').nth(i).getAttribute('aria-pressed'),
        'true',
      );
    }
    await page
      .locator('.viewport-topbar')
      .screenshot({ path: `../output/electron/verification/geometry-choices-${language}.png` });
    await page
      .locator('.activity-rail')
      .screenshot({ path: `../output/electron/verification/navigation-insets-${language}.png` });
    await page.locator('.title-actions > button:has([data-icon="sidebar-left"])').click();
    await page.locator('.title-actions > button:has([data-icon="panel-right"])').click();
    await page.setViewportSize({ width: 560, height: 800 });
    const narrow = await page.locator('.viewport-topbar').evaluate((bar) => {
      const bounds = bar.getBoundingClientRect();
      return {
        overflow: bar.scrollWidth - bar.clientWidth,
        last: bar.lastElementChild.getBoundingClientRect().right,
        right: bounds.right,
      };
    });
    assert.ok(narrow.overflow <= 1 && narrow.last <= narrow.right, JSON.stringify(narrow));
    await page.locator('.viewport-topbar').screenshot({
      path: `../output/electron/verification/geometry-choices-narrow-${language}.png`,
    });
    assert.deepEqual(errors, []);
    await page.close();
  });
}

for (const appearance of ['dark', 'light'])
  test(`selected viewport tools retain solid semantic paint through hover and press in ${appearance}`, async () => {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript(installPaintStyle);
    await page.addInitScript((appearance) => {
      localStorage.setItem(
        'mojive.electron.settings.v1',
        JSON.stringify({ language: 'en', appearance }),
      );
    }, appearance);
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
    await page.waitForSelector('.tool-palette');
    const palette = page.locator('.tool-palette');
    const snap = palette.locator('button:has([data-icon="snap"])');
    if ((await snap.getAttribute('aria-pressed')) !== 'true') await snap.click();
    const selected = palette.locator('button').first();
    const paint = async (button) => {
      await settlePaint(page);
      return button.evaluate((node) => {
        const style = window.paintStyle(node);
        const resolve = (token) => {
          const probe = document.createElement('span');
          probe.style.color = `var(${token})`;
          node.append(probe);
          const color = getComputedStyle(probe).color;
          probe.remove();
          return color;
        };
        return {
          background: style.backgroundColor,
          color: style.color,
          rest: resolve('--button-primary'),
          hover: resolve('--button-primary-hover'),
          pressed: resolve('--button-primary-pressed'),
          ink: resolve('--button-primary-text'),
          active: node.getAttribute('aria-pressed'),
        };
      });
    };
    for (const [name, button] of [
      ['select', selected],
      ['snap', snap],
    ]) {
      await page.mouse.move(900, 60);
      let state = await paint(button);
      assert.equal(state.active, 'true');
      assert.equal(state.background, state.rest);
      assert.equal(state.color, state.ink);
      await palette.screenshot({
        path: `../output/electron/verification/tool-${name}-${appearance}-idle.png`,
      });
      await button.hover();
      state = await paint(button);
      assert.equal(state.background, state.hover);
      assert.equal(state.color, state.ink);
      assert.equal(state.active, 'true');
      await palette.screenshot({
        path: `../output/electron/verification/tool-${name}-${appearance}-hover.png`,
      });
      await page.mouse.down();
      state = await paint(button);
      assert.equal(state.background, state.pressed);
      assert.equal(state.color, state.ink);
      await palette.screenshot({
        path: `../output/electron/verification/tool-${name}-${appearance}-held.png`,
      });
      await page.mouse.move(900, 60);
      await page.mouse.up();
      state = await paint(button);
      assert.equal(state.active, 'true');
      assert.equal(state.background, state.rest);
    }
    await selected.focus();
    await page.keyboard.down('Space');
    let state = await paint(selected);
    assert.equal(state.background, state.pressed);
    await page.keyboard.up('Space');
    state = await paint(selected);
    assert.equal(state.background, state.rest);
    assert.equal(state.active, 'true');
    assert.deepEqual(errors, []);
    await page.close();
  });
