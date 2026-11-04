import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser;
before(async () => {
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
    await page
      .locator('.viewport-topbar')
      .screenshot({ path: `../output/electron/verification/geometry-choices-narrow-${language}.png` });
    assert.deepEqual(errors, []);
    await page.close();
  });
}
