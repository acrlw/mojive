import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, base;
const measurements = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-rail-buttons-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await writeFile(
    '../output/electron/verification/rail-button-measurements.json',
    JSON.stringify(measurements, null, 2),
  );
  await browser?.close();
  await server?.close();
});
for (const appearance of ['dark', 'light'])
  for (const scale of [1, 1.25, 2])
    test(`rail main button edges remain equal in ${appearance} at ${scale}x, including hover`, async () => {
      const page = await browser.newPage({ viewport: { width: 1700, height: 1100 } });
      const errors = [];
      page.on('pageerror', (e) => errors.push(e.message));
      await page.addInitScript(
        (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
        { language: 'en', appearance, uiScale: scale },
      );
      await page.goto(base);
      await page.getByRole('treeitem').first().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const rail = page.locator('.activity-rail');
      for (const name of ['Scene', 'Settings']) {
        const button = rail.getByTitle(name, { exact: true });
        await button.hover();
        const measured = await rail.evaluate((rail) => {
          const r = rail.getBoundingClientRect(),
            buttons = [...rail.querySelectorAll(':scope > button')],
            first = buttons[0].getBoundingClientRect(),
            last = buttons.at(-1).getBoundingClientRect();
          return {
            insets: [
              first.top - r.top,
              r.bottom - last.bottom,
              first.left - r.left,
              r.right - first.right,
            ],
            buttons: buttons.map((b) => {
              const r = b.getBoundingClientRect(),
                i = b.querySelector('.icon').getBoundingClientRect();
              return {
                title: b.title,
                width: r.width,
                height: r.height,
                icon: [i.width, i.height],
                insets: [i.left - r.left, r.right - i.right, i.top - r.top, r.bottom - i.bottom],
                left: r.left,
                right: r.right,
                padding: getComputedStyle(b).padding,
              };
            }),
          };
        });
        for (const inset of measured.insets)
          assert.ok(Math.abs(inset - 7 * scale) < 0.02, JSON.stringify(measured));
        for (const b of measured.buttons) {
          assert.ok(Math.abs(b.width - 34 * scale) < 0.02);
          assert.equal(b.width, b.height);
          for (const inset of b.insets)
            assert.ok(Math.abs(inset - 6 * scale) < 0.02, JSON.stringify(b));
        }
        assert.notEqual(
          await button.evaluate((b) => getComputedStyle(b).backgroundColor),
          'rgba(0, 0, 0, 0)',
        );
        await rail.screenshot({
          path: `../output/electron/verification/rail-${name.toLowerCase()}-hover-${appearance}-${scale}.png`,
        });
        measurements.push({ appearance, scale, name, ...measured });
      }
      // Exercise an actual aria-label-only icon button, rather than a synthetic title consumer.
      await page.locator('.title-actions').getByTitle('Settings', { exact: true }).click();
      await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('camera');
      const clear = page
        .locator('.settings-search')
        .getByRole('button', { name: 'Clear search', exact: true });
      assert.equal(await clear.getAttribute('title'), 'Clear search');
      await clear.click();
      assert.equal(
        await page.getByRole('textbox', { name: 'Search all settings', exact: true }).inputValue(),
        '',
      );
      assert.deepEqual(errors, []);
      await page.close();
    });

test('shared Button preserves explicit hints, translates semantic aria-label fallback and keeps caption insets', async () => {
  const page = await browser.newPage({ viewport: { width: 800, height: 200 } });
  await page.goto(`${base}/tests/harness/button-review.html`);
  await page.getByTestId('fallback').waitFor();
  assert.equal(await page.getByTestId('fallback').getAttribute('title'), 'Copy');
  assert.equal(await page.getByTestId('explicit').getAttribute('title'), 'Paste');
  assert.equal(await page.getByTestId('suppressed').getAttribute('title'), null);
  assert.equal(await page.getByTestId('unnamed').getAttribute('title'), null);
  assert.equal(await page.getByTestId('caption').getAttribute('title'), null);
  const caption = page.getByTestId('caption');
  await caption.hover();
  const spacing = await caption.evaluate((button) => {
    const b = button.getBoundingClientRect(),
      icon = button.querySelector('.icon').getBoundingClientRect(),
      range = document.createRange();
    range.selectNodeContents(
      [...button.childNodes].find((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim()),
    );
    return {
      leading: icon.left - b.left,
      trailing: b.right - range.getBoundingClientRect().right,
      padding: getComputedStyle(button).padding,
    };
  });
  assert.equal(spacing.leading, 8);
  assert.equal(spacing.trailing, 8);
  assert.equal(spacing.padding, '0px 8px');
  await page.screenshot({ path: '../output/electron/verification/shared-button-hint-and-insets.png' });
  await page.evaluate(() => window.renderButtons('zh'));
  await page.waitForTimeout(25);
  assert.equal(await page.getByTestId('fallback').getAttribute('title'), '复制');
  assert.equal(await page.getByTestId('explicit').getAttribute('title'), '粘贴');
  await page.close();
});
