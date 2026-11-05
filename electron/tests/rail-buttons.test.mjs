import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
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
      await page.addInitScript(installPaintStyle);
      await page.evaluate(installPaintStyle);
      const errors = [];
      page.on('pageerror', (e) => errors.push(e.message));
      await page.addInitScript(
        (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
        { language: 'en', appearance, uiScale: scale },
      );
      await page.goto(base);
      await page.getByRole('treeitem').first().waitFor();
      await page.evaluate(() => document.fonts.ready);
      await settlePaint(page);
      const rail = page.locator('.activity-rail');
      const indicator = await rail
        .locator('.activity-indicator')
        .first()
        .evaluate((element) => {
          const box = element.getBoundingClientRect();
          const cap = element.firstElementChild;
          const source = cap.getBoundingClientRect();
          return {
            bounds: [box.width, box.height],
            source: [source.width, source.height, source.right - box.right],
            kind: cap.dataset.smoothKind,
            amount: cap.dataset.smoothAmount,
            hit: getComputedStyle(element).pointerEvents,
          };
        });
      assert.deepEqual(indicator.bounds, [2 * scale, 16 * scale]);
      assert.deepEqual(indicator.source, [4 * scale, 16 * scale, 0]);
      assert.equal(indicator.kind, 'capsule');
      assert.equal(indicator.amount, '0.382');
      assert.equal(indicator.hit, 'none');
      for (const name of ['Scene', 'Settings']) {
        const button = rail.getByRole('button', { name: name, exact: true });
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
                title: b.dataset.tooltip,
                width: r.width,
                height: r.height,
                icon: [i.width, i.height],
                insets: [i.left - r.left, r.right - i.right, i.top - r.top, r.bottom - i.bottom],
                left: r.left,
                right: r.right,
                padding: window.paintStyle(b).padding,
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
          await button.evaluate((b) => window.paintStyle(b).backgroundColor),
          'rgba(0, 0, 0, 0)',
        );
        await rail.screenshot({
          path: `../output/electron/verification/rail-${name.toLowerCase()}-hover-${appearance}-${scale}.png`,
        });
        measurements.push({ appearance, scale, name, ...measured });
      }
      // Exercise an actual aria-label-only icon button, rather than a synthetic title consumer.
      await page
        .locator('.title-actions')
        .getByRole('button', { name: 'Settings', exact: true })
        .click();
      await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('camera');
      const clear = page
        .locator('.settings-search')
        .getByRole('button', { name: 'Clear search', exact: true });
      assert.equal(await clear.getAttribute('data-tooltip'), 'Clear search');
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
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  await page.goto(`${base}/tests/harness/button-review.html`);
  await page.getByTestId('fallback').waitFor();
  assert.equal(await page.getByTestId('fallback').getAttribute('data-tooltip'), 'Copy');
  assert.equal(await page.getByTestId('explicit').getAttribute('data-tooltip'), 'Paste');
  assert.equal(await page.getByTestId('suppressed').getAttribute('data-tooltip'), null);
  assert.equal(await page.getByTestId('unnamed').getAttribute('data-tooltip'), null);
  assert.equal(await page.getByTestId('caption').getAttribute('data-tooltip'), null);
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
      padding: window.paintStyle(button).padding,
    };
  });
  assert.equal(spacing.leading, 8);
  assert.equal(spacing.trailing, 8);
  assert.equal(spacing.padding, '0px 8px');
  await page.screenshot({
    path: '../output/electron/verification/shared-button-hint-and-insets.png',
  });
  await page.evaluate(() => window.renderButtons('zh'));
  await page.waitForTimeout(25);
  assert.equal(await page.getByTestId('fallback').getAttribute('data-tooltip'), '复制');
  assert.equal(await page.getByTestId('explicit').getAttribute('data-tooltip'), '粘贴');
  await page.close();
});
