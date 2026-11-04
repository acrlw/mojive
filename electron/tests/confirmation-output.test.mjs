import { createServer as createTcpServer } from 'node:net';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
let server, browser, page, url;
before(async () => {
  const socket = createTcpServer();
  await new Promise((resolve) => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise((resolve) => socket.close(resolve));
  server = await createServer({
    root: fileURLToPath(new URL('../', import.meta.url)),
    server: { host: '127.0.0.1', port, strictPort: true },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1100, height: 820 } });
  page.setDefaultTimeout(5000);
  url = `http://127.0.0.1:${port}`;
  await page.goto(url + '/tests/harness/ui-confirm-review.html');
  await page.waitForFunction(() => window.confirmHarness);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('confirmation uses one complete sentence, equal buttons, safe initial focus and a keyboard trap', async () => {
  await page.getByRole('button', { name: 'Open confirmation' }).click();
  const dialog = page.getByRole('alertdialog');
  assert.equal(
    await dialog.locator('p').textContent(),
    '关闭前，是否保存“robot arm.mojive.json”的更改？',
  );
  assert.equal(await dialog.locator('.icon').count(), 0);
  assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), '取消');
  const widths = await dialog
    .locator('button')
    .evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().width));
  assert.ok(Math.max(...widths) - Math.min(...widths) < 1);
  await page.keyboard.press('Shift+Tab');
  assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), '保存');
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.textContent.trim()), '取消');
  await page.keyboard.press('Enter');
  assert.equal(await dialog.count(), 0);
  assert.deepEqual(await page.evaluate(() => window.confirmEvents), ['cancel']);
});
test('save failure retains the modal; pending save cannot discard and successful save resolves once', async () => {
  await page.evaluate(() => {
    window.confirmEvents = [];
    window.confirmSaveMode = 'pending';
    window.confirmHarness.show('open');
  });
  await page.getByRole('button', { name: '保存', exact: true }).click();
  assert.equal(await page.getByRole('alertdialog').getAttribute('aria-busy'), 'true');
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('alertdialog').count(), 1);
  assert.deepEqual(await page.evaluate(() => window.confirmEvents), ['save']);
  await page.evaluate(() => window.resolveConfirmSave(false));
  await page.getByRole('button', { name: '保存', exact: true }).waitFor({ state: 'visible' });
  assert.equal(await page.getByRole('button', { name: '保存', exact: true }).isEnabled(), true);
  await page.evaluate(() => (window.confirmSaveMode = 'ok'));
  await page.getByRole('button', { name: '保存', exact: true }).click();
  assert.equal(await page.getByRole('alertdialog').count(), 0);
  assert.deepEqual(await page.evaluate(() => window.confirmEvents), ['save', 'save']);
});
test('discard and Escape remain separate actions; long filenames fit at200% scale', async () => {
  await page.evaluate(() => {
    window.confirmEvents = [];
    document.getElementById('root').style.zoom = '2';
    window.confirmHarness.show('close', 'industrial-robot-with-a-long-scene-filename.mojive.json');
  });
  const box = await page.getByRole('alertdialog').boundingBox();
  assert.ok(box.x >= 0 && box.y >= 0 && box.x + box.width <= 1100 && box.y + box.height <= 820);
  await page.getByRole('button', { name: '不保存', exact: true }).click();
  assert.deepEqual(await page.evaluate(() => window.confirmEvents), ['discard']);
  await page.evaluate(() => window.confirmHarness.show());
  await page.getByRole('alertdialog').waitFor();
  await page.keyboard.press('Escape');
  assert.deepEqual(await page.evaluate(() => window.confirmEvents), ['discard', 'cancel']);
});
test('output search shares the toolbar row and all toolbar icons use20px masters; narrow width wraps without overflow', async () => {
  await page.goto(url + '/tests/harness/ui-settings-review.html');
  await page.waitForFunction(() => window.uiState);
  await page.getByRole('button', { name: '搜索输出', exact: true }).click();
  let geometry = await page.locator('.output-toolbar').evaluate((node) => {
    const groups = ['.output-severities', '.output-search', '.output-actions'].map((selector) => {
      const r = node.querySelector(selector).getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    });
    return {
      groups,
      icons: [...node.querySelectorAll('.icon')].map((icon) => icon.getBoundingClientRect().width),
    };
  });
  assert.ok(
    Math.max(...geometry.groups.map((g) => g.y)) - Math.min(...geometry.groups.map((g) => g.y)) < 1,
  );
  assert.ok(geometry.icons.every((size) => size === 20));
  assert.ok(geometry.groups[1].x > geometry.groups[0].x + geometry.groups[0].width);
  assert.ok(geometry.groups[1].x + geometry.groups[1].width <= geometry.groups[2].x);
  await page.locator('.mojive-output').evaluate((node) => (node.style.width = '400px'));
  const narrow = await page.locator('.output-toolbar').evaluate((node) => {
    const search = node.querySelector('.output-search').getBoundingClientRect(),
      toolbar = node.getBoundingClientRect();
    return {
      search: { left: search.left, right: search.right, top: search.top },
      toolbar: { left: toolbar.left, right: toolbar.right, top: toolbar.top },
    };
  });
  assert.ok(narrow.search.top > narrow.toolbar.top);
  assert.ok(
    narrow.search.left >= narrow.toolbar.left && narrow.search.right <= narrow.toolbar.right,
  );
  assert.equal(
    await page
      .locator('.output-message .icon')
      .first()
      .evaluate((node) => node.getBoundingClientRect().width),
    16,
  );
});
test('actual Settings switches retain circular thumbs and capsule tracks in dark and light themes', async () => {
  await page.locator('.mojive-output').evaluate((node) => (node.style.width = ''));
  await page.getByRole('button', { name: '设置', exact: true }).click();
  await page.getByRole('button', { name: '渲染', exact: true }).click();
  for (const theme of ['dark', 'light']) {
    await page.evaluate((theme) => window.uiActions.updateSettings({ appearance: theme }), theme);
    const toggle = page.getByRole('switch').first();
    for (let index = 0; index < 2; index++) {
      const geometry = await toggle.evaluate((node) => {
        const track = node.querySelector('.custom-switch'),
          thumb = track.querySelector('span'),
          style = getComputedStyle(track),
          thumbStyle = getComputedStyle(thumb);
        return {
          track: style.cornerShape,
          thumb: thumbStyle.cornerShape,
          width: thumb.getBoundingClientRect().width,
          height: thumb.getBoundingClientRect().height,
          radius: thumbStyle.borderRadius,
        };
      });
      assert.match(geometry.track, /^(round|superellipse\(1\))$/);
      assert.match(geometry.thumb, /^(round|superellipse\(1\))$/);
      assert.equal(geometry.width, geometry.height);
      assert.equal(geometry.radius, '50%');
      await toggle.click();
    }
  }
});
test('primary and secondary actions retain native semantic ink through hover, pressed and disabled states', async () => {
  await page.goto(url + '/tests/harness/ui-confirm-review.html');
  await page.waitForFunction(() => window.confirmHarness);
  const palettes = {
    dark: {
      primary: ['rgb(156, 191, 141)', 'rgb(103, 135, 90)', 'rgb(184, 210, 172)'],
      secondary: ['rgb(43, 47, 52)', 'rgb(54, 59, 65)', 'rgb(64, 71, 78)'],
      text: 'rgb(26, 29, 32)',
    },
    light: {
      primary: ['rgb(76, 112, 64)', 'rgb(69, 104, 58)', 'rgb(62, 97, 53)'],
      secondary: ['rgb(202, 208, 199)', 'rgb(192, 200, 186)', 'rgb(181, 192, 175)'],
      text: 'rgb(255, 255, 255)',
    },
  };
  for (const theme of ['dark', 'light']) {
    await page.evaluate((theme) => {
      window.confirmHarness.update({ appearance: theme });
      window.confirmHarness.show();
    }, theme);
    const dialog = page.getByRole('alertdialog');
    await dialog.waitFor();
    for (const variant of ['primary', 'secondary']) {
      const button = dialog.locator(`[data-button-variant=${variant}]`).first();
      for (let state = 0; state < 3; state++) {
        if (!state) await page.mouse.move(2, 2);
        else await button.hover();
        if (state === 2) await page.mouse.down();
        await page.waitForTimeout(160);
        const ink = await button.evaluate((node) => ({
          background: getComputedStyle(node).backgroundColor,
          foreground: getComputedStyle(node).color,
        }));
        assert.equal(ink.background, palettes[theme][variant][state]);
        if (variant === 'primary') assert.equal(ink.foreground, palettes[theme].text);
      }
      await page.mouse.move(2, 2);
      await page.mouse.up();
      await button.evaluate((node) => (node.disabled = true));
      const disabled = await button.evaluate((node) => ({
        background: getComputedStyle(node).backgroundColor,
        opacity: getComputedStyle(node).opacity,
      }));
      assert.equal(disabled.background, palettes[theme][variant][0]);
      assert.equal(disabled.opacity, '0.6');
      await button.evaluate((node) => (node.disabled = false));
    }
    await page.keyboard.press('Escape');
  }
  await page.goto(url + '/tests/harness/ui-controls-review.html');
  await page.waitForFunction(() => window.controlState);
  for (const theme of ['dark', 'light']) {
    await page.evaluate(async (theme) => {
      const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
      applySettings({ ...DEFAULT_SETTINGS, language: 'en', appearance: theme });
    }, theme);
    await page.getByRole('button', { name: 'Material color', exact: true }).click();
    const done = page.getByRole('button', { name: 'Done', exact: true });
    assert.equal(await done.getAttribute('data-button-variant'), 'primary');
    await done.hover();
    await page.waitForTimeout(160);
    assert.equal(
      await done.evaluate((node) => getComputedStyle(node).backgroundColor),
      palettes[theme].primary[1],
    );
    await page.keyboard.press('Escape');
  }
});
