import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
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
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
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
test('output uses the shared 16px search and 20px toolbar actions; narrow width wraps without overflow', async () => {
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
      icons: [...node.querySelectorAll('.output-severities .icon, .output-actions .icon')].map(
        (icon) => icon.getBoundingClientRect().width,
      ),
      searchIcon: node.querySelector('.search-field > .icon').getBoundingClientRect().width,
    };
  });
  assert.ok(
    Math.max(...geometry.groups.map((g) => g.y)) - Math.min(...geometry.groups.map((g) => g.y)) < 1,
  );
  assert.ok(geometry.icons.every((size) => size === 20));
  assert.equal(geometry.searchIcon, 16);
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
test('output icons track first-line ink across fonts, languages and fractional scale', async () => {
  await page.goto(url + '/tests/harness/ui-settings-review.html');
  await page.waitForFunction(() => window.uiActions);
  const { mkdir } = await import('node:fs/promises');
  await mkdir('../output/electron/verification', { recursive: true });
  for (const [language, appearance, outputFont, outputFontSize, scale] of [
    ['en', 'dark', 'mono', 14, 1],
    ['zh', 'dark', 'mono', 14, 2],
    ['zh', 'light', 'body', 18, 1.25],
    ['en', 'light', 'mono', 12, 1.25],
  ]) {
    await page.evaluate(
      ({ language, appearance, outputFont, outputFontSize, scale }) => {
        window.uiActions.updateSettings({ language, appearance, outputFont, outputFontSize });
        window.uiActions.replaceEntries([
          {
            id: 1,
            level: 'info',
            time: '10:50:26',
            text:
              language === 'zh'
                ? '已加载 Joint Gizmos · 45 个对象 · 9 个关节'
                : 'Loaded Joint Gizmos · 45 objects · 9 joints',
          },
          {
            id: 2,
            level: 'warn',
            time: '10:56:11',
            text:
              language === 'zh'
                ? 'H国 · 资源加载警告\n检查网格资源路径。'
                : 'H · Resource warning\nCheck the mesh resource path.',
          },
          {
            id: 3,
            level: 'err',
            time: '10:56:12',
            text: language === 'zh' ? 'H国 · 无法加载网格' : 'H · Could not load mesh',
          },
        ]);
        document.getElementById('root').style.zoom = scale;
      },
      { language, appearance, outputFont, outputFontSize, scale },
    );
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(
      () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
    );
    const metrics = await page
      .locator('.output-message')
      .nth(1)
      .evaluate(
        (row, { language, scale }) => {
          const text = row.querySelector('.output-message-text'),
            icon = row.querySelector('.icon'),
            style = window.paintStyle(text);
          const marker = document.createElement('span');
          marker.style.cssText = 'display:inline-block;width:0;height:0;vertical-align:baseline';
          text.prepend(marker);
          const baseline = marker.getBoundingClientRect().top,
            box = icon.getBoundingClientRect();
          marker.remove();
          const context = document.createElement('canvas').getContext('2d');
          context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
          const ink = context.measureText(language === 'zh' ? 'H国' : 'H');
          return {
            center: (box.top + box.bottom) / 2,
            ink:
              baseline - ((ink.actualBoundingBoxAscent - ink.actualBoundingBoxDescent) * scale) / 2,
            size: box.width / scale,
            overflow: row.scrollWidth - row.clientWidth,
          };
        },
        { language, scale },
      );
    assert.ok(
      Math.abs(metrics.center - metrics.ink) < 0.65 * scale,
      JSON.stringify({ language, outputFont, outputFontSize, scale, metrics }),
    );
    assert.ok(Math.abs(metrics.size - (outputFontSize * 16) / 14) < 0.1);
    assert.ok(metrics.overflow <= 1);
    await page.locator('.mojive-output').screenshot({
      path: `../output/electron/verification/output-alignment-${language}-${appearance}-${outputFont}-${outputFontSize}-${scale}.png`,
    });
  }
  await page.goto(url + '/tests/harness/ui-settings-review.html');
  await page.waitForFunction(() => window.uiActions);
  await page.evaluate(() =>
    window.uiActions.updateSettings({ language: 'zh', outputFont: 'mono', outputFontSize: 14 }),
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
          style = window.paintStyle(track),
          thumbStyle = window.paintStyle(thumb);
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
          background: window.paintStyle(node).backgroundColor,
          foreground: window.paintStyle(node).color,
        }));
        assert.equal(ink.background, palettes[theme][variant][state]);
        if (variant === 'primary') assert.equal(ink.foreground, palettes[theme].text);
      }
      // Releasing a modifier must not discard the still-held pointer owner.
      await page.keyboard.press('Shift');
      await page.mouse.move(2, 2);
      await page.mouse.up();
      await settlePaint(page);
      assert.equal(
        await button.evaluate((node) => window.paintStyle(node).backgroundColor),
        palettes[theme][variant][0],
        'Release outside restores idle paint',
      );
      await button.evaluate((node) => (node.disabled = true));
      await settlePaint(page);
      const disabled = await button.evaluate((node) => ({
        background: window.paintStyle(node).backgroundColor,
        opacity: window.paintStyle(node).opacity,
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
      await done.evaluate((node) => window.paintStyle(node).backgroundColor),
      palettes[theme].primary[1],
    );
    await page.keyboard.press('Escape');
  }
});
