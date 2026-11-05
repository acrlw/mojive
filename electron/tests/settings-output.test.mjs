import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
import { createServer as createTcpServer } from 'node:net';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
let server, browser, page;
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
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-settings-review.html`,
  );
  await page.waitForFunction(() => window.uiState);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('settings sanitize persisted data and keep independent native numeric/output typography', async () => {
  const result = await page.evaluate(() =>
    window.uiModules.normalizeSettings({
      language: 'zh',
      uiScale: 99,
      cameraFov: -3,
      orbitSpeed: 'NaN',
      numericFont: 'body',
      numericFontSize: 18,
      outputFont: 'mono',
      outputFontSize: 13,
      appearance: 'bad',
      showGrid: false,
    }),
  );
  assert.equal(result.uiScale, 2);
  assert.equal(result.cameraFov, 10);
  assert.equal(result.orbitSpeed, 0.35);
  assert.equal(result.numericFontSize, 18);
  assert.equal(result.outputFontSize, null);
  assert.equal(result.numericFont, 'body');
  assert.equal(result.appearance, 'dark');
  assert.equal(result.showGrid, false);
});
test('Chinese is default; settings translate immediately, persist and apply exact native light palette', async () => {
  assert.equal(await page.locator('html').getAttribute('lang'), 'zh-CN');
  await page.getByRole('button', { name: '设置', exact: true }).click();
  await page.getByRole('combobox', { name: '语言', exact: true }).click();
  await page.getByRole('option', { name: 'English', exact: true }).click();
  await page.getByRole('dialog', { name: 'Settings', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
  assert.equal(
    await page.evaluate(() =>
      window.paintStyle(document.documentElement).getPropertyValue('--panel-color').trim(),
    ),
    '#F1F3EF',
  );
  await page.getByRole('combobox', { name: 'Numeric Font family', exact: true }).click();
  await page.getByRole('option', { name: 'MiSans', exact: true }).click();
  await page.getByRole('combobox', { name: 'Numeric Font size', exact: true }).click();
  await page.getByRole('option', { name: '18', exact: true }).click();
  assert.match(
    await page.evaluate(() =>
      window.paintStyle(document.documentElement).getPropertyValue('--font-mono'),
    ),
    /MiSans/,
  );
  assert.equal(
    await page.evaluate(() =>
      window.paintStyle(document.documentElement).getPropertyValue('--numeric-font-size').trim(),
    ),
    '18px',
  );
  await page.reload();
  await page.waitForFunction(() => window.uiState?.settings.language === 'en');
  assert.equal(await page.evaluate(() => window.uiState.settings.numericFontSize), 18);
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
});
test('settings search supports Chinese labels and change immediately without altering independent fields', async () => {
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search all settings' }).fill('缩放速度');
  await page.getByRole('textbox', { name: 'Zoom speed', exact: true }).fill('2.5');
  await page.getByRole('textbox', { name: 'Zoom speed', exact: true }).press('Enter');
  assert.equal(await page.evaluate(() => window.uiState.settings.zoomSpeed), 2.5);
  assert.equal(await page.evaluate(() => window.uiState.settings.panSpeed), 1);
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('dialog').count(), 0);
});
test('output supports all-token search, severity filters, sorting, selection copy and clear', async () => {
  await page.getByRole('button', { name: 'Search output', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search output', exact: true }).fill('missing mesh');
  assert.equal(await page.getByRole('option').count(), 2);
  await page.getByRole('button', { name: 'Warning', exact: true }).click();
  assert.equal(await page.getByRole('option').count(), 1);
  assert.match(await page.getByRole('option').innerText(), /Missing arm/);
  await page.getByRole('button', { name: 'Warning', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search output', exact: true }).fill('');
  await page.getByRole('button', { name: 'Sort output', exact: true }).click();
  assert.match(await page.getByRole('option').first().innerText(), /Missing arm/);
  await page.getByRole('option').first().click();
  await page
    .getByRole('option')
    .nth(1)
    .click({ modifiers: ['Shift'] });
  assert.equal(await page.getByRole('option', { selected: true }).count(), 2);
  const copied = await page.evaluate(() =>
    window.uiModules.copyOutput(window.uiState.entries, new Set([2, 3])),
  );
  assert.equal(copied, 'Missing wheel mesh\nMissing arm mesh');
  await page.getByRole('button', { name: 'Clear output', exact: true }).click();
  assert.equal(await page.getByRole('option').count(), 0);
  await page.getByText('No messages', { exact: true }).waitFor();
});

test('output follows new history and manual browsing suspends follow', async () => {
  await page.evaluate(() =>
    window.uiActions.replaceEntries(
      Array.from({ length: 100 }, (_, i) => ({
        id: i + 1,
        level: 'info',
        time: '11:00:00',
        text: 'Message ' + i,
      })),
    ),
  );
  await page.getByRole('button', { name: 'Sort output', exact: true }).click();
  const follow = page.getByRole('button', { name: 'Auto-scroll', exact: true });
  if ((await follow.getAttribute('aria-pressed')) !== 'true') await follow.click();
  await page.waitForFunction(() => {
    const node = document.querySelector('.output-messages');
    return node.scrollHeight - node.scrollTop - node.clientHeight < 2;
  });
  await page.locator('.output-messages').hover();
  await page.mouse.wheel(0, -200);
  await page.waitForFunction(
    () =>
      document.querySelector('button[aria-label="Auto-scroll"]').getAttribute('aria-pressed') ===
      'false',
  );
  await follow.click();
  await page.waitForFunction(() => {
    const node = document.querySelector('.output-messages');
    return node.scrollHeight - node.scrollTop - node.clientHeight < 2;
  });
});
test('settings category dimensions stay fixed and segments share one equal connected frame', async () => {
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('button', { name: 'General', exact: true }).click();
  const general = await page.getByRole('dialog').evaluate((node) => ({
    height: node.getBoundingClientRect().height,
    surface: node.querySelector('.settings-surface').getBoundingClientRect().height,
    width: node.getBoundingClientRect().width,
  }));
  assert.ok(general.width <= 640);
  assert.ok(Math.abs(general.height - general.surface) < 2, JSON.stringify(general));
  const segment = await page.locator('.settings-segment').evaluate((node) => ({
    padding: window.paintStyle(node).padding,
    gap: window.paintStyle(node).gap,
    children: [...node.querySelectorAll('button')].map((child) => ({
      width: child.getBoundingClientRect().width,
      left: child.getBoundingClientRect().left,
      right: child.getBoundingClientRect().right,
      radius: window.paintStyle(child).borderRadius,
    })),
  }));
  assert.equal(segment.padding, '0px');
  assert.equal(segment.gap, '0px');
  assert.ok(Math.abs(segment.children[0].width - segment.children[1].width) < 1);
  assert.equal(segment.children[0].right, segment.children[1].left);
  assert.equal(segment.children[1].radius, '0px');
  await page.getByRole('button', { name: 'Camera', exact: true }).click();
  const camera = await page.getByRole('dialog').boundingBox();
  assert.equal(camera.height, general.height);
  assert.equal(camera.width, general.width);
  await page.getByRole('button', { name: 'Close', exact: true }).click();
});

test('settings stays inside the viewport at200% shell scale and shared buttons retain native hit/focus behavior', async () => {
  await page.evaluate(() => (document.getElementById('root').style.zoom = '2'));
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  const bounds = await page.getByRole('dialog').evaluate((node) => ({
    x: node.getBoundingClientRect().x,
    y: node.getBoundingClientRect().y,
    right: node.getBoundingClientRect().right,
    bottom: node.getBoundingClientRect().bottom,
    w: innerWidth,
    h: innerHeight,
  }));
  assert.ok(
    bounds.x >= 0 && bounds.y >= 0 && bounds.right <= bounds.w && bounds.bottom <= bounds.h,
    JSON.stringify(bounds),
  );
  await page.getByRole('button', { name: 'General', exact: true }).click();
  await page.getByRole('combobox', { name: 'Language', exact: true }).click();
  const popup = await page.locator('.select-popover').evaluate((node) => {
    const rect = node.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      width: innerWidth,
      height: innerHeight,
    };
  });
  assert.ok(
    popup.left >= 0 && popup.top >= 0 && popup.right <= popup.width && popup.bottom <= popup.height,
    JSON.stringify(popup),
  );
  await page.getByRole('option', { name: 'English', exact: true }).click();
  const search = page.getByRole('textbox', { name: 'Search all settings', exact: true });
  await search.click();
  await settlePaint(page);
  const focus = await search.evaluate((node) => ({
    inner: window.paintStyle(node).outlineStyle,
    outer: window.paintStyle(node.parentElement).outlineColor,
    border: window.paintStyle(document.documentElement).getPropertyValue('--ui-border').trim(),
    surface: node.parentElement.getAttribute('data-smooth-surface'),
    paint: node.parentElement.style.getPropertyValue('--smooth-paint'),
  }));
  assert.equal(focus.inner, 'none');
  assert.equal(focus.outer, 'rgba(0, 0, 0, 0.15)');
  assert.ok(focus.surface);
  assert.match(decodeURIComponent(focus.paint), /<path[^>]+d=/);
  const close = page.getByRole('dialog').getByRole('button', { name: 'Close', exact: true });
  await close.focus();
  assert.equal(await close.evaluate((node) => document.activeElement === node), true);
  assert.equal(await close.evaluate((node) => window.paintStyle(node).clipPath), 'none');
  assert.ok(
    Math.abs(
      (await close.evaluate((node) => parseFloat(window.paintStyle(node).borderRadius))) - 4,
    ) < 0.001,
  );
  await close.press('Enter');
  assert.equal(await page.getByRole('dialog').count(), 0);
  await page.evaluate(() => (document.getElementById('root').style.zoom = '1'));
});
test('current camera, light, timeline and export captions remain localized without confusing appearance and entity names', async () => {
  const result = await page.evaluate(async () => {
    const { message } = await import('/src/i18n.ts');
    const { t, normalizeSettings } = window.uiModules;
    return {
      captions: [
        'System',
        'Light',
        'Light source',
        message('Added {{kind}}', { kind: message('light') }),
        message('{{label}} {{axis}}', { label: message('Eye'), axis: 'X' }),
        message('{{label}} {{axis}}', { label: message('Target'), axis: 'Z' }),
        'Selected frustum / light range',
        'Follow locked',
        'Record from playhead',
        'Light direction',
        'Save camera image',
        'Video exceeds the 500 MiB export limit.',
      ].map((text) => t(text, 'zh')),
      defaults: normalizeSettings({}).showEntityInfluence,
      persisted: normalizeSettings({ showEntityInfluence: false }).showEntityInfluence,
    };
  });
  assert.deepEqual(result.captions, [
    '系统',
    '浅色',
    '灯光',
    '已添加灯光',
    '视点 X',
    '目标 Z',
    '选中相机视锥与灯光范围',
    '锁定跟随',
    '从播放头开始录制',
    '灯光方向',
    '保存相机图像',
    '视频超过 500 MiB 的导出限制。',
  ]);
  assert.equal(result.defaults, true);
  assert.equal(result.persisted, false);
});

test('corner smoothing updates G2 surface paint without changing authored radii or layout and persists', async () => {
  const view = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  await view.addInitScript(() =>
    localStorage.setItem('mojive.electron.settings.v1', JSON.stringify({ language: 'en' })),
  );
  await view.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-settings-review.html`,
  );
  await view.getByRole('button', { name: 'Settings', exact: true }).click();
  await view
    .getByRole('textbox', { name: 'Search all settings', exact: true })
    .fill('Corner smoothing');
  const field = view.getByRole('textbox', { name: 'Corner smoothing', exact: true });
  await field.waitFor();
  assert.equal(await field.inputValue(), '60');
  const inspect = () =>
    view.locator('.settings-surface').evaluate((element) => ({
      radii: JSON.parse(element.dataset.smoothRadii),
      amount: Number(element.dataset.smoothAmount),
      paint: element.style.getPropertyValue('--smooth-paint'),
      width: element.getBoundingClientRect().width,
      height: element.getBoundingClientRect().height,
    }));
  const initial = await inspect();
  for (const value of [0, 100, 35]) {
    await field.fill(String(value));
    await field.press('Enter');
    await view.waitForFunction(
      (value) =>
        Number(document.querySelector('.settings-surface')?.dataset.smoothAmount) === value / 100,
      value,
    );
    const state = await inspect();
    assert.deepEqual(state.radii, [16, 16, 16, 16]);
    assert.equal(state.width, initial.width);
    assert.equal(state.height, initial.height);
    assert.notEqual(state.paint, initial.paint);
    assert.equal(
      await view.evaluate(
        () => JSON.parse(localStorage.getItem('mojive.electron.settings.v1')).cornerSmoothing,
      ),
      value,
    );
  }
  // Reload without the initial-setting script overwriting the saved preference.
  const saved = await view.evaluate(() => localStorage.getItem('mojive.electron.settings.v1'));
  assert.equal(JSON.parse(saved).cornerSmoothing, 35);
  assert.equal(await view.evaluate(() => window.uiModules.readSettings().cornerSmoothing), 35);
  const search = view.getByRole('textbox', { name: 'Search all settings', exact: true });
  await search.fill('Capsule smoothing');
  const capsule = view.getByRole('textbox', { name: 'Capsule smoothing', exact: true });
  assert.equal(await capsule.inputValue(), '38.2');
  await capsule.fill('25');
  await capsule.press('Enter');
  await view.waitForFunction(
    () => Number(document.querySelector('.settings-search')?.dataset.smoothAmount) === 0.25,
  );
  assert.equal((await inspect()).amount, 0.35);
  assert.equal(await view.evaluate(() => window.uiModules.readSettings().capsuleSmoothing), 25);
  await search.fill('');
  await view.getByRole('button', { name: 'Restore defaults', exact: true }).click();
  await settlePaint(view);
  assert.equal(await view.evaluate(() => window.uiState.settings.cornerSmoothing), 60);
  assert.equal(await view.evaluate(() => window.uiState.settings.capsuleSmoothing), 38.2);
  await view.close();
});
