import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page, base;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-notice-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime'] },
    logLevel: 'error',
  });
  server.middlewares.use('/__notice', (_, res) => {
    res.setHeader('Content-Type', 'text/html');
    res.end(
      '<!doctype html><div id="fixture" style="position:relative;margin:40px;width:700px;height:360px;background:var(--panel-color)"></div><button id="outside">Outside</button>',
    );
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}/__notice`;
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  page = await browser.newPage({ viewport: { width: 1000, height: 600 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.setDefaultTimeout(5000);
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.clock.install({ time: new Date('2026-11-04T00:00:00Z') });
  await page.goto(base);
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite-notice-test/deps/react.js');
    const { default: DOM } = await import(
      '/node_modules/.vite-notice-test/deps/react-dom_client.js'
    );
    const { OperationNotice } = await import('/src/OperationNotice.tsx');
    const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
    await import('/src/design-tokens.css');
    await import('/src/styles.css');
    applySettings({ ...DEFAULT_SETTINGS, language: 'en', appearance: 'dark' });
    window.notices = { actions: [], dismissed: [], keys: [] };
    window.settings = (patch) => applySettings({ ...DEFAULT_SETTINGS, ...patch });
    window.addEventListener('keydown', (event) => window.notices.keys.push(event.key));
    function Harness() {
      const [notice, setNotice] = React.useState(null);
      window.publish = ({ id, message, action, level }) =>
        setNotice({
          id,
          message,
          level,
          action: action
            ? {
                label: action,
                onClick: () => {
                  window.notices.actions.push({ id, action });
                  window.publish({ id: id + 1, message: 'Edit undone', action: 'Redo' });
                },
              }
            : undefined,
        });
      return (
        notice &&
        React.createElement(OperationNotice, {
          notice,
          onDismiss: (id) => {
            window.notices.dismissed.push(id);
            setNotice((current) => (current?.id === id ? null : current));
          },
        })
      );
    }
    DOM.createRoot(document.getElementById('fixture')).render(React.createElement(Harness));
  });
  await page.waitForFunction(() => typeof window.publish === 'function');
  await page.clock.pauseAt(new Date('2026-11-04T01:00:00Z'));
  await page.mouse.move(900, 10);
});
afterEach(async () => {
  await page?.close();
});
async function publish(data) {
  await page.evaluate((data) => window.publish(data), data);
  await page.locator(`[data-notice-id="${data.id}"]`).waitFor();
}

test('notice lifetime pauses under the pointer and resumes its remaining duration', async () => {
  await publish({ id: 1, message: 'Added box', action: 'Undo' });
  await page.clock.runFor(1000);
  await page.getByRole('status').hover();
  await page.clock.runFor(10000);
  assert.equal(await page.getByRole('status').count(), 1);
  await page.mouse.move(900, 10);
  await page.clock.runFor(3000);
  assert.equal(await page.getByRole('status').count(), 1);
  await page.clock.runFor(500);
  await page.getByRole('status').waitFor({ state: 'detached' });
  assert.deepEqual(await page.evaluate(() => window.notices.dismissed), [1]);
});

test('keyboard focus pauses expiry across action buttons and owns edit keys and Escape', async () => {
  await publish({ id: 1, message: 'Deleted arm', action: 'Undo' });
  await page.clock.runFor(1000);
  await page.getByRole('button', { name: 'Undo', exact: true }).focus();
  await page.clock.runFor(10000);
  await page.keyboard.press('Delete');
  assert.deepEqual(await page.evaluate(() => window.notices.keys), []);
  await page.keyboard.press('Tab');
  assert.equal(
    await page
      .getByRole('button', { name: 'Dismiss notification' })
      .evaluate((el) => el === document.activeElement),
    true,
  );
  await page.clock.runFor(10000);
  assert.equal(await page.getByRole('status').count(), 1);
  await page.locator('#outside').focus();
  await page.clock.runFor(3499);
  assert.equal(await page.getByRole('status').count(), 1);
  await page.clock.runFor(1);
  await page.getByRole('status').waitFor({ state: 'detached' });
  await publish({ id: 2, message: 'Edit restored' });
  await page.getByRole('button', { name: 'Dismiss notification' }).focus();
  await page.keyboard.press('Escape');
  await page.getByRole('status').waitFor({ state: 'detached' });
  assert.deepEqual(await page.evaluate(() => window.notices.dismissed), [1, 2]);
});

test('actions can replace their notice without an old timer dismissing the replacement', async () => {
  await publish({ id: 1, message: 'Added box', action: 'Undo' });
  await page.clock.runFor(4000);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await page.locator('[data-notice-id="2"]').waitFor();
  assert.deepEqual(await page.evaluate(() => window.notices.actions), [{ id: 1, action: 'Undo' }]);
  assert.equal(await page.getByRole('button', { name: 'Redo', exact: true }).count(), 1);
  await page.mouse.move(900, 10);
  await page.locator('#outside').focus();
  await page.clock.runFor(500);
  assert.equal(await page.locator('[data-notice-id="2"]').count(), 1);
  await page.clock.runFor(3999);
  assert.equal(await page.locator('[data-notice-id="2"]').count(), 1);
  await page.clock.runFor(1);
  await page.getByRole('status').waitFor({ state: 'detached' });
  assert.deepEqual(await page.evaluate(() => window.notices.dismissed), [2]);
});

test('a repeated message has a fresh publication lifetime and removing its action does not replay it', async () => {
  await publish({ id: 1, message: 'Edit undone' });
  await page.clock.runFor(4000);
  await publish({ id: 2, message: 'Edit undone' });
  await page.clock.runFor(500);
  assert.equal(await page.getByRole('status').count(), 1);
  await page.clock.runFor(4000);
  await page.getByRole('status').waitFor({ state: 'detached' });
  assert.deepEqual(await page.evaluate(() => window.notices.dismissed), [2]);
  await publish({ id: 3, message: 'Deleted arm', action: 'Undo' });
  await publish({ id: 3, message: 'Deleted arm' });
  assert.equal(await page.getByRole('button', { name: 'Undo', exact: true }).count(), 0);
  assert.deepEqual(await page.evaluate(() => window.notices.actions), []);
});

test('native severity, localized messages and corner placement remain legible in both themes and narrow viewports', async () => {
  for (const appearance of ['dark', 'light'])
    for (const language of ['en', 'zh']) {
      await page.evaluate(({ appearance, language }) => window.settings({ appearance, language }), {
        appearance,
        language,
      });
      for (const width of [700, 280]) {
        await page.locator('#fixture').evaluate((el, width) => {
          el.style.width = width + 'px';
          el.style.zoom = '1.25';
        }, width);
        await publish({
          id: `${appearance}-${language}-${width}`,
          message: 'Edit undone',
          action: 'Redo',
        });
        assert.equal(
          await page.locator('.operation-notice-message').textContent(),
          language === 'zh' ? '已撤销操作' : 'Edit undone',
        );
        const metrics = await page.getByRole('status').evaluate((el) => {
          const r = el.getBoundingClientRect(),
            parent = el.parentElement.getBoundingClientRect(),
            s = window.paintStyle(el);
          const button = el.querySelector('.operation-notice-action').getBoundingClientRect();
          return {
            left: r.left - parent.left,
            bottom: parent.bottom - r.bottom,
            width: r.width,
            parentWidth: parent.width,
            height: r.height,
            radius: s.borderRadius,
            fontSize: s.fontSize,
            color: s.color,
            buttonHeight: button.height,
            transform: s.transform,
            role: el.getAttribute('aria-live'),
          };
        });
        assert.equal(metrics.left, 15);
        assert.equal(metrics.bottom, 15);
        assert.ok(metrics.width <= Math.min(440 * 1.25, metrics.parentWidth - 30));
        assert.equal(metrics.height, 55);
        assert.equal(metrics.radius, '12px');
        assert.equal(metrics.buttonHeight, 35);
        assert.equal(metrics.fontSize, '14px');
        assert.equal(metrics.transform, 'none');
        assert.equal(metrics.role, 'polite');
        assert.equal(await page.evaluate(() => document.activeElement.id), '');
        await page.locator('#fixture').screenshot({
          path: `../output/electron/verification/operation-notice-${appearance}-${language}-${width}.png`,
        });
      }
    }
  await publish({
    id: 'warning',
    message: 'Stop recording before changing scene structure.',
    level: 'warning',
  });
  assert.equal(await page.getByRole('status').getAttribute('data-level'), 'warning');
  assert.equal(await page.locator('.operation-notice [data-icon="warn"]').count(), 1);
  await publish({ id: 'error', message: 'A texture image could not be decoded.', level: 'error' });
  assert.equal(await page.locator('.operation-notice [data-icon="err"]').count(), 1);
});
