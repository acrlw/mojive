import { installPaintStyle } from './helpers/surface-style.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer as createTcpServer } from 'node:net';
import { mkdtemp, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createServer } from 'vite';
import { _electron } from 'playwright';
test('Electron picker Copy/Paste traverses the explicit preload bridge and CSS corners preserve overflow', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const socket = createTcpServer();
  await new Promise((resolve) => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise((resolve) => socket.close(resolve));
  const server = await createServer({
    root,
    plugins: [
      {
        name: 'control-fixture',
        configureServer(server) {
          server.middlewares.use((req, _res, next) => {
            if (req.url === '/') req.url = '/tests/harness/ui-controls-review.html';
            next();
          });
        },
      },
    ],
    server: { host: '127.0.0.1', port, strictPort: true },
    logLevel: 'error',
  });
  const profile = await mkdtemp(
    path.join(root, '../output/electron/verification/controls-profile-'),
  );
  let app;
  try {
    await server.listen();
    app = await _electron.launch({
      args: [path.join(root, 'main/main.cjs'), '--hidden', '--user-data-dir=' + profile],
      env: { ...process.env, MOJIVE_DEV_URL: `http://127.0.0.1:${port}` },
    });
    const page = await app.firstWindow();
    await page.addInitScript(installPaintStyle);
    await page.evaluate(installPaintStyle);
    page.setDefaultTimeout(5000);
    await page.waitForFunction(() => window.controlState && window.mojive);
    // Keep the real IPC route while isolating the test from the user's OS clipboard.
    await app.evaluate(({ clipboard }) => {
      globalThis.colorTestClipboard = '#336699';
      clipboard.readText = () => globalThis.colorTestClipboard;
      clipboard.writeText = (text) => {
        globalThis.colorTestClipboard = text;
      };
    });
    await page.getByRole('button', { name: 'Material color', exact: true }).click();
    await page.getByRole('button', { name: 'Paste', exact: true }).click();
    await page.waitForFunction(() => window.controlState.color === '#336699');
    await page.getByRole('textbox', { name: 'Hex color', exact: true }).fill('#123456');
    await page.getByRole('textbox', { name: 'Hex color', exact: true }).press('Enter');
    await page.getByRole('button', { name: 'Copy', exact: true }).click();
    await page.waitForFunction(async () => (await window.mojive.readClipboardText()) === '#123456');
    assert.equal(await app.evaluate(() => globalThis.colorTestClipboard), '#123456');
    await page.getByRole('button', { name: 'Done', exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.controlEvents), [
      'color begin',
      'color commit',
    ]);
    const shape = await page.evaluate(() => {
      const frame = document.createElement('div');
      frame.style.cssText =
        'position:absolute;left:500px;top:100px;width:80px;height:40px;border-radius:8px;background:red;box-shadow:0 0 8px black';
      const child = document.createElement('button');
      child.textContent = 'Overflow test';
      child.style.cssText = 'position:absolute;left:100px;top:0;width:120px;height:28px';
      frame.append(child);
      document.body.append(frame);
      return new Promise((resolve) =>
        requestAnimationFrame(() =>
          requestAnimationFrame(() =>
            resolve({
              corner: window.paintStyle(frame).cornerShape,
              clip: window.paintStyle(frame).clipPath,
              hit: document.elementFromPoint(620, 114) === child,
            }),
          ),
        ),
      );
    });
    assert.match(shape.corner, /^(round|superellipse\(1\))$/);
    assert.equal(shape.clip, 'none');
    assert.equal(shape.hit, true);
  } finally {
    await app?.close();
    await server.close();
    await rm(profile, { recursive: true, force: true });
  }
});
