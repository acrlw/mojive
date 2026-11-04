import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: path.resolve('.'),
    cacheDir: 'node_modules/.vite-layers-test',
    server: { host: '127.0.0.1', port: 0 },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime'] },
    logLevel: 'error',
  });
  server.middlewares.use('/__layers_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><button id="opener"><span data-icon="layers"></span>Layers</button><input id="outside" aria-label="Outside input"><div id="fixture" style="position:relative;width:320px;height:440px"></div>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 600, height: 550 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__layers_test`);
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite-layers-test/deps/react.js'),
      { default: ReactDOM } = await import(
        '/node_modules/.vite-layers-test/deps/react-dom_client.js'
      ),
      { LayersPopup, DEFAULT_VIEWPORT_LAYERS } = await import('/src/ViewportLayers.tsx'),
      { setLanguage } = await import('/src/i18n.ts');
    await import('/src/design-tokens.css');
    await import('/src/styles.css');
    const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
    applySettings({ ...DEFAULT_SETTINGS, language: 'en' });
    setLanguage('en');
    await document.fonts.load('14px MiSans');
    const root = ReactDOM.createRoot(document.getElementById('fixture'));
    const f = (window.layerFixture = {
      value: { ...DEFAULT_VIEWPORT_LAYERS },
      open: true,
      changes: [],
    });
    f.render = () =>
      root.render(
        f.open
          ? React.createElement(
              'div',
              { className: 'viewport-layers-host' },
              React.createElement(LayersPopup, {
                value: f.value,
                onChange: (value) => {
                  f.value = value;
                  f.changes.push(value);
                  f.render();
                },
                onClose: () => {
                  f.open = false;
                  f.render();
                },
              }),
            )
          : null,
      );
    document.getElementById('opener').onclick = () => {
      f.open = !f.open;
      f.render();
    };
    document.getElementById('opener').focus();
    f.render();
  });
  await page.getByRole('dialog', { name: 'Viewport layers' }).waitFor();
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('native layer groups use real shared switches and reset controlled visibility', async () => {
  const switches = page.getByRole('switch');
  assert.equal(await switches.count(), 7);
  await page.getByRole('switch', { name: 'Grid', exact: true }).click();
  assert.equal(
    await page.getByRole('switch', { name: 'Grid', exact: true }).getAttribute('aria-checked'),
    'false',
  );
  assert.equal(await page.evaluate(() => window.layerFixture.value.grid), false);
  await page.getByRole('button', { name: 'Reset viewport layers', exact: true }).click();
  assert.equal(
    await page.getByRole('switch', { name: 'Grid', exact: true }).getAttribute('aria-checked'),
    'true',
  );
  await page
    .getByRole('dialog', { name: 'Viewport layers' })
    .screenshot({ path: '../output/electron/verification/viewport-layers.png' });
});
test('Escape restores the opener, while outside dismissal preserves native input focus', async () => {
  await page.getByRole('switch', { name: 'Grid', exact: true }).press('Escape');
  await page.getByRole('dialog', { name: 'Viewport layers' }).waitFor({ state: 'detached' });
  assert.equal(
    await page.locator('#opener').evaluate((element) => document.activeElement === element),
    true,
  );
  await page.locator('#opener').click();
  await page.getByRole('dialog', { name: 'Viewport layers' }).waitFor();
  await page.getByLabel('Outside input').click();
  await page.getByRole('dialog', { name: 'Viewport layers' }).waitFor({ state: 'detached' });
  assert.equal(
    await page
      .getByLabel('Outside input')
      .evaluate((element) => document.activeElement === element),
    true,
  );
});
test('short narrow viewport constrains the popup and keeps every layer reachable by scrolling', async () => {
  await page.evaluate(() => {
    document.getElementById('fixture').style.width = '220px';
    document.getElementById('fixture').style.height = '210px';
  });
  await page.locator('#opener').click();
  const popup = page.getByRole('dialog', { name: 'Viewport layers' });
  await popup.waitFor();
  const size = await popup.evaluate((element) => ({
    width: element.clientWidth,
    height: element.clientHeight,
    scroll: element.scrollHeight,
  }));
  assert.ok(size.width <= 196);
  assert.ok(size.height <= 190);
  assert.ok(size.scroll > size.height);
  await page.getByRole('switch', { name: 'Selection highlight', exact: true }).click();
  assert.equal(await page.evaluate(() => window.layerFixture.value.selection), false);
});
