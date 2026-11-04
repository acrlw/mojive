import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, base;
const master = await readFile(new URL('../design/icons/camera-align.svg', import.meta.url), 'utf8');
before(async () => {
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-camera-align-test',
    plugins: [
      {
        name: 'camera-align-review',
        configureServer(server) {
          server.middlewares.use('/__camera-align-icons', (_, res) => {
            res.setHeader('Content-Type', 'text/html');
            res.end('<!doctype html><div id="gallery"></div>');
          });
        },
      },
    ],
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime', 'three'] },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
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

test('camera alignment has an original canonical master and a reproducible generated public asset', async () => {
  const local = JSON.parse(
    await readFile(new URL('../design/icons/sources.json', import.meta.url), 'utf8'),
  );
  const native = JSON.parse(
    await readFile(
      new URL('../../examples/ui_design/native/svg/sources.json', import.meta.url),
      'utf8',
    ),
  );
  assert.deepEqual(local.style, native.style);
  assert.equal(local.icons['camera-align'], 'camera-align.svg');
  assert.equal(
    await readFile(new URL('../public/icons/camera-align.svg', import.meta.url), 'utf8'),
    master,
  );
  const provenance = JSON.parse(
    await readFile(new URL('../public/icons/sources.json', import.meta.url), 'utf8'),
  );
  assert.deepEqual(provenance.electron.icons, local.icons);
  assert.equal(
    native.icons['camera-align'],
    undefined,
    'Electron extension must not silently change the native icon inventory',
  );
});

test('actual shared SVG controls preserve the camera aperture, stroke and grid at 16/20/24 and fractional scales', async () => {
  for (const appearance of ['dark', 'light'])
    for (const deviceScaleFactor of [1, 2]) {
      const page = await browser.newPage({
        viewport: { width: 1120, height: 1250 },
        deviceScaleFactor,
      });
      try {
        await page.goto(base + '/__camera-align-icons');
        const result = await page.evaluate(
          async ({ appearance, deviceScaleFactor }) => {
            const { default: React } = await import(
              '/node_modules/.vite-camera-align-test/deps/react.js'
            );
            const { default: DOM } = await import(
              '/node_modules/.vite-camera-align-test/deps/react-dom_client.js'
            );
            const { Icon, Button } = await import('/src/components.tsx');
            const { icons } = await import('/src/icons.ts');
            const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
            await import('/src/design-tokens.css');
            await import('/src/styles.css');
            applySettings({ ...DEFAULT_SETTINGS, language: 'en', appearance });
            document.body.style.overflow = 'auto';
            const style = document.createElement('style');
            style.textContent = `
          #gallery{padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:16px;color:var(--muted-color)}
          .icon-scale{padding:12px;background:var(--floating-color);border-radius:12px}
          .icon-scale h2{margin:0 0 12px;font-size:14px;font-weight:600;color:var(--text-color)}
          .icon-matrix{display:grid;grid-template-columns:110px repeat(4,1fr);gap:4px;align-items:center}
          .icon-label{font-size:13px}.icon-size{font:12px var(--font-mono);text-align:center}
          .icon-cell{height:88px;display:flex;align-items:center;justify-content:center}.icon-cell .button{flex-shrink:0}
        `;
            document.head.append(style);
            const names = ['camera', 'camera-align', 'fit', 'reset', 'perspective'],
              sizes = [16, 20, 24, 48],
              scales = [0.75, 1, 1.25, 1.5];
            const sections = scales.map((scale) =>
              React.createElement(
                'section',
                { className: 'icon-scale', key: scale },
                React.createElement(
                  'h2',
                  null,
                  `${appearance} · ${Math.round(scale * 100)}% · DPR ${deviceScaleFactor}`,
                ),
                React.createElement(
                  'div',
                  { className: 'icon-matrix' },
                  React.createElement('span', null),
                  ...sizes.map((size) =>
                    React.createElement(
                      'span',
                      { key: 'size' + size, className: 'icon-size' },
                      size,
                    ),
                  ),
                  ...names.flatMap((name) => [
                    React.createElement('span', { key: name, className: 'icon-label' }, name),
                    ...sizes.map((size) =>
                      React.createElement(
                        'div',
                        { className: 'icon-cell', key: name + size },
                        React.createElement(
                          'div',
                          { style: { zoom: scale } },
                          React.createElement(
                            Button,
                            {
                              title: name,
                              style: { width: size + 8, height: size + 8 },
                              'data-size': size,
                              'data-scale': scale,
                            },
                            React.createElement(Icon, { name, size }),
                          ),
                        ),
                      ),
                    ),
                  ]),
                ),
              ),
            );
            DOM.createRoot(document.getElementById('gallery')).render(
              React.createElement(React.Fragment, null, ...sections),
            );
            const apertures = [];
            for (const scale of scales)
              for (const size of [16, 20, 24]) {
                const width = size * scale * deviceScaleFactor,
                  svg = icons['camera-align'].replaceAll('currentColor', '#ffffff');
                const image = new Image();
                image.src = 'data:image/svg+xml;base64,' + btoa(svg);
                await image.decode();
                const canvas = document.createElement('canvas');
                canvas.width = canvas.height = Math.ceil(width);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(image, 0, 0, width, width);
                apertures.push({
                  size,
                  scale,
                  alpha: ctx.getImageData(
                    Math.floor((8.5 / 24) * width),
                    Math.floor((16.5 / 24) * width),
                    1,
                    1,
                  ).data[3],
                });
              }
            return {
              source: icons['camera-align'],
              distinct:
                icons['camera-align'] !== icons.reset && icons['camera-align'] !== icons.camera,
              apertures,
            };
          },
          { appearance, deviceScaleFactor },
        );
        assert.equal(result.source, master);
        assert.equal(result.distinct, true);
        for (const sample of result.apertures)
          assert.ok(
            sample.alpha < 128,
            `Lens aperture remains open: ${JSON.stringify({ appearance, deviceScaleFactor, ...sample })}`,
          );
        await page.locator('[data-icon="camera-align"]').first().waitFor();
        await page.evaluate(() => document.fonts.ready);
        const metrics = await page.locator('[data-icon="camera-align"]').evaluateAll((nodes) =>
          nodes.map((node) => {
            const svg = node.querySelector('svg'),
              b = svg.getBBox(),
              r = node.getBoundingClientRect(),
              button = node.closest('button');
            return {
              size: Number(button.dataset.size),
              scale: Number(button.dataset.scale),
              width: r.width,
              height: r.height,
              grid: svg.getAttribute('viewBox'),
              stroke: svg.getAttribute('stroke-width'),
              cap: svg.getAttribute('stroke-linecap'),
              join: svg.getAttribute('stroke-linejoin'),
              bbox: [b.x, b.y, b.width, b.height],
              transforms: svg.querySelectorAll('[transform]').length,
            };
          }),
        );
        for (const item of metrics) {
          assert.equal(item.width, item.size * item.scale);
          assert.equal(item.height, item.width);
          assert.equal(item.grid, '0 0 24 24');
          assert.equal(item.stroke, '1.5');
          assert.equal(item.cap, 'round');
          assert.equal(item.join, 'round');
          assert.equal(item.transforms, 0);
          assert.deepEqual(item.bbox, [2.5, 3, 18.5, 18]);
        }
        await page.locator('[data-icon="camera-align"]').nth(5).hover();
        await page.locator('#gallery').screenshot({
          path: `../output/electron/verification/camera-align-icons-${appearance}-${deviceScaleFactor}x.png`,
        });
      } finally {
        await page.close();
      }
    }
});

test('actual camera Inspector and camera menu use the new alignment glyph in rest and hover states', async () => {
  for (const appearance of ['dark', 'light']) {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    try {
      await page.addInitScript((appearance) => {
        localStorage.setItem(
          'mojive.electron.settings.v1',
          JSON.stringify({ language: 'en', appearance, uiScale: 1.25 }),
        );
        localStorage.setItem(
          'mojive.layout',
          JSON.stringify({ left: 280, right: 360, bottom: 260 }),
        );
      }, appearance);
      await page.goto(base);
      await page.getByRole('treeitem').first().waitFor();
      await page.locator('.activity-rail').getByTitle('Cameras', { exact: true }).click();
      const card = page.locator('.camera-entry').first();
      await card.locator('.camera-entry-name').click();
      const align = page.locator('.camera-properties').getByTitle('Align to view', { exact: true });
      await align.waitFor();
      assert.equal(await align.locator('[data-icon="camera-align"]').count(), 1);
      assert.equal(await align.locator('[data-icon="reset"]').count(), 0);
      await page
        .locator('.camera-properties .section')
        .first()
        .screenshot({ path: `../output/electron/verification/camera-align-inspector-${appearance}-rest.png` });
      await align.hover();
      await page
        .locator('.camera-properties .section')
        .first()
        .screenshot({ path: `../output/electron/verification/camera-align-inspector-${appearance}-hover.png` });
      await card.getByRole('button', { name: 'More camera actions', exact: true }).click();
      const menu = page.getByRole('menu', { name: /More camera actions/ }),
        menuAlign = menu.getByRole('menuitem', { name: 'Align to view', exact: true });
      assert.equal(await menuAlign.locator('[data-icon="camera-align"]').count(), 1);
      await menuAlign.hover();
      await menu.screenshot({ path: `../output/electron/verification/camera-align-menu-${appearance}-hover.png` });
      assert.deepEqual(errors, []);
    } finally {
      await page.close();
    }
  }
});
