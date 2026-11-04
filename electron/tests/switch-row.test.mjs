import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: fileURLToPath(new URL('../', import.meta.url)),
    cacheDir: 'node_modules/.vite-switch-row-test',
    server: { host: '127.0.0.1', port: 0, watch: null },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime'] },
    logLevel: 'error',
  });
  server.middlewares.use('/__switch_row_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><main style="width:360px;margin:24px"><button id="before">Before</button><div id="fixture"></div><button id="after">After</button></main>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 440, height: 300 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__switch_row_test`);
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite-switch-row-test/deps/react.js'),
      { default: ReactDOM } = await import(
        '/node_modules/.vite-switch-row-test/deps/react-dom_client.js'
      ),
      { SwitchRow } = await import('/src/CustomControls.tsx'),
      { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
    await import('/src/design-tokens.css');
    await import('/src/styles.css');
    await import('/src/controls.css');
    const root = ReactDOM.createRoot(document.getElementById('fixture'));
    const fixture = (window.switchRowFixture = {
      checked: false,
      disabled: false,
      changes: [],
      label: 'Show grid',
      name: undefined,
      title: undefined,
    });
    fixture.render = () =>
      root.render(
        React.createElement(SwitchRow, {
          id: 'grid-switch',
          className: 'test-switch-row',
          label: fixture.label,
          'aria-label': fixture.name,
          title: fixture.title,
          checked: fixture.checked,
          disabled: fixture.disabled,
          onChange: (checked) => {
            fixture.checked = checked;
            fixture.changes.push(checked);
            fixture.render();
          },
        }),
      );
    fixture.theme = (appearance, language = 'en') => {
      applySettings({ ...DEFAULT_SETTINGS, appearance, language });
      fixture.render();
    };
    fixture.theme('dark');
    await document.fonts.load('14px MiSans');
  });
  await page.getByRole('switch', { name: 'Show grid', exact: true }).waitFor();
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('label, row whitespace and track activate the same switch once', async () => {
  const row = page.getByRole('switch', { name: 'Show grid', exact: true });
  assert.equal(await row.evaluate((node) => node.tagName), 'BUTTON');
  assert.equal(await row.locator('button,input,[tabindex]').count(), 0);
  assert.equal(await row.locator('.custom-switch').getAttribute('aria-hidden'), 'true');
  assert.equal(await row.getAttribute('title'), null);
  assert.equal(await row.getAttribute('id'), 'grid-switch');
  await row.locator('.switch-row-label').click();
  await row.click({ position: { x: 180, y: 14 } });
  await row.locator('.custom-switch').click();
  assert.deepEqual(await page.evaluate(() => window.switchRowFixture.changes), [true, false, true]);
  assert.equal(await row.getAttribute('aria-checked'), 'true');
});

test('one tab stop owns native Space and Enter activation and a single focus outline', async () => {
  await page.evaluate(() => (window.switchRowFixture.changes = []));
  await page.locator('#before').focus();
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'grid-switch');
  const row = page.locator('#grid-switch');
  const outlines = await row.evaluate((node) =>
    [node, ...node.querySelectorAll('*')].map((item) => getComputedStyle(item).outlineStyle),
  );
  assert.deepEqual(outlines, ['solid', 'none', 'none', 'none']);
  await page.keyboard.press('Space');
  await page.keyboard.press('Enter');
  assert.deepEqual(await page.evaluate(() => window.switchRowFixture.changes), [false, true]);
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'after');
  await page.keyboard.press('Shift+Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'grid-switch');
});

test('disabled row cannot activate from any pointer region or enter the tab sequence', async () => {
  await page.evaluate(() => {
    const fixture = window.switchRowFixture;
    fixture.disabled = true;
    fixture.changes = [];
    fixture.render();
  });
  const row = page.locator('#grid-switch');
  await page.waitForFunction(() => document.getElementById('grid-switch').disabled);
  for (const x of [10, 180, 330]) await row.click({ position: { x, y: 14 }, force: true });
  assert.deepEqual(await page.evaluate(() => window.switchRowFixture.changes), []);
  assert.equal(await row.getAttribute('aria-checked'), 'true');
  await page.locator('#before').focus();
  await page.keyboard.press('Tab');
  assert.equal(await page.evaluate(() => document.activeElement.id), 'after');
  await page.evaluate(() => {
    window.switchRowFixture.disabled = false;
    window.switchRowFixture.render();
  });
});

test('translated captions, explicit names and hints retain their separate semantics', async () => {
  await page.evaluate(() => window.switchRowFixture.theme('dark', 'zh'));
  await page.getByRole('switch', { name: '显示网格', exact: true }).waitFor();
  await page.evaluate(() => {
    const fixture = window.switchRowFixture;
    fixture.name = 'Grid';
    fixture.title = 'Show grid';
    fixture.render();
  });
  const row = page.getByRole('switch', { name: '网格', exact: true });
  await row.waitFor();
  assert.equal(await row.locator('.switch-row-label').textContent(), '显示网格');
  assert.equal(await row.getAttribute('title'), '显示网格');
  await page.evaluate(() => {
    const fixture = window.switchRowFixture;
    fixture.name = undefined;
    fixture.title = undefined;
    fixture.theme('dark');
  });
});

test('whole-row hover and hold preserve native switch geometry and paint in both themes', async () => {
  const row = page.locator('#grid-switch');
  const visual = () =>
    row.evaluate((node) => {
      const track = node.querySelector('.custom-switch'),
        thumb = track.firstElementChild,
        bounds = (item) => {
          const { x, y, width, height, right } = item.getBoundingClientRect();
          return { x, y, width, height, right };
        },
        canvas = document.createElement('canvas'),
        context = canvas.getContext('2d');
      context.fillStyle = getComputedStyle(track).backgroundColor;
      context.fillRect(0, 0, 1, 1);
      return {
        row: bounds(node),
        track: bounds(track),
        thumb: bounds(thumb),
        color: [...context.getImageData(0, 0, 1, 1).data].slice(0, 3),
        thumbColor: getComputedStyle(thumb).backgroundColor,
        shape: getComputedStyle(thumb).cornerShape,
        radius: getComputedStyle(thumb).borderRadius,
        rowBackground: getComputedStyle(node).backgroundColor,
      };
    });
  // Rounded native draw_switch palette outputs; CSS keeps fractional channels until paint.
  const expected = {
    dark: {
      off: [
        [54, 58, 63],
        [77, 80, 85],
        [99, 103, 107],
      ],
      on: [
        [109, 131, 104],
        [123, 143, 120],
        [94, 112, 93],
      ],
    },
    light: {
      off: [
        [214, 219, 210],
        [191, 197, 188],
        [168, 174, 165],
      ],
      on: [
        [139, 162, 130],
        [126, 148, 119],
        [158, 177, 150],
      ],
    },
  };
  await mkdir(new URL('../../output/electron/verification/switch-row/', import.meta.url), { recursive: true });
  for (const theme of ['dark', 'light']) {
    for (const checked of [false, true]) {
      await page.evaluate(
        ({ theme, checked }) => {
          const fixture = window.switchRowFixture;
          fixture.checked = checked;
          fixture.theme(theme);
        },
        { theme, checked },
      );
      await page.waitForFunction(
        (checked) =>
          document.getElementById('grid-switch').getAttribute('aria-checked') === String(checked),
        checked,
      );
      let thumbColor;
      for (const state of ['normal', 'hover', 'held']) {
        if (state === 'normal') await page.mouse.move(420, 260);
        else await row.hover({ position: { x: 180, y: 14 } });
        if (state === 'held') await page.mouse.down();
        await page.waitForTimeout(140);
        const value = await visual();
        assert.equal(value.track.width, 54);
        assert.equal(value.track.height, 24);
        assert.equal(value.row.right - value.track.right, 8);
        assert.ok(Math.abs(value.thumb.width - (state === 'held' ? 16.56 : 18)) < 0.01);
        assert.equal(value.thumb.width, value.thumb.height);
        assert.equal(value.radius, '50%');
        assert.match(value.shape, /^(round|superellipse\(1\))$/);
        if (state === 'normal') thumbColor = value.thumbColor;
        else assert.equal(value.thumbColor, thumbColor);
        const palette =
          expected[theme][checked ? 'on' : 'off'][['normal', 'hover', 'held'].indexOf(state)];
        value.color.forEach((channel, index) => assert.ok(Math.abs(channel - palette[index]) <= 1));
        assert.equal(value.rowBackground === 'rgba(0, 0, 0, 0)', state === 'normal');
        await row.screenshot({
          path: fileURLToPath(
            new URL(
              `../../output/electron/verification/switch-row/${theme}-${checked ? 'on' : 'off'}-${state}.png`,
              import.meta.url,
            ),
          ),
        });
      }
      await page.mouse.move(420, 260);
      await page.mouse.up();
    }
  }
});
