import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
let server, browser;
const out = '../output/electron/verification';
const measurements = [];
before(async () => {
  await mkdir(out, { recursive: true });
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'error' });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await writeFile(`${out}/nested-corners.json`, JSON.stringify(measurements, null, 2));
  await browser?.close();
  await server?.close();
});
async function corner(page, parent, child, edges, name) {
  const data = await page
    .locator(parent)
    .first()
    .evaluate(
      (outer, { child }) => {
        const inner = outer.querySelector(child),
          a = outer.getBoundingClientRect(),
          b = inner.getBoundingClientRect();
        const os = window.paintStyle(outer),
          ins = window.paintStyle(inner);
        const scale = a.width / parseFloat(os.width);
        return {
          profiles: [outer.dataset.smoothAmount, inner.dataset.smoothAmount],
          outer: parseFloat(os.borderTopLeftRadius),
          inner: parseFloat(ins.borderTopLeftRadius),
          insets: {
            left: (b.left - a.left) / scale,
            top: (b.top - a.top) / scale,
            right: (a.right - b.right) / scale,
            bottom: (a.bottom - b.bottom) / scale,
          },
        };
      },
      { child },
    );
  measurements.push({ name, ...data });
  const insets = edges.map((edge) => data.insets[edge]);
  assert.ok(
    Math.max(...insets) - Math.min(...insets) < 0.12,
    `${name}: unequal insets ${JSON.stringify(data)}`,
  );
  assert.ok(
    Math.abs(data.inner - Math.max(0, data.outer - insets[0])) < 0.12,
    `${name}: corner budget ${JSON.stringify(data)}`,
  );
}
for (const [theme, language, scale] of [
  ['dark', 'en', 1],
  ['light', 'zh', 1.25],
]) {
  test(`production nested surfaces: ${theme}/${language}/${scale}`, async () => {
    const page = await browser.newPage({
      viewport: { width: 1780, height: 1100 },
      deviceScaleFactor: 2,
    });
    await page.addInitScript(installPaintStyle);
    await page.evaluate(installPaintStyle);
    page.setDefaultTimeout(6000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript(
      (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
      { language, appearance: theme, uiScale: scale },
    );
    const base = `http://127.0.0.1:${server.httpServer.address().port}`;
    await page.goto(base);
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    await singleCornerPaint(page, '.geometry-selector', 'workspace geometry choices');
    await corner(page, '.tool-palette', 'button', ['left', 'top', 'right'], 'tool palette');
    await page.locator('[data-menu=Edit]').click();
    await page.locator('.menu-edit').waitFor();
    await corner(page, '.menu-edit', 'button:first-child', ['left', 'top', 'right'], 'menu first');
    await corner(page, '.menu-edit', 'button:last-child', ['left', 'bottom', 'right'], 'menu last');
    await page.locator('.menu-edit button').last().hover();
    await page.locator('.menu-edit').screenshot({ path: `${out}/corners-menu-${theme}.png` });
    await page.keyboard.press('Escape');
    await page.locator('.activity-rail > button:has([data-icon=pose])').click();
    const heading = page.locator('.entity-card-heading').nth(1);
    await heading.click();
    await heading.hover();
    await corner(
      page,
      '.entity-card:nth-child(2)',
      '.entity-card-heading',
      ['left', 'top', 'right'],
      'joint heading',
    );
    await corner(
      page,
      '.entity-card:nth-child(2)',
      '.number-field',
      ['right', 'bottom'],
      'joint value',
    );
    await page
      .locator('.joint-panel-list')
      .screenshot({ path: `${out}/corners-joints-${theme}.png` });
    await page.locator('.joint-panel-tools button:has([data-icon=search])').click();
    await page.locator('.joint-panel-search input').fill('01');
    await settlePaint(page);
    const search = await page.locator('.joint-panel-search').evaluate((el) => {
      const button = el.querySelector('button'),
        a = el.getBoundingClientRect(),
        b = button.getBoundingClientRect();
      return {
        radius: window.paintStyle(button).borderRadius,
        sides: [b.top - a.top, a.bottom - b.bottom, a.right - b.right],
      };
    });
    assert.equal(search.radius, '50%');
    assert.ok(
      search.sides.every((gap, index) => Math.abs(gap - (index === 2 ? 4.24723 : 4) * scale) < 0.1),
      JSON.stringify(search),
    );
    await page.locator('[data-menu=Add]').click();
    await page.locator('.menu-add button:has([data-icon=camera])').click();
    await singleCornerPaint(page, '.inspector .axis-field', 'workspace vector field');
    await singleCornerPaint(page, '.camera-projection', 'workspace camera projection');
    await corner(
      page,
      '.operation-notice',
      '.operation-notice-close',
      ['right', 'top', 'bottom'],
      'notice close',
    );
    await page.locator('.activity-rail > button:has([data-icon=camera])').click();
    await corner(page, '.camera-entry', '.camera-entry-name', ['left', 'top'], 'camera name');
    await corner(
      page,
      '.camera-entry',
      '.camera-entry-more button',
      ['right', 'top'],
      'camera more',
    );
    await corner(
      page,
      '.camera-entry',
      '.camera-entry-actions button:last-child',
      ['right', 'bottom'],
      'camera actions',
    );
    await page.locator('.camera-entry-name').first().hover();
    await page
      .locator('.camera-entry')
      .first()
      .screenshot({ path: `${out}/corners-camera-${theme}.png` });
    await page.locator('.camera-entry-more button').first().click();
    await corner(
      page,
      '.camera-entry-menu',
      'button:first-child',
      ['left', 'top', 'right'],
      'camera menu first',
    );
    await corner(
      page,
      '.camera-entry-menu',
      'button:last-child',
      ['left', 'bottom', 'right'],
      'camera menu last',
    );
    await page.keyboard.press('Escape');
    await page.locator('.viewport-topbar button:has([data-icon=layers])').click();
    await corner(
      page,
      '.viewport-layers-surface',
      'header .button',
      ['right', 'top'],
      'layers reset',
    );
    await page
      .locator('.viewport-layers-popup')
      .screenshot({ path: `${out}/corners-layers-${theme}.png` });
    await page.keyboard.press('Escape');
    await page.locator('.title-actions > button:has([data-icon=settings])').click();
    await singleCornerPaint(page, '.settings-segment', 'workspace appearance choices');
    await corner(
      page,
      '.settings-surface',
      '.settings-header .button',
      ['right', 'top'],
      'settings close',
    );
    await page.locator('.settings-navigation button:has([data-icon=mouse])').click();
    await corner(
      page,
      '.binding-control',
      '.binding-clear',
      ['right', 'top', 'bottom'],
      'binding clear',
    );
    assert.equal(
      await page
        .locator('.binding-clear')
        .first()
        .evaluate((el) => window.paintStyle(el).width),
      '24px',
    );
    await page.keyboard.press('Escape');
    await page.locator('[data-menu=Help]').click();
    await page.locator('.menu-help button').first().click();
    await corner(page, '.info-dialog', '.dialog-heading .button', ['right', 'top'], 'guide close');
    await page.locator('.info-dialog').screenshot({ path: `${out}/corners-guide-${theme}.png` });
    await page.keyboard.press('Escape');
    await page.locator('[data-menu=Add]').click();
    await page.locator('.menu-add button').nth(5).click();
    await corner(
      page,
      '.inspector .color-trigger',
      '.color-swatch',
      ['left', 'top', 'bottom'],
      'workspace color trigger',
    );
    await page.locator('.inspector .color-trigger').click();
    await singleCornerPaint(page, '.color-format', 'workspace color format');
    await page.locator('.color-hex').fill('#F19322');
    await page.locator('.color-hex').press('Enter');
    await page.locator('.color-footer button').last().click();
    await page.locator('.inspector .color-trigger').click();
    await corner(
      page,
      '.color-picker-surface',
      '.color-comparison-item',
      ['left', 'top'],
      'workspace color shell',
    );
    await corner(
      page,
      '.color-comparison-item',
      '.color-swatch',
      ['left', 'top', 'bottom'],
      'workspace color comparison',
    );
    await page.locator('.color-comparison-item').first().hover();
    await page.locator('.color-picker-surface').screenshot({
      path: `${out}/corners-color-workspace-${theme}.png`,
    });
    await page.locator('.color-format button').nth(1).click();
    await singleCornerPaint(page, '.color-channel-fields .axis-field', 'workspace color channel');
    await page.locator('.color-picker-surface').screenshot({
      path: `${out}/color-rgb-workspace-${theme}.png`,
    });
    await page.goto(base + '/tests/harness/ui-controls-review.html');
    await page.evaluate((scale) => (document.getElementById('root').style.zoom = scale), scale);
    await page.evaluate(
      async ({ theme, language }) => {
        const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
        applySettings({ ...DEFAULT_SETTINGS, appearance: theme, language });
      },
      { theme, language },
    );
    await page.getByRole('combobox').click();
    await corner(page, '.select-options', 'button', ['left', 'top', 'right'], 'select options');
    await page.keyboard.press('Escape');
    await page.locator('.color-trigger').click();
    await corner(
      page,
      '.color-picker-surface',
      '.color-footer button:last-child',
      ['right', 'bottom'],
      'color footer',
    );
    await corner(
      page,
      '.color-comparison-item',
      '.color-swatch',
      ['left', 'top', 'bottom'],
      'color comparison',
    );
    await page.locator('.color-comparison-item').first().hover();
    await page
      .locator('.color-picker-surface')
      .screenshot({ path: `${out}/corners-color-${theme}.png` });
    await page.goto(base + '/tests/harness/ui-confirm-review.html');
    await page.evaluate((scale) => (document.getElementById('root').style.zoom = scale), scale);
    await page.getByRole('button', { name: 'Open confirmation' }).click();
    await corner(
      page,
      '.confirm-scene-surface',
      '.confirm-scene-actions button:last-child',
      ['right', 'bottom'],
      'confirmation',
    );
    assert.deepEqual(errors, []);
    await page.close();
  });
}

async function swatchPixels(page, swatch) {
  const png = await swatch.screenshot();
  return page.evaluate(async (encoded) => {
    const image = new Image();
    image.src = 'data:image/png;base64,' + encoded;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    return {
      width: canvas.width,
      height: canvas.height,
      rgba: Array.from(context.getImageData(0, 0, canvas.width, canvas.height).data),
    };
  }, png.toString('base64'));
}

async function singleCornerPaint(page, selector, name) {
  const surface = page.locator(selector).first();
  await surface.scrollIntoViewIfNeeded();
  const styles = await surface.evaluate((element) => {
    const nodes = [element, ...element.querySelectorAll('*')];
    const saved = nodes.map((node) => node.getAttribute('style'));
    for (const node of nodes) {
      const background = window.paintStyle(node).backgroundColor;
      node.style.setProperty('color', 'transparent', 'important');
      node.style.setProperty('-webkit-text-fill-color', 'transparent', 'important');
      node.style.setProperty('caret-color', 'transparent', 'important');
      node.style.setProperty('box-shadow', 'none', 'important');
      node.style.setProperty('outline', 'none', 'important');
      if (node.tagName.toLowerCase() === 'svg') node.style.visibility = 'hidden';
      if (background !== 'rgba(0, 0, 0, 0)' && background !== 'transparent')
        node.style.setProperty('background', '#4F545E', 'important');
    }
    return saved;
  });
  try {
    const actual = await swatchPixels(page, surface);
    await surface.evaluate((element) => {
      // Keep segment geometry identical: Chromium rasterizes asymmetric corners
      // differently from a four-corner rectangle at fractional device positions.
      element.style.setProperty('background', 'transparent', 'important');
      element.style.setProperty('overflow', 'visible', 'important');
      element.style.setProperty('clip-path', 'none', 'important');
    });
    const reference = await swatchPixels(page, surface);
    assert.equal(actual.rgba.length, reference.rgba.length);
    let fringe = 0,
      maxError = 0;
    const differences = [];
    for (let i = 0; i < actual.rgba.length; i += 4) {
      const error = Math.max(
        ...[0, 1, 2].map((channel) =>
          Math.abs(actual.rgba[i + channel] - reference.rgba[i + channel]),
        ),
      );
      if (error > 2) {
        fringe++;
        if (differences.length < 8)
          differences.push({
            x: (i / 4) % actual.width,
            y: Math.floor(i / 4 / actual.width),
            actual: actual.rgba.slice(i, i + 3),
            reference: reference.rgba.slice(i, i + 3),
          });
      }
      maxError = Math.max(maxError, error);
    }
    measurements.push({ name, fringe, maxError, differences });
    assert.ok(
      fringe === 0 && maxError === 0,
      `${name}: duplicate corner painting differs from one continuous fill (max ${maxError})`,
    );
  } finally {
    await surface.evaluate((element, styles) => {
      [element, ...element.querySelectorAll('*')].forEach((node, index) => {
        if (styles[index] === null) node.removeAttribute('style');
        else node.setAttribute('style', styles[index]);
      });
    }, styles);
  }
}

for (const density of [1, 2]) {
  test(`color swatch corners contain no checker fringe at ${density}x framebuffer density`, async () => {
    const page = await browser.newPage({
      viewport: { width: 1400, height: 1400 },
      deviceScaleFactor: density,
    });
    await page.addInitScript(installPaintStyle);
    await page.evaluate(installPaintStyle);
    await page.goto(
      `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-controls-review.html`,
    );
    await page.locator('.color-trigger').waitFor();
    await page.evaluate(() => window.controlActions.setColor('#4F545E'));
    for (const scale of [1, 1.25, 2]) {
      for (const offset of [0, 0.25]) {
        await page.evaluate(
          ({ scale, offset }) => {
            const root = document.getElementById('root');
            root.style.zoom = scale;
            root.style.transform = `translate(${offset}px, ${offset}px)`;
          },
          { scale, offset },
        );
        await page.locator('.color-trigger').click();
        await page.locator('.color-comparison-item').first().hover();
        for (const swatch of await page.locator('.color-swatch').all()) {
          const { rgba } = await swatchPixels(page, swatch);
          let fringe = 0;
          for (let i = 0; i < rgba.length; i += 4)
            if (rgba[i] > 80 || rgba[i + 1] > 85 || rgba[i + 2] > 95) fringe++;
          assert.equal(
            fringe,
            0,
            `Opaque #4F545E cannot contain brighter checker pixels: scale ${scale}, offset ${offset}`,
          );
        }
        if (offset === 0)
          await page
            .locator('.color-comparison')
            .screenshot({ path: `${out}/color-swatch-dark-${density}x-${scale}.png` });
        await page.keyboard.press('Escape');
      }
    }
    await page.close();
  });
}

test('transparent swatches retain the correctly composited checker colors in both themes', async () => {
  const page = await browser.newPage({
    viewport: { width: 900, height: 900 },
    deviceScaleFactor: 2,
  });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-controls-review.html`,
  );
  await page.locator('.color-trigger').waitFor();
  for (const theme of ['dark', 'light']) {
    for (const alpha of [0, 128]) {
      await page.evaluate(
        async ({ theme, alpha }) => {
          const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
          applySettings({ ...DEFAULT_SETTINGS, appearance: theme, language: 'en' });
          window.controlActions.setAlpha(true);
          window.controlActions.setColor('#4F545E' + alpha.toString(16).padStart(2, '0'));
        },
        { theme, alpha },
      );
      await page.locator('.color-trigger').click();
      const colors = [170, 238].map((gray) =>
        [79, 84, 94]
          .map((color) => Math.round((color * alpha) / 255 + gray * (1 - alpha / 255)))
          .join(','),
      );
      for (const swatch of await page.locator('.color-comparison .color-swatch').all()) {
        const { width, height, rgba } = await swatchPixels(page, swatch);
        const interior = new Set();
        for (let y = 10; y < height - 10; y++)
          for (let x = 10; x < width - 10; x++)
            interior.add(rgba.slice((y * width + x) * 4, (y * width + x) * 4 + 3).join(','));
        assert.deepEqual([...interior].sort(), colors.sort());
      }
      await page
        .locator('.color-comparison')
        .screenshot({ path: `${out}/color-swatch-alpha-${theme}-${alpha}.png` });
      await page.keyboard.press('Escape');
    }
  }
  await page.close();
});

test('Canvas readouts share the continuous contour at normal and fractional scale', async () => {
  const page = await browser.newPage({
    viewport: { width: 900, height: 800 },
    deviceScaleFactor: 2,
  });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-controls-review.html`,
  );
  const results = await page.evaluate(async () => {
    const { fillSurface } = await import('/src/viewport/surfaces.ts');
    const output = [];
    for (const scale of [1, 1.25, 2]) {
      const canvas = document.createElement('canvas');
      canvas.width = 120 * scale * 2;
      canvas.height = 64 * scale * 2;
      const ctx = canvas.getContext('2d');
      ctx.scale(scale * 2, scale * 2);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, 120, 64);
      ctx.fillStyle = '#000';
      fillSurface(ctx, 10, 8, 100, 48, 16);
      const at = (x, y) =>
        ctx.getImageData(Math.floor(x * scale * 2), Math.floor(y * scale * 2), 1, 1).data[0];
      output.push({
        scale,
        inside: at(18, 16),
        outside: at(10, 8),
        top: at(60, 7),
        fill: at(60, 9),
      });
    }
    document.body.innerHTML = '';
    document.body.style.cssText = 'background:#f0f3ee;margin:24px;display:flex;gap:24px';
    const css = document.createElement('div');
    css.style.cssText = 'width:320px;height:192px;border-radius:64px;background:#4f545e';
    document.body.append(css);
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 384;
    canvas.style.cssText = 'width:320px;height:192px';
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);
    ctx.fillStyle = '#4f545e';
    fillSurface(ctx, 0, 0, 320, 192, 64);
    document.body.append(canvas);
    return output;
  });
  for (const result of results) {
    assert.equal(result.inside, 0, JSON.stringify(result));
    assert.equal(result.outside, 255, JSON.stringify(result));
    assert.equal(result.top, 255, JSON.stringify(result));
    assert.equal(result.fill, 0, JSON.stringify(result));
  }
  await page.screenshot({ path: `${out}/continuous-corners-css-canvas.png` });
  await page.close();
});
