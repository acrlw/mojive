import { openEditableJointScene } from './helpers/editable-scene.mjs';
import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

let server, browser;
const evidence = [];
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-switch-row-integration',
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
  await writeFile(
    '../output/electron/verification/switch-row-integration.json',
    JSON.stringify(evidence, null, 2),
  );
});
async function resize(page, width, scale) {
  const handle = page.locator('[role=separator][aria-orientation=vertical]').last(),
    r = await handle.boundingBox(),
    pane = await page.locator('.inspector').boundingBox();
  await page.mouse.move(r.x + r.width / 2, r.y + 100 * scale);
  await page.mouse.down();
  await page.mouse.move(r.x + r.width / 2 - (width * scale - pane.width), r.y + 100 * scale, {
    steps: 8,
  });
  await page.mouse.up();
}
async function expectChecked(row, value) {
  assert.equal(await row.getAttribute('aria-checked'), String(value));
}

for (const [language, appearance, scale] of [
  ['en', 'light', 1],
  ['zh', 'dark', 1],
  ['en', 'light', 1.25],
  ['zh', 'dark', 1.25],
  ['en', 'dark', 2],
  ['zh', 'light', 2],
])
  test(`actual full-row switches and cross-section Inspector columns ${language}/${appearance}/${scale}x`, async () => {
    const page = await browser.newPage({
        viewport: { width: Math.round(1700 * scale), height: Math.round(1100 * scale) },
      }),
      errors = [];
    await page.addInitScript(installPaintStyle);
    await page.evaluate(installPaintStyle);
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript(
      (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
      { language, appearance, uiScale: scale },
    );
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
    await openEditableJointScene(page);
    await page.locator('.tree-row.type-camera').first().click();
    await page.evaluate(() => document.fonts.ready);
    for (const width of [280, 480]) {
      await resize(page, width, scale);
      const rows = page.locator('.inspector .switch-row');
      assert.equal(await rows.count(), 4);
      const geometry = await rows.evaluateAll(
        (rows, scale) =>
          rows.map((row) => {
            const a = row.getBoundingClientRect(),
              track = row.querySelector('.custom-switch'),
              b = track.getBoundingClientRect(),
              thumb = track.firstElementChild,
              c = thumb.getBoundingClientRect();
            return {
              tag: row.tagName,
              type: row.type,
              role: row.getAttribute('role'),
              disabled: row.disabled,
              tabIndex: row.tabIndex,
              nested: row.querySelectorAll('button,input').length,
              trackRight: (a.right - b.right) / scale,
              captionLeft: (row.firstElementChild.getBoundingClientRect().left - a.left) / scale,
              trackWidth: b.width / scale,
              trackHeight: b.height / scale,
              thumbWidth: c.width / scale,
              thumbHeight: c.height / scale,
              thumbRadius: window.paintStyle(thumb).borderRadius,
              rowWidth: a.width / scale,
              overflow: row.scrollWidth - row.clientWidth,
            };
          }),
        scale,
      );
      for (const row of geometry) {
        assert.equal(row.tag, 'BUTTON');
        assert.equal(row.type, 'button');
        assert.equal(row.role, 'switch');
        assert.equal(row.tabIndex, 0);
        assert.equal(row.nested, 0);
        assert.ok(
          Math.abs(row.trackRight - 8) < 0.1 && Math.abs(row.captionLeft - 8) < 0.1,
          JSON.stringify(row),
        );
        assert.ok(Math.abs(row.trackWidth - 54) < 0.5 && Math.abs(row.trackHeight - 24) < 0.5);
        assert.ok(Math.abs(row.thumbWidth - 18) < 0.5 && Math.abs(row.thumbHeight - 18) < 0.5);
        assert.equal(row.thumbRadius, '50%');
        assert.ok(row.overflow <= 1);
      }
      const visible = rows.first();
      await visible.scrollIntoViewIfNeeded();
      await expectChecked(visible, true);
      for (const target of ['caption', 'gap', 'track']) {
        const point = await visible.evaluate((row, target) => {
          const a = row.firstElementChild.getBoundingClientRect(),
            b = row.lastElementChild.getBoundingClientRect();
          return {
            x:
              target === 'caption'
                ? a.left + 3
                : target === 'gap'
                  ? (a.right + b.left) / 2
                  : b.left + b.width / 2,
            y: b.top + b.height / 2,
          };
        }, target);
        await page.mouse.move(point.x, point.y);
        await settlePaint(page);
        const hover = await visible.evaluate((row) => window.paintStyle(row).backgroundColor);
        assert.notEqual(hover, 'rgba(0, 0, 0, 0)');
        if (target === 'gap')
          await page.locator('.inspector').screenshot({
            path: `../output/electron/verification/switch-row-inspector-${language}-${appearance}-${width}-${scale}x-hover.png`,
          });
        await page.mouse.down();
        await settlePaint(page);
        await expectChecked(visible, true);
        const held = await visible.evaluate((row) => window.paintStyle(row).backgroundColor);
        assert.notEqual(held, hover);
        if (target === 'gap')
          await page.locator('.inspector').screenshot({
            path: `../output/electron/verification/switch-row-inspector-${language}-${appearance}-${width}-${scale}x-held.png`,
          });
        await page.mouse.up();
        await expectChecked(visible, false);
        await page.keyboard.press('ControlOrMeta+z');
        await expectChecked(visible, true);
      }
      await visible.focus();
      await visible.press('Space');
      await expectChecked(visible, false);
      await visible.press('Enter');
      await expectChecked(visible, true);
      await page.keyboard.press('ControlOrMeta+z');
      await expectChecked(visible, false);
      await page.keyboard.press('ControlOrMeta+z');
      await expectChecked(visible, true);
      const fields = await page.locator('.camera-property-row').evaluateAll((rows) =>
        rows.map((row) => {
          const r = row.lastElementChild.getBoundingClientRect(),
            label = row.firstElementChild.getBoundingClientRect();
          return {
            left: r.left,
            right: r.right,
            gap: r.left - label.right,
            inline: r.top < label.bottom && r.bottom > label.top,
          };
        }),
      );
      assert.ok(fields.length >= 8);
      for (const field of fields) {
        assert.deepEqual(field, fields[0]);
        assert.equal(field.inline, true, 'scalar fields stay beside labels across sections');
        assert.ok(Math.abs(field.gap - 12 * scale) < 0.1);
      }
      await page.locator('.camera-properties').screenshot({
        path: `../output/electron/verification/switch-row-camera-columns-${language}-${appearance}-${width}-${scale}x.png`,
      });
      evidence.push({ language, appearance, scale, width, geometry, fields });
    }
    const position = page.locator('.inspector-content > .section .vector-group').first();
    const xInput = position.locator('input').first();
    await xInput.scrollIntoViewIfNeeded();
    await xInput.click({ modifiers: ['Meta'] });
    await xInput.fill('-999.999');
    await xInput.press('Enter');
    assert.equal(await xInput.inputValue(), '-999.999');
    const numericBudget = await xInput.evaluate((input, scale) => {
      const style = window.paintStyle(input),
        context = document.createElement('canvas').getContext('2d');
      context.font = `${style.fontSize} ${style.fontFamily}`;
      return {
        available:
          input.getBoundingClientRect().width / scale -
          parseFloat(style.paddingLeft) -
          parseFloat(style.paddingRight) -
          parseFloat(style.borderLeftWidth) -
          parseFloat(style.borderRightWidth),
        text: context.measureText(input.value).width,
      };
    }, scale);
    assert.ok(numericBudget.available >= numericBudget.text, JSON.stringify(numericBudget));
    const inputBox = await xInput.boundingBox();
    await page.mouse.move(inputBox.x + inputBox.width / 2, inputBox.y + inputBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(inputBox.x + inputBox.width / 2 + 10, inputBox.y + inputBox.height / 2, {
      steps: 5,
    });
    assert.ok(
      Number(await xInput.inputValue()) > -999.999 && Number(await xInput.inputValue()) < -999.8,
    );
    await position.screenshot({
      path: `../output/electron/verification/switch-row-negative-drag-${language}-${appearance}-${scale}x.png`,
    });
    await page.keyboard.press('Escape');
    await page.mouse.up();
    assert.equal(await xInput.inputValue(), '-999.999');
    await page.keyboard.press('ControlOrMeta+z');
    assert.equal(await xInput.inputValue(), '0.000');
    // Collapsing another section changes the shared budget without changing the panel width.
    const projection = page
      .locator('.camera-properties > .section')
      .nth(1)
      .locator('.section-disclosure');
    const minimumInlineWidth = async () => {
      await settlePaint(page);
      return page.evaluate(() => {
        const section = document.querySelector('.inspector-content > .section .section-content');
        const grid = section.closest('.inspector-content');
        const input = section.querySelector('input'),
          style = getComputedStyle(input);
        const axis = section.querySelector('.axis'),
          axisStyle = getComputedStyle(axis);
        const canvas = document.createElement('canvas').getContext('2d');
        canvas.font = `${style.fontSize} ${style.fontFamily}`;
        const field =
          canvas.measureText('-999.999').width +
          parseFloat(style.paddingLeft) +
          parseFloat(style.paddingRight) +
          parseFloat(axisStyle.width);
        const paneInset =
          parseFloat(getComputedStyle(section.closest('.inspector')).width) - section.clientWidth;
        return (
          parseFloat(getComputedStyle(grid).gridTemplateColumns) + 12 + field * 3 + 8 + paneInset
        );
      });
    };
    const expandedWidth = await minimumInlineWidth();
    await projection.click();
    const collapsedWidth = await minimumInlineWidth();
    assert.ok(expandedWidth > collapsedWidth, 'Projection captions change the shared label budget');
    await projection.click();
    await resize(page, Math.round((expandedWidth + collapsedWidth) / 2), scale);
    const transform = page
      .locator('.inspector-content > .section')
      .first()
      .locator('.section-content');
    await page.waitForFunction(
      () =>
        document.querySelector('.inspector-content > .section .section-content')?.dataset
          .vectorLabels === 'above',
    );
    await projection.click();
    await page.waitForFunction(
      () =>
        document.querySelector('.inspector-content > .section .section-content')?.dataset
          .vectorLabels === 'inline',
    );
    await projection.click();
    await page.waitForFunction(
      () =>
        document.querySelector('.inspector-content > .section .section-content')?.dataset
          .vectorLabels === 'above',
    );
    assert.equal(await transform.getAttribute('data-vector-fields'), 'row');
    // Settings, light properties and both Layers surfaces use the same complete row owner.
    await page.locator('.activity-rail button:has([data-icon=settings])').click();
    await page.locator('.settings-navigation > button').last().click();
    const grid = page.locator('.settings-fields .switch-row').first();
    await expectChecked(grid, true);
    await grid.locator('.switch-row-label').click();
    await expectChecked(grid, false);
    await grid.press('Space');
    await expectChecked(grid, true);
    await page.locator('.settings-dialog').screenshot({
      path: `../output/electron/verification/switch-row-settings-${language}-${appearance}-${scale}x.png`,
    });
    await page.locator('.settings-header button:has([data-icon=close])').click();
    await page.locator('.tree-row.type-light').first().click();
    const light = page.locator('.light-properties .switch-row').first();
    await expectChecked(light, true);
    await light.locator('.switch-row-label').click();
    await expectChecked(light, false);
    await page.keyboard.press('ControlOrMeta+z');
    await expectChecked(light, true);
    await page.locator('.viewport-navigation button:has([data-icon=layers])').click();
    const layerRows = page.locator('.viewport-layers-popup .switch-row');
    const helpers = layerRows.nth(3),
      influence = layerRows.nth(4);
    await expectChecked(helpers, true);
    await helpers.locator('.switch-row-label').click();
    await expectChecked(helpers, false);
    assert.equal(await influence.isDisabled(), true);
    const disabledState = await influence.getAttribute('aria-checked');
    await influence.click({ force: true });
    assert.equal(await influence.getAttribute('aria-checked'), disabledState);
    await helpers.focus();
    await page.keyboard.press('Tab');
    assert.equal(
      await layerRows.nth(5).evaluate((row) => document.activeElement === row),
      true,
      'Tab skips the disabled influence row',
    );
    await page.locator('.viewport-layers-popup').screenshot({
      path: `../output/electron/verification/switch-row-layers-${language}-${appearance}-${scale}x.png`,
    });
    await layerRows.nth(5).press('Escape');
    assert.deepEqual(errors, []);
    await page.close();
  });
