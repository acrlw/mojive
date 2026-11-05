import { before, after, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';

let server, browser, base;
const out = '../output/electron/verification/capsules';
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  await mkdir(out, { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function settings(page, values) {
  await page.evaluate(async (values) => {
    const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
    applySettings({ ...DEFAULT_SETTINGS, language: 'en', ...values });
  }, values);
  await page.evaluate(() => document.fonts.ready);
  await settlePaint(page);
}
async function capsuleLayout(page) {
  return page.locator('[data-capsule-layout]').evaluateAll(async (elements) => {
    const { smoothCapsule, capsuleEndPadding } = await import('/src/smooth-geometry.ts');
    return elements
      .filter((el) => el.getBoundingClientRect().width)
      .map((el) => {
        const box = el.getBoundingClientRect(),
          style = getComputedStyle(el);
        const width = parseFloat(style.width),
          height = parseFloat(style.height),
          scale = box.width / width;
        const end = capsuleEndPadding(
          smoothCapsule(width, height, Number(el.dataset.smoothAmount)),
        );
        const bounds = (child) => {
          const b = child.getBoundingClientRect();
          return {
            x: (b.x + b.width / 2 - box.x) / scale,
            y: (b.y + b.height / 2 - box.y) / scale,
          };
        };
        if (el.matches('.custom-switch')) {
          const checked = el.closest('[role=switch]').getAttribute('aria-checked') === 'true';
          return {
            kind: 'switch',
            end,
            height,
            centers: [{ ...bounds(el.firstElementChild), target: checked ? width - end : end }],
          };
        }
        const icon = el.querySelector(':scope > .icon'),
          clear = el.querySelector(':scope > button');
        return {
          kind: el.matches('.filter-chip') ? 'chip' : 'search',
          end,
          height,
          centers: [
            { ...bounds(icon), target: end },
            ...(clear ? [{ ...bounds(clear), target: width - end }] : []),
          ],
        };
      });
  });
}
function assertCenters(rows) {
  for (const row of rows)
    for (const center of row.centers) {
      assert.ok(Math.abs(center.x - center.target) < 0.04, JSON.stringify(row));
      assert.ok(Math.abs(center.y - row.height / 2) < 0.04, JSON.stringify(row));
    }
}
test('search, chip icons and switch discs follow fitted optical centers after smoothing and scale changes', async () => {
  const page = await browser.newPage({ viewport: { width: 1700, height: 1100 } });
  await page.addInitScript(() =>
    localStorage.setItem('mojive.electron.settings.v1', JSON.stringify({ language: 'en' })),
  );
  await page.goto(base);
  await page.locator('.tree-row').first().waitFor();
  await page.locator('.left-panel').getByRole('button', { name: 'Search', exact: true }).click();
  await page.locator('.panel-search input').fill('capsule');
  const records = [];
  for (const [appearance, uiScale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    for (const capsuleSmoothing of [0, 38.2, 100]) {
      await settings(page, { appearance, uiScale, capsuleSmoothing });
      const rows = await capsuleLayout(page);
      assert.ok(
        rows.some((row) => row.kind === 'chip') && rows.some((row) => row.kind === 'search'),
      );
      assertCenters(rows);
      records.push({ appearance, uiScale, capsuleSmoothing, rows });
    }
    await settings(page, { appearance, uiScale });
    await page
      .locator('.scene-controls')
      .screenshot({ path: `${out}/optical-chips-${appearance}.png` });
    await page.locator('.panel-search button').hover();
    await settlePaint(page);
    await page
      .locator('.panel-search')
      .screenshot({ path: `${out}/optical-search-${appearance}.png` });
  }
  await page.goto(base + '/tests/harness/ui-controls-review.html');
  await page.locator('[role=switch]').waitFor();
  for (const capsuleSmoothing of [0, 38.2, 100]) {
    await settings(page, { capsuleSmoothing, uiScale: 1.25 });
    for (let i = 0; i < 2; i++) {
      await page.locator('[role=switch]').click();
      await page.waitForTimeout(150);
      assertCenters(await capsuleLayout(page));
    }
  }
  const mutations = await page.evaluate(async () => {
    let count = 0;
    const observer = new MutationObserver((records) => (count += records.length));
    document
      .querySelectorAll('[data-capsule-layout]')
      .forEach((el) => observer.observe(el, { attributes: true, attributeFilter: ['style'] }));
    await new Promise((resolve) => setTimeout(resolve, 150));
    observer.disconnect();
    return count;
  });
  assert.equal(mutations, 0, 'Optical layout must settle without a measurement loop.');
  await writeFile(`${out}/optical-layout.json`, JSON.stringify(records, null, 2));
  await page.close();
});
test('XYZ and RGB share numeric typography, balanced character space and compact frames', async () => {
  const page = await browser.newPage({
    viewport: { width: 1500, height: 1100 },
    deviceScaleFactor: 2,
  });
  await page.goto(base + '/tests/harness/ui-controls-review.html');
  await page.locator('.axis-field').first().waitFor();
  await page.locator('.color-trigger').click();
  await page.getByRole('button', { name: 'RGB 255', exact: true }).click();
  const records = [];
  for (const numericFontSize of [14, 18])
    for (const [appearance, uiScale] of [
      ['dark', 1],
      ['light', 1.25],
    ]) {
      await settings(page, { appearance, uiScale, numericFontSize });
      const fields = await page.locator('.axis-field').evaluateAll((elements) =>
        elements.map((el) => {
          const badge = el.querySelector('.axis'),
            input = el.querySelector('input');
          const b = getComputedStyle(badge),
            s = getComputedStyle(input);
          const canvas = document.createElement('canvas').getContext('2d');
          canvas.font = `${s.fontSize} ${s.fontFamily}`;
          const ink = canvas.measureText(badge.textContent.trim());
          return {
            badge: badge.textContent.trim(),
            badgeFont: b.fontFamily,
            inputFont: s.fontFamily,
            badgeSize: b.fontSize,
            inputSize: s.fontSize,
            height: parseFloat(b.height),
            width: parseFloat(b.width),
            verticalSpace:
              (parseFloat(b.height) - ink.actualBoundingBoxAscent - ink.actualBoundingBoxDescent) /
              2,
            horizontalSpace: (parseFloat(b.width) - ink.width) / 2,
            inputWidth: input.clientWidth,
            textWidth: canvas.measureText(input.value).width,
          };
        }),
      );
      assert.equal(fields.map((field) => field.badge).join(''), 'XYZRGB');
      for (const field of fields) {
        assert.equal(field.badgeFont, field.inputFont);
        assert.match(field.badgeFont, /JetBrains Mono/);
        assert.equal(field.badgeSize, field.inputSize);
        assert.equal(field.height, 28);
        assert.ok(
          Math.abs(field.verticalSpace - field.horizontalSpace) < 0.6,
          JSON.stringify(field),
        );
        assert.ok(field.inputWidth >= field.textWidth + 8, JSON.stringify(field));
      }
      records.push({ numericFontSize, appearance, uiScale, fields });
      await page
        .locator('.vector-fields')
        .screenshot({ path: `${out}/channels-xyz-${appearance}-${numericFontSize}.png` });
      await page
        .locator('.color-channel-fields')
        .screenshot({ path: `${out}/channels-rgb-${appearance}-${numericFontSize}.png` });
    }
  await page.evaluate(() => window.controlActions.setAlpha(true));
  await page.getByRole('button', { name: 'RGBA 1', exact: true }).click();
  await settlePaint(page);
  const widths = await page.locator('.color-channel-fields input').evaluateAll((inputs) =>
    inputs.map((input) => {
      const style = getComputedStyle(input),
        ctx = document.createElement('canvas').getContext('2d');
      ctx.font = style.font;
      return {
        available: input.clientWidth,
        required:
          ctx.measureText('1.000').width +
          parseFloat(style.paddingLeft) +
          parseFloat(style.paddingRight),
      };
    }),
  );
  assert.equal(widths.length, 4);
  for (const width of widths) assert.ok(width.available >= width.required, JSON.stringify(width));
  await page
    .locator('.color-picker-surface')
    .screenshot({ path: `${out}/channels-rgba-large.png` });
  await writeFile(`${out}/channel-typography.json`, JSON.stringify(records, null, 2));
  await page.close();
});

test('production field and segmented families retain the same radius, height and state colors', async () => {
  const records = [];
  for (const [appearance, uiScale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    const page = await browser.newPage({ viewport: { width: 2000, height: 1200 } });
    await page.addInitScript(installPaintStyle);
    await page.addInitScript(
      (values) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(values)),
      { language: 'en', appearance, uiScale },
    );
    await page.goto(base);
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    const readGroup = async (selector) => {
      const group = page.locator(selector).first();
      const first = group.locator(':scope > button.active').first();
      await first.click();
      await page.mouse.move(0, 0);
      await settlePaint(page);
      const geometry = await group.evaluate((el) => {
        const own = window.paintStyle(el);
        return {
          radius: own.borderRadius,
          gap: own.gap,
          children: [...el.children].map((button) => {
            const s = window.paintStyle(button);
            return {
              radius: [
                s.borderTopLeftRadius,
                s.borderTopRightRadius,
                s.borderBottomRightRadius,
                s.borderBottomLeftRadius,
              ],
              height: parseFloat(s.height),
              overflow: button.scrollWidth - button.clientWidth,
              shadow: s.boxShadow,
            };
          }),
        };
      });
      assert.equal(geometry.radius, '4px', selector);
      assert.equal(geometry.gap, '0px', selector);
      geometry.children.forEach((child, index) => {
        assert.equal(child.height, 28, selector);
        assert.ok(child.overflow <= 1, selector + JSON.stringify(child));
        assert.equal(child.shadow, 'none', selector);
        assert.deepEqual(
          child.radius,
          index === 0
            ? ['4px', '0px', '0px', '4px']
            : index === geometry.children.length - 1
              ? ['0px', '4px', '4px', '0px']
              : ['0px', '0px', '0px', '0px'],
          selector,
        );
      });
      const colors = [];
      for (const state of ['rest', 'hover', 'press']) {
        if (state === 'hover') await first.hover();
        if (state === 'press') await page.mouse.down();
        await settlePaint(page);
        colors.push(
          await first.evaluate((el) => ({
            background: window.paintStyle(el).backgroundColor,
            ink: getComputedStyle(el).color,
          })),
        );
      }
      await group.screenshot({ path: `${out}/family-${selector.slice(1)}-${appearance}.png` });
      await page.mouse.up();
      records.push({ appearance, uiScale, selector, geometry, colors });
    };
    await readGroup('.geometry-selector');
    await readGroup('.timeline-follow');
    await page.locator('.title-actions button:has([data-icon=settings])').click();
    await readGroup('.settings-segment');
    await page.keyboard.press('Escape');
    await page.locator('[data-menu=Add]').click();
    await page.locator('.menu-add button:has([data-icon=camera])').click();
    await readGroup('.camera-projection');
    const heights = await page
      .locator('.inspector .number-field, .inspector .select-trigger, .inspector .axis-field')
      .evaluateAll((nodes) =>
        nodes.map((el) => ({
          kind: el.className,
          height: parseFloat(getComputedStyle(el).height),
        })),
      );
    assert.ok(heights.length > 10);
    for (const field of heights) assert.equal(field.height, 28, JSON.stringify(field));
    await page.locator('.inspector').screenshot({ path: `${out}/form-height-${appearance}.png` });
    await page.locator('.tree-row[data-tooltip="04_free_6dof"]').click();
    const xyz = await page
      .locator('.inspector .axis-field')
      .first()
      .evaluate((el) => window.paintStyle(el).borderRadius);
    await page
      .locator('.inspector-tabs')
      .getByRole('button', { name: 'Material', exact: true })
      .click();
    await page.locator('.inspector .color-trigger').first().click();
    await readGroup('.color-format');
    await page.getByRole('button', { name: 'RGB 255', exact: true }).click();
    await settlePaint(page);
    for (const node of await page.locator('.color-channel-fields .axis-field').all()) {
      assert.equal(await node.evaluate((el) => window.paintStyle(el).borderRadius), xyz);
      assert.equal(await node.evaluate((el) => parseFloat(getComputedStyle(el).height)), 28);
    }
    await page
      .locator('.color-picker-surface')
      .screenshot({ path: `${out}/picker-family-${appearance}.png` });
    const themeRows = records.filter((row) => row.appearance === appearance);
    assert.equal(themeRows.length, 5);
    for (const row of themeRows) assert.deepEqual(row.colors, themeRows[0].colors, row.selector);
    await page.close();
  }
  await writeFile(`${out}/component-inventory.json`, JSON.stringify(records, null, 2));
});
