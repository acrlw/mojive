import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

let server, browser;
const evidence = [];
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-inspector-spacing',
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
    '../output/electron/verification/inspector-spacing.json',
    JSON.stringify(evidence, null, 2),
  );
});

async function resize(page, width, scale) {
  const splitter = page.locator('[role=separator][aria-orientation=vertical]').last();
  const handle = await splitter.boundingBox(),
    pane = await page.locator('.inspector').boundingBox();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 100 * scale);
  await page.mouse.down();
  await page.mouse.move(
    handle.x + handle.width / 2 - (width * scale - pane.width),
    handle.y + 100 * scale,
    { steps: 8 },
  );
  await page.mouse.up();
  await page.waitForTimeout(50);
}

async function properties(page, selector, scale) {
  return page.locator(selector).evaluateAll(
    (groups, scale) =>
      groups
        .map((group) => {
          const rows = [...group.children].filter((row) =>
            row.matches('.property-row,.camera-property-row,.vector-group'),
          );
          const metrics = rows.map((row) => {
            const label = row.firstElementChild,
              control = row.lastElementChild;
            const a = label.getBoundingClientRect(),
              b = control.getBoundingClientRect();
            const range = document.createRange();
            range.selectNodeContents(label);
            const ink = range.getBoundingClientRect();
            return {
              label: label.textContent,
              labelWidth: a.width / scale,
              inkWidth: ink.width / scale,
              left: a.left / scale,
              controlLeft: b.left / scale,
              controlRight: b.right / scale,
              gap: (b.left - a.right) / scale,
              switch: control.matches('[role=switch]'),
              stacked: b.top >= a.bottom,
              overflow: row.scrollWidth - row.clientWidth,
              vectors: [...control.querySelectorAll('.axis-field')].map((field) => {
                const r = field.getBoundingClientRect(),
                  input = field.querySelector('input');
                const style = window.paintStyle(input),
                  context = document.createElement('canvas').getContext('2d');
                context.font = `${style.fontSize} ${style.fontFamily}`;
                return {
                  x: r.x / scale,
                  y: r.y / scale,
                  width: r.width / scale,
                  available:
                    input.getBoundingClientRect().width / scale -
                    parseFloat(style.paddingLeft) -
                    parseFloat(style.paddingRight) -
                    parseFloat(style.borderLeftWidth) -
                    parseFloat(style.borderRightWidth),
                  budget: context.measureText('-999.999').width,
                };
              }),
            };
          });
          return {
            rows: metrics,
            width: group.clientWidth,
            labelColumn:
              parseFloat(
                getComputedStyle(group.closest('.inspector-content')).gridTemplateColumns,
              ) || 0,
            layout: group.dataset.vectorLabels,
            fields: group.dataset.vectorFields,
            overflow: group.scrollWidth - group.clientWidth,
            switchBleed: group.querySelector(':scope > .switch-row') ? 8 : 0,
          };
        })
        .filter((group) => group.rows.length),
    scale,
  );
}

for (const scale of [1, 1.25, 2])
  for (const language of ['en', 'zh'])
    for (const appearance of ['dark', 'light']) {
      test(`Inspector shares intrinsic columns across sections with 12px gap and numeric budgets: ${language}/${appearance}/${scale}x`, async () => {
        const page = await browser.newPage({
          viewport: { width: Math.round(1700 * scale), height: Math.round(1050 * scale) },
        });
        await page.addInitScript(installPaintStyle);
        await page.evaluate(installPaintStyle);
        const errors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        await page.addInitScript(
          (settings) => {
            localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
            localStorage.removeItem('mojive.layout');
          },
          { language, appearance, uiScale: scale },
        );
        await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
        await page.locator('.tree-row.type-link').first().waitFor();
        await page.evaluate(() => document.fonts.ready);
        for (const width of [280, 480]) {
          await resize(page, width, scale);
          for (const kind of ['link', 'camera', 'light', 'material']) {
            await page
              .locator(`.tree-row.type-${kind === 'material' ? 'link' : kind}`)
              .first()
              .click();
            await page
              .locator('.inspector-tabs > button')
              .nth(kind === 'material' ? 1 : 0)
              .click();
            await page.waitForTimeout(40);
            const groups = await properties(page, '.inspector .section-content', scale);
            for (const group of groups) {
              // Full-row switch paint extends into the existing section inset.
              assert.ok(
                group.overflow <= group.switchBleed + 1,
                JSON.stringify({ kind, width, group }),
              );
              for (const row of group.rows) {
                assert.ok(row.overflow <= 1, JSON.stringify({ kind, width, row }));
                if (!row.stacked) {
                  assert.ok(Math.abs(row.gap - 12) < 0.8, JSON.stringify({ kind, width, row }));
                  assert.ok(
                    Math.abs(row.controlLeft - group.rows[0].controlLeft) < 0.8,
                    JSON.stringify(group),
                  );
                }
                for (const field of row.vectors)
                  assert.ok(
                    field.available + 1 >= field.budget,
                    JSON.stringify({ kind, width, row }),
                  );
                if (row.vectors.length === 3) {
                  const ys = row.vectors.map((field) => field.y);
                  assert.ok(
                    ys.every((y) => Math.abs(y - ys[0]) < 0.8) || (ys[0] < ys[1] && ys[1] < ys[2]),
                  );
                }
              }
            }
            assert.equal(
              await page
                .locator('.inspector-content')
                .evaluate((element) => element.scrollWidth - element.clientWidth),
              0,
            );
            const inline = groups.flatMap((group) => group.rows).filter((row) => !row.stacked);
            for (const row of inline) {
              assert.ok(
                Math.abs(row.controlLeft - inline[0].controlLeft) < 0.8,
                JSON.stringify({ kind, width, inline }),
              );
              assert.ok(
                Math.abs(row.labelWidth - inline[0].labelWidth) < 0.8,
                JSON.stringify({ kind, width, inline }),
              );
            }
            if (width === 480)
              assert.ok(
                Math.abs(inline[0].labelWidth - Math.max(...inline.map((row) => row.inkWidth))) <
                  1.5,
                JSON.stringify({ kind, width, inline }),
              );
            if (kind !== 'material') {
              const vector = groups.find((group) => group.layout);
              const channels = vector.rows.find((row) => row.vectors.length === 3).vectors;
              const fieldsWidth = channels.reduce(
                (sum, field) => sum + field.budget + field.width - field.available,
                8,
              );
              const labelWidth = Math.max(
                vector.labelColumn,
                ...groups.flatMap((group) => group.rows).map((row) => row.inkWidth),
              );
              assert.equal(
                vector.layout,
                vector.width >= labelWidth + 12 + fieldsWidth ? 'inline' : 'above',
                JSON.stringify({ kind, width, labelWidth, fieldsWidth, vector }),
              );
              assert.equal(vector.fields, vector.width >= fieldsWidth ? 'row' : 'stack');
            } else assert.ok(groups.length > 0);
            evidence.push({ scale, language, appearance, width, kind, groups });
            if (scale === 1 || (scale === 2 && kind === 'camera')) {
              const target =
                kind === 'camera'
                  ? '.camera-properties'
                  : kind === 'light'
                    ? '.light-properties'
                    : '.inspector .section';
              await page.locator(target).first().scrollIntoViewIfNeeded();
              await page.locator('.inspector').screenshot({
                path: `../output/electron/verification/inspector-spacing-${kind}-${language}-${appearance}-${width}-${scale}x.png`,
              });
            }
            if (kind === 'camera') {
              await page.locator('.camera-projection [role=radio]').last().click();
              const orthographic = await properties(
                page,
                '.camera-properties .section-content',
                scale,
              );
              assert.ok(
                orthographic
                  .flatMap((group) => group.rows)
                  .some((row) => /Orthographic height|正交高度/.test(row.label)),
              );
              for (const group of orthographic)
                for (const row of group.rows) {
                  assert.ok(row.overflow <= 1, JSON.stringify(row));
                  if (!row.stacked) assert.ok(Math.abs(row.gap - 12) < 0.8);
                }
              await page.locator('.camera-projection [role=radio]').first().click();
            }
          }
        }
        assert.deepEqual(errors, []);
        await page.close();
      });
    }
