import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

let server, browser, page, base;
before(async () => {
  server = await createServer({
    root: process.cwd(),
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1700, height: 1050 } });
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto(base);
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function workspace(appearance, uiScale = 1) {
  await page.evaluate(
    ({ appearance, uiScale }) => {
      localStorage.setItem(
        'mojive.electron.settings.v1',
        JSON.stringify({ language: 'en', appearance, uiScale }),
      );
      localStorage.removeItem('mojive.layout');
    },
    { appearance, uiScale },
  );
  await page.reload();
  await page.getByRole('treeitem').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
}

test('material color and numeric controls share both edges after real Inspector resizing at normal and fractional scale', async () => {
  for (const scale of [1, 1.25]) {
    await workspace(scale === 1 ? 'dark' : 'light', scale);
    await page.locator('.tree-row[title="05_multi_slide_x"]').click();
    await page
      .locator('.inspector-tabs')
      .getByRole('button', { name: 'Material', exact: true })
      .click();
    await page.locator('.material-property').first().waitFor();
    const inspector = page.locator('.inspector');
    const splitter = page.getByRole('separator', { name: 'Resize Inspector', exact: true });
    for (const width of [480, 280, 400]) {
      const box = await splitter.boundingBox(),
        previous = await inspector.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + 100);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 - (width * scale - previous.width), box.y + 100, {
        steps: 8,
      });
      await page.mouse.up();
      const groups = await page.locator('.inspector .section-content').evaluateAll((sections) =>
        sections
          .map((section) =>
            [...section.querySelectorAll('.material-property')].map((row) => {
              const field = row.lastElementChild.getBoundingClientRect(),
                bounds = row.getBoundingClientRect();
              return {
                left: field.left,
                right: field.right,
                width: field.width,
                rowRight: bounds.right,
                scroll: row.scrollWidth,
                available: row.clientWidth,
              };
            }),
          )
          .filter((rows) => rows.length),
      );
      assert.ok(groups.length >= 2, 'compound body exposes multiple material groups');
      for (const rows of groups) {
        assert.equal(rows.length, 3);
        for (const row of rows) {
          assert.ok(Math.abs(row.left - rows[0].left) < 0.75, JSON.stringify(rows));
          assert.ok(Math.abs(row.right - rows[0].right) < 0.75);
          assert.ok(Math.abs(row.right - row.rowRight) < 0.75);
          assert.ok(row.width >= 139 * scale);
          assert.ok(row.scroll <= row.available + 1);
        }
      }
    }
    await inspector.screenshot({
      path: `../output/electron/verification/material-aligned-${scale === 1 ? 'dark' : 'light-125'}.png`,
    });
  }
});

test('scene row selection retains each entity hue in both themes and in filtered results', async () => {
  const roles = {
    robot: 'node-robot',
    link: 'node-link',
    geom: 'node-geom',
    joint: 'purple-bright',
    camera: 'node-camera',
    light: 'node-light',
  };
  for (const theme of ['dark', 'light']) {
    await workspace(theme);
    const backgrounds = [];
    for (const [kind, role] of Object.entries(roles)) {
      const row = page.locator(`.tree-row.type-${kind}`).first();
      await row.click();
      await page.mouse.move(1000, 40);
      const colors = await row.evaluate((element, role) => {
        const probe = document.createElement('span');
        probe.style.color = `var(--ui-${role})`;
        document.body.append(probe);
        const expected = getComputedStyle(probe).color;
        probe.remove();
        return {
          selected: element.getAttribute('aria-selected'),
          icon: getComputedStyle(element.querySelector(':scope > .icon')).color,
          expected,
          background: getComputedStyle(element).backgroundColor,
          text: getComputedStyle(element).color,
        };
      }, role);
      assert.equal(colors.selected, 'true');
      assert.equal(colors.icon, colors.expected, kind);
      assert.equal(colors.text, colors.expected, kind);
      backgrounds.push(colors.background);
      if (kind === 'joint')
        await page
          .locator('.left-panel')
          .screenshot({ path: `../output/electron/verification/tree-semantic-${theme}.png` });
    }
    assert.equal(
      new Set(backgrounds).size,
      Object.keys(roles).length,
      'selection fills retain type identity',
    );
    await page.getByRole('button', { name: 'Joint', exact: true }).click();
    const joint = page.locator('.tree-row.type-joint').first();
    await joint.click();
    assert.equal(
      await joint.evaluate(
        (element) => getComputedStyle(element.querySelector(':scope > .icon')).color,
      ),
      theme === 'dark' ? 'rgb(199, 180, 207)' : 'rgb(121, 86, 135)',
    );
    assert.equal(await joint.locator('button.tree-chevron').count(), 0);
  }
});

test('section reset stays beside disclosure with an independent hit target and preserves collapse state', async () => {
  await workspace('dark', 1.25);
  await page.locator('.tree-row[title="05_multi_anchor"]').click();
  const section = page.locator('.inspector .section:has(.section-actions)').first();
  const x = section.getByRole('textbox', { name: 'Position X', exact: true });
  await x.fill('1.25');
  await x.press('Enter');
  const reset = section.getByRole('button', { name: 'Reset transform', exact: true });
  const disclosure = section.locator('.section-disclosure');
  const a = await reset.boundingBox(),
    b = await disclosure.boundingBox();
  assert.equal(a.width, 35);
  assert.equal(a.height, 35);
  assert.equal(b.x - a.x - a.width, 5);
  assert.equal(b.y, a.y);
  await reset.hover();
  await section.screenshot({ path: '../output/electron/verification/inspector-section-actions.png' });
  await reset.click();
  assert.equal(await disclosure.getAttribute('aria-expanded'), 'true');
  assert.equal(Number(await x.inputValue()), 0);
  await x.fill('2');
  await x.press('Enter');
  await disclosure.click();
  assert.equal(await section.locator('.section-content').count(), 0);
  await disclosure.press('Enter');
  assert.equal(Number(await x.inputValue()), 2);
  await page.locator('.viewport-navigation > button:has([data-icon=fit])').waitFor();
});
