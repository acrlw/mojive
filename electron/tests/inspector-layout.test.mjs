import { openEditableJointScene } from './helpers/editable-scene.mjs';
import { installPaintStyle } from './helpers/surface-style.mjs';
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
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
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
  await openEditableJointScene(page);
  await page.evaluate(() => document.fonts.ready);
}
test('selected timeline lanes retain the divider and scalar Inspector joints retain their semantic label', async () => {
  for (const scale of [1, 1.25]) {
    const theme = scale === 1 ? 'dark' : 'light';
    await workspace(theme, scale);
    for (const index of [0, 2]) {
      await page.locator('.lane-label').nth(index).click();
      const bounds = await page.locator('.lane-labels').evaluate((node) => {
        const parent = node.getBoundingClientRect(),
          style = window.paintStyle(node);
        const zoom = parent.width / parseFloat(style.width);
        return {
          edge: parent.right - parseFloat(style.borderRightWidth) * zoom,
          border: parseFloat(style.borderRightWidth),
          children: [...node.querySelectorAll('.lane-label')].map(
            (child) => child.getBoundingClientRect().right,
          ),
        };
      });
      assert.ok(bounds.border > 0);
      assert.ok(
        bounds.children.every((right) => right <= bounds.edge + 0.1),
        JSON.stringify(bounds),
      );
    }
    await page
      .locator('.timeline')
      .screenshot({ path: `../output/electron/verification/timeline-divider-${theme}.png` });
    for (const [name, caption, value] of [
      ['01_revolute_y', 'Joint Angle', '25.000'],
      ['02_prismatic_x', 'Joint Position', '0.125'],
    ]) {
      await page.locator(`.tree-row[data-tooltip="${name}"]`).click();
      const row = page.locator('.joint-value-property');
      assert.equal(await row.locator(':scope > span').innerText(), caption);
      const input = row.getByRole('textbox', { name: `${name} value`, exact: true });
      await input.fill(value);
      await input.press('Enter');
      assert.equal(await input.inputValue(), value);
      const geometry = await row.evaluate((el) => {
        const label = el.firstElementChild.getBoundingClientRect(),
          fields = el.lastElementChild.getBoundingClientRect();
        return {
          overflow: el.scrollWidth - el.clientWidth,
          separate: label.right <= fields.left || label.bottom <= fields.top,
        };
      });
      assert.ok(geometry.overflow <= 1 && geometry.separate, JSON.stringify(geometry));
      await page
        .locator('.inspector')
        .screenshot({ path: `../output/electron/verification/inspector-${name}-${theme}.png` });
    }
  }
});

test('material color and numeric controls share both edges after real Inspector resizing at normal and fractional scale', async () => {
  for (const scale of [1, 1.25]) {
    await workspace(scale === 1 ? 'dark' : 'light', scale);
    await page.locator('.tree-row[data-tooltip="05_multi_slide_x"]').click();
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
                bounds = row.getBoundingClientRect(),
                previous = row.previousElementSibling?.lastElementChild.getBoundingClientRect(),
                heading = section.previousElementSibling.getBoundingClientRect(),
                group = section.parentElement.getBoundingClientRect();
              return {
                left: field.left,
                right: field.right,
                width: field.width,
                rowRight: bounds.right,
                scroll: row.scrollWidth,
                available: row.clientWidth,
                height: field.height,
                precedingGap: field.top - (previous?.bottom ?? heading.bottom),
                bottomGap: group.bottom - field.bottom,
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
          assert.ok(Math.abs(row.height - 28 * scale) < 0.1);
          assert.ok(Math.abs(row.precedingGap - 8 * scale) < 0.1);
        }
        assert.ok(Math.abs(rows.at(-1).bottomGap - 12 * scale) <= 1.1);
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
        const expected = window.paintStyle(probe).color;
        probe.remove();
        return {
          selected: element.getAttribute('aria-selected'),
          icon: window.paintStyle(element.querySelector(':scope > .icon')).color,
          expected,
          background: window.paintStyle(element).backgroundColor,
          text: window.paintStyle(element).color,
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
        (element) => window.paintStyle(element.querySelector(':scope > .icon')).color,
      ),
      theme === 'dark' ? 'rgb(199, 180, 207)' : 'rgb(121, 86, 135)',
    );
    assert.equal(await joint.locator('button.tree-chevron').count(), 0);
  }
});

test('section reset stays beside disclosure with an independent hit target and preserves collapse state', async () => {
  await workspace('dark', 1.25);
  await page.locator('.tree-row[data-tooltip="05_multi_anchor"]').click();
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
  await section.screenshot({
    path: '../output/electron/verification/inspector-section-actions.png',
  });
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

test('light has one activation control and shared section titles keep rounded, padded hover', async () => {
  for (const [theme, scale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    await workspace(theme, scale);
    const row = page.locator('.tree-row.type-light').first();
    await row.click();
    const inspector = page.locator('.inspector');
    assert.equal(await inspector.getByRole('switch', { name: 'Visible', exact: true }).count(), 0);
    const enabled = inspector.getByRole('switch', { name: 'Light enabled', exact: true });
    assert.equal(await enabled.count(), 1);
    assert.equal(await enabled.getAttribute('aria-checked'), 'true');
    await enabled.click();
    assert.equal(await enabled.getAttribute('aria-checked'), 'false');
    assert.equal(await row.locator('.tree-visibility').getAttribute('data-tooltip'), 'Show object');
    await page.keyboard.press('ControlOrMeta+z');
    assert.equal(await enabled.getAttribute('aria-checked'), 'true');
    await row.locator('.tree-visibility').click();
    assert.equal(await enabled.getAttribute('aria-checked'), 'false');
    await page.keyboard.press('ControlOrMeta+z');
    const title = inspector.getByRole('button', { name: 'Transform', exact: true });
    await title.hover();
    const geometry = await title.evaluate((el) => {
      const style = window.paintStyle(el),
        r = el.getBoundingClientRect(),
        range = document.createRange();
      range.selectNodeContents(el);
      return {
        radius: parseFloat(style.borderRadius),
        padding: parseFloat(style.paddingLeft),
        left: range.getBoundingClientRect().left,
        r: r.left,
        bg: style.backgroundColor,
      };
    });
    assert.equal(geometry.radius, 4);
    assert.equal(geometry.padding, 8);
    assert.ok(Math.abs(geometry.left - geometry.r - 8 * scale) < 0.2);
    assert.notEqual(geometry.bg, 'rgba(0, 0, 0, 0)');
    await inspector.screenshot({
      path: `../output/electron/verification/inspector-light-${theme}.png`,
    });
  }
});

test('free-joint coordinates share the Transform label column when the Inspector has room', async () => {
  for (const scale of [1, 1.25]) {
    await workspace(scale === 1 ? 'dark' : 'light', scale);
    const splitter = page.getByRole('separator', { name: 'Resize Inspector', exact: true });
    const box = await splitter.boundingBox();
    const current = Number(await splitter.getAttribute('aria-valuenow'));
    await page.mouse.move(box.x + box.width / 2, box.y + 100);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 - (480 - current) * scale, box.y + 100, {
      steps: 8,
    });
    await page.mouse.up();
    await page.locator('.tree-row[data-tooltip="04_free_6dof"]').click();
    const rows = await page.locator('.inspector .vector-group').evaluateAll((nodes) =>
      nodes.map((node) => {
        const label = node.querySelector('label').getBoundingClientRect();
        const fields = node.querySelector('.vector-fields').getBoundingClientRect();
        const channels = [...node.querySelectorAll('.axis-field')].map((field) =>
          field.getBoundingClientRect().toJSON(),
        );
        return { label: label.toJSON(), fields: fields.toJSON(), channels };
      }),
    );
    assert.equal(rows.length, 5);
    for (const row of rows) {
      assert.ok(
        Math.abs(row.label.y + row.label.height / 2 - row.fields.y - row.fields.height / 2) < 1,
        JSON.stringify(row),
      );
      assert.ok(Math.abs(row.fields.x - rows[0].fields.x) < 1);
      assert.ok(row.channels.every((channel) => Math.abs(channel.height - 28 * scale) < 0.1));
    }
    await page.locator('.inspector').screenshot({
      path: `../output/electron/verification/inspector-free-joint-inline-${scale}.png`,
    });
  }
});
