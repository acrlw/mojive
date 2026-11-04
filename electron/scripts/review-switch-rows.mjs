import assert from 'node:assert/strict';
import { _electron } from 'playwright';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd(),
  out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
const evidence = [],
  errors = [];
async function review(language, appearance, scale) {
  const app = await _electron.launch({
    args: [root, '--user-data-dir=' + (await mkdtemp(path.join(out, 'switch-row-profile-')))],
    cwd: root,
    timeout: 45000,
  });
  try {
    const page = await app.firstWindow();
    await app.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows()[0].setContentSize(1700, 1100),
    );
    page.setDefaultTimeout(15000);
    page.on('pageerror', (error) => errors.push(error.message));
    await page.locator('.tree-row').first().waitFor();
    assert.ok(page.url().startsWith('mojive-app://'));
    await page.evaluate(
      (settings) => {
        localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
        localStorage.removeItem('mojive.layout');
      },
      { language, appearance, uiScale: scale },
    );
    await page.reload();
    await page.locator('.tree-row.type-camera').first().click();
    await page.evaluate(() => document.fonts.ready);
    for (const width of [280, 480]) {
      const h = await page
          .locator('[role=separator][aria-orientation=vertical]')
          .last()
          .boundingBox(),
        pane = await page.locator('.inspector').boundingBox();
      await page.mouse.move(h.x + h.width / 2, h.y + 80 * scale);
      await page.mouse.down();
      await page.mouse.move(h.x + h.width / 2 - (width * scale - pane.width), h.y + 80 * scale, {
        steps: 8,
      });
      await page.mouse.up();
      const row = page.locator('.inspector .switch-row').first();
      await row.scrollIntoViewIfNeeded();
      const geom = await row.evaluate((row, scale) => {
        const a = row.getBoundingClientRect(),
          b = row.querySelector('.switch-row-label').getBoundingClientRect(),
          c = row.querySelector('.custom-switch').getBoundingClientRect();
        return {
          leftInset: (b.left - a.left) / scale,
          rightInset: (a.right - c.right) / scale,
          click: { x: (b.right + c.left) / 2, y: c.y + c.height / 2 },
        };
      }, scale);
      assert.equal(geom.leftInset, 8);
      assert.equal(geom.rightInset, 8);
      const prefix = `switch-row-electron-${language}-${appearance}-${width}-${scale}x`;
      await page.mouse.move(geom.click.x, geom.click.y);
      const hover = await row.evaluate((n) => getComputedStyle(n).backgroundColor);
      await page.locator('.inspector').screenshot({ path: path.join(out, prefix + '-hover.png') });
      await page.mouse.down();
      assert.equal(await row.getAttribute('aria-checked'), 'true');
      const held = await row.evaluate((n) => getComputedStyle(n).backgroundColor);
      assert.notEqual(hover, held);
      await page.locator('.inspector').screenshot({ path: path.join(out, prefix + '-held.png') });
      await page.mouse.up();
      assert.equal(await row.getAttribute('aria-checked'), 'false');
      await app.evaluate(({ Menu }) =>
        Menu.getApplicationMenu().getMenuItemById('edit.undo').click(),
      );
      await page.waitForFunction(
        () =>
          document.querySelector('.inspector .switch-row').getAttribute('aria-checked') === 'true',
      );
      await row.focus();
      await page.keyboard.press('Space');
      assert.equal(await row.getAttribute('aria-checked'), 'false');
      await page.keyboard.press('Enter');
      assert.equal(await row.getAttribute('aria-checked'), 'true');
      const fields = await page.locator('.camera-property-row').evaluateAll(
        (rows, scale) =>
          rows.map((n) => {
            const a = n.firstElementChild.getBoundingClientRect(),
              b = n.lastElementChild.getBoundingClientRect();
            return {
              left: b.left,
              right: b.right,
              gap: (b.left - a.right) / scale,
              inline: b.top < a.bottom && b.bottom > a.top,
            };
          }),
        scale,
      );
      for (const f of fields) {
        assert.deepEqual(f, fields[0]);
        assert.equal(f.inline, true);
        assert.equal(f.gap, 12);
      }
      await page
        .locator('.camera-properties > .section')
        .nth(1)
        .evaluate((n) => n.scrollIntoView({ block: 'start' }));
      await page
        .locator('.inspector')
        .screenshot({ path: path.join(out, prefix + '-columns.png') });
      evidence.push({ language, appearance, scale, width, geom, hover, held, fields });
    }
    assert.deepEqual(errors, []);
  } finally {
    await app.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
    );
    await app.close();
  }
}
for (const [language, appearance, scale] of [
  ['en', 'light', 1],
  ['zh', 'dark', 1.25],
])
  await review(language, appearance, scale);
await writeFile(
  path.join(out, 'switch-row-electron.json'),
  JSON.stringify({ evidence, errors }, null, 2),
);
console.log(JSON.stringify({ cases: evidence.length, errors, captures: out }, null, 2));
