import assert from 'node:assert/strict';
import { _electron } from 'playwright';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = process.cwd(),
  out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
const app = await _electron.launch({
  args: [root, '--user-data-dir=' + (await mkdtemp(path.join(out, 'scrollbar-profile-')))],
  cwd: root,
  timeout: 45000,
});
const results = [],
  errors = [];
try {
  const page = await app.firstWindow();
  page.setDefaultTimeout(10000);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.locator('.tree-row').first().waitFor();
  for (const [appearance, uiScale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    await page.evaluate(
      (settings) => {
        localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
        localStorage.removeItem('mojive.layout');
      },
      { language: 'en', appearance, uiScale },
    );
    await page.reload();
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page
      .locator('.activity-rail')
      .getByRole('button', { name: 'Project files', exact: true })
      .click();
    await page.getByRole('button', { name: 'Open project folder', exact: true }).click();
    const location = page.getByRole('textbox', { name: 'Location', exact: true });
    await location.waitFor();
    await page.waitForFunction(
      () => document.querySelector('.file-dialog-list')?.getAttribute('aria-busy') === 'false',
    );
    await location.fill(path.join(root, 'src'));
    await page.getByRole('button', { name: 'Select folder', exact: true }).click();
    await page.locator('.file-dialog').waitFor({ state: 'hidden' });
    await page.locator('.file-row').first().waitFor();
    const owner = page.locator('.left-content');
    const g = await owner.evaluate((element) => {
      const r = element.getBoundingClientRect();
      return {
        right: r.right,
        top: r.top,
        width: r.width,
        client: element.clientHeight,
        total: element.scrollHeight,
      };
    });
    assert.ok(g.total > g.client);
    await page.waitForTimeout(1100);
    assert.equal(
      await owner.evaluate((element) => element.hasAttribute('data-scrollbar-visible')),
      false,
    );
    await page.mouse.move(g.right - 3.5 * uiScale, g.top + 12 * uiScale);
    assert.equal(await owner.getAttribute('data-scrollbar-hover'), 'y');
    await owner.screenshot({ path: path.join(out, `scrollbar-electron-${appearance}-hover.png`) });
    await page.mouse.down();
    await page.waitForTimeout(1100);
    assert.equal(await owner.getAttribute('data-scrollbar-pressed'), 'y');
    assert.equal(
      await owner.evaluate((element) => element.hasAttribute('data-scrollbar-visible')),
      true,
    );
    await owner.screenshot({ path: path.join(out, `scrollbar-electron-${appearance}-held.png`) });
    await page.mouse.move(g.right - 3.5 * uiScale, g.top + 150 * uiScale, { steps: 8 });
    await page.mouse.up();
    const scroll = await owner.evaluate((element) => element.scrollTop);
    assert.ok(scroll > 100);
    assert.equal(await owner.getAttribute('data-scrollbar-pressed'), null);
    await page.waitForTimeout(1100);
    assert.equal(
      await owner.evaluate((element) => element.hasAttribute('data-scrollbar-visible')),
      false,
    );
    await owner.screenshot({ path: path.join(out, `scrollbar-electron-${appearance}-idle.png`) });
    await page.mouse.move(g.right - 3.5 * uiScale, g.top + 151 * uiScale);
    assert.equal(
      await owner.evaluate((element) => element.hasAttribute('data-scrollbar-visible')),
      true,
    );
    await page.screenshot({
      path: path.join(out, `scrollbar-electron-${appearance}-workspace.png`),
    });
    results.push({
      appearance,
      uiScale,
      files: await page.locator('.file-row').count(),
      scroll,
      heldIdleProtected: true,
      releaseIdleHidden: true,
    });
  }
  assert.deepEqual(errors, []);
  await writeFile(
    path.join(out, 'scrollbar-electron.json'),
    JSON.stringify({ results, errors }, null, 2),
  );
  console.log(JSON.stringify(results));
} finally {
  await app.close();
}
