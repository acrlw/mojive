import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const preflight = process.argv.includes('--preflight');
const name = preflight ? 'bundle-preflight' : 'bundle';
const out = path.join(root, '../output/electron/verification');
const executable = path.join(root, '../output/electron/release/Mojive.app/Contents/MacOS/Mojive');
await mkdir(out, { recursive: true });
const errors = [];
const app = await _electron.launch({
  executablePath: executable,
  args: ['--hidden', '--user-data-dir=' + (await mkdtemp(path.join(out, 'bundle-profile-')))],
  cwd: root,
  timeout: 45000,
});
try {
  const page = await app.firstWindow();
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.waitForSelector('[role="treeitem"]', { timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(800);
  const status = await page.evaluate(() => {
    const canvas = document.querySelector('[data-testid="viewport-canvas"]');
    return {
      url: location.href,
      title: document.querySelector('.document-title')?.textContent,
      objects: Number(document.querySelector('.type-all')?.textContent),
      nativeMenu: window.mojive?.hasNativeMenu,
      logo: !!document.querySelector('.brand-mark'),
      miSans: document.fonts.check('14px MiSans'),
      jetBrainsMono: document.fonts.check('12px "JetBrains Mono"'),
      bridge: typeof window.mojive?.openUrdfPath,
      mediaBridge: typeof window.mojive?.saveBinaryFile,
      camerasPanel: !!document.querySelector('.activity-rail [data-icon=camera]'),
      layersPanel: !!document.querySelector('.activity-rail [data-icon=layers]'),
      timelineTracks: document.querySelectorAll('[data-lane]').length,
      nodeUnavailable: typeof window.require === 'undefined',
      viewportWidth: canvas?.width,
      viewportHeight: canvas?.height,
      renderedImageBytes: canvas?.toDataURL('image/png').length,
    };
  });
  assert.equal(status.url, 'mojive-app://app/index.html');
  assert.match(status.title, /Joint/);
  assert.equal(status.objects, 45);
  assert.equal(status.nativeMenu, true);
  assert.equal(status.logo, false);
  assert.equal(status.miSans, true);
  assert.equal(status.jetBrainsMono, true);
  assert.equal(status.bridge, 'function');
  assert.equal(status.mediaBridge, 'function');
  assert.equal(status.camerasPanel, true);
  assert.equal(status.layersPanel, true);
  assert.equal(status.timelineTracks, 3);
  assert.equal(status.nodeUnavailable, true);
  assert.ok(status.viewportWidth > 500 && status.viewportHeight > 300);
  assert.ok(status.renderedImageBytes > 10000);
  for (const name of ['碰撞', '叠加', '外观', '默认']) {
    await page.getByRole('button', { name, exact: true }).click();
    await page.waitForTimeout(150);
  }
  await page.screenshot({ path: path.join(out, `${name}.png`) });
  await page.locator('.activity-rail > button:has([data-icon=camera])').click();
  await page.locator('.camera-entry-name').first().click();
  await page.locator('.camera-properties').waitFor();
  assert.equal(await page.locator('.inspector').getByText('显示', { exact: true }).count(), 1);
  assert.equal(await page.locator('.inspector').getByText('标识', { exact: true }).count(), 1);
  await page.locator('.inspector').getByRole('switch', { name: '预览', exact: true }).click();
  await page.locator('.camera-preview-image').waitFor();
  await page.waitForTimeout(350);
  assert.ok(
    await page
      .locator('.camera-preview-image')
      .evaluate((canvas) => canvas.toDataURL().length > 10000),
  );
  const cardActions = page.locator('.camera-entry-actions > button');
  for (const index of [0, 1]) {
    await cardActions.nth(index).click();
    await page.locator('.scene-camera-return').waitFor();
    await page.waitForTimeout(350);
    assert.equal(await cardActions.nth(index).getAttribute('aria-pressed'), 'true');
    await cardActions.nth(index).click();
    await page.locator('.scene-camera-return').waitFor({ state: 'detached' });
    await page.waitForTimeout(350);
  }
  assert.equal(await page.locator('.timeline-follow > button').count(), 3);
  assert.equal(await page.locator('.timeline-settings-popover').count(), 0);
  const timeline = page.locator('.timeline');
  const captureActions = await timeline
    .locator('.timeline-capture-actions > button')
    .evaluateAll((buttons) => buttons.map((button) => button.dataset.tooltip));
  assert.deepEqual(captureActions.slice(0, 2), ['录制场景帧', '导出视频']);
  assert.equal(await timeline.getByRole('textbox', { name: '时间轴终点', exact: true }).count(), 0);
  assert.equal(await timeline.locator('.timeline-details').count(), 0);
  const laneEnd = await timeline.locator('.lane-labels').evaluate((labels) => {
    const last = labels.lastElementChild;
    return [labels.getBoundingClientRect().bottom, last.getBoundingClientRect().bottom];
  });
  assert.equal(laneEnd[0], laneEnd[1]);
  await page.screenshot({ path: path.join(out, `${name}-camera.png`) });
  await page.locator('.inspector').getByRole('switch', { name: '预览', exact: true }).click();
  await page.locator('.activity-rail > button:has([data-icon=pose])').click();
  const freeJoint = page.locator('.joint-entry').filter({ hasText: '04_free_6dof' });
  await freeJoint.locator('.entity-card-heading[aria-expanded]').click();
  const divider = await page.getByRole('separator').first().boundingBox();
  await page.mouse.move(divider.x + divider.width / 2, divider.y + 120);
  await page.mouse.down();
  await page.mouse.move(divider.x + divider.width / 2 + 150, divider.y + 120, { steps: 8 });
  await page.mouse.up();
  await freeJoint.scrollIntoViewIfNeeded();
  const axisRows = await freeJoint
    .locator('.vector-fields')
    .first()
    .evaluate((row) => [...row.children].map((field) => field.getBoundingClientRect().top));
  assert.equal(axisRows.length, 3);
  assert.ok(axisRows.every((top) => Math.abs(top - axisRows[0]) < 1));
  await page.mouse.move(900, 35);
  await page.screenshot({ path: path.join(out, `${name}-joints.png`) });
  const runtime = await app.evaluate(({ app }) => ({
    packaged: app.isPackaged,
    name: app.getName(),
    version: app.getVersion(),
    electronVersion: process.versions.electron,
  }));
  assert.equal(runtime.packaged, true);
  assert.equal(runtime.name, 'Mojive');
  assert.equal(runtime.electronVersion, '39.8.10');
  assert.deepEqual(errors, []);
  const manifest = JSON.parse(
    await readFile(path.join(root, '../output/electron/release/package-manifest.json'), 'utf8'),
  );
  await writeFile(
    path.join(out, `${name}.json`),
    JSON.stringify(
      {
        preflight,
        scope:
          'Actual packaged Mojive.app launch; embedded fonts, URDF, sandbox bridge, native menu, Chinese UI, four geometry modes, live camera preview and View/Pilot toggles',
        runtime,
        status,
        errors,
        packageManifest: manifest,
      },
      null,
      2,
    ) + '\n',
  );
  console.log(`${preflight ? 'Preflight' : 'Final'} packaged bundle smoke passed: ${executable}`);
} catch (error) {
  await writeFile(
    path.join(out, `${name}-failure.json`),
    JSON.stringify({ preflight, error: String(error), errors }, null, 2),
  );
  throw error;
} finally {
  await app.close();
}
