import { chooseSampleScene } from '../tests/helpers/editable-scene.mjs';
import assert from 'node:assert/strict';
import { _electron } from 'playwright';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
const profile = await mkdtemp(path.join(out, 'timeline-wheel-profile-'));
const env = { ...process.env };
delete env.MOJIVE_DEV_URL;
const report = {
  application: 'Production Electron App',
  indexSha256: createHash('sha256')
    .update(await readFile(path.join(root, 'dist/index.html')))
    .digest('hex'),
  states: [],
  errors: [],
};
let app;
try {
  app = await _electron.launch({
    args: [root, '--hidden', '--user-data-dir=' + profile],
    cwd: root,
    env,
    timeout: 45000,
  });
  const page = await app.firstWindow();
  page.setDefaultTimeout(15000);
  page.on('pageerror', (error) => report.errors.push(error.stack));
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].setContentSize(1700, 1000),
  );
  await page.locator('.tree-row').first().waitFor();
  assert.equal(new URL(page.url()).protocol, 'mojive-app:');
  const panel = page.getByTestId('timeline');
  const body = panel.locator('.timeline-body');
  const track = panel.locator('.timeline-track');
  const numeric = async (label, value) => {
    const field = panel.getByLabel(label, { exact: true }).filter({ visible: true });
    await field.dblclick();
    await field.fill(String(value));
    await field.press('Enter');
  };
  const view = () =>
    body.evaluate((el) => ({ start: +el.dataset.viewStart, end: +el.dataset.viewEnd }));
  const span = (range) => range.end - range.start;
  const settle = () =>
    page.evaluate(
      () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
    );
  const follow = (mode) =>
    panel.locator(`.timeline-follow button:has([data-icon=follow-${mode}])`).click();
  const fit = () => panel.locator('.timeline-toolbar button:has([data-icon=fit])').click();
  const wheel = async (ratio, delta, mode = 0, shift = false) => {
    const b = await track.boundingBox();
    const cssWidth = await track.evaluate((el) => el.clientWidth);
    const scale = b.width / cssWidth;
    const x = b.x + (12 + ratio * (cssWidth - 24)) * scale;
    const y = b.y + 14 * scale;
    const previousEvent = await page.evaluate(() => window.timelineWheelCount || 0);
    if (mode === 0) {
      await page.mouse.move(x, y);
      if (shift) await page.keyboard.down('Shift');
      await page.mouse.wheel(0, delta);
      if (shift) await page.keyboard.up('Shift');
    } else {
      await track.dispatchEvent('wheel', {
        clientX: x,
        clientY: y,
        deltaY: delta,
        deltaMode: mode,
        bubbles: true,
        cancelable: true,
      });
    }
    await page.waitForFunction((count) => window.timelineWheelCount > count, previousEvent);
    await settle();
    const actualX = await page.evaluate(() => window.lastTimelineWheelX);
    return ((actualX - b.x) / scale - 12) / (cssWidth - 24);
  };
  for (const [appearance, uiScale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    if (report.states.length) {
      // Discard only this isolated profile's test recording through the normal scene flow.
      // Reloading a dirty document correctly opens the application's close confirmation.
      await chooseSampleScene(page, 'Joint gizmos');
      await page.getByRole('alertdialog').getByRole('button', { name: "Don't Save" }).click();
      await page.getByRole('alertdialog').waitFor({ state: 'hidden' });
      await page.waitForFunction(
        () => Number(document.querySelector('.time-readout input')?.value) === 0,
      );
    }
    await page.evaluate(
      (settings) => {
        localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
        localStorage.setItem(
          'mojive.layout',
          JSON.stringify({ left: 260, right: 300, bottom: 300 }),
        );
      },
      { appearance, uiScale, language: 'en' },
    );
    await page.reload();
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() =>
      document.addEventListener(
        'wheel',
        (e) => {
          window.lastTimelineWheelX = e.clientX;
          window.timelineWheelCount = (window.timelineWheelCount || 0) + 1;
        },
        true,
      ),
    );
    await panel.waitFor();
    await follow('off');
    await fit();
    await numeric('View end', 10);
    await numeric('View start', 0);
    const checks = [];
    for (const [ratio, delta, mode] of [
      [0.62, -100, 0],
      [0.62, 100, 0],
      [0.2, -2, 0],
      [0.2, 2, 0],
      [0.7, -3, 1],
      [0.7, 1, 2],
    ]) {
      const before = await view();
      const actualRatio = await wheel(ratio, delta, mode);
      const after = await view();
      const expectedStart = Math.max(0, before.start + actualRatio * (span(before) - span(after)));
      const error = Math.abs(expectedStart - after.start);
      assert.ok(error < 1e-7, `Cursor anchor error: ${error}`);
      const units = delta / (mode === 0 ? 100 : mode === 1 ? 3 : 1);
      const expectedFactor = Math.exp(Math.max(-4, Math.min(4, units)) * 0.18);
      assert.ok(
        Math.abs(span(after) / span(before) - expectedFactor) < 1e-9,
        `Incorrect wheel magnitude for delta ${delta}, mode ${mode}`,
      );
      checks.push({ delta, mode, ratio: actualRatio, before, after, anchorErrorSeconds: error });
    }
    const beforePan = await view();
    await wheel(0.5, 100, 0, true);
    const afterPan = await view();
    assert.ok(Math.abs(span(afterPan) - span(beforePan)) < 1e-7);
    assert.notEqual(afterPan.start, beforePan.start);
    // Preserve the native locked-follow exception: the playhead, not the pointer, anchors zoom.
    await numeric('Current time', 2);
    await follow('locked');
    const beforeLock = await view();
    const fraction = (2 - beforeLock.start) / span(beforeLock);
    await wheel(0.9, -100);
    const afterLock = await view();
    assert.ok(Math.abs((2 - afterLock.start) / span(afterLock) - fraction) < 1e-7);
    await follow('off');
    await fit();
    for (let i = 0; i < 24; i++) await wheel(0.58, 1000000);
    const maximum = await view();
    assert.ok(span(maximum) <= 60.000001, `Runaway view: ${JSON.stringify(maximum)}`);
    assert.equal(maximum.start, 0, 'zoom-out cannot expose negative time');
    await wheel(0.5, -1000000, 0, true);
    assert.equal((await view()).start, 0, 'pan cannot cross the zero boundary');
    assert.ok(
      (await panel.locator('.ruler-tick').allTextContents()).every((t) => parseFloat(t) >= 0),
    );
    const boundaryCapture = `timeline-zero-${appearance}-${uiScale}x.png`;
    await page.screenshot({ path: path.join(out, boundaryCapture) });
    for (let i = 0; i < 24; i++) await wheel(0.58, -1000000);
    const minimum = await view();
    assert.ok(span(minimum) >= 0.00099999);
    assert.ok(span(minimum) < 0.001001);
    await numeric('Current time', 0);
    await fit();
    await follow('page');
    const record = panel.locator('.record-button');
    await record.click();
    await settle();
    assert.ok((await view()).start >= 0, 'recording follow starts at zero');
    await page.waitForFunction(
      () => Number(document.querySelector('.time-readout input')?.value) > 10.2,
      undefined,
      { timeout: 15000 },
    );
    await record.click();
    const recordedThrough = Number(await panel.locator('.time-readout input').inputValue());
    assert.ok(recordedThrough > 10);
    await fit();
    assert.equal(await panel.locator('.lane-label').count(), 3);
    await page.mouse.move(20, 20);
    await settle();
    const capture = `timeline-wheel-${appearance}-${uiScale}x.png`;
    await page.screenshot({ path: path.join(out, capture) });
    report.states.push({
      appearance,
      uiScale,
      checks,
      beforePan,
      afterPan,
      beforeLock,
      afterLock,
      maximum,
      minimum,
      boundaryCapture,
      recordedThrough,
      capture,
    });
    console.log(
      `PASS ${appearance}/${uiScale}: zero boundary, cursor anchoring, bounded wheel, pan, locked follow, recording > 10 s`,
    );
  }
  assert.deepEqual(report.errors, []);
} catch (error) {
  report.errors.push(error.stack || String(error));
  if (app) {
    const page = await app.firstWindow().catch(() => null);
    await page?.screenshot({ path: path.join(out, 'timeline-wheel-failure.png') }).catch(() => {});
  }
  throw error;
} finally {
  await writeFile(path.join(out, 'timeline-wheel-electron.json'), JSON.stringify(report, null, 2));
  if (app) {
    await app
      .evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
      )
      .catch(() => {});
    await app.close();
  }
  await rm(profile, { recursive: true, force: true });
}
