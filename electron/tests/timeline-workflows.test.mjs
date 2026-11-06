import { openEditableScene } from './helpers/editable-scene.mjs';
import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page, base;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-timeline-workflows-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime', 'three'],
    },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    );
    localStorage.setItem('mojive.layout', JSON.stringify({ left: 280, right: 360, bottom: 320 }));
  });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  errors.length = 0;
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await openEditableScene(page);
  await page.getByTestId('viewport-canvas').waitFor();
  await page.getByTestId('timeline').waitFor();
  await page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'Add', exact: true })
    .click();
  await page.getByRole('menuitem', { name: 'Box', exact: true }).click();
  await page.getByLabel('Position X', { exact: true }).waitFor();
});
const timeline = () => page.getByTestId('timeline');
const xField = () => page.getByLabel('Position X', { exact: true });
const timeField = () => timeline().getByLabel('Current time', { exact: true });
async function setting(label) {
  return timeline().getByLabel(label, { exact: true }).filter({ visible: true });
}
async function numeric(locator, value) {
  await locator.dblclick();
  await locator.fill(String(value));
  await locator.press('Enter');
}
async function x(value) {
  if (value === undefined) return Number(await xField().inputValue());
  await numeric(xField(), value);
}
async function time(value) {
  if (value === undefined) return Number(await timeField().inputValue());
  await numeric(timeField(), value);
}
async function captureKey() {
  await timeline()
    .getByRole('button', { name: 'Capture scene pose at playhead (K)', exact: true })
    .click();
}
const keys = () => timeline().locator('[data-scene-frame]');
const details = () => timeline().locator('.timeline-details');
async function assertNoErrors() {
  assert.deepEqual(errors, []);
}
async function roundTripWorkspace() {
  // Exercise the production save/open dialogs while replacing only the desktop filesystem boundary.
  await page.evaluate(() => {
    window.recordingFile = null;
    window.mojive = {
      platform: 'linux',
      hasNativeMenu: false,
      onMenuAction: () => () => {},
      updateMenuState: async () => {},
      browseFiles: async () => ({
        path: '/workspace',
        parent: null,
        shortcuts: [],
        truncated: false,
        entries: window.recordingFile
          ? [
              {
                name: 'timeline-roundtrip.mojive.json',
                path: '/workspace/timeline-roundtrip.mojive.json',
                directory: false,
              },
            ]
          : [],
      }),
      saveScenePath: async (path, content) => {
        window.recordingFile = { path, content };
        return path;
      },
      openScenePath: async () => window.recordingFile,
    };
  });
  const fileMenu = page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'File', exact: true });
  await fileMenu.click();
  await page.getByRole('menuitem', { name: /^Save Scene As…/ }).click();
  const saveDialog = page.getByRole('dialog', { name: 'Save scene as', exact: true });
  await saveDialog.locator('.file-dialog-list[aria-busy="false"]').waitFor();
  await saveDialog
    .getByRole('textbox', { name: 'File name', exact: true })
    .fill('timeline-roundtrip.mojive.json');
  await saveDialog.getByRole('button', { name: 'Save', exact: true }).click();
  await saveDialog.waitFor({ state: 'hidden' });
  const saved = await page.evaluate(() => JSON.parse(window.recordingFile.content));
  await fileMenu.click();
  await page.getByRole('menuitem', { name: /^Open Scene…/ }).click();
  const openDialog = page.getByRole('dialog', { name: 'Open saved scene', exact: true });
  await openDialog
    .getByRole('option', { name: 'timeline-roundtrip.mojive.json', exact: true })
    .dblclick();
  await openDialog.waitFor({ state: 'hidden' });
  return saved;
}

test('actual App retains two same-time keys and distinguishes selection, Load and metadata Apply', async () => {
  await x(1);
  await time(0.1);
  await captureKey();
  await x(2);
  await captureKey();
  assert.equal(await keys().count(), 2, 'Capture appends a new preset even at the same time');
  await x(3);
  const second = keys().last();
  await second.click();
  assert.equal(await x(), 3, 'Selecting a key only selects its marker/time');
  assert.equal(await time(), 0.1);
  await second.dblclick();
  assert.equal(await x(), 2, 'Double-click loads the captured pose');
  await x(4);
  await details().getByLabel('Keyframe name', { exact: true }).fill('Reach pose');
  await numeric(details().getByLabel('Keyframe time', { exact: true }), 0.2);
  await details().getByRole('button', { name: 'Apply keyframe details', exact: true }).click();
  assert.equal(await x(), 4, 'Apply metadata never recaptures or loads the current scene');
  assert.match(await second.getAttribute('data-tooltip'), /Reach pose.*0\.200/);
  await details().getByRole('button', { name: 'Load', exact: true }).click();
  assert.equal(await x(), 2, 'Saved pose survives metadata changes');
  assert.equal(await time(), 0.2);
  await details().getByLabel('Keyframe name', { exact: true }).fill('Discard this draft');
  await details().getByLabel('Keyframe name', { exact: true }).press('Escape');
  assert.equal(
    await details().getByLabel('Keyframe name', { exact: true }).inputValue(),
    'Reach pose',
  );
  await second.screenshot({ path: '../output/electron/verification/timeline-key-marker.png' });
  await page.screenshot({ path: '../output/electron/verification/timeline-three-tracks.png' });
  await assertNoErrors();
});

test('empty Take lane has no metadata footer and its header retains real context actions', async () => {
  const take = timeline().locator('.take-lane');
  await take.click({ position: { x: 30, y: 30 } });
  assert.equal(await details().count(), 0);
  await timeline().locator('.take-label').click({ button: 'right' });
  const menu = page.getByRole('menu', { name: 'Timeline actions', exact: true });
  assert.equal(
    await menu.getByRole('menuitem', { name: 'New Take', exact: true }).isEnabled(),
    true,
  );
  assert.equal(await menu.getByRole('menuitem', { name: 'Delete take', exact: true }).count(), 0);
  await menu.press('Escape');
  await assertNoErrors();
});

test('actual App snapshot captures timeline time and restores its independent pose', async () => {
  await x(1.25);
  await time(0.125);
  await timeline().getByRole('button', { name: 'Capture snapshot', exact: true }).click();
  await x(9);
  await time(0.75);
  await page.waitForTimeout(140);
  const snapshot = timeline().locator('[data-lane="snapshots"] button').first();
  await snapshot.click();
  assert.equal(await x(), 9, 'Selecting a snapshot does not load it');
  assert.match(await details().locator('.detail-time').innerText(), /0\.125 s/);
  await details().getByRole('button', { name: 'Load', exact: true }).click();
  assert.equal(await x(), 1.25);
  assert.equal(await time(), 0.125, 'Snapshot time is its capture playhead, not wallclock');
  await details().getByRole('button', { name: 'Delete selected item', exact: true }).click();
  assert.equal(await timeline().locator('[data-lane="snapshots"] button').count(), 0);
  assert.equal(await x(), 1.25, 'Deleting snapshot metadata does not undo the loaded pose');
  await assertNoErrors();
});

test('actual App records manual scene edits into take frames, steps samples and switches independent takes', async () => {
  await x(0);
  await time(0);
  await timeline()
    .getByRole('button', { name: /^Record (scene frames|from playhead)$/ })
    .click();
  await page.waitForTimeout(40);
  await x(2);
  await page.waitForTimeout(60);
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const end = await time();
  assert.ok(end > 0);
  await timeline()
    .locator('[data-lane="take"]')
    .click({ position: { x: 30, y: 20 } });
  const count = Number(
    (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
      / · (\d+) frames$/,
    )[1],
  );
  assert.ok(count >= 2, `Recorded actual initial/final samples: ${count}`);
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await x(), 0, 'Initial frame was retained rather than overwritten at Stop');
  assert.equal(await time(), 0);
  await timeline().getByRole('button', { name: 'Next frame', exact: true }).click();
  const next = await time();
  assert.ok(next > 0 && next <= end);
  await timeline().getByRole('button', { name: 'Previous frame', exact: true }).click();
  assert.equal(await time(), 0);
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await x(), 2);
  assert.equal(await time(), end);
  await x(5);
  await timeline().getByRole('button', { name: 'New Take', exact: true }).click();
  await timeline()
    .getByRole('button', { name: /^Record (scene frames|from playhead)$/ })
    .click();
  await page.waitForTimeout(50);
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const take = timeline().getByRole('combobox', { name: 'Take', exact: true });
  await take.click();
  assert.equal(await page.getByRole('option').count(), 2);
  await page.getByRole('option', { name: 'Take 1', exact: true }).click();
  assert.equal(await x(), 2, 'Choosing Take1 restores its remembered final-frame pose');
  await take.click();
  await page.getByRole('option', { name: 'Take 2', exact: true }).click();
  assert.equal(await x(), 5);
  await assertNoErrors();
});

test('actual App manually stopped take can loop past its tail and pause without freezing', async () => {
  await time(0);
  await x(0);
  await timeline()
    .getByRole('button', { name: /^Record (scene frames|from playhead)$/ })
    .click();
  await page.waitForTimeout(60);
  await x(1);
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="Current time"]').value) >= 0.35,
  );
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const recordedEnd = await time();
  assert.ok(recordedEnd >= 0.35);
  const loopButton = timeline().getByRole('button', { name: 'Loop playback', exact: true });
  if (!(await loopButton.getAttribute('class')).includes('active')) await loopButton.click();
  await timeline().getByRole('button', { name: 'Play (Space)', exact: true }).click();
  const samples = await page.evaluate(async (recordedEnd) => {
    const samples = [];
    // Stop-button delivery varies with renderer load; verify two complete loops
    // using the actual captured span instead of assuming a sub-800ms take.
    const deadline = performance.now() + recordedEnd * 4000 + 500;
    let wraps = 0;
    while (wraps < 2 && performance.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20));
      const value = Number(document.querySelector('[aria-label="Current time"]').value);
      if (samples.length && value < samples.at(-1) - 0.03) wraps++;
      samples.push(value);
    }
    return samples;
  }, recordedEnd);
  const wraps = samples.filter((value, index) => index && value < samples[index - 1] - 0.03).length;
  assert.ok(wraps >= 2, `Several loops must wrap rather than stall at the last sample: ${samples}`);
  assert.equal(
    await timeline().getByRole('button', { name: 'Pause (Space)', exact: true }).count(),
    1,
  );
  await timeline().getByRole('button', { name: 'Pause (Space)', exact: true }).click();
  const stopped = await time();
  await page.waitForTimeout(100);
  assert.equal(await time(), stopped);
  await assertNoErrors();
});

test('actual App recording passes ten seconds, resumes past its tail and saves every frame without an End limit', async () => {
  assert.equal(await timeline().getByLabel('Timeline end', { exact: true }).count(), 0);
  await time(0);
  await x(0);
  await timeline().getByRole('button', { name: 'Record scene frames', exact: true }).click();
  // Real elapsed time exercises RAF, the recording clock, retained frames and extent growth together.
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="Current time"]').value) > 10.25,
    undefined,
    { timeout: 15000 },
  );
  assert.equal(
    await timeline().getByRole('button', { name: 'Stop recording', exact: true }).isVisible(),
    true,
  );
  await x(7);
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const firstEnd = await time();
  assert.ok(firstEnd > 10.25);
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await time(), 0);
  assert.equal(await x(), 0);
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await time(), firstEnd);
  assert.equal(await x(), 7, 'The final pose after the former ten-second boundary is retained');

  await numeric(await setting('Loop start'), 0.1);
  await numeric(await setting('Loop end'), 0.3);
  assert.ok(Number(await (await setting('Loop end')).inputValue()) < 0.5);
  await time(firstEnd);
  await timeline().getByRole('button', { name: 'Record from playhead', exact: true }).click();
  await page.waitForFunction(
    (previousEnd) =>
      Number(document.querySelector('[aria-label="Current time"]').value) > previousEnd + 0.2,
    firstEnd,
  );
  assert.equal(
    await timeline().getByRole('button', { name: 'Stop recording', exact: true }).isVisible(),
    true,
  );
  await x(9);
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const resumedEnd = await time();
  assert.ok(
    resumedEnd > firstEnd + 0.2,
    'Neither the existing take tail nor the playback range stops recording',
  );
  const count = Number(
    (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
      / · (\d+) frames$/,
    )[1],
  );
  const selectedName = await page.locator('.inspector-heading').getAttribute('data-tooltip');

  const saved = await roundTripWorkspace();
  const frames = saved.timeline.takes[0].frames;
  assert.equal(
    frames.length,
    count,
    'Persistence includes all retained samples, not only visible ticks',
  );
  assert.equal(frames[0].time, 0);
  assert.ok(Math.abs(frames.at(-1).time - resumedEnd) <= 0.001);
  assert.ok(saved.duration >= resumedEnd, 'Saved extent grows to encompass the complete take');

  await page.locator('.tree-row').filter({ hasText: selectedName }).click();
  assert.equal(
    Number(
      (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
        / · (\d+) frames$/,
      )[1],
    ),
    count,
  );
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await x(), 0);
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await time(), resumedEnd);
  assert.equal(await x(), 9, 'The round-tripped final frame is still loadable');
  await page.screenshot({
    path: '../output/electron/verification/timeline-recording-beyond-ten-seconds.png',
  });
  await assertNoErrors();
});

test('Undo after recording across an extent boundary keeps the cursor saveable for new keys and snapshots', async () => {
  await time(9.9);
  await x(0);
  await timeline().getByRole('button', { name: 'Record scene frames', exact: true }).click();
  await x(3);
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="Current time"]').value) > 10.2,
  );
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const cursor = await time();
  const selectedName = await page.locator('.inspector-heading').getAttribute('data-tooltip');
  const undo = async (redo = false) => {
    await page
      .getByRole('navigation', { name: 'Application menu' })
      .getByRole('button', { name: 'Edit', exact: true })
      .click();
    await page.getByRole('menuitem', { name: redo ? /^Redo/ : /^Undo/ }).click();
  };
  await undo();
  assert.equal(await time(), cursor, 'Undo preserves the navigation cursor beyond ten seconds');
  assert.equal(await x(), 0, 'Undo restores the scene pose from before the complete recording');
  assert.equal(await timeline().getByRole('combobox', { name: 'Take', exact: true }).count(), 0);
  await undo(true);
  assert.equal(await x(), 3, 'Redo restores the complete recording and its final pose');
  assert.equal(await timeline().getByRole('combobox', { name: 'Take', exact: true }).count(), 1);
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await x(), 0);
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await x(), 3);
  assert.equal(await time(), cursor);
  await undo();
  await captureKey();
  await timeline().getByRole('button', { name: 'Capture snapshot', exact: true }).click();
  assert.equal(await keys().count(), 1);
  assert.equal(await timeline().locator('[data-lane="snapshots"] button').count(), 1);
  const saved = await roundTripWorkspace();
  assert.ok(saved.duration >= cursor, 'Undo cannot shrink the saved extent below the kept cursor');
  assert.equal(saved.timeline.takes.length, 0);
  assert.equal(saved.keys.length, 1);
  assert.equal(saved.timeline.snapshots.length, 1);
  assert.ok(Math.abs(saved.keys[0].time - cursor) < 0.001);
  assert.ok(Math.abs(saved.timeline.snapshots[0].time - cursor) < 0.001);
  await page.locator('.tree-row').filter({ hasText: selectedName }).click();
  await x(8);
  await keys().first().dblclick();
  assert.equal(await x(), 0, 'The post-Undo keyframe remains loadable after saving and reopening');
  await x(9);
  await timeline().locator('[data-lane="snapshots"] button').first().click();
  await details().getByRole('button', { name: 'Load', exact: true }).click();
  assert.equal(await x(), 0, 'The post-Undo snapshot remains loadable after saving and reopening');
  await assertNoErrors();
});

async function threeKeys() {
  for (const [value, at] of [
    [1, 0.5],
    [2, 1],
    [3, 1.5],
  ]) {
    await x(value);
    await time(at);
    await captureKey();
  }
}
const selectedKeys = () =>
  keys()
    .filter({ has: page.locator('.primary-marker') })
    .locator('xpath=self::*[contains(@class,"selected")]');
async function pointAt(at, lane = 'keyframes', row = 35) {
  const ticks = await timeline()
    .locator('.ruler-tick')
    .evaluateAll((nodes) =>
      nodes
        .slice(0, 2)
        .map((node) => ({ time: parseFloat(node.textContent), x: node.getBoundingClientRect().x })),
    );
  assert.equal(ticks.length, 2, 'Visible ruler ticks define the actual timeline projection');
  const laneBox = await timeline().locator(`[data-lane="${lane}"]`).boundingBox();
  const pixelsPerSecond = (ticks[1].x - ticks[0].x) / (ticks[1].time - ticks[0].time);
  return { x: ticks[0].x + (at - ticks[0].time) * pixelsPerSecond, y: laneBox.y + row };
}
async function drag(from, to, modifiers = []) {
  for (const key of modifiers) await page.keyboard.down(key);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 8 });
  await page.mouse.up();
  for (const key of modifiers) await page.keyboard.up(key);
}

test('actual App key selection, clipboard, duplicate, nudge and Delete keep captured poses independent', async () => {
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await threeKeys();
  await keys().first().click();
  await keys()
    .last()
    .click({ modifiers: ['ControlOrMeta'] });
  assert.equal(await selectedKeys().count(), 2);
  await keys()
    .nth(1)
    .click({ modifiers: ['Shift'] });
  assert.equal(
    await selectedKeys().count(),
    2,
    'Shift-click selects the interval from the primary key',
  );
  await timeline().press('ControlOrMeta+a');
  assert.equal(await selectedKeys().count(), 3);
  await timeline().press('ControlOrMeta+c');
  await timeline().press('ControlOrMeta+d');
  assert.equal(await keys().count(), 6);
  assert.equal(await selectedKeys().count(), 3);
  await timeline().press('Delete');
  assert.equal(await keys().count(), 3);
  assert.equal(
    await page.getByLabel('Position X', { exact: true }).count(),
    1,
    'Timeline Delete never deletes the selected scene object',
  );
  await time(2);
  await timeline().focus();
  await timeline().press('ControlOrMeta+v');
  await page.waitForFunction(() => document.querySelectorAll('[data-scene-frame]').length === 6);
  assert.equal(await selectedKeys().count(), 3);
  const pasted = await selectedKeys().evaluateAll((items) =>
    items.map((item) => item.dataset.sceneFrame),
  );
  await timeline().press('ArrowRight');
  for (const [index, id] of pasted.entries()) {
    assert.match(
      await timeline().locator(`[data-scene-frame="${id}"]`).getAttribute('aria-label'),
      new RegExp(`${(2 + index * 0.5 + 1 / 60).toFixed(2)} seconds`),
    );
  }
  await timeline().locator(`[data-scene-frame="${pasted[0]}"]`).dblclick();
  assert.equal(await x(), 1, 'Pasted preset contains its source pose, not the current scene');
  await timeline().press('ControlOrMeta+a');
  await timeline().press('Delete');
  assert.equal(await keys().count(), 0);
  assert.equal(await x(), 1);
  await assertNoErrors();
});

test('actual App box selection and held key drag preview cancel atomically before the next gesture', async () => {
  await threeKeys();
  await keys().last().click();
  await timeline().getByRole('button', { name: 'Previous keyframe', exact: true }).click();
  assert.equal(await x(), 2);
  await timeline().getByRole('button', { name: 'Next keyframe', exact: true }).click();
  assert.equal(await x(), 3);
  await drag(await pointAt(0.25), await pointAt(1.25));
  assert.equal(await selectedKeys().count(), 2);
  const first = keys().first(),
    id = await first.getAttribute('data-scene-frame');
  const rect = await first.boundingBox(),
    from = { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  const before = await first.getAttribute('data-tooltip');
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(from.x + 88, from.y, { steps: 8 });
  assert.ok(
    (await first.boundingBox()).x > rect.x + 70,
    'Held drag paints its uncommitted preview',
  );
  assert.equal(
    await first.getAttribute('data-tooltip'),
    before,
    'Stored key metadata stays unchanged before release',
  );
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.equal(await first.getAttribute('data-tooltip'), before);
  assert.ok(Math.abs((await first.boundingBox()).x - rect.x) < 0.1);
  await drag(from, { x: from.x + 88, y: from.y });
  const moved = timeline().locator(`[data-scene-frame="${id}"]`);
  assert.match(await moved.getAttribute('data-tooltip'), /1\.500 s/);
  await assertNoErrors();
});

test('actual App wheel zoom, right pan, follow modes and context menu remain usable above panel clipping', async () => {
  await threeKeys();
  const body = timeline().locator('.timeline-body'),
    track = timeline().locator('.timeline-track');
  const before = await body.evaluate(
    (el) => Number(el.dataset.viewEnd) - Number(el.dataset.viewStart),
  );
  const p = await pointAt(2);
  await page.mouse.move(p.x, p.y);
  await page.mouse.wheel(0, -800);
  await page.waitForTimeout(50);
  assert.ok(
    (await body.evaluate((el) => Number(el.dataset.viewEnd) - Number(el.dataset.viewStart))) <
      before / 1.5,
  );
  const scrollBefore = await body.evaluate((el) => el.scrollLeft);
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(p.x - 90, p.y, { steps: 6 });
  await page.mouse.up({ button: 'right' });
  assert.ok((await body.evaluate((el) => el.scrollLeft)) > scrollBefore + 70);
  assert.match(
    await timeline().getByRole('button', { name: 'Follow off', exact: true }).getAttribute('class'),
    /active/,
  );
  await timeline().getByRole('button', { name: 'Follow locked', exact: true }).click();
  assert.match(
    await timeline()
      .getByRole('button', { name: 'Follow locked', exact: true })
      .getAttribute('class'),
    /active/,
  );
  await timeline().getByRole('button', { name: 'Follow page', exact: true }).click();
  assert.match(
    await timeline()
      .getByRole('button', { name: 'Follow page', exact: true })
      .getAttribute('class'),
    /active/,
  );
  await setting('Timeline zoom');
  await timeline()
    .getByRole('button', { name: 'Fit timeline', exact: true })
    .filter({ visible: true })
    .click();
  await keys().first().click();
  const keyRect = await keys().first().boundingBox();
  await page.mouse.click(keyRect.x + keyRect.width / 2, keyRect.y + keyRect.height / 2, {
    button: 'right',
  });
  const menu = page.getByRole('menu', { name: 'Timeline actions', exact: true });
  await menu.waitFor();
  assert.equal(await menu.evaluate((el) => el.parentElement.matches(':popover-open')), true);
  const copy = menu.getByRole('menuitem', { name: /Copy keyframes/ });
  const c = await copy.boundingBox();
  assert.equal(
    await copy.evaluate((el, p) => el.contains(document.elementFromPoint(p.x, p.y)), {
      x: c.x + c.width / 2,
      y: c.y + c.height / 2,
    }),
    true,
    'Menu is in top layer, not clipped by bottom panel',
  );
  await page.screenshot({ path: '../output/electron/verification/timeline-context-dark.png' });
  await page.keyboard.press('Escape');
  assert.equal(await menu.count(), 0);
  await page.setViewportSize({ width: 1120, height: 800 });
  await page.waitForTimeout(150);
  await page.screenshot({ path: '../output/electron/verification/timeline-compact.png' });
  await page.setViewportSize({ width: 1600, height: 1000 });
  await assertNoErrors();
});

test('actual App loop range selection, boundary stepping, numeric range and frame deletion use actual take samples', async () => {
  await time(0);
  await timeline()
    .getByRole('button', { name: /^Record (scene frames|from playhead)$/ })
    .click();
  await page.waitForTimeout(60);
  await x(1);
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="Current time"]').value) >= 0.5,
  );
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  await numeric(await setting('Timeline zoom'), 1000);
  await timeline()
    .locator('.timeline-body')
    .evaluate(
      (el) =>
        (el.scrollLeft =
          (-Number(el.dataset.scrollLower) / Number(el.dataset.scrollSpan)) *
          Number(el.dataset.scrollTravel)),
    );
  await page.waitForTimeout(60);
  await drag(await pointAt(0.1, 'take'), await pointAt(0.3, 'take'), ['Shift']);
  assert.equal(await timeline().getByTestId('timeline-loop-range').count(), 1);
  const start = Number(await (await setting('Loop start')).inputValue());
  const end = Number(
    await timeline().getByRole('textbox', { name: 'Loop end', exact: true }).inputValue(),
  );
  assert.ok(start > 0.05 && start < 0.15 && end > 0.25 && end < 0.35, `${start},${end}`);
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await time(), start);
  assert.equal(
    await timeline().getByRole('button', { name: 'Previous frame', exact: true }).isDisabled(),
    true,
  );
  await timeline().focus();
  await timeline().press('ArrowLeft');
  assert.equal(await time(), start);
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await time(), end);
  assert.equal(
    await timeline().getByRole('button', { name: 'Next frame', exact: true }).isDisabled(),
    true,
  );
  await timeline().focus();
  await timeline().press('ArrowRight');
  assert.equal(await time(), end);
  await numeric(await setting('Loop start'), 0.15);
  const shifted = Number(await (await setting('Loop start')).inputValue());
  assert.ok(shifted >= start && shifted < end);
  await page.screenshot({ path: '../output/electron/verification/timeline-actions-dark.png' });
  const p = await pointAt(0.2, 'take');
  await page.keyboard.down('Shift');
  await page.mouse.click(p.x, p.y, { button: 'right' });
  await page.keyboard.up('Shift');
  assert.equal(await timeline().getByTestId('timeline-loop-range').count(), 0);
  await drag(await pointAt(0.1, 'take'), await pointAt(0.2, 'take'));
  const count = Number(
    (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
      / · (\d+) frames$/,
    )[1],
  );
  await timeline().press('Delete');
  const remaining = Number(
    (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
      / · (\d+) frames$/,
    )[1],
  );
  assert.ok(
    remaining < count && remaining >= 2,
    `Only selected samples removed: ${count} -> ${remaining}`,
  );
  await timeline().press('Delete');
  assert.equal(
    Number(
      (await timeline().locator('.take-label').getAttribute('data-tooltip')).match(
        / · (\d+) frames$/,
      )[1],
    ),
    remaining,
    'Delete without a sample selection never removes the whole take',
  );
  await timeline().locator('.take-label').click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Delete take', exact: true }).click();
  assert.equal(
    await timeline().getByRole('combobox', { name: 'Take', exact: true }).count(),
    0,
    'Deleting a take remains an explicit available action',
  );
  assert.equal(
    await details().count(),
    0,
    'Deleting the final take does not create an empty footer',
  );
  await assertNoErrors();
});

test('actual App Record from playhead restores the nearest saved sample and truncates its future', async () => {
  await time(0);
  await x(0);
  await timeline()
    .getByRole('button', { name: /^Record (scene frames|from playhead)$/ })
    .click();
  await page.waitForTimeout(100);
  await x(2);
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="Current time"]').value) >= 0.8,
  );
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const oldEnd = await time();
  await time(0.08);
  await timeline().getByRole('button', { name: 'Next frame', exact: true }).click();
  const sampleTime = await time(),
    savedPose = await x();
  await x(99);
  await timeline().getByRole('button', { name: 'Record from playhead', exact: true }).click();
  assert.equal(
    await x(),
    savedPose,
    'Resume restores the selected saved pose before recording; it does not overwrite that sample with an unrelated edit',
  );
  await page.waitForTimeout(20);
  await timeline().getByRole('button', { name: 'Stop recording', exact: true }).click();
  const newEnd = await time();
  assert.ok(
    newEnd > sampleTime && newEnd < oldEnd,
    `Resumed tail replaces the previous future: ${sampleTime} -> ${newEnd}`,
  );
  await timeline().getByRole('button', { name: 'Go to end', exact: true }).click();
  assert.equal(await time(), newEnd);
  assert.equal(await x(), savedPose);
  await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
  assert.equal(await time(), 0, 'Resume preserves frames before the selected sample');
  assert.equal(await x(), 0);
  await assertNoErrors();
});

test('timeline exposes all controls directly, preserves numeric transactions and retains a pinned Follow group', async () => {
  await threeKeys();
  assert.equal(await timeline().getByLabel('Timeline end', { exact: true }).count(), 0);
  await numeric(await setting('Playback speed'), 2);
  assert.equal(Number(await (await setting('Playback speed')).inputValue()), 2);
  await numeric(await setting('Timeline zoom'), 150);
  await timeline().getByRole('button', { name: 'Follow locked', exact: true }).click();
  assert.equal(
    await timeline()
      .getByRole('button', { name: 'Follow locked', exact: true })
      .getAttribute('aria-pressed'),
    'true',
  );
  const count = await keys().count();
  await timeline().getByRole('button', { name: 'Follow locked', exact: true }).press('Delete');
  assert.equal(await keys().count(), count, 'Focused controls do not delete timeline keys');
  await timeline()
    .getByRole('button', { name: 'Follow locked', exact: true })
    .press('ControlOrMeta+z');
  assert.equal(await keys().count(), count - 1, 'Focused Follow forwards the global undo command');
  await timeline()
    .getByRole('button', { name: 'Follow locked', exact: true })
    .press('ControlOrMeta+Shift+z');
  assert.equal(await keys().count(), count);
  for (const [label, value, draft] of [
    ['Playback speed', 2, '3.5'],
    ['Timeline zoom', 150, '250'],
  ]) {
    const field = await setting(label);
    await field.dblclick();
    await field.fill(draft);
    await field.press('Escape');
    assert.equal(
      Number(await field.inputValue()),
      value,
      `${label}: Escape cancels the numeric draft`,
    );
  }
  assert.equal(
    await timeline().locator('.timeline-actions-popup,.timeline-actions-trigger').count(),
    0,
  );
  for (const width of [1280, 1000]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const title of [
      'Go to start',
      'Go to end',
      'Loop playback',
      'Previous keyframe',
      'Next keyframe',
      'Capture snapshot',
      'Follow off',
      'Follow page',
      'Follow locked',
    ]) {
      const control = timeline().getByRole('button', { name: title, exact: true });
      assert.equal(await control.isVisible(), true, `${width}: ${title}`);
      assert.equal(
        await control.evaluate(
          (el) => !!el.closest('.timeline-toolbar') && !el.closest('[popover]'),
        ),
        true,
      );
    }
    for (const label of ['View start', 'View end', 'Playback speed', 'Timeline zoom'])
      assert.equal(
        await (await setting(label)).isVisible(),
        true,
        `${width}: ${label} remains reachable`,
      );
    assert.equal(
      await timeline()
        .getByRole('button', { name: 'Fit timeline', exact: true })
        .filter({ visible: true })
        .isVisible(),
      true,
    );
    assert.equal(
      await timeline().getByRole('button', { name: 'Export video', exact: true }).isVisible(),
      true,
    );
    await timeline().getByRole('button', { name: 'Go to start', exact: true }).click();
    assert.equal(await time(), 0);
    assert.equal(
      await timeline()
        .locator('.timeline-toolbar')
        .evaluate((el) => el.scrollWidth <= el.clientWidth),
      true,
    );
  }
  await assertNoErrors();
});

test('actual App timeline keeps every control inline, Follow pinned, complete numeric budgets and only three painted lane labels', async () => {
  const layoutReport = [];
  const themes = JSON.parse(
    await readFile(new URL('../../python/design/resources/themes.json', import.meta.url), 'utf8'),
  );
  for (const language of ['en', 'zh'])
    for (const appearance of ['dark', 'light'])
      for (const uiScale of [1, 1.25])
        for (const panelWidth of [312, 600, 1000, 1600]) {
          const review = await browser.newPage({
            viewport: { width: panelWidth + 900, height: 1100 },
          });
          await review.addInitScript(installPaintStyle);
          await review.evaluate(installPaintStyle);
          const reviewErrors = [];
          review.on('pageerror', (error) => reviewErrors.push(error.message));
          try {
            await review.addInitScript(
              (settings) => {
                localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
                localStorage.setItem(
                  'mojive.layout',
                  JSON.stringify({ left: 280, right: 360, bottom: 320 }),
                );
              },
              { language, appearance, uiScale },
            );
            await review.goto(base, { waitUntil: 'domcontentloaded' });
            const panel = review.getByTestId('timeline');
            await panel.waitFor();
            await review.evaluate(() => document.fonts.ready);
            // Both side panels have proportional maxima; a narrow window changes those
            // maxima too. Measure the real panel after each resize, rather than assuming fixed sidebars.
            for (let attempt = 0; attempt < 8; attempt++) {
              const actual = (await panel.boundingBox()).width;
              if (Math.abs(actual - panelWidth) < 1) break;
              await review.setViewportSize({
                width: Math.round(review.viewportSize().width + panelWidth - actual),
                height: 1100,
              });
              await review.evaluate(
                () =>
                  new Promise((resolve) =>
                    requestAnimationFrame(() => requestAnimationFrame(resolve)),
                  ),
              );
            }
            assert.ok(
              Math.abs((await panel.boundingBox()).width - panelWidth) < 1,
              `Measured timeline panel must reach ${panelWidth}px`,
            );
            const input = panel.locator('.time-readout input');
            await input.dblclick();
            await input.fill('1.055');
            await input.press('Enter');
            // Fit makes the playhead visible even when only a fraction of a second
            // fits at the unchanged default pixel scale in a narrow panel.
            await panel.locator('.timeline-toolbar button:has([data-icon=fit])').click();
            const context = `${language}/${appearance}/${uiScale}/${panelWidth}`;
            const metrics = await panel.evaluate((el) => {
              const rect = (node) => {
                const b = node.getBoundingClientRect();
                return {
                  x: b.x,
                  y: b.y,
                  width: b.width,
                  height: b.height,
                  right: b.right,
                  bottom: b.bottom,
                };
              };
              const toolbar = el.querySelector('.timeline-toolbar'),
                follow = el.querySelector('.timeline-follow');
              return {
                toolbar: rect(toolbar),
                overflow: toolbar.scrollWidth > toolbar.clientWidth,
                groups: [...toolbar.querySelectorAll('.timeline-control-group')]
                  .filter((n) => n.offsetWidth)
                  .map(rect),
                controls: [...toolbar.querySelectorAll('.button,.number-field')]
                  .filter((n) => n.offsetWidth)
                  .map((node) => ({
                    ...rect(node),
                    field: node.matches('.number-field') || !!node.closest('.segmented'),
                  })),
                core: {
                  ...rect(toolbar.querySelector('.timeline-primary-controls')),
                  client: toolbar.querySelector('.timeline-primary-controls').clientWidth,
                  scroll: toolbar.querySelector('.timeline-primary-controls').scrollWidth,
                },
                range: rect(
                  [...toolbar.querySelectorAll('.timeline-range-fields')].find(
                    (n) => n.offsetWidth,
                  ),
                ),
                icons: [...toolbar.querySelectorAll('.button > .icon')]
                  .filter((n) => n.offsetWidth)
                  .map((node) => ({
                    ...rect(node),
                    parent: rect(node.parentElement),
                  })),
                lanes: [...el.querySelectorAll('.key-lane,.take-lane,.snapshot-lane')].map(rect),
                grid: window.paintStyle(el.querySelector('.timeline-grid')).backgroundImage,
                ruler: rect(el.querySelector('.time-ruler')),
                head: {
                  ...rect(el.querySelector('.playhead-head')),
                  color: window.paintStyle(el.querySelector('.playhead-head')).backgroundColor,
                },
                line: {
                  ...rect(el.querySelector('.playhead > i')),
                  opacity: window.paintStyle(el.querySelector('.playhead > i')).opacity,
                },
                follow: {
                  ...rect(follow),
                  padding: window.paintStyle(follow).padding,
                  gap: window.paintStyle(follow).gap,
                },
                cells: [...follow.children].map(rect),
                inputs: [...toolbar.querySelectorAll('.number-field input')]
                  .filter((n) => n.offsetWidth)
                  .map((input) => {
                    const style = window.paintStyle(input),
                      ctx = document.createElement('canvas').getContext('2d');
                    ctx.font = style.font;
                    const budget = input.closest('.timeline-speed')
                      ? '4.00'
                      : input.closest('.timeline-zoom')
                        ? '8000.0'
                        : '3600.000';
                    return {
                      text: ctx.measureText(budget).width,
                      width:
                        parseFloat(style.width) -
                        parseFloat(style.paddingLeft) -
                        parseFloat(style.paddingRight),
                    };
                  }),
              };
            });
            assert.equal(metrics.overflow, false, context);
            const rowYs = metrics.controls.map((g) => g.y + g.height / 2);
            assert.ok(
              Math.max(...rowYs) - Math.min(...rowYs) < 0.1,
              `${context}: every visible toolbar control stays on one line`,
            );
            assert.ok(
              Math.abs(metrics.toolbar.height - 45 * uiScale) <= 0.5,
              `${context}: fixed one-row height includes the quantized border`,
            );
            for (const control of metrics.controls)
              assert.ok(
                Math.abs(control.height - 28 * uiScale) < 0.1,
                `${context}: shared field or command height`,
              );
            assert.ok(
              metrics.follow.x >= metrics.toolbar.x &&
                metrics.follow.right <= metrics.toolbar.right,
              `${context}: Follow remains inside the viewport`,
            );
            if (panelWidth / uiScale >= 480)
              assert.ok(
                metrics.range.x >= metrics.follow.right &&
                  metrics.range.right <= metrics.toolbar.right,
                `${context}: both inline range editors remain inside the viewport`,
              );
            else {
              const fields = panel.locator('.timeline-scrolling-range .number-field');
              for (let i = 0; i < 2; i++) {
                await fields.nth(i).scrollIntoViewIfNeeded();
                const box = await fields.nth(i).boundingBox();
                const core = await panel.locator('.timeline-primary-controls').boundingBox();
                assert.ok(
                  box.x >= core.x - 1 && box.x + box.width <= core.x + core.width + 1,
                  `${context}: endpoint ${i} remains fully reachable within the scrolling core`,
                );
                assert.equal(
                  (await panel.locator('.timeline-follow').boundingBox()).x,
                  metrics.follow.x,
                );
              }
              await panel.screenshot({
                path: `../output/electron/verification/timeline-inline-range-visible-${language}-${appearance}-${uiScale}-${panelWidth}.png`,
              });
            }
            assert.equal(
              await panel
                .locator('.timeline-toolbar .timeline-range-fields input')
                .filter({ visible: true })
                .count(),
              2,
            );
            assert.equal(
              await panel.locator('.timeline-range-fields button').count(),
              0,
              'Range is not a choice/dropdown',
            );
            assert.ok(
              metrics.core.width >= (await panel.locator('.time-readout').boundingBox()).width,
              `${context}: the complete current-time format can be browsed in the core`,
            );
            if (metrics.toolbar.width / uiScale >= 1250)
              assert.equal(
                metrics.core.scroll,
                metrics.core.client,
                `${context}: complete inline toolbar fits within 1250 logical pixels`,
              );
            const followBefore = metrics.follow.x;
            const scrolled = await panel.locator('.timeline-primary-controls').evaluate((el) => {
              el.scrollLeft = el.scrollWidth;
              const nav = el.querySelector('.timeline-key-navigation').getBoundingClientRect();
              const box = el.getBoundingClientRect();
              return { navRight: nav.right, right: box.right };
            });
            assert.ok(
              scrolled.navRight <= scrolled.right + 1,
              `${context}: high-frequency tail is reachable even in extreme widths`,
            );
            assert.equal((await panel.locator('.timeline-follow').boundingBox()).x, followBefore);
            await panel.locator('.timeline-primary-controls').evaluate((el) => {
              el.scrollLeft = 0;
            });
            layoutReport.push({
              language,
              appearance,
              uiScale,
              panelWidth,
              toolbar: metrics.toolbar,
              rows: 1,
              overflow: metrics.overflow,
              core: metrics.core,
              follow: metrics.follow,
              range: metrics.range,
            });
            for (const icon of metrics.icons) {
              assert.ok(Math.abs(icon.height - 20 * uiScale) < 0.1, `${context}: icon height`);
              assert.ok(
                Math.abs(icon.y + icon.height / 2 - icon.parent.y - icon.parent.height / 2) < 0.1,
                `${context}: icon center`,
              );
            }
            assert.deepEqual(
              metrics.lanes.map((lane) => lane.height),
              [44, 44, 44].map((h) => h * uiScale),
              context,
            );
            assert.equal((metrics.grid.match(/linear-gradient/g) || []).length, 1);
            const rgb = themes[appearance].danger
              .match(/[0-9a-f]{2}/gi)
              .map((hex) => parseInt(hex, 16));
            assert.equal(metrics.head.color, `rgb(${rgb.join(', ')})`, context);
            assert.equal(metrics.head.width, 8 * uiScale, context);
            assert.equal(metrics.head.height, 9 * uiScale, context);
            assert.equal(metrics.head.y, metrics.ruler.y, context);
            assert.equal(metrics.line.opacity, '0.85', context);
            assert.equal(metrics.line.width, uiScale, context);
            assert.equal(metrics.follow.padding, '0px');
            assert.equal(metrics.follow.gap, '0px');
            assert.equal(metrics.cells[0].x, metrics.follow.x);
            assert.equal(metrics.cells.at(-1).right, metrics.follow.right);
            for (const cell of metrics.cells) {
              assert.equal(cell.y, metrics.follow.y);
              assert.equal(cell.bottom, metrics.follow.bottom);
            }
            for (const field of metrics.inputs)
              assert.ok(
                field.width >= field.text,
                `${context}: numeric budget ${JSON.stringify(field)}`,
              );
            const sampleHead = async () => {
              const png = await panel.locator('.playhead-head').screenshot();
              return review.evaluate(
                async ({ png, scale }) => {
                  const image = new Image();
                  image.src = 'data:image/png;base64,' + png;
                  await image.decode();
                  const canvas = document.createElement('canvas');
                  canvas.width = image.width;
                  canvas.height = image.height;
                  const ctx = canvas.getContext('2d');
                  ctx.drawImage(image, 0, 0);
                  return [
                    ...ctx.getImageData(Math.floor(image.width / 2), Math.round(2 * scale), 1, 1)
                      .data,
                  ];
                },
                { png: png.toString('base64'), scale: uiScale },
              );
            };
            assert.deepEqual(await sampleHead(), [...rgb, 255], context);
            await panel.screenshot({
              path: `../output/electron/verification/timeline-inline-${language}-${appearance}-${uiScale}-${panelWidth}.png`,
            });
            assert.equal(
              await panel.locator('.timeline-actions-trigger,.timeline-actions-popup').count(),
              0,
            );
            assert.equal(
              await panel.locator('.timeline-details').count(),
              0,
              'No empty metadata footer',
            );
            const labels = await panel.locator('.lane-labels').boundingBox();
            const lastLabel = await panel.locator('.lane-label').last().boundingBox();
            assert.equal(labels.height, 160 * uiScale, `${context}: labels have content height`);
            assert.equal(
              labels.y + labels.height,
              lastLabel.y + lastLabel.height,
              `${context}: no painted label strip below the third lane`,
            );
            for (const selector of [
              '.timeline-speed .number-field',
              '.timeline-zoom .number-field',
              '.timeline-video-button',
            ]) {
              const control = panel.locator(selector);
              await control.scrollIntoViewIfNeeded();
              const box = await control.boundingBox();
              const core = await panel.locator('.timeline-primary-controls').boundingBox();
              assert.ok(
                box.x >= core.x - 1 && box.x + box.width <= core.x + core.width + 1,
                `${context}: ${selector} reachable intact without a popup`,
              );
              assert.equal(
                (await panel.locator('.timeline-follow').boundingBox()).x,
                metrics.follow.x,
              );
            }
            await panel.screenshot({
              path: `../output/electron/verification/timeline-inline-controls-${language}-${appearance}-${uiScale}-${panelWidth}.png`,
            });
            const lastTrack = await panel.locator('.timeline-body').evaluate((el) => {
              el.scrollTop = el.scrollHeight;
              const viewport = el.getBoundingClientRect(),
                last = el.querySelector('.snapshot-lane').getBoundingClientRect();
              return {
                top: last.top,
                bottom: last.bottom,
                viewportTop: viewport.top,
                viewportBottom: viewport.bottom,
                height: viewport.height,
              };
            });
            assert.ok(
              lastTrack.height >= 44 * uiScale &&
                lastTrack.top >= lastTrack.viewportTop &&
                // CSSOM scrollTop rounds at fractional zoom; allow one physical pixel at the final border.
                lastTrack.bottom <= lastTrack.viewportBottom + 1,
              `${context}: last track remains fully reachable by scrolling ${JSON.stringify(lastTrack)}`,
            );
            if (panelWidth === 600 && uiScale === 1.25)
              await panel.screenshot({
                path: `../output/electron/verification/timeline-inline-bottom-${language}-${appearance}.png`,
              });
            await panel.locator('.timeline-body').evaluate((el) => {
              el.scrollTop = 0;
            });
            if (uiScale === 1.25) {
              const splitter = review.locator('.splitter-y');
              // Reduce available track height just enough to exercise the sticky ruler.
              for (let i = 0; i < 10; i++) await splitter.press('ArrowDown');
              await panel.locator('.timeline-body').evaluate((el) => {
                el.scrollTop = 30;
              });
              assert.ok((await panel.locator('.timeline-body').evaluate((el) => el.scrollTop)) > 0);
              assert.deepEqual(await sampleHead(), [...rgb, 255], `${context}: sticky playhead`);
            }
            assert.deepEqual(reviewErrors, [], context);
          } finally {
            await review.close();
          }
        }
  await writeFile(
    '../output/electron/verification/timeline-inline-layout.json',
    JSON.stringify(layoutReport, null, 2),
  );
});
