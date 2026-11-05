import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page, base;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-notice-history-test',
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
  page.setDefaultTimeout(7000);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    );
    localStorage.setItem('mojive.layout', JSON.stringify({ left: 280, right: 360, bottom: 300 }));
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
  await page.getByRole('treeitem').first().waitFor();
  await page.getByTestId('viewport-canvas').waitFor();
});
const notice = () => page.locator('.operation-notice');
const action = (label) => notice().getByRole('button', { name: label, exact: true });
const titles = () =>
  page
    .getByRole('treeitem')
    .evaluateAll((nodes) => nodes.map((node) => node.dataset.tooltip).sort());
async function add(kind = 'Box') {
  await page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'Add', exact: true })
    .click();
  await page.getByRole('menuitem', { name: kind, exact: true }).click();
  await action('Undo').waitFor();
  return page.locator('.inspector-heading strong').textContent();
}
async function numeric(locator, value) {
  await locator.dblclick();
  await locator.fill(String(value));
  await locator.press('Enter');
}
async function rememberAction(key) {
  // React's DOM event props are read only in this test; production exposes no test/history hook.
  await page.locator('.operation-notice-action').evaluate((node, key) => {
    const propsKey = Object.keys(node).find((name) => name.startsWith('__reactProps$'));
    if (!propsKey || typeof node[propsKey]?.onClick !== 'function')
      throw Error('Notice action handler is unavailable');
    window.__noticeCallbacks ||= {};
    window.__noticeCallbacks[key] = node[propsKey].onClick;
  }, key);
}
async function invokeRemembered(key) {
  await page.evaluate((key) => window.__noticeCallbacks[key](), key);
  await page.evaluate(() => new Promise(requestAnimationFrame));
}
async function chooseScene(name = 'Scene objects') {
  await page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'File', exact: true })
    .click();
  await page.getByRole('menuitem', { name, exact: true }).click();
  await page.getByRole('alertdialog').waitFor();
}

test('committed operation notice undoes exactly that transaction and its replacement safely redoes it', async () => {
  const before = await titles(),
    name = await add();
  assert.ok((await titles()).includes(name));
  assert.match(await notice().textContent(), /Added box/);
  await action('Undo').click();
  await action('Redo').waitFor();
  assert.deepEqual(await titles(), before);
  assert.match(await notice().textContent(), /Edit undone/);
  await action('Redo').click();
  await action('Undo').waitFor();
  assert.ok((await titles()).includes(name));
  assert.match(await notice().textContent(), /Edit restored/);
  assert.deepEqual(errors, []);
});

test('starting then cancelling a numeric edit invalidates the earlier notice even when its snapshot is still the latest', async () => {
  const name = await add();
  await rememberAction('before-edit');
  const x = page.getByLabel('Position X', { exact: true }),
    original = Number(await x.inputValue());
  await x.dblclick();
  await page.locator('.operation-notice-action').waitFor({ state: 'detached' });
  await x.fill('3');
  await invokeRemembered('before-edit');
  assert.ok((await titles()).includes(name), 'Pending edits reject the old notice callback');
  await x.press('Escape');
  assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), original);
  assert.equal(
    await page.locator('.operation-notice-action').count(),
    0,
    'Cancel never revives an old action',
  );
  await invokeRemembered('before-edit');
  assert.ok(
    (await titles()).includes(name),
    'The old action stays invalid after cancel restores the same history head',
  );
  assert.deepEqual(errors, []);
});

test('a retained old callback cannot undo a newer operation or revive when its old entry becomes top again', async () => {
  const box = await add('Box');
  await rememberAction('box');
  const sphere = await add('Sphere'),
    both = await titles();
  await invokeRemembered('box');
  assert.deepEqual(await titles(), both, 'Old action cannot pop the Sphere transaction');
  await action('Undo').click();
  await action('Redo').waitFor();
  assert.ok((await titles()).includes(box));
  assert.ok(!(await titles()).includes(sphere));
  await invokeRemembered('box');
  assert.ok(
    (await titles()).includes(box),
    'Epoch/publication guard rejects old Box action even though its Snapshot is top again',
  );
  await action('Redo').click();
  await action('Undo').waitFor();
  assert.deepEqual(await titles(), both);
  assert.deepEqual(errors, []);
});

test('loading invalidates actions before confirmation and document replacement clears all prior notice identities', async () => {
  const box = await add();
  await rememberAction('old-document');
  await chooseScene();
  await page.locator('.operation-notice-action').waitFor({ state: 'detached' });
  await invokeRemembered('old-document');
  assert.ok(
    (await titles()).includes(box),
    'Busy/confirmation cannot execute the old history action',
  );
  await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.equal(await page.locator('.operation-notice-action').count(), 0);
  await invokeRemembered('old-document');
  assert.ok(
    (await titles()).includes(box),
    'Cancelling replacement does not revive the earlier callback',
  );
  await chooseScene();
  await page
    .getByRole('alertdialog')
    .getByRole('button', { name: "Don't Save", exact: true })
    .click();
  await page.getByRole('alertdialog').waitFor({ state: 'detached' });
  await notice().waitFor({ state: 'detached' });
  const replacement = await titles();
  assert.ok(!replacement.includes(box));
  await invokeRemembered('old-document');
  assert.deepEqual(
    await titles(),
    replacement,
    'A stale callback cannot restore the discarded document',
  );
  assert.deepEqual(errors, []);
});

test('seeking, recording and playback permanently invalidate previously published history actions', async () => {
  const box = await add(),
    timeline = page.getByTestId('timeline');
  await rememberAction('before-seek');
  await numeric(timeline.locator('.time-readout input'), 1);
  assert.equal(await page.locator('.operation-notice-action').count(), 0);
  await invokeRemembered('before-seek');
  assert.ok((await titles()).includes(box));
  await timeline
    .getByRole('button', { name: 'Capture scene pose at playhead (K)', exact: true })
    .click();
  await rememberAction('before-record');
  const keyCount = await timeline.locator('.keyframe').count();
  await timeline.getByRole('button', { name: 'Record scene frames', exact: true }).click();
  await page.locator('.operation-notice-action').waitFor({ state: 'detached' });
  await invokeRemembered('before-record');
  assert.equal(await timeline.locator('.keyframe').count(), keyCount);
  await page.waitForTimeout(100);
  await timeline.getByRole('button', { name: 'Stop recording', exact: true }).click();
  assert.equal(await page.locator('.operation-notice-action').count(), 0);
  await timeline.getByRole('button', { name: 'Go to start', exact: true }).click();
  await timeline
    .getByRole('button', { name: 'Capture scene pose at playhead (K)', exact: true })
    .click();
  await rememberAction('before-play');
  const savedKeys = await timeline.locator('.keyframe').count();
  await timeline.getByRole('button', { name: 'Play (Space)', exact: true }).click();
  await page.locator('.operation-notice-action').waitFor({ state: 'detached' });
  await invokeRemembered('before-play');
  assert.equal(await timeline.locator('.keyframe').count(), savedKeys);
  await timeline.getByRole('button', { name: 'Pause (Space)', exact: true }).click();
  await invokeRemembered('before-play');
  assert.equal(await timeline.locator('.keyframe').count(), savedKeys);
  assert.equal(await page.locator('.operation-notice-action').count(), 0);
  assert.deepEqual(errors, []);
});

test('actual Chinese operation messages and history actions appear at the viewport lower-left in both themes', async () => {
  for (const appearance of ['dark', 'light']) {
    const review = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    await review.addInitScript(installPaintStyle);
    await review.evaluate(installPaintStyle);
    try {
      await review.addInitScript((appearance) => {
        localStorage.setItem(
          'mojive.electron.settings.v1',
          JSON.stringify({ language: 'zh', appearance, uiScale: 1.25 }),
        );
        localStorage.setItem(
          'mojive.layout',
          JSON.stringify({ left: 280, right: 360, bottom: 300 }),
        );
      }, appearance);
      await review.goto(base, { waitUntil: 'domcontentloaded' });
      await review.getByRole('treeitem').first().waitFor();
      await review.evaluate(() => document.fonts.ready);
      const labels = await review.evaluate(async () => {
        const { t } = await import('/src/i18n.ts');
        return { add: t('Add'), box: t('Box'), undo: t('Undo'), redo: t('Redo') };
      });
      await review.locator('nav').getByRole('button', { name: labels.add, exact: true }).click();
      await review.getByRole('menuitem', { name: labels.box, exact: true }).click();
      const notification = review.locator('.operation-notice');
      await notification.getByRole('button', { name: labels.undo, exact: true }).waitFor();
      assert.match(
        await notification.locator('.operation-notice-message').textContent(),
        /^已添加/,
      );
      await notification.getByRole('button', { name: labels.undo, exact: true }).click();
      await notification.getByRole('button', { name: labels.redo, exact: true }).waitFor();
      assert.equal(
        await notification.locator('.operation-notice-message').textContent(),
        '已撤销操作',
      );
      const metrics = await notification.evaluate((el) => {
        const r = el.getBoundingClientRect(),
          p = el.closest('.viewport-content').getBoundingClientRect();
        return {
          left: r.left - p.left,
          bottom: p.bottom - r.bottom,
          right: p.right - r.right,
          height: r.height,
          font: window.paintStyle(el).fontFamily,
        };
      });
      assert.ok(Math.abs(metrics.left - 15) < 0.1);
      assert.ok(Math.abs(metrics.bottom - 15) < 0.1);
      assert.ok(
        metrics.right > 15,
        'Notice is anchored to the left instead of centered or spanning the viewport',
      );
      assert.equal(metrics.height, 55);
      assert.match(metrics.font, /MiSans/);
      await review.locator('.viewport-content').screenshot({
        path: `../output/electron/verification/notice-history-${appearance}-zh-125.png`,
      });
    } finally {
      await review.close();
    }
  }
});
