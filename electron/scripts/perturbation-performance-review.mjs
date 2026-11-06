import { createServer } from 'vite';
import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const first = path.resolve(option('--first', path.join(root, '../assets/joint_gizmo.xml')));
const model = path.resolve(option('--model', first));
const body = option('--body', '03_ball');
const out = path.resolve(
  option('--output', path.join(root, '../output/electron/perturbation-performance')),
);
const moves = Number(option('--moves', '120'));
const repeats = Number(option('--repeat', '2'));
await mkdir(out, { recursive: true });

// Install observation around production classes before the real App mounts. In particular,
// the gesture includes App undo/history callbacks, not only renderer and worker timings.
function observe(THREE, PhysicsDocument, ViewportPipeline, ViewportMetrics, PerturbationGesture) {
  localStorage.setItem(
    'mojive.electron.settings.v1',
    JSON.stringify({ language: 'en', appearance: 'dark' }),
  );
  const state = (window.__performanceReview = {
    runtime: null,
    camera: null,
    active: false,
    data: null,
    restores: 0,
  });
  const fresh = () => ({
    frame: [],
    raf: [],
    commands: [],
    serialize: [],
    clone: [],
    compare: [],
    begin: [],
    pendingMax: 0,
    begins: 0,
  });
  state.start = () => {
    state.data = fresh();
    state.active = true;
  };
  state.stop = () => {
    state.active = false;
    return state.data;
  };
  const load = PhysicsDocument.load;
  PhysicsDocument.load = async function (...args) {
    const runtime = await load.apply(this, args);
    state.runtime = runtime;
    const execute = runtime.session.execute;
    let pending = 0;
    runtime.session.execute = async function (command) {
      const data = state.active ? state.data : null;
      const start = performance.now();
      pending++;
      if (data) data.pendingMax = Math.max(data.pendingMax, pending);
      try {
        const result = await execute.call(this, command);
        if (command.type === 'restore') state.restores++;
        return result;
      } finally {
        pending--;
        if (data) data.commands.push({ type: command.type, ms: performance.now() - start });
      }
    };
    return runtime;
  };
  const render = ViewportPipeline.prototype.render;
  ViewportPipeline.prototype.render = function (scene, camera, ...args) {
    state.camera = camera;
    return render.call(this, scene, camera, ...args);
  };
  const record = ViewportMetrics.prototype.record;
  ViewportMetrics.prototype.record = function (start, end, workload) {
    if (state.active) state.data.frame.push({ ms: end - start, ...workload });
    return record.call(this, start, end, workload);
  };
  const begin = PerturbationGesture.prototype.begin;
  PerturbationGesture.prototype.begin = function (...args) {
    const start = performance.now(),
      result = begin.apply(this, args);
    if (state.active) {
      state.data.begin.push(performance.now() - start);
      if (result) state.data.begins++;
    }
    return result;
  };
  const serialize = THREE.Object3D.prototype.toJSON;
  THREE.Object3D.prototype.toJSON = function (...args) {
    const watched = state.active && this === state.runtime?.document.root,
      start = performance.now();
    try {
      return serialize.apply(this, args);
    } finally {
      if (watched) state.data.serialize.push(performance.now() - start);
    }
  };
  const clone = window.structuredClone;
  window.structuredClone = function (value, ...args) {
    const watched = state.active && value?.root && value?.nodes && value?.files,
      start = performance.now();
    try {
      return clone.call(this, value, ...args);
    } finally {
      if (watched) state.data.clone.push(performance.now() - start);
    }
  };
  const stringify = JSON.stringify;
  JSON.stringify = function (value, ...args) {
    const watched = state.active && value?.document && value?.timeline,
      start = performance.now();
    try {
      return stringify.call(this, value, ...args);
    } finally {
      if (watched) state.data.compare.push(performance.now() - start);
    }
  };
  let previous = performance.now();
  function frame(now) {
    if (state.active) state.data.raf.push(now - previous);
    previous = now;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  state.point = (name) => {
    const node = state.runtime.document.nodes.find((node) => node.name === name);
    if (!node) throw Error('Missing body ' + name);
    node.object.updateWorldMatrix(true, true);
    const point = new THREE.Box3()
      .setFromObject(node.object)
      .getCenter(new THREE.Vector3())
      .project(state.camera);
    const rect = document.querySelector('.viewport-stage canvas').getBoundingClientRect();
    return {
      x: rect.left + ((point.x + 1) * rect.width) / 2,
      y: rect.top + ((1 - point.y) * rect.height) / 2,
    };
  };
}

const bootstrap = `import * as THREE from 'three';
import { PhysicsDocument } from '/src/physics/PhysicsDocument.ts';
import { ViewportPipeline } from '/src/viewport/pipeline.ts';
import { ViewportMetrics } from '/src/viewport/metrics.ts';
import { PerturbationGesture } from '/src/viewport/perturbation.ts';
(${observe.toString()})(THREE, PhysicsDocument, ViewportPipeline, ViewportMetrics, PerturbationGesture);
await import('/src/main.tsx');`;
const server = await createServer({
  root,
  configFile: false,
  plugins: [
    {
      name: 'production-performance-observation',
      transformIndexHtml: (html) => html.replace('/src/main.tsx', '/__performance_bootstrap.js'),
      resolveId: (id) => (id === '/__performance_bootstrap.js' ? '\0performance-bootstrap' : null),
      load: (id) => (id === '\0performance-bootstrap' ? bootstrap : null),
    },
  ],
  server: { host: '127.0.0.1', port: 0, hmr: false, watch: null },
});
await server.listen();
let app;
const errors = [],
  results = [],
  checks = [];
const distribution = (values) => {
  const sorted = values.toSorted((a, b) => a - b);
  return {
    count: values.length,
    median: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
    p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
    max: sorted.at(-1) ?? 0,
  };
};
try {
  app = await electron.launch({
    args: [root, '--hidden', '--user-data-dir=' + (await mkdtemp(path.join(out, 'profile-')))],
    cwd: root,
    env: { ...process.env, MOJIVE_DEV_URL: server.resolvedUrls.local[0] },
    timeout: 45000,
  });
  const page = await app.firstWindow();
  page.on('pageerror', (error) => errors.push(error.stack || error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  page.setDefaultTimeout(30000);
  await page.waitForSelector('.simulation-toolbar');
  await page.evaluate(() => document.fonts.ready);
  const action = (id) =>
    app.evaluate(
      ({ BrowserWindow }, id) =>
        BrowserWindow.getAllWindows()[0].webContents.send('mojive:menu-action', id),
      id,
    );
  const capture = () => page.evaluate(() => window.__performanceReview.runtime.session.capture());
  async function restoreHistory(direction, expected, preserveDocument = true) {
    const previous = await page.evaluate(() => window.__performanceReview.restores);
    await action(`edit.${direction}`);
    await page.waitForFunction(
      (previous) => window.__performanceReview.restores > previous,
      previous,
    );
    assert.deepEqual(
      await capture(),
      expected,
      `${direction} restores the complete simulation state`,
    );
    if (preserveDocument)
      assert.equal(
        await page.evaluate(
          () => window.__performanceReview.document === window.__performanceReview.runtime.document,
        ),
        true,
        `${direction} retains the existing scene document and GPU resources`,
      );
  }
  async function open(file) {
    await app.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows()[0].webContents.send(
        'mojive:menu-action',
        'file.openSimulation',
      ),
    );
    const dialog = page.locator('.file-dialog');
    await dialog.waitFor();
    await dialog.getByRole('textbox', { name: 'Location', exact: true }).fill(path.dirname(file));
    await dialog.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    await dialog
      .locator('.file-dialog-entry')
      .filter({
        has: page.locator('span[data-tooltip]').filter({
          hasText: new RegExp(
            '^' + path.basename(file).replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$',
          ),
        }),
      })
      .dblclick();
    await dialog.waitFor({ state: 'detached', timeout: 120000 });
    const discard = page.getByRole('button', { name: "Don't Save", exact: true });
    if (await discard.isVisible()) await discard.click();
    await page.waitForFunction(
      (name) => window.__performanceReview.runtime?.document.name === name,
      path.basename(file),
      { timeout: 120000 },
    );
    await page.waitForTimeout(500);
    console.log('Loaded', file);
  }
  await open(first);
  await open(model);
  await page.locator('.panel-header').getByRole('button', { name: 'Search', exact: true }).click();
  await page.locator('.panel-search input').fill(body);
  await page.getByRole('treeitem', { name: body, exact: true }).click();
  for (let cycle = 0; cycle < repeats; cycle++)
    for (const playing of [false, true])
      for (const mode of ['Visual', 'Collision'])
        for (const button of ['left', 'right']) {
          await page.evaluate(async () => {
            const session = window.__performanceReview.runtime.session;
            session.pause();
            await session.execute({ type: 'reset' });
          });
          await page
            .locator('.geometry-selector')
            .getByRole('button', { name: mode, exact: true })
            .click();
          await page.waitForTimeout(250);
          const before = await capture();
          await page.evaluate(() => {
            window.__performanceReview.document = window.__performanceReview.runtime.document;
          });
          const point = await page.evaluate((body) => window.__performanceReview.point(body), body);
          await page.mouse.move(point.x, point.y);
          await page.keyboard.down('Control');
          await page.evaluate(() => window.__performanceReview.start());
          const start = performance.now();
          await page.mouse.down({ button });
          const downMs = performance.now() - start;
          if (playing) await page.evaluate(() => window.__performanceReview.runtime.session.play());
          for (let i = 0; i < moves; i++) {
            await page.mouse.move(point.x + Math.sin(i / 15) * 32, point.y + Math.cos(i / 18) * 20);
            await page.waitForTimeout(8);
          }
          const release = performance.now();
          await page.mouse.up({ button });
          const upMs = performance.now() - release;
          await page.keyboard.up('Control');
          await page.evaluate(async () => {
            const session = window.__performanceReview.runtime.session;
            session.pause();
            await session.capture();
          });
          const drainMs = performance.now() - release;
          await page.waitForTimeout(250);
          const samples = await page.evaluate(() => window.__performanceReview.stop());
          assert.equal(samples.begins, 1, `Gesture must activate: ${mode} ${button}`);
          assert.deepEqual(
            [samples.serialize.length, samples.clone.length, samples.compare.length],
            [0, 0, 0],
            'Physical gestures must not serialize or compare the authored document',
          );
          const result = {
            cycle,
            playing,
            mode,
            button,
            downMs,
            upMs,
            drainMs,
            begin: distribution(samples.begin),
            frame: distribution(samples.frame.map((frame) => frame.ms)),
            raf: distribution(samples.raf),
            serialize: distribution(samples.serialize),
            clone: distribution(samples.clone),
            compare: distribution(samples.compare),
            command: distribution(samples.commands.map((command) => command.ms)),
            pendingMax: samples.pendingMax,
            workload: samples.frame.at(-1),
            samples,
          };
          results.push(result);
          await writeFile(
            path.join(out, 'results.json'),
            JSON.stringify({ first, model, body, errors, results, checks }, null, 2),
          );
          console.log(JSON.stringify({ ...result, samples: undefined }));
          if (!cycle && !playing && button === 'left')
            await page.screenshot({ path: path.join(out, mode.toLowerCase() + '.png') });
          const after = await capture();
          if (playing) {
            assert.notDeepEqual(after, before, `${mode} ${button} changes the simulation state`);
            if (!cycle && mode === 'Collision' && button === 'right') {
              await page.evaluate(() => {
                const state = window.__performanceReview;
                const session = state.runtime.session;
                state.originalExecute = session.execute;
                state.rejectedRestore = false;
                session.execute = async function (command) {
                  if (command.type === 'restore' && !state.rejectedRestore) {
                    state.rejectedRestore = true;
                    throw Error('Expected performance review restore rejection');
                  }
                  return state.originalExecute.call(this, command);
                };
              });
              await action('edit.undo');
              await page.waitForFunction(() => window.__performanceReview.rejectedRestore);
              await page.waitForTimeout(100);
              assert.deepEqual(await capture(), after, 'A rejected Undo leaves physics unchanged');
              assert.equal(
                await page.evaluate(
                  () =>
                    window.__performanceReview.document ===
                    window.__performanceReview.runtime.document,
                ),
                true,
              );
              await page.evaluate(() => {
                const state = window.__performanceReview;
                state.runtime.session.execute = state.originalExecute;
              });
              checks.push({ rejectedRestore: true, unchangedState: true });
            }
            await restoreHistory('undo', before);
            await restoreHistory('redo', after);
            checks.push({
              cycle,
              playing,
              mode,
              button,
              undo: true,
              redo: true,
              retainedDocument: true,
            });
          } else {
            assert.deepEqual(
              after,
              before,
              'Paused perturbation clears its transient force without advancing physics',
            );
            checks.push({ cycle, playing, mode, button, unchangedState: true });
          }
        }
  console.log('Checking cancellation');
  await page.evaluate(async () => {
    const session = window.__performanceReview.runtime.session;
    session.pause();
    await session.execute({ type: 'reset' });
    window.__performanceReview.document = window.__performanceReview.runtime.document;
  });
  const cancelBefore = await capture();
  const cancelPoint = await page.evaluate((body) => window.__performanceReview.point(body), body);
  await page.mouse.move(cancelPoint.x, cancelPoint.y);
  await page.keyboard.down('Control');
  await page.mouse.down();
  await page.evaluate(() => window.__performanceReview.runtime.session.play());
  for (let i = 0; i < 12; i++) {
    await page.mouse.move(cancelPoint.x + i * 2, cancelPoint.y + i);
    await page.waitForTimeout(12);
  }
  await page.evaluate(async () => {
    const session = window.__performanceReview.runtime.session;
    session.pause();
    await session.capture();
  });
  const previousRestores = await page.evaluate(() => window.__performanceReview.restores);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await page.keyboard.up('Control');
  await page.waitForFunction(
    (previous) => window.__performanceReview.restores > previous,
    previousRestores,
  );
  assert.deepEqual(await capture(), cancelBefore, 'Escape restores the entire pre-gesture state');
  assert.equal(
    await page.evaluate(
      () => window.__performanceReview.document === window.__performanceReview.runtime.document,
    ),
    true,
  );
  checks.push({ cancellation: true, unchangedState: true, retainedDocument: true });
  await page.locator('.bottom-tabs').getByRole('button', { name: 'Timeline', exact: true }).click();
  console.log('Checking recording Undo/Redo');
  // Recording owns one document transaction, including any physical gestures during the take.
  await page.evaluate(async () => {
    const session = window.__performanceReview.runtime.session;
    session.pause();
    await session.execute({ type: 'reset' });
  });
  const recordingBefore = await capture();
  await page.getByRole('button', { name: 'Record scene frames', exact: true }).click();
  await page.getByRole('button', { name: 'Stop recording', exact: true }).waitFor();
  await page.evaluate(() => window.__performanceReview.runtime.session.pause());
  const point = await page.evaluate((body) => window.__performanceReview.point(body), body);
  await page.mouse.move(point.x, point.y);
  await page.keyboard.down('Control');
  await page.mouse.down();
  await page.evaluate(() => window.__performanceReview.runtime.session.play());
  for (let i = 0; i < 30; i++) {
    await page.mouse.move(point.x + i, point.y + i / 2);
    await page.waitForTimeout(12);
  }
  await page.mouse.up();
  await page.keyboard.up('Control');
  await page.getByRole('button', { name: 'Stop recording', exact: true }).click();
  await page.getByRole('button', { name: 'Record from playhead', exact: true }).waitFor();
  await page.waitForTimeout(250);
  const recordingAfter = await capture();
  assert.equal(await page.locator('.take-strip').count(), 1);
  const recordedTicks = await page.locator('.take-frame-tick').count();
  assert.ok(recordedTicks > 1, 'The recording contains multiple simulation frames');
  await page.screenshot({ path: path.join(out, 'recording.png') });
  await restoreHistory('undo', recordingBefore, false);
  assert.equal(
    await page.locator('.take-strip').count(),
    0,
    'One Undo removes the whole recording, including the gesture',
  );
  await restoreHistory('redo', recordingAfter, false);
  assert.equal(await page.locator('.take-strip').count(), 1);
  assert.equal(await page.locator('.take-frame-tick').count(), recordedTicks);
  checks.push({ recording: true, undo: true, redo: true, recordedTicks });
  await writeFile(
    path.join(out, 'results.json'),
    JSON.stringify({ first, model, body, errors, results, checks }, null, 2),
  );
} finally {
  if (app) {
    // Test-owned windows must bypass the application's unsaved-change close confirmation.
    await app
      .evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
      )
      .catch(() => {});
    await app.close();
  }
  await server.close();
}
assert.deepEqual(errors, [], 'Renderer errors');
