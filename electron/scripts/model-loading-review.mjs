import assert from 'node:assert/strict';
import { mkdir, mkdtemp, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { _electron as electron } from 'playwright';

const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const root = await realpath(
  path.resolve(option('--root', path.join(path.dirname(fileURLToPath(import.meta.url)), '..'))),
);
const first = path.resolve(option('--first', path.join(root, '../assets/joint_gizmo.xml')));
const model = path.resolve(option('--model', first));
const output = path.resolve(
  option('--output', path.join(root, '../output/electron/model-loading')),
);
const baseline = args.includes('--baseline');
const openAction = option('--open-action', 'file.openSimulation');
const cancelStage = option('--cancel-stage', 'resources');
const captureStage = option('--capture-stage', null);
const timeout = Number(option('--timeout', '180000'));
await mkdir(output, { recursive: true });

// Observe the real App's file dialog, resource loader, worker, binding and first render.
// The wrappers preserve arguments/results and do not substitute a test loading path.
function observe(PhysicsDocument, WorkerBackend, SceneBinding, ViewportPipeline) {
  localStorage.setItem(
    'mojive.electron.settings.v1',
    JSON.stringify({ language: 'en', appearance: 'dark' }),
  );
  const documentIds = new WeakMap();
  let nextId = 0;
  const identify = (document) => {
    if (!documentIds.has(document)) documentIds.set(document, ++nextId);
    return documentIds.get(document);
  };
  const state = (window.__modelLoadingReview = {
    runtime: null,
    document: null,
    measurement: null,
    cancelAtStage: null,
    previousRaf: performance.now(),
  });
  const mark = (stage, detail = {}) => {
    if (state.measurement)
      state.measurement.events.push({
        stage,
        at: performance.now() - state.measurement.startedAt,
        ...detail,
      });
  };
  state.start = (label, cancelAtStage = null) => {
    state.cancelAtStage = cancelAtStage;
    state.measurement = {
      label,
      startedAt: performance.now(),
      events: [],
      raf: [],
      longTasks: [],
      renders: [],
      progress: [],
      originalDocument: state.document ? identify(state.document) : null,
    };
    mark('request');
  };
  state.identity = () => (state.document ? identify(state.document) : null);
  state.finish = () => {
    const result = {
      ...state.measurement,
      elapsed: performance.now() - state.measurement.startedAt,
      finalDocument: state.identity(),
      finalName: state.document?.name,
    };
    state.measurement = null;
    return result;
  };
  const load = PhysicsDocument.load;
  PhysicsDocument.load = async function (...args) {
    const measurement = state.measurement;
    mark('physics-document-start', {
      files: args[0].files.length,
      bytes: args[0].files.reduce((sum, file) => sum + file.bytes.byteLength, 0),
    });
    try {
      const runtime = await load.apply(this, args);
      state.runtime = runtime;
      if (measurement === state.measurement)
        mark('physics-document-ready', { document: identify(runtime.document) });
      return runtime;
    } catch (error) {
      if (measurement === state.measurement)
        mark('physics-document-rejected', { error: String(error) });
      throw error;
    }
  };
  const workerLoad = WorkerBackend.load;
  WorkerBackend.load = async function (...args) {
    const measurement = state.measurement;
    const worker = args[0];
    const terminate = worker.terminate;
    worker.terminate = function (...terminationArgs) {
      if (measurement === state.measurement) mark('worker-terminated');
      return terminate.apply(this, terminationArgs);
    };
    mark('worker-start');
    try {
      const backend = await workerLoad.apply(this, args);
      if (measurement === state.measurement) mark('worker-ready');
      return backend;
    } catch (error) {
      if (measurement === state.measurement) mark('worker-rejected', { error: String(error) });
      throw error;
    }
  };
  if (typeof SceneBinding.create === 'function') {
    const create = SceneBinding.create;
    SceneBinding.create = async function (...args) {
      const measurement = state.measurement;
      mark('binding-start');
      const binding = await create.apply(this, args);
      if (measurement === state.measurement) mark('binding-ready');
      return binding;
    };
  }
  const render = ViewportPipeline.prototype.render;
  ViewportPipeline.prototype.render = function (...args) {
    const start = performance.now();
    const result = render.apply(this, args);
    state.document = args[2];
    if (state.measurement) {
      const id = identify(state.document);
      if (!state.measurement.renders.some((entry) => entry.document === id)) {
        const meshes = [];
        state.document.root.traverse((object) => {
          if (object.isMesh) meshes.push(object);
        });
        mark('first-render-submitted', {
          document: id,
          name: state.document.name,
          meshes: meshes.length,
        });
        state.measurement.renders.push({ document: id, duration: performance.now() - start });
      }
    }
    return result;
  };
  function heartbeat(now) {
    if (state.measurement && state.previousRaf >= state.measurement.startedAt)
      state.measurement.raf.push(now - state.previousRaf);
    state.previousRaf = now;
    requestAnimationFrame(heartbeat);
  }
  requestAnimationFrame(heartbeat);
  new PerformanceObserver((list) => {
    if (state.measurement)
      for (const entry of list.getEntries()) {
        if (entry.startTime >= state.measurement.startedAt)
          state.measurement.longTasks.push({
            at: entry.startTime - state.measurement.startedAt,
            duration: entry.duration,
          });
      }
  }).observe({ type: 'longtask', buffered: true });
  let lastProgress = '';
  const inspectProgress = () => {
    const element = document.querySelector('[data-testid="scene-load-progress"]');
    const signature = element ? `${element.dataset.stage}:${element.dataset.startedAt}` : '';
    if (signature !== lastProgress) {
      lastProgress = signature;
      if (state.measurement)
        state.measurement.progress.push({
          stage: element?.dataset.stage ?? 'dismissed',
          text: element?.textContent,
          at: performance.now() - state.measurement.startedAt,
        });
      if (element && element.dataset.stage === state.cancelAtStage) {
        state.cancelAtStage = null;
        // Cancel through the production button after the stage has had a paint
        // opportunity. A screenshot round-trip can outlast a small model's load.
        requestAnimationFrame(() => {
          const cancel = document.querySelector('[data-testid="scene-load-cancel"]');
          if (cancel) {
            const bar = element.querySelector('[role="progressbar"]');
            const fill = bar?.firstElementChild;
            mark('cancel-request', {
              loadingStage: element.dataset.stage,
              progressPresent: Boolean(bar),
              progressValue: bar?.getAttribute('aria-valuenow') ?? null,
              progressFill: fill
                ? {
                    width: fill.getBoundingClientRect().width,
                    height: fill.getBoundingClientRect().height,
                    animation: getComputedStyle(fill).animationName,
                    running: fill
                      .getAnimations()
                      .some((animation) => animation.playState === 'running'),
                  }
                : null,
            });
            cancel.click();
          }
        });
      }
    }
  };
  new MutationObserver(inspectProgress).observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['data-stage', 'data-started-at'],
  });
}

const bootstrap = `import { PhysicsDocument } from '/src/physics/PhysicsDocument.ts';
import { WorkerBackend } from '/src/physics/WorkerBackend.ts';
import { SceneBinding } from '/src/physics/SceneBinding.ts';
import { ViewportPipeline } from '/src/viewport/pipeline.ts';
(${observe.toString()})(PhysicsDocument, WorkerBackend, SceneBinding, ViewportPipeline);
await import('/src/main.tsx');`;
const server = await createServer({
  root,
  configFile: false,
  plugins: [
    {
      name: 'production-loading-observation',
      transformIndexHtml: (html) => html.replace('/src/main.tsx', '/__loading_bootstrap.js'),
      resolveId: (id) => (id === '/__loading_bootstrap.js' ? '\0loading-bootstrap' : null),
      load: (id) => (id === '\0loading-bootstrap' ? bootstrap : null),
    },
  ],
  server: { host: '127.0.0.1', port: 0, hmr: false, watch: null },
});
await server.listen();
const results = [],
  checks = [],
  errors = [];
let app, page, failure;
const distribution = (values) => {
  const sorted = values.toSorted((a, b) => a - b);
  return {
    count: sorted.length,
    median: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
    p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
    max: sorted.at(-1) ?? 0,
  };
};
try {
  app = await electron.launch({
    args: [root, '--hidden', '--user-data-dir=' + (await mkdtemp(path.join(output, 'profile-')))],
    cwd: root,
    env: { ...process.env, MOJIVE_DEV_URL: server.resolvedUrls.local[0] },
    timeout: 45000,
  });
  app.process().stderr?.on('data', (chunk) => process.stderr.write(chunk));
  page = await app.firstWindow();
  page.setDefaultTimeout(30000);
  page.on('pageerror', (error) => errors.push(error.stack || error.message));
  await page.waitForSelector('.viewport-stage canvas');
  await page.waitForFunction(
    () =>
      window.__modelLoadingReview.document &&
      !document.querySelector('.loading-scrim, [data-testid="scene-load-progress"]'),
  );
  await page.evaluate(() => document.fonts.ready);
  const progress = page.getByTestId('scene-load-progress');
  const action = (id) =>
    app.evaluate(
      ({ BrowserWindow }, id) =>
        BrowserWindow.getAllWindows()[0].webContents.send('mojive:menu-action', id),
      id,
    );
  async function choose(file, label, cancelAtStage = null) {
    await action(openAction);
    const dialog = page.locator('.file-dialog');
    await dialog.waitFor();
    await dialog.getByRole('textbox', { name: 'Location', exact: true }).fill(path.dirname(file));
    await dialog.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    const entry = dialog.locator('.file-dialog-entry').filter({
      has: page.locator('span[data-tooltip]').filter({
        hasText: new RegExp('^' + path.basename(file).replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$'),
      }),
    });
    await entry.waitFor();
    await page.evaluate(
      ([label, cancelAtStage]) => window.__modelLoadingReview.start(label, cancelAtStage),
      [label, cancelAtStage],
    );
    await entry.dblclick();
  }
  async function finish(label) {
    await page.waitForTimeout(100);
    const result = await page.evaluate(() => window.__modelLoadingReview.finish());
    result.rafDistribution = distribution(result.raf);
    result.longTaskDistribution = distribution(result.longTasks.map((entry) => entry.duration));
    results.push(result);
    await page.screenshot({ path: path.join(output, `${label}.png`) });
    console.log(
      JSON.stringify({
        label,
        elapsed: result.elapsed,
        raf: result.rafDistribution,
        events: result.events,
        progress: result.progress,
      }),
    );
    return result;
  }
  async function waitInstalled(file) {
    const discard = page.getByRole('button', { name: "Don't Save", exact: true });
    await page.waitForFunction(
      (name) => {
        const state = window.__modelLoadingReview;
        return (
          (state.document?.name === name &&
            state.identity() !== state.measurement.originalDocument) ||
          [...document.querySelectorAll('button')].some(
            (button) => button.textContent?.trim() === "Don't Save",
          )
        );
      },
      path.basename(file),
      { timeout },
    );
    if (await discard.isVisible()) await discard.click();
    await page.waitForFunction(
      (name) => {
        const state = window.__modelLoadingReview;
        return (
          state.document?.name === name && state.identity() !== state.measurement.originalDocument
        );
      },
      path.basename(file),
      { timeout },
    );
    await page.locator('.file-dialog').waitFor({ state: 'detached', timeout });
    if (!baseline) await progress.waitFor({ state: 'detached', timeout });
  }
  for (const [file, label] of [
    [first, 'first-model'],
    [model, 'second-model'],
  ]) {
    await choose(file, label);
    if (!baseline) {
      await page.waitForFunction(
        () => window.__modelLoadingReview.measurement.progress.length > 0,
        null,
        { timeout: 2000 },
      );
      checks.push(`${label}: progress appears before completion`);
      if (captureStage && label === 'second-model')
        await page.waitForFunction(
          (stage) =>
            document.querySelector('[data-testid="scene-load-progress"]')?.dataset.stage === stage,
          captureStage,
          { timeout },
        );
      if (await progress.isVisible())
        await page.screenshot({ path: path.join(output, `${label}-loading.png`) });
    }
    await waitInstalled(file);
    const result = await finish(label);
    assert.notEqual(
      result.originalDocument,
      result.finalDocument,
      'A successful load installs a new scene',
    );
    if (!baseline)
      assert.ok(
        result.progress.some((entry) => entry.stage !== 'dismissed'),
        'Real load stages are visible',
      );
  }
  if (!baseline) {
    await choose(model, 'cancel-load', cancelStage);
    await page.waitForFunction(
      () =>
        window.__modelLoadingReview.measurement.events.some(
          (event) => event.stage === 'cancel-request',
        ),
      null,
      { timeout },
    );
    await progress.waitFor({ state: 'detached', timeout: 5000 });
    await page.waitForTimeout(300);
    const cancelled = await finish('cancel-load');
    assert.equal(
      cancelled.finalDocument,
      cancelled.originalDocument,
      'Cancellation preserves the displayed scene',
    );
    if (cancelStage === 'compile') {
      const cancelRequest = cancelled.events.find((event) => event.stage === 'cancel-request');
      assert.equal(cancelRequest.progressPresent, true, 'Compilation renders a progress indicator');
      assert.equal(
        cancelRequest.progressValue,
        null,
        'Model compilation reports indeterminate progress, not an invented percentage',
      );
      assert.ok(
        cancelRequest.progressFill?.width > 0 &&
          cancelRequest.progressFill?.height > 0 &&
          cancelRequest.progressFill?.running,
        'Indeterminate compilation renders an animated progress fill',
      );
      checks.push('Compilation progress is indeterminate');
      assert.ok(
        cancelled.events.some((event) => event.stage === 'worker-terminated'),
        'Cancelling compilation terminates the candidate worker',
      );
      checks.push('Compilation cancellation terminates the candidate worker');
    }
    checks.push('Cancellation preserves the existing scene and dismisses loading');
    // The custom picker may remain available for a retry after cancellation.
    if (await page.locator('.file-dialog').isVisible())
      await page
        .locator('.file-dialog')
        .getByRole('button', { name: 'Cancel', exact: true })
        .click();
    const invalid = path.join(output, 'invalid-model.xml');
    await writeFile(
      invalid,
      '<mujoco><worldbody><body><geom type="not-a-geom"/></body></worldbody></mujoco>',
    );
    await choose(invalid, 'failed-load');
    await progress.waitFor({ state: 'detached', timeout });
    await page.waitForFunction(
      () =>
        window.__modelLoadingReview.measurement.events.some((event) =>
          event.stage.endsWith('rejected'),
        ),
      null,
      { timeout },
    );
    const failed = await finish('failed-load');
    assert.equal(
      failed.finalDocument,
      failed.originalDocument,
      'Failure preserves the displayed scene',
    );
    checks.push('Compile failure preserves the existing scene');
    if (await page.locator('.file-dialog').isVisible())
      await page
        .locator('.file-dialog')
        .getByRole('button', { name: 'Cancel', exact: true })
        .click();
    await choose(model, 'retry-load');
    await waitInstalled(model);
    const retry = await finish('retry-load');
    assert.notEqual(
      retry.finalDocument,
      retry.originalDocument,
      'A model can load after cancellation and failure',
    );
    checks.push('A real model loads again after cancellation and failure');
  }
  assert.deepEqual(errors, [], 'The renderer has no unhandled errors');
} catch (error) {
  failure = error;
  if (page) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
} finally {
  await writeFile(
    path.join(output, 'report.json'),
    JSON.stringify(
      {
        baseline,
        first,
        model,
        cancelStage,
        results,
        checks,
        errors,
        failure: failure?.stack ?? null,
      },
      null,
      2,
    ),
  );
  await app?.close();
  await server.close();
}
if (failure) throw failure;
console.log(`Model loading review: ${path.join(output, 'report.json')}`);
