import { build, createServer, preview } from 'vite';
import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const first = path.resolve(option('--first', path.join(root, '../assets/joint_gizmo.xml')));
const model = path.resolve(option('--model', path.join(root, '../../mujoco_menagerie/unitree_g1/scene.xml')));
const jointName = option('--joint', 'left_elbow_joint');
const out = path.resolve(option('--output', path.join(root, '../output/electron/joint-performance')));
const moves = Number(option('--moves', '120'));
const repeats = Number(option('--repeat', '2'));
const width = Number(option('--width', '2048'));
const height = Number(option('--height', '1047'));
const scaleFactor = Number(option('--scale-factor', '2'));
const profileCPU = args.includes('--cpu-profile');
const traceFrames = args.includes('--trace');
const development = args.includes('--dev');
assert.ok(Number.isInteger(moves) && moves >= 12, '--moves must be an integer of at least 12');
assert.ok(Number.isInteger(repeats) && repeats >= 1, '--repeat must be a positive integer');
assert.ok(Number.isInteger(width) && width >= 960, '--width must be an integer of at least 960');
assert.ok(Number.isInteger(height) && height >= 600, '--height must be an integer of at least 600');
assert.ok(Number.isFinite(scaleFactor) && scaleFactor >= 1 && scaleFactor <= 3, '--scale-factor must be between 1 and 3');
await mkdir(out, { recursive: true });

// Observe the production App, worker, history and renderer. Pointer input below still goes
// through Playwright; private geometry is read only to locate the actual rendered handle.
function observe(THREE, PhysicsDocument, ViewportPipeline, ViewportMetrics, GizmoOverlay, NativeRaster) {
  localStorage.setItem('mojive.electron.settings.v1', JSON.stringify({ language: 'en', appearance: 'dark' }));
  const state = window.__jointPerformance = {
    runtime: null, overlay: null, active: false, data: null, restores: 0,
    pending: new Set(), backendPending: 0, jointName: '', document: null,
    phase: '', lastInvalidation: performance.now(),
  };
  state.start = (phase = '') => {
    state.phase = phase;
    state.data = {
      startedAt: performance.now(), timeOrigin: performance.timeOrigin,
      requests: [], dispatched: [], acknowledgements: [], applied: [], renders: [],
      frame: [], passes: [], raf: [], rafEvents: [], gizmo: [], serialize: [], clone: [], compare: [],
      invalidations: [], pointer: [], pointerDown: [], raster: [], rasterResize: [], longTasks: [],
      pendingMax: 0, backendPendingMax: 0,
    };
    state.active = true;
  };
  state.stop = () => { state.active = false; return state.data; };
  new PerformanceObserver(list => {
    if (state.active) for (const task of list.getEntries())
      state.data.longTasks.push({ at: task.startTime, ms: task.duration });
  }).observe({ type: 'longtask', buffered: false });
  const resize = NativeRaster.prototype.resize;
  NativeRaster.prototype.resize = function (...args) {
    const before = [this.canvas.width, this.canvas.height], start = performance.now();
    try { return resize.apply(this, args); }
    finally {
      const after = [this.canvas.width, this.canvas.height];
      if (state.active && before.some((value, index) => value !== after[index]))
        state.data.rasterResize.push({ at: start, ms: performance.now() - start, before, after });
    }
  };
  const flush = NativeRaster.prototype.flush;
  NativeRaster.prototype.flush = function (...args) {
    const count = this.count, start = performance.now();
    const stages = [];
    const observeCall = (owner, name) => {
      const method = owner[name];
      owner[name] = function (...values) {
        const at = performance.now();
        try { return method.apply(this, values); }
        finally { stages.push({ name, ms: performance.now() - at }); }
      };
      return () => { owner[name] = method; };
    };
    const restore = state.active && count ? [
      ...['viewport', 'clear', 'bufferSubData', 'drawArrays'].map(name => observeCall(this.gl, name)),
      observeCall(args[0], 'drawImage'),
    ] : [];
    try { return flush.apply(this, args); }
    finally {
      restore.forEach(reset => reset());
      if (state.active && count) state.data.raster.push({ at: start, ms: performance.now() - start,
        width: this.canvas.width, height: this.canvas.height, vertices: count / 6, stages });
    }
  };
  const sourceJoint = () => state.runtime?.session.current.source?.joints.find(joint => joint.name === state.jointName);
  const documentJoint = () => state.runtime?.document.joints.find(joint => joint.name === state.jointName);
  const frameValue = frame => {
    const joint = sourceJoint();
    return joint && frame ? frame.jointPositions[joint.offset] : null;
  };
  const load = PhysicsDocument.load;
  PhysicsDocument.load = async function (...args) {
    const runtime = await load.apply(this, args);
    state.runtime = runtime;
    const session = runtime.session;
    const update = session.updateTarget;
    session.updateTarget = function (command) {
      const data = state.active ? state.data : null;
      if (data && command.type === 'jointPose')
        data.requests.push({ at: performance.now(), id: command.id, position: [...command.position] });
      const completion = update.call(this, command);
      if (!state.pending.has(completion)) {
        state.pending.add(completion);
        const settled = () => state.pending.delete(completion);
        void completion.then(settled, settled);
      }
      if (data) data.pendingMax = Math.max(data.pendingMax, state.pending.size);
      return completion;
    };
    const backend = session.backend;
    const executeBackend = backend.execute;
    backend.execute = async function (command) {
      const data = state.active && command.type === 'jointPose' ? state.data : null;
      const start = performance.now();
      state.backendPending++;
      if (data) {
        data.backendPendingMax = Math.max(data.backendPendingMax, state.backendPending);
        data.dispatched.push({ at: start, id: command.id, position: [...command.position] });
      }
      try {
        const frame = await executeBackend.call(this, command);
        if (command.type === 'restore') state.restores++;
        if (data) data.acknowledgements.push({
          at: performance.now(), ms: performance.now() - start,
          id: command.id, requested: [...command.position], value: frameValue(frame),
        });
        return frame;
      } finally { state.backendPending--; }
    };
    let previous = session.current.frame;
    session.subscribe(({ frame, playing }) => {
      if (state.active && frame && frame !== previous)
        state.data.applied.push({ at: performance.now(), value: frameValue(frame), documentValue: documentJoint()?.value, playing });
      previous = frame;
    });
    return runtime;
  };
  const render = ViewportPipeline.prototype.render;
  ViewportPipeline.prototype.render = function (...args) {
    if (!this.renderer.__observedPasses) {
      const renderer = this.renderer, original = renderer.render.bind(renderer);
      const gl = renderer.getContext(), timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
      const pending = [];
      renderer.__observedPasses = true;
      renderer.render = (scene, camera) => {
        for (let i = pending.length - 1; i >= 0; i--) {
          const { query, entry } = pending[i];
          if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) continue;
          if (!gl.getParameter(timer.GPU_DISJOINT_EXT))
            entry.gpuMs = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
          gl.deleteQuery(query);
          pending.splice(i, 1);
        }
        if (!state.active) return original(scene, camera);
        const target = renderer.getRenderTarget();
        const entry = { at: performance.now(), target: this.targets.indexOf(target),
          kind: scene === this.screen ? (this.quad.material === this.outline ? 'outline' : 'composite')
            : scene.overrideMaterial ? 'depth' : target === this.targets[2] ? 'mask' : 'scene' };
        const calls = renderer.info.render.calls, triangles = renderer.info.render.triangles;
        const query = timer ? gl.createQuery() : null;
        if (query) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
        try { return original(scene, camera); }
        finally {
          if (query) { gl.endQuery(timer.TIME_ELAPSED_EXT); pending.push({ query, entry }); }
          entry.cpuMs = performance.now() - entry.at;
          entry.calls = renderer.info.render.calls - calls;
          entry.triangles = renderer.info.render.triangles - triangles;
          state.data.passes.push(entry);
        }
      };
    }
    if (state.active) state.data.renders.push({
      at: performance.now(), value: documentJoint()?.value,
      frameValue: frameValue(state.runtime?.session.current.frame),
      camera: { position: args[1].position.toArray(), quaternion: args[1].quaternion.toArray() },
    });
    return render.apply(this, args);
  };
  const record = ViewportMetrics.prototype.record;
  ViewportMetrics.prototype.record = function (start, end, workload) {
    if (state.active) state.data.frame.push({ at: start, ms: end - start, ...workload });
    return record.call(this, start, end, workload);
  };
  for (const method of ['update', 'begin', 'move', 'end']) {
    const original = GizmoOverlay.prototype[method];
    GizmoOverlay.prototype[method] = function (...args) {
      state.overlay = this;
      const start = performance.now();
      const result = original.apply(this, args);
      if (state.active) state.data.gizmo.push({
        method, at: start, ms: performance.now() - start,
        accepted: method === 'begin' ? Boolean(result) : undefined,
        dragging: Boolean(this.drag), total: this.drag?.total,
        requested: this.drag ? this.drag.jointValue + this.drag.total : undefined,
      });
      return result;
    };
  }
  const serialize = THREE.Object3D.prototype.toJSON;
  THREE.Object3D.prototype.toJSON = function (...args) {
    const watched = state.active && this === state.runtime?.document.root, start = performance.now();
    try { return serialize.apply(this, args); }
    finally { if (watched) state.data.serialize.push(performance.now() - start); }
  };
  const clone = window.structuredClone;
  window.structuredClone = function (value, ...args) {
    const watched = state.active && value?.root && value?.nodes && value?.files, start = performance.now();
    try { return clone.call(this, value, ...args); }
    finally { if (watched) state.data.clone.push(performance.now() - start); }
  };
  const stringify = JSON.stringify;
  JSON.stringify = function (value, ...args) {
    const watched = state.active && value?.document && value?.timeline, start = performance.now();
    try { return stringify.call(this, value, ...args); }
    finally { if (watched) state.data.compare.push(performance.now() - start); }
  };
  let previous = performance.now();
  function frame(now) {
    if (state.active) {
      state.data.raf.push(now - previous);
      state.data.rafEvents.push({ at: now, deliveredAt: performance.now(), ms: now - previous, phase: state.phase });
    }
    previous = now;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  document.addEventListener('pointerdown', event => {
    if (state.active && state.phase === 'orbit' && event.button === 0 &&
      event.target?.dataset?.testid === 'viewport-canvas')
      state.data.pointerDown.push({ at: performance.now(), x: event.clientX, y: event.clientY });
  }, true);
  document.addEventListener('pointermove', event => {
    if (state.active && state.phase === 'orbit' && event.buttons === 1 &&
      event.target?.dataset?.testid === 'viewport-canvas')
      state.data.pointer.push({ at: performance.now(), x: event.clientX, y: event.clientY });
  }, true);
  state.handlePoint = () => {
    const overlay = state.overlay, joint = sourceJoint();
    if (!overlay || overlay.tool !== 'joint' || overlay.joint?.name !== state.jointName)
      throw Error('The selected production joint gizmo is not ready.');
    if (!joint || joint.width !== 1 || joint.type === 'prismatic')
      throw Error('Choose a scalar revolute joint for the ring drag review.');
    const handle = overlay.handles.find(handle => handle.key === 'r2' && handle.kind === 'ring');
    if (!handle) throw Error('The selected joint has no visible ring handle.');
    const rect = overlay.canvas.getBoundingClientRect();
    const points = handle.path.filter(point =>
      point[0] > 90 && point[0] < rect.width - 95 && point[1] > 100 && point[1] < rect.height - 80 &&
      overlay.hit(point)?.key === handle.key);
    if (!points.length) throw Error('No ring point is hit-testable inside the unobstructed viewport.');
    points.sort((a, b) => Math.min(b[0], rect.width - b[0], b[1], rect.height - b[1]) -
      Math.min(a[0], rect.width - a[0], a[1], rect.height - a[1]));
    return { x: rect.left + points[0][0], y: rect.top + points[0][1] };
  };
  state.dragPath = count => {
    const overlay = state.overlay, drag = overlay?.drag, joint = sourceJoint();
    if (!drag || drag.handle.kind !== 'ring') throw Error('Mouse down did not acquire the real ring drag.');
    const rect = overlay.canvas.getBoundingClientRect();
    const value = documentJoint().value, limits = joint.limits ?? [-Infinity, Infinity];
    const positiveRoom = limits[1] - value, negativeRoom = value - limits[0];
    const direction = positiveRoom >= negativeRoom ? 1 : -1;
    const angle = Math.min(0.55, Math.max(positiveRoom, negativeRoom) * 0.5);
    if (angle < 0.02) throw Error('Insufficient joint range for a measurable drag.');
    const points = Array.from({ length: count }, (_, index) => {
      const amount = angle * (index + 1) / count * direction;
      if (drag.linear) return [drag.start[0] + drag.tangent[0] * amount * 100,
        drag.start[1] + drag.tangent[1] * amount * 100];
      const segments = drag.ring.length - 1;
      const position = (amount / (Math.PI * 2) * segments + segments) % segments;
      const left = Math.floor(position), blend = position - left;
      const a = drag.ring[left], b = drag.ring[(left + 1) % segments];
      return [a[0] + (b[0] - a[0]) * blend, a[1] + (b[1] - a[1]) * blend];
    });
    if (points.some(point => point[0] < 70 || point[0] > rect.width - 10 || point[1] < 80 || point[1] > rect.height - 15))
      throw Error('The ring drag path leaves the unobstructed viewport.');
    return { linear: drag.linear, direction, angle,
      points: points.map(point => ({ x: rect.left + point[0], y: rect.top + point[1] })) };
  };
}

const bootstrap = `import * as THREE from 'three';
import { PhysicsDocument } from '/src/physics/PhysicsDocument.ts';
import { ViewportPipeline } from '/src/viewport/pipeline.ts';
import { ViewportMetrics } from '/src/viewport/metrics.ts';
import { GizmoOverlay } from '/src/viewport/gizmo.ts';
import { NativeRaster } from '/src/viewport/viewcube.ts';
(${observe.toString()})(THREE, PhysicsDocument, ViewportPipeline, ViewportMetrics, GizmoOverlay, NativeRaster);
await import('/src/main.tsx');`;
const configuration = {
  root, configFile: false, base: './',
  build: { outDir: path.join(out, 'build'), emptyOutDir: true, target: 'esnext', chunkSizeWarningLimit: 1600 },
  plugins: [{
    name: 'production-joint-performance-observation',
    transformIndexHtml: { order: 'pre', handler: html => html.replace('/src/main.tsx', '/__joint_performance_bootstrap.js') },
    resolveId: id => id === '/__joint_performance_bootstrap.js' ? '\0joint-performance-bootstrap' : null,
    load: id => id === '\0joint-performance-bootstrap' ? bootstrap : null,
    transform(code, id) {
      if (!id.split('?')[0].endsWith('/src/Viewport.tsx')) return null;
      const marker = 'function invalidate() {';
      assert.ok(code.includes(marker), 'Production viewport invalidation observation point exists');
      return code.replace(marker, `${marker}
        const observation = window.__jointPerformance;
        if (observation) {
          observation.lastInvalidation = performance.now();
          if (observation.active && ['settle', 'idle'].includes(observation.phase))
            observation.data.invalidations.push({ at: performance.now(), stack: new Error().stack });
        }
      `);
    },
  }],
  server: { host: '127.0.0.1', port: 0, hmr: false, watch: null },
  preview: { host: '127.0.0.1', port: 0 },
};
// Development React adds substantial work to every scene update. Benchmark the bundled
// application by default; retain --dev for diagnosing development-only regressions.
if (!development) await build(configuration);
const server = development ? await createServer(configuration) : await preview(configuration);
if (development) await server.listen();
const distribution = values => {
  const sorted = values.toSorted((a, b) => a - b);
  return { count: values.length, median: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
    p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0, max: sorted.at(-1) ?? 0 };
};
function assertMonotonic(values, direction, minimumSpan, label) {
  assert.ok(values.length >= 2 && values.every(Number.isFinite), `${label} contains measurable finite poses`);
  assert.ok(direction * (values.at(-1) - values[0]) >= minimumSpan,
    `${label} must travel in the requested direction by at least ${minimumSpan} radians`);
  for (let index = 1; index < values.length; index++)
    assert.ok(direction * (values[index] - values[index - 1]) >= -1e-7,
      `${label} must not alternate between old and new poses at sample ${index}`);
}
function assertLightweight(data, label) {
  assert.equal(data.rasterResize.length, 0, `${label}: no WebGL drawing-buffer resize during interaction`);
  assert.equal(data.serialize.length, 0, `${label}: no full scene serialization`);
  assert.equal(data.clone.length, 0, `${label}: no full document cloning`);
  assert.equal(data.compare.length, 0, `${label}: no document/timeline JSON comparison`);
}
let app;
const report = { first, model, jointName, moves, repeats, build: development ? 'development' : 'production', results: [], errors: [] };
try {
  app = await electron.launch({
    args: [root, ...(args.includes('--visible') ? [] : ['--hidden']),
      `--force-device-scale-factor=${scaleFactor}`, '--user-data-dir=' + await mkdtemp(path.join(out, 'profile-'))],
    cwd: root, env: { ...process.env, MOJIVE_DEV_URL: server.resolvedUrls.local[0] }, timeout: 45000,
  });
  const page = await app.firstWindow();
  report.gpu = await app.evaluate(({ app }) => app.getGPUInfo('complete'));
  if (traceFrames) await app.evaluate(({ contentTracing }) => contentTracing.startRecording({
    included_categories: ['toplevel', 'blink', 'cc', 'gpu', 'viz', 'benchmark', 'input',
      'latencyInfo', 'renderer.scheduler', 'v8', 'devtools.timeline',
      'disabled-by-default-devtools.timeline.frame'],
  }));
  await app.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setContentSize(...size), [width, height]);
  const cdp = profileCPU ? await page.context().newCDPSession(page) : null;
  if (cdp) await cdp.send('Profiler.enable');
  report.window = await page.evaluate(() => ({ width: innerWidth, height: innerHeight, dpr: devicePixelRatio }));
  page.setDefaultTimeout(30000);
  page.on('pageerror', error => report.errors.push(error.stack || error.message));
  page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
  await page.waitForSelector('.viewport-stage canvas');
  await page.evaluate(() => document.fonts.ready);
  const action = id => app.evaluate(({ BrowserWindow }, id) =>
    BrowserWindow.getAllWindows()[0].webContents.send('mojive:menu-action', id), id);
  const capture = () => page.evaluate(() => window.__jointPerformance.runtime.session.capture());
  async function open(file) {
    await action('file.openSimulation');
    const dialog = page.locator('.file-dialog');
    await dialog.waitFor();
    const location = dialog.getByRole('textbox', { name: 'Location', exact: true });
    await location.fill(path.dirname(file));
    await location.press('Enter');
    await dialog.locator('.file-dialog-entry').filter({ has: page.locator('span[data-tooltip]').filter({
      hasText: new RegExp('^' + path.basename(file).replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$'),
    }) }).dblclick();
    await dialog.waitFor({ state: 'detached', timeout: 120000 });
    const discard = page.getByRole('button', { name: "Don't Save", exact: true });
    if (await discard.isVisible()) await discard.click();
    await page.waitForFunction(name => window.__jointPerformance.runtime?.document.name === name,
      path.basename(file), { timeout: 120000 });
    await page.waitForTimeout(500);
    console.log('Loaded', file);
  }
  async function history(direction, expected) {
    const previous = await page.evaluate(() => window.__jointPerformance.restores);
    await action(`edit.${direction}`);
    await page.waitForFunction(previous => window.__jointPerformance.restores > previous, previous);
    assert.deepEqual(await capture(), expected, `${direction} restores the complete physics snapshot`);
    assert.equal(await page.evaluate(() => window.__jointPerformance.document === window.__jointPerformance.runtime.document),
      true, `${direction} preserves scene document identity and GPU resources`);
  }
  await open(first);
  await open(model);
  await page.evaluate(name => { window.__jointPerformance.jointName = name; }, jointName);
  await page.locator('.panel-header').getByRole('button', { name: 'Search', exact: true }).click();
  await page.locator('.panel-search input').fill(jointName);
  const treeJoint = page.getByRole('treeitem', { name: jointName, exact: true });
  if (cdp) await cdp.send('Profiler.start');
  if (cdp) {
    await cdp.send('Performance.enable');
    report.profileClock = { metrics: (await cdp.send('Performance.getMetrics')).metrics,
      pageTime: await page.evaluate(() => performance.now()) };
  }
  await page.evaluate(() => window.__jointPerformance.start('activation'));
  await treeJoint.dblclick();
  await page.locator('.tool-palette').getByRole('button', { name: /^Joint(?:\s|$)/ }).click();
  await page.waitForTimeout(700);
  report.activation = await page.evaluate(() => window.__jointPerformance.stop());
  assert.equal(report.activation.rasterResize.length, 0, 'first activation must not resize the WebGL drawing buffer');
  if (cdp) {
    const { profile } = await cdp.send('Profiler.stop');
    await writeFile(path.join(out, 'activation.cpuprofile'), JSON.stringify(profile));
  }
  // Keep the arm and body visible around the selected elbow, using the same wheel input
  // as the user. Native focus intentionally frames a joint closely.
  const canvas = page.locator('[data-testid="viewport-canvas"]');
  const initialBounds = await canvas.boundingBox();
  await page.mouse.move(initialBounds.x + initialBounds.width / 2, initialBounds.y + initialBounds.height / 2);
  if (args.includes('--frame-all'))
    await page.getByRole('button', { name: 'Frame all', exact: true }).click();
  else await page.mouse.wheel(0, 600);
  await page.waitForTimeout(400);
  for (let cycle = 0; cycle < repeats; cycle++) for (const mode of option('--modes', 'Visual,Collision').split(',')) {
    const label = `${cycle + 1}-${mode.toLowerCase()}`;
    await page.evaluate(async () => {
      const state = window.__jointPerformance;
      state.runtime.session.pause();
      await state.runtime.session.execute({ type: 'reset' });
      state.document = state.runtime.document;
    });
    await page.locator('.geometry-selector').getByRole('button', { name: mode, exact: true }).click();
    await page.waitForTimeout(300);
    const before = await capture();
    const bounds = await canvas.boundingBox();
    const origin = { x: bounds.x + bounds.width * 0.5, y: bounds.y + 25 };
    const cameraBefore = await page.evaluate(() => {
      const camera = window.__jointPerformance.overlay.camera;
      return { position: camera.position.toArray(), quaternion: camera.quaternion.toArray() };
    });
    const point = await page.evaluate(() => window.__jointPerformance.handlePoint());
    await page.mouse.move(point.x, point.y);
    if (cdp) await cdp.send('Profiler.start');
    await page.evaluate(() => window.__jointPerformance.start('drag'));
    await page.mouse.down();
    const trajectory = await page.evaluate(count => window.__jointPerformance.dragPath(count), moves);
    const start = performance.now();
    for (const point of trajectory.points) await page.mouse.move(point.x, point.y);
    await page.mouse.up();
    const released = performance.now();
    const orbitStartedAt = await page.evaluate(() => {
      window.__jointPerformance.phase = 'orbit';
      return performance.now();
    });
    // Start the next real gesture before waiting for a worker barrier, screenshot or
    // settling delay. Keep one trace so late joint ACKs remain visible during orbit.
    await page.mouse.move(origin.x, origin.y);
    await page.mouse.down();
    const captured = capture().then(snapshot => ({ snapshot, ms: performance.now() - released }), error => ({ error }));
    for (let index = 1; index <= 24; index++) await page.mouse.move(origin.x + index * 6, origin.y);
    for (let index = 23; index >= 0; index--) await page.mouse.move(origin.x + index * 6, origin.y);
    await page.mouse.up();
    await page.waitForTimeout(100);
    const captureResult = await captured;
    if (captureResult.error) throw captureResult.error;
    const after = captureResult.snapshot;
    const captureDrainMs = captureResult.ms;
    const drain = await page.evaluate(() => {
      const state = window.__jointPerformance;
      return { pending: state.pending.size, backend: state.backendPending,
        target: Boolean(state.runtime.session.pendingTarget), drag: Boolean(state.overlay.drag),
        playing: state.runtime.session.current.playing };
    });
    const drag = await page.evaluate(() => window.__jointPerformance.stop());
    if (cdp) {
      const { profile } = await cdp.send('Profiler.stop');
      await writeFile(path.join(out, `${label}.cpuprofile`), JSON.stringify(profile));
    }
    const dragFrames = drag.frame.filter(frame => frame.at < orbitStartedAt);
    const frameGaps = dragFrames.slice(1).map((frame, index) => frame.at - dragFrames[index].at);
    const result = { label, trajectory: { linear: trajectory.linear, direction: trajectory.direction, angle: trajectory.angle },
      pointerDurationMs: released - start, captureDrainMs, orbitStartedAt, drain, drag,
      cadence: { frames: dragFrames.length,
        fps: frameGaps.length ? 1000 * frameGaps.length / (dragFrames.at(-1).at - dragFrames[0].at) : 0,
        interval: distribution(frameGaps) },
      latency: { worker: distribution(drag.acknowledgements.map(ack => ack.ms)),
        render: distribution(dragFrames.map(frame => frame.ms)),
        raf: distribution(drag.raf) } };
    report.results.push(result);
    assert.notDeepEqual(after.state, before.state, `${label}: dragging changes the simulation state`);
    assert.deepEqual(drain, { pending: 0, backend: 0, target: false, drag: false, playing: false },
      `${label}: release/capture drains every target and ends pointer ownership`);
    assert.ok(drag.requests.length > 5 && drag.acknowledgements.length > 0, `${label}: real pointer motion reaches the worker`);
    assert.ok(drag.pendingMax <= 3, `${label}: target backlog remains bounded`);
    for (const ack of drag.acknowledgements)
      assert.ok(Math.abs(ack.value - ack.requested[0]) < 1e-8, `${label}: worker ACK equals its submitted joint target`);
    for (const frame of drag.applied)
      assert.ok(Math.abs(frame.value - frame.documentValue) < 1e-8, `${label}: scene pose follows the published physics frame`);
    for (const frame of drag.renders)
      assert.ok(Math.abs(frame.value - frame.frameValue) < 1e-8, `${label}: rendered pose never jumps ahead of its physics frame`);
    assertMonotonic(drag.requests.map(request => request.position[0]), trajectory.direction, trajectory.angle * 0.5,
      `${label}: requested trajectory`);
    assertMonotonic(drag.acknowledgements.map(ack => ack.value), trajectory.direction, trajectory.angle * 0.5,
      `${label}: acknowledged trajectory`);
    assertMonotonic(drag.renders.map(frame => frame.value), trajectory.direction, trajectory.angle * 0.5,
      `${label}: displayed trajectory`);
    assert.ok(Math.abs(drag.requests.at(-1).position[0] - drag.acknowledgements.at(-1).value) < 1e-8,
      `${label}: the final pointer target survives coalescing and the capture barrier`);
    assertLightweight(drag, label);
    result.orbit = {
      pointer: drag.pointer,
      requests: drag.requests.filter(request => request.at >= orbitStartedAt),
      renders: drag.renders.filter(frame => frame.at >= orbitStartedAt),
      frame: drag.frame.filter(frame => frame.at >= orbitStartedAt),
      releaseToStartMs: drag.pointerDown[0]?.at - drag.gizmo.findLast(event =>
        event.method === 'end' && event.at <= orbitStartedAt)?.at,
    };
    result.orbit.latency = distribution(result.orbit.pointer.map(pointer => {
      const rendered = result.orbit.renders.find(frame => frame.at >= pointer.at);
      return rendered ? rendered.at - pointer.at : null;
    }).filter(Number.isFinite));
    assert.ok(result.orbit.pointer.length >= 40, `${label}: production viewport receives the next orbit gesture`);
    assert.ok(result.orbit.releaseToStartMs >= 0 && result.orbit.releaseToStartMs < 150,
      `${label}: orbit starts immediately after joint release (${result.orbit.releaseToStartMs} ms)`);
    assert.ok(result.orbit.renders.some(frame => frame.camera.quaternion.some((value, index) =>
      Math.abs(value - cameraBefore.quaternion[index]) > 1e-3)), `${label}: orbit changes the real render camera`);
    assert.ok(result.orbit.latency.count >= 40 && result.orbit.latency.p95 < 100 && result.orbit.latency.max < 250,
      `${label}: orbit remains responsive immediately after joint release (${JSON.stringify(result.orbit.latency)})`);
    assert.equal(result.orbit.requests.length, 0, `${label}: orbit does not submit stale joint targets`);
    assert.deepEqual(await capture(), after, `${label}: editor orbit leaves the physics snapshot unchanged`);
    assert.equal(await page.evaluate(() => Boolean(window.__jointPerformance.overlay.drag)), false,
      `${label}: orbit release retains no gizmo pointer ownership`);
    await page.screenshot({ path: path.join(out, `${label}.png`) });

    // Precision-rail dwell and pending React commits are finite transitions. Trace each
    // invalidation while they settle, then measure mouse motion in a quiescent window.
    await page.evaluate(() => window.__jointPerformance.start('settle'));
    await page.mouse.move(bounds.x + bounds.width * 0.5, bounds.y + 25);
    await page.waitForFunction(() => performance.now() - window.__jointPerformance.lastInvalidation >= 750,
      null, { timeout: 6000 });
    result.settle = await page.evaluate(() => window.__jointPerformance.stop());
    await page.evaluate(() => window.__jointPerformance.start('idle'));
    for (let index = 0; index < 48; index++)
      await page.mouse.move(bounds.x + bounds.width * (0.35 + index / 240), bounds.y + 25);
    await page.waitForTimeout(350);
    result.idle = await page.evaluate(() => window.__jointPerformance.stop());
    assert.equal(result.idle.requests.length, 0, `${label}: idle mouse motion submits no physics target`);
    assert.ok(result.idle.renders.length <= 3, `${label}: idle mouse motion does not continually redraw (${result.idle.renders.length} renders)`);
    assertLightweight(result.idle, `${label} idle`);
    await page.evaluate(() => window.__jointPerformance.start('history'));
    await history('undo', before);
    await history('redo', after);
    result.history = await page.evaluate(() => window.__jointPerformance.stop());
    assertLightweight(result.history, `${label} history`);

    // Escape must restore qpos/qvel/act and leave the last committed edit as the undo head.
    const cancelPoint = await page.evaluate(() => window.__jointPerformance.handlePoint());
    await page.mouse.move(cancelPoint.x, cancelPoint.y);
    const previousRestores = await page.evaluate(() => window.__jointPerformance.restores);
    await page.evaluate(() => window.__jointPerformance.start('cancel'));
    await page.mouse.down();
    const cancelledPath = await page.evaluate(count => window.__jointPerformance.dragPath(count), 24);
    for (const point of cancelledPath.points) await page.mouse.move(point.x, point.y);
    await page.keyboard.press('Escape');
    await page.mouse.up();
    await page.waitForFunction(previous => window.__jointPerformance.restores > previous, previousRestores);
    assert.deepEqual(await capture(), after, `${label}: Escape restores the complete pre-gesture state`);
    await history('undo', before);
    await history('redo', after);
    result.cancel = await page.evaluate(() => window.__jointPerformance.stop());
    assertLightweight(result.cancel, `${label} cancel`);
    console.log(label, JSON.stringify({ captureDrainMs, requests: drag.requests.length,
      acknowledgements: drag.acknowledgements.length, idleRenders: result.idle.renders.length,
      settleRenders: result.settle.renders.length, cadence: result.cadence,
      orbitLatency: result.orbit.latency, latency: result.latency }));
  }
  assert.deepEqual(report.errors, [], 'No production browser errors');
  report.passed = true;
} catch (error) {
  report.passed = false;
  report.failure = error.stack || String(error);
  if (app?.windows().length) {
    const page = app.windows()[0];
    report.failurePhase = await page.evaluate(() => {
      const state = window.__jointPerformance;
      if (!state) return null;
      const data = state.active ? state.stop() : state.data;
      return { phase: state.phase, data, now: performance.now(), lastInvalidation: state.lastInvalidation,
        pending: state.pending.size, backendPending: state.backendPending,
        dragging: Boolean(state.overlay?.drag), playing: state.runtime?.session.current.playing };
    }).catch(() => null);
    await page.screenshot({ path: path.join(out, 'failure.png') }).catch(() => {});
  }
  throw error;
} finally {
  await writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  if (app) {
    if (traceFrames) await app.evaluate(({ contentTracing }, destination) =>
      contentTracing.stopRecording(destination), path.join(out, 'frames.trace.json'));
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach(window => window.destroy())).catch(() => {});
    await app.close();
  }
  if (development) await server.close();
  else await new Promise((resolve, reject) => server.httpServer.close(error => error ? reject(error) : resolve()));
}
console.log(path.join(out, 'report.json'));
