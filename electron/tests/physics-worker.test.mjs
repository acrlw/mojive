import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('official MuJoCo runs in a browser Worker without Electron or SharedArrayBuffer', async (t) => {
  const server = await createServer({ configFile: false, root: path.resolve('.'),
    optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  t.after(() => server.close());
  const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const result = await page.evaluate(async () => {
    const { loadMujocoBackend } = await import('/src/physics/WorkerBackend.ts');
    const { createModelBundle } = await import('/src/resources/modelBundle.ts');
    const xml = '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>';
    const bundle = await createModelBundle('Worker model', 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(xml) }]);
    const stages = [];
    const backend = await loadMujocoBackend(bundle, { onProgress: (progress) => stages.push(progress.stage) });
    try {
      const first = await backend.frame();
      const frame = await backend.execute({ type: 'step', count: 6000 });
      const snapshot = await backend.capture();
      await backend.execute({ type: 'reset' });
      await backend.execute({ type: 'restore', snapshot });
      const restored = await backend.capture();
      const controller = new AbortController();
      let cancelled = false;
      try {
        await loadMujocoBackend(bundle, { signal: controller.signal, onProgress: (progress) => {
          if (progress.stage === 'runtime') controller.abort();
        } });
      } catch (error) { cancelled = error.name === 'AbortError'; }
      return { stages, cancelled, retainedPackage: new TextDecoder().decode(bundle.files[0].bytes) === xml, initial: first.time, time: frame.time, sourceId: frame.sourceId,
        restored: JSON.stringify(restored) === JSON.stringify(snapshot),
        typed: frame.bodyTransforms instanceof Float64Array, isolated: crossOriginIsolated,
        available: typeof window.mojive !== 'undefined' };
    } finally { backend.dispose(); }
  });
  assert.equal(result.initial, 0);
  assert.equal(result.cancelled, true);
  assert.equal(result.retainedPackage, true);
  assert.deepEqual([...new Set(result.stages)], ['runtime', 'verify', 'compile', 'source']);
  assert.ok(result.time > 11.99);
  assert.equal(result.restored, true);
  assert.equal(result.typed, true);
  assert.equal(result.isolated, false);
  assert.equal(result.available, false);
});
