import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';
const reference = JSON.parse(
  await readFile(new URL('./drag-guides-native-reference.json', import.meta.url), 'utf8'),
);
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: fileURLToPath(new URL('../', import.meta.url)),
    server: { host: '127.0.0.1', port: 0 },
    optimizeDeps: { noDiscovery: true, include: [] },
    logLevel: 'error',
  });
  server.middlewares.use('/__guides_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Guide projection verification</title>');
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__guides_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function verify(kind) {
  return page.evaluate(
    async ({ kind, fixtures }) => {
      const THREE = await import('/node_modules/three/build/three.module.js'),
        guides = await import('/src/viewport/drag_guides.ts'),
        errors = [];
      const same = (actual, expected, name, tolerance = 2e-5) => {
        if (actual == null || expected == null) {
          if (actual !== expected) errors.push(name + ': null mismatch');
          return;
        }
        const a = Array.isArray(actual) ? actual.flat(Infinity) : [actual],
          b = Array.isArray(expected) ? expected.flat(Infinity) : [expected];
        if (a.length !== b.length) errors.push(name + ': length ' + a.length + ' != ' + b.length);
        else {
          const max = Math.max(0, ...a.map((n, i) => Math.abs(n - b[i])));
          if (max > tolerance) errors.push(name + ': max difference ' + max);
        }
      };
      for (const [i, s] of fixtures.entries()) {
        const camera = s.orthographic
          ? new THREE.OrthographicCamera()
          : new THREE.PerspectiveCamera();
        camera.position.fromArray(s.eye);
        camera.near = s.near;
        camera.matrixWorldInverse.set(...s.view.flat());
        camera.matrixWorld.copy(camera.matrixWorldInverse).invert();
        camera.matrix.copy(camera.matrixWorld);
        camera.matrixAutoUpdate = false;
        camera.projectionMatrix.set(...s.projection.flat());
        camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        if (kind === 'translation') {
          const result = guides.translationRuler(
              camera,
              new THREE.Vector3(...s.current),
              new THREE.Vector3(...s.start),
              new THREE.Vector3(...s.axis),
              900,
              600,
              { slide: s.slide },
            ),
            actual = [
              ...(result?.segments ?? []).map(([start, end]) => ({
                start,
                end,
                alpha: 0.92,
                width: 1.2,
                roundStart: false,
                roundEnd: false,
              })),
              ...(result?.ticks ?? []),
            ];
          same(actual.length, s.lines.length, `${i} line count`);
          for (let j = 0; j < Math.min(actual.length, s.lines.length); j++) {
            const a = actual[j],
              b = s.lines[j];
            for (const key of ['start', 'end', 'alpha', 'width', 'roundStart', 'roundEnd'])
              same(a[key], b[key], `${i} line${j}.${key}`);
          }
        }
        if (kind === 'rotation') {
          const result = guides.rotationGuide(
            camera,
            new THREE.Vector3(...s.origin),
            new THREE.Vector3(...s.axis),
            new THREE.Vector3(...s.radial),
            900,
            600,
            {
              angle: s.angle,
              screen: s.screen,
              joint: s.joint ?? undefined,
              snapping: true,
              projectionFade: !!s.joint,
            },
          );
          same(result?.sector ?? [], s.sectors[0] ?? [], `${i} sector`);
          same(result?.stroke ?? [], s.strokes[0] ?? [], `${i} radial-cut stroke`);
          same(result?.reference ?? [], s.reference[0] ?? [], `${i} reference`);
          same(result?.ticks.length ?? 0, s.ticks.length, `${i} tick count`);
          if (result) same(result.anchor, s.anchor, `${i} readout anchor`);
          for (let j = 0; j < Math.min(result?.ticks.length ?? 0, s.ticks.length); j++)
            for (const key of ['start', 'end', 'alpha', 'width', 'roundStart', 'roundEnd'])
              same(result.ticks[j][key], s.ticks[j][key], `${i} tick${j}.${key}`);
        }
        if (kind === 'axis') {
          const result = guides.rotationAxisGuide(
            camera,
            new THREE.Vector3(...s.origin),
            new THREE.Vector3(...s.axis),
            900,
            600,
            { hinge: true },
          );
          same(result?.alpha ?? null, s.expected?.alpha ?? null, `${i} axis alpha`);
          same(result?.polygons ?? [], s.expected?.polygons ?? [], `${i} axis polygons`);
          same(
            result?.hitPolygon ?? [],
            s.expected?.hitPolygon ?? [],
            `${i} original axis hit polygon`,
          );
        }
      }
      return errors;
    },
    { kind, fixtures: reference[kind] },
  );
}
test('translation snap ruler matches24 native projected/clipped/center-masked cases', async () =>
  assert.deepEqual(await verify('translation'), []));
test('rotation sectors, endpoint radial cuts, screen dials and snap ticks match16 native cases', async () =>
  assert.deepEqual(await verify('rotation'), []));
test('hinge axis rear-disk occlusion and dashed G3 arrow match12 native projections', async () =>
  assert.deepEqual(await verify('axis'), []));
