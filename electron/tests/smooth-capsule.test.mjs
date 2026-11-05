import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { parseCorner, normalGapError } from './helpers/corner-measurement.mjs';
import { settlePaint } from './helpers/surface-style.mjs';
const out = path.resolve('../output/electron/verification/capsules');
let geometry, browser, server;
before(async () => {
  await mkdir(out, { recursive: true });
  const outfile = path.join(out, 'geometry.mjs');
  await build({
    entryPoints: ['src/smooth-geometry.ts'],
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    logLevel: 'silent',
  });
  geometry = await import(pathToFileURL(outfile));
});
after(async () => {
  await browser?.close();
  await server?.close();
});
function capSegments(shape) {
  return parseCorner(geometry.surfacePath(shape.width, shape.height, shape).split(' L ')[0]);
}
function translated(segments, delta) {
  const p = (v) => ({ x: v.x - delta, y: v.y - delta });
  return segments.map((s) =>
    s.type === 'cubic'
      ? { ...s, points: s.points.map(p) }
      : { ...s, start: p(s.start), end: p(s.end), center: p(s.center) },
  );
}
test('G3 capsule contours match native samples, preserve bounds and fit the short-capsule limit', async () => {
  const reference = JSON.parse(
    await readFile(new URL('./fixtures/capsule-native.json', import.meta.url), 'utf8'),
  );
  const results = [];
  for (const sample of reference.cases) {
    const shape = geometry.smoothCapsule(sample.width, sample.height, sample.smoothing, 0.005);
    assert.ok(Math.abs(shape.effectiveSmoothing - sample.effectiveSmoothing) < 2e-10);
    assert.ok(Math.abs(shape.minimumRadius - sample.minimumRadius) < 1e-9);
    assert.ok(Math.abs(shape.depth - sample.depth) < 1e-7);
    const error = normalGapError(
      capSegments(shape),
      sample.samples.slice(1, -1).map(([x, y, nx, ny]) => ({ p: { x, y }, n: { x: nx, y: ny } })),
      0,
    );
    assert.ok(error < 0.005, `${sample.width}/${sample.height}/${sample.smoothing}: ${error}`);
    assert.deepEqual(
      sample.bounds.map((v) => Math.round(v * 1e9) / 1e9),
      [0, 0, sample.width, sample.height],
    );
    const end = geometry.capsuleCapSample(shape, 0),
      mid = geometry.capsuleCapSample(shape, 0.5),
      tail = geometry.capsuleCapSample(shape, 1);
    assert.ok(Math.hypot(end.p.x - sample.samples[0][0], end.p.y - sample.samples[0][1]) < 1e-7);
    assert.ok(
      Math.hypot(tail.p.x - sample.samples.at(-1)[0], tail.p.y - sample.samples.at(-1)[1]) < 1e-7,
    );
    assert.ok(Math.abs(end.curvatureSlope) < 1e-10 && Math.abs(tail.curvatureSlope) < 1e-10);
    if (shape.effectiveSmoothing > 0)
      assert.ok(Math.abs(end.curvature) < 1e-12 && Math.abs(tail.curvature) < 1e-12);
    assert.ok(
      Math.abs(
        (sample.width >= sample.height ? mid.p.x : mid.p.y) - Math.max(sample.width, sample.height),
      ) < 1e-9,
    );
    const svg = geometry.surfacePath(shape.width, shape.height, shape);
    assert.equal((svg.match(/ L /g) ?? []).length, 1, 'only the straight side uses L');
    assert.equal(svg.includes(' C '), shape.effectiveSmoothing > 0);
    results.push({
      width: sample.width,
      height: sample.height,
      smoothing: sample.smoothing,
      effective: shape.effectiveSmoothing,
      normalError: error,
    });
  }
  await writeFile(path.join(out, 'native-comparison.json'), JSON.stringify(results, null, 2));
});
test('capsule nesting and stroke offsets share one analytic contour with regular curvature', () => {
  for (const [width, height] of [
    [120, 28],
    [28, 120],
    [30, 28],
  ]) {
    const base = geometry.smoothCapsule(width, height, 0.6, 0.005);
    for (const delta of [-4, -0.5, 4]) {
      const derived = geometry.offsetCapsule(base, delta, 0.005);
      const samples = Array.from({ length: 129 }, (_, i) => {
        const sample = geometry.capsuleCapSample(base, i / 128);
        return { p: sample.p, n: { x: sample.tangent.y, y: -sample.tangent.x } };
      });
      assert.ok(normalGapError(translated(capSegments(derived), delta), samples, delta) < 0.005);
      assert.equal(derived.width, width + 2 * delta);
      assert.equal(derived.height, height + 2 * delta);
      const chain = geometry.offsetCapsule(
        geometry.offsetCapsule(base, delta / 2),
        delta / 2,
        0.005,
      );
      assert.equal(
        geometry.surfacePath(chain.width, chain.height, chain),
        geometry.surfacePath(derived.width, derived.height, derived),
      );
      const curves = capSegments(derived).filter((s) => s.type === 'cubic');
      for (let i = 1; i < curves.length; i++) {
        const a = geometry.evaluateCubic(curves[i - 1].points, 1),
          b = geometry.evaluateCubic(curves[i].points, 0);
        if (Math.hypot(a.p.x - b.p.x, a.p.y - b.p.y) > 1e-8) continue;
        assert.ok(Math.hypot(a.tangent.x - b.tangent.x, a.tangent.y - b.tangent.y) < 1e-7);
        assert.ok(Math.abs(a.curvature - b.curvature) < 4e-6);
      }
    }
    assert.throws(() => geometry.offsetCapsule(base, -base.minimumRadius), /regularity limit/);
  }
});
function meanEndRadius(shape, center) {
  const width = Math.max(shape.width, shape.height),
    height = Math.min(shape.width, shape.height);
  const cap = Array.from({ length: 1025 }, (_, i) => {
    const p = geometry.capsuleCapSample(shape, i / 1024).p;
    return shape.height > shape.width ? { x: p.y, y: shape.width - p.x } : p;
  });
  const polygon = [...cap, ...cap.map((p) => ({ x: width - p.x, y: height - p.y }))];
  let mean = 0;
  for (let i = 0; i <= 720; i++) {
    const angle = Math.PI / 2 + (Math.PI * i) / 720;
    const dx = Math.cos(angle),
      dy = Math.sin(angle);
    let nearest = Infinity;
    for (let j = 0; j < polygon.length; j++) {
      const a = polygon[j],
        b = polygon[(j + 1) % polygon.length];
      const ex = b.x - a.x,
        ey = b.y - a.y,
        vx = a.x - center,
        vy = a.y - height / 2;
      const denominator = dx * ey - dy * ex;
      if (Math.abs(denominator) < 1e-12) continue;
      const distance = (vx * ey - vy * ex) / denominator;
      const along = (vx * dy - vy * dx) / denominator;
      if (distance > 0 && along >= -1e-12 && along <= 1 + 1e-12)
        nearest = Math.min(nearest, distance);
    }
    assert.ok(Number.isFinite(nearest));
    mean += (nearest * (i === 0 || i === 720 ? 0.5 : 1)) / 720;
  }
  return mean;
}

test('optical capsule spacing matches native sampling and the actual fitted or offset profile', async () => {
  const reference = JSON.parse(
    await readFile(new URL('./fixtures/capsule-native.json', import.meta.url), 'utf8'),
  );
  const results = [];
  const started = performance.now();
  for (const native of reference.cases) {
    const shape = geometry.smoothCapsule(native.width, native.height, native.smoothing);
    const end = geometry.capsuleEndPadding(shape);
    const short = Math.min(native.width, native.height),
      long = Math.max(native.width, native.height);
    assert.ok(end >= short / 2 && end <= long / 2);
    // Native uses 181 equal-weight ray samples; the shared solver integrates
    // the continuous angle distribution. At these sizes their difference is <0.006px.
    assert.ok(Math.abs(end - native.opticalEndPadding) < 0.006);
    const rotated = geometry.smoothCapsule(native.height, native.width, native.smoothing);
    assert.equal(geometry.capsuleEndPadding(rotated), end);
    const scaled = geometry.smoothCapsule(
      native.width * 2.25,
      native.height * 2.25,
      native.smoothing,
    );
    assert.ok(Math.abs(geometry.capsuleEndPadding(scaled) - end * 2.25) < 1e-9);
    results.push({
      width: native.width,
      height: native.height,
      smoothing: native.smoothing,
      end,
      native: native.opticalEndPadding,
    });
  }
  const profilesMs = performance.now() - started;
  const measurements = [];
  for (const smoothing of [0, 0.382, 1]) {
    for (const [width, height] of [
      [120, 28],
      [28, 120],
      [30, 28],
    ]) {
      const base = geometry.smoothCapsule(width, height, smoothing);
      for (const delta of [-4, 0, 4]) {
        const shape = geometry.offsetCapsule(base, delta);
        const end = geometry.capsuleEndPadding(shape);
        const mean = meanEndRadius(shape, end),
          radius = Math.min(shape.width, shape.height) / 2;
        assert.ok(
          Math.abs(mean - radius) < 0.0002,
          `${width}/${height}/${smoothing}/${delta}: ${mean - radius}`,
        );
        measurements.push({ width, height, smoothing, delta, end, mean, radius });
      }
    }
  }
  const base = geometry.smoothCapsule(120, 28, 0.382);
  const warm = performance.now();
  for (let i = 0; i < 10000; i++) geometry.capsuleEndPadding(base);
  await writeFile(
    path.join(out, 'optical-spacing.json'),
    JSON.stringify(
      { results, measurements, profilesMs, cached10000Ms: performance.now() - warm },
      null,
      2,
    ),
  );
});

test('production search, switch, chips and vertical capsule use curved SVG while circular content stays circular', async () => {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'error' });
  await server.listen();
  const records = [];
  for (const [theme, scale] of [
    ['dark', 1],
    ['light', 1.25],
  ]) {
    const page = await browser.newPage({
      viewport: { width: 1520, height: 1080 },
      deviceScaleFactor: 2,
    });
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    await page.addInitScript(
      (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
      { language: 'en', appearance: theme, uiScale: scale },
    );
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
    await page.locator('.filter-chip').first().waitFor();
    await settlePaint(page);
    assert.ok(
      await page
        .locator('.filter-chip')
        .first()
        .evaluate(
          (n) =>
            n.dataset.smoothKind === 'capsule' &&
            decodeURIComponent(n.style.getPropertyValue('--smooth-paint')).includes(' C '),
        ),
    );
    await page.locator('.type-filters').screenshot({ path: path.join(out, `chips-${theme}.png`) });
    await page.getByRole('button', { name: 'Settings', exact: true }).first().click();
    await page.getByRole('button', { name: 'Rendering', exact: true }).click();
    await settlePaint(page);
    const switchTrack = page.locator('.settings-dialog .custom-switch').first();
    const actualSwitch = await switchTrack.evaluate((n) => ({
      kind: n.dataset.smoothKind,
      amount: Number(n.dataset.smoothAmount),
      svg: decodeURIComponent(n.style.getPropertyValue('--smooth-paint')),
      thumb: n.querySelector('span').dataset.smoothKind,
      radius: getComputedStyle(n.querySelector('span')).borderRadius,
    }));
    assert.equal(actualSwitch.kind, 'capsule');
    assert.equal(actualSwitch.amount, 0.382);
    assert.match(actualSwitch.svg, / C /);
    assert.equal(actualSwitch.thumb, undefined);
    assert.equal(actualSwitch.radius, '50%');
    await switchTrack.screenshot({ path: path.join(out, `switch-${theme}.png`) });
    await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('grid');
    await settlePaint(page);
    const search = page.locator('.settings-search');
    const actualSearch = await search.evaluate((n) => ({
      kind: n.dataset.smoothKind,
      amount: Number(n.dataset.smoothAmount),
      svg: decodeURIComponent(n.style.getPropertyValue('--smooth-paint')),
      clearKind: n.querySelector('button').dataset.smoothKind,
      clearRadius: getComputedStyle(n.querySelector('button')).borderRadius,
    }));
    assert.equal(actualSearch.kind, 'capsule');
    assert.equal(actualSearch.amount, 0.382);
    assert.match(actualSearch.svg, / C /);
    assert.equal(actualSearch.clearKind, undefined);
    assert.equal(actualSearch.clearRadius, '50%');
    await search.screenshot({ path: path.join(out, `search-${theme}.png`) });
    await page.evaluate(() => {
      const outer = document.createElement('div');
      outer.id = 'capsule-chain';
      outer.style.cssText =
        'position:fixed;z-index:100000;left:700px;top:400px;width:120px;height:40px;padding:4px;box-sizing:border-box;border-radius:9999px;background:#515f67;outline:1px solid #889990';
      const inner = document.createElement('div');
      inner.style.cssText = 'width:112px;height:32px;border-radius:9999px;background:#9cbf8d';
      outer.append(inner);
      document.querySelector('.settings-dialog').append(outer);
      const vertical = document.querySelector('.filter-chip').cloneNode(true);
      vertical.id = 'vertical-capsule';
      vertical.replaceChildren(vertical.querySelector('.icon'));
      vertical.setAttribute('aria-label', 'Vertical category chip');
      vertical.removeAttribute('data-smooth-surface');
      vertical.style.cssText =
        'position:fixed;z-index:100000;left:850px;top:400px;width:28px;height:120px;min-width:0;padding:0;border-radius:9999px';
      document.querySelector('.settings-dialog').append(vertical);
    });
    await settlePaint(page);
    const nested = await page.locator('#capsule-chain').evaluate((n) =>
      [n, n.firstChild].map((e) => ({
        kind: e.dataset.smoothKind,
        refs: JSON.parse(e.dataset.smoothReferences),
      })),
    );
    assert.equal(nested[0].kind, 'capsule');
    assert.equal(nested[1].kind, 'capsule');
    assert.ok(nested[0].refs.every((r) => r.radius === 20 && r.offset === 0));
    assert.ok(nested[1].refs.every((r) => r.radius === 20 && r.offset === -4));
    assert.ok(
      await page
        .locator('#vertical-capsule')
        .evaluate((n) =>
          decodeURIComponent(n.style.getPropertyValue('--smooth-paint')).includes(' C '),
        ),
    );
    await page
      .locator('#capsule-chain')
      .screenshot({ path: path.join(out, `nested-${theme}.png`) });
    await page
      .locator('#vertical-capsule')
      .screenshot({ path: path.join(out, `vertical-${theme}.png`) });
    assert.deepEqual(errors, []);
    records.push({ theme, scale, switch: actualSwitch.kind, search: actualSearch.kind, nested });
    await page.close();
  }
  await writeFile(path.join(out, 'production.json'), JSON.stringify(records, null, 2));
});
