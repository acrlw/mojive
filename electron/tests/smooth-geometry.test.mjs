import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { parseCorner, normalGapError } from './helpers/corner-measurement.mjs';
const out = path.resolve('../output/electron/verification/nested-smoothing');
let geometry, browser, server;
before(async () => {
  await mkdir(out, { recursive: true });
  const outfile = path.join(out, 'geometry-test.mjs');
  await build({
    entryPoints: ['src/smooth-geometry.ts'],
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    logLevel: 'silent',
  });
  geometry = await import(pathToFileURL(outfile));
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('G2 cubic construction preserves radius, bounds smoothing, and offsets one outer reference', () => {
  const { smoothCorner, offsetCorner, evaluateCubic, maximumCurvature } = geometry;
  for (const smoothing of [0.001, 0.01, 0.25, 0.6, 1]) {
    const base = smoothCorner(16, smoothing, 100);
    assert.ok(Math.abs(evaluateCubic(base.leading[0], 0).curvature) < 1e-12);
    assert.ok(Math.abs(evaluateCubic(base.leading[0], 1).curvature * 16 - 1) < 1e-10);
    assert.equal(base.footprint, 16 * (1 + smoothing));
  }
  assert.deepEqual(geometry.normalizeRadii(2, 16, [0, 2, 2, 0]), [0, 2, 2, 0]);
  assert.deepEqual(
    geometry.normalizeRadii(10, 8, [8, 4, 2, 6]),
    [8, 4, 2, 6].map((r) => (r * 4) / 7),
  );
  assert.deepEqual(geometry.normalizeRadii(10, 8, [0, 0, 0, 0]), [0, 0, 0, 0]);
  const base = smoothCorner(16, 0.6, 100);
  assert.equal(offsetCorner(base, 0), base);
  assert.equal(offsetCorner(smoothCorner(0, 0.6, 0), 4).radius, 0);
  assert.ok(Math.abs(maximumCurvature(base.leading[0]) * 16 - 1.29197308112959) < 1e-11);
  assert.throws(() => offsetCorner(base, -13), /regularity limit/);
  assert.throws(() => offsetCorner(smoothCorner(4, 0, 100), -4), /removes the rounded corner/);
  const middle = offsetCorner(base, -8, 0.005),
    inner = offsetCorner(middle, -4, 0.005);
  assert.equal(inner.reference, base);
  assert.equal(inner.offset, -12);
  assert.equal(inner.radius, 4);
  assert.equal(inner.footprint, 13.600000000000001);
  assert.deepEqual(inner.leading, offsetCorner(base, -12, 0.005).leading);
  const saturated = smoothCorner(4, 1, 4.8);
  assert.ok(Math.abs(saturated.footprint - 4.8) < 1e-12);
  assert.ok(Math.abs(evaluateCubic(saturated.leading[0], 1).curvature * 4 - 1) < 1e-12);
  assert.equal(smoothCorner(4, 1, 4).leading.length, 0);
});

test('adaptive cubic output retains endpoint tangents and curvature, including budget saturation and original arc joins', () => {
  const { smoothCorner, offsetCorner, evaluateCubic, cornerCommands } = geometry;
  for (const smoothing of [0, 0.01, 0.25, 0.6, 0.9, 1])
    for (const [radius, delta, budget] of [
      [4, 4, 10],
      [4, 12, 10],
      [16, -12, 30],
      [4, 2, 4],
      [12, 4, 12],
    ]) {
      const base = smoothCorner(radius, smoothing, budget),
        offset = offsetCorner(base, delta, 0.005);
      assert.ok(offset.sampledError <= 0.005);
      if (offset.leading.length) {
        assert.ok(Math.abs(evaluateCubic(offset.leading[0], 0).curvature) < 4e-6);
        assert.ok(
          Math.abs(evaluateCubic(offset.leading.at(-1), 1).curvature - 1 / offset.radius) < 4e-6,
        );
        assert.ok(
          Math.abs(evaluateCubic(offset.trailing[0], 0).curvature - 1 / offset.radius) < 4e-6,
        );
      }
      const tokens = cornerCommands(offset).match(/[A-Z]|[-+]?\d*\.?\d+(?:e[-+]?\d+)?/gi) ?? [];
      let cursor = { x: -offset.footprint, y: 0 },
        previous = null;
      while (tokens.length) {
        const command = tokens.shift();
        if (command === 'A') {
          const values = tokens.splice(0, 7).map(Number);
          cursor = { x: values[5], y: values[6] };
          previous = null;
          continue;
        }
        assert.equal(command, 'C');
        const v = tokens.splice(0, 6).map(Number),
          curve = [cursor, { x: v[0], y: v[1] }, { x: v[2], y: v[3] }, { x: v[4], y: v[5] }];
        if (previous) {
          const a = evaluateCubic(previous, 1),
            b = evaluateCubic(curve, 0);
          assert.ok(Math.hypot(a.tangent.x - b.tangent.x, a.tangent.y - b.tangent.y) < 1e-7);
          assert.ok(
            Math.abs(a.curvature - b.curvature) < 4e-6,
            `${smoothing}/${radius}/${delta}: ${a.curvature} vs ${b.curvature}`,
          );
        }
        cursor = curve[3];
        previous = curve;
      }
    }
});

test('serialized SVG intersections retain constant normal clearance while independent radii drift', async () => {
  const { smoothCorner, offsetCorner, evaluateCubic, cornerCommands } = geometry;
  const page = await browser.newPage();
  const measurements = [];
  for (const smoothing of [0, 0.01, 0.6, 1])
    for (const [baseRadius, delta] of [
      [4, 0.1],
      [4, 4],
      [4, 12],
      [16, -12],
    ]) {
      const base = smoothCorner(baseRadius, smoothing, 100),
        derived = offsetCorner(base, delta, 0.005),
        independent = smoothCorner(baseRadius + delta, smoothing, 100);
      const points = [];
      if (base.leading.length)
        for (let i = 0; i <= 128; i++) {
          const v = evaluateCubic(base.leading[0], i / 128);
          points.push({ p: v.p, n: { x: v.tangent.y, y: -v.tangent.x } });
        }
      if (base.arc)
        for (let i = 0; i <= 32; i++) {
          const start = Math.atan2(base.arc.start.x + baseRadius, baseRadius - base.arc.start.y),
            end = Math.atan2(base.arc.end.x + baseRadius, baseRadius - base.arc.end.y),
            theta = start + ((end - start) * i) / 32,
            n = { x: Math.sin(theta), y: -Math.cos(theta) };
          points.push({
            p: { x: -baseRadius + baseRadius * n.x, y: baseRadius + baseRadius * n.y },
            n,
          });
        }
      const makePath = (c) =>
        `M ${-c.footprint + delta} ${-delta} ${cornerCommands(c, (p) => ({ x: p.x + delta, y: p.y - delta }))}`;
      const errors = [
        normalGapError(parseCorner(makePath(derived)), points, delta),
        delta > 0 ? normalGapError(parseCorner(makePath(independent)), points, delta) : null,
      ];
      measurements.push({
        smoothing,
        delta,
        cubics: derived.leading.length * 2,
        sampledPositionError: derived.sampledError,
        normalGapError: errors[0],
        independentRadiusGapError: errors[1],
      });
      assert.ok(
        errors[0] < Math.min(0.005, Math.abs(delta) / 100),
        JSON.stringify(measurements.at(-1)),
      );
      if (smoothing === 0.6 && delta === 12) assert.ok(errors[1] > 0.1);
    }
  await writeFile(path.join(out, 'normal-clearance.json'), JSON.stringify(measurements, null, 2));
  await page.close();
});

test('production picker uses one outer reference through all three surfaces and keeps the shell symmetric', async () => {
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'error' });
  await server.listen();
  const page = await browser.newPage({
    viewport: { width: 1100, height: 1000 },
    deviceScaleFactor: 2,
  });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-controls-review.html`,
  );
  await page.waitForFunction(() => window.controlState);
  for (const scale of [1, 1.25, 2]) {
    await page.evaluate((scale) => (document.getElementById('root').style.zoom = scale), scale);
    await page.locator('.color-trigger').click();
    await page.locator('.color-comparison-item').first().hover();
    await page.waitForFunction(
      () => document.querySelector('.color-picker-surface')?.dataset.smoothReferences,
    );
    const profiles = await page.evaluate(() =>
      [
        '.color-picker-surface',
        '.color-comparison-item',
        '.color-comparison-item .color-swatch',
      ].map((selector) => JSON.parse(document.querySelector(selector).dataset.smoothReferences)),
    );
    assert.deepEqual(
      profiles[0].map((c) => [c.radius, c.offset]),
      [
        [16, 0],
        [16, 0],
        [16, 0],
        [16, 0],
      ],
    );
    assert.deepEqual(
      profiles[1].map((c) => [c.radius, c.offset]),
      [
        [16, -8],
        [16, -8],
        [16, -8],
        [16, -8],
      ],
    );
    assert.ok(profiles[2].every((c) => c.radius === 16 && c.offset === -12));
    await page
      .locator('.color-comparison')
      .screenshot({ path: path.join(out, `production-comparison-${scale}.png`) });
    await page
      .locator('.color-picker-surface')
      .screenshot({ path: path.join(out, `production-picker-${scale}.png`) });
    await page.keyboard.press('Escape');
  }
  await page.evaluate(() => {
    const outer = document.createElement('div');
    outer.id = 'deep-corner-chain';
    outer.style.cssText =
      'position:fixed;left:600px;top:40px;width:120px;height:120px;box-sizing:border-box;padding:30px;border-radius:32px;background:#39434a';
    const inner = document.createElement('div');
    inner.style.cssText = 'width:60px;height:60px;border-radius:2px;background:#9cbf8d';
    outer.append(inner);
    document.body.append(outer);
  });
  await page.waitForFunction(
    () => document.querySelector('#deep-corner-chain > div')?.dataset.smoothReferences,
  );
  const deep = await page.evaluate(() =>
    [...document.querySelectorAll('#deep-corner-chain, #deep-corner-chain > div')].map((e) =>
      JSON.parse(e.dataset.smoothReferences),
    ),
  );
  assert.ok(deep[0].every((c) => c.radius === 2 && c.offset === 30));
  assert.ok(deep[1].every((c) => c.radius === 2 && c.offset === 0));
  assert.deepEqual(errors, []);
  await page.close();
});

test('focus outlines retain their signed offset and share nested curves without replacing borders', async () => {
  const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto(
    `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/ui-controls-review.html`,
  );
  await page.locator('.color-trigger').click();
  await page.keyboard.press('Tab');
  const original = page.locator('.color-comparison-item').first();
  await original.focus();
  const settle = () =>
    page.evaluate(
      () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
    );
  await settle();
  const inspect = (locator) =>
    locator.evaluate((el) => {
      const raw = el.style.getPropertyValue('--smooth-paint');
      const svg = new DOMParser().parseFromString(
        decodeURIComponent(raw.slice(raw.indexOf(',') + 1, -2)),
        'image/svg+xml',
      ).documentElement;
      return {
        outline: getComputedStyle(el).outlineStyle,
        focused: el.matches(':focus-visible'),
        references: JSON.parse(el.dataset.smoothReferences),
        viewBox: svg.getAttribute('viewBox').split(' ').map(Number),
        edges: [...svg.querySelectorAll('[data-edge]')].map((edge) => ({
          kind: edge.getAttribute('data-edge'),
          transform: edge.getAttribute('transform'),
          width: Number(edge.getAttribute('stroke-width')),
          path: edge.getAttribute('d'),
        })),
      };
    });
  const nested = await inspect(original);
  assert.ok(nested.focused);
  assert.equal(nested.outline, 'none');
  assert.ok(nested.references.every((ref) => ref.radius === 16 && ref.offset === -8));
  assert.equal(nested.edges.at(-1).transform, 'translate(0.5 0.5)');
  assert.match(nested.edges.at(-1).path, / C /);
  await page
    .locator('.color-picker-surface')
    .screenshot({ path: path.join(out, 'picker-focus.png') });
  await page.keyboard.press('Escape');
  await page.evaluate(async () => {
    await import('/src/joint-panel.css');
    const rail = document.createElement('div');
    rail.id = 'focus-rail';
    rail.className = 'joint-rail';
    rail.tabIndex = 0;
    rail.style.cssText =
      'position:fixed;left:600px;top:40px;width:180px;border:1px solid red;box-shadow:inset 0 0 0 2px blue';
    document.body.append(rail);
  });
  const rail = page.locator('#focus-rail');
  await rail.focus();
  await settle();
  const outside = await inspect(rail);
  assert.equal(outside.outline, 'none');
  assert.deepEqual(
    outside.edges.map((edge) => edge.kind),
    ['inset', 'border', 'outline'],
  );
  assert.equal(outside.edges.at(-1).transform, 'translate(-2.5 -2.5)');
  assert.equal(outside.edges.at(-1).width, 1);
  assert.deepEqual(outside.viewBox.slice(0, 2), [-3, -3]);
  assert.match(outside.edges.at(-1).path, / C /);
  await rail.evaluate((el) => (el.style.width = '300px'));
  await settle();
  const resized = await inspect(rail);
  assert.equal(resized.viewBox[2] - outside.viewBox[2], 120);
  assert.equal(resized.viewBox[3], outside.viewBox[3]);
  assert.deepEqual(resized.references, outside.references);
  await page.locator('button').last().focus();
  await settle();
  assert.deepEqual(
    (await inspect(rail)).edges.map((edge) => edge.kind),
    ['inset', 'border'],
  );
  assert.deepEqual(errors, []);
  await page.close();
});
