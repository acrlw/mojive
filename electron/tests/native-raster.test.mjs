import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: path.resolve('.'),
    server: { host: '127.0.0.1', port: 0 },
    optimizeDeps: { noDiscovery: true, include: [] },
    logLevel: 'error',
  });
  server.middlewares.use('/__raster_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Native raster compositing</title>');
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__raster_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('bounded native raster preserves alpha, clipping and Canvas2D paint order at both DPRs', async () => {
  const results = await page.evaluate(async () => {
    const { NativeRaster } = await import('/src/viewport/viewcube.ts');
    // Keep the original full-frame transport independent of the bounded path.
    // Geometry is shared; its triangulation has separate viewcube/gizmo coverage.
    class FullFrameRaster extends NativeRaster {
      resize(width, height, dpr) {
        super.resize(width, height, dpr);
        const w = Math.round(width * dpr),
          h = Math.round(height * dpr);
        if (this.canvas.width !== w) this.canvas.width = w;
        if (this.canvas.height !== h) this.canvas.height = h;
      }
      flush(context) {
        if (!this.count) return;
        const gl = this.gl;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.disable(gl.SCISSOR_TEST);
        gl.useProgram(this.program);
        gl.uniform2f(this.extent, this.width, this.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertices.subarray(0, this.count));
        gl.drawArrays(gl.TRIANGLES, 0, this.count / 6);
        context.drawImage(this.canvas, 0, 0, this.width, this.height);
        this.reset();
      }
    }
    const circle = (x, y, radius) =>
      Array.from({ length: 36 }, (_, i) => [
        x + Math.cos((i / 36) * Math.PI * 2) * radius,
        y + Math.sin((i / 36) * Math.PI * 2) * radius,
      ]);
    const results = [];
    for (const dpr of [1, 2]) {
      for (const scenario of [
        'overlap-and-text',
        'viewport-clipping',
        'moving-dirty-regions',
        'rotating-dirty-bounds',
        'fractional-dimensions',
      ]) {
        const width = scenario === 'fractional-dimensions' ? 799.25 : 800;
        const height = scenario === 'fractional-dimensions' ? 499.25 : 500;
        const renders = [];
        for (const mode of ['baseline', 'bounded', 'batched']) {
          const raster = mode === 'baseline' ? new FullFrameRaster() : new NativeRaster();
          const canvas = document.createElement('canvas');
          canvas.width = Math.round(width * dpr);
          canvas.height = Math.round(height * dpr);
          const context = canvas.getContext('2d', { willReadFrequently: true });
          context.setTransform(dpr, 0, 0, dpr, 0, 0);
          let copiedPixels = 0,
            copies = 0;
          const capacities = [];
          const drawImage = context.drawImage.bind(context);
          context.drawImage = (source, ...args) => {
            copiedPixels += args.length === 8 ? args[2] * args[3] : source.width * source.height;
            copies++;
            drawImage(source, ...args);
          };
          const paint = (operation) => {
            operation();
            if (mode !== 'batched') raster.flush(context);
          };
          try {
            const frames = ['moving-dirty-regions', 'rotating-dirty-bounds'].includes(scenario) ? 3 : 1;
            for (let frame = 0; frame < frames; frame++) {
              raster.resize(width, height, dpr);
              context.fillStyle = '#212529';
              context.fillRect(0, 0, width, height);
              const dx = frame * 190;
              if (scenario === 'rotating-dirty-bounds') {
                const angle = frame * Math.PI / 4;
                const ring = circle(0, 0, 1).map(([x, y]) => [
                  225 + 90 * x * Math.cos(angle) - 75 * y * Math.sin(angle),
                  190 + 90 * x * Math.sin(angle) + 75 * y * Math.cos(angle),
                ]);
                paint(() => raster.stroke(ring, '#98bc87', 3, 0.8));
                raster.flush(context);
                capacities.push([raster.canvas.width, raster.canvas.height]);
                continue;
              } else if (scenario === 'viewport-clipping') {
                paint(() => raster.fill(circle(-20, 40, 80), '#a0c490', 1, false, 0.4));
                paint(() => raster.stroke(circle(780, 490, 58), '#ffa341', 4, 0.7));
                paint(() =>
                  raster.fill(
                    [
                      [790, 230],
                      [840, 280],
                      [780, 320],
                    ],
                    '#91d6ee',
                    1,
                    false,
                    0.6,
                  ),
                );
                paint(() => raster.fill(circle(-300, -400, 10), '#ff0000'));
              } else {
                paint(() => raster.fill(circle(225 + dx, 190, 74), '#98bc87', 1, false, 0.28));
                paint(() => raster.stroke(circle(225 + dx, 190, 70), '#98bc87', 3, 0.8));
                paint(() =>
                  raster.fill(
                    [
                      [188 + dx, 142],
                      [280 + dx, 205],
                      [202 + dx, 238],
                    ],
                    '#df7586',
                    1,
                    true,
                    0.54,
                  ),
                );
                paint(() =>
                  raster.stroke(
                    [
                      [125 + dx, 190],
                      [325 + dx, 190],
                    ],
                    '#e8e8e8',
                    4,
                    0.68,
                    false,
                  ),
                );
                paint(() =>
                  raster.fan(
                    [[225 + dx, 190], ...circle(225 + dx, 190, 65).slice(0, 9)],
                    '#3e8fcd',
                    0.31,
                  ),
                );
              }
              // Canvas feedback may interrupt a batch. Its background, marker,
              // text and subsequent native geometry must retain this exact order.
              raster.flush(context);
              context.fillStyle = '#2b3038';
              context.fillRect(320 + dx, 235, 175, 30);
              paint(() => raster.fill(circle(333 + dx, 250, 3), '#98bc87'));
              raster.flush(context);
              context.fillStyle = '#e4e6e9';
              context.font = '13px monospace';
              context.fillText('joint +26.7°', 344 + dx, 255);
              paint(() =>
                raster.stroke(
                  [
                    [310 + dx, 240],
                    [340 + dx, 270],
                  ],
                  '#ffa341',
                  2,
                  0.7,
                  false,
                ),
              );
              raster.flush(context);
            }
            renders.push({
              mode,
              data: context.getImageData(0, 0, canvas.width, canvas.height).data,
              image: canvas.toDataURL(),
              copiedPixels,
              copies,
              capacities,
            });
          } finally {
            raster.dispose();
          }
        }
        const baseline = renders[0];
        for (const render of renders.slice(1)) {
          let max = 0,
            total = 0,
            changed = 0;
          for (let i = 0; i < baseline.data.length; i += 4) {
            let pixelMax = 0;
            for (let channel = 0; channel < 4; channel++) {
              const difference = Math.abs(baseline.data[i + channel] - render.data[i + channel]);
              max = Math.max(max, difference);
              pixelMax = Math.max(pixelMax, difference);
              total += difference;
            }
            if (pixelMax) changed++;
          }
          results.push({
            scenario,
            dpr,
            mode: render.mode,
            max,
            mean: total / baseline.data.length,
            changed,
            copiedPixels: render.copiedPixels,
            baselineCopiedPixels: baseline.copiedPixels,
            copies: render.copies,
            baselineCopies: baseline.copies,
            image: render.image,
            capacities: render.capacities,
          });
        }
      }
    }
    return results;
  });
  const directory = path.resolve('../output/electron/verification/native-raster');
  await mkdir(directory, { recursive: true });
  for (const result of results) {
    await writeFile(
      path.join(directory, `${result.scenario}-dpr${result.dpr}-${result.mode}.png`),
      Buffer.from(result.image.split(',')[1], 'base64'),
    );
    delete result.image;
  }
  await writeFile(path.join(directory, 'results.json'), JSON.stringify(results, null, 2));
  for (const result of results) {
    const label = `${result.scenario} DPR${result.dpr} ${result.mode}`;
    // A batch avoids repeated premultiplied 8-bit Canvas2D quantization. Bound
    // the resulting rounding, while rejecting changed edge coverage or paint order.
    assert.ok(
      result.max <= (result.mode === 'bounded' ? 1 : 4),
      `${label}: max difference ${result.max}`,
    );
    assert.ok(result.mean < 0.02, `${label}: mean difference ${result.mean}`);
    if (result.scenario === 'viewport-clipping') assert.equal(result.max, 0, label);
    if (result.scenario === 'rotating-dirty-bounds') {
      assert.deepEqual(result.capacities[1], result.capacities[0], `${label}: retained rotation capacity`);
      assert.deepEqual(result.capacities[2], result.capacities[0], `${label}: retained rotation capacity`);
    }
    if (result.scenario === 'overlap-and-text' && result.mode === 'batched') {
      assert.ok(
        result.copies < result.baselineCopies,
        `${label}: paint boundaries should coalesce copies`,
      );
      assert.ok(
        result.copiedPixels < result.baselineCopiedPixels * 0.1,
        `${label}: localized geometry should not copy a full viewport`,
      );
    }
  }
});
