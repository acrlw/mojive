import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
const measurements = [];
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: process.cwd(),
    server: { host: '127.0.0.1', port: 0 },
    optimizeDeps: { noDiscovery: true, include: [] },
    logLevel: 'error',
  });
  server.middlewares.use('/__markers', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><style>body{margin:0;background:#202529;color:white;font:14px sans-serif}canvas{display:inline-block;width:200px;height:200px}</style>',
    );
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 800, height: 700 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__markers`);
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await writeFile(
    '../output/electron/verification/entity-marker-sizes.json',
    JSON.stringify(measurements, null, 2),
  );
  await browser?.close();
  await server?.close();
});
for (const kind of ['camera', 'light'])
  test(`native ${kind} glyph keeps its 24-point master and 13-point pick radius at UI and display scales`, async () => {
    const results = await page.evaluate(async (kind) => {
      const THREE = await import('/node_modules/three/build/three.module.js'),
        { SceneEntityMarkers } = await import('/src/viewport/entities.ts'),
        { createPrimitive } = await import('/src/scene.ts');
      const results = [];
      for (const dpr of [1, 2])
        for (const scale of [1, 1.5, 2]) {
          const node = createPrimitive(kind),
            root = new THREE.Group();
          node.object.position.set(0, 0, 0);
          root.add(node.object);
          root.updateMatrixWorld(true);
          const renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true,
            preserveDrawingBuffer: true,
          });
          renderer.setPixelRatio(dpr);
          renderer.setSize(200, 200, false);
          renderer.setClearColor(0, 0);
          renderer.clear();
          const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
          camera.position.set(0, 0, 5);
          camera.lookAt(0, 0, 0);
          camera.updateMatrixWorld();
          const markers = new SceneEntityMarkers();
          markers.render(renderer, camera, { root, nodes: [node] }, null, 200, 200, scale);
          const gl = renderer.getContext(),
            width = gl.drawingBufferWidth,
            height = gl.drawingBufferHeight,
            pixels = new Uint8Array(width * height * 4);
          gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          let left = width,
            right = -1,
            top = height,
            bottom = -1;
          for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++)
              if (pixels[(y * width + x) * 4 + 3] > 8) {
                left = Math.min(left, x);
                right = Math.max(right, x);
                top = Math.min(top, y);
                bottom = Math.max(bottom, y);
              }
          results.push({
            kind,
            dpr,
            scale,
            pixels: [right - left + 1, bottom - top + 1],
            inside: markers.pick([100 + 13 * scale - 1e-6, 100]) === node.id,
            boundary: markers.pick([100 + 13 * scale, 100]) === node.id,
            outside: markers.pick([100 + 13 * scale + 1e-6, 100]) === null,
          });
          if (dpr === 2) {
            const image = document.createElement('img');
            image.src = renderer.domElement.toDataURL();
            image.style.cssText = 'width:200px;height:200px';
            image.alt = `${kind} · ${scale * 100}%`;
            const card = document.createElement('div');
            card.style.cssText = 'display:inline-flex;flex-direction:column;text-align:center';
            card.append(image, document.createTextNode(`${kind} · ${scale * 100}%`));
            document.body.append(card);
          }
          markers.dispose();
          renderer.dispose();
        }
      return results;
    }, kind);
    // Independent production scene_icon_geometry bounds on its 24-unit grid.
    const native = kind === 'camera' ? [21.5, 17.5] : [13.5, 21.5];
    for (const result of results) {
      for (let i = 0; i < 2; i++)
        assert.ok(
          Math.abs(result.pixels[i] - native[i] * result.scale * result.dpr) <= 2,
          JSON.stringify(result),
        );
      assert.ok(result.inside && result.boundary && result.outside, JSON.stringify(result));
      measurements.push(result);
    }
  });
test('native camera/light glyph scale comparison capture', async () => {
  await page.screenshot({ path: '../output/electron/verification/entity-marker-scales.png' });
});
