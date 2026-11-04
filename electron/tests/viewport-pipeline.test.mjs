import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
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
  });
  server.middlewares.use('/__pipeline_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><link rel="icon" href="data:,"><title>GPU parity</title>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__pipeline_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('selection shader closes all viewport borders and preserves explicit light visibility', async () => {
  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(320, 240);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    document.body.append(renderer.domElement);
    const scene = new T.Scene();
    scene.background = new T.Color('#22272d');
    const root = new T.Group();
    scene.add(root);
    const mesh = new T.Mesh(
      new T.BoxGeometry(20, 20, 1),
      new T.MeshStandardMaterial({ color: '#748b92' }),
    );
    mesh.userData = { nodeId: 'box', role: 'visual' };
    root.add(mesh);
    const light = new T.PointLight(0xffffff, 10);
    light.position.z = 3;
    light.userData = { kind: 'light', role: 'visual', nodeId: 'light' };
    light.visible = false;
    root.add(light);
    scene.add(new T.AmbientLight(0xffffff, 1));
    const camera = new T.PerspectiveCamera(42, 320 / 240, 0.01, 100);
    camera.position.z = 5;
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    scene.updateMatrixWorld(true);
    const doc = {
      name: 'test',
      root,
      nodes: [
        { id: 'box', object: mesh },
        { id: 'light', object: light },
      ],
      joints: [],
      files: [],
      warnings: [],
    };
    const pipeline = new ViewportPipeline(renderer);
    pipeline.resize(320, 240);
    pipeline.render(scene, camera, doc, 'box', 'visual');
    const capture = renderer.domElement.toDataURL(),
      ctx = document.createElement('canvas').getContext('2d');
    ctx.canvas.width = 320;
    ctx.canvas.height = 240;
    ctx.drawImage(renderer.domElement, 0, 0);
    const rgba = ctx.getImageData(0, 0, 320, 240).data;
    const orange = (x, y) => {
      const i = (y * 320 + x) * 4;
      return rgba[i] > 235 && rgba[i + 1] > 130 && rgba[i + 1] < 190 && rgba[i + 2] < 80;
    };
    const edges = [orange(0, 120), orange(319, 120), orange(160, 0), orange(160, 239)];
    pipeline.render(scene, camera, doc, 'box', 'collision');
    const hidden = !light.visible;
    const error = renderer.getContext().getError();
    pipeline.dispose();
    renderer.dispose();
    return { edges, hidden, error, capture };
  });
  await mkdir('../output/electron/verification', { recursive: true });
  await writeFile(
    '../output/electron/verification/outline-all-borders.png',
    Buffer.from(result.capture.split(',')[1], 'base64'),
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.edges, [true, true, true, true]);
  assert.equal(result.hidden, true);
  assert.equal(result.error, 0);
});
test('Both composes nearest role surfaces independently of internal overdraw and draw order', async () => {
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(200, 200);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    const scene = new T.Scene(),
      root = new T.Group();
    scene.add(root, new T.AmbientLight(0xffffff, 1));
    const camera = new T.PerspectiveCamera(42, 1, 0.01, 100);
    camera.position.z = 4;
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const make = (role, z, size) => {
      const group = new T.Group();
      group.userData = { role, roleGroup: true };
      const mesh = new T.Mesh(
        new T.BoxGeometry(size, size, 0.2),
        new T.MeshStandardMaterial({ color: role === 'visual' ? '#80a1b2' : '#cc731f' }),
      );
      mesh.userData = { role, nodeId: role };
      mesh.position.z = z;
      group.add(mesh);
      root.add(group);
      return group;
    };
    make('visual', 0, 1.5);
    const outer = make('collision', 0.3, 1.7);
    const doc = { name: 'test', root, nodes: [], joints: [], files: [], warnings: [] },
      pipeline = new ViewportPipeline(renderer);
    pipeline.resize(200, 200);
    const pixel = () => {
      scene.updateMatrixWorld(true);
      pipeline.render(scene, camera, doc, null, 'both');
      const out = new Uint8Array(4);
      renderer
        .getContext()
        .readPixels(
          100,
          100,
          1,
          1,
          renderer.getContext().RGBA,
          renderer.getContext().UNSIGNED_BYTE,
          out,
        );
      return Array.from(out);
    };
    const first = pixel();
    const inner = make('collision', -0.5, 0.6);
    const second = pixel();
    root.remove(outer, inner);
    root.add(inner, outer);
    const third = pixel(),
      error = renderer.getContext().getError();
    pipeline.dispose();
    renderer.dispose();
    return { first, second, third, error };
  });
  assert.deepEqual(result.second, result.first);
  assert.deepEqual(result.third, result.first);
  assert.equal(result.error, 0);
  assert.ok(result.first[0] > 30);
});

test('Default preserves authored role visibility and materials while explicit presets override only role filters', async () => {
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ preserveDrawingBuffer: true });
    renderer.setSize(80, 80);
    const scene = new T.Scene(),
      root = new T.Group(),
      camera = new T.PerspectiveCamera(42, 1, 0.01, 100);
    camera.position.z = 4;
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    scene.add(root);
    function role(name, color, z, visible) {
      const group = new T.Group();
      group.userData = { role: name, roleGroup: true };
      group.visible = visible;
      const mesh = new T.Mesh(new T.BoxGeometry(1, 1, 0.2), new T.MeshBasicMaterial({ color }));
      mesh.position.z = z;
      group.add(mesh);
      root.add(group);
      return group;
    }
    role('visual', '#0000ff', 0, true);
    const collision = role('collision', '#ff0000', 0.3, false),
      helper = new T.GridHelper();
    helper.userData.viewportHelper = true;
    helper.visible = false;
    scene.add(helper);
    const doc = { root, nodes: [], joints: [], files: [], warnings: [] },
      pipeline = new ViewportPipeline(renderer);
    pipeline.resize(80, 80);
    function sample(mode) {
      scene.updateMatrixWorld(true);
      pipeline.render(scene, camera, doc, null, mode);
      const gl = renderer.getContext(),
        out = new Uint8Array(4);
      gl.readPixels(40, 40, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
      return Array.from(out);
    }
    const authored = sample('default'),
      forced = sample('collision'),
      hiddenAfter = !collision.visible;
    collision.visible = true;
    const shown = sample('default'),
      hiddenHelper = !helper.visible;
    pipeline.dispose();
    renderer.dispose();
    return { authored, forced, shown, hiddenAfter, hiddenHelper };
  });
  assert.deepEqual(result.authored, [0, 0, 255, 255]);
  assert.deepEqual(result.shown, [255, 0, 0, 255]);
  assert.notDeepEqual(result.forced, result.shown);
  assert.equal(result.hiddenAfter, true);
  assert.equal(result.hiddenHelper, true);
});
