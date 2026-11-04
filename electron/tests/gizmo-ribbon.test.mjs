import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    server: { host: '127.0.0.1', port: 0, watch: null },
    logLevel: 'error',
  });
  server.middlewares.use('/__ribbon_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><style>body{margin:0}canvas{position:absolute;inset:0}</style>',
    );
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 900, height: 600 } });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('Atelier near-full hinge ring retains an open center at front, oblique and edge views', async () => {
  const errors = [],
    results = [];
  page.on('pageerror', (error) => errors.push(error.stack));
  for (const [name, eye] of [
    ['front', [0, 0, 1]],
    ['oblique', [0.55, -0.65, 0.78]],
    ['edge', [0.1, -1, 0.4]],
  ]) {
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__ribbon_test`);
    const result = await page.evaluate(
      async ({ eye }) => {
        const THREE = await import('/node_modules/three/build/three.module.js'),
          { createDemo, setJoint } = await import('/src/scene.ts'),
          { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
          { createViewportScene } = await import('/src/viewport/scene-environment.ts'),
          { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
        const doc = await createDemo(),
          joint = doc.joints.find((j) => j.name === 'base_rotation');
        setJoint(doc, joint.id, (-90.045 * Math.PI) / 180);
        const camera = new THREE.PerspectiveCamera(42, 1.5, 0.01, 100);
        camera.up.set(0, 0, 1);
        camera.position.fromArray(eye);
        camera.lookAt(0, 0, 0.17);
        camera.updateMatrixWorld();
        const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        renderer.setSize(900, 600);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;
        const { scene } = createViewportScene();
        scene.add(doc.root);
        scene.updateMatrixWorld(true);
        const pipeline = new ViewportPipeline(renderer);
        pipeline.resize(900, 600);
        pipeline.render(scene, camera, doc, joint.id, 'default');
        document.body.append(renderer.domElement);
        const canvas = document.createElement('canvas');
        document.body.append(canvas);
        const gizmo = new GizmoOverlay(canvas, { onChange() {}, invalidate() {} });
        gizmo.update(camera, doc, joint.id, 'joint', 'world', 900, 600);
        const image = canvas.getContext('2d').getImageData(0, 0, 900, 600).data;
        let painted = 0,
          total = 0;
        for (let y = 295; y <= 305; y++)
          for (let x = 430; x <= 470; x++) {
            if (Math.abs(x - 450) < 6) continue;
            total++;
            if (image[(y * 900 + x) * 4 + 3] > 128) painted++;
          }
        return {
          eye,
          lower: joint.lower,
          upper: joint.upper,
          value: joint.value,
          painted,
          total,
          ratio: painted / total,
        };
      },
      { eye },
    );
    results.push({ name, ...result });
    await page.screenshot({
      path: `../output/electron/verification/atelier-hinge-${name}-${process.env.RIBBON_CAPTURE ?? 'after'}.png`,
    });
  }
  await writeFile(
    `../output/electron/verification/atelier-hinge-${process.env.RIBBON_CAPTURE ?? 'after'}.json`,
    JSON.stringify(results, null, 2),
  );
  assert.deepEqual(errors, []);
  for (const r of results) assert.ok(r.ratio < 0.2, JSON.stringify(r));
});
