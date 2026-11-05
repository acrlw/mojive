import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
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
  server.middlewares.use('/__feedback_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><link rel="stylesheet" href="/src/design-tokens.css"><link rel="stylesheet" href="/src/styles.css"><style>body{margin:0;background:#22272d}canvas{display:block}</style><title>Native active feedback</title>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 900, height: 600 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__feedback_test`);
  await page.evaluate(async () => {
    await document.fonts.load('16px "JetBrains Mono"');
    await document.fonts.ready;
  });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function feedback(kind) {
  return page.evaluate(async (kind) => {
    const THREE = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { worldScale, project, frontArc } = await import('/src/viewport/geometry.ts'),
      { createPrimitive } = await import('/src/scene.ts');
    document.querySelectorAll('canvas').forEach((c) => c.remove());
    const node = createPrimitive('box', 'Test box'),
      doc = {
        name: 'test',
        root: new THREE.Group(),
        nodes: [node],
        joints: [],
        files: [],
        warnings: [],
      };
    doc.root.add(node.object);
    node.object.position.set(0, 0, 0);
    doc.root.updateMatrixWorld(true);
    const canvas = document.createElement('canvas');
    document.body.append(canvas);
    const labels = [],
      ctx = canvas.getContext('2d'),
      fillText = ctx.fillText.bind(ctx);
    ctx.fillText = (text, ...args) => {
      labels.push(text);
      fillText(text, ...args);
    };
    const gizmo = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => {} }),
      camera = new THREE.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -4, kind === 'edge' ? 0.6 : 2.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const origin = new THREE.Vector3(),
      scale = worldScale(origin, camera, 600),
      center = project(origin, camera, 900, 600),
      tip = project(new THREE.Vector3(scale, 0, 0), camera, 900, 600),
      delta = [tip[0] - center[0], tip[1] - center[1]],
      length = Math.hypot(...delta);
    let start = [center[0] + delta[0] * 0.75, center[1] + delta[1] * 0.75],
      end = [start[0] + (delta[0] / length) * 76, start[1] + (delta[1] / length) * 76];
    const trackball = kind.startsWith('trackball'),
      rotation = trackball || ['rotate', 'edge'].includes(kind);
    if (trackball) {
      const theme = (await (await fetch('/src/locales/themes.json')).json())[
        kind.endsWith('light') ? 'light' : 'dark'
      ];
      gizmo.configure({
        readoutStyle: {
          font: '"JetBrains Mono","MiSans",monospace',
          size: 16,
          surface: theme.floating,
          ink: theme.text,
        },
      });
    }
    gizmo.update(camera, doc, node.id, rotation ? 'rotation' : 'position', 'world', 900, 600);
    if (rotation) {
      const path = frontArc(origin, new THREE.Vector3(0, 0, 1), scale * 0.78, camera, 900, 600);
      start = path[8];
      end = path[24];
    }
    if (trackball) {
      start = [center[0], center[1]];
      end = [center[0] - 90, center[1] - 19];
    }
    const acquired = gizmo.begin(start);
    gizmo.move(end, false, !trackball && kind !== 'move');
    const pixels = ctx.getImageData(0, 0, 900, 600).data;
    let ink = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) ink++;
    const composite = document.createElement('canvas');
    composite.width = 900;
    composite.height = 600;
    const output = composite.getContext('2d');
    output.fillStyle = '#22272d';
    output.fillRect(0, 0, 900, 600);
    output.drawImage(canvas, 0, 0);
    const result = {
      acquired,
      ink,
      labels,
      image: composite.toDataURL(),
      value: node.object.position.x,
      disk: Array.from({ length: 8 }, (_, i) => {
        const angle = (i * Math.PI) / 4;
        const x = Math.round(center[0] + Math.cos(angle) * 14),
          y = Math.round(center[1] + Math.sin(angle) * 14);
        return pixels[(y * 900 + x) * 4 + 3];
      }),
    };
    gizmo.end(true);
    gizmo.dispose();
    return result;
  }, kind);
}
for (const kind of ['move', 'snap', 'rotate', 'edge'])
  test(`native active ${kind} feedback paints held-input geometry and source readout`, async () => {
    const result = await feedback(kind);
    assert.equal(result.acquired, true);
    assert.ok(result.ink > 500);
    const label = result.labels.at(-1);
    if (kind === 'move') assert.match(label, /X \+\d+\.\d{3} m$/);
    else assert.match(label, / · SNAP (?:5°|0.1 m)$/);
    assert.ok(!label.includes('rad'));
    await mkdir('../output/electron/verification', { recursive: true });
    await writeFile(
      `../output/electron/verification/active-${kind}.png`,
      Buffer.from(result.image.split(',')[1], 'base64'),
    );
  });
test('translation connector preserves production shader equations verbatim', async () => {
  const original = await readFile('../python/render/opengl/shaders/debug_drag_link.frag', 'utf8');
  const port = await page.evaluate(
    async () => (await import('/src/viewport/drag-link-shader.ts')).dragLinkFragment,
  );
  const equations = (s) => s.slice(s.indexOf('float capsule_sdf'));
  assert.equal(equations(port).trim(), equations(original).trim());
});

for (const theme of ['light', 'dark'])
  test(`trackball has complete native64segment disk and readable ${theme} feedback`, async () => {
    const result = await feedback('trackball-' + theme);
    assert.equal(result.acquired, true);
    assert.ok(
      result.disk.every((alpha) => alpha >= 50),
      `Full disk coverage in all octants: ${result.disk}`,
    );
    assert.equal(result.labels.at(-1), 'Trackball -10.9° -51.6°');
    await writeFile(
      `../output/electron/verification/trackball-${theme}.png`,
      Buffer.from(result.image.split(',')[1], 'base64'),
    );
  });
