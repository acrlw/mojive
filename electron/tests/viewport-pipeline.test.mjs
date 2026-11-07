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
test('unlit passes preserve moving shadows, selection and helper occlusion without redrawing shadow maps', async () => {
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js');
    const { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(320, 240);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    const scene = new T.Scene(), root = new T.Group(), selected = new T.Group();
    scene.background = new T.Color('#22272d');
    scene.add(root, new T.AmbientLight(0xffffff, 0.3));
    root.add(selected);
    const light = new T.DirectionalLight(0xffffff, 3);
    light.position.set(2, 5, 3);
    light.castShadow = true;
    light.shadow.mapSize.set(256, 256);
    scene.add(light);
    const floor = new T.Mesh(new T.BoxGeometry(8, 0.1, 8), new T.MeshStandardMaterial({ color: '#70898a' }));
    floor.position.y = -0.7;
    floor.receiveShadow = true;
    root.add(floor);
    let shadowDraws = 0;
    for (const [role, x] of [['visual', -0.2], ['collision', 0.2]]) {
      const mesh = new T.Mesh(new T.BoxGeometry(0.8, 1, 0.7), new T.MeshStandardMaterial({ color: '#b79d66' }));
      mesh.userData.role = role;
      mesh.position.x = x;
      mesh.castShadow = true;
      mesh.onBeforeShadow = () => shadowDraws++;
      selected.add(mesh);
    }
    const helpers = new T.Scene();
    const line = new T.Line(new T.BufferGeometry().setFromPoints([
      new T.Vector3(-2, 0, 0), new T.Vector3(2, 0, 0),
    ]), new T.LineBasicMaterial({ color: '#ca8b8b', depthTest: true }));
    helpers.add(line);
    const camera = new T.PerspectiveCamera(42, 4 / 3, 0.1, 100);
    camera.position.set(3, 2, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const doc = { root, nodes: [{ id: 'body', kind: 'link', object: selected }], joints: [] };
    const pipeline = new ViewportPipeline(renderer);
    pipeline.resize(320, 240);
    const unlit = pipeline.renderUnlit;
    const gl = renderer.getContext(), actual = new Uint8Array(320 * 240 * 4), reference = actual.slice();
    const cases = [];
    let actualPng, referencePng;
    for (const mode of ['visual', 'collision', 'both']) for (const helper of [false, true]) {
      for (const x of [-0.8, 0.5]) {
        selected.position.x = x;
        scene.updateMatrixWorld(true);
        const renderHelpers = () => {
          if (!helper) return;
          renderer.autoClear = false;
          renderer.render(helpers, camera);
        };
        shadowDraws = 0;
        pipeline.renderUnlit = unlit;
        pipeline.render(scene, camera, doc, 'body', mode, { renderHelpers: helper ? renderHelpers : undefined });
        const optimizedShadows = shadowDraws;
        gl.readPixels(0, 0, 320, 240, gl.RGBA, gl.UNSIGNED_BYTE, actual);
        actualPng = renderer.domElement.toDataURL();
        shadowDraws = 0;
        pipeline.renderUnlit = (s, c) => renderer.render(s, c);
        // The former caller requested union depth even when its helper draw was empty.
        pipeline.render(scene, camera, doc, 'body', mode, { renderHelpers });
        gl.readPixels(0, 0, 320, 240, gl.RGBA, gl.UNSIGNED_BYTE, reference);
        referencePng = renderer.domElement.toDataURL();
        let different = 0;
        for (let i = 0; i < actual.length; i++) different += Number(actual[i] !== reference[i]);
        cases.push({ mode, helper, x, different, optimizedShadows, referenceShadows: shadowDraws });
      }
    }
    pipeline.renderUnlit = unlit;
    // A manually scheduled shadow refresh must survive an unlit pass, including failure.
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = true;
    const render = renderer.render;
    renderer.render = () => { throw new Error('Interrupted render'); };
    try { pipeline.renderUnlit(scene, camera); } catch {}
    renderer.render = render;
    const restored = !renderer.shadowMap.autoUpdate && renderer.shadowMap.needsUpdate;
    const error = gl.getError();
    pipeline.dispose();
    renderer.dispose();
    return { cases, restored, error, actualPng, referencePng };
  });
  const out = path.resolve('../output/electron/shadow-pass-parity');
  await mkdir(out, { recursive: true });
  for (const name of ['actual', 'reference'])
    await writeFile(path.join(out, `${name}.png`), Buffer.from(result[`${name}Png`].split(',')[1], 'base64'));
  assert.equal(result.error, 0);
  assert.equal(result.restored, true);
  for (const item of result.cases) {
    assert.equal(item.different, 0, JSON.stringify(item));
    assert.equal(item.optimizedShadows, item.mode === 'both' ? 2 : 1);
    assert.ok(item.referenceShadows > item.optimizedShadows, JSON.stringify(item));
  }
});
test('bounded selection outline matches the full-frame shader at every density and viewport edge', async () => {
  const result = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js');
    const { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    const scene = new T.Scene();
    scene.background = new T.Color('#22272d');
    scene.add(new T.AmbientLight(0xffffff, 1));
    const root = new T.Group(),
      selected = new T.Group();
    scene.add(root);
    root.add(selected);
    const geometry = new T.BoxGeometry(0.8, 0.9, 0.5);
    geometry.computeBoundingBox();
    const visual = new T.Mesh(geometry, new T.MeshStandardMaterial({ color: '#748b92' }));
    const collision = new T.Mesh(geometry, new T.MeshStandardMaterial({ color: '#6f9b55' }));
    visual.userData.role = 'visual';
    collision.userData.role = 'collision';
    collision.position.x = 0.2;
    selected.add(visual, collision);
    const doc = { root, nodes: [{ id: 'body', kind: 'link', object: selected }], joints: [] };
    const pipeline = new ViewportPipeline(renderer);
    const bounded = pipeline.selectedScissor;
    const failures = [];
    let mismatches = 0,
      cases = 0,
      restricted = 0,
      fallback = 0;
    const gl = renderer.getContext();
    for (const density of [1, 1.25, 2]) {
      renderer.setPixelRatio(density);
      renderer.setSize(160, 120);
      const width = renderer.domElement.width,
        height = renderer.domElement.height;
      pipeline.resize(width, height);
      const actual = new Uint8Array(width * height * 4),
        reference = actual.slice();
      const perspective = new T.PerspectiveCamera(42, 4 / 3, 0.1, 100);
      const ortho = new T.OrthographicCamera(-2, 2, 1.5, -1.5, 0.1, 100);
      for (const camera of [perspective, ortho]) {
        camera.position.z = 5;
        camera.updateMatrixWorld();
        for (const outlineWidth of [1, 3, 8]) {
          for (const mode of ['default', 'visual', 'collision', 'both']) {
            // Motion also checks that old mask pixels cannot leak outside the new bounds.
            for (const [x, y, z] of [
              [0, 0, 0],
              [-2, 0, 0],
              [2, 0, 0],
              [0, -1.5, 0],
              [0, 1.5, 0],
              [0, 0, 4.9],
            ]) {
              selected.position.set(x, y, z);
              selected.rotation.set(0.2, 0.15, 0.4);
              scene.updateMatrixWorld(true);
              // Exercise composition into an existing viewport scissor as well.
              renderer.setScissor(4, 3, 152, 114);
              renderer.setScissorTest(false);
              renderer.clear();
              const clipped = cases % 2 === 1;
              renderer.setScissorTest(clipped);
              pipeline.selectedScissor = function (...args) {
                const bounds = bounded.apply(this, args);
                if (bounds && bounds.z * bounds.w < (width * height) / 2) restricted++;
                if (!bounds) fallback++;
                return bounds;
              };
              pipeline.render(scene, camera, doc, 'body', mode, { outlineWidth });
              gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, actual);
              const scissor = renderer.getScissor(new T.Vector4()).toArray();
              if (scissor.join() !== '4,3,152,114' || renderer.getScissorTest() !== clipped)
                throw new Error('Outline did not restore renderer scissor state');
              pipeline.selectedScissor = () => null;
              pipeline.render(scene, camera, doc, 'body', mode, { outlineWidth });
              gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, reference);
              let different = 0, maxDelta = 0, first;
              for (let i = 0; i < actual.length; i++) if (actual[i] !== reference[i]) {
                different++;
                maxDelta = Math.max(maxDelta, Math.abs(actual[i] - reference[i]));
                first ??= [Math.floor(i / 4) % width, Math.floor(i / 4 / width), actual[i], reference[i]];
              }
              mismatches += different;
              if (different && failures.length < 12) failures.push({density, camera:camera.type, outlineWidth, mode, x,y,z,clipped,different,maxDelta,first});
              cases++;
            }
          }
        }
      }
    }
    const error = gl.getError();
    pipeline.dispose();
    geometry.dispose();
    visual.material.dispose();
    collision.material.dispose();
    renderer.dispose();
    return { mismatches, cases, restricted, fallback, error, failures };
  });
  assert.equal(result.cases, 432);
  assert.equal(result.mismatches, 0, JSON.stringify(result));
  assert.ok(result.restricted > 0, 'Normal selections must restrict the outline draw');
  assert.ok(result.fallback > 0, 'Near-plane intersections must use the full-frame path');
  assert.equal(result.error, 0);
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
