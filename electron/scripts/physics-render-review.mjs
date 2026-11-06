import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { routeReviewResources } from './review-resources.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const diagnostics = process.argv.includes('--diagnostics');
const files = process.argv.slice(2).filter((p) => p !== '--diagnostics').map((p) => path.resolve(p));
const out = path.resolve(root, '../output/electron/physics-render');
await mkdir(out, { recursive: true });
const server = await createServer({ root, configFile: false, server: { host: '127.0.0.1', port: 0, hmr: false, watch: null, fs: { allow: [root, ...files.map(path.dirname)] } } });
server.middlewares.use('/__review', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<html><body style="margin:0"></body></html>'); });
await server.listen();
const browser = await chromium.launch({ headless: true });
const results = [];
try {
  for (const file of files) {
    const page = await browser.newPage({ viewport: { width: 1000, height: 750 } });
    await routeReviewResources(page, path.dirname(file));
    const errors = [];
    await page.route('**/__review', (route) => route.fulfill({ contentType: 'text/html', body: '<html><body style="margin:0"></body></html>' }));
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (e) => { if (e.type() === 'error') errors.push(e.text()); });
    await page.goto(server.resolvedUrls.local[0] + '__review');
    const result = await page.evaluate(async (file) => {
      const THREE = await import('/node_modules/three/build/three.module.js');
      const { default: load } = await import('/node_modules/@mujoco/mujoco/mujoco.js');
      const { loadModelBundle } = await import('/src/resources/loadModelBundle.ts');
      const { MujocoBackend } = await import('/src/physics/backends/mujoco/MujocoBackend.ts');
      const { SceneBinding } = await import('/src/physics/SceneBinding.ts');
      const { createViewportScene } = await import('/src/viewport/scene-environment.ts');
      const { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
      const { sceneFocusBounds } = await import('/src/viewport/focus-bounds.ts');
      const { CameraFocusTransition } = await import('/src/viewport/camera-focus.ts');
      const mj = await load({ locateFile: () => '/node_modules/@mujoco/mujoco/mujoco.wasm' });
      const name = file.slice(file.lastIndexOf('/') + 1);
      const bundle = await loadModelBundle(name, name, async (resource) => {
        const response = await fetch('/__model_resource?path=' + encodeURIComponent(resource));
        if (!response.ok) throw new Error(resource + ': ' + response.status);
        return new Uint8Array(await response.arrayBuffer());
      });
      const backend = await MujocoBackend.load(mj, bundle);
      const binding = new SceneBinding(backend.source, await backend.frame());
      const environment = createViewportScene();
      environment.scene.add(binding.document.root);
      environment.configure(binding.document.root); environment.grid.visible = false;
      // Use the editor's Frame All bounds and optics instead of a separate near-plane heuristic.
      const bounds = sceneFocusBounds(binding.document);
      const center = new THREE.Vector3().fromArray(bounds.center);
      const start = {
        projection: 'perspective', fov: 42, near: .01, far: 1000,
        orthoHeight: 4, sensorWidth: 35, focalLength: 35, resolution: [1000, 750],
        position: center.clone().add(new THREE.Vector3(1, -1.7, 1)).toArray(),
        target: bounds.center, up: [0, 0, 1],
      };
      const view = new CameraFocusTransition(start, bounds, { frameScene: true, duration: 0 }).end;
      const camera = new THREE.PerspectiveCamera(view.fov, 4 / 3, view.near, view.far);
      camera.up.fromArray(view.up); camera.position.fromArray(view.position); camera.lookAt(center);
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      renderer.setSize(1000, 750); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.1;
      const pipeline = new ViewportPipeline(renderer); pipeline.resize(1000, 750);
      document.body.appendChild(renderer.domElement); environment.update(camera);
      const render = (mode = 'default') => pipeline.render(environment.scene, camera, binding.document, null, mode);
      render();
      window.review = { backend, binding, renderer, environment, camera, pipeline, render };
      return { pipeline: 'ViewportPipeline', view, exposure: renderer.toneMappingExposure,
        textures: backend.source.textures.map((t) => ({ name: t.name, type: t.type, width: t.width, height: t.height })), materials: backend.source.geometries.filter((g) => g.material).map((g) => ({ name: g.name, color: g.color, material: g.material })), warnings: backend.source.warnings };
    }, file);
    const filename = path.basename(path.dirname(file)) + '-' + path.basename(file, '.xml') + '.png';
    await page.screenshot({ path: path.join(out, filename) });
    if (diagnostics) {
      await page.evaluate(() => {
        const { environment, render } = window.review;
        environment.scene.traverse((object) => {
          if (object.isLight) { object.userData.reviewCastShadow = object.castShadow; object.castShadow = false; }
        });
        render();
      });
      await page.screenshot({ path: path.join(out, filename.replace('.png', '-no-shadows.png')) });
      await page.evaluate(() => {
        const { environment, render } = window.review;
        environment.scene.traverse((object) => {
          if (object.isLight) object.castShadow = object.userData.reviewCastShadow;
        });
        render('visual');
      });
      await page.screenshot({ path: path.join(out, filename.replace('.png', '-visual.png')) });
    }
    results.push({ file, ...result, errors });
    console.log(file, JSON.stringify({ errors, textures: result.textures.length, warnings: result.warnings }));
    await page.close();
  }
} finally { await browser.close(); await server.close(); }
await writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2));
if (results.some((r) => r.errors.length)) process.exitCode = 1;
