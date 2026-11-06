import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const files = process.argv.slice(2).map((p) => path.resolve(p));
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
      const mj = await load({ locateFile: () => '/node_modules/@mujoco/mujoco/mujoco.wasm' });
      const dir = file.slice(0, file.lastIndexOf('/') + 1), name = file.slice(file.lastIndexOf('/') + 1);
      const bundle = await loadModelBundle(name, name, async (resource) => {
        const response = await fetch('/@fs' + dir + resource);
        if (!response.ok) throw new Error(resource + ': ' + response.status);
        return new Uint8Array(await response.arrayBuffer());
      });
      const backend = await MujocoBackend.load(mj, bundle);
      const binding = new SceneBinding(backend.source, await backend.frame());
      binding.document.root.traverse((o) => { if (o.userData.defaultVisible === false) o.visible = false; });
      const environment = createViewportScene();
      environment.scene.add(binding.document.root);
      environment.configure(binding.document.root); environment.grid.visible = false;
      const box = new THREE.Box3();
      binding.document.root.updateMatrixWorld(true);
      binding.document.root.traverse((object) => {
        if (object.isMesh && !object.userData.fitExcluded) box.union(new THREE.Box3().setFromObject(object));
      });
      const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length() || 1;
      const camera = new THREE.PerspectiveCamera(42, 4 / 3, .001, Math.max(100, size * 20));
      camera.up.set(0, 0, 1); camera.position.copy(center).add(new THREE.Vector3(1, -1.7, 1).normalize().multiplyScalar(size * 1.3)); camera.lookAt(center);
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      renderer.setSize(1000, 750); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      document.body.appendChild(renderer.domElement); environment.update(camera);
      try { renderer.render(environment.scene, camera); } catch (error) {
        console.error(String(error));
        binding.document.root.traverse((o) => { if(o.isMesh) console.error(o.material.type, o.material.map?.isCubeTexture, String(o.material.onBeforeCompile)); });
      }
      window.review = { backend, binding, renderer, environment, camera };
      return { textures: backend.source.textures.map((t) => ({ name: t.name, type: t.type, width: t.width, height: t.height })), materials: backend.source.geometries.filter((g) => g.material).map((g) => ({ name: g.name, color: g.color, material: g.material })), warnings: backend.source.warnings };
    }, file);
    const filename = path.basename(path.dirname(file)) + '-' + path.basename(file, '.xml') + '.png';
    await page.screenshot({ path: path.join(out, filename) });
    results.push({ file, ...result, errors });
    console.log(file, JSON.stringify({ errors, textures: result.textures.length, warnings: result.warnings }));
    await page.close();
  }
} finally { await browser.close(); await server.close(); }
await writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2));
if (results.some((r) => r.errors.length)) process.exitCode = 1;
