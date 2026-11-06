import { createServer } from 'vite';
import { chromium } from 'playwright';
import { readdir, mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const roots = process.argv.slice(2).map((p) => path.resolve(p));
if (!roots.length) throw new Error('Pass one or more model directories.');
const out = path.resolve(root, '../output/electron/physics-corpus');
await mkdir(out, { recursive: true });
const files = [];
async function discover(dir, base) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await discover(full, base);
    else if (/\.(xml|mjcf|urdf)$/i.test(entry.name)) files.push({ base, entry: path.relative(base, full) });
  }
}
for (const base of roots) await discover(base, base);
files.sort((a, b) => a.entry.localeCompare(b.entry));
const server = await createServer({ root, configFile: false, server: { host: '127.0.0.1', port: 0, hmr: false, watch: null,
  fs: { allow: [root, ...roots] } } });
server.middlewares.use('/__review', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<html><body></body></html>'); });
await server.listen();
const browser = await chromium.launch({ headless: true });
const results = process.env.CORPUS_RESUME ? JSON.parse(await readFile(path.join(out, 'results.json'), 'utf8')) : [];
const pending = files.filter((item) => !results.some((r) => r.base === item.base && r.entry === item.entry));
try {
  // Each model gets an isolated WASM heap, including models near the memory limit.
  for (let start = 0; start < pending.length; start += 1) {
    const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
    await page.route('**/__review', (route) => route.fulfill({ contentType: 'text/html', body: '<html><body></body></html>' }));
    await page.goto(server.resolvedUrls.local[0] + '__review');
    await page.evaluate(async () => {
      const { default: load } = await import('/node_modules/@mujoco/mujoco/mujoco.js');
      window.mj = await load({ locateFile: () => '/node_modules/@mujoco/mujoco/mujoco.wasm' });
    });
    for (const item of pending.slice(start, start + 1)) {
      const result = await page.evaluate(async ({ base, entry }) => {
        let backend;
        try {
          const { loadModelBundle } = await import('/src/resources/loadModelBundle.ts');
          const { MujocoBackend } = await import('/src/physics/backends/mujoco/MujocoBackend.ts');
          const { SceneBinding } = await import('/src/physics/SceneBinding.ts');
          const bundle = await loadModelBundle(entry, entry, async (name) => {
            const response = await fetch('/@fs' + base + '/' + name.split('/').map(encodeURIComponent).join('/'));
            if (!response.ok) throw new Error('Resource ' + name + ': ' + response.status);
            return new Uint8Array(await response.arrayBuffer());
          });
          const docs = bundle.files.filter((f) => /\.(xml|mjcf|urdf)$/i.test(f.path)).map((f) =>
            new DOMParser().parseFromString(new TextDecoder().decode(f.bytes), 'application/xml'));
          if (!['mujoco', 'robot'].includes(docs[0].documentElement.tagName) ||
            (docs[0].documentElement.tagName === 'mujoco' && !docs.some((d) => d.querySelector('worldbody')))) return { status: 'fragment' };
          const plugins = docs.flatMap((d) => [...d.querySelectorAll('plugin[plugin], geom[type="sdf"]')].map((e) => e.getAttribute('plugin') || 'sdf'));
          if (plugins.length) return { status: 'plugin', plugins: [...new Set(plugins)] };
          backend = await MujocoBackend.load(window.mj, bundle);
          const initial = await backend.frame(), binding = new SceneBinding(backend.source, initial);
          const snapshot = await backend.capture();
          await backend.execute({ type: 'step', count: 2 });
          await backend.execute({ type: 'restore', snapshot });
          const restored = await backend.frame();
          if (restored.bodyTransforms.some((v, i) => Math.abs(v - initial.bodyTransforms[i]) > 1e-9)) throw new Error('State replay mismatch');
          const source = backend.source;
          return { status: source.geometries.length ? 'passed' : 'empty', files: bundle.files.length,
            geometries: source.geometries.length, textures: source.textures.length, keys: source.keyframes.length,
            warnings: source.warnings, nodes: binding.document.nodes.length };
        } catch (error) { return { status: 'failed', message: String(error) }; }
        finally { backend?.dispose(); }
      }, item);
      results.push({ ...item, ...result });
      console.log(`${results.length}/${files.length} ${result.status} ${item.entry}${result.message ? ': ' + result.message.slice(0, 200) : ''}`);
      await writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2));
    }
    await page.close();
  }
} finally { await browser.close(); await server.close(); }
const counts = Object.fromEntries([...new Set(results.map((r) => r.status))].map((s) => [s, results.filter((r) => r.status === s).length]));
console.log(JSON.stringify(counts));
process.exitCode = results.some((r) => r.status === 'failed') ? 1 : 0;
