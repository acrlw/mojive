import { createServer } from 'vite';
import { chromium } from 'playwright';
import { readdir, mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { routeReviewResources } from './review-resources.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const roots = process.argv.slice(2).map((p) => path.resolve(p));
if (!roots.length) throw new Error('Pass one or more model directories.');
const out = path.resolve(process.env.CORPUS_OUT || path.join(root, '../output/electron/physics-corpus'));
if (process.env.CORPUS_RESUME) throw new Error('Run a fresh corpus review; results from different builds must not be combined.');
await mkdir(out, { recursive: true });
const files = [];
async function discover(dir, base) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await discover(full, base);
    else if (/\.(xml|mjcf|urdf)$/i.test(entry.name) && (!process.env.CORPUS_FILTER || full.includes(process.env.CORPUS_FILTER))) {
      files.push({ base, entry: path.relative(base, full) });
    }
  }
}
for (const base of roots) await discover(base, base);
files.sort((a, b) => a.entry.localeCompare(b.entry));
const metadata = {
  startedAt: new Date().toISOString(), roots, filter: process.env.CORPUS_FILTER || null,
  revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
  sourceHash: '', runtimeVersion: '', models: files.length,
  checks: ['resource resolution', 'compile', 'scene binding', 'step', 'full-state restore', 'deterministic continuation'],
  limitations: 'Compilation and state checks do not certify rendered appearance or every desktop timeline interaction.',
};
const hash = createHash('sha256');
async function hashSources(dir) {
  for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await hashSources(full);
    else hash.update(path.relative(root, full)).update(await readFile(full));
  }
}
await hashSources(path.join(root, 'src'));
hash.update(await readFile(fileURLToPath(import.meta.url))).update(await readFile(path.join(root, 'package-lock.json')));
hash.update(await readFile(new URL('./review-resources.mjs', import.meta.url)));
metadata.sourceHash = hash.digest('hex');
await writeFile(path.join(out, 'metadata.json'), JSON.stringify(metadata, null, 2));
const server = await createServer({ root, configFile: false, server: { host: '127.0.0.1', port: 0, hmr: false, watch: null,
  fs: { allow: [root, ...roots] } } });
server.middlewares.use('/__review', (_req, res) => { res.setHeader('Content-Type', 'text/html'); res.end('<html><body></body></html>'); });
await server.listen();
const browser = await chromium.launch({ headless: true });
const results = [];
const pending = files;
try {
  // Each model gets an isolated WASM heap, including models near the memory limit.
  for (let start = 0; start < pending.length; start += 1) {
    const page = await browser.newPage({ viewport: { width: 900, height: 640 } });
    await routeReviewResources(page, pending[start].base);
    await page.route('**/__review', (route) => route.fulfill({ contentType: 'text/html', body: '<html><body></body></html>' }));
    await page.goto(server.resolvedUrls.local[0] + '__review');
    metadata.runtimeVersion = await page.evaluate(async () => {
      const { default: load } = await import('/node_modules/@mujoco/mujoco/mujoco.js');
      window.mj = await load({ locateFile: () => '/node_modules/@mujoco/mujoco/mujoco.wasm' });
      return window.mj.mj_versionString();
    });
    for (const item of pending.slice(start, start + 1)) {
      const result = await page.evaluate(async ({ entry }) => {
        let backend, phase = 'resources', bundleId;
        try {
          const { loadModelBundle } = await import('/src/resources/loadModelBundle.ts');
          const { MujocoBackend } = await import('/src/physics/backends/mujoco/MujocoBackend.ts');
          const { SceneBinding } = await import('/src/physics/SceneBinding.ts');
          const bundle = await loadModelBundle(entry, entry, async (name) => {
            const response = await fetch('/__model_resource?path=' + encodeURIComponent(name));
            if (!response.ok) throw new Error('Resource ' + name + ': ' + response.status);
            return new Uint8Array(await response.arrayBuffer());
          });
          bundleId = bundle.id;
          phase = 'entry';
          const docs = bundle.files.filter((f) => /\.(xml|mjcf|urdf)$/i.test(f.path)).map((f) =>
            new DOMParser().parseFromString(new TextDecoder().decode(f.bytes), 'application/xml'));
          const entryFile = bundle.files.find((f) => f.path === bundle.entry);
          const entryRoot = new DOMParser().parseFromString(new TextDecoder().decode(entryFile.bytes), 'application/xml').documentElement.tagName;
          if (!['mujoco', 'robot'].includes(entryRoot) ||
            (entryRoot === 'mujoco' && !docs.some((d) => d.querySelector('worldbody')))) return { status: 'fragment', bundleId };
          const plugins = docs.flatMap((d) => [...d.querySelectorAll('plugin[plugin], geom[type="sdf"]')].map((e) => e.getAttribute('plugin') || 'sdf'));
          if (plugins.length) return { status: 'plugin', plugins: [...new Set(plugins)], bundleId };
          phase = 'compile';
          backend = await MujocoBackend.load(window.mj, bundle);
          phase = 'scene';
          const initial = await backend.frame(), binding = new SceneBinding(backend.source, initial);
          const snapshot = await backend.capture();
          phase = 'step';
          await backend.execute({ type: 'step', count: 2 });
          const expected = await backend.capture();
          if (expected.state.some((v) => !Number.isFinite(v))) throw new Error('Non-finite simulation state');
          phase = 'restore';
          await backend.execute({ type: 'restore', snapshot });
          if (JSON.stringify(await backend.capture()) !== JSON.stringify(snapshot)) throw new Error('Full-state replay mismatch');
          const restored = await backend.frame();
          if (restored.bodyTransforms.some((v, i) => Math.abs(v - initial.bodyTransforms[i]) > 1e-9)) throw new Error('State replay mismatch');
          phase = 'continuation';
          await backend.execute({ type: 'step', count: 2 });
          if (JSON.stringify(await backend.capture()) !== JSON.stringify(expected)) throw new Error('State continuation mismatch');
          const source = backend.source;
          return { status: source.geometries.length ? 'passed' : 'empty', bundleId, files: bundle.files.length,
            geometries: source.geometries.length, textures: source.textures.length, keys: source.keyframes.length,
            warnings: source.warnings, nodes: binding.document.nodes.length };
        } catch (error) { return { status: 'failed', phase, bundleId, code: error.code, message: String(error),
          cause: error.cause ? String(error.cause) : undefined }; }
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
await writeFile(path.join(out, 'metadata.json'), JSON.stringify({ ...metadata, finishedAt: new Date().toISOString(), counts }, null, 2));
console.log(JSON.stringify(counts));
process.exitCode = results.some((r) => r.status === 'failed') ? 1 : 0;
