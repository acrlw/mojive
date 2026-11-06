import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import loadMujoco from '@mujoco/mujoco';

const encode = (text) => Array.from(new TextEncoder().encode(text));
function tetrahedron() {
  const bytes = new Uint8Array(84 + 4 * 50);
  const view = new DataView(bytes.buffer);
  view.setUint32(80, 4, true);
  const vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
  [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]].forEach((face, i) => {
    face.forEach((vertex, j) => vertices[vertex].forEach((value, axis) => {
      view.setFloat32(84 + i * 50 + 12 + j * 12 + axis * 4, value, true);
    }));
  });
  return Array.from(bytes);
}

test('model packages follow MuJoCo include and resource resolution', async (t) => {
  const server = await createServer({ configFile: false, root: path.resolve('.'),
    optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  t.after(() => server.close());
  const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const mj = await loadMujoco();
  const packageFiles = (entry, resources, htmlFallback = false) => page.evaluate(async ({ entry, resources, htmlFallback }) => {
    const { loadModelBundle } = await import('/src/resources/loadModelBundle.ts');
    const bundle = await loadModelBundle('Resource fixture', entry, async (file) => {
      if (resources[file]) return new Uint8Array(resources[file]);
      if (htmlFallback) return new TextEncoder().encode('<!doctype html><html><body>App shell</body></html>');
      throw new Error('Not found');
    });
    return { ...bundle, files: bundle.files.map((file) => ({ path: file.path, bytes: Array.from(file.bytes) })) };
  }, { entry, resources, htmlFallback });
  const compile = (bundle) => {
    const vfs = new mj.MjVFS();
    let model;
    try {
      for (const file of bundle.files) vfs.addBuffer(file.path, new Uint8Array(file.bytes));
      model = mj.MjModel.from_xml_path(bundle.entry, vfs);
      return { meshes: model.nmesh, geoms: model.ngeom, flexes: model.nflex };
    } finally { model?.delete(); vfs.delete(); }
  };

  await t.test('resource reads are bounded, shared, and stop reporting after cancellation', async () => {
    const result = await page.evaluate(async () => {
      const { loadModelBundle } = await import('/src/resources/loadModelBundle.ts');
      const xml = '<mujoco><asset>' + Array.from({ length: 24 }, (_, i) =>
        `<mesh name="m${i}" file="mesh${i % 12}.stl"/>`).join('') + '</asset></mujoco>';
      let active = 0, peak = 0;
      const reads = {}, stages = [];
      const read = async (path) => {
        reads[path] = (reads[path] || 0) + 1;
        active++; peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 2));
        active--;
        return new TextEncoder().encode(path === 'main.xml' ? xml : 'mesh');
      };
      const loaded = await loadModelBundle('Parallel resources', 'main.xml', read,
        { onProgress: (progress) => stages.push(progress.stage) });
      const controller = new AbortController();
      const cancelledProgress = [];
      let cancelled = false;
      try {
        await loadModelBundle('Cancelled resources', 'main.xml', read, {
          signal: controller.signal, onProgress: (progress) => {
            cancelledProgress.push(progress.stage);
            if (progress.completed === 2) controller.abort();
          },
        });
      } catch (error) { cancelled = error.name === 'AbortError'; }
      const afterCancel = cancelledProgress.length;
      await new Promise((resolve) => setTimeout(resolve, 20));
      return { peak, files: loaded.files.length, reads: Object.values(reads), stages,
        cancelled, quiet: afterCancel === cancelledProgress.length };
    });
    assert.ok(result.peak > 1 && result.peak <= 8);
    assert.equal(result.files, 13);
    assert.ok(result.reads.every((count) => count <= 2));
    assert.equal(result.stages.at(-1), 'verify');
    assert.equal(result.cancelled, true);
    assert.equal(result.quiet, true);
  });

  await t.test('included asset files resolve beside their declaring XML after a missing entry-relative path', async () => {
    const bundle = await packageFiles('models/main.xml', {
      'models/main.xml': encode('<mujoco><include file="assets/asset/meshes.xml"/><worldbody><geom type="mesh" mesh="tetra"/></worldbody></mujoco>'),
      'models/assets/asset/meshes.xml': encode('<mujoco><asset><mesh name="tetra" file="../geometry/tetra.stl"/></asset></mujoco>'),
      'models/assets/geometry/tetra.stl': tetrahedron(),
    }, true);
    assert.deepEqual(bundle.files.map((file) => file.path), [
      'models/assets/asset/meshes.xml', 'models/assets/geometry/tetra.stl', 'models/main.xml',
    ]);
    assert.deepEqual(compile(bundle), { meshes: 1, geoms: 1, flexes: 0 });
  });

  await t.test('entry compiler paths take precedence over included-file paths', async () => {
    const bundle = await packageFiles('main.xml', {
      'main.xml': encode('<mujoco><compiler meshdir="shared"/><include file="parts/meshes.xml"/><worldbody><geom type="mesh" mesh="tetra"/></worldbody></mujoco>'),
      'parts/meshes.xml': encode('<mujoco><asset><mesh name="tetra" file="tetra.stl"/></asset></mujoco>'),
      'shared/tetra.stl': tetrahedron(), 'parts/tetra.stl': encode('This candidate must not be loaded'),
    });
    assert.ok(bundle.files.some((file) => file.path === 'shared/tetra.stl'));
    assert.ok(!bundle.files.some((file) => file.path === 'parts/tetra.stl'));
    assert.deepEqual(compile(bundle), { meshes: 1, geoms: 1, flexes: 0 });
  });

  await t.test('nested includes prefer the main directory and otherwise use the including directory', async () => {
    const bundle = await packageFiles('main.xml', {
      'main.xml': encode('<mujoco><include file="parts/container.xml"/></mujoco>'),
      'parts/container.xml': encode('<mujoco><include file="body.xml"/><include file="sibling.xml"/></mujoco>'),
      'body.xml': encode('<mujoco><worldbody><geom type="sphere" size="1"/></worldbody></mujoco>'),
      'parts/body.xml': encode('This candidate must not be loaded'),
      'parts/sibling.xml': encode('<mujoco><worldbody><geom type="sphere" size="1" pos="3 0 0"/></worldbody></mujoco>'),
    });
    assert.ok(!bundle.files.some((file) => file.path === 'parts/body.xml'));
    assert.deepEqual(compile(bundle), { meshes: 0, geoms: 2, flexes: 0 });
  });

  await t.test('asset, mesh and texture compiler directories remain independent', async () => {
    const bundle = await packageFiles('main.xml', {
      'main.xml': encode('<mujoco><compiler assetdir="assets" meshdir="meshes" texturedir="textures"/><asset><mesh file="mesh.stl"/><hfield file="height.png"/><skin file="skin.skn"/><texture file="color.png"/></asset></mujoco>'),
      'meshes/mesh.stl': tetrahedron(), 'assets/height.png': [1], 'assets/skin.skn': [2], 'textures/color.png': [3],
    });
    assert.deepEqual(bundle.files.map((file) => file.path), ['assets/height.png', 'assets/skin.skn', 'main.xml', 'meshes/mesh.stl', 'textures/color.png']);
  });

  await t.test('flexcomp files use meshdir declared by an included scene', async () => {
    const bundle = await packageFiles('flex/model.xml', {
      'flex/model.xml': encode('<mujoco><include file="scene.xml"/><worldbody><flexcomp name="surface" type="mesh" file="tetra.stl" dim="2" radius=".001" mass="1"/></worldbody></mujoco>'),
      'flex/scene.xml': encode('<mujoco><compiler meshdir="asset"/></mujoco>'),
      'flex/asset/tetra.stl': tetrahedron(),
    }, true);
    assert.deepEqual(bundle.files.map((file) => file.path), ['flex/asset/tetra.stl', 'flex/model.xml', 'flex/scene.xml']);
    assert.deepEqual(compile(bundle), { meshes: 0, geoms: 0, flexes: 1 });
  });

  await t.test('attached models resolve their own compiler directories', async () => {
    const bundle = await packageFiles('main.xml', {
      'main.xml': encode('<mujoco><compiler meshdir="parent-meshes"/><asset><model name="child" file="child/model.xml"/></asset><worldbody><attach model="child" body="part" prefix="child/"/></worldbody></mujoco>'),
      'child/model.xml': encode('<mujoco><compiler meshdir="meshes"/><asset><mesh name="tetra" file="tetra.stl"/></asset><worldbody><body name="part"><geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>'),
      'child/meshes/tetra.stl': tetrahedron(),
    });
    assert.deepEqual(compile(bundle), { meshes: 1, geoms: 1, flexes: 0 });
  });

  await t.test('existing malformed resources are preserved instead of replaced by a fallback', async () => {
    const bundle = await packageFiles('main.xml', {
      'main.xml': encode('<mujoco><compiler meshdir="shared"/><include file="parts/meshes.xml"/><worldbody><geom type="mesh" mesh="tetra"/></worldbody></mujoco>'),
      'parts/meshes.xml': encode('<mujoco><asset><mesh name="tetra" file="tetra.stl"/></asset></mujoco>'),
      'shared/tetra.stl': encode('Invalid STL file'), 'parts/tetra.stl': tetrahedron(),
    });
    assert.ok(!bundle.files.some((file) => file.path === 'parts/tetra.stl'));
    assert.throws(() => compile(bundle));
  });

  await t.test('HTTP 200 app-shell responses report the missing declaration and attempted path', async () => {
    await page.route('**/resources/**', async (route) => {
      const isEntry = route.request().url().endsWith('/model.xml');
      await route.fulfill({ status: 200, contentType: isEntry ? 'application/xml' : 'text/html',
        body: isEntry ? '<mujoco><asset><mesh name="missing" file="meshes/head.stl"/></asset></mujoco>'
          : '<!doctype html><html><body>App shell</body></html>' });
    });
    await assert.rejects(page.evaluate(async () => {
      const { loadModelResource } = await import('/src/resources/loadModelBundle.ts');
      return loadModelResource('model.xml', `${location.origin}/resources/project/`);
    }), /Cannot resolve <mesh file="meshes\/head.stl"> in project\/model.xml.*project\/meshes\/head.stl: received HTML/);
  });
});
