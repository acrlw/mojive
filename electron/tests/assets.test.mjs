import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { PreviewSession } from '../src/assets/PreviewSession.ts';
import { readFile } from 'node:fs/promises';

test('the desktop joint example ships the native MJCF without conversion', async () => {
  assert.deepEqual(await readFile('public/demo/joint_gizmo.xml'), await readFile('../assets/joint_gizmo.xml'));
});

// A dismissed or superseded preview must release its late resources exactly once.
test('preview generations dispose late models without adopting them', async () => {
  const session = new PreviewSession();
  let finish, releases = 0;
  const first = session.load({}, () => new Promise((resolve) => { finish = resolve; }));
  const second = { document: {}, dispose: () => { releases++; } };
  assert.equal(await session.load({}, async () => second), second);
  finish({ document: {}, dispose: () => { releases++; } });
  assert.equal(await first, null);
  session.dispose(); session.dispose();
  assert.equal(releases, 2);
});

test('browser assets resolve resources, isolate preview, and run the real physics sample', async (t) => {
  const server = await createServer({ configFile: false, root: process.cwd(), logLevel: 'error',
    optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } });
  await server.listen(); t.after(() => server.close());
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const result = await page.evaluate(async () => {
    const { browserBundles, bundleUrls } = await import('/src/resources/browserBundle.ts');
    const { SAMPLE_ASSETS, sampleAsset, readAssetLibrary, writeAssetLibrary, rememberAsset } = await import('/src/assets/catalog.ts');
    const { loadAsset, loadAssetPreview } = await import('/src/assets/loadAsset.ts');
    const { setJoint, setJointConfiguration, disposeDocument } = await import('/src/scene.ts');
    const files = [new File(['<robot name="example"><link name="base"/></robot>'], 'example.urdf'),
      new File(['<mujocoinclude/>'], 'part.xml'), new File(['mesh'], 'part.stl')];
    const bundles = await browserBundles(files);
    const urls = bundleUrls(bundles[0]);
    const text = await (await fetch(urls.resolve(urls.base + 'part.stl'))).text();
    let missing = false;
    try { urls.resolve(urls.base + 'missing.stl'); } catch { missing = true; }
    urls.dispose();
    let duplicate = false;
    try { await browserBundles([files[0], files[0]]); } catch { duplicate = true; }
    const native = SAMPLE_ASSETS.find((asset) => asset.source.scene === 'geometry_lab');
    const model = SAMPLE_ASSETS.find((asset) => asset.source.scene === 'physics_pendulum');
    const preview = await loadAssetPreview(native), simulation = await loadAssetPreview(model);
    const positions = preview.document.root.children.map((object) => object.position.toArray());
    const independent = preview.document !== simulation.document && preview.document.root !== simulation.document.root;
    let saved;
    const upload = { id: 'upload', name: 'upload', format: 'URDF', source: { kind: 'bundle', bundle: bundles[0] } };
    const library = rememberAsset(rememberAsset({ recent: [], favorites: [model, upload] }, native), native);
    writeAssetLibrary({ setItem: (_, value) => { saved = value; } }, { ...library, recent: [...library.recent, upload] });
    const restored = readAssetLibrary({ getItem: () => saved });
    // Previously saved examples must also enter through the authoritative physics backend.
    const legacy = { ...sampleAsset('joint_gizmo'), format: 'Scene' };
    const migrated = readAssetLibrary({ getItem: () => JSON.stringify({ recent: [legacy], favorites: [legacy] }) });
    const jointExample = await loadAsset(legacy);
    const { document: jointDoc, runtime } = jointExample;
    if (!runtime) throw new Error('The default joint example did not load MuJoCo.');
    const initial = await runtime.session.capture();
    setJoint(jointDoc, '01_revolute_y', 0.25);
    setJoint(jointDoc, '02_prismatic_x', 0.1);
    setJointConfiguration(jointDoc, '03_ball_xyz', { rotation: [0, 0, Math.sin(0.1), Math.cos(0.1)] });
    setJointConfiguration(jointDoc, '04_free_6dof', { position: [1.7, 0.1, 0.7] });
    await runtime.session.capture(); // Wait for commands routed from production controls.
    const pose = jointDoc.joints.map((joint) => ({ name: joint.name, value: joint.value,
      position: joint.position?.toArray(), rotation: joint.rotation?.toArray() }));
    await runtime.session.execute({ type: 'step', count: 2 });
    const stepped = await runtime.session.capture();
    await runtime.session.stepBack();
    const returnedToInitial = JSON.stringify(await runtime.session.capture()) === JSON.stringify(initial);
    await runtime.session.execute({ type: 'restore', snapshot: stepped });
    const restoredStep = JSON.stringify(await runtime.session.capture()) === JSON.stringify(stepped);
    const jointPreview = await loadAssetPreview(sampleAsset('joint_gizmo'));
    const previewIsolated = jointPreview.document !== jointDoc && jointPreview.document.joints[0].value === 0;
    jointPreview.dispose();
    const result = { entries: bundles.length, text, missing, duplicate, independent,
      previewNodes: preview.document.nodes.length, joints: simulation.document.joints.length,
      unchanged: JSON.stringify(positions) === JSON.stringify(preview.document.root.children.map((object) => object.position.toArray())),
      recent: restored.recent.length, favorites: restored.favorites.length,
      jointExample: { formats: [migrated.recent[0].format, migrated.favorites[0].format],
        name: jointDoc.name, count: jointDoc.joints.length, types: [...new Set(jointDoc.joints.map((joint) => joint.type))].sort(),
        pose, returnedToInitial, restoredStep, previewIsolated } };
    runtime.dispose(); disposeDocument(jointDoc);
    preview.dispose(); simulation.dispose();
    return result;
  });
  assert.equal(result.entries, 1); assert.equal(result.text, 'mesh');
  assert.equal(result.missing, true); assert.equal(result.duplicate, true);
  assert.equal(result.independent, true); assert.equal(result.unchanged, true);
  assert.ok(result.previewNodes > 0); assert.equal(result.joints, 1);
  assert.equal(result.recent, 1); assert.equal(result.favorites, 1);
  assert.deepEqual(result.jointExample.formats, ['MJCF', 'MJCF']);
  assert.equal(result.jointExample.name, 'Joint gizmos');
  assert.equal(result.jointExample.count, 9);
  assert.deepEqual(result.jointExample.types, ['ball', 'free', 'prismatic', 'revolute']);
  const pose = (name) => result.jointExample.pose.find((joint) => joint.name === name);
  assert.equal(pose('01_revolute_y').value, 0.25);
  assert.equal(pose('02_prismatic_x').value, 0.1);
  assert.deepEqual(pose('04_free_6dof').position, [1.7, 0.1, 0.7]);
  assert.ok(Math.abs(pose('03_ball_xyz').rotation[2] - Math.sin(0.1)) < 1e-12);
  assert.equal(result.jointExample.returnedToInitial, true);
  assert.equal(result.jointExample.restoredStep, true);
  assert.equal(result.jointExample.previewIsolated, true);
});
