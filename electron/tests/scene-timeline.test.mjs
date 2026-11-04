import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import path from 'node:path';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
    root: path.resolve('.'),
    server: { host: '127.0.0.1', port: 0 },
    logLevel: 'error',
  });
  server.middlewares.use('/__timeline_domain', (_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Timeline domain</title>');
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__timeline_domain`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('named keys, snapshots and takes persist complete scene poses with legacy workspace compatibility', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { setJoint, setJointConfiguration, authorJointTransform } = await import('/src/scene.ts');
    const { capturePose, applyPose, assertPoseCompatible, saveWorkspace, loadWorkspace } =
      await import('/src/editor.ts');
    const doc = await createCatalogScene('joint_gizmo');
    setJoint(doc, '01_revolute_y', 0.3);
    setJointConfiguration(doc, '03_ball_xyz', { rotation: [0.1, 0, 0, Math.sqrt(0.99)] });
    setJointConfiguration(doc, '04_free_6dof', { position: [0.4, 0.2, 0.1] });
    const joint = doc.joints[0];
    joint.object.position.z += 0.2;
    authorJointTransform(doc, joint.id);
    const geom = doc.nodes.find((n) => n.kind === 'geom');
    geom.object.position.x += 0.12;
    const pose = capturePose(doc),
      original = JSON.stringify(pose),
      timeline = {
        snapshots: [{ id: 'snapshot', name: 'Complete pose', time: 1, pose }],
        takes: [
          {
            id: 'take',
            name: 'Authored motion',
            frames: [
              { time: 0, pose },
              { time: 2, pose },
            ],
          },
        ],
        activeTakeId: 'take',
      };
    setJoint(doc, '01_revolute_y', -0.7);
    geom.object.position.set(9, 9, 9);
    assertPoseCompatible(doc, pose);
    applyPose(doc, pose);
    const difference = (a, b) => {
      if (typeof a === 'number') return Math.abs(a - b);
      if (a && typeof a === 'object')
        return Math.max(0, ...Object.keys(a).map((key) => difference(a[key], b[key])));
      return a === b ? 0 : Infinity;
    };
    const restored = loadWorkspace(
      saveWorkspace(doc, [{ id: 'key', name: 'Ready', time: 1, pose }], 4, timeline),
    );
    const legacy = loadWorkspace(saveWorkspace(doc, [{ id: 'old', time: 0, pose }], 4));
    return {
      same: JSON.stringify(pose) === original,
      difference: difference(pose, capturePose(doc)),
      saved: restored.timeline,
      keyName: restored.keys[0].name,
      legacy: legacy.timeline,
      nodes: doc.nodes.length,
      transformCount: Object.keys(pose.transforms).length,
      joints: doc.joints.length,
      originCount: Object.keys(pose.jointOrigins).length,
    };
  });
  assert.equal(result.same, true);
  assert.ok(result.difference < 1e-10);
  assert.equal(result.nodes, result.transformCount);
  assert.equal(result.joints, result.originCount);
  assert.equal(result.saved.activeTakeId, 'take');
  assert.equal(result.saved.snapshots[0].name, 'Complete pose');
  assert.equal(result.saved.takes[0].frames.length, 2);
  assert.equal(result.keyName, 'Ready');
  assert.deepEqual(result.legacy, { snapshots: [], takes: [], activeTakeId: null });
});
test('long timeline keys, snapshots and recorded samples roundtrip beyond an hour without truncation', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { disposeDocument, setJoint } = await import('/src/scene.ts');
    const { capturePose, sampleTake, saveWorkspace, loadWorkspace } = await import(
      '/src/editor.ts'
    );
    const doc = await createCatalogScene('joint_gizmo');
    const times = [0, 10.125, 3600.125, 12345.6789];
    const frames = times.map((time, i) => {
      setJoint(doc, '01_revolute_y', i * 0.1);
      return { time, pose: capturePose(doc) };
    });
    const last = frames.at(-1);
    const saved = loadWorkspace(
      saveWorkspace(doc, [{ id: 'key', name: 'Late key', ...last }], 12350, {
        snapshots: [{ id: 'snapshot', name: 'Late snapshot', ...last }],
        takes: [{ id: 'take', name: 'Long recording', frames }],
        activeTakeId: 'take',
      }),
    );
    const index = sampleTake(saved.doc, saved.timeline.takes[0], last.time, 'floor');
    const result = {
      times: saved.timeline.takes[0].frames.map((frame) => frame.time),
      key: saved.keys[0].time,
      snapshot: saved.timeline.snapshots[0].time,
      extent: saved.duration,
      index,
      joint: saved.doc.joints.find((joint) => joint.name === '01_revolute_y').value,
    };
    disposeDocument(doc);
    disposeDocument(saved.doc);
    return result;
  });
  assert.deepEqual(result.times, [0, 10.125, 3600.125, 12345.6789]);
  assert.equal(result.key, 12345.6789);
  assert.equal(result.snapshot, 12345.6789);
  assert.equal(result.extent, 12350);
  assert.equal(result.index, 3);
  assert.ok(Math.abs(result.joint - 0.3) < 1e-12);
});
test('take sampling selects recorded frames with native nearest/floor rules and rejects partial scene mixing atomically', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { setJoint, createPrimitive } = await import('/src/scene.ts');
    const { capturePose, sampleTake, assertPoseCompatible } = await import('/src/editor.ts');
    const doc = await createCatalogScene('joint_gizmo');
    const frames = [0, 0.4, 0.8].map((value, index) => {
      setJoint(doc, '01_revolute_y', value);
      return { time: index * 2, pose: capturePose(doc) };
    });
    const take = { id: 'take', name: 'Recorded', frames };
    const samples = [];
    for (const [time, mode] of [
      [-1, 'nearest'],
      [1, 'nearest'],
      [1.01, 'nearest'],
      [1.99, 'floor'],
      [2, 'floor'],
      [3.9, 'floor'],
      [99, 'nearest'],
    ]) {
      const index = sampleTake(doc, take, time, mode);
      samples.push([index, doc.joints[0].value]);
    }
    const primitive = createPrimitive('box');
    doc.nodes.push(primitive);
    doc.root.add(primitive.object);
    const before = JSON.stringify(capturePose(doc));
    let error = '';
    try {
      sampleTake(doc, take, 0);
    } catch (failure) {
      error = failure.message;
    }
    const unchanged = before === JSON.stringify(capturePose(doc));
    let missingOrigin = '';
    const partial = capturePose(doc);
    delete partial.jointOrigins;
    try {
      assertPoseCompatible(doc, partial);
    } catch (failure) {
      missingOrigin = failure.message;
    }
    return {
      samples,
      error,
      unchanged,
      missingOrigin,
      empty: sampleTake(doc, { id: 'empty', name: 'Empty', frames: [] }, 0),
    };
  });
  assert.deepEqual(result.samples, [
    [0, 0],
    [0, 0],
    [1, 0.4],
    [0, 0],
    [1, 0.4],
    [1, 0.4],
    [2, 0.8],
  ]);
  assert.match(result.error, /incompatible/);
  assert.match(result.missingOrigin, /joint origins/);
  assert.equal(result.unchanged, true);
  assert.equal(result.empty, -1);
});
test('malformed snapshot/take names, poses, times, ordering and references reject before scene replacement', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { capturePose, saveWorkspace, loadWorkspace } = await import('/src/editor.ts');
    const doc = await createCatalogScene('joint_gizmo'),
      pose = capturePose(doc);
    const original = saveWorkspace(doc, [{ id: 'key', time: 0, pose }], 4, {
      snapshots: [
        { id: 'one', name: 'One', time: 1, pose },
        { id: 'two', name: 'Two', time: 2, pose },
      ],
      takes: [
        {
          id: 'take',
          name: 'Take',
          frames: [
            { time: 0, pose },
            { time: 2, pose },
          ],
        },
      ],
      activeTakeId: 'take',
    });
    const changes = [
      (data) => data.timeline.snapshots.reverse(),
      (data) => (data.timeline.snapshots[0].time = null),
      (data) => (data.timeline.snapshots[0].name = ''),
      (data) => (data.timeline.snapshots[1].id = 'one'),
      (data) => (data.timeline.activeTakeId = 'missing'),
      (data) => data.timeline.takes[0].frames.reverse(),
      (data) => (data.timeline.takes[0].frames[1].time = 0),
      (data) => (data.timeline.takes[0].frames[1].time = 5),
      (data) => (data.timeline.takes[0].frames[0].pose.transforms.world.q = [0, 0, 0, 2]),
      (data) =>
        delete data.timeline.snapshots[0].pose.jointConfigurations[
          doc.joints.find((j) => j.name === '03_ball_xyz').id
        ],
      (data) => (data.timeline.takes[0].name = ' '),
      (data) => (data.keys[0].name = 9),
    ];
    const outcomes = changes.map((change) => {
      const data = JSON.parse(original);
      change(data);
      try {
        loadWorkspace(JSON.stringify(data));
        return '';
      } catch (error) {
        return error.message;
      }
    });
    const unchanged =
      original ===
      saveWorkspace(doc, [{ id: 'key', time: 0, pose }], 4, JSON.parse(original).timeline);
    return { outcomes, unchanged };
  });
  assert.ok(result.outcomes.every(Boolean), JSON.stringify(result.outcomes));
  assert.equal(result.unchanged, true);
});
