import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('physics workspaces preserve resources, defaults, full integration state and viewer edits', async (t) => {
  const server = await createServer({ configFile: false, root: process.cwd(),
    optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } });
  await server.listen(); t.after(() => server.close());
  const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  const result = await page.evaluate(async () => {
    const { loadMujocoBackend } = await import('/src/physics/WorkerBackend.ts');
    const { createModelBundle } = await import('/src/resources/modelBundle.ts');
    const { PhysicsDocument } = await import('/src/physics/PhysicsDocument.ts');
    const { savePhysicsWorkspace, loadPhysicsWorkspace } = await import('/src/physics/workspace.ts');
    const { createPrimitive, disposeDocument, createDemo, removeNode } = await import('/src/scene.ts');
    const { updateCameraView } = await import('/src/scene-camera.ts');
    const { captureSceneState, applySceneState } = await import('/src/physics/sceneState.ts');
    const { encodeDocument, decodeDocument, applyPose } = await import('/src/editor.ts');
    const { updateLight } = await import('/src/scene-light.ts');
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { sceneCommands } = await import('/src/scene-commands.ts');
    const { PerturbationGesture } = await import('/src/viewport/perturbation.ts');
    const encoder = new TextEncoder();
    const bundle = await createModelBundle('Saved model', 'model.xml', [
      { path: 'model.xml', bytes: encoder.encode('<mujoco><option timestep=".002" gravity="0 0 -9.81"/><include file="world.xml"/></mujoco>') },
      { path: 'world.xml', bytes: encoder.encode(`<mujoco><worldbody><geom type="plane" size="3 3 .1"/>
        <body name="ball" pos="0 0 1"><freejoint name="free"/><geom type="sphere" size=".1"/>
          <camera name="attached" pos="0 -2 1"/><light name="attached_light" type="spot" pos="0 0 2"/>
        </body></worldbody></mujoco>`) },
    ]);
    const runtime = await PhysicsDocument.load(bundle, loadMujocoBackend);
    const timeline = { snapshots: [], takes: [], activeTakeId: null };
    let reopened;
    try {
      const doc = runtime.document;
      const gesture = new PerturbationGesture(), perturbation = [];
      const port = sceneCommands(doc).perturbation;
      const movable = doc.nodes.find((node) => node.kind === 'geom' && port.bodyAt(node.id));
      const fixed = doc.nodes.find((node) => node.kind === 'geom' && !port.bodyAt(node.id));
      let begins = 0;
      for (const view of [new THREE.PerspectiveCamera(60, 1, .02, 100), new THREE.OrthographicCamera(-2, 2, 2, -2, .02, 100)]) {
        view.up.set(0, 0, 1); view.position.set(0, -5, 1); view.lookAt(0, 0, 1); view.updateMatrixWorld(true);
        const begin = (id, pointer = 7, button = 0) => gesture.begin(doc, id, view, [200, 200], 400, 400, pointer, button, () => begins++);
        perturbation.push(!begin(fixed.id), begin(movable.id), !begin(movable.id, 8));
        perturbation.push(gesture.releases(7, 0), !gesture.releases(8, 0), !gesture.releases(7, 2));
        gesture.move([240, 200]);
        const held = (await runtime.session.capture()).interaction.target;
        const expected = view.isOrthographicCamera ? .4 : 2 * 5 * Math.tan(Math.PI / 6) * .1;
        perturbation.push(Math.abs(held.position[0] - expected) < 1e-8, held.mode === 'translation');
        perturbation.push(gesture.end(), !gesture.end(), (await runtime.session.capture()).interaction === undefined);
        perturbation.push(begin(movable.id, 9, 2));
        gesture.move([240, 220]);
        const rotated = (await runtime.session.capture()).interaction.target;
        perturbation.push(rotated.mode === 'rotation', Math.abs(Math.hypot(...rotated.rotation) - 1) < 1e-8,
          Math.abs(rotated.rotation[3]) < .999);
        gesture.end();
        perturbation.push((await runtime.session.capture()).interaction === undefined);
      }
      perturbation.push(begins === 4);
      await runtime.session.execute({ type: 'reset' });
      const camera = doc.nodes.find((node) => node.kind === 'camera');
      const light = doc.nodes.find((node) => node.kind === 'light');
      const originalTargetParent = light.object.target.parent;
      const ownership = [];
      for (const edit of [
        () => removeNode(doc, camera.id),
        () => removeNode(doc, doc.nodes.find((node) => node.kind === 'world').id),
        () => updateCameraView(doc, camera.id, { position: [1, 2, 3] }),
        () => updateCameraView(doc, camera.id, { projection: 'orthographic' }),
        () => updateLight(doc, light.id, { direction: [0, 1, 0] }),
        () => updateLight(doc, light.id, { type: 'point' }),
      ]) {
        try { edit(); ownership.push(false); } catch { ownership.push(true); }
      }
      updateLight(doc, light.id, { intensity: 2 });
      ownership.push(light.object.target.parent === originalTargetParent);
      const box = createPrimitive('box');
      box.parentId = doc.nodes.find((node) => node.kind === 'world').id;
      doc.nodes.find((node) => node.id === box.parentId).object.add(box.object);
      doc.nodes.push(box);
      box.object.position.set(2, 3, 4);
      await runtime.session.execute({ type: 'environment', values: { timestep: .005, gravity: [0, 0, -4] } });
      await runtime.session.execute({ type: 'step', count: 83 });
      const snapshot = await runtime.session.capture();
      const recorded = captureSceneState(doc, runtime);
      const detached = decodeDocument(encodeDocument(doc));
      applyPose(detached, recorded.pose);
      const replayPose = captureSceneState(detached, null).pose;
      const visualReplay = JSON.stringify(replayPose) === JSON.stringify(recorded.pose);
      disposeDocument(detached);
      await runtime.session.execute({ type: 'step', count: 2500 });
      const passesTenSeconds = runtime.session.current.frame.time > 10;
      await applySceneState(doc, recorded, runtime);
      const replayMatches = JSON.stringify(await runtime.session.capture()) === JSON.stringify(snapshot);
      timeline.snapshots.push({ id: 'snapshot', name: 'Snapshot', time: .415, ...recorded });
      timeline.takes.push({ id: 'take', name: 'Take', frames: [{ time: .415, ...recorded }] });
      timeline.activeTakeId = 'take';
      const saved = await savePhysicsWorkspace(doc, [], 10, timeline, runtime);
      reopened = await loadPhysicsWorkspace(saved, () => loadMujocoBackend);
      const restored = await reopened.runtime.session.capture();
      const stateMatches = JSON.stringify(snapshot) === JSON.stringify(restored);
      const boxMatches = reopened.doc.nodes.find((node) => node.id === box.id).object.position.toArray().join() === '2,3,4';
      await runtime.session.execute({ type: 'step', count: 130 });
      await reopened.runtime.session.execute({ type: 'step', count: 130 });
      const continues = JSON.stringify(await runtime.session.capture()) === JSON.stringify(await reopened.runtime.session.capture());
      await reopened.runtime.session.execute({ type: 'resetEnvironment' });
      const defaults = reopened.runtime.session.current.environment;
      const rejected = [];
      for (const corrupt of [
        (data) => { delete data.physics; },
        (data) => { data.physics.bundle.files[0].base64 = btoa('corrupt'); },
        (data) => { data.physics.snapshot.sourceId = 'wrong-source'; },
        (data) => { data.document.joints[0].lower = -10; },
        (data) => { data.physics.snapshot.state = [0]; },
        (data) => { delete data.timeline.takes[0].frames[0].simulation; },
        (data) => { data.timeline.snapshots[0].simulation.format = 'invalid'; },
      ]) {
        const data = JSON.parse(saved); corrupt(data);
        try {
          const invalid = await loadPhysicsWorkspace(JSON.stringify(data), () => loadMujocoBackend);
          invalid.runtime?.dispose(); disposeDocument(invalid.doc); rejected.push(false);
        } catch { rejected.push(true); }
      }
      const native = await createDemo();
      const nativeSaved = await savePhysicsWorkspace(native, [], 10, { snapshots: [], takes: [], activeTakeId: null }, null);
      const nativeLoaded = await loadPhysicsWorkspace(nativeSaved, () => { throw Error('Engine must not load'); });
      const nativeWorks = nativeLoaded.runtime === null && nativeLoaded.doc.joints.length === native.joints.length;
      disposeDocument(native); disposeDocument(nativeLoaded.doc);
      return { visualReplay, passesTenSeconds, replayMatches, stateMatches, boxMatches, continues, rejected, nativeWorks, ownership, perturbation,
        defaults: [defaults.timestep, defaults.gravity], limited: reopened.doc.joints[0].lower === -Infinity,
        resources: reopened.runtime.bundle.files.length };
    } finally {
      runtime.dispose(); disposeDocument(runtime.document);
      if (reopened) { reopened.runtime?.dispose(); disposeDocument(reopened.doc); }
    }
  });
  assert.equal(result.stateMatches, true); assert.equal(result.continues, true);
  assert.equal(result.boxMatches, true); assert.equal(result.nativeWorks, true);
  assert.equal(result.limited, true); assert.equal(result.resources, 2);
  assert.deepEqual(result.defaults, [.002, [0, 0, -9.81]]);
  assert.deepEqual(result.rejected, [true, true, true, true, true, true, true]);
  assert.equal(result.visualReplay, true); assert.equal(result.passesTenSeconds, true); assert.equal(result.replayMatches, true);
  assert.deepEqual(result.ownership, [true, true, true, true, true, true, true]);
  assert.ok(result.perturbation.every(Boolean), `Perturbation gesture checks: ${result.perturbation}`);
});
