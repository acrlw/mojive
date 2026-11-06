import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
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
  });
  server.middlewares.use('/__gizmo_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Gizmo gestures</title>');
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__gizmo_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function gesture(kind) {
  return page.evaluate(async (kind) => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { GizmoOverlay } = await import('/src/viewport/gizmo.ts');
    const { worldScale, project, frontArc } = await import('/src/viewport/geometry.ts');
    const { createPrimitive } = await import('/src/scene.ts');
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
    let begins = 0,
      commits = 0,
      cancels = 0,
      changes = 0;
    const gizmo = new GizmoOverlay(canvas, {
        onBegin: () => begins++,
        onCommit: () => commits++,
        onCancel: kind === 'cancel-callback' ? () => cancels++ : undefined,
        onChange: () => changes++,
        invalidate: () => {},
      }),
      camera = new THREE.PerspectiveCamera(42, 900 / 600, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -4, 2.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const origin = new THREE.Vector3(),
      scale = worldScale(origin, camera, 600),
      center = project(origin, camera, 900, 600),
      tip = project(new THREE.Vector3(scale, 0, 0), camera, 900, 600),
      delta = [tip[0] - center[0], tip[1] - center[1]],
      length = Math.hypot(...delta);
    let start = [center[0] + delta[0] * 0.75, center[1] + delta[1] * 0.75],
      end = [start[0] + (delta[0] / length) * 25, start[1] + (delta[1] / length) * 25];
    gizmo.update(
      camera,
      doc,
      node.id,
      kind === 'rotate' ? 'rotation' : 'position',
      'world',
      900,
      600,
    );
    if (kind === 'rotate') {
      const path = frontArc(origin, new THREE.Vector3(0, 0, 1), scale * 0.78, camera, 900, 600);
      start = path[12];
      end = path[17];
    }
    const acquired = gizmo.begin(start);
    gizmo.move(end, false, kind === 'snap');
    const moved = node.object.position.toArray(),
      q = node.object.quaternion.toArray();
    gizmo.end(kind === 'cancel' || kind === 'cancel-callback');
    const restored = node.object.position.toArray();
    gizmo.dispose();
    canvas.remove();
    return {
      acquired,
      begins,
      commits,
      cancels,
      changes,
      moved,
      restored,
      q,
      expected: (25 * scale) / length,
    };
  }, kind);
}
test('axis drag uses native projected travel and one command transaction', async () => {
  const r = await gesture('move');
  assert.equal(r.acquired, true);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
  assert.equal(r.changes, 1);
  assert.ok(Math.abs(r.moved[0] - r.expected) < 1e-8);
  assert.ok(Math.abs(r.moved[1]) + Math.abs(r.moved[2]) < 1e-9);
});
test('Shift snapping rounds axis travel to the native 0.1 m interval', async () => {
  const r = await gesture('snap');
  assert.ok(Math.abs(r.moved[0] * 10 - Math.round(r.moved[0] * 10)) < 1e-8);
});
test('cancel restores the pose after held input and closes the edit', async () => {
  const r = await gesture('cancel');
  assert.ok(r.moved[0] > 0.01);
  assert.deepEqual(r.restored, [0, 0, 0]);
  assert.equal(r.commits, 1);
});
test('cancel uses the cancellation boundary without committing the held edit', async () => {
  const r = await gesture('cancel-callback');
  assert.equal(r.acquired, true);
  assert.ok(r.moved[0] > 0.01);
  assert.deepEqual(r.restored, [0, 0, 0]);
  assert.equal(r.begins, 1);
  assert.equal(r.cancels, 1);
  assert.equal(r.commits, 0);
});
test('oblique rotation follows the projected ring and rotates only the world axis', async () => {
  const r = await gesture('rotate');
  assert.equal(r.acquired, true);
  assert.ok(Math.abs(r.q[2]) > 0.02);
  assert.ok(Math.abs(r.q[0]) + Math.abs(r.q[1]) < 1e-7);
  assert.ok(Math.abs(r.q.reduce((s, v) => s + v * v, 0) - 1) < 1e-9);
});

test('position-axis conversion stays frozen across multiple projected motions', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { worldScale, project } = await import('/src/viewport/geometry.ts'),
      { createPrimitive } = await import('/src/scene.ts');
    const node = createPrimitive('box', 'Axis trace'),
      doc = { root: new T.Group(), nodes: [node], joints: [], files: [], warnings: [] };
    node.object.position.set(0, 0, 0);
    doc.root.add(node.object);
    const camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(3, -4, 2.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const canvas = document.createElement('canvas'),
      g = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => {} });
    g.update(camera, doc, node.id, 'position', 'world', 900, 600);
    const origin = new T.Vector3(),
      scale = worldScale(origin, camera, 600),
      center = project(origin, camera, 900, 600),
      tip = project(new T.Vector3(scale, 0, 0), camera, 900, 600),
      dx = tip[0] - center[0],
      dy = tip[1] - center[1],
      length = Math.hypot(dx, dy),
      start = [center[0] + dx * 0.75, center[1] + dy * 0.75];
    g.begin(start);
    const actual = [];
    for (const travel of [12, 40, 85, 130, 55, -20]) {
      g.move([start[0] + (dx / length) * travel, start[1] + (dy / length) * travel]);
      actual.push(node.object.position.x);
    }
    g.end(true);
    g.dispose();
    return {
      actual,
      expected: [12, 40, 85, 130, 55, -20].map((travel) => (travel * scale) / length),
    };
  });
  for (let i = 0; i < r.actual.length; i++) assert.ok(Math.abs(r.actual[i] - r.expected[i]) < 1e-9);
});

for (const jointName of ['01_revolute_y', '02_prismatic_x']) {
  test(`${jointName} drag is independent of delayed physics acknowledgements`, async () => {
    const result = await page.evaluate(async (jointName) => {
      const T = await import('/node_modules/three/build/three.module.js'),
        { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
        { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
        { setJoint, disposeDocument } = await import('/src/scene.ts'),
        { bindSceneCommands } = await import('/src/scene-commands.ts'),
        xml = await (await fetch('/demo/joint_gizmo.xml')).text();
      const trace = (delayed) => {
        const doc = createNativeJointScene(xml),
          joint = doc.joints.find((item) => item.name === jointName),
          canvas = document.createElement('canvas'),
          g = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => {} }),
          camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100);
        doc.root.updateMatrixWorld(true);
        const center = joint.object.getWorldPosition(new T.Vector3());
        camera.up.set(0, 0, 1);
        camera.position.copy(center).add(new T.Vector3(3, -4, 2.5));
        camera.lookAt(center);
        camera.updateMatrixWorld();
        g.configure({ activeJoint: joint.id });
        g.update(camera, doc, joint.bodyId, 'joint', 'world', 900, 600);
        const values = [];
        let unbind;
        const commands = {
          setJointPosition(id, position) {
            values.push(position[0]);
            if (!delayed) {
              unbind();
              setJoint(doc, id, position[0]);
              unbind = bindSceneCommands(doc, commands);
            }
          },
        };
        unbind = bindSceneCommands(doc, commands);
        // Use the actual projected hit target and frozen drag basis, including
        // travel beyond a limit followed by reversal away from the clamp.
        const start = g.jointProjection.current,
          acquired = g.begin(start),
          direction = joint.type === 'prismatic' ? g.drag?.axisScreen : g.drag?.tangent;
        if (direction)
          for (const distance of [12, 40, 85, 300, 600, 625, 600, 575, 550])
            g.move([start[0] + direction[0] * distance, start[1] + direction[1] * distance]);
        g.end();
        g.dispose();
        unbind();
        disposeDocument(doc);
        return { acquired, values, lower: joint.lower, upper: joint.upper };
      };
      return { immediate: trace(false), delayed: trace(true) };
    }, jointName);
    assert.equal(result.immediate.acquired, true);
    assert.equal(result.delayed.acquired, true);
    assert.ok(result.immediate.values.length >= 3);
    assert.ok(result.immediate.values.length < 9, 'repeated clamped targets are not resent');
    assert.equal(result.delayed.values.length, result.immediate.values.length);
    for (let i = 0; i < result.immediate.values.length; i++)
      assert.ok(Math.abs(result.immediate.values[i] - result.delayed.values[i]) < 1e-9);
    const { values, lower, upper } = result.delayed;
    assert.ok(
      values.some((value) => value === lower || value === upper),
      'drag reaches a limit',
    );
    assert.ok(values.at(-1) > lower && values.at(-1) < upper, 'reversing leaves the limit');
  });
}

test('precision rail sends a return to the original value before earlier targets are acknowledged', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
      { disposeDocument } = await import('/src/scene.ts'),
      { bindSceneCommands } = await import('/src/scene-commands.ts'),
      doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text()),
      joint = doc.joints.find((item) => item.name === '06_precision_hinge_y'),
      camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100),
      values = [],
      unbind = bindSceneCommands(doc, { setJointPosition: (_, value) => values.push(value[0]) }),
      g = new GizmoOverlay(document.createElement('canvas'), {
        onChange: () => {},
        invalidate: () => {},
      });
    doc.root.updateMatrixWorld(true);
    const center = joint.object.getWorldPosition(new T.Vector3()),
      original = joint.value;
    camera.up.set(0, 0, 1);
    camera.position.copy(center).add(new T.Vector3(3, -4, 2.5));
    camera.lookAt(center);
    camera.updateMatrixWorld();
    g.configure({ activeJoint: joint.id });
    g.update(camera, doc, joint.bodyId, 'joint', 'world', 900, 600);
    g.hoverAt(g.jointProjection.current);
    await new Promise((resolve) => setTimeout(resolve, 650));
    const rail = g.jointProjection.rail,
      pointAt = (t) => [rail.start[0] + (rail.end[0] - rail.start[0]) * t, rail.start[1]],
      start = pointAt((original - rail.lower) / (rail.upper - rail.lower)),
      target = pointAt(0.75),
      acquired = g.begin(start),
      kind = g.jointEdit?.kind;
    for (const point of [target, target, start, start]) g.move(point);
    g.end();
    const acknowledged = joint.value;
    g.dispose();
    unbind();
    disposeDocument(doc);
    return { acquired, kind, original, acknowledged, values };
  });
  assert.equal(r.acquired, true);
  assert.equal(r.kind, 'rail');
  assert.equal(r.acknowledged, r.original, 'the delayed worker has not acknowledged a target');
  assert.equal(r.values.length, 2, 'identical requested targets are deduplicated');
  assert.notEqual(r.values[0], r.original);
  assert.ok(Math.abs(r.values[1] - r.original) < 1e-12, 'return to origin is the final request');
});

for (const [jointName, tool, handleKind] of [
  ['03_ball_xyz', 'joint', 'trackball'],
  ['03_ball_xyz', 'joint', 'ring'],
  ['04_free_6dof', 'position', 'axis'],
  ['04_free_6dof', 'rotation', 'trackball'],
]) {
  test(`${jointName} ${handleKind} keeps delayed physics frames authoritative and cancels`, async () => {
    const r = await page.evaluate(
      async ({ jointName, tool, handleKind }) => {
        const T = await import('/node_modules/three/build/three.module.js'),
          { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
          { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
          { setJointConfiguration, disposeDocument } = await import('/src/scene.ts'),
          { bindSceneCommands } = await import('/src/scene-commands.ts'),
          xml = await (await fetch('/demo/joint_gizmo.xml')).text();
        const trace = (mode) => {
          const doc = createNativeJointScene(xml),
            joint = doc.joints.find((item) => item.name === jointName),
            camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100),
            requests = [],
            unchangedUntilAck = [];
          let cancels = 0,
            commits = 0,
            unbind = () => {};
          const g = new GizmoOverlay(document.createElement('canvas'), {
              onChange: () => {},
              invalidate: () => {},
              onCancel: () => cancels++,
              onCommit: () => commits++,
            }),
            pose = () => [...joint.object.position.toArray(), ...joint.object.quaternion.toArray()],
            original = pose(),
            apply = (value) => {
              unbind();
              setJointConfiguration(doc, joint.id, {
                position: joint.type === 'free' ? value.slice(0, 3) : undefined,
                rotation: joint.type === 'free' ? value.slice(3) : value,
              });
              if (mode !== 'authored') unbind = bindSceneCommands(doc, commands);
            },
            commands = {
              setJointPosition(_, value) {
                requests.push([...value]);
                if (mode === 'immediate') apply(value);
              },
            };
          if (mode !== 'authored') unbind = bindSceneCommands(doc, commands);
          doc.root.updateMatrixWorld(true);
          const center = joint.object.getWorldPosition(new T.Vector3());
          camera.up.set(0, 0, 1);
          camera.position.copy(center).add(new T.Vector3(3, -4, 2.5));
          camera.lookAt(center);
          camera.updateMatrixWorld();
          g.configure({ activeJoint: joint.id });
          g.update(camera, doc, joint.bodyId, tool, 'world', 900, 600);
          const candidates = g.handles.filter((item) => item.kind === handleKind),
            start = candidates
              .flatMap((handle) => handle.path)
              .find((point) => g.hit(point)?.kind === handleKind),
            acquired = !!start && g.begin(start),
            acquiredKind = g.drag?.handle.kind;
          let moved = false;
          if (acquired) {
            for (const [index, distance] of [10, 35, 75, 100].entries()) {
              const before = pose();
              g.move([start[0] + distance, start[1] + distance * 0.4]);
              unchangedUntilAck.push(
                pose().every((value, i) => Math.abs(value - before[i]) < 1e-12),
              );
              if (mode === 'delayed' && index >= 2) apply(requests[index - 2]);
            }
            moved = pose().some((value, i) => Math.abs(value - original[i]) > 1e-6);
            const beforeCancel = pose();
            g.end(true);
            if (mode === 'delayed') {
              unchangedUntilAck.push(
                pose().every((value, i) => Math.abs(value - beforeCancel[i]) < 1e-12),
              );
              for (const value of requests.slice(2)) apply(value);
            }
          }
          const restored = pose();
          g.dispose();
          unbind();
          disposeDocument(doc);
          return {
            acquired,
            acquiredKind,
            requests,
            unchangedUntilAck,
            original,
            restored,
            moved,
            cancels,
            commits,
          };
        };
        return {
          immediate: trace('immediate'),
          delayed: trace('delayed'),
          authored: trace('authored'),
        };
      },
      { jointName, tool, handleKind },
    );
    for (const result of Object.values(r)) {
      assert.equal(result.acquired, true);
      assert.equal(result.acquiredKind, handleKind);
      assert.equal(result.cancels, 1);
      assert.equal(result.commits, 0);
      assert.ok(result.moved, 'a nonzero gesture changes the acknowledged/authored pose');
      for (let i = 0; i < result.original.length; i++)
        assert.ok(
          Math.abs(result.restored[i] - result.original[i]) < 1e-9,
          'cancel restores the original pose',
        );
    }
    assert.equal(r.delayed.requests.length, 5);
    assert.ok(
      r.delayed.unchangedUntilAck.every(Boolean),
      'only acknowledgements mutate physics proxies',
    );
    for (let i = 0; i < r.immediate.requests.length; i++)
      for (let j = 0; j < r.immediate.requests[i].length; j++)
        assert.ok(
          Math.abs(r.immediate.requests[i][j] - r.delayed.requests[i][j]) < 1e-9,
          'target pose is independent of acknowledgement timing',
        );
  });
}

test('joint hover settles after precision dwell and wakes only for state changes', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
      { disposeDocument } = await import('/src/scene.ts'),
      doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text()),
      joint = doc.joints.find((item) => item.name === '06_precision_hinge_y'),
      camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100),
      canvas = document.createElement('canvas');
    doc.root.updateMatrixWorld(true);
    const center = joint.object.getWorldPosition(new T.Vector3());
    camera.up.set(0, 0, 1);
    camera.position.copy(center).add(new T.Vector3(3, -4, 2.5));
    camera.lookAt(center);
    camera.updateMatrixWorld();
    let invalidations = 0;
    const g = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => invalidations++ });
    g.configure({ activeJoint: joint.id });
    g.update(camera, doc, joint.bodyId, 'joint', 'world', 900, 600);
    const point = g.jointProjection.current;
    const hit = g.hoverAt(point),
      first = invalidations;
    for (let i = 0; i < 100; i++) g.hoverAt(point);
    const repeated = invalidations;
    await new Promise((resolve) => setTimeout(resolve, 650));
    const shown = g.precisionVisible,
      settled = invalidations;
    await new Promise((resolve) => setTimeout(resolve, 250));
    const idle = invalidations;
    g.hoverAt([-1000, -1000]);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const exitGrace = g.precisionVisible;
    await new Promise((resolve) => setTimeout(resolve, 500));
    const hidden = !g.precisionVisible,
      left = invalidations;
    await new Promise((resolve) => setTimeout(resolve, 250));
    const afterLeave = invalidations;
    g.dispose();
    disposeDocument(doc);
    return { hit, first, repeated, shown, settled, idle, exitGrace, hidden, left, afterLeave };
  });
  assert.equal(r.hit, true);
  assert.equal(r.repeated, r.first);
  assert.equal(r.shown, true);
  assert.equal(r.idle, r.settled);
  assert.equal(r.exitGrace, true);
  assert.equal(r.hidden, true);
  assert.ok(r.left > r.idle);
  assert.equal(r.afterLeave, r.left);
});
