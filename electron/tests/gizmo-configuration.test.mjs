import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
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
    logLevel: 'error',
  });
  server.middlewares.use('/__configuration_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><link rel="stylesheet" href="/src/design-tokens.css"><title>Joint configuration gestures</title>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__configuration_test`);
  await page.evaluate(async () => {
    await document.fonts.load('16px "JetBrains Mono"');
    await document.fonts.ready;
  });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function gesture(kind) {
  const result = await page.evaluate(async (kind) => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
      { worldScale, project, frontArc } = await import('/src/viewport/geometry.ts');
    const xml = await (await fetch('/demo/joint_gizmo.xml')).text(),
      doc = createNativeJointScene(xml),
      joint = doc.joints.find(
        (j) =>
          j.name ===
          (kind.startsWith('ball')
            ? '03_ball_xyz'
            : kind === 'multi'
              ? '05_multi_slide_y'
              : '04_free_6dof'),
      ),
      body = doc.nodes.find((n) => n.id === joint.bodyId),
      canvas = document.createElement('canvas');
    document.body.append(canvas);
    const labels = [],
      ctx = canvas.getContext('2d'),
      fill = ctx.fillText.bind(ctx);
    ctx.fillText = (text, ...args) => {
      labels.push(text);
      fill(text, ...args);
    };
    let begins = 0,
      commits = 0;
    const gizmo = new GizmoOverlay(canvas, {
      onBegin: () => begins++,
      onCommit: () => commits++,
      onChange: () => {},
      invalidate: () => {},
    });
    if (kind === 'ball-body' || kind === 'ball-world') {
      const { setJointConfiguration } = await import('/src/scene.ts');
      setJointConfiguration(doc, joint.id, {
        rotation: new T.Quaternion().setFromAxisAngle(new T.Vector3(1, 0, 0), 0.45).toArray(),
      });
    }
    const camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    doc.root.updateMatrixWorld(true);
    const center = joint.object.getWorldPosition(new T.Vector3());
    camera.position.copy(center).add(new T.Vector3(3, -4, 2.5));
    camera.lookAt(center);
    camera.updateMatrixWorld();
    const origin = joint.origin.toArray(),
      originalPosition = joint.position?.toArray(),
      originalRotation = joint.rotation?.toArray(),
      scale = worldScale(center, camera, 600);
    gizmo.configure({
      snapEnabled: true,
      translationSnap: 0.25,
      rotationSnap: 30,
      activeJoint: kind === 'multi' ? joint.id : null,
    });
    gizmo.update(
      camera,
      doc,
      body.id,
      kind === 'free-rotation' ? 'rotation' : 'joint',
      kind === 'ball-body' ? 'body' : 'world',
      900,
      600,
    );
    let start, end;
    if (kind.startsWith('ball') || kind === 'free-rotation') {
      const axis = new T.Vector3(0, 0, 1);
      if (kind === 'ball-body')
        axis.applyQuaternion(joint.object.getWorldQuaternion(new T.Quaternion()));
      const arc = frontArc(center, axis, scale * 0.78, camera, 900, 600);
      start = arc[9];
      end = arc[23];
    } else {
      const direction = kind === 'multi' ? new T.Vector3(0, 1, 0) : new T.Vector3(1, 0, 0),
        screen = project(center, camera, 900, 600),
        tip = project(center.clone().addScaledVector(direction, scale), camera, 900, 600),
        delta = [tip[0] - screen[0], tip[1] - screen[1]],
        length = Math.hypot(...delta);
      start = [screen[0] + delta[0] * 0.75, screen[1] + delta[1] * 0.75];
      if (kind === 'multi') {
        const { projectJoint } = await import('/src/viewport/joint_interaction.ts');
        const p = projectJoint(joint, center, direction, scale, camera, 900, 600);
        start = p.current;
      }
      end = [start[0] + (delta[0] / length) * 35, start[1] + (delta[1] / length) * 35];
    }
    const acquired = gizmo.begin(start);
    gizmo.move(end);
    const { ViewportPipeline } = await import('/src/viewport/pipeline.ts');
    const renderer = new T.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(900, 600);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    const scene = new T.Scene();
    scene.background = new T.Color('#22272d');
    const light = new T.HemisphereLight(0xcad7e4, 0x67645d, 2.6);
    light.position.set(0, 0, 5);
    scene.add(doc.root, light);
    const key = new T.DirectionalLight(0xfff4df, 3.1);
    key.position.set(4, -3, 7);
    scene.add(key);
    scene.updateMatrixWorld(true);
    const pipeline = new ViewportPipeline(renderer);
    pipeline.resize(900, 600);
    pipeline.render(scene, camera, doc, body.id, 'default');
    const output = document.createElement('canvas');
    output.width = 900;
    output.height = 600;
    const paint = output.getContext('2d');
    paint.drawImage(renderer.domElement, 0, 0);
    paint.drawImage(canvas, 0, 0);
    const image = output.toDataURL();
    pipeline.dispose();
    renderer.dispose();
    const moved = {
      position: joint.position?.toArray(),
      rotation: joint.rotation?.toArray(),
      value: joint.value,
      label: labels.at(-1),
      origin: joint.origin.toArray(),
    };
    gizmo.end(true);
    const restored = {
      position: joint.position?.toArray(),
      rotation: joint.rotation?.toArray(),
      value: joint.value,
    };
    gizmo.dispose();
    canvas.remove();
    return {
      acquired,
      begins,
      commits,
      origin,
      originalPosition,
      originalRotation,
      moved,
      restored,
      image,
    };
  }, kind);
  await mkdir('../output/electron/verification', { recursive: true });
  await writeFile(
    `../output/electron/verification/native-joint-${kind}-drag.png`,
    Buffer.from(result.image.split(',')[1], 'base64'),
  );
  return result;
}
test('native ball joint uses snapped quaternion rotation and restores its authored frame on cancel', async () => {
  const r = await gesture('ball');
  assert.equal(r.acquired, true);
  assert.equal(r.begins, 1);
  assert.equal(r.commits, 1);
  assert.ok(Math.abs(r.moved.rotation[2]) > 0.01);
  assert.ok(Math.abs(r.moved.rotation[0]) + Math.abs(r.moved.rotation[1]) < 1e-8);
  assert.match(r.moved.label, /SNAP 30°$/);
  assert.deepEqual(r.moved.origin, r.origin);
  assert.deepEqual(r.restored.rotation, r.originalRotation);
});
test('native free joint uses normal position handles and stores local configuration without rewriting origin', async () => {
  const r = await gesture('free');
  assert.equal(r.acquired, true);
  assert.ok(r.moved.position[0] > 0.1);
  assert.ok(Math.abs(r.moved.position[0] / 0.25 - Math.round(r.moved.position[0] / 0.25)) < 1e-8);
  assert.match(r.moved.label, /SNAP 0.25 m$/);
  assert.deepEqual(r.moved.origin, r.origin);
  assert.deepEqual(r.restored.position, r.originalPosition);
});
test('explicit direct-joint picker controls the chosen multi-joint body coordinate', async () => {
  const r = await gesture('multi');
  assert.equal(r.acquired, true);
  assert.ok(r.moved.value > 0);
  assert.match(r.moved.label, /05_multi_slide_y/);
  assert.equal(r.restored.value, 0);
  assert.deepEqual(r.moved.origin, r.origin);
});

test('free joint rotation writes a normalized quaternion while preserving translation and authored origin', async () => {
  const r = await gesture('free-rotation');
  assert.equal(r.acquired, true);
  assert.ok(Math.abs(r.moved.rotation[2]) > 0.01);
  assert.ok(Math.abs(r.moved.rotation.reduce((sum, v) => sum + v * v, 0) - 1) < 1e-9);
  assert.deepEqual(r.moved.position, r.originalPosition);
  assert.deepEqual(r.moved.origin, r.origin);
  assert.deepEqual(r.restored.rotation, r.originalRotation);
});

test('native slide accepts deliberate cardinal escape and reverses immediately after a clamped limit', async () => {
  const r = await page.evaluate(async () => {
    const T = await import('/node_modules/three/build/three.module.js'),
      { loadUrdf } = await import('/src/scene.ts'),
      { GizmoOverlay } = await import('/src/viewport/gizmo.ts'),
      { worldScale, project } = await import('/src/viewport/geometry.ts');
    const doc = await loadUrdf(
        '<robot name="slide"><link name="a"/><link name="b"/><joint name="slide" type="prismatic"><parent link="a"/><child link="b"/><axis xyz="1 0 0"/><limit lower="-1" upper="1"/></joint></robot>',
      ),
      j = doc.joints[0],
      camera = new T.PerspectiveCamera(42, 1.5, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -4, 1.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const canvas = document.createElement('canvas'),
      g = new GizmoOverlay(canvas, { onChange: () => {}, invalidate: () => {} });
    g.update(camera, doc, j.id, 'joint', 'world', 900, 600);
    const center = project(new T.Vector3(), camera, 900, 600);
    const acquired = g.begin(center);
    g.move([center[0], center[1] - 20]);
    const escaped = j.value;
    g.move([center[0], center[1] - 1000]);
    const limited = j.value;
    g.move([center[0], center[1] - 990]);
    const reversed = j.value;
    g.end(true);
    g.dispose();
    return { acquired, escaped, limited, reversed, restored: j.value };
  });
  assert.equal(r.acquired, true);
  assert.ok(r.escaped > 0);
  assert.equal(r.limited, 1);
  assert.ok(r.reversed < 0.999 && r.reversed > 0.8);
  assert.equal(r.restored, 0);
});

for (const frame of ['body', 'world'])
  test(`ball ${frame} frame composes rotation in the native order and cancellation restores initial quaternion`, async () => {
    const r = await gesture('ball-' + frame);
    const delta = await page.evaluate(
      ({ start, current, frame }) => {
        const multiply = (a, b) => [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
          ],
          inverse = [-start[0], -start[1], -start[2], start[3]];
        return frame === 'body' ? multiply(inverse, current) : multiply(current, inverse);
      },
      { start: r.originalRotation, current: r.moved.rotation, frame },
    );
    assert.equal(r.acquired, true);
    assert.ok(Math.abs(delta[0]) + Math.abs(delta[1]) < 1e-8);
    assert.ok(Math.abs(delta[2]) > 0.01);
    assert.deepEqual(r.restored.rotation, r.originalRotation);
    assert.deepEqual(r.moved.origin, r.origin);
  });
