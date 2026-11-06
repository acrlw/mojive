import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { mkdtemp, mkdir, writeFile, symlink, rm, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createServer } from 'vite';
import { chromium } from 'playwright';
const require = createRequire(import.meta.url);
const { Projects } = require('../main/resource-paths.cjs');
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
    root: path.resolve('.'),
    server: { host: '127.0.0.1', port: 0 },
  });
  server.middlewares.use('/__scene_test', (_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><base href="/"><title>Scene tests</title>');
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__scene_test`);
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('demo parses link identity, hierarchy, joint limits and mimic transforms', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, setJoint, disposeDocument } = await import('/src/scene.ts');
    const doc = await createDemo();
    setJoint(doc, 'shoulder', 99);
    setJoint(doc, 'gripper', 0.02);
    const meshes = [];
    doc.root.traverse((object) => {
      if (object.isMesh) meshes.push({ role: object.userData.role, id: object.userData.nodeId });
    });
    const shoulder = doc.joints.find((item) => item.name === 'shoulder');
    const left = doc.joints.find((item) => item.name === 'gripper'),
      right = doc.joints.find((item) => item.name === 'gripper_mimic');
    const result = {
      count: doc.nodes.length,
      joints: doc.joints.length,
      meshes,
      limit: shoulder.value,
      left: left.object.position.x,
      right: right.object.position.x,
      mimic: right.value,
      actuators: doc.root.userData.actuators.length,
      validParents: doc.nodes.every(
        (node) => !node.parentId || doc.nodes.some((parent) => parent.id === node.parentId),
      ),
    };
    disposeDocument(doc);
    return result;
  });
  assert.equal(result.joints, 7);
  assert.equal(result.count, 45);
  assert.equal(result.actuators, 3);
  assert.equal(result.limit, 1.9);
  assert.equal(result.mimic, 0.02);
  assert.ok(Math.abs(result.left - 0.09) < 1e-9);
  assert.ok(Math.abs(result.right + 0.09) < 1e-9);
  assert.ok(result.validParents);
  assert.ok(result.meshes.some((item) => item.role === 'collision'));
  assert.ok(result.meshes.every((item) => item.id.startsWith('geom:')));
});

test('URDF origin uses fixed-axis Rz Ry Rx and prismatic motion uses local joint axis', async () => {
  const result = await page.evaluate(async () => {
    const { loadUrdf, setJoint, disposeDocument } = await import('/src/scene.ts');
    const doc = await loadUrdf(
      '<robot name="frames"><link name="a"/><link name="b"/><joint name="slide" type="prismatic"><parent link="a"/><child link="b"/><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/><limit lower="-2" upper="2"/></joint></robot>',
    );
    setJoint(doc, 'slide', 0.5);
    const object = doc.joints[0].object;
    const result = object.position.toArray();
    disposeDocument(doc);
    return result;
  });
  assert.ok(Math.abs(result[0] - 1) < 1e-9);
  assert.ok(Math.abs(result[1] - 2.5) < 1e-9);
  assert.equal(result[2], 3);
});

test('invalid hierarchy, malformed numbers and mimic cycles reject without a partial result', async () => {
  const results = await page.evaluate(async () => {
    const { loadUrdf } = await import('/src/scene.ts');
    const documents = [
      '<robot name="bad"><link name="a"/><link name="a"/></robot>',
      '<robot name="bad"><link name="a"/><link name="b"/><joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint></robot>',
      '<robot name="bad"><link name="a"><visual><geometry><box size="1 NaN 1"/></geometry></visual></link></robot>',
      '<robot name="bad"><link name="a"/><link name="b"/><link name="c"/><joint name="j1" type="revolute"><parent link="a"/><child link="b"/><mimic joint="j2"/></joint><joint name="j2" type="revolute"><parent link="b"/><child link="c"/><mimic joint="j1"/></joint></robot>',
    ];
    return Promise.all(
      documents.map(async (text) => {
        try {
          await loadUrdf(text);
          return '';
        } catch (error) {
          return error.message;
        }
      }),
    );
  });
  assert.match(results[0], /Duplicate link/);
  assert.match(results[1], /cycle/);
  assert.match(results[2], /Invalid box size/);
  assert.match(results[3], /Mimic joint cycle/);
});

test('relative STL resources load and missing resources reject explicitly', async () => {
  const result = await page.evaluate(async () => {
    const { loadUrdf, disposeDocument } = await import('/src/scene.ts');
    const xml = (filename) =>
      `<robot name="mesh"><link name="body"><visual><origin xyz="0 0 2"/><geometry><mesh filename="${filename}" scale="2 2 2"/></geometry></visual></link></robot>`;
    const doc = await loadUrdf(xml('triangle.stl'), new URL('/demo/', location.href).href);
    let positions = 0;
    doc.root.traverse((object) => {
      if (object.isMesh) positions = object.geometry.getAttribute('position').count;
    });
    disposeDocument(doc);
    let failure = '';
    try {
      await loadUrdf(xml('missing.stl'), new URL('/demo/', location.href).href);
    } catch (error) {
      failure = error.message;
    }
    return { positions, failure };
  });
  assert.equal(result.positions, 3);
  assert.ok(result.failure);
});

test('local resources preserve spaces and package paths and reject symlink escapes', async () => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), 'mojive-resources-'));
  try {
    const root = path.join(temporary, 'my robot');
    await mkdir(path.join(root, 'meshes'), { recursive: true });
    const mesh = path.join(root, 'meshes', 'part one.stl');
    await writeFile(mesh, 'solid a\nendsolid a');
    await writeFile(
      path.join(root, 'package.xml'),
      '<package><name>robot_description</name></package>',
    );
    const outside = path.join(temporary, 'private.txt');
    await writeFile(outside, 'not a resource');
    await symlink(outside, path.join(root, 'escape.stl'));
    const projects = new Projects(),
      token = await projects.register(root);
    assert.equal(
      await projects.resolve(`mojive-resource://project/${token}/meshes/part%20one.stl`),
      await realpath(mesh),
    );
    assert.equal(
      await projects.resolve(
        `mojive-resource://project/${token}/@package/robot_description/meshes/part%20one.stl`,
      ),
      await realpath(mesh),
    );
    await assert.rejects(
      projects.resolve(`mojive-resource://project/${token}/escape.stl`),
      /outside/,
    );
    await assert.rejects(projects.authorized(outside), /outside/);
    await assert.rejects(
      projects.resolve(`mojive-resource://project/${token}/%2Fprivate.txt`),
      /Invalid resource/,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test('serialized scenes preserve editable identities, joint origins, authored entities and pose', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, setJoint, createPrimitive, disposeDocument } = await import(
      '/src/scene.ts'
    );
    const { encodeDocument, decodeDocument, saveWorkspace, loadWorkspace, capturePose, applyPose } =
      await import('/src/editor.ts');
    const doc = await createDemo();
    const primitive = createPrimitive('box');
    doc.root.add(primitive.object);
    doc.nodes.push(primitive);
    primitive.object.position.set(2, 3, 4);
    setJoint(doc, 'elbow', 0.5);
    const pose = capturePose(doc);
    setJoint(doc, 'elbow', 0.8);
    applyPose(doc, pose);
    const data = encodeDocument(doc);
    const loaded = decodeDocument(data);
    const saved = loadWorkspace(saveWorkspace(loaded, [{ id: 'key', time: 1, pose }], 10));
    const rootIsCollection = saved.doc.nodes.every((node) => node.object !== saved.doc.root);
    const elbow = saved.doc.joints.find((joint) => joint.name === 'elbow');
    setJoint(saved.doc, elbow.id, 0.75);
    const robot = saved.doc.nodes.find((node) => node.id === 'robot');
    robot.object.removeFromParent();
    const box = saved.doc.nodes.find((node) => node.id === primitive.id);
    const result = {
      rootIsCollection,
      identity: box.object.uuid === primitive.object.uuid,
      position: box.object.position.toArray(),
      parentIsScene: box.object.parent === saved.doc.root,
      elbow: elbow.value,
      keyCount: saved.keys.length,
      actuators: saved.doc.root.userData.actuators.length,
    };
    disposeDocument(doc);
    disposeDocument(loaded);
    disposeDocument(saved.doc);
    return result;
  });
  assert.equal(result.rootIsCollection, true);
  assert.equal(result.identity, true);
  assert.deepEqual(result.position, [2, 3, 4]);
  assert.equal(result.parentIsScene, true);
  assert.equal(result.elbow, 0.75);
  assert.equal(result.keyCount, 1);
  assert.equal(result.actuators, 3);
});

test('snapshot serialization records immediate transform edits before any render frame', async () => {
  const result = await page.evaluate(async () => {
    const { loadUrdf, createPrimitive, disposeDocument } = await import('/src/scene.ts');
    const { encodeDocument, decodeDocument, saveWorkspace, loadWorkspace } = await import(
      '/src/editor.ts'
    );
    const doc = await loadUrdf('<robot name="Immediate"><link name="base"/></robot>', '');
    const node = createPrimitive('box');
    doc.root.add(node.object);
    doc.nodes.push(node);
    const before = encodeDocument(doc);
    // Keep both snapshots in one JS task: no renderer/RAF may update cached matrices.
    node.object.position.set(2, 3, 4);
    node.object.rotation.set(0.2, -0.3, 0.4);
    node.object.scale.set(2, 3, 4);
    const after = encodeDocument(doc);
    const undo = decodeDocument(before),
      redo = decodeDocument(after);
    const saved = loadWorkspace(saveWorkspace(doc, [], 10));
    const transform = (document) => {
      const object = document.nodes.find((entry) => entry.id === node.id).object;
      return {
        p: object.position.toArray(),
        q: object.quaternion.toArray(),
        s: object.scale.toArray(),
      };
    };
    const output = {
      before: transform(undo),
      after: transform(redo),
      saved: transform(saved.doc),
      expected: transform(doc),
      changed: JSON.stringify(before) !== JSON.stringify(after),
    };
    [doc, undo, redo, saved.doc].forEach(disposeDocument);
    return output;
  });
  assert.ok(result.changed, 'immediate edits must form their own undo transaction');
  assert.deepEqual(result.before.p, [0, 0, 0.25]);
  for (const state of [result.after, result.saved])
    for (const key of ['p', 'q', 's'])
      state[key].forEach((value, index) =>
        assert.ok(Math.abs(value - result.expected[key][index]) < 1e-12),
      );
});

test('embedded material textures survive project serialization without external resource URLs', async () => {
  const result = await page.evaluate(async () => {
    const { loadUrdf, disposeDocument } = await import('/src/scene.ts');
    const { encodeDocument, decodeDocument } = await import('/src/editor.ts');
    const png =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6QmcAAAAASUVORK5CYII=';
    const doc = await loadUrdf(
      `<robot name="texture"><material name="test"><texture filename="${png}"/></material><link name="body"><visual><geometry><box size="1 1 1"/></geometry><material name="test"/></visual></link></robot>`,
    );
    const data = encodeDocument(doc);
    const loaded = decodeDocument(data);
    let hasMap = false;
    loaded.root.traverse((object) => {
      if (object.isMesh) hasMap = Boolean(object.material.map);
    });
    const result = {
      embedded: data.root.images?.every(
        (image) => typeof image.url === 'string' && image.url.startsWith('data:'),
      ),
      hasMap,
    };
    disposeDocument(doc);
    disposeDocument(loaded);
    return result;
  });
  assert.equal(result.embedded, true);
  assert.equal(result.hasMap, true);
});

test('joint frame authoring and timeline interpolation preserve origins across position changes', async () => {
  const result = await page.evaluate(async () => {
    const { loadUrdf, setJoint, authorJointTransform, disposeDocument } = await import(
      '/src/scene.ts'
    );
    const { capturePose, applyPose, samplePose } = await import('/src/editor.ts');
    const doc = await loadUrdf(
      '<robot name="authoring"><link name="a"/><link name="b"/><joint name="hinge" type="revolute"><parent link="a"/><child link="b"/><origin xyz="1 2 3"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint></robot>',
    );
    const joint = doc.joints[0];
    setJoint(doc, joint.id, 0.5);
    joint.object.position.x = 4;
    joint.object.rotation.x = 0.3;
    authorJointTransform(doc, joint.id);
    const authored = joint.object.matrix.toArray();
    setJoint(doc, joint.id, 0.5);
    const stable = authored.every(
      (value, index) => Math.abs(value - joint.object.matrix.elements[index]) < 1e-9,
    );
    const a = capturePose(doc);
    joint.object.position.x = 8;
    authorJointTransform(doc, joint.id);
    const b = capturePose(doc);
    samplePose(
      doc,
      [
        { id: 'a', time: 0, pose: a },
        { id: 'b', time: 2, pose: b },
      ],
      1,
    );
    const middle = joint.object.position.x;
    setJoint(doc, joint.id, 0.9);
    const afterValue = joint.object.position.x;
    applyPose(doc, a);
    const restored = joint.object.position.x;
    disposeDocument(doc);
    return { stable, middle, afterValue, restored };
  });
  assert.equal(result.stable, true);
  assert.ok(Math.abs(result.middle - 6) < 1e-9);
  assert.ok(Math.abs(result.afterValue - 6) < 1e-9);
  assert.ok(Math.abs(result.restored - 4) < 1e-9);
});

test('malformed saved keyframes and disconnected joint identities reject before replacing a valid scene', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, disposeDocument } = await import('/src/scene.ts');
    const { saveWorkspace, loadWorkspace, capturePose } = await import('/src/editor.ts');
    const source = await createDemo();
    const original = saveWorkspace(
      source,
      [{ id: 'pose', time: 1, pose: capturePose(source) }],
      10,
    );
    const badKey = JSON.parse(original);
    badKey.keys = [{}];
    const badPosition = JSON.parse(original);
    badPosition.keys[0].pose.transforms.robot.p = [1, 2];
    const badJoint = JSON.parse(original);
    badJoint.document.joints[0].child = 'missing';
    const badOrigin = JSON.parse(original);
    badOrigin.document.joints[0].origin = Array(16).fill(0);
    const badIdentity = JSON.parse(original);
    badIdentity.document.nodes[1].id = 'robot';
    const outcomes = [badKey, badPosition, badJoint, badOrigin, badIdentity].map((data) => {
      try {
        loadWorkspace(JSON.stringify(data));
        return '';
      } catch (error) {
        return error.message;
      }
    });
    const unchanged =
      saveWorkspace(source, [{ id: 'pose', time: 1, pose: capturePose(source) }], 10) === original;
    const restored = loadWorkspace(original);
    const count = restored.doc.nodes.length;
    disposeDocument(source);
    disposeDocument(restored.doc);
    return { outcomes, unchanged, count };
  });
  assert.ok(result.outcomes.every(Boolean));
  assert.equal(result.unchanged, true);
  assert.equal(result.count, 45);
});

test('deleting a link removes its incoming joint and dependent mimic branch without dangling references', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, removeNode, disposeDocument } = await import('/src/scene.ts');
    const { saveWorkspace, loadWorkspace, capturePose } = await import('/src/editor.ts');
    const doc = await createDemo();
    const pose = capturePose(doc);
    const removed = removeNode(doc, 'link:left_finger');
    const linkNames = new Set(
      doc.nodes.filter((node) => node.kind === 'link').map((node) => node.name),
    );
    const jointNames = new Set(doc.joints.map((joint) => joint.name));
    const valid = doc.joints.every(
      (joint) =>
        linkNames.has(joint.parent) &&
        linkNames.has(joint.child) &&
        (!joint.mimic || jointNames.has(joint.mimic.joint)),
    );
    const loaded = loadWorkspace(saveWorkspace(doc, [{ id: 'prior-pose', time: 0, pose }], 10));
    const first = { removed, valid, count: loaded.doc.nodes.length };
    removeNode(doc, 'link:upper_arm');
    const remaining = doc.joints.map((joint) => joint.name),
      actuators = doc.root.userData.actuators.length;
    disposeDocument(doc);
    disposeDocument(loaded.doc);
    return { ...first, remaining, actuators };
  });
  assert.deepEqual(
    result.removed.filter((id) => !id.startsWith('geom:')).sort(),
    ['joint:gripper', 'joint:gripper_mimic', 'link:left_finger', 'link:right_finger'].sort(),
  );
  assert.equal(result.valid, true);
  assert.equal(result.count, 35);
  assert.equal(result.removed.filter((id) => id.startsWith('geom:')).length, 6);
  assert.deepEqual(result.remaining, ['base_rotation']);
  assert.equal(result.actuators, 0);
});

test('deleting objects retains shared resources until the last owner and leaves remaining authored objects usable', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, createPrimitive, removeNode, disposeDocument } = await import(
      '/src/scene.ts'
    );
    const { saveWorkspace, loadWorkspace } = await import('/src/editor.ts');
    const doc = await createDemo();
    const a = createPrimitive('box'),
      b = createPrimitive('box');
    b.object.geometry.dispose();
    b.object.material.dispose();
    b.object.geometry = a.object.geometry;
    b.object.material = a.object.material;
    doc.root.add(a.object, b.object);
    doc.nodes.push(a, b);
    let disposed = 0;
    a.object.geometry.addEventListener('dispose', () => disposed++);
    removeNode(doc, a.id);
    const retained = disposed === 0 && b.object.parent === doc.root;
    removeNode(doc, 'robot');
    const saved = loadWorkspace(saveWorkspace(doc, [], 10));
    const remaining = saved.doc.nodes.map((node) => node.id);
    removeNode(doc, b.id);
    const released = disposed === 1;
    disposeDocument(doc);
    disposeDocument(saved.doc);
    return { retained, remaining, b: b.id, released };
  });
  assert.equal(result.retained, true);
  assert.deepEqual(result.remaining, [result.b]);
  assert.equal(result.released, true);
});

test('document replacement disposes shadow materials, light targets and skybox exactly once', async () => {
  const counts = await page.evaluate(async () => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { disposeDocument } = await import('/src/scene.ts');
    const { setSceneRendering } = await import('/src/scene-rendering.ts');
    const root = new THREE.Group();
    const texture = new THREE.Texture(),
      skybox = new THREE.CubeTexture();
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshStandardMaterial({ map: texture });
    const depth = new THREE.MeshDepthMaterial({ map: texture });
    const distance = new THREE.MeshDistanceMaterial({ map: texture });
    for (let index = 0; index < 2; index++) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.customDepthMaterial = depth;
      mesh.customDistanceMaterial = distance;
      root.add(mesh);
    }
    const light = new THREE.DirectionalLight();
    light.shadow.map = new THREE.WebGLRenderTarget(16, 16);
    light.shadow.mapPass = new THREE.WebGLRenderTarget(16, 16);
    root.add(light);
    setSceneRendering(root, { skybox });
    const resources = {
      geometry,
      material,
      depth,
      distance,
      texture,
      skybox,
      shadow: light.shadow.map,
      shadowPass: light.shadow.mapPass,
    };
    const counts = Object.fromEntries(Object.keys(resources).map((key) => [key, 0]));
    for (const [key, resource] of Object.entries(resources))
      resource.addEventListener('dispose', () => counts[key]++);
    disposeDocument({ root });
    return counts;
  });
  assert.deepEqual(counts, {
    geometry: 1,
    material: 1,
    depth: 1,
    distance: 1,
    texture: 1,
    skybox: 1,
    shadow: 1,
    shadowPass: 1,
  });
});

test('deleting a mesh retains shared shadow materials and textures until the last owner', async () => {
  const result = await page.evaluate(async () => {
    const THREE = await import('/node_modules/three/build/three.module.js');
    const { createDemo, createPrimitive, removeNode, disposeDocument } = await import(
      '/src/scene.ts'
    );
    const doc = await createDemo();
    const texture = new THREE.Texture();
    const depth = new THREE.MeshDepthMaterial({ map: texture });
    const distance = new THREE.MeshDistanceMaterial({ map: texture });
    const nodes = [createPrimitive('box'), createPrimitive('box')];
    for (const node of nodes) {
      node.object.customDepthMaterial = depth;
      node.object.customDistanceMaterial = distance;
      doc.root.add(node.object);
      doc.nodes.push(node);
    }
    const counts = { depth: 0, distance: 0, texture: 0 };
    for (const [key, resource] of Object.entries({ depth, distance, texture }))
      resource.addEventListener('dispose', () => counts[key]++);
    removeNode(doc, nodes[0].id);
    const retained = { ...counts };
    removeNode(doc, nodes[1].id);
    disposeDocument(doc);
    return { retained, released: counts };
  });
  assert.deepEqual(result.retained, { depth: 0, distance: 0, texture: 0 });
  assert.deepEqual(result.released, { depth: 1, distance: 1, texture: 1 });
});

test('Z-up cylinder and capsule vertices survive Undo snapshots, saved scenes and timeline sampling', async () => {
  const result = await page.evaluate(async () => {
    const { createDemo, createPrimitive, loadUrdf, disposeDocument, setJoint } = await import(
      '/src/scene.ts'
    );
    const {
      encodeDocument,
      decodeDocument,
      saveWorkspace,
      loadWorkspace,
      capturePose,
      samplePose,
    } = await import('/src/editor.ts');
    const doc = await createDemo();
    for (const kind of ['cylinder', 'capsule']) {
      const node = createPrimitive(kind);
      doc.root.add(node.object);
      doc.nodes.push(node);
    }
    const unusual = await loadUrdf(
      '<robot name="baked"><link name="body"><visual><geometry><capsule radius="0.1" length="0.7"/></geometry></visual><collision><geometry><cylinder radius="0.2" length="1.3"/></geometry></collision></link></robot>',
    );
    const fingerprint = (document) => {
      const result = new Map();
      document.root.updateMatrixWorld(true);
      document.root.traverse((object) => {
        if (object.isMesh) {
          object.geometry.computeBoundingBox();
          result.set(object.uuid, {
            position: Array.from(object.geometry.getAttribute('position').array),
            normal: Array.from(object.geometry.getAttribute('normal').array),
            bounds: [
              ...object.geometry.boundingBox.min.toArray(),
              ...object.geometry.boundingBox.max.toArray(),
            ],
          });
        }
      });
      return result;
    };
    const mismatch = (a, b) => {
      let count = 0;
      for (const [id, value] of a) if (JSON.stringify(value) !== JSON.stringify(b.get(id))) count++;
      return count;
    };
    const initial = fingerprint(doc),
      extraInitial = fingerprint(unusual);
    const undo = decodeDocument(encodeDocument(doc)),
      redo = decodeDocument(encodeDocument(undo));
    const extra = decodeDocument(encodeDocument(unusual));
    const poseA = capturePose(doc);
    setJoint(doc, 'elbow', -0.5);
    const poseB = capturePose(doc);
    const keys = [
      { id: 'a', time: 0, pose: poseA },
      { id: 'b', time: 2, pose: poseB },
    ];
    const saved = loadWorkspace(saveWorkspace(doc, keys, 10));
    samplePose(doc, keys, 1);
    samplePose(saved.doc, saved.keys, 1);
    const worldBounds = (document) => {
      const positions = [];
      document.root.updateMatrixWorld(true);
      document.root.traverse((object) => {
        if (object.isMesh) {
          const attr = object.geometry.getAttribute('position'),
            p = object.position.clone();
          for (let i = 0; i < attr.count; i++) {
            p.fromBufferAttribute(attr, i).applyMatrix4(object.matrixWorld);
            positions.push(...p.toArray());
          }
        }
      });
      return positions;
    };
    const before = worldBounds(doc),
      after = worldBounds(saved.doc);
    const result = {
      undo: mismatch(initial, fingerprint(undo)),
      redo: mismatch(initial, fingerprint(redo)),
      saved: mismatch(initial, fingerprint(saved.doc)),
      extra: mismatch(extraInitial, fingerprint(extra)),
      worldError: Math.max(...before.map((value, index) => Math.abs(value - after[index]))),
    };
    for (const value of [doc, undo, redo, unusual, extra, saved.doc]) disposeDocument(value);
    return result;
  });
  assert.equal(result.undo, 0);
  assert.equal(result.redo, 0);
  assert.equal(result.saved, 0);
  assert.equal(result.extra, 0);
  assert.ok(result.worldError < 1e-9);
});

test('native joint catalog reproduces all nine joint cases and independent geometry identities', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene, SCENE_CATALOG } = await import('/src/SceneCatalog.ts');
    const { disposeDocument } = await import('/src/scene.ts');
    const { saveWorkspace, loadWorkspace } = await import('/src/editor.ts');
    const results = [];
    for (const item of SCENE_CATALOG) {
      const doc = await createCatalogScene(item.id);
      const restored = loadWorkspace(saveWorkspace(doc, [], 10));
      results.push({
        id: item.id,
        count: doc.nodes.length,
        joints: doc.joints.map((j) => ({
          name: j.name,
          type: j.type,
          lower: j.lower,
          upper: j.upper,
          bodyId: j.bodyId,
        })),
        geometries: doc.nodes.filter((n) => n.kind === 'geom').length,
        lightDirection: restored.doc.nodes
          .find((n) => n.kind === 'light' && n.object.isDirectionalLight)
          ?.object.target.position.toArray(),
        validParents: restored.doc.nodes.every(
          (n) => !n.parentId || restored.doc.nodes.some((p) => p.id === n.parentId),
        ),
      });
      disposeDocument(doc);
      disposeDocument(restored.doc);
    }
    return results;
  });
  assert.equal(result.length, 4);
  assert.ok(result.every((item) => item.validParents));
  const native = result.find((item) => item.id === 'joint_gizmo');
  assert.equal(native.count, 45);
  assert.equal(native.joints.length, 9);
  assert.equal(native.geometries, 19);
  assert.deepEqual(native.lightDirection, [-0.15, 0.35, -1]);
  assert.equal(native.joints.filter((j) => j.bodyId === 'link:05_multi_joint').length, 3);
  assert.ok(native.joints.some((j) => j.type === 'ball'));
  assert.ok(native.joints.some((j) => j.type === 'free'));
  assert.ok(
    Math.abs(native.joints.find((j) => j.name === '06_precision_hinge_y').upper - Math.PI / 90) <
      1e-12,
  );
  assert.equal(native.joints.find((j) => j.name === '07_precision_slide_x').upper, 0.005);
});

test('ball/free configurations retain local frame, limits, timeline slerp and save roundtrips', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene, createNativeJointScene } = await import('/src/SceneCatalog.ts');
    const { setJointConfiguration, removeNode, disposeDocument } = await import('/src/scene.ts');
    const { capturePose, samplePose, saveWorkspace, loadWorkspace } = await import(
      '/src/editor.ts'
    );
    const doc = await createCatalogScene('joint_gizmo');
    const ball = doc.joints.find((j) => j.type === 'ball'),
      free = doc.joints.find((j) => j.type === 'free');
    const initial = capturePose(doc);
    setJointConfiguration(doc, ball.id, {
      rotation: [0, Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)],
    });
    setJointConfiguration(doc, free.id, {
      position: [0.4, 0.2, 0.6],
      rotation: [0, 0, Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)],
    });
    const keys = [
      { id: 'a', time: 0, pose: initial },
      { id: 'b', time: 2, pose: capturePose(doc) },
    ];
    samplePose(doc, keys, 1);
    const restored = loadWorkspace(saveWorkspace(doc, keys, 10));
    const restoredBall = restored.doc.joints.find((j) => j.type === 'ball'),
      restoredFree = restored.doc.joints.find((j) => j.type === 'free');
    const limited = createNativeJointScene(
      doc.source.replace('type="ball"', 'type="ball" range="0 25"'),
    );
    const limitedBall = limited.joints.find((j) => j.type === 'ball');
    setJointConfiguration(limited, limitedBall.id, { rotation: [0, 1, 0, 0] });
    const removed = removeNode(restored.doc, 'link:05_multi_joint');
    const valid = loadWorkspace(saveWorkspace(restored.doc, keys, 10));
    const result = {
      ballAngle: 2 * Math.acos(restoredBall.rotation.w),
      position: restoredFree.position.toArray(),
      freeAngle: 2 * Math.acos(restoredFree.rotation.w),
      limitedAngle: 2 * Math.acos(limitedBall.rotation.w),
      removedJoints: removed.filter((id) => id.startsWith('joint:')).length,
      remaining: valid.doc.joints.length,
    };
    [doc, restored.doc, limited, valid.doc].forEach(disposeDocument);
    return result;
  });
  assert.ok(Math.abs(result.ballAngle - Math.PI / 4) < 1e-9);
  assert.deepEqual(result.position, [0.2, 0.1, 0.3]);
  assert.ok(Math.abs(result.freeAngle - Math.PI / 6) < 1e-9);
  assert.ok(Math.abs(result.limitedAngle - (25 * Math.PI) / 180) < 1e-9);
  assert.equal(result.removedJoints, 3);
  assert.equal(result.remaining, 6);
});

test('malformed multi-DOF saves reject translated ball joints and incomplete pose configurations', async () => {
  const result = await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { disposeDocument } = await import('/src/scene.ts');
    const { capturePose, saveWorkspace, loadWorkspace } = await import('/src/editor.ts');
    const doc = await createCatalogScene('joint_gizmo');
    const ball = doc.joints.find((j) => j.type === 'ball');
    const text = saveWorkspace(doc, [{ id: 'pose', time: 0, pose: capturePose(doc) }], 10);
    const translated = JSON.parse(text);
    translated.document.joints.find((j) => j.type === 'ball').position = [1, 0, 0];
    const translatedPose = JSON.parse(text);
    translatedPose.keys[0].pose.jointConfigurations[ball.id].position = [1, 0, 0];
    const missing = JSON.parse(text);
    delete missing.keys[0].pose.jointConfigurations[ball.id];
    const results = [translated, translatedPose, missing].map((value) => {
      try {
        loadWorkspace(JSON.stringify(value));
        return '';
      } catch (error) {
        return error.message;
      }
    });
    disposeDocument(doc);
    return results;
  });
  assert.ok(result.every(Boolean));
  assert.match(result[0], /cannot translate/);
  assert.match(result[1], /cannot translate/);
  assert.match(result[2], /missing its joint configuration/);
});
