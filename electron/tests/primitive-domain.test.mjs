import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import { mkdir } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';

let api;
before(async () => {
  const output = path.resolve('../output/electron/verification/primitive-domain');
  await mkdir(output, { recursive: true });
  const outfile = path.join(output, 'domain.mjs');
  await build({
    stdin: {
      contents: `export * as THREE from 'three';
        export * from './src/scene.ts';
        export * from './src/editor.ts';
        export * from './src/scene-tree.ts';`,
      resolveDir: process.cwd(),
    },
    outfile,
    bundle: true,
    platform: 'node',
    format: 'esm',
    logLevel: 'silent',
  });
  api = await import(pathToFileURL(outfile));
});

function emptyDocument() {
  return {
    name: 'Primitives',
    root: new api.THREE.Group(),
    nodes: [],
    joints: [],
    files: [],
    warnings: [],
  };
}
function add(document, kind) {
  const node = api.createPrimitive(kind);
  document.nodes.push(node);
  document.root.add(node.object);
  return node;
}
function close(actual, expected, label) {
  assert.equal(actual.length, expected.length, label);
  actual.forEach((value, index) =>
    assert.ok(Math.abs(value - expected[index]) < 1e-6, `${label}[${index}]`),
  );
}
function geometryState(node) {
  const geometry = node.object.geometry;
  const components = (array) => Array.from(array, (value) => (value === 0 ? 0 : value));
  return {
    position: components(geometry.getAttribute('position').array),
    normal: components(geometry.getAttribute('normal').array),
    uv: components(geometry.getAttribute('uv').array),
    index: [...geometry.index.array],
  };
}

function closePose(actual, expected) {
  assert.deepEqual(Object.keys(actual.transforms), Object.keys(expected.transforms));
  for (const [id, transform] of Object.entries(expected.transforms)) {
    for (const component of ['p', 'q', 's'])
      close(actual.transforms[id][component], transform[component], `${id}.${component}`);
  }
  assert.deepEqual(actual.joints, expected.joints);
  assert.deepEqual(actual.jointOrigins, expected.jointOrigins);
  assert.deepEqual(actual.jointConfigurations, expected.jointConfigurations);
}

test('existing geometric primitive defaults match native sizes while cameras and lights retain their setup', () => {
  const document = emptyDocument();
  try {
    for (const kind of ['box', 'sphere', 'cylinder']) {
      const node = add(document, kind),
        bounds = new api.THREE.Box3().setFromObject(node.object);
      close(bounds.min.toArray(), [-0.5, -0.5, 0], `${kind} minimum`);
      close(bounds.max.toArray(), [0.5, 0.5, 1], `${kind} maximum`);
    }
    const ground = add(document, 'ground'),
      bounds = new api.THREE.Box3().setFromObject(ground.object);
    close(bounds.min.toArray(), [-4, -4, 0], 'ground minimum');
    close(bounds.max.toArray(), [4, 4, 0], 'ground maximum');
    assert.equal(ground.properties.Width, 8);
    assert.equal(ground.properties.Length, 8);
    assert.equal(ground.object.userData.viewportPickable, false);
    const camera = add(document, 'camera'),
      light = add(document, 'light');
    close(camera.object.position.toArray(), [3, -3, 2], 'camera position');
    assert.equal(camera.object.fov, 45);
    close(light.object.position.toArray(), [2, -2, 3], 'light position');
    assert.equal(light.object.intensity, 14);
  } finally {
    api.disposeDocument(document);
  }
});

test('cone and ellipsoid use native Z-up shape dimensions and participate in geometry filters', () => {
  const document = emptyDocument();
  try {
    const cone = add(document, 'cone'),
      ellipsoid = add(document, 'ellipsoid');
    const coneBounds = new api.THREE.Box3().setFromObject(cone.object);
    const ellipsoidBounds = new api.THREE.Box3().setFromObject(ellipsoid.object);
    close(coneBounds.min.toArray(), [-0.5, -0.5, 0], 'cone minimum');
    close(coneBounds.max.toArray(), [0.5, 0.5, 1], 'cone maximum');
    close(ellipsoidBounds.min.toArray(), [-0.65, -0.45, 0], 'ellipsoid minimum');
    close(ellipsoidBounds.max.toArray(), [0.65, 0.45, 0.7], 'ellipsoid maximum');
    const positions = cone.object.geometry.getAttribute('position');
    let apex = 0;
    for (let i = 0; i < positions.count; i++)
      if (positions.getZ(i) > 0.49999) {
        close([positions.getX(i), positions.getY(i)], [0, 0], 'positive Z apex');
        apex++;
      }
    assert.ok(apex > 0);
    assert.deepEqual(
      api.filterSceneNodes(document.nodes, '', 'geom').map((node) => node.id),
      [cone.id, ellipsoid.id],
    );
    assert.deepEqual(
      api.filterSceneNodes(document.nodes, 'ellipsoid', 'geom').map((node) => node.id),
      [ellipsoid.id],
    );
    for (const node of document.nodes) {
      assert.equal(node.object.userData.physicsOwned, undefined);
      assert.equal(node.object.userData.nodeId, node.id);
      assert.equal(node.object.userData.role, 'visual');
      assert.equal(api.canRemoveNode(document, node.id), true);
    }
  } finally {
    api.disposeDocument(document);
  }
});

test('primitive creation and deletion restore through the same document snapshots used by Undo and Redo', () => {
  const documents = [],
    document = emptyDocument();
  documents.push(document);
  try {
    const before = api.encodeDocument(document);
    const nodes = ['cone', 'ellipsoid', 'sphere', 'cylinder', 'capsule'].map((kind) =>
      add(document, kind),
    );
    const expected = nodes.map(geometryState),
      after = api.encodeDocument(document);
    const undo = api.decodeDocument(before),
      redo = api.decodeDocument(after);
    documents.push(undo, redo);
    assert.equal(undo.nodes.length, 0);
    assert.deepEqual(
      redo.nodes.map((node) => node.id),
      nodes.map((node) => node.id),
    );
    assert.deepEqual(redo.nodes.map(geometryState), expected);
    assert.deepEqual(api.removeNode(redo, nodes[0].id), [nodes[0].id]);
    assert.deepEqual(
      redo.nodes.map((node) => node.kind),
      ['ellipsoid', 'sphere', 'cylinder', 'capsule'],
    );
    const restored = api.decodeDocument(after);
    documents.push(restored);
    assert.deepEqual(restored.nodes.map(geometryState), expected);
  } finally {
    documents.forEach(api.disposeDocument);
  }
});

test('saved projects reopen cone and ellipsoid geometry, edits and full-scene timeline poses without physics', () => {
  const document = emptyDocument();
  let reopened;
  try {
    const cone = add(document, 'cone'),
      ellipsoid = add(document, 'ellipsoid');
    for (const kind of ['sphere', 'cylinder', 'capsule']) add(document, kind);
    const expected = document.nodes.map(geometryState);
    const start = api.capturePose(document);
    cone.object.position.set(2, 3, 4);
    cone.object.rotation.set(0.4, -0.2, 0.7);
    ellipsoid.object.scale.set(1.2, 0.7, 1.8);
    ellipsoid.object.material.color.set('#cc3344');
    const end = api.capturePose(document);
    const keys = [
      { id: 'start', time: 0, pose: start },
      { id: 'end', time: 2, pose: end },
    ];
    reopened = api.loadWorkspace(api.saveWorkspace(document, keys, 2));
    assert.deepEqual(
      reopened.doc.nodes.map((node) => node.kind),
      ['cone', 'ellipsoid', 'sphere', 'cylinder', 'capsule'],
    );
    assert.deepEqual(reopened.doc.nodes.map(geometryState), expected);
    closePose(api.capturePose(reopened.doc), end);
    assert.equal(reopened.doc.nodes[1].object.material.color.getHexString(), 'cc3344');
    api.samplePose(reopened.doc, reopened.keys, 0);
    closePose(api.capturePose(reopened.doc), start);
    api.samplePose(reopened.doc, reopened.keys, 2);
    closePose(api.capturePose(reopened.doc), end);
    assert.equal(api.canRemoveNode(reopened.doc, ellipsoid.id), true);
  } finally {
    api.disposeDocument(document);
    if (reopened) api.disposeDocument(reopened.doc);
  }
});
