import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import * as THREE from 'three';
import { CameraFocusTransition } from '../src/viewport/camera-focus.ts';
import { cameraForTransition } from '../src/viewport/camera-transition.ts';
import { sceneFocusBounds, sceneNodeBounds } from '../src/viewport/focus-bounds.ts';

const reference = JSON.parse(
  await readFile(new URL('./fixtures/camera-focus-native.json', import.meta.url), 'utf8'),
);
const close = (actual, expected, label, tolerance = reference.tolerance) => {
  const a = Array.isArray(actual) ? actual.flat(Infinity) : [actual],
    b = Array.isArray(expected) ? expected.flat(Infinity) : [expected];
  assert.equal(a.length, b.length, `${label}: length`);
  for (let index = 0; index < a.length; index++)
    assert.ok(
      Number.isFinite(a[index]) && Math.abs(a[index] - b[index]) <= tolerance,
      `${label}[${index}]: ${a[index]} != ${b[index]}`,
    );
};
const rows = (matrix) => matrix.clone().transpose().elements;

for (const fixture of reference.cases)
  test(`native focus animation and framing: ${fixture.id}`, () => {
    const transition = new CameraFocusTransition(fixture.start, fixture.bounds, fixture.options);
    let reused;
    for (const step of fixture.steps) {
      const actual = transition.sample(step.elapsed),
        expected = step.expected.view,
        label = `${fixture.id} ${step.elapsed}s`;
      for (const key of ['position', 'target', 'up', 'fov', 'near', 'far', 'orthoHeight'])
        close(actual[key], expected[key], `${label} ${key}`);
      reused = cameraForTransition(actual, actual.resolution[0] / actual.resolution[1], reused);
      close(rows(reused.matrixWorldInverse), step.expected.viewMatrix, `${label} view`);
      close(rows(reused.projectionMatrix), step.expected.projectionMatrix, `${label} projection`);
    }
  });

function scopeFixture() {
  const root = new THREE.Group(),
    nodes = [],
    joints = [];
  const add = (id, kind, parent = root, position = [0, 0, 0]) => {
    const object =
      kind === 'geom' ? new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2)) : new THREE.Group();
    object.position.fromArray(position);
    parent.add(object);
    object.userData.nodeId = id;
    object.userData.kind = kind;
    const node = { id, name: id, kind, object, visible: true, parentId: parent.userData.nodeId };
    nodes.push(node);
    return node;
  };
  const world = add('world', 'world'),
    model = add('model', 'robot', world.object),
    body = add('body', 'link', model.object),
    own = add('own', 'geom', body.object),
    joint = add('joint', 'joint', body.object, [10, 0, 0]),
    child = add('child', 'link', joint.object),
    childGeom = add('childGeom', 'geom', child.object),
    camera = add('camera', 'camera', body.object, [2, 3, 4]);
  add('cameraAttachment', 'geom', camera.object, [500, 0, 0]);
  joints.push({ id: 'joint', bodyId: 'child', child: 'child', object: joint.object });
  return { doc: { root, nodes, joints }, add, body, own, model, joint, child, childGeom, camera };
}

test('focus scopes distinguish own body, descendant model and the selected geometry', () => {
  const { doc, own, childGeom } = scopeFixture();
  // Camera/light attachments do not enlarge the bounds of their owning body.
  close(sceneNodeBounds(doc, 'body').center, [0, 0, 0], 'body center');
  close(sceneNodeBounds(doc, 'body').half, [1, 1, 1], 'body own geometry');
  close(sceneNodeBounds(doc, 'joint').center, [10, 0, 0], 'joint child body');
  close(sceneNodeBounds(doc, 'joint', 'local').center, [0, 0, 0], 'joint body local frame');
  close(sceneNodeBounds(doc, 'own').half, [1, 1, 1], 'single geom');
  own.object.visible = false;
  childGeom.object.visible = false;
  close(sceneNodeBounds(doc, 'body').half, [1, 1, 1], 'hidden geometry remains included');
  assert.ok(
    sceneNodeBounds(doc, 'model').half[0] > 200,
    'model includes descendant finite geometry',
  );
});

test('camera/light fallback is centered at authored world position and scales with finite scene bounds', () => {
  const { doc, camera, add, body } = scopeFixture();
  const light = add('light', 'light', body.object, [-2, 4, 1]);
  const ground = add('ground', 'geom', doc.root, [10000, 0, 0]);
  ground.object.userData.fitExcluded = true;
  const scene = sceneFocusBounds(doc),
    half = (Math.hypot(...scene.half) * 2 * 0.025) / Math.sqrt(3);
  for (const node of [camera, light]) {
    assert.equal(sceneNodeBounds(doc, node.id), null);
    close(
      sceneFocusBounds(doc, node.id).center,
      node.object.getWorldPosition(new THREE.Vector3()).toArray(),
      node.id + ' position',
    );
    close(
      sceneFocusBounds(doc, node.id).half,
      [half, half, half],
      node.id + ' scene relative extent',
    );
  }
  assert.deepEqual(sceneFocusBounds(doc, 'world'), scene);
  assert.equal(sceneNodeBounds(doc, 'ground'), null);
  assert.ok(scene.half[0] < 1000, 'infinite ground is excluded from full scene bounds');
});

test('local bounds apply the combined transform once and remain finite with malformed vertex samples', () => {
  const { doc, body, own } = scopeFixture();
  body.object.position.set(2, -1, 3);
  body.object.rotation.z = Math.PI / 4;
  own.object.rotation.z = -Math.PI / 4;
  close(sceneNodeBounds(doc, 'body').half, [1, 1, 1], 'canceling rotations in world');
  close(sceneNodeBounds(doc, 'body', 'local').half, [Math.SQRT2, Math.SQRT2, 1], 'body frame');
  own.object.geometry = new THREE.BufferGeometry();
  own.object.geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-1, -1, -1, 1, 1, 1, NaN, 2, 3], 3),
  );
  close(sceneNodeBounds(doc, 'body').half, [1, 1, 1], 'finite vertices');
});

test('geometry-free scenes use native half-unit scene fallback without arbitrary per-object cubes', () => {
  const root = new THREE.Group(),
    camera = new THREE.PerspectiveCamera();
  camera.position.set(7, -2, 3);
  root.add(camera);
  const doc = {
    root,
    nodes: [{ id: 'camera', name: 'camera', kind: 'camera', object: camera, visible: true }],
    joints: [],
  };
  close(sceneFocusBounds(doc).half, [0.5, 0.5, 0.5], 'empty scene');
  close(
    sceneFocusBounds(doc, 'camera').half,
    [0.025, 0.025, 0.025],
    'empty scene camera placeholder',
  );
});

test('selected body/model/geom bounds match independent native geometry fixtures in both frames', async () => {
  const fixture = JSON.parse(
    await readFile(new URL('./fixtures/focus-bounds-native.json', import.meta.url), 'utf8'),
  );
  const root = new THREE.Group(),
    nodes = [],
    byId = new Map();
  for (const item of fixture.nodes) {
    const object = new THREE.Group(),
      parent = byId.get(item.parent)?.object ?? root;
    const world = new THREE.Matrix4().set(...item.matrix.flat());
    parent.updateWorldMatrix(true, false);
    const local = parent.matrixWorld.clone().invert().multiply(world);
    local.decompose(object.position, object.quaternion, object.scale);
    object.userData = { nodeId: item.id, kind: item.kind, fitExcluded: item.infinite };
    object.visible = item.visible;
    parent.add(object);
    if (item.size) {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2));
      mesh.scale.fromArray(item.size);
      object.add(mesh);
    }
    const node = {
      id: item.id,
      name: item.id,
      kind: item.kind,
      object,
      parentId: item.parent,
      visible: item.visible,
    };
    nodes.push(node);
    byId.set(item.id, node);
  }
  const doc = { root, nodes, joints: [] };
  for (const check of fixture.checks)
    for (const space of ['local', 'world']) {
      const actual = sceneNodeBounds(doc, check.id, space),
        expected = check[space];
      if (expected === null) assert.equal(actual, null, `${check.id} ${space}`);
      else {
        close(actual.center, expected.center, `${check.id} ${space} center`);
        close(actual.half, expected.half, `${check.id} ${space} half`);
      }
    }
  const scene = sceneFocusBounds(doc);
  close(scene.center, fixture.scene.center, 'finite scene center');
  close(scene.half, fixture.scene.half, 'finite scene half');
});
