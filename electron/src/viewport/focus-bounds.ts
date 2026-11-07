import * as THREE from 'three';
import type { SceneDocument, SceneNode } from '../contracts';

export interface FocusBounds {
  center: number[];
  half: number[];
}

function centered(box: THREE.Box3): FocusBounds | null {
  if (box.isEmpty()) return null;
  return {
    center: box.getCenter(new THREE.Vector3()).toArray(),
    half: box.getSize(new THREE.Vector3()).multiplyScalar(0.5).toArray(),
  };
}

const geometryBounds = new WeakMap<THREE.BufferGeometry, {
  positions: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
  version: number;
  box: THREE.Box3;
}>();

function finiteGeometryBox(mesh: THREE.Mesh): THREE.Box3 | null {
  const positions = mesh.geometry.getAttribute('position');
  if (!positions) return null;
  const version = positions instanceof THREE.InterleavedBufferAttribute
    ? positions.data.version : positions.version;
  const cached = geometryBounds.get(mesh.geometry);
  // Rigid poses only change the world transform. Vertex edits invalidate through
  // the same attribute version used for GPU uploads, including interleaved data.
  if (cached?.positions === positions && cached.version === version)
    return cached.box.isEmpty() ? null : cached.box.clone();
  const box = new THREE.Box3(),
    point = new THREE.Vector3();
  for (let index = 0; index < positions.count; index++) {
    point.fromBufferAttribute(positions, index);
    if (Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z))
      box.expandByPoint(point);
  }
  geometryBounds.set(mesh.geometry, { positions, version, box });
  return box.isEmpty() ? null : box.clone();
}

function boundsFor(
  doc: SceneDocument,
  owner: THREE.Object3D,
  subtree: boolean,
  frameOwner?: THREE.Object3D,
): FocusBounds | null {
  doc.root.updateWorldMatrix(true, true);
  const nodes = new Map(doc.nodes.map((node) => [node.object, node])),
    box = new THREE.Box3(),
    frame = frameOwner ? frameOwner.matrixWorld.clone().invert() : new THREE.Matrix4(),
    world = new THREE.Matrix4();
  const visit = (object: THREE.Object3D) => {
    // Infinite planes never contribute to native bounds, even when selected.
    if (object.userData.fitExcluded) return;
    const node = nodes.get(object);
    if (
      object !== owner &&
      !subtree &&
      node &&
      ['link', 'joint', 'robot', 'world', 'camera', 'light'].includes(node.kind)
    )
      return;
    if (object instanceof THREE.Mesh) {
      const part = finiteGeometryBox(object);
      if (part) box.union(part.applyMatrix4(world.multiplyMatrices(frame, object.matrixWorld)));
    }
    for (const child of object.children) visit(child);
  };
  visit(owner);
  return centered(box);
}

function bodyNode(doc: SceneDocument, node: SceneNode): SceneNode | undefined {
  if (node.kind !== 'joint') return node;
  const joint = doc.joints.find((candidate) => candidate.id === node.id);
  if (!joint) return undefined;
  return doc.nodes.find((candidate) =>
    joint.bodyId
      ? candidate.id === joint.bodyId
      : candidate.kind === 'link' && candidate.name === joint.child,
  );
}

/** Body/link bounds contain their own geometry; model roots contain descendant geometry. */
export function sceneNodeBounds(
  doc: SceneDocument,
  id: string,
  space: 'world' | 'local' = 'world',
): FocusBounds | null {
  const selected = doc.nodes.find((candidate) => candidate.id === id);
  if (!selected || selected.kind === 'camera' || selected.kind === 'light') return null;
  const owner = bodyNode(doc, selected);
  if (!owner) return null;
  let frame: THREE.Object3D | undefined;
  if (space === 'local') {
    if (owner.kind === 'robot') return null;
    frame = owner.object;
    if (owner.kind !== 'link') {
      const byObject = new Map(doc.nodes.map((node) => [node.object, node]));
      while (frame.parent && byObject.get(frame)?.kind !== 'link') frame = frame.parent;
    }
  }
  // Electron's robot root is the imported model container, equivalent to native MODEL.
  return boundsFor(doc, owner.object, owner.kind === 'robot', frame);
}

/** Resolve tree focus, including native scene-relative placeholders for geometry-free entities. */
export function sceneFocusBounds(doc: SceneDocument, id?: string): FocusBounds {
  const scene = boundsFor(doc, doc.root, true) ?? {
    center: [0, 0, 0],
    half: [0.5, 0.5, 0.5],
  };
  if (id === undefined) return scene;
  const node = doc.nodes.find((candidate) => candidate.id === id);
  if (!node) throw Error(`Scene node is unavailable: ${id}`);
  if (node.kind === 'world' || node.kind === 'environment') return scene;
  const bounds = sceneNodeBounds(doc, id);
  if (bounds) return bounds;
  const radius = Math.hypot(...scene.half) * 2 * 0.025;
  return {
    center: node.object.getWorldPosition(new THREE.Vector3()).toArray(),
    half: Array(3).fill(radius / Math.sqrt(3)),
  };
}
