import * as THREE from 'three';
import type { SceneDocument, SceneNode } from './contracts';

/** Viewport selection is disabled by any ancestor; tree editing remains independent. */
export function isViewportPickable(object: THREE.Object3D): boolean {
  for (let current: THREE.Object3D | null = object; current; current = current.parent)
    if (current.userData.viewportPickable === false) return false;
  return true;
}

/** Stored on the scene object so save/load, duplication and undo preserve the preference. */
export function setViewportPickable(object: THREE.Object3D, enabled: boolean): void {
  object.userData.viewportPickable = enabled;
}

/** One render owner represents a logical selection; hierarchy containers own no mesh ID. */
export function selectionGeometryOwner(
  doc: SceneDocument,
  selected: string | null,
): SceneNode | null {
  if (selected === null) return null;
  const node = doc.nodes.find((node) => node.id === selected);
  if (!node || !isViewportPickable(node.object)) return null;
  if (['world', 'robot', 'camera', 'light'].includes(node.kind)) return null;
  if (node.kind !== 'joint') return node;
  const joint = doc.joints.find((joint) => joint.id === node.id);
  if (!joint) return null;
  return (
    doc.nodes.find((candidate) =>
      joint.bodyId
        ? candidate.id === joint.bodyId
        : candidate.kind === 'link' && candidate.name === joint.child,
    ) ?? null
  );
}

/** Match native render-ID ownership instead of highlighting a selected subtree. */
export function selectionGeometry(
  doc: SceneDocument,
  selected: string | null,
  result = new Set<THREE.Mesh>(),
  nodes = new Map<THREE.Object3D, SceneNode>(),
): Set<THREE.Mesh> {
  result.clear();
  nodes.clear();
  const owner = selectionGeometryOwner(doc, selected);
  if (!owner) return result;
  for (const node of doc.nodes) nodes.set(node.object, node);
  const body = owner.kind === 'link';
  const visit = (object: THREE.Object3D) => {
    const node = nodes.get(object);
    // A body's geometry children share its render owner. Child bodies, joints,
    // primitives and helpers retain their own identities. A geom is exclusive.
    if (object !== owner.object && node && !(body && node.kind === 'geom')) return;
    if (object instanceof THREE.Mesh && isViewportPickable(object)) result.add(object);
    for (const child of object.children) visit(child);
  };
  visit(owner.object);
  return result;
}
