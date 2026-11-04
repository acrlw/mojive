import type { Object3D } from 'three';
/** Supported content groups; unavailable physics/debug publishers are not represented as switches. */
export interface ViewportLayers {
  geometry: boolean;
  grid: boolean;
  helpers: boolean;
  influence: boolean;
  viewCube: boolean;
  gizmos: boolean;
  selection: boolean;
}
export const DEFAULT_VIEWPORT_LAYERS: Readonly<ViewportLayers> = Object.freeze({
  geometry: true,
  grid: true,
  helpers: true,
  influence: true,
  viewCube: true,
  gizmos: true,
  selection: true,
});
export function objectGizmosEnabled(object?: Object3D): boolean {
  for (let current = object; current; current = current.parent ?? undefined) {
    if (typeof current.userData.gizmoEnabled === 'boolean') return current.userData.gizmoEnabled;
  }
  return true;
}
