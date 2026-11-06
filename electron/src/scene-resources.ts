import * as THREE from 'three';
import type { SceneDocument } from './contracts';
import { getSceneRendering } from './scene-rendering.ts';

export function collectResources(roots: THREE.Object3D[]) {
  const geometries = new Set<THREE.BufferGeometry>(),
    materials = new Set<THREE.Material>(),
    textures = new Set<THREE.Texture>(),
    shadows = new Set<THREE.LightShadow>();
  for (const root of roots) {
    const skybox = getSceneRendering(root)?.skybox;
    if (skybox) textures.add(skybox);
    root.traverse((object) => {
      if (
        object instanceof THREE.Mesh ||
        object instanceof THREE.Line ||
        object instanceof THREE.Points
      ) {
        geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          materials.add(material);
      } else if (object instanceof THREE.Sprite) materials.add(object.material);
      if (object instanceof THREE.Mesh) {
        if (object.customDepthMaterial) materials.add(object.customDepthMaterial);
        if (object.customDistanceMaterial) materials.add(object.customDistanceMaterial);
      }
      if (
        object instanceof THREE.DirectionalLight ||
        object instanceof THREE.SpotLight ||
        object instanceof THREE.PointLight
      )
        shadows.add(object.shadow);
    });
  }
  materials.forEach((mat) => {
    for (const value of Object.values(mat)) if (value instanceof THREE.Texture) textures.add(value);
  });
  return { geometries, materials, textures, shadows };
}

export function disposeResources(
  resources: ReturnType<typeof collectResources>,
  live?: ReturnType<typeof collectResources>,
) {
  resources.geometries.forEach((value) => {
    if (!live?.geometries.has(value)) value.dispose();
  });
  resources.materials.forEach((value) => {
    if (!live?.materials.has(value)) value.dispose();
  });
  resources.textures.forEach((value) => {
    if (!live?.textures.has(value)) value.dispose();
  });
  resources.shadows.forEach((value) => {
    if (!live?.shadows.has(value)) value.dispose();
  });
}

export function disposeDocument(doc: SceneDocument): void {
  // Shadow materials, shadow targets and scene backgrounds are document-owned too.
  disposeResources(collectResources([doc.root]));
}
