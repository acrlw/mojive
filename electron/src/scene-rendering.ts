import * as THREE from 'three';

/** Scene-owned lighting; editor, previews and captures consume the same description. */
export interface SceneRendering {
  ambient: [number, number, number]; headlight: [number, number, number];
  fog: [number, number, number]; fogStart: number; fogEnd: number;
  skybox?: THREE.CubeTexture;
}
const environments = new WeakMap<THREE.Object3D, SceneRendering>();
export const setSceneRendering = (root: THREE.Object3D, value: SceneRendering) => environments.set(root, value);
export const getSceneRendering = (root: THREE.Object3D) => environments.get(root);

/** Object-coordinate cubemaps are surface textures, not reflection environment maps. */
export function installSurfaceTexture(material: THREE.Material): void {
  if (!('map' in material) || !(material.map instanceof THREE.CubeTexture)) return;
  material.customProgramCacheKey = () => 'object-cube-texture-v1';
  material.onBeforeCompile = (shader) => {
    shader.uniforms.surfaceScale = { value: new THREE.Vector3().fromArray(material.userData.surfaceScale ?? [1, 1, 1]) };
    shader.vertexShader = 'uniform vec3 surfaceScale; varying vec3 surfaceDirection;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nsurfaceDirection = position * surfaceScale;');
    shader.fragmentShader = 'varying vec3 surfaceDirection;\n' + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_pars_fragment>', 'uniform samplerCube map;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>',
      'diffuseColor *= textureCube(map, surfaceDirection);');
  };
}
