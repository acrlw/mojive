import * as THREE from 'three';
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js';
let areaInitialized = false;

/** Shared editor/export lighting and ground guide. */
export function createViewportScene() {
  if (!areaInitialized) {
    RectAreaLightUniformsLib.init();
    areaInitialized = true;
  }
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#22272d');
  const hemi = new THREE.HemisphereLight(0xcad7e4, 0x67645d, 2.6);
  hemi.position.set(0, 0, 5);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xfff4df, 3.1);
  key.position.set(4, -3, 7);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -8;
  key.shadow.camera.right = 8;
  key.shadow.camera.top = 8;
  key.shadow.camera.bottom = -8;
  key.shadow.bias = -0.0002;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9eb7dc, 1);
  fill.position.set(-4, 2, 4);
  scene.add(fill);
  const grid = new THREE.GridHelper(20, 40, 0x4d5862, 0x343c44);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.002;
  grid.userData.viewportHelper = true;
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.6;
  scene.add(grid);
  return {
    scene,
    grid,
    dispose() {
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      key.shadow.map?.dispose();
    },
  };
}
