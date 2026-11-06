import * as THREE from 'three';
import type { PhysicsGeometry, PhysicsSource, PhysicsTexture } from './contracts';
import { setSceneRendering, installSurfaceTexture } from '../scene-rendering.ts';

export function createTexture(source: PhysicsTexture): THREE.DataTexture | THREE.CubeTexture {
  // Expand all compiled channel layouts to RGBA. Do not reload or reinterpret XML assets.
  const pixels = new Uint8Array(source.width * source.height * 4);
  for (let i = 0; i < source.width * source.height; i++) {
    for (let channel = 0; channel < 3; channel++)
      pixels[i * 4 + channel] = source.pixels[i * source.channels + (source.channels < 3 ? 0 : channel)];
    pixels[i * 4 + 3] = source.channels === 4 ? source.pixels[i * 4 + 3] : 255;
  }
  let texture: THREE.DataTexture | THREE.CubeTexture;
  if (source.type === '2d') {
    texture = new THREE.DataTexture(pixels, source.width, source.height);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  } else {
    const length = source.width * source.width * 4;
    const faces = Array.from({ length: 6 }, (_, face) => {
      const offset = source.height === source.width ? 0 : face * length;
      return new THREE.DataTexture(pixels.slice(offset, offset + length), source.width, source.width);
    });
    // WebGLTextures accepts DataTexture cube faces; the public typings only
    // describe the DOM-image form of CubeTexture.images.
    texture = new THREE.CubeTexture(faces as unknown as HTMLImageElement[]);
  }
  texture.name = source.name;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

export function applySurface(material: THREE.MeshStandardMaterial, geometry: THREE.BufferGeometry,
  source: PhysicsGeometry, textures: THREE.Texture[]): void {
  const description = source.material;
  if (!description) return;
  material.name = description.name;
  material.emissive.copy(material.color).multiplyScalar(description.emission);
  const texture = textures[description.texture];
  if (!texture) return;
  material.map = texture.clone();
  material.map.needsUpdate = true;
  if (texture instanceof THREE.CubeTexture) {
    geometry.computeBoundingBox();
    const size = geometry.boundingBox!.getSize(new THREE.Vector3()).multiplyScalar(.5);
    if (source.shape.type === 'capsule') size.setScalar(source.shape.radius);
    material.userData.surfaceScale = description.uniform ? [1, 1, 1] : size.toArray().map((v) => 1 / (v || 1));
    installSurfaceTexture(material);
  } else {
    material.map.repeat.set(...description.repeat);
    if (source.shape.type !== 'mesh' || !geometry.hasAttribute('uv')) {
      const positions = geometry.getAttribute('position');
      const uv = new Float32Array(positions.count * 2);
      geometry.computeBoundingBox();
      const size = geometry.boundingBox!.getSize(new THREE.Vector3());
      for (let i = 0; i < positions.count; i++) {
        uv[i * 2] = positions.getX(i) / (description.uniform ? 2 : size.x || 1) * description.repeat[0] + .5;
        uv[i * 2 + 1] = -positions.getY(i) / (description.uniform ? 2 : size.y || 1) * description.repeat[1] + .5;
      }
      material.map.repeat.set(1, 1);
      geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    }
  }
}

export function bindAppearance(root: THREE.Object3D, source: PhysicsSource): void {
  if (source.visual) {
    const sky = source.textures?.find((texture) => texture.type === 'skybox');
    setSceneRendering(root, { ...source.visual, skybox: sky ? createTexture(sky) as THREE.CubeTexture : undefined });
  }
  root.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) installSurfaceTexture(material);
      const material = Array.isArray(object.material) ? object.material[0] : object.material;
      if (material.map instanceof THREE.CubeTexture) {
        object.customDepthMaterial = new THREE.MeshDepthMaterial({ map: material.map, depthPacking: THREE.RGBADepthPacking });
        object.customDistanceMaterial = new THREE.MeshDistanceMaterial({ map: material.map });
        object.customDepthMaterial.userData.surfaceScale = material.userData.surfaceScale;
        object.customDistanceMaterial.userData.surfaceScale = material.userData.surfaceScale;
        installSurfaceTexture(object.customDepthMaterial);
        installSurfaceTexture(object.customDistanceMaterial);
      }
    }
  });
}
