import * as THREE from 'three';
import type { GeometryShape } from './contracts';

/** Engine-neutral geometry conversion; baked buffers survive workspace serialization. */
export function createGeometry(shape: GeometryShape): THREE.BufferGeometry {
  let geometry: THREE.BufferGeometry;
  switch (shape.type) {
    case 'box': geometry = new THREE.BoxGeometry(...shape.size); break;
    case 'plane': geometry = new THREE.PlaneGeometry((shape.size[0] || 10) * 2, (shape.size[1] || 10) * 2); break;
    case 'sphere': geometry = new THREE.SphereGeometry(shape.radius, 32, 24); break;
    case 'ellipsoid':
      geometry = new THREE.SphereGeometry(1, 32, 24);
      geometry.scale(...shape.size); break;
    case 'cylinder':
      geometry = new THREE.CylinderGeometry(shape.radius, shape.radius, shape.length, 32);
      geometry.rotateX(Math.PI / 2); break;
    case 'capsule':
      geometry = new THREE.CapsuleGeometry(shape.radius, shape.length, 12, 32);
      geometry.rotateX(Math.PI / 2); break;
    case 'mesh':
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(shape.positions.slice(), 3));
      if (shape.normals) geometry.setAttribute('normal', new THREE.BufferAttribute(shape.normals.slice(), 3));
      if (shape.uv) geometry.setAttribute('uv', new THREE.BufferAttribute(shape.uv.slice(), 2));
      if (shape.indices) geometry.setIndex(new THREE.BufferAttribute(shape.indices.slice(), 1));
      if (!shape.normals) geometry.computeVertexNormals();
      // Prepare raycast broad-phase bounds during asynchronous scene construction.
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();
      return geometry;
    case 'heightfield': {
      const { rows, columns, size, heights } = shape;
      geometry = new THREE.PlaneGeometry(size[0] * 2, size[1] * 2, columns - 1, rows - 1);
      const positions = geometry.getAttribute('position');
      for (let row = 0; row < rows; row++)
        for (let column = 0; column < columns; column++)
          positions.setZ(row * columns + column, heights[(rows - row - 1) * columns + column] * size[2]);
      geometry.computeVertexNormals(); break;
    }
  }
  const buffer = new THREE.BufferGeometry().copy(geometry);
  geometry.dispose();
  buffer.computeBoundingBox();
  buffer.computeBoundingSphere();
  return buffer;
}
