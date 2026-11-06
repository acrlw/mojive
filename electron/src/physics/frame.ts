import type { PhysicsFrame, PhysicsSource } from './contracts';

const valid = (buffer: Float64Array, size: number) =>
  buffer instanceof Float64Array && buffer.length === size && buffer.every(Number.isFinite);

/** Validate the entire frame before any scene object is changed. */
export function validateFrame(source: PhysicsSource, frame: PhysicsFrame): void {
  const positions = source.joints.reduce((size, joint) => Math.max(size, joint.offset + joint.width), 0);
  if (frame.sourceId !== source.id || !Number.isFinite(frame.time) || frame.time < 0 ||
    !valid(frame.bodyTransforms, source.bodies.length * 7) ||
    !valid(frame.jointFrames, source.joints.length * 6) ||
    !valid(frame.cameraTransforms, source.cameras.length * 7) ||
    !valid(frame.lightFrames, source.lights.length * 6) ||
    !valid(frame.jointPositions, positions) || !valid(frame.controls, source.actuators.length))
    throw new Error('Simulation frame is incompatible with the scene source.');
  const meshes = source.geometries.filter((geometry) => geometry.deformable);
  const updates = frame.geometryUpdates ?? [];
  if (updates.length !== meshes.length || updates.some((update, i) => {
    const mesh = meshes[i], shape = mesh.shape;
    return update.id !== mesh.id || shape.type !== 'mesh' ||
      !(update.positions instanceof Float32Array) || !(update.normals instanceof Float32Array) ||
      update.positions.length !== shape.positions.length || update.normals.length !== shape.positions.length ||
      !update.positions.every(Number.isFinite) || !update.normals.every(Number.isFinite) ||
      !Number.isInteger(update.count) || update.count < 0 || update.count % 3 !== 0 ||
      update.count > (shape.indices?.length ?? shape.positions.length / 3);
  })) throw new Error('Simulation mesh updates are incompatible with the scene source.');
}
