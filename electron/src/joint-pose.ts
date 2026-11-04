import * as THREE from 'three';
import type { Joint, SceneDocument } from './contracts';
import { setJointConfiguration } from './scene';

/** Native free coordinates describe the world pose; ball coordinates stay joint-local. */
export function jointDisplayPose(joint: Joint) {
  if (joint.type === 'free') {
    joint.object.updateWorldMatrix(true, false);
    return {
      position: joint.object.getWorldPosition(new THREE.Vector3()),
      rotation: joint.object.getWorldQuaternion(new THREE.Quaternion()),
    };
  }
  return {
    position: (joint.position ?? new THREE.Vector3()).clone(),
    rotation: (joint.rotation ?? new THREE.Quaternion()).clone(),
  };
}
export function setJointDisplayPose(
  doc: SceneDocument,
  joint: Joint,
  pose: { position?: number[]; rotation?: number[] },
) {
  if (joint.type !== 'free') {
    setJointConfiguration(doc, joint.id, pose);
    return;
  }
  joint.object.parent?.updateWorldMatrix(true, false);
  const frame = new THREE.Matrix4().multiplyMatrices(
    joint.object.parent?.matrixWorld ?? new THREE.Matrix4(),
    joint.origin,
  );
  const configuration: { position?: number[]; rotation?: number[] } = {};
  if (pose.position)
    configuration.position = new THREE.Vector3()
      .fromArray(pose.position)
      .applyMatrix4(frame.clone().invert())
      .toArray();
  if (pose.rotation) {
    const frameRotation = new THREE.Quaternion();
    frame.decompose(new THREE.Vector3(), frameRotation, new THREE.Vector3());
    configuration.rotation = frameRotation
      .invert()
      .multiply(new THREE.Quaternion().fromArray(pose.rotation))
      .normalize()
      .toArray();
  }
  setJointConfiguration(doc, joint.id, configuration);
}
