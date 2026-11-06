import * as THREE from 'three';
import type { SceneDocument, SceneNode } from './contracts';
export type SceneLight =
  | THREE.PointLight
  | THREE.SpotLight
  | THREE.DirectionalLight
  | THREE.RectAreaLight
  | THREE.HemisphereLight;
export type LightKind = 'point' | 'spot' | 'directional' | 'area' | 'hemisphere';
export interface LightState {
  type: LightKind;
  color: string;
  groundColor: string;
  intensity: number;
  range: number;
  decay: number;
  angle: number;
  penumbra: number;
  width: number;
  height: number;
  direction: number[];
  enabled: boolean;
  castShadow: boolean;
}
const types: Record<string, LightKind> = {
  PointLight: 'point',
  SpotLight: 'spot',
  DirectionalLight: 'directional',
  RectAreaLight: 'area',
  HemisphereLight: 'hemisphere',
};
export function readLight(node: SceneNode): LightState {
  const light = node.object as SceneLight;
  if (node.kind !== 'light' || !types[light.type]) throw Error('Scene light is unavailable.');
  light.updateWorldMatrix(true, false);
  let direction = new THREE.Vector3(0, 0, -1).applyQuaternion(
    light.getWorldQuaternion(new THREE.Quaternion()),
  );
  if (light instanceof THREE.DirectionalLight || light instanceof THREE.SpotLight)
    direction = light.target
      .getWorldPosition(new THREE.Vector3())
      .sub(light.getWorldPosition(new THREE.Vector3()))
      .normalize();
  return {
    type: types[light.type],
    color: '#' + light.color.getHexString(),
    groundColor:
      light instanceof THREE.HemisphereLight ? '#' + light.groundColor.getHexString() : '#777777',
    intensity: light.intensity,
    range: 'distance' in light ? light.distance : 20,
    decay: 'decay' in light ? light.decay : 2,
    angle: light instanceof THREE.SpotLight ? THREE.MathUtils.radToDeg(light.angle) : 45,
    penumbra: light instanceof THREE.SpotLight ? light.penumbra : 0.2,
    width: light instanceof THREE.RectAreaLight ? light.width : 1,
    height: light instanceof THREE.RectAreaLight ? light.height : 1,
    direction: direction.toArray(),
    enabled: light.visible,
    castShadow: light.castShadow,
  };
}
export function updateLight(doc: SceneDocument, id: string, patch: Partial<LightState>): SceneNode {
  const node = doc.nodes.find((node) => node.id === id);
  if (!node) throw Error('Scene light is unavailable.');
  const old = node.object as SceneLight,
    previous = readLight(node),
    next = { ...previous, ...patch };
  if (old.userData.physicsOwned && (patch.direction ||
      (patch.type !== undefined && patch.type !== previous.type)))
    throw new Error('The simulation backend owns this light pose and type.');
  if (
    !['point', 'spot', 'directional', 'area', 'hemisphere'].includes(next.type) ||
    !/^#[0-9a-f]{6}$/i.test(next.color) ||
    !/^#[0-9a-f]{6}$/i.test(next.groundColor)
  )
    throw Error('Invalid light type or color.');
  for (const key of [
    'intensity',
    'range',
    'decay',
    'angle',
    'penumbra',
    'width',
    'height',
  ] as const)
    if (!Number.isFinite(next[key])) throw Error('Light properties must be finite.');
  if (
    next.intensity < 0 ||
    next.intensity > 1e6 ||
    next.range < 0 ||
    next.range > 1e6 ||
    next.decay < 0 ||
    next.decay > 10 ||
    next.angle < 0.1 ||
    next.angle > 89.9 ||
    next.penumbra < 0 ||
    next.penumbra > 1 ||
    next.width <= 0 ||
    next.width > 1e6 ||
    next.height <= 0 ||
    next.height > 1e6
  )
    throw Error('Light properties are outside their valid range.');
  if (
    next.direction.length !== 3 ||
    !next.direction.every(Number.isFinite) ||
    new THREE.Vector3().fromArray(next.direction).length() < 1e-8
  )
    throw Error('Light direction must be a nonzero XYZ vector.');
  let light = old;
  if (previous.type !== next.type) {
    light =
      next.type === 'point'
        ? new THREE.PointLight()
        : next.type === 'spot'
          ? new THREE.SpotLight()
          : next.type === 'directional'
            ? new THREE.DirectionalLight()
            : next.type === 'area'
              ? new THREE.RectAreaLight()
              : new THREE.HemisphereLight();
    THREE.Object3D.prototype.copy.call(light, old, false);
    light.uuid = old.uuid;
    const parent = old.parent,
      index = parent?.children.indexOf(old) ?? -1;
    if (parent) {
      parent.remove(old);
      parent.add(light);
      parent.children.splice(parent.children.indexOf(light), 1);
      parent.children.splice(index, 0, light);
    }
    for (const child of [...old.children])
      if (!child.userData.lightTarget && (!('target' in old) || child !== old.target))
        light.add(child);
    if ('shadow' in old) old.shadow?.dispose();
    node.object = light;
  }
  light.color.set(next.color);
  light.intensity = next.intensity;
  light.visible = next.enabled;
  node.visible = next.enabled;
  light.castShadow =
    next.castShadow &&
    !(light instanceof THREE.RectAreaLight || light instanceof THREE.HemisphereLight);
  if (light instanceof THREE.PointLight || light instanceof THREE.SpotLight) {
    light.distance = next.range;
    light.decay = next.decay;
    if (light.distance > 0) {
      light.shadow.camera.far = light.distance;
      light.shadow.camera.near = Math.min(0.5, light.distance * 0.01);
    } else {
      light.shadow.camera.near = 0.01;
      light.shadow.camera.far = 2000;
    }
    light.shadow.camera.updateProjectionMatrix();
  }
  if (light instanceof THREE.SpotLight) {
    light.angle = THREE.MathUtils.degToRad(next.angle);
    light.penumbra = next.penumbra;
  }
  if (light instanceof THREE.RectAreaLight) {
    light.width = next.width;
    light.height = next.height;
  }
  if (light instanceof THREE.HemisphereLight) light.groundColor.set(next.groundColor);
  const needsDirection =
    (light instanceof THREE.DirectionalLight || light instanceof THREE.SpotLight) &&
    !light.target.userData.lightTarget;
  if (!light.userData.physicsOwned && (patch.direction !== undefined || patch.type !== undefined || needsDirection)) {
    const direction = new THREE.Vector3().fromArray(next.direction).normalize(),
      eye = light.getWorldPosition(new THREE.Vector3()),
      up = Math.abs(direction.z) > 0.999 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
    const rotation = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(eye, eye.clone().add(direction), up),
    );
    if (light.parent)
      rotation.premultiply(light.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
    light.quaternion.copy(rotation);
  }
  if (!light.userData.physicsOwned && (light instanceof THREE.DirectionalLight || light instanceof THREE.SpotLight)) {
    let target =
      light.children.find((child) => child.userData.lightTarget) ??
      (light.target.parent === light ? light.target : undefined);
    if (!target) {
      target = new THREE.Object3D();
      target.name = 'Light direction';
      light.add(target);
    }
    target.userData.lightTarget = true;
    target.position.set(0, 0, -1);
    light.target = target;
  }
  node.properties = { ...node.properties, Type: next.type[0].toUpperCase() + next.type.slice(1) };
  light.updateMatrixWorld(true);
  return node;
}
