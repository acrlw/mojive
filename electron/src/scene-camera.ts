import * as THREE from 'three';
import type { SceneDocument, SceneNode } from './contracts';
export type SceneCamera = THREE.PerspectiveCamera | THREE.OrthographicCamera;
export interface CameraViewState {
  projection: 'perspective' | 'orthographic';
  position: number[];
  target: number[];
  up: number[];
  fov: number;
  near: number;
  far: number;
  orthoHeight: number;
  sensorWidth: number;
  focalLength: number;
  resolution: [number, number];
}
export type CameraPatch = Partial<CameraViewState>;
export interface SceneCameraState {
  id: string | null;
  pilot: boolean;
}
export function cameraNode(doc: SceneDocument, id: string): SceneNode {
  const node = doc.nodes.find((node) => node.id === id && node.kind === 'camera');
  if (
    !node ||
    (!(node.object instanceof THREE.PerspectiveCamera) &&
      !(node.object instanceof THREE.OrthographicCamera))
  )
    throw Error('Scene camera is unavailable.');
  return node;
}
export function readCameraView(node: SceneNode): CameraViewState {
  const camera = node.object as SceneCamera,
    meta = camera.userData.camera ?? {};
  camera.updateWorldMatrix(true, false);
  const position = camera.getWorldPosition(new THREE.Vector3()),
    rotation = camera.getWorldQuaternion(new THREE.Quaternion());
  const resolution: [number, number] = meta.resolution ?? [1920, 1080],
    aspect = resolution[0] / resolution[1];
  const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : (meta.fov ?? 45);
  const sensorWidth =
    camera instanceof THREE.PerspectiveCamera ? camera.filmGauge : (meta.sensorWidth ?? 35);
  const distance = meta.targetDistance ?? 5;
  return {
    projection: camera instanceof THREE.OrthographicCamera ? 'orthographic' : 'perspective',
    position: position.toArray(),
    target: position
      .clone()
      .add(new THREE.Vector3(0, 0, -distance).applyQuaternion(rotation))
      .toArray(),
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(rotation).toArray(),
    fov,
    near: camera.near,
    far: camera.far,
    orthoHeight:
      camera instanceof THREE.OrthographicCamera
        ? (camera.top - camera.bottom) / camera.zoom
        : (meta.orthoHeight ?? 4),
    sensorWidth,
    focalLength:
      sensorWidth / Math.max(aspect, 1) / (2 * Math.tan(THREE.MathUtils.degToRad(fov / 2))),
    resolution: [...resolution],
  };
}
export function validateCameraView(view: CameraViewState): void {
  if (!['perspective', 'orthographic'].includes(view.projection))
    throw Error('Invalid camera projection.');
  for (const vector of [view.position, view.target, view.up])
    if (!Array.isArray(vector) || vector.length !== 3 || !vector.every(Number.isFinite))
      throw Error('Camera position and direction must be finite XYZ vectors.');
  for (const value of [
    view.fov,
    view.near,
    view.far,
    view.orthoHeight,
    view.sensorWidth,
    view.focalLength,
  ])
    if (!Number.isFinite(value)) throw Error('Camera optics must be finite.');
  if (
    view.fov < 1 ||
    view.fov > 179 ||
    view.near < 1e-8 ||
    view.far <= view.near ||
    view.far > 1e9 ||
    view.orthoHeight < 1e-4 ||
    view.orthoHeight > 1e6 ||
    view.sensorWidth <= 0 ||
    view.focalLength <= 0
  )
    throw Error('Camera optics are outside their valid range.');
  if (
    !Array.isArray(view.resolution) ||
    view.resolution.length !== 2 ||
    view.resolution.some((value) => !Number.isInteger(value) || value < 1 || value > 8192) ||
    view.resolution[0] * view.resolution[1] > 16777216
  )
    throw Error('Camera resolution must fit within 8192 px per axis and 16 megapixels.');
  const direction = new THREE.Vector3()
      .fromArray(view.target)
      .sub(new THREE.Vector3().fromArray(view.position)),
    up = new THREE.Vector3().fromArray(view.up);
  if (
    direction.length() < 1e-6 ||
    up.length() < 1e-6 ||
    direction.normalize().cross(up.normalize()).length() < 1e-6
  )
    throw Error('Camera target and up direction must define a valid view.');
}
/** Change projection without changing selection identity, parentage, or dependent children. */
export function updateCameraView(doc: SceneDocument, id: string, patch: CameraPatch): SceneNode {
  const node = cameraNode(doc, id),
    old = node.object as SceneCamera,
    previous = readCameraView(node),
    next = { ...previous, ...patch };
  if (old.userData.physicsOwned && (patch.position || patch.target || patch.up ||
      (patch.projection !== undefined && patch.projection !== previous.projection)))
    throw new Error('The simulation backend owns this camera pose and projection type.');
  if (
    patch.focalLength !== undefined ||
    (patch.sensorWidth !== undefined && patch.fov === undefined) ||
    (patch.resolution !== undefined && patch.fov === undefined)
  )
    next.fov = THREE.MathUtils.radToDeg(
      2 *
        Math.atan(
          next.sensorWidth /
            Math.max(next.resolution[0] / next.resolution[1], 1) /
            (2 * next.focalLength),
        ),
    );
  next.focalLength =
    next.sensorWidth /
    Math.max(next.resolution[0] / next.resolution[1], 1) /
    (2 * Math.tan(THREE.MathUtils.degToRad(next.fov / 2)));
  validateCameraView(next);
  let camera: SceneCamera = old;
  if (old instanceof THREE.OrthographicCamera !== (next.projection === 'orthographic')) {
    camera =
      next.projection === 'orthographic'
        ? new THREE.OrthographicCamera()
        : new THREE.PerspectiveCamera();
    THREE.Object3D.prototype.copy.call(camera, old, false);
    camera.uuid = old.uuid;
    const parent = old.parent,
      index = parent?.children.indexOf(old) ?? -1;
    if (parent) {
      parent.remove(old);
      parent.add(camera);
      parent.children.splice(parent.children.indexOf(camera), 1);
      parent.children.splice(index, 0, camera);
    }
    for (const child of [...old.children]) camera.add(child);
    node.object = camera;
  }
  camera.near = next.near;
  camera.far = next.far;
  camera.zoom = 1;
  const aspect = next.resolution[0] / next.resolution[1];
  if (camera instanceof THREE.PerspectiveCamera) {
    camera.fov = next.fov;
    camera.aspect = aspect;
    camera.filmGauge = next.sensorWidth;
  } else {
    camera.left = (-next.orthoHeight * aspect) / 2;
    camera.right = -camera.left;
    camera.top = next.orthoHeight / 2;
    camera.bottom = -camera.top;
  }
  const eye = new THREE.Vector3().fromArray(next.position),
    target = new THREE.Vector3().fromArray(next.target),
    up = new THREE.Vector3().fromArray(next.up).normalize();
  const rotation = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().lookAt(eye, target, up),
  );
  camera.parent?.updateWorldMatrix(true, false);
  const worldScale = camera.getWorldScale(new THREE.Vector3()),
    world = new THREE.Matrix4().compose(eye, rotation, worldScale);
  if (camera.parent) world.premultiply(camera.parent.matrixWorld.clone().invert());
  world.decompose(camera.position, camera.quaternion, camera.scale);
  camera.up.copy(up);
  camera.userData.camera = {
    resolution: [...next.resolution],
    sensorWidth: next.sensorWidth,
    fov: next.fov,
    orthoHeight: next.orthoHeight,
    targetDistance: eye.distanceTo(target),
  };
  node.properties = {
    ...node.properties,
    Projection: next.projection === 'perspective' ? 'Perspective' : 'Orthographic',
  };
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return node;
}
export function cameraFromView(view: CameraViewState): SceneCamera {
  validateCameraView(view);
  const aspect = view.resolution[0] / view.resolution[1];
  const camera: SceneCamera =
    view.projection === 'orthographic'
      ? new THREE.OrthographicCamera(
          (-view.orthoHeight * aspect) / 2,
          (view.orthoHeight * aspect) / 2,
          view.orthoHeight / 2,
          -view.orthoHeight / 2,
          view.near,
          view.far,
        )
      : new THREE.PerspectiveCamera(view.fov, aspect, view.near, view.far);
  if (camera instanceof THREE.PerspectiveCamera) camera.filmGauge = view.sensorWidth;
  camera.position.fromArray(view.position);
  camera.up.fromArray(view.up);
  camera.lookAt(new THREE.Vector3().fromArray(view.target));
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return camera;
}
/** Reuse a detached render camera; scene transforms remain authoritative while viewing. */
export function renderSceneCamera(
  node: SceneNode,
  aspect?: number,
  reuse?: SceneCamera,
): SceneCamera {
  const source = node.object as SceneCamera,
    metadata = source.userData.camera ?? {},
    resolution = metadata.resolution ?? [1920, 1080];
  const camera =
    reuse && reuse.type === source.type
      ? reuse
      : source instanceof THREE.PerspectiveCamera
        ? new THREE.PerspectiveCamera()
        : new THREE.OrthographicCamera();
  source.updateWorldMatrix(true, false);
  source.matrixWorld.decompose(camera.position, camera.quaternion, camera.scale);
  camera.scale.set(1, 1, 1);
  camera.near = source.near;
  camera.far = source.far;
  camera.zoom = source.zoom;
  const outputAspect = resolution[0] / resolution[1];
  if (camera instanceof THREE.PerspectiveCamera && source instanceof THREE.PerspectiveCamera) {
    camera.fov = source.fov;
    camera.aspect = outputAspect;
    camera.filmGauge = source.filmGauge;
    camera.filmOffset = source.filmOffset;
  } else if (
    camera instanceof THREE.OrthographicCamera &&
    source instanceof THREE.OrthographicCamera
  ) {
    camera.top = (source.top - source.bottom) / 2;
    camera.bottom = -camera.top;
    camera.left = -camera.top * outputAspect;
    camera.right = -camera.left;
  }
  camera.updateProjectionMatrix();
  if (aspect !== undefined) {
    const x = Math.min(1, outputAspect / aspect),
      y = Math.min(1, aspect / outputAspect),
      m = camera.projectionMatrix.elements;
    for (const index of [0, 4, 8, 12]) m[index] *= x;
    for (const index of [1, 5, 9, 13]) m[index] *= y;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
  camera.userData.projectionBlend = camera instanceof THREE.OrthographicCamera ? 1 : 0;
  camera.updateMatrixWorld(true);
  return camera;
}
export function cameraViewFromRender(
  camera: THREE.Camera,
  target: THREE.Vector3,
  resolution: [number, number],
): CameraViewState {
  const eye = camera.getWorldPosition(new THREE.Vector3()),
    q = camera.getWorldQuaternion(new THREE.Quaternion()),
    perspective = camera as THREE.PerspectiveCamera,
    ortho = camera as THREE.OrthographicCamera;
  const fov = perspective.isPerspectiveCamera ? perspective.fov : 45,
    sensorWidth = perspective.isPerspectiveCamera ? perspective.filmGauge : 35;
  return {
    projection: ortho.isOrthographicCamera ? 'orthographic' : 'perspective',
    position: eye.toArray(),
    target: target.toArray(),
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray(),
    fov,
    near: perspective.near,
    far: perspective.far,
    orthoHeight: ortho.isOrthographicCamera
      ? (ortho.top - ortho.bottom) / ortho.zoom
      : 2 * eye.distanceTo(target) * Math.tan(THREE.MathUtils.degToRad(fov / 2)),
    sensorWidth,
    focalLength:
      sensorWidth /
      Math.max(resolution[0] / resolution[1], 1) /
      (2 * Math.tan(THREE.MathUtils.degToRad(fov / 2))),
    resolution,
  };
}
