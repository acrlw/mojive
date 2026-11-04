import * as THREE from 'three';
import type { CameraViewState, SceneCamera } from '../scene-camera';

export interface TransitionCameraView extends CameraViewState {
  projectionBlend?: number;
  viewportRelativeFocal?: boolean;
  normalizedFocal?: [number, number];
  principalOffset?: [number, number];
}

function blend(view: TransitionCameraView): number {
  return THREE.MathUtils.clamp(
    view.projectionBlend ?? (view.projection === 'orthographic' ? 1 : 0),
    0,
    1,
  );
}

function focal(view: TransitionCameraView, aspect = view.resolution[0] / view.resolution[1]) {
  const fy = 0.5 / Math.tan(THREE.MathUtils.degToRad(view.fov) / 2);
  return view.normalizedFocal ?? [fy / aspect, fy];
}

function distance(view: CameraViewState) {
  return Math.hypot(...view.position.map((value, index) => value - view.target[index]));
}

/** Native camera handoff; elapsed time is absolute so callers can sample moving targets. */
export class CameraViewTransition {
  readonly start: TransitionCameraView;
  readonly duration: number;
  private readonly rotation = new THREE.Quaternion();
  private readonly targetRotation = new THREE.Quaternion();
  private readonly interpolatedRotation = new THREE.Quaternion();
  private readonly basis = new THREE.Matrix4();
  private readonly eye = new THREE.Vector3();
  private readonly target = new THREE.Vector3();
  private readonly up = new THREE.Vector3();
  private readonly startFocal: number[];
  private readonly startPrincipal: number[];

  constructor(start: TransitionCameraView, duration = 0.28) {
    this.start = {
      ...start,
      position: [...start.position],
      target: [...start.target],
      up: [...start.up],
      resolution: [...start.resolution],
      normalizedFocal: start.normalizedFocal && [...start.normalizedFocal],
      principalOffset: start.principalOffset && [...start.principalOffset],
    };
    this.duration = duration;
    this.startFocal = [...focal(start)];
    this.startPrincipal = [...(start.principalOffset ?? [0, 0])];
    this.readRotation(start, this.rotation);
  }

  private readRotation(view: CameraViewState, rotation: THREE.Quaternion) {
    this.eye.fromArray(view.position);
    this.target.fromArray(view.target);
    this.up.fromArray(view.up);
    this.basis.lookAt(this.eye, this.target, this.up);
    // Match the native float32 view-matrix boundary, including the sign chosen
    // for exactly opposing rotations before the shortest quaternion path test.
    for (const index of [0, 1, 2, 4, 5, 6, 8, 9, 10])
      this.basis.elements[index] = Math.fround(this.basis.elements[index]);
    rotation.setFromRotationMatrix(this.basis);
    rotation.set(
      Math.fround(rotation.x),
      Math.fround(rotation.y),
      Math.fround(rotation.z),
      Math.fround(rotation.w),
    );
  }

  sample(target: TransitionCameraView, elapsedSeconds: number): TransitionCameraView {
    const elapsed = Math.max(0, elapsedSeconds);
    if (elapsed >= this.duration) return target;
    if (elapsed === 0) return { ...this.start, resolution: [...target.resolution] };
    const t = 1 - (1 - elapsed / this.duration) ** 3,
      start = this.start;
    this.readRotation(target, this.targetRotation);
    let dot = this.rotation.dot(this.targetRotation);
    if (dot < 0) {
      this.targetRotation.set(
        -this.targetRotation.x,
        -this.targetRotation.y,
        -this.targetRotation.z,
        -this.targetRotation.w,
      );
      dot = -dot;
    }
    let a = 1 - t,
      b = t;
    if (dot <= 0.9995) {
      const angle = Math.acos(THREE.MathUtils.clamp(dot, 0, 1)),
        sine = Math.sin(angle);
      a = Math.sin((1 - t) * angle) / sine;
      b = Math.sin(t * angle) / sine;
    }
    this.interpolatedRotation
      .set(
        this.rotation.x * a + this.targetRotation.x * b,
        this.rotation.y * a + this.targetRotation.y * b,
        this.rotation.z * a + this.targetRotation.z * b,
        this.rotation.w * a + this.targetRotation.w * b,
      )
      .normalize();
    this.basis.makeRotationFromQuaternion(this.interpolatedRotation);
    const matrix = this.basis.elements,
      eye = start.position.map((value, index) => value * (1 - t) + target.position[index] * t),
      span = Math.max(0.001, distance(start) * (1 - t) + distance(target) * t),
      targetFocal = focal(target),
      targetPrincipal = target.principalOffset ?? [0, 0];
    return {
      ...target,
      position: eye,
      target: eye.map((value, index) => value - Math.fround(matrix[8 + index]) * span),
      up: [Math.fround(matrix[4]), Math.fround(matrix[5]), Math.fround(matrix[6])],
      fov: start.fov * (1 - t) + target.fov * t,
      near: start.near * (1 - t) + target.near * t,
      far: start.far * (1 - t) + target.far * t,
      orthoHeight: start.orthoHeight * (1 - t) + target.orthoHeight * t,
      projectionBlend: blend(start) * (1 - t) + blend(target) * t,
      normalizedFocal: [
        this.startFocal[0] * (1 - t) + targetFocal[0] * t,
        this.startFocal[1] * (1 - t) + targetFocal[1] * t,
      ],
      principalOffset: [
        this.startPrincipal[0] * (1 - t) + targetPrincipal[0] * t,
        this.startPrincipal[1] * (1 - t) + targetPrincipal[1] * t,
      ],
    };
  }
}

const renderTarget = new THREE.Vector3(),
  perspectiveMatrix = new THREE.Matrix4(),
  orthographicMatrix = new THREE.Matrix4();

/** Render native normalized perspective/orthographic morphing without changing scene cameras. */
export function cameraForTransition(
  view: TransitionCameraView,
  aspect: number,
  reuse?: THREE.Camera,
): SceneCamera {
  const amount = blend(view),
    camera =
      amount >= 1
        ? reuse instanceof THREE.OrthographicCamera
          ? reuse
          : new THREE.OrthographicCamera()
        : reuse instanceof THREE.PerspectiveCamera
          ? reuse
          : new THREE.PerspectiveCamera(),
    near = view.near,
    far = view.far,
    [fx, fy] = focal(view, aspect),
    [px, py] = view.principalOffset ?? [0, 0],
    height = Math.max(view.orthoHeight, 1e-6),
    width = height * Math.max(aspect, 1e-6);
  camera.position.fromArray(view.position);
  camera.up.fromArray(view.up);
  camera.scale.set(1, 1, 1);
  camera.lookAt(renderTarget.fromArray(view.target));
  camera.near = near;
  camera.far = far;
  camera.zoom = 1;
  if (camera instanceof THREE.PerspectiveCamera) {
    camera.fov = view.fov;
    camera.aspect = aspect;
    camera.filmGauge = view.sensorWidth;
  } else {
    camera.left = -width / 2;
    camera.right = width / 2;
    camera.bottom = -height / 2;
    camera.top = height / 2;
  }
  perspectiveMatrix.set(
    2 * fx,
    0,
    2 * px,
    0,
    0,
    2 * fy,
    -2 * py,
    0,
    0,
    0,
    (far + near) / (near - far),
    (2 * far * near) / (near - far),
    0,
    0,
    -1,
    0,
  );
  orthographicMatrix.set(
    2 / width,
    0,
    0,
    0,
    0,
    2 / height,
    0,
    0,
    0,
    0,
    -2 / (far - near),
    -(far + near) / (far - near),
    0,
    0,
    0,
    1,
  );
  if (amount <= 0) camera.projectionMatrix.copy(perspectiveMatrix);
  else if (amount >= 1) camera.projectionMatrix.copy(orthographicMatrix);
  else {
    const perspective = perspectiveMatrix.elements,
      ortho = orthographicMatrix.elements,
      matrix = camera.projectionMatrix.elements,
      span = Math.max(distance(view), 1e-6);
    for (let index = 0; index < 16; index++)
      matrix[index] = (perspective[index] / span) * (1 - amount) + ortho[index] * amount;
  }
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.userData.projectionBlend = amount;
  camera.updateMatrixWorld(true);
  return camera;
}
