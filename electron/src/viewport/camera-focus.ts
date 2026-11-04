import * as THREE from 'three';
import type { TransitionCameraView } from './camera-transition';
import type { FocusBounds } from './focus-bounds';

export interface CameraFocusOptions {
  margin?: number;
  duration?: number;
  easing?:
    | 'linear'
    | 'smoothstep'
    | 'smootherstep'
    | 'ease_out_quad'
    | 'ease_out_cubic'
    | 'ease_out_quart';
  eyeDirection?: number[] | null;
  minDistance?: number;
  maxDistance?: number;
  sphereRadius?: number;
  sphereMargin?: number;
  frameScene?: boolean;
}

const radians = THREE.MathUtils.degToRad,
  degrees = THREE.MathUtils.radToDeg,
  clamp = THREE.MathUtils.clamp,
  vector = (value: number[]) => new THREE.Vector3().fromArray(value),
  wrap = (value: number) => ((((value + 180) % 360) + 360) % 360) - 180,
  rounded = (value: THREE.Vector3) =>
    value.set(Math.fround(value.x), Math.fround(value.y), Math.fround(value.z));

function normalized(value: THREE.Vector3, fallback: THREE.Vector3) {
  if (!Number.isFinite(value.length()) || value.length() <= 1e-9) value.copy(fallback);
  return value.divideScalar(Math.max(value.length(), 1e-9));
}

/** Native focus keeps azimuth from eye-to-new-center and raises views below 30 degrees. */
export function elevatedFocusDirection(
  eyeOffset: number[],
  fallback: number[],
  minimum = 30,
): number[] {
  const direction = normalized(vector(eyeOffset), vector(fallback)),
    elevation = radians(clamp(minimum, 0, 89.9)),
    minZ = Math.sin(elevation);
  if (direction.z >= minZ) return direction.toArray();
  const horizontal = direction.clone().setZ(0);
  if (horizontal.length() <= 1e-9) horizontal.copy(vector(fallback)).setZ(0);
  if (horizontal.length() <= 1e-9) horizontal.set(1, 0, 0);
  return horizontal.normalize().multiplyScalar(Math.cos(elevation)).setZ(minZ).toArray();
}

function lookRotation(backward: THREE.Vector3, suppliedUp: THREE.Vector3): THREE.Matrix4 {
  const back = normalized(backward.clone(), new THREE.Vector3(0, 0, 1)),
    forward = back.clone().negate(),
    right = new THREE.Vector3().crossVectors(forward, suppliedUp);
  if (right.length() <= 1e-9)
    right.crossVectors(
      forward,
      Math.abs(forward.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0),
    );
  right.normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  return new THREE.Matrix4().makeBasis(rounded(right), rounded(up), rounded(back));
}

function cameraBasis(view: TransitionCameraView) {
  const backward = rounded(vector(view.position).sub(vector(view.target)).normalize()),
    right = new THREE.Vector3().crossVectors(backward.clone().negate(), vector(view.up));
  if (right.length() <= 1e-9)
    right.crossVectors(
      backward.clone().negate(),
      Math.abs(backward.z) < 0.95 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0),
    );
  right.normalize();
  return { right, up: new THREE.Vector3().crossVectors(backward, right), backward };
}

function orbitBasis(yaw: number, pitch: number) {
  const y = radians(yaw),
    p = radians(pitch),
    cy = Math.cos(y),
    sy = Math.sin(y),
    cp = Math.cos(p),
    sp = Math.sin(p);
  return {
    right: new THREE.Vector3(-sy, cy, 0),
    up: new THREE.Vector3(-sp * cy, -sp * sy, cp),
    backward: new THREE.Vector3(cp * cy, cp * sy, sp),
  };
}

function adoptedYaw(backward: THREE.Vector3, up: number[]) {
  if (Math.hypot(backward.x, backward.y) > 1e-9) return degrees(Math.atan2(backward.y, backward.x));
  const screenUp = vector(up).addScaledVector(backward, -vector(up).dot(backward));
  if (Math.hypot(screenUp.x, screenUp.y) <= 1e-9) return -90;
  const sign = backward.z >= 0 ? -1 : 1;
  return degrees(Math.atan2(sign * screenUp.y, sign * screenUp.x));
}

function roll(view: TransitionCameraView, yaw: number, pitch: number) {
  const right = new THREE.Vector3().setFromMatrixColumn(
      lookRotation(vector(view.position).sub(vector(view.target)), vector(view.up)),
      0,
    ),
    level = orbitBasis(yaw, pitch);
  return degrees(Math.atan2(right.dot(level.up), right.dot(level.right)));
}

function easing(value: number, kind: CameraFocusOptions['easing']) {
  const t = clamp(value, 0, 1);
  if (kind === 'linear') return t;
  if (kind === 'smootherstep') return t * t * t * (t * (t * 6 - 15) + 10);
  if (kind === 'ease_out_cubic') return 1 - (1 - t) ** 3;
  if (kind === 'ease_out_quart') return 1 - (1 - t) ** 4;
  if (kind === 'ease_out_quad') return t * (2 - t);
  return t * t * (3 - 2 * t);
}

/** Native focus_bounds / focus_target animation, separate from scene-camera handoffs. */
export class CameraFocusTransition {
  readonly start: TransitionCameraView;
  readonly end: TransitionCameraView;
  readonly duration: number;
  private readonly options: CameraFocusOptions;
  private readonly startDistance: number;
  private readonly endDistance: number;
  private readonly startYaw: number;
  private readonly endYaw: number;
  private readonly startPitch: number;
  private readonly endPitch: number;
  private readonly startRoll: number;
  private readonly endRoll: number;
  private readonly startSpan: number;
  private readonly endSpan: number;
  private readonly aimed: boolean;
  private readonly spherical: boolean;
  private readonly frameNear?: number;

  constructor(start: TransitionCameraView, bounds: FocusBounds, options: CameraFocusOptions = {}) {
    this.start = structuredClone(start);
    start = this.start;
    this.options = { easing: options.frameScene ? 'ease_out_quad' : 'smoothstep', ...options };
    this.duration = options.duration ?? (options.frameScene ? 0.35 : 0.36);
    this.spherical = options.sphereRadius !== undefined || options.frameScene === true;
    this.aimed = options.eyeDirection !== null;
    const center = vector(bounds.center),
      half = vector(bounds.half).set(
        Math.abs(bounds.half[0]),
        Math.abs(bounds.half[1]),
        Math.abs(bounds.half[2]),
      ),
      offset = vector(start.position).sub(vector(start.target)),
      initialDistance = Math.max(offset.length(), 0.001),
      initialDirection = offset.clone().divideScalar(initialDistance),
      direction =
        options.eyeDirection ??
        elevatedFocusDirection(
          vector(start.position).sub(center).toArray(),
          initialDirection.toArray(),
        ),
      minimum = options.minDistance ?? 0.001,
      maximum = options.maxDistance ?? 1e6,
      bounded = (value: number) => clamp(value, minimum, maximum),
      margin = Math.max(1, options.margin ?? 1.15),
      aspect = start.resolution[0] / start.resolution[1],
      heightPerDistance = 2 * Math.tan(radians(start.fov) / 2);
    this.startDistance = initialDistance;
    this.startYaw = adoptedYaw(initialDirection, start.up);
    this.startPitch = clamp(degrees(Math.asin(clamp(initialDirection.z, -1, 1))), -89.9, 89.9);
    this.startRoll = this.endRoll = 0;
    this.startSpan = this.endSpan = 0;

    if (this.spherical) {
      const aim = options.frameScene
          ? orbitBasis(this.startYaw, Math.max(this.startPitch, 30)).backward
          : normalized(vector(direction), initialDirection),
        rawRadius = options.frameScene ? half.length() : options.sphereRadius!,
        radius = options.frameScene && rawRadius < 1e-6 ? 0.5 : Math.max(rawRadius, 1e-6),
        halfY = radians(start.fov) / 2,
        halfX = Math.atan(Math.tan(halfY) * Math.max(aspect, 0.001)),
        padding = options.frameScene
          ? 1.15
          : Math.max(1, ((options.sphereMargin ?? 1.5) * margin) / 1.15);
      this.endDistance = bounded(
        Math.max(0.001, (radius / Math.max(Math.sin(Math.min(halfX, halfY)), 0.001)) * padding),
      );
      this.endYaw = this.startYaw + wrap(degrees(Math.atan2(aim.y, aim.x)) - this.startYaw);
      this.endPitch = clamp(degrees(Math.asin(clamp(aim.z, -1, 1))), -89.9, 89.9);
      const backward = orbitBasis(this.endYaw, this.endPitch).backward;
      if (options.frameScene) this.frameNear = Math.max(0.001, (this.endDistance - radius) * 0.25);
      this.end = {
        ...start,
        position: rounded(center.clone().addScaledVector(backward, this.endDistance)).toArray(),
        target: rounded(center.clone()).toArray(),
        up: [0, 0, 1],
        near:
          this.frameNear === undefined
            ? Math.min(start.near, Math.max(0.0001, (this.endDistance - radius) * 0.25))
            : Math.min(this.frameNear, this.endDistance * 0.0025),
        far: options.frameScene
          ? (this.endDistance + radius) * 32
          : Math.max(start.far, (this.endDistance + radius) * 2),
        orthoHeight: this.endDistance * heightPerDistance,
        normalizedFocal: undefined,
        principalOffset: undefined,
        projectionBlend: undefined,
      };
      return;
    }

    const focal = start.normalizedFocal ?? [1 / heightPerDistance / aspect, 1 / heightPerDistance],
      p00 = Math.fround(2 * focal[0]),
      p11 = Math.fround(2 * focal[1]),
      p02 = Math.fround(2 * (start.principalOffset?.[0] ?? 0)),
      p12 = Math.fround(-2 * (start.principalOffset?.[1] ?? 0));
    let aimed = start;
    if (this.aimed) {
      const rotation = lookRotation(
        normalized(vector(direction), initialDirection),
        new THREE.Vector3(0, 0, 1),
      );
      if (start.projection !== 'orthographic')
        rotation.multiply(
          lookRotation(
            new THREE.Vector3(-p02 / p00, -p12 / p11, 1),
            new THREE.Vector3(0, 1, 0),
          ).transpose(),
        );
      aimed = {
        ...start,
        position: center
          .clone()
          .add(new THREE.Vector3().setFromMatrixColumn(rotation, 2))
          .toArray(),
        target: center.toArray(),
        up: new THREE.Vector3().setFromMatrixColumn(rotation, 1).toArray(),
      };
    }
    const { right, up, backward } = cameraBasis(aimed),
      support = (value: THREE.Vector3) =>
        Math.abs(value.x) * half.x + Math.abs(value.y) * half.y + Math.abs(value.z) * half.z,
      horizontal = right.clone().multiplyScalar(p00).addScaledVector(backward, p02),
      vertical = up.clone().multiplyScalar(p11).addScaledVector(backward, p12);
    let distance = Math.max(0.001, support(backward) + 0.001);
    for (const axis of [horizontal, vertical])
      for (const sign of [-1, 1])
        distance = Math.max(
          distance,
          support(backward.clone().addScaledVector(axis, sign * margin)),
        );
    distance = bounded(distance);
    let height = Math.max(
      0.001,
      2 * margin * support(up),
      (2 * margin * support(right)) / Math.max(aspect, 0.001),
    );
    height = bounded(height / heightPerDistance) * heightPerDistance;
    const target = center.clone();
    if (start.projection !== 'orthographic') {
      target
        .addScaledVector(right, (-distance * p02) / p00)
        .addScaledVector(up, (-distance * p12) / p11);
      height = distance * heightPerDistance;
    }
    const depth = support(backward);
    this.end = {
      ...aimed,
      position: target.clone().addScaledVector(backward, distance).toArray(),
      target: target.toArray(),
      near: Math.min(start.near, Math.max(0.0001, (distance - depth) * 0.25)),
      far: Math.max(start.far, (distance + depth) * 2),
      orthoHeight: height,
      projectionBlend: undefined,
    };
    this.endDistance = distance;
    this.endYaw = this.startYaw + wrap(adoptedYaw(backward, up.toArray()) - this.startYaw);
    this.endPitch = degrees(Math.asin(clamp(backward.z, -1, 1)));
    if (this.aimed) {
      this.startPitch = degrees(Math.asin(clamp(cameraBasis(start).backward.z, -1, 1)));
      this.startRoll = roll(start, this.startYaw, this.startPitch);
      this.endRoll =
        this.startRoll + wrap(roll(this.end, this.endYaw, this.endPitch) - this.startRoll);
    }
    if (start.projection === 'orthographic') {
      this.startSpan = start.orthoHeight;
      this.endSpan = height;
    } else if (this.aimed) {
      this.startSpan = center.distanceTo(vector(start.position));
      this.endSpan = center.distanceTo(vector(this.end.position));
    } else {
      this.startSpan = vector(start.position).sub(center).dot(cameraBasis(start).backward);
      this.endSpan = distance;
    }
  }

  sample(elapsedSeconds: number): TransitionCameraView {
    if (elapsedSeconds >= this.duration || this.duration <= 0) return this.end;
    const start = this.start,
      end = this.end,
      progress = easing(Math.max(0, elapsedSeconds) / this.duration, this.options.easing),
      t =
        !this.spherical && this.startSpan > 0.001
          ? (progress * this.startSpan) /
            (this.endSpan * (1 - progress) + this.startSpan * progress)
          : progress,
      distance = this.spherical
        ? Math.exp(
            Math.log(Math.max(this.startDistance, 0.001)) * (1 - t) +
              Math.log(Math.max(this.endDistance, 0.001)) * t,
          )
        : this.startDistance * (1 - t) + this.endDistance * t,
      yaw = this.startYaw + (this.endYaw - this.startYaw) * progress,
      pitch = this.startPitch + (this.endPitch - this.startPitch) * progress,
      pivot = vector(start.target).lerp(vector(end.target), t),
      basis = orbitBasis(yaw, pitch),
      height = start.orthoHeight + (end.orthoHeight - start.orthoHeight) * t;
    if (this.spherical)
      return {
        ...end,
        position: rounded(pivot.clone().addScaledVector(basis.backward, distance)).toArray(),
        target: rounded(pivot).toArray(),
        orthoHeight: height,
        near: this.frameNear === undefined ? end.near : Math.min(this.frameNear, distance * 0.0025),
      };
    const eye = vector(start.position).lerp(vector(end.position), t);
    let up = end.up;
    if (this.aimed) {
      const angle = radians(this.startRoll + (this.endRoll - this.startRoll) * progress);
      pivot.copy(eye).addScaledVector(basis.backward, -distance);
      up = basis.right
        .multiplyScalar(-Math.sin(angle))
        .addScaledVector(basis.up, Math.cos(angle))
        .toArray();
    }
    return {
      ...end,
      position: eye.toArray(),
      target: pivot.toArray(),
      up,
      orthoHeight: height,
      projectionBlend:
        (start.projectionBlend ?? (start.projection === 'orthographic' ? 1 : 0)) * (1 - t) +
        (end.projection === 'orthographic' ? 1 : 0) * t,
    };
  }
}
