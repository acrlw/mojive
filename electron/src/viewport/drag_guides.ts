/** Native active-guide projection, from python/ui/gizmo/{guides,projection}.py. */
import * as THREE from 'three';
import {
  type P,
  SIZE,
  worldScale,
  ringAlpha,
  axisAlpha,
  basisForAxis,
  roundPolygon,
  arrow,
} from './geometry';
import { arcRibbonMesh } from './viewcube';
export type Segment = [P, P];
export interface GuideTick {
  start: P;
  end: P;
  alpha: number;
  active: boolean;
  width: number;
  roundStart: boolean;
  roundEnd: boolean;
}
const tau = Math.PI * 2,
  clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const add = (a: P, b: P, n = 1): P => [a[0] + b[0] * n, a[1] + b[1] * n];
const sub = (a: P, b: P): P => [a[0] - b[0], a[1] - b[1]];
const dot = (a: P, b: P) => a[0] * b[0] + a[1] * b[1];
function blend(camera: THREE.Camera) {
  return Number(
    camera.userData.projectionBlend ?? (camera instanceof THREE.OrthographicCamera ? 1 : 0),
  );
}
function view(camera: THREE.Camera, origin: THREE.Vector3) {
  const t = blend(camera);
  return origin
    .clone()
    .sub(camera.position)
    .normalize()
    .multiplyScalar(1 - t)
    .addScaledVector(camera.getWorldDirection(new THREE.Vector3()), t)
    .normalize();
}
function projected(
  camera: THREE.Camera,
  p: THREE.Vector3,
  w: number,
  h: number,
): [number, number, number] {
  const v = new THREE.Vector4(p.x, p.y, p.z, 1)
    .applyMatrix4(camera.matrixWorldInverse)
    .applyMatrix4(camera.projectionMatrix);
  return [(v.x / v.w + 1) * w * 0.5, (1 - v.y / v.w) * h * 0.5, v.w];
}
const xy = (p: number[]): P => [p[0], p[1]];
export function clipLine(
  origin: P,
  direction: P,
  rect: [number, number, number, number],
  finite = false,
): Segment | null {
  if (Math.hypot(...direction) < 1e-6) return null;
  let lo = finite ? 0 : -Infinity,
    hi = finite ? 1 : Infinity;
  for (let i = 0; i < 2; i++) {
    const lower = rect[i],
      upper = rect[i] + rect[i + 2];
    if (Math.abs(direction[i]) < 1e-9) {
      if (origin[i] < lower || origin[i] > upper) return null;
      continue;
    }
    const a = (lower - origin[i]) / direction[i],
      b = (upper - origin[i]) / direction[i];
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
  }
  return lo > hi ? null : [add(origin, direction, lo), add(origin, direction, hi)];
}
export function splitAroundInterval(
  start: P,
  end: P,
  origin: P,
  direction: P,
  lower: number,
  upper: number,
): Segment[] {
  const n = Math.hypot(...direction);
  if (n < 1e-9 || lower >= upper) return [[start, end]];
  direction = [direction[0] / n, direction[1] / n];
  let a = dot(sub(start, origin), direction),
    b = dot(sub(end, origin), direction);
  if (a > b) {
    [start, end] = [end, start];
    [a, b] = [b, a];
  }
  const result: Segment[] = [];
  if (a < lower) result.push([start, add(origin, direction, Math.min(lower, b))]);
  if (b > upper) result.push([add(origin, direction, Math.max(upper, a)), end]);
  return result;
}
export function splitAroundPoint(start: P, end: P, center: P, radius: number): Segment[] {
  const delta = sub(end, start),
    length = Math.hypot(...delta);
  if (length < 1e-9 || radius <= 0) return [[start, end]];
  const direction: P = [delta[0] / length, delta[1] / length],
    along = dot(sub(center, start), direction),
    perp = Math.hypot(...sub(center, add(start, direction, along)));
  if (perp >= radius) return [[start, end]];
  const half = Math.sqrt(radius * radius - perp * perp),
    lo = Math.max(0, along - half),
    hi = Math.min(length, along + half);
  if (lo >= hi) return [[start, end]];
  const result: Segment[] = [];
  if (lo > 1e-6) result.push([start, add(start, direction, lo)]);
  if (hi < length - 1e-6) result.push([add(start, direction, hi), end]);
  return result;
}
export function snapTickAlpha(offset: number) {
  const distance = Math.abs(offset);
  if (distance <= 5) return 1;
  if (distance >= 10) return 0;
  const t = (distance - 5) / 5;
  return 1 - t * t * (3 - 2 * t);
}
function lineParameters(
  camera: THREE.Camera,
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  segment: Segment,
  w: number,
  h: number,
): [number, number] | null {
  const matrix = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    ),
    a = new THREE.Vector4(origin.x, origin.y, origin.z, 1).applyMatrix4(matrix),
    b = new THREE.Vector4(axis.x, axis.y, axis.z, 0).applyMatrix4(matrix),
    values: number[] = [];
  for (const p of segment) {
    const ndc = [(p[0] / w) * 2 - 1, 1 - (p[1] / h) * 2],
      den = [b.x - ndc[0] * b.w, b.y - ndc[1] * b.w],
      i = Math.abs(den[0]) >= Math.abs(den[1]) ? 0 : 1;
    if (Math.abs(den[i]) < 1e-10) return null;
    values.push((ndc[i] * a.w - a.getComponent(i)) / den[i]);
  }
  return values as [number, number];
}
export function translationRuler(
  camera: THREE.Camera,
  current: THREE.Vector3,
  start: THREE.Vector3,
  axis: THREE.Vector3,
  w: number,
  h: number,
  options: { step?: number; slide?: boolean; styleScale?: number } = {},
) {
  const { step = 0.1, slide = false, styleScale = 1 } = options,
    scale = worldScale(current, camera, h) * styleScale,
    points = [
      projected(camera, current, w, h),
      projected(camera, current.clone().addScaledVector(axis, scale), w, h),
    ];
  if (points.some((p) => p[2] <= 0) || step <= 0) return null;
  const origin = xy(points[0]),
    delta = sub(xy(points[1]), origin),
    length = Math.hypot(...delta);
  if (length < 1e-6) return null;
  const direction: P = [delta[0] / length, delta[1] / length],
    normal: P = [-direction[1], direction[0]],
    segment = clipLine(origin, direction, [0, 0, w, h]);
  if (!segment) return null;
  const bounds = lineParameters(camera, current, axis, segment, w, h);
  if (!bounds) return null;
  const distance = current.clone().sub(start).dot(axis),
    currentStep = distance / step,
    lo = Math.max(Math.ceil((Math.min(...bounds) + distance) / step), Math.ceil(currentStep - 10)),
    hi = Math.min(
      Math.floor((Math.max(...bounds) + distance) / step),
      Math.floor(currentStep + 10),
    );
  if (lo > hi) return null;
  const mask = SIZE * 0.11 * styleScale,
    hidden: [number, number] | null = slide ? null : [-mask, length],
    segments = hidden ? splitAroundInterval(...segment, origin, direction, ...hidden) : [segment],
    raw: { a: P; b: P; alpha: number; active: boolean }[] = [];
  for (let index = lo; index <= hi; index++) {
    const value = index * step,
      p = projected(camera, current.clone().addScaledVector(axis, value - distance), w, h),
      alpha = length * step >= 2 * styleScale ? snapTickAlpha(index - currentStep) : 0;
    if (p[2] <= 0 || alpha <= 0.01) continue;
    const half = (Math.abs(value - Math.round(value)) < 1e-6 ? 7 : 3.5) * styleScale;
    raw.push({ a: add(xy(p), normal, -half), b: add(xy(p), normal, half), alpha, active: false });
  }
  if (!slide)
    raw.push({
      a: add(origin, normal, -14 * styleScale),
      b: add(origin, normal, 14 * styleScale),
      alpha: 1,
      active: true,
    });
  const ticks: GuideTick[] = [];
  for (const { a, b, alpha, active } of raw) {
    const along = dot(sub([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], origin), direction);
    if (!active && hidden && along >= hidden[0] && along <= hidden[1]) continue;
    for (const [start, end] of splitAroundPoint(a, b, origin, mask))
      ticks.push({
        start,
        end,
        alpha,
        active,
        width: (active ? 2.2 : 1.2) * styleScale,
        roundStart: true,
        roundEnd: true,
      });
  }
  return { segments, ticks };
}
export class RotationDial {
  private scale: number;
  private normal: THREE.Vector3;
  private radial: THREE.Vector3;
  private tangent: THREE.Vector3;
  private center: [number, number, number];
  private screenRadial: P = [1, 0];
  private screenTangent: P = [0, 1];
  constructor(
    private camera: THREE.Camera,
    private origin: THREE.Vector3,
    axis: THREE.Vector3,
    start: THREE.Vector3,
    private w: number,
    private h: number,
    private screen = false,
    private styleScale = 1,
  ) {
    this.normal = axis.clone().normalize();
    this.radial = start.clone().addScaledVector(this.normal, -start.dot(this.normal)).normalize();
    this.tangent = new THREE.Vector3().crossVectors(this.normal, this.radial);
    this.scale = worldScale(origin, camera, h) * styleScale;
    this.center = projected(camera, origin, w, h);
    if (screen) {
      const a = sub(xy(this.worldPoint(1, 0)), xy(this.center)),
        b = sub(xy(this.worldPoint(1, Math.PI * 0.5)), xy(this.center)),
        length = Math.hypot(...a);
      if (length > 1e-9) this.screenRadial = [a[0] / length, a[1] / length];
      const left: P = [-this.screenRadial[1], this.screenRadial[0]],
        sign = dot(left, b) >= 0 ? 1 : -1;
      this.screenTangent = [left[0] * sign, left[1] * sign];
    }
  }
  private worldPoint(radius: number, angle: number) {
    return projected(
      this.camera,
      this.origin
        .clone()
        .addScaledVector(this.radial, Math.cos(angle) * radius * this.scale)
        .addScaledVector(this.tangent, Math.sin(angle) * radius * this.scale),
      this.w,
      this.h,
    );
  }
  point(radius: number, angle: number): [number, number, number] {
    if (!this.screen) return this.worldPoint(radius, angle);
    const q = SIZE * this.styleScale * radius;
    return [
      this.center[0] +
        q * (Math.cos(angle) * this.screenRadial[0] + Math.sin(angle) * this.screenTangent[0]),
      this.center[1] +
        q * (Math.cos(angle) * this.screenRadial[1] + Math.sin(angle) * this.screenTangent[1]),
      this.center[2],
    ];
  }
  tick(radius: number, angle: number, length: number): Segment | null {
    const a = this.point(radius, angle),
      b = this.point(radius + 1 / SIZE, angle),
      direction = sub(xy(b), xy(a)),
      n = Math.hypot(...direction);
    if (a[2] <= 0 || b[2] <= 0 || n < 1e-6) return null;
    return [xy(a), add(xy(a), direction, length / n)];
  }
}
function roundEven(n: number) {
  const floor = Math.floor(n),
    fraction = n - floor;
  return fraction === 0.5 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(n);
}
export function rotationSweep(angle: number) {
  const shown = THREE.MathUtils.degToRad(roundEven(THREE.MathUtils.radToDeg(angle) * 10) / 10);
  return Math.sign(shown) * (Math.abs(shown) % tau);
}
export function shortestRotationSweep(angle: number) {
  const wrapped = ((((angle + Math.PI) % tau) + tau) % tau) - Math.PI;
  return Math.abs(wrapped + Math.PI) <= 1e-9 && angle > 0 ? Math.PI : wrapped;
}
export function rotationTickLength(degrees: number) {
  degrees = ((degrees % 360) + 360) % 360;
  const rounded = Math.round(degrees);
  return Math.abs(degrees - rounded) < 1e-6 && rounded % 90 === 0
    ? 8
    : Math.abs(degrees / 45 - Math.round(degrees / 45)) < 1e-6
      ? 7
      : Math.abs(degrees / 15 - Math.round(degrees / 15)) < 1e-6
        ? 5.5
        : 4;
}
export interface RotationGuideOptions {
  angle: number;
  screen?: boolean;
  joint?: { start: number; current: number; lower: number; upper: number };
  snapping?: boolean;
  projectionFade?: boolean;
  styleScale?: number;
  tickScale?: number;
  stepDegrees?: number;
}
export function rotationGuide(
  camera: THREE.Camera,
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  radial: THREE.Vector3,
  w: number,
  h: number,
  options: RotationGuideOptions,
) {
  const {
      screen = false,
      joint,
      snapping = false,
      projectionFade = false,
      styleScale = 1,
      tickScale = 1.25,
      stepDegrees = 5,
    } = options,
    facing = axis.dot(view(camera, origin)),
    alpha = projectionFade && !screen ? ringAlpha(facing) : 1;
  if (alpha <= 0) return null;
  const ambiguous = !!joint && joint.upper - joint.lower >= tau - 1e-9,
    start = joint?.start ?? 0,
    raw = rotationSweep(joint ? joint.current - start : options.angle),
    sweep = ambiguous ? shortestRotationSweep(raw) : raw,
    radius = screen ? 0.96 : 0.78,
    dial = new RotationDial(
      camera,
      origin,
      axis,
      joint ? basisForAxis(axis)[0] : radial,
      w,
      h,
      screen,
      styleScale,
    ),
    count = Math.ceil(64 * Math.min(4, 1 / Math.max(Math.sqrt(Math.abs(facing)), 0.25))),
    number = Math.max(2, Math.ceil((count * Math.abs(sweep)) / tau) + 1),
    angles = Array.from({ length: number }, (_, i) => start + (sweep * i) / (number - 1)),
    arc3 = angles.map((a) => dial.point(radius, a)),
    center = projected(camera, origin, w, h);
  if (center[2] <= 0 || arc3.some((p) => p[2] <= 0)) return null;
  const arc = arc3.map(xy),
    sector = [xy(center), ...arc],
    reference = joint
      ? []
      : Array.from({ length: count }, (_, i) => dial.point(radius, (tau * i) / count)),
    startTick = dial.tick(radius, start, 1),
    endTick = dial.tick(radius, start + sweep, 1),
    atLimit = (value: number) =>
      !!joint &&
      (value <= joint.lower + Math.max(1e-9, Math.abs(joint.upper - joint.lower) * 1e-6) ||
        value >= joint.upper - Math.max(1e-9, Math.abs(joint.upper - joint.lower) * 1e-6)),
    roundedStart = !!joint && !ambiguous && atLimit(start),
    roundedEnd = !!joint && !ambiguous && atLimit(joint.current),
    strokeMesh =
      Math.abs(sweep) > 1e-6
        ? arcRibbonMesh(
            arc,
            3.5 * styleScale,
            roundedStart,
            roundedEnd,
            startTick ? sub(startTick[1], startTick[0]) : null,
            endTick ? sub(endTick[1], endTick[0]) : null,
          )
        : { vertices: [], indices: [] };
  const ticks: GuideTick[] = [];
  const addTick = (angle: number, active: boolean) => {
    const a = dial.point(radius, angle),
      b = dial.point(radius, angle + THREE.MathUtils.degToRad(stepDegrees));
    if (a[2] <= 0 || b[2] <= 0 || Math.hypot(...sub(xy(b), xy(a))) < 2 * styleScale) return;
    const tick = dial.tick(
      radius,
      angle,
      rotationTickLength(THREE.MathUtils.radToDeg(angle)) * tickScale * styleScale,
    );
    if (tick)
      ticks.push({
        start: tick[0],
        end: tick[1],
        alpha,
        active,
        width: (active ? 2.2 : 1.1) * styleScale,
        roundStart: !joint || active,
        roundEnd: true,
      });
  };
  if (snapping && (screen || alpha >= 0.5) && stepDegrees > 0) {
    for (let degrees = 0; degrees < 360; degrees += stepDegrees) {
      const angle = THREE.MathUtils.degToRad(degrees);
      if (joint && !ambiguous) {
        const lower = joint.lower,
          span = joint.upper - lower,
          offset = (((angle - lower) % tau) + tau) % tau;
        if (offset > span + 1e-9) continue;
      }
      addTick(angle, false);
    }
    if (!joint) addTick(options.angle, true);
  }
  return {
    alpha,
    sweep,
    fillAlpha: Math.abs(sweep) > 1e-6 ? 0.24 * alpha : 0,
    sector,
    arc,
    stroke: strokeMesh.vertices,
    strokeIndices: strokeMesh.indices,
    reference: reference.every((p) => p[2] > 0) ? reference.map(xy) : [],
    ticks,
    anchor: xy(dial.point(radius, joint?.current ?? options.angle)),
  };
}
export function finiteAxisSegment(
  camera: THREE.Camera,
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  extent: number,
  w: number,
  h: number,
  inset = 0,
): Segment | null {
  if (axis.length() < 1e-9 || extent <= 0) return null;
  axis = axis.clone().normalize();
  let lo = -extent,
    hi = extent;
  if (blend(camera) < 1 - 1e-6) {
    const forward = camera.getWorldDirection(new THREE.Vector3()),
      depth = origin.clone().sub(camera.position).dot(forward),
      axisDepth = axis.dot(forward),
      near = Math.max(Number((camera as THREE.PerspectiveCamera).near ?? 0.01) * 1.5, 1e-4);
    if (depth <= near) return null;
    if (axisDepth > 1e-9) lo = Math.max(lo, (near - depth) / axisDepth);
    else if (axisDepth < -1e-9) hi = Math.min(hi, (near - depth) / axisDepth);
    if (lo >= hi) return null;
  }
  const a = projected(camera, origin.clone().addScaledVector(axis, lo), w, h),
    b = projected(camera, origin.clone().addScaledVector(axis, hi), w, h);
  if (a[2] <= 0 || b[2] <= 0) return null;
  inset = clamp(inset, 0, Math.min(w, h) * 0.5);
  return clipLine(xy(a), sub(xy(b), xy(a)), [inset, inset, w - 2 * inset, h - 2 * inset], true);
}
function circularCapsule(x: number, y: number, width: number, height: number): P[] {
  if (width <= 0 || height <= 0) return [];
  const long = Math.max(width, height),
    short = Math.min(width, height),
    radius = short * 0.5,
    step = Math.sqrt((8 * 0.025) / radius),
    half = Math.PI * 0.5,
    count = Math.ceil(half / step),
    angles = [
      ...Array.from({ length: count }, (_, i) => (half * i) / count),
      ...Array.from({ length: count + 1 }, (_, i) => half + (half * i) / count),
    ],
    cap = angles.map((a) => [Math.sin(a) * radius, -Math.cos(a) * radius] as P),
    right = cap.map((p) => [p[0] + long - radius, p[1] + radius] as P),
    left = cap.map((p) => [-p[0] + radius, -p[1] + radius] as P),
    all = [...right, ...left].map((p) =>
      height > width ? ([short - p[1] + x, p[0] + y] as P) : ([p[0] + x, p[1] + y] as P),
    );
  return all.filter(
    (p, i) =>
      Math.hypot(...sub(p, all[(i + all.length - 1) % all.length])) >
      Math.min(0.025 * 0.01, Math.max(1, short) * 1e-7),
  );
}
export function rotationAxisGuide(
  camera: THREE.Camera,
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  w: number,
  h: number,
  options: { hinge?: boolean; styleScale?: number } = {},
) {
  const { hinge = false, styleScale = 1 } = options,
    scale = worldScale(origin, camera, h) * styleScale;
  if (scale <= 0) return null;
  if (!hinge) {
    const depth = origin
        .clone()
        .sub(camera.position)
        .dot(camera.getWorldDirection(new THREE.Vector3())),
      ortho = 2 / camera.projectionMatrix.elements[5],
      extent = Math.max(
        scale * 8,
        blend(camera) >= 1 - 1e-6 ? ortho * 0.75 * Math.hypot(w / h, 1) : depth * 2.4,
      ),
      segment = finiteAxisSegment(
        camera,
        origin,
        axis,
        extent,
        w,
        h,
        Math.min(6 * styleScale, 0.1 * Math.min(w, h)),
      );
    return !segment || Math.hypot(...sub(segment[1], segment[0])) < 8 * styleScale
      ? null
      : { alpha: 0.82, segment, polygons: [] as P[][], hitPolygon: null as P[] | null };
  }
  const alpha = axisAlpha(axis.dot(view(camera, origin)));
  if (alpha <= 0) return null;
  const segment = finiteAxisSegment(camera, origin, axis, scale * 1.05, w, h),
    center = projected(camera, origin, w, h);
  if (!segment || center[2] <= 0) return null;
  const [start, tip] = segment,
    delta = sub(tip, start),
    length = Math.hypot(...delta);
  if (length < 8 * styleScale) return null;
  const direction: P = [delta[0] / length, delta[1] / length],
    side: P = [-direction[1], direction[0]],
    head = Math.min(12 * styleScale, length * 0.3),
    width = 3.5 * styleScale,
    shaft = width * 0.5,
    bodyEnd = length - head - width,
    join = 0.3 * styleScale,
    support = Math.min(length - head, join * 2.5),
    shape = roundPolygon(
      [
        [-support, -shaft],
        [0, -shaft],
        [0, -7 * styleScale],
        [head, 0],
        [0, 7 * styleScale],
        [0, shaft],
        [-support, shaft],
      ],
      0.6 * styleScale,
      [1, 2, 3, 4, 5],
      { 1: join, 5: join },
    ).map(([x, y]): P => [x + length - head, y]);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((center[0] / w) * 2 - 1, 1 - (center[1] / h) * 2), camera);
  const toward = ray.ray.direction.clone().negate(),
    facing = toward.dot(axis),
    radial = toward.addScaledVector(axis, -facing).normalize(),
    rim = [
      projected(camera, origin.clone().addScaledVector(radial, scale * 0.78), w, h),
      projected(camera, origin.clone().addScaledVector(radial, -scale * 0.78), w, h),
    ],
    centerT = dot(sub(xy(center), start), direction),
    rimT = rim.map((p) => dot(sub(xy(p), start), direction)),
    edgeT = facing > 0 ? Math.min(...rimT) : Math.max(...rimT),
    lo = clamp(Math.min(centerT, edgeT), 0, bodyEnd),
    hi = clamp(Math.max(centerT, edgeT), 0, bodyEnd),
    intervals: [number, number][] = [[0, lo]];
  for (let n = lo; n < hi; n += 9 * styleScale)
    intervals.push([n, Math.min(n + 5 * styleScale, hi)]);
  intervals.push([hi, length]);
  const merged: [number, number][] = [];
  for (const [a, b] of intervals) {
    if (b - a <= 1e-5) continue;
    const previous = merged.at(-1);
    if (previous && a - previous[1] <= 1e-5) previous[1] = b;
    else merged.push([a, b]);
  }
  const polygons = merged.map(([a, b]) => {
    let points: P[];
    if (b < length) points = circularCapsule(a, -shaft, b - a, width);
    else {
      const count = Math.ceil((Math.PI * 0.5) / Math.sqrt((8 * 0.025) / shaft)),
        angles = [
          ...Array.from({ length: count }, (_, i) => (Math.PI * 0.5 * i) / count),
          ...Array.from(
            { length: count + 1 },
            (_, i) => Math.PI * 0.5 + (Math.PI * 0.5 * i) / count,
          ),
        ],
        cap = angles
          .map((t) => [a + shaft - Math.sin(t) * shaft, -Math.cos(t) * shaft] as P)
          .reverse();
      points = [...shape.filter((p) => p[0] > bodyEnd), ...cap];
    }
    return points.map((p) => add(add(start, direction, p[0]), side, p[1]));
  });
  // Hit ownership covers the original solid arrow, including its occluded dash gaps.
  const hitPolygon = arrow(start, tip, true, {
    width,
    headLength: head,
    headWidth: 14 * styleScale,
    cornerRadius: 0.6 * styleScale,
  });
  return { alpha: 0.82 * alpha, segment: null, polygons, hitPolygon };
}
