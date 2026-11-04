// Direct port of python/geometry2d/curves.py and interaction/gizmo.py.
// Keep the G3 curvature profile and screen-space dimensions tied to Mojive.
import * as THREE from 'three';
export type P = [number, number];
const gx = [
  -0.9602898564975363, -0.7966664774136267, -0.525532409916329, -0.1834346424956498,
  0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363,
];
const gw = [
  0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.362683783378362, 0.362683783378362,
  0.3137066458778873, 0.2223810344533745, 0.1012285362903763,
];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const length = (p: P) => Math.hypot(...p);
const sub = (a: P, b: P): P => [a[0] - b[0], a[1] - b[1]];
const mul = (a: P, n: number): P => [a[0] * n, a[1] * n];
const dot = (a: P, b: P) => a[0] * b[0] + a[1] * b[1];
const cross = (a: P, b: P) => a[0] * b[1] - a[1] * b[0];
function theta(s: number, angle: number, smoothing = 0.618) {
  const phi = Math.abs(angle),
    r = smoothing * phi,
    total = phi + r,
    arc = phi - r;
  if (!r) return Math.sign(angle) * s;
  const down = s > r + arc,
    u = clamp((down ? total - s : s) / r, 0, 1),
    t = r * (u ** 3 - 0.5 * u ** 4);
  return Math.sign(angle) * (down ? phi - t : s > r ? s - r * 0.5 : t);
}
function integrate(lo: number, hi: number, angle: number, smoothing = 0.618): P {
  const h = (hi - lo) * 0.5,
    m = (hi + lo) * 0.5;
  let x = 0,
    y = 0;
  for (let i = 0; i < 8; i++) {
    const t = theta(m + h * gx[i], angle, smoothing);
    x += Math.cos(t) * gw[i];
    y += Math.sin(t) * gw[i];
  }
  return [x * h, y * h];
}
function turnEnd(angle: number, smoothing = 0.618): P {
  const r = smoothing * angle,
    up = integrate(0, r, angle, smoothing),
    ca = Math.cos(angle),
    sa = Math.sin(angle);
  return [
    up[0] + Math.sin(angle - r * 0.5) - Math.sin(r * 0.5) + ca * up[0] + sa * up[1],
    up[1] + Math.cos(r * 0.5) - Math.cos(angle - r * 0.5) + sa * up[0] - ca * up[1],
  ];
}
function smoothTurn(angle: number, scale: number, smoothing = 0.618): P[] {
  const phi = Math.abs(angle),
    r = smoothing * phi,
    total = phi + r,
    breaks = [...new Set([0, r, total * 0.5, phi, total])].sort((a, b) => a - b),
    step = Math.sqrt((8 * 0.025) / scale),
    out: P[] = [[0, 0]];
  let x = 0,
    y = 0;
  for (let k = 1; k < breaks.length; k++) {
    const lo = breaks[k - 1],
      hi = breaks[k],
      count = Math.max(1, Math.ceil((hi - lo) / step));
    for (let j = 0; j < count; j++) {
      const v = integrate(
        lo + ((hi - lo) * j) / count,
        lo + ((hi - lo) * (j + 1)) / count,
        angle,
        smoothing,
      );
      x += v[0];
      y += v[1];
      out.push([x * scale, y * scale]);
    }
  }
  const end = turnEnd(phi, smoothing);
  out[out.length - 1] = [end[0] * scale, Math.sign(angle) * end[1] * scale];
  return out;
}
export function roundPolygon(
  points: P[],
  radius: number,
  selected: number[],
  radii?: Record<number, number>,
): P[] {
  const n = points.length,
    edges = points.map((p, i) => sub(points[(i + 1) % n], p)),
    lengths = edges.map(length),
    dirs = edges.map((e, i) => mul(e, 1 / Math.max(lengths[i], 1e-30)));
  const turns = dirs.map((d, i) =>
    Math.atan2(cross(dirs[(i + n - 1) % n], d), dot(dirs[(i + n - 1) % n], d)),
  );
  const runs = turns.map((t, i) =>
    selected.includes(i) &&
    Math.abs(t) < Math.PI - 1e-4 &&
    Math.min(lengths[i], lengths[(i + n - 1) % n]) > 1e-9
      ? Math.max(0, radii?.[i] ?? radius) * Math.tan(Math.abs(t) * 0.5)
      : 0,
  );
  const factors = lengths.map((l, i) =>
    Math.min(1, (l * 0.99) / Math.max(runs[i] + runs[(i + 1) % n], 1e-30)),
  );
  const result: P[] = [];
  for (let i = 0; i < n; i++) {
    const incoming = dirs[(i + n - 1) % n],
      outgoing = dirs[i],
      angle = turns[i],
      rad = radii?.[i] ?? radius;
    if (
      !selected.includes(i) ||
      rad <= 0 ||
      Math.abs(angle) < 1e-6 ||
      Math.abs(angle) > Math.PI - 1e-4
    ) {
      result.push(points[i]);
      continue;
    }
    const run = radii
      ? runs[i] * Math.min(factors[i], factors[(i + n - 1) % n])
      : Math.min(
          rad * Math.tan(Math.abs(angle) * 0.5),
          0.45 * Math.min(lengths[i], lengths[(i + n - 1) % n]),
        );
    const end = turnEnd(Math.abs(angle)),
      bis: P = [1 + Math.cos(angle), Math.sin(Math.abs(angle))],
      unit = dot(end, bis) / dot(bis, bis);
    const curve = smoothTurn(angle, run / unit);
    for (const p of curve)
      result.push([
        points[i][0] - incoming[0] * run + incoming[0] * p[0] - incoming[1] * p[1],
        points[i][1] - incoming[1] * run + incoming[1] * p[0] + incoming[0] * p[1],
      ]);
    result[result.length - 1] = [
      points[i][0] + outgoing[0] * run,
      points[i][1] + outgoing[1] * run,
    ];
  }
  return result;
}
const arrowCache = new Map<string, P[]>();
export function arrow(
  start: P,
  end: P,
  roundTail = false,
  style: { width?: number; headLength?: number; headWidth?: number; cornerRadius?: number } = {},
): P[] {
  const delta = sub(end, start),
    len = length(delta);
  if (len < 1e-6) return [];
  const d = mul(delta, 1 / len),
    side: P = [-d[1], d[0]],
    width = style.width ?? 4.4,
    shaft = width * 0.5,
    wing = Math.max(width, style.headWidth ?? 14) * 0.5,
    radius = style.cornerRadius ?? 1.2,
    join = radius * 0.5,
    head = Math.min(style.headLength ?? 12, len * 0.42),
    support = Math.min(len - head, Math.max(join, 0.001) * 2.5),
    cacheKey = [head, shaft, wing, radius, support, join].join(':');
  let local = arrowCache.get(cacheKey);
  if (!local) {
    local = roundPolygon(
      [
        [-support, -shaft],
        [0, -shaft],
        [0, -wing],
        [head, 0],
        [0, wing],
        [0, shaft],
        [-support, shaft],
      ],
      radius,
      [1, 2, 3, 4, 5],
      { 1: join, 5: join },
    );
    if (arrowCache.size > 128) arrowCache.clear();
    arrowCache.set(cacheKey, local);
  }
  const neck: P = [end[0] - d[0] * head, end[1] - d[1] * head],
    path = local.map(
      (p): P => [neck[0] + p[0] * d[0] + p[1] * side[0], neck[1] + p[0] * d[1] + p[1] * side[1]],
    );
  if (!roundTail)
    return support === len - head
      ? path
      : [
          [start[0] - side[0] * shaft, start[1] - side[1] * shaft],
          ...path,
          [start[0] + side[0] * shaft, start[1] + side[1] * shaft],
        ];
  const maxInset = Math.max(0, len - head - support) * 0.45;
  let smoothing = 0.618;
  const ratio = (q: number) => {
    const ramp = q * Math.PI;
    let x = 0,
      y = 0;
    for (let i = 0; i < 8; i++) {
      const u = (1 + gx[i]) * 0.5,
        t = ramp * (u ** 3 - 0.5 * u ** 4);
      x += gw[i] * 0.5 * Math.cos(t);
      y += gw[i] * 0.5 * Math.sin(t);
    }
    return (ramp * x + 1 - Math.sin(ramp * 0.5)) / (2 * (ramp * y + Math.cos(ramp * 0.5)));
  };
  if (maxInset / width + 0.5 < ratio(smoothing)) {
    let lo = 0,
      hi = smoothing;
    for (let i = 0; i < 44; i++) {
      const q = (lo + hi) * 0.5;
      if (ratio(q) > maxInset / width + 0.5) hi = q;
      else lo = q;
    }
    smoothing = lo;
  }
  const shape = smoothTurn(Math.PI, width / turnEnd(Math.PI, smoothing)[1], smoothing),
    depth = Math.max(...shape.map((p) => p[0]));
  shape[0] = [0, 0];
  shape[shape.length - 1] = [0, width];
  const cap = shape
    .map(
      ([x, y]): P => [
        start[0] - (x + width * 0.5 - depth) * d[0] + (y - width * 0.5) * side[0],
        start[1] - (x + width * 0.5 - depth) * d[1] + (y - width * 0.5) * side[1],
      ],
    )
    .reverse();
  let result = [cap[cap.length - 1], ...path, ...cap.slice(0, -1)];
  if (support === len - head)
    result = result.filter(
      (p, i) =>
        Math.hypot(
          p[0] - result[(i + result.length - 1) % result.length][0],
          p[1] - result[(i + result.length - 1) % result.length][1],
        ) > 1e-8,
    );
  return result;
}
const cornerTemplate = (() => {
  const points = smoothTurn(Math.PI * 0.5, 4),
    end = points[points.length - 1];
  return points.map((p): P => [p[0] / end[0], p[1] / end[1]]);
})();
export function affineCorners(points: P[], radius = 2): P[] {
  return points.flatMap((p, i) => {
    const incoming = sub(p, points[(i + points.length - 1) % points.length]),
      outgoing = sub(points[(i + 1) % points.length], p),
      a = length(incoming),
      b = length(outgoing);
    if (Math.min(a, b) < 1e-9) return [p];
    const run = Math.min(radius, 0.4 * a, 0.4 * b),
      before = mul(incoming, run / a),
      after = mul(outgoing, run / b);
    return cornerTemplate.map(
      (t): P => [
        p[0] - before[0] + t[0] * before[0] + t[1] * after[0],
        p[1] - before[1] + t[0] * before[1] + t[1] * after[1],
      ],
    );
  });
}

/** Native dimensions handle: square is centered on the endpoint, including foreshortened shafts. */
export function dimensionAxis(start: P, end: P): P[] {
  const delta = sub(end, start),
    len = length(delta);
  if (len <= 1e-6) return [];
  const d = mul(delta, 1 / len),
    side: P = [-d[1], d[0]],
    half = 5.5,
    shaft = 1.25,
    radius = 1.1,
    join = 0.55;
  const transform = (p: P): P => [
    end[0] + p[0] * d[0] + p[1] * side[0],
    end[1] + p[0] * d[1] + p[1] * side[1],
  ];
  if (len <= half) {
    const r = radius,
      curve = smoothTurn(Math.PI * 0.5, r / turnEnd(Math.PI * 0.5)[0]);
    const local: P[] = [
      ...curve.map(([x, y]): P => [-half + y, -half + r - x]),
      ...curve.map(([x, y]): P => [half - r + x, -half + y]),
      ...curve.map(([x, y]): P => [half - y, half - r + x]),
      ...curve.map(([x, y]): P => [-half + r - x, half - y]),
    ];
    return local.map(transform);
  }
  const support = Math.min(len - half, join * 2.5),
    head = roundPolygon(
      [
        [-support, -shaft],
        [0, -shaft],
        [0, -half],
        [2 * half, -half],
        [2 * half, half],
        [0, half],
        [0, shaft],
        [-support, shaft],
      ],
      radius,
      [1, 2, 3, 4, 5, 6],
      { 1: join, 6: join },
    ).map(([x, y]): P => [x - half, y]);
  const points = support === len - half ? head : [[-len, -shaft] as P, ...head, [-len, shaft] as P];
  return points.map(transform);
}

export const AXIS_COLORS = ['#dc7773', '#52aa5c', '#6f94e5'];
export const JOINT_COLOR = '#9cbf8d';
export const HOVER_COLOR = '#b8d2ac';
export const SIZE = 88;
export function axisAlpha(facing: number) {
  return clamp((Math.sqrt(Math.max(0, 1 - facing * facing)) - 0.08) / 0.2, 0, 1);
}
export function ringAlpha(facing: number) {
  return clamp((Math.abs(facing) - 0.08) / 0.2, 0, 1);
}
export function project(point: THREE.Vector3, camera: THREE.Camera, w: number, h: number): P {
  const ndc = point.clone().project(camera);
  return [(ndc.x + 1) * w * 0.5, (1 - ndc.y) * h * 0.5];
}
export function worldScale(origin: THREE.Vector3, camera: THREE.Camera, height: number) {
  const v = new THREE.Vector4(origin.x, origin.y, origin.z, 1)
    .applyMatrix4(camera.matrixWorldInverse)
    .applyMatrix4(camera.projectionMatrix);
  return v.w <= 0
    ? 0
    : (2 * v.w * SIZE) / (camera.projectionMatrix.elements[5] * Math.max(1, height));
}
export function segmentDistance(point: P, a: P, b: P) {
  const d = sub(b, a),
    t = clamp(dot(sub(point, a), d) / Math.max(dot(d, d), 1e-12), 0, 1);
  return Math.hypot(point[0] - a[0] - t * d[0], point[1] - a[1] - t * d[1]);
}
export function pathDistance(p: P, path: P[], closed = false) {
  let d = Infinity;
  for (let i = 1; i < path.length; i++) d = Math.min(d, segmentDistance(p, path[i - 1], path[i]));
  if (closed && path.length) d = Math.min(d, segmentDistance(p, path[path.length - 1], path[0]));
  return d;
}
export function inside(p: P, path: P[]) {
  let yes = false;
  for (let i = 0, j = path.length - 1; i < path.length; j = i++) {
    const a = path[i],
      b = path[j];
    if (
      a[1] > p[1] !== b[1] > p[1] &&
      p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]
    )
      yes = !yes;
  }
  return yes;
}
export function basisForAxis(normal: THREE.Vector3): [THREE.Vector3, THREE.Vector3] {
  const z = normal.clone().normalize(),
    u = (
      Math.abs(z.z) > 0.9 ? new THREE.Vector3(z.z, 0, -z.x) : new THREE.Vector3(-z.y, z.x, 0)
    ).normalize();
  return [u, new THREE.Vector3().crossVectors(z, u).normalize()];
}
export function dial(
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  start: THREE.Vector3,
  radius: number,
  lo: number,
  hi: number,
  camera: THREE.Camera,
  w: number,
  h: number,
  count = 64,
): P[] {
  const tangent = new THREE.Vector3().crossVectors(axis, start).normalize();
  return Array.from({ length: count + 1 }, (_, i) => {
    const angle = lo + ((hi - lo) * i) / count;
    return project(
      origin
        .clone()
        .addScaledVector(start, Math.cos(angle) * radius)
        .addScaledVector(tangent, Math.sin(angle) * radius),
      camera,
      w,
      h,
    );
  });
}
export function frontArc(
  origin: THREE.Vector3,
  normal: THREE.Vector3,
  radius: number,
  camera: THREE.Camera,
  w: number,
  h: number,
): P[] {
  const toEye =
      camera instanceof THREE.OrthographicCamera
        ? camera.getWorldDirection(new THREE.Vector3()).negate()
        : camera.position.clone().sub(origin).normalize(),
    front = toEye.clone().addScaledVector(normal, -toEye.dot(normal));
  if (front.length() < 1e-9) front.copy(basisForAxis(normal)[0]);
  else front.normalize();
  const tangent = new THREE.Vector3().crossVectors(front, normal),
    matrix = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    ),
    center = new THREE.Vector4(origin.x, origin.y, origin.z, 1).applyMatrix4(matrix),
    cosine = new THREE.Vector4(
      tangent.x * radius,
      tangent.y * radius,
      tangent.z * radius,
      0,
    ).applyMatrix4(matrix),
    sine = new THREE.Vector4(front.x * radius, front.y * radius, front.z * radius, 0).applyMatrix4(
      matrix,
    ),
    e0 = center.clone().add(cosine),
    e1 = center.clone().sub(cosine),
    pixels: P = [w * 0.5, -h * 0.5],
    chord: P = [(e1.x / e1.w - e0.x / e0.w) * pixels[0], (e1.y / e1.w - e0.y / e0.w) * pixels[1]];
  const d = (v: THREE.Vector4) => chord[0] * v.x * pixels[0] + chord[1] * v.y * pixels[1],
    a = d(cosine),
    b = d(sine),
    c = d(center),
    aa = b * center.w - c * sine.w,
    bb = c * cosine.w - a * center.w,
    cc = b * cosine.w - a * sine.w,
    mag = Math.hypot(aa, bb);
  let lo = 0,
    hi = Math.PI;
  if (mag > 1e-12 && Math.abs(cc) < mag) {
    const phase = Math.atan2(bb, aa),
      delta = Math.acos(-cc / mag),
      tau = 2 * Math.PI,
      roots = [(((phase - delta) % tau) + tau) % tau, (((phase + delta) % tau) + tau) % tau]
        .filter((t) => t > 0 && t < Math.PI)
        .sort((x, y) => x - y);
    if (roots.length && aa + cc < -mag * 1e-12) lo = roots[0];
    if (roots.length && -aa + cc < -mag * 1e-12) hi = roots[roots.length - 1];
  }
  return Array.from({ length: 33 }, (_, i) => {
    const a = lo + ((hi - lo) * i) / 32,
      v = center.clone().addScaledVector(cosine, Math.cos(a)).addScaledVector(sine, Math.sin(a));
    return [(v.x / v.w) * pixels[0] + w * 0.5, (v.y / v.w) * pixels[1] + h * 0.5] as P;
  });
}
