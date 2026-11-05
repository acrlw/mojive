export type Point = { x: number; y: number };
export type Cubic = [Point, Point, Point, Point];
export type Corner = {
  key: string;
  radius: number;
  footprint: number;
  leading: Cubic[];
  trailing: Cubic[];
  arc: { start: Point; end: Point; radius: number } | null;
  reference?: Corner;
  offset: number;
  sampledError: number;
};
const point = (x: number, y: number): Point => ({ x, y });
const add = (a: Point, b: Point) => point(a.x + b.x, a.y + b.y);
const sub = (a: Point, b: Point) => point(a.x - b.x, a.y - b.y);
const mul = (p: Point, n: number) => point(p.x * n, p.y * n);
const cross = (a: Point, b: Point) => a.x * b.y - a.y * b.x;
const length = (p: Point) => Math.hypot(p.x, p.y);

// Smoothing is the fraction of the quarter-circle replaced by two cubic transitions.
// The design radius stays fixed; p = (1 + smoothing) * radius sets the edge footprint.
export function smoothCorner(radius: number, smoothing: number, budget: number): Corner {
  const amount = radius > 0 ? Math.max(0, Math.min(1, smoothing, budget / radius - 1)) : 0;
  const key = `${radius}:${amount}:${budget}`;
  if (radius <= 0)
    return {
      key,
      radius: 0,
      footprint: 0,
      leading: [],
      trailing: [],
      arc: null,
      offset: 0,
      sampledError: 0,
    };
  if (amount === 0)
    return {
      key,
      radius,
      footprint: radius,
      leading: [],
      trailing: [],
      arc: { start: point(-radius, 0), end: point(0, radius), radius },
      offset: 0,
      sampledError: 0,
    };
  const phi = (Math.PI * amount) / 4;
  const tangent = Math.tan(phi / 2);
  const cosine = Math.cos(phi);
  const sine = Math.sin(phi);
  const p = (1 + amount) * radius;
  const b = (1.5 * radius * tangent) / (1 + cosine);
  const a = radius * (amount + tangent) - b;
  const d = 2 * radius * Math.sin(phi / 2) ** 2;
  const end = point(-radius + radius * sine, d);
  // P0/P1/P2 are collinear (zero starting curvature). P2 lies on the arc's
  // tangent and b solves kappa(1) = 1/radius, giving G2 at both original joins.
  const curve: Cubic = [point(-p, 0), point(-p + a, 0), point(-radius + radius * tangent, 0), end];
  const leading = [curve];
  const trailing = [[...curve].reverse().map((p) => point(-p.y, -p.x)) as Cubic];
  return {
    key,
    radius,
    footprint: p,
    leading,
    trailing,
    arc: amount < 1 ? { start: end, end: point(-d, radius - radius * sine), radius } : null,
    offset: 0,
    sampledError: 0,
  };
}
export function evaluateCubic(curve: Cubic, t: number) {
  const [p0, p1, p2, p3] = curve,
    u = 1 - t;
  const p = add(
    add(mul(p0, u ** 3), mul(p1, 3 * u * u * t)),
    add(mul(p2, 3 * u * t * t), mul(p3, t ** 3)),
  );
  const v = add(
    add(mul(sub(p1, p0), 3 * u * u), mul(sub(p2, p1), 6 * u * t)),
    mul(sub(p3, p2), 3 * t * t),
  );
  const a = add(mul(add(sub(p2, mul(p1, 2)), p0), 6 * u), mul(add(sub(p3, mul(p2, 2)), p1), 6 * t));
  const speed = length(v);
  if (!Number.isFinite(speed) || speed < 1e-14) throw new Error('Degenerate cubic tangent');
  return { p, v, speed, tangent: mul(v, 1 / speed), curvature: cross(v, a) / speed ** 3 };
}
export function offsetPoint(curve: Cubic, t: number, delta: number) {
  const value = evaluateCubic(curve, t),
    factor = 1 + delta * value.curvature;
  if (!(factor > 1e-8)) throw new Error('Offset reaches the regularity limit');
  return {
    p: add(value.p, mul(point(value.tangent.y, -value.tangent.x), delta)),
    tangent: value.tangent,
    derivative: mul(value.v, factor),
    curvature: value.curvature / factor,
  };
}
function polyAdd(a: number[], b: number[], scale = 1) {
  return Array.from(
    { length: Math.max(a.length, b.length) },
    (_, i) => (a[i] ?? 0) + scale * (b[i] ?? 0),
  );
}
function polyMul(a: number[], b: number[]) {
  const result = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) result[i + j] += a[i] * b[j];
  return result;
}
function roots(coefficients: number[]): number[] {
  const scale = Math.max(...coefficients.map(Math.abs));
  if (scale === 0) return [];
  const p = coefficients.map((c) => c / scale);
  while (p.length > 1 && Math.abs(p.at(-1)!) < 1e-14) p.pop();
  if (p.length < 2) return [];
  if (p.length === 2) {
    const root = -p[0] / p[1];
    return root > 0 && root < 1 ? [root] : [];
  }
  const critical = roots(p.slice(1).map((c, i) => c * (i + 1)));
  const value = (t: number) => p.reduceRight((sum, c) => sum * t + c, 0);
  const points = [0, ...critical, 1],
    result = critical.filter((t) => Math.abs(value(t)) < 1e-12);
  for (let i = 1; i < points.length; i++) {
    let a = points[i - 1],
      b = points[i],
      va = value(a);
    if (va * value(b) >= 0) continue;
    for (let j = 0; j < 55; j++) {
      const mid = (a + b) / 2,
        vm = value(mid);
      if (va * vm <= 0) b = mid;
      else {
        a = mid;
        va = vm;
      }
    }
    result.push((a + b) / 2);
  }
  return result.sort((a, b) => a - b);
}
export function maximumCurvature(curve: Cubic) {
  const derivative = (axis: 'x' | 'y') => [
    3 * (curve[1][axis] - curve[0][axis]),
    6 * (curve[2][axis] - 2 * curve[1][axis] + curve[0][axis]),
    3 * (curve[3][axis] - 3 * curve[2][axis] + 3 * curve[1][axis] - curve[0][axis]),
  ];
  const x = derivative('x'),
    y = derivative('y'),
    ax = [x[1], 2 * x[2]],
    ay = [y[1], 2 * y[2]];
  const vv = polyAdd(polyMul(x, x), polyMul(y, y)),
    va = polyAdd(polyMul(x, ax), polyMul(y, ay));
  const vCrossA = polyAdd(polyMul(x, ay), polyMul(y, ax), -1),
    vCrossJ = polyAdd(
      x.map((n) => n * 2 * y[2]),
      y.map((n) => n * 2 * x[2]),
      -1,
    );
  const extrema = roots(polyAdd(polyMul(vCrossJ, vv), polyMul(vCrossA, va), -3));
  return Math.max(...[0, 1, ...extrema].map((t) => evaluateCubic(curve, t).curvature));
}
type CurveSample = ReturnType<typeof offsetPoint>;
type CurveProfile = (t: number) => CurveSample;
function fit(sample: CurveProfile, start: number, end: number): Cubic | null {
  const a = sample(start),
    b = sample(end),
    chord = sub(b.p, a.p),
    distance = length(chord);
  if (distance < 1e-12) return null;
  const d = mul(chord, 1 / distance),
    t0 = a.tangent,
    t1 = b.tangent,
    k0 = a.curvature * distance,
    k1 = b.curvature * distance,
    crossT = cross(t0, t1);
  let alpha = (length(a.derivative) * (end - start)) / (3 * distance),
    beta = (length(b.derivative) * (end - start)) / (3 * distance);
  const hermite: Cubic = [
    a.p,
    add(a.p, mul(t0, alpha * distance)),
    sub(b.p, mul(t1, beta * distance)),
    b.p,
  ];
  // Near a singular G2 Hermite system, use derivative handles only when their
  // analytic endpoint curvatures satisfy the absolute 2e-6 error budget.
  const approximate = () =>
    Math.max(
      Math.abs(evaluateCubic(hermite, 0).curvature - a.curvature),
      Math.abs(evaluateCubic(hermite, 1).curvature - b.curvature),
    ) < 2e-6
      ? hermite
      : null;
  const residual = (alpha: number, beta: number) => [
    3 * k0 * alpha * alpha - 2 * cross(t0, d) + 2 * beta * crossT,
    3 * k1 * beta * beta - 2 * cross(d, t1) + 2 * alpha * crossT,
  ];
  for (let iteration = 0; iteration < 30; iteration++) {
    const [f0, f1] = residual(alpha, beta),
      error = Math.max(Math.abs(f0), Math.abs(f1));
    if (error < 1e-12)
      return [a.p, add(a.p, mul(t0, alpha * distance)), sub(b.p, mul(t1, beta * distance)), b.p];
    const j0 = 6 * k0 * alpha,
      j1 = 2 * crossT,
      j2 = 6 * k1 * beta,
      det = j0 * j2 - j1 * j1;
    if (Math.abs(det) < 1e-28) return approximate();
    const da = (f0 * j2 - f1 * j1) / det,
      db = (j0 * f1 - j1 * f0) / det;
    let accepted = false;
    for (let damping = 1; damping > 1e-7; damping /= 2) {
      const na = alpha - damping * da,
        nb = beta - damping * db;
      if (na > 0 && nb > 0 && Math.max(...residual(na, nb).map(Math.abs)) < error) {
        alpha = na;
        beta = nb;
        accepted = true;
        break;
      }
    }
    if (!accepted) return approximate();
  }
  return approximate();
}
function fitProfile(sample: CurveProfile, tolerance: number) {
  let sampledError = 0;
  const fitRange = (start: number, end: number, depth: number): Cubic[] => {
    const candidate = fit(sample, start, end);
    let error = Infinity;
    if (candidate) {
      error = 0;
      for (let i = 1; i < 32; i++)
        error = Math.max(
          error,
          length(
            sub(evaluateCubic(candidate, i / 32).p, sample(start + ((end - start) * i) / 32).p),
          ),
        );
    }
    if (candidate && error <= tolerance * 0.75) {
      for (let i = 1; i < 128; i++)
        sampledError = Math.max(
          sampledError,
          length(
            sub(evaluateCubic(candidate, i / 128).p, sample(start + ((end - start) * i) / 128).p),
          ),
        );
      return [candidate];
    }
    if (depth === 14)
      throw new Error(
        `Offset cubic fitting did not converge within ${tolerance}px at [${start},${end}] error ${error}`,
      );
    const mid = (start + end) / 2;
    return [...fitRange(start, mid, depth + 1), ...fitRange(mid, end, depth + 1)];
  };
  const curves = fitRange(0, 1, 0);
  return { curves, sampledError };
}
export function offsetCorner(corner: Corner, delta: number, tolerance = 0.01): Corner {
  if (
    Math.abs(delta) <= 8 * Number.EPSILON * Math.max(1, corner.radius, corner.footprint) ||
    corner.radius === 0
  )
    return corner;
  tolerance = Math.min(tolerance, Math.abs(delta) / 100);
  const reference = corner.reference ?? corner,
    offset = corner.offset + delta;
  const radius = reference.radius + offset;
  if (!(radius > 0)) throw new Error('Offset removes the rounded corner');
  const peak = Math.max(1 / reference.radius, ...reference.leading.map(maximumCurvature));
  if (!(1 + offset * peak > 1e-8))
    throw new Error(
      `Offset reaches the regularity limit (radius ${reference.radius}, offset ${offset}, maximum curvature ${peak})`,
    );
  let sampledError = 0;
  const shift = (p: Point) => point(p.x - offset, p.y + offset);
  const leading = reference.leading
    .flatMap((curve) => {
      const fitted = fitProfile((t) => offsetPoint(curve, t, offset), tolerance);
      sampledError = Math.max(sampledError, fitted.sampledError);
      return fitted.curves;
    })
    .map((curve) => curve.map(shift) as Cubic);
  const trailing = [...leading]
    .reverse()
    .map((curve) => [...curve].reverse().map((p) => point(-p.y, -p.x)) as Cubic);
  let arc = null;
  if (reference.arc) {
    const start = reference.arc.start,
      end = reference.arc.end;
    const center = point(-reference.radius, reference.radius);
    const moved = (p: Point) => shift(add(center, mul(sub(p, center), radius / reference.radius)));
    arc = { start: moved(start), end: moved(end), radius };
  }
  return {
    key: `${reference.key}:${offset}:${tolerance}`,
    radius,
    footprint: reference.footprint + offset,
    leading,
    trailing,
    arc,
    reference,
    offset,
    sampledError,
  };
}
const number = (value: number) => String(Number(value.toFixed(12)));
export function cornerCommands(corner: Corner, transform: (p: Point) => Point = (p) => p) {
  const xy = (p: Point) => {
    const q = transform(p);
    return `${number(q.x)} ${number(q.y)}`;
  };
  const cubic = (c: Cubic) => `C ${xy(c[1])} ${xy(c[2])} ${xy(c[3])}`;
  return [
    ...corner.leading.map(cubic),
    ...(corner.arc
      ? [`A ${number(corner.arc.radius)} ${number(corner.arc.radius)} 0 0 1 ${xy(corner.arc.end)}`]
      : []),
    ...corner.trailing.map(cubic),
  ].join(' ');
}

export type Capsule = {
  kind: 'capsule';
  key: string;
  width: number;
  height: number;
  smoothing: number;
  effectiveSmoothing: number;
  minimumRadius: number;
  depth: number;
  offset: number;
  tolerance: number;
  reference?: Capsule;
};
const gaussX = [
  -0.9602898564975363, -0.7966664774136267, -0.525532409916329, -0.1834346424956498,
  0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363,
];
const gaussW = [
  0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.362683783378362, 0.362683783378362,
  0.3137066458778873, 0.2223810344533745, 0.1012285362903763,
];
// Native geometry2d.curves: unit-peak-curvature smoothstep ramp, integrated in arc length.
function rampPoint(distance: number, ramp: number) {
  let x = 0,
    y = 0;
  for (let i = 0; i < 8; i++) {
    const u = (distance * (1 + gaussX[i])) / (2 * ramp);
    const theta = ramp * (u ** 3 - 0.5 * u ** 4);
    x += Math.cos(theta) * gaussW[i];
    y += Math.sin(theta) * gaussW[i];
  }
  return point((x * distance) / 2, (y * distance) / 2);
}
function capDimensions(smoothing: number) {
  const ramp = Math.PI * smoothing;
  const up = ramp ? rampPoint(ramp, ramp) : point(0, 0);
  return { up, height: 2 * (up.y + Math.cos(ramp / 2)), depth: up.x + 1 - Math.sin(ramp / 2) };
}
const capsuleCache = new Map<string, Capsule>();
export function smoothCapsule(
  width: number,
  height: number,
  smoothing: number,
  tolerance = 0.01,
): Capsule {
  const key = `capsule:${width}:${height}:${smoothing}:${tolerance}`;
  const cached = capsuleCache.get(key);
  if (cached) return cached;
  const long = Math.max(width, height),
    short = Math.min(width, height);
  if (!(short > 0)) throw new Error('Capsule dimensions must be positive');
  const limit = long / (2 * short);
  let q = Math.max(0, Math.min(1, smoothing));
  const dimensions = capDimensions(q);
  if (limit <= 0.5) q = 0;
  else if (dimensions.depth / dimensions.height > limit) {
    let low = 0,
      high = q;
    for (let i = 0; i < 44; i++) {
      const mid = (low + high) / 2,
        size = capDimensions(mid);
      if (size.depth / size.height > limit) high = mid;
      else low = mid;
    }
    q = low;
  }
  const fitted = capDimensions(q),
    scale = short / fitted.height;
  const capsule: Capsule = {
    kind: 'capsule',
    key,
    width,
    height,
    smoothing,
    effectiveSmoothing: q,
    minimumRadius: scale,
    depth: fitted.depth * scale,
    offset: 0,
    tolerance,
  };
  if (capsuleCache.size >= 256) capsuleCache.delete(capsuleCache.keys().next().value!);
  capsuleCache.set(key, capsule);
  return capsule;
}
export function offsetCapsule(capsule: Capsule, delta: number, tolerance = 0.01): Capsule {
  // Radius normalization can subtract two algebraically equal values one ULP apart.
  if (Math.abs(delta) <= 8 * Number.EPSILON * Math.max(1, capsule.width, capsule.height))
    return capsule;
  const reference = capsule.reference ?? capsule,
    offset = capsule.offset + delta;
  if (!(reference.minimumRadius + offset > 1e-8))
    throw new Error('Capsule offset reaches the regularity limit');
  return {
    ...reference,
    reference,
    offset,
    tolerance: Math.min(tolerance, Math.abs(delta) / 100),
    key: `${reference.key}:offset:${offset}:${tolerance}`,
    width: reference.width + 2 * offset,
    height: reference.height + 2 * offset,
    minimumRadius: reference.minimumRadius + offset,
    depth: reference.depth + offset,
  };
}
function capSample(
  reference: Capsule,
  distance: number,
  delta: number,
): CurveSample & { curvatureSlope: number } {
  const q = reference.effectiveSmoothing,
    ramp = q * Math.PI,
    total = Math.PI + ramp;
  const scale = reference.minimumRadius,
    short = Math.min(reference.width, reference.height);
  const down = distance > Math.PI;
  const s = down ? total - distance : distance;
  let p: Point,
    theta: number,
    k: number,
    slope = 0;
  if (ramp && s <= ramp) {
    const u = Math.max(0, Math.min(1, s / ramp));
    p = mul(rampPoint(s, ramp), scale);
    theta = ramp * (u ** 3 - 0.5 * u ** 4);
    k = (u * u * (3 - 2 * u)) / scale;
    slope = (6 * u * (1 - u)) / (ramp * scale * scale);
  } else {
    const up = ramp ? rampPoint(ramp, ramp) : point(0, 0);
    theta = s - ramp / 2;
    p = point(
      (up.x + Math.sin(theta) - Math.sin(ramp / 2)) * scale,
      (up.y + Math.cos(ramp / 2) - Math.cos(theta)) * scale,
    );
    k = 1 / scale;
  }
  if (down) {
    p.y = short - p.y;
    theta = Math.PI - theta;
    slope = -slope;
  }
  const tangent = point(Math.cos(theta), Math.sin(theta)),
    factor = 1 + delta * k;
  if (!(factor > 1e-8)) throw new Error('Capsule offset reaches the regularity limit');
  return {
    p: add(p, mul(point(tangent.y, -tangent.x), delta)),
    tangent,
    derivative: mul(tangent, scale * factor),
    curvature: k / factor,
    curvatureSlope: slope / factor ** 3,
  };
}
/** Analytic right-cap sample in the resulting surface's coordinates; t spans the full cap. */
export function capsuleCapSample(capsule: Capsule, t: number) {
  const reference = capsule.reference ?? capsule,
    delta = capsule.offset;
  const total = Math.PI * (1 + reference.effectiveSmoothing);
  const sample = capSample(reference, Math.max(0, Math.min(1, t)) * total, delta);
  const long = Math.max(reference.width, reference.height);
  let p = add(sample.p, point(long - reference.depth + delta, delta));
  let tangent = sample.tangent;
  if (reference.height > reference.width) {
    p = point(capsule.width - p.y, p.x);
    tangent = point(-tangent.y, tangent.x);
  }
  return { ...sample, p, tangent, derivative: mul(tangent, length(sample.derivative) * total) };
}
const capsulePaddingCache = new Map<string, number>();
/** Place a circular end item so its mean end clearance equals its straight-side clearance. */
export function capsuleEndPadding(capsule: Capsule) {
  const reference = capsule.reference ?? capsule;
  const radius = Math.min(capsule.width, capsule.height) / 2;
  const long = Math.max(capsule.width, capsule.height);
  const q = reference.effectiveSmoothing;
  if (q === 0) return radius;
  const delta = capsule.offset;
  // The cap profile is invariant under rotation and uniform scale. Derived
  // contours also depend on their normal offset, not their new box aspect ratio.
  const key = `${q}:${delta / reference.minimumRadius}`;
  let ratio = capsulePaddingCache.get(key);
  if (ratio === undefined) {
    const half = (Math.PI * (1 + q)) / 2;
    const ramp = Math.PI * q;
    const depth = capsule.depth / radius;
    const sample = (distance: number) => {
      const value = capSample(reference, distance, delta);
      return {
        x: (reference.depth - value.p.x + delta) / radius,
        y: (value.p.y + delta) / radius,
        dx: -value.derivative.x / radius,
        dy: value.derivative.y / radius,
      };
    };
    const integrate = (f: (t: number) => number, start: number, end: number): number => {
      const sum = (a: number, b: number) => {
        let value = 0;
        for (let i = 0; i < 8; i++) value += gaussW[i] * f((a + b + (b - a) * gaussX[i]) / 2);
        return (value * (b - a)) / 2;
      };
      const split = (a: number, b: number, whole: number, depth: number): number => {
        const mid = (a + b) / 2,
          left = sum(a, mid),
          right = sum(mid, b);
        if (Math.abs(left + right - whole) < 1e-9 * (b - a)) return left + right;
        if (depth === 10) throw new Error('Capsule optical spacing integration did not converge');
        return split(a, mid, left, depth + 1) + split(mid, b, right, depth + 1);
      };
      return end > start ? split(start, end, sum(start, end), 0) : 0;
    };
    const meanRadius = (center: number) => {
      let lo = 0,
        hi = half;
      if (center < depth) {
        for (let i = 0; i < 36; i++) {
          const mid = (lo + hi) / 2;
          if (sample(mid).x > center) lo = mid;
          else hi = mid;
        }
      } else hi = 0;
      const f = (t: number) => {
        const { x, y, dx, dy } = sample(t);
        const vx = x - center,
          vy = y - 1;
        // r d(theta) = |cross(P-C, dP)| / |P-C| along the visible half-cap.
        return (vy * dx - vx * dy) / Math.hypot(vx, vy);
      };
      const curve =
        hi < ramp ? integrate(f, hi, ramp) + integrate(f, ramp, half) : integrate(f, hi, half);
      return (2 / Math.PI) * (curve + Math.asinh(Math.max(0, center - depth)));
    };
    let lo = 1,
      hi = depth;
    for (let i = 0; i < 32; i++) {
      const mid = (lo + hi) / 2;
      if (meanRadius(mid) < 1) lo = mid;
      else hi = mid;
    }
    ratio = (lo + hi) / 2;
    if (capsulePaddingCache.size >= 128)
      capsulePaddingCache.delete(capsulePaddingCache.keys().next().value!);
    capsulePaddingCache.set(key, ratio);
  }
  return Math.min(long / 2, radius * ratio);
}
const capCurveCache = new Map<string, Cubic[]>();
function capsulePath(capsule: Capsule) {
  const reference = capsule.reference ?? capsule,
    delta = capsule.offset;
  const long = Math.max(reference.width, reference.height),
    short = Math.min(reference.width, reference.height);
  const ramp = Math.PI * reference.effectiveSmoothing;
  // Long-axis resizing translates the same cap. Reuse its fitted curves across
  // widths instead of fitting again during panel and search-field resizing.
  const capKey = `${short}:${ramp}:${reference.minimumRadius}:${delta}:${capsule.tolerance}`;
  let leading = capCurveCache.get(capKey);
  if (!leading) {
    leading = ramp
      ? fitProfile((t) => {
          const value = capSample(reference, t * ramp, delta);
          return { ...value, derivative: mul(value.derivative, ramp) };
        }, capsule.tolerance).curves
      : [];
    if (capCurveCache.size >= 256) capCurveCache.delete(capCurveCache.keys().next().value!);
    capCurveCache.set(capKey, leading);
  }
  const trailing = [...leading]
    .reverse()
    .map((curve) => [...curve].reverse().map((p) => point(p.x, short - p.y)) as Cubic);
  const transform = (p: Point, left: boolean) => {
    const q = left
      ? point(reference.depth - p.x + delta, short - p.y + delta)
      : point(long - reference.depth + p.x + delta, p.y + delta);
    return reference.height > reference.width ? point(capsule.width - q.y, q.x) : q;
  };
  const xy = (p: Point, left: boolean) => {
    const q = transform(p, left);
    return `${number(q.x)} ${number(q.y)}`;
  };
  const start = capSample(reference, 0, delta).p;
  let path = `M ${xy(start, false)}`;
  for (const left of [false, true]) {
    if (left) path += ` L ${xy(start, true)}`;
    for (const curve of leading)
      path += ` C ${curve
        .slice(1)
        .map((p) => xy(p, left))
        .join(' ')}`;
    if (reference.effectiveSmoothing < 1)
      path += ` A ${number(reference.minimumRadius + delta)} ${number(reference.minimumRadius + delta)} 0 0 1 ${xy(capSample(reference, Math.PI, delta).p, left)}`;
    for (const curve of trailing)
      path += ` C ${curve
        .slice(1)
        .map((p) => xy(p, left))
        .join(' ')}`;
  }
  return path + ' Z';
}

export type SurfaceCorners = [Corner, Corner, Corner, Corner];
export type SurfaceGeometry = SurfaceCorners | Capsule;
const cornerCache = new Map<string, Corner>();
export function derivedCorner(
  radius: number,
  smoothing: number,
  budget: number,
  offset: number,
  tolerance: number,
) {
  const key = `${radius}:${smoothing}:${budget}:${offset}:${tolerance}`;
  let corner = cornerCache.get(key);
  if (!corner) {
    corner = offsetCorner(smoothCorner(radius, smoothing, budget), offset, tolerance);
    if (cornerCache.size >= 256) cornerCache.delete(cornerCache.keys().next().value!);
    cornerCache.set(key, corner);
  }
  return corner;
}
export function shiftedCorner(corner: Corner, delta: number, tolerance: number) {
  if (corner.radius === 0 || delta === 0) return corner;
  const key = `${corner.key}:shift:${delta}:${tolerance}`;
  let result = cornerCache.get(key);
  if (!result) {
    result = offsetCorner(corner, delta, tolerance);
    if (cornerCache.size >= 256) cornerCache.delete(cornerCache.keys().next().value!);
    cornerCache.set(key, result);
  }
  return result;
}
/** CSS scales all circular radii together only when an adjacent pair exceeds its side. */
export function normalizeRadii(width: number, height: number, radii: readonly number[]) {
  const values = radii.map((radius) => Math.max(0, radius));
  const ratio = (side: number, a: number, b: number) =>
    values[a] + values[b] > 0 ? side / (values[a] + values[b]) : Infinity;
  const scale = Math.min(
    1,
    ratio(width, 0, 1),
    ratio(width, 3, 2),
    ratio(height, 0, 3),
    ratio(height, 1, 2),
  );
  return values.map((radius) => radius * scale);
}
export function cornerBudgets(width: number, height: number, radii: number[]) {
  const sides = [
    [1, 3, width, height],
    [0, 2, width, height],
    [3, 1, width, height],
    [2, 0, width, height],
  ];
  return sides.map(([horizontal, vertical, w, h], i) => {
    const radius = radii[i];
    if (radius === 0) return 0;
    // Each side is shared in proportion to its radii; opposing budgets cannot overlap.
    return Math.min(
      (w * radius) / (radius + radii[horizontal]),
      (h * radius) / (radius + radii[vertical]),
    );
  });
}
const pathCache = new Map<string, string>();
export function surfacePath(width: number, height: number, corners: SurfaceGeometry) {
  if (!Array.isArray(corners)) {
    const cached = pathCache.get(corners.key);
    if (cached) return cached;
    const path = capsulePath(corners);
    if (pathCache.size >= 512) pathCache.delete(pathCache.keys().next().value!);
    pathCache.set(corners.key, path);
    return path;
  }
  const key = `${width}:${height}:${corners.map((c) => c.key).join('|')}`;
  const cached = pathCache.get(key);
  if (cached) return cached;
  const transforms = [
    (p: Point) => point(p.y, -p.x),
    (p: Point) => point(width + p.x, p.y),
    (p: Point) => point(width - p.y, height + p.x),
    (p: Point) => point(-p.x, height - p.y),
  ];
  let path = `M ${number(corners[0].footprint)} 0`;
  for (const i of [1, 2, 3, 0]) {
    const start = transforms[i](point(-corners[i].footprint, 0));
    path += ` L ${number(start.x)} ${number(start.y)} ${cornerCommands(corners[i], transforms[i])}`;
  }
  path += ' Z';
  if (pathCache.size >= 512) pathCache.delete(pathCache.keys().next().value!);
  pathCache.set(key, path);
  return path;
}
