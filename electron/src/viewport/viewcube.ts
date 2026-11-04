/** Direct geometry/behavior port of python/ui/viewcube.py and geometry2d/curves.py. */
import type { Camera } from 'three';
export type Axis = 0 | 1 | 2;
export type Point = [number, number];
export interface Ball {
  axis: Axis;
  sign: 1 | -1;
  x: number;
  y: number;
  radius: number;
  depth: number;
  alpha: number;
}
export interface ViewCubeOptions {
  onAxis: (axis: Axis, sign: 1 | -1) => void;
  onProjectionToggle?: () => void;
  onFocus?: () => void;
  onOrbit: (dx: number, dy: number) => void;
  invalidate?: () => void;
}
const RADIUS = 34,
  BALL = 9.5,
  LINE = 2,
  BACKDROP = 45.5,
  INSET = 5;
const COLORS = [
  [220, 119, 115],
  [82, 170, 92],
  [111, 148, 229],
];
const DARK_BASE = [25.5, 28.05, 33.15],
  TOLERANCE = 0.025;
const GAUSS_X = [
  -0.9602898564975363, -0.7966664774136267, -0.525532409916329, -0.1834346424956498,
  0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363,
];
const GAUSS_W = [
  0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.362683783378362, 0.362683783378362,
  0.3137066458778873, 0.2223810344533745, 0.1012285362903763,
];
const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export function backAlpha(depth: number): number {
  const t = clamp((0.96 - depth) / (0.96 - 0.72));
  return t * t * (3 - 2 * t);
}
export function yawPitchForAxis(axis: Axis, sign: 1 | -1): Point {
  return axis === 2 ? [-90, 89.9 * sign] : [(axis === 0 ? 0 : 90) + (sign < 0 ? 180 : 0), 0];
}
export function viewCubeLayout(
  right: number[],
  up: number[],
  forward: number[],
  center: Point = [45, 45],
  scale = 90 / 91,
): Ball[] {
  const out: Ball[] = [];
  for (let axis = 0; axis < 3; axis++)
    for (const sign of [1, -1] as const) {
      const depth = forward[axis] * sign;
      out.push({
        axis: axis as Axis,
        sign,
        x: center[0] + right[axis] * sign * RADIUS * scale,
        y: center[1] - up[axis] * sign * RADIUS * scale,
        radius: BALL * scale,
        depth,
        alpha: backAlpha(depth),
      });
    }
  return out.sort((a, b) => b.depth - a.depth);
}
export function hitTest(balls: Ball[], p: Point): Ball | null {
  let best: Ball | null = null;
  for (const b of balls)
    if (
      b.alpha > 0.1 &&
      (p[0] - b.x) ** 2 + (p[1] - b.y) ** 2 <= b.radius ** 2 &&
      (!best || b.depth < best.depth)
    )
      best = b;
  return best;
}
function thetaAt(s: number, phi: number, q: number) {
  const ramp = q * phi,
    arc = (1 - q) * phi,
    total = 2 * ramp + arc;
  if (ramp === 0) return s;
  const down = s > ramp + arc,
    u = clamp((down ? total - s : s) / ramp),
    theta = ramp * (u ** 3 - 0.5 * u ** 4);
  return down ? phi - theta : s > ramp ? s - ramp * 0.5 : theta;
}
function turnEnd(phi: number, q: number): Point {
  const ramp = q * phi,
    arc = (1 - q) * phi;
  let x = 0,
    y = 0;
  for (let i = 0; i < 8; i++) {
    const theta = thetaAt((1 + GAUSS_X[i]) * ramp * 0.5, phi, q);
    x += Math.cos(theta) * GAUSS_W[i] * ramp * 0.5;
    y += Math.sin(theta) * GAUSS_W[i] * ramp * 0.5;
  }
  const mx = arc === 0 ? 0 : Math.sin(phi - ramp * 0.5) - Math.sin(ramp * 0.5),
    my = arc === 0 ? 0 : Math.cos(ramp * 0.5) - Math.cos(phi - ramp * 0.5);
  return [
    x + mx + Math.cos(phi) * x + Math.sin(phi) * y,
    y + my + Math.sin(phi) * x - Math.cos(phi) * y,
  ];
}
function turnPoints(phi: number, scale: number, q: number): Point[] {
  const ramp = q * phi,
    arc = (1 - q) * phi,
    total = 2 * ramp + arc,
    step = Math.sqrt((8 * TOLERANCE) / scale);
  const breaks = [...new Set([0, ramp, total * 0.5, ramp + arc, total])].sort((a, b) => a - b),
    s = [0];
  for (let i = 1; i < breaks.length; i++) {
    const lo = breaks[i - 1],
      hi = breaks[i],
      n = Math.max(1, Math.ceil((hi - lo) / step));
    for (let j = 1; j <= n; j++) s.push(lo + ((hi - lo) * j) / n);
  }
  let x = 0,
    y = 0;
  const out: Point[] = [[0, 0]];
  for (let i = 1; i < s.length; i++) {
    const half = (s[i] - s[i - 1]) * 0.5,
      mid = (s[i] + s[i - 1]) * 0.5;
    for (let j = 0; j < 8; j++) {
      const t = thetaAt(mid + half * GAUSS_X[j], phi, q);
      x += half * Math.cos(t) * GAUSS_W[j];
      y += half * Math.sin(t) * GAUSS_W[j];
    }
    out.push([x * scale, y * scale]);
  }
  const end = turnEnd(phi, q);
  out[out.length - 1] = [end[0] * scale, end[1] * scale];
  return out;
}
function capDepthRatio(q: number): number {
  const ramp = q * Math.PI;
  let x = 0,
    y = 0;
  for (let i = 0; i < 8; i++) {
    const u = (1 + GAUSS_X[i]) * 0.5,
      phase = u ** 3 - 0.5 * u ** 4,
      w = GAUSS_W[i] * 0.5;
    x += w * Math.cos(ramp * phase);
    y += w * Math.sin(ramp * phase);
  }
  return (ramp * x + 1 - Math.sin(ramp * 0.5)) / (2 * (ramp * y + Math.cos(ramp * 0.5)));
}
function lineCap(distance: number, width: number, q: number, maxInset: number): Point[] {
  const limit = maxInset / width + 0.5;
  if (limit < capDepthRatio(q)) {
    if (limit <= 0.5) q = 0;
    else {
      let lo = 0,
        hi = q;
      for (let i = 0; i < 44; i++) {
        const mid = (lo + hi) * 0.5;
        if (capDepthRatio(mid) > limit) hi = mid;
        else lo = mid;
      }
      q = lo;
    }
  }
  const height = turnEnd(Math.PI, q)[1],
    cap = turnPoints(Math.PI, width / height, q);
  for (const p of cap) p[1] -= width * 0.5;
  cap[0] = [0, -width * 0.5];
  cap[cap.length - 1] = [0, width * 0.5];
  const depth = Math.max(...cap.map((p) => p[0]));
  return cap.map((p) => [-distance - (p[0] + width * 0.5 - depth), p[1]]);
}
function sampleBezier(control: Point[]): Point[] {
  const result: Point[] = [control[0]];
  function visit(p: Point[]) {
    const dx = p[p.length - 1][0] - p[0][0],
      dy = p[p.length - 1][1] - p[0][1],
      len = dx * dx + dy * dy;
    const error = Math.max(
      ...p.map((v) => {
        const t = clamp(((v[0] - p[0][0]) * dx + (v[1] - p[0][1]) * dy) / Math.max(len, 1e-30));
        return Math.hypot(v[0] - p[0][0] - t * dx, v[1] - p[0][1] - t * dy);
      }),
    );
    if (error <= TOLERANCE) {
      result.push(p[p.length - 1]);
      return;
    }
    const levels = [p];
    for (let i = 1; i < p.length; i++) {
      const prev = levels[i - 1];
      levels.push(
        prev.slice(1).map((v, j) => [(v[0] + prev[j][0]) * 0.5, (v[1] + prev[j][1]) * 0.5]),
      );
    }
    visit(levels.map((l) => l[0]));
    visit([...levels].reverse().map((l) => l[l.length - 1]));
  }
  visit(control);
  return result;
}
function lollipopHead(radius: number, half: number, run: number, q: number): Point[] {
  const angle = -Math.PI + Math.asin(half / radius) + 0.35 * q,
    radial: Point = [Math.cos(angle), Math.sin(angle)],
    tangent: Point = [-radial[1], radial[0]],
    start: Point = [-Math.sqrt(radius * radius - half * half) - run, -half],
    end: Point = [radius * radial[0], radius * radial[1]],
    speed = Math.hypot(end[0] - start[0], end[1] - start[1]);
  const p: Point[] = Array.from({ length: 8 }, () => [0, 0]);
  for (let i = 0; i < 4; i++) p[i] = [start[0] + (i * speed) / 7, start[1]];
  p[7] = end;
  p[6] = [end[0] - (speed * tangent[0]) / 7, end[1] - (speed * tangent[1]) / 7];
  for (const d of [0, 1]) {
    p[5][d] = (-((speed * speed) / radius) * radial[d]) / 42 + 2 * p[6][d] - p[7][d];
    p[4][d] = 3 * p[5][d] - 3 * p[6][d] + p[7][d] + ((speed ** 3 / radius ** 2) * tangent[d]) / 210;
  }
  const bridge = q > 0 ? sampleBezier(p) : [end],
    n = Math.max(3, Math.ceil((-2 * angle) / Math.sqrt((8 * TOLERANCE) / radius)) + 1),
    arc: Point[] = [];
  for (let i = 1; i < n - 1; i++) {
    const a = angle - (2 * angle * i) / (n - 1);
    arc.push([radius * Math.cos(a), radius * Math.sin(a)]);
  }
  return [...bridge, ...arc, ...[...bridge].reverse().map((p) => [p[0], -p[1]] as Point)];
}
/** Same adaptive seventh-order neck, cap curvature quadrature and collapsing-head limit as native. */
export function lollipopPoints(
  distance: number,
  radius: number,
  width: number,
  smoothing = 0.618,
): Point[] {
  if (distance <= radius) return lollipopHead(radius, 0, 0, 0).slice(0, -1);
  const visible = clamp((distance - radius) / width),
    half = Math.min(width * 0.5, radius * 0.5) * visible,
    q = smoothing * visible,
    run = Math.min(radius * 0.6, (distance - radius) * 0.4) * q;
  const head = lollipopHead(radius, half, run, q),
    cap = lineCap(distance, half * 2, q, Math.max(0, distance + head[0][0]) * 0.4);
  let points = [cap[0], ...head, ...cap.slice(1).reverse()];
  if (visible < 1) {
    const blend = visible * visible * (3 - 2 * visible);
    points = points.map((p) => {
      const k = blend + ((1 - blend) * radius) / Math.hypot(...p);
      return [p[0] * k, p[1] * k];
    });
    const edge = Math.min(TOLERANCE * 0.01, Math.max(1, radius) * 1e-7);
    points = points.filter(
      (p, i) =>
        Math.hypot(
          p[0] - points[(i + points.length - 1) % points.length][0],
          p[1] - points[(i + points.length - 1) % points.length][1],
        ) > edge,
    );
  }
  return points;
}
export interface ArcRibbonMesh {
  vertices: Point[];
  /** Null requests triangulation after native folded-offset exterior repair. */
  indices: number[] | null;
}
/** Native arc_ribbon_mesh preserves strip topology unless an offset triangle folds. */
export function arcRibbonMesh(
  input: Point[],
  width: number,
  roundStart = true,
  roundEnd = true,
  startRadial: Point | null = null,
  endRadial: Point | null = null,
): ArcRibbonMesh {
  if (input.length < 2 || width <= 0) return { vertices: [], indices: [] };
  const origin = input[0],
    points = input.map(
      (p) =>
        [
          Math.round((p[0] - origin[0]) * 1e9) / 1e9,
          Math.round((p[1] - origin[1]) * 1e9) / 1e9,
        ] as Point,
    ),
    n = points.length;
  const sub = (a: Point, b: Point): Point => [a[0] - b[0], a[1] - b[1]],
    cross = (a: Point, b: Point) => a[0] * b[1] - a[1] * b[0];
  const tangents = points.map((p, i) => {
    let t = sub(points[Math.min(n - 1, i + 1)], points[Math.max(0, i - 1)]);
    if (Math.hypot(...t) < 1e-6) {
      const alternatives = [];
      if (i > 0) alternatives.push(sub(p, points[i - 1]));
      if (i < n - 1) alternatives.push(sub(points[i + 1], p));
      t = alternatives.sort((a, b) => Math.hypot(...b) - Math.hypot(...a))[0];
    }
    return t;
  });
  if (tangents.some((t) => Math.hypot(...t) < 1e-6)) return { vertices: [], indices: [] };
  const dirs = tangents.map((t) => {
      const d = Math.hypot(...t);
      return [t[0] / d, t[1] / d] as Point;
    }),
    offset = dirs.map((t) => [-t[1], t[0]] as Point),
    half = width * 0.5;
  for (const [index, radial, rounded] of [
    [0, startRadial, roundStart],
    [n - 1, endRadial, roundEnd],
  ] as [number, Point | null, boolean][]) {
    if (!radial || rounded) continue;
    const length = Math.hypot(...radial);
    if (length < 1e-6) continue;
    const direction: Point = [radial[0] / length, radial[1] / length],
      sign = direction[0] * offset[index][0] + direction[1] * offset[index][1] < 0 ? -1 : 1;
    offset[index] = [direction[0] * sign, direction[1] * sign];
  }
  const a = points.map((p, i) => [p[0] - offset[i][0] * half, p[1] - offset[i][1] * half] as Point),
    b = points.map((p, i) => [p[0] + offset[i][0] * half, p[1] + offset[i][1] * half] as Point),
    count = Math.max(8, Math.ceil(Math.PI * half)),
    start: Point[] = [],
    end: Point[] = [];
  for (let i = 1; i < count; i++) {
    const angle = (i * Math.PI) / count,
      c = Math.cos(angle),
      s = Math.sin(angle);
    end.push([
      points[n - 1][0] + half * (-c * offset[n - 1][0] + s * dirs[n - 1][0]),
      points[n - 1][1] + half * (-c * offset[n - 1][1] + s * dirs[n - 1][1]),
    ]);
    start.push([
      points[0][0] + half * (c * offset[0][0] - s * dirs[0][0]),
      points[0][1] + half * (c * offset[0][1] - s * dirs[0][1]),
    ]);
  }
  let outline = [...a, ...(roundEnd ? end : []), ...b.reverse(), ...(roundStart ? start : [])];
  const endCount = roundEnd ? end.length : 0,
    startCount = roundStart ? start.length : 0,
    indices: number[] = [];
  for (let a = 0; a < n - 1; a++) {
    const b = n * 2 + endCount - 1 - a;
    indices.push(a, a + 1, b - 1, a, b - 1, b);
  }
  for (let i = n; i < n + endCount; i++) indices.push(n - 1, i, i + 1);
  const startIndex = n * 2 + endCount;
  for (let i = startIndex; i < startIndex + startCount; i++)
    indices.push(startIndex - 1, i, i + 1 < startIndex + startCount ? i + 1 : 0);
  const world = (points: Point[]): Point[] =>
    points.map((p) => [p[0] + origin[0], p[1] + origin[1]]);
  let folded = false;
  for (let i = 0; i < indices.length; i += 3)
    if (
      cross(
        sub(outline[indices[i + 1]], outline[indices[i]]),
        sub(outline[indices[i + 2]], outline[indices[i]]),
      ) < -1e-9
    ) {
      folded = true;
      break;
    }
  // Near-full arcs can have overlapping end caps without folded offset triangles.
  // Their strip indices retain the hole; exterior repair would turn them into disks.
  if (!folded) return { vertices: world(outline), indices };
  const area = (path: Point[]) =>
    path.reduce((sum, p, i) => sum + cross(p, path[(i + 1) % path.length]), 0) * 0.5;
  for (let repair = 0; repair < input.length * 2; repair++) {
    let found: false | [number, number, Point] = false;
    for (let first = 0; first < outline.length && !found; first++) {
      const p = outline[first],
        d = sub(outline[(first + 1) % outline.length], p);
      for (let second = first + 2; second < outline.length; second++) {
        const q = outline[second],
          other = sub(outline[(second + 1) % outline.length], q),
          den = cross(d, other);
        if (Math.abs(den) <= 1e-9) continue;
        const delta = sub(q, p),
          t = cross(delta, other) / den,
          u = cross(delta, d) / den;
        if (t > 1e-9 && t < 1 - 1e-9 && u > 1e-9 && u < 1 - 1e-9) {
          found = [first, second, [p[0] + d[0] * t, p[1] + d[1] * t]];
          break;
        }
      }
    }
    if (!found) return { vertices: world(outline), indices: null };
    const [first, second, p] = found,
      inside = [p, ...outline.slice(first + 1, second + 1)],
      outside = [...outline.slice(0, first + 1), p, ...outline.slice(second + 1)];
    outline = Math.abs(area(inside)) > Math.abs(area(outside)) ? inside : outside;
  }
  throw new Error('ViewCube stroke exterior did not converge');
}
/** Boundary compatibility for callers that inspect native stroke geometry. */
export function openStrokeOutline(
  input: Point[],
  width: number,
  roundStart = true,
  roundEnd = true,
  startRadial: Point | null = null,
  endRadial: Point | null = null,
): Point[] {
  return arcRibbonMesh(input, width, roundStart, roundEnd, startRadial, endRadial).vertices;
}

/** G3 round-ended two-point stroke used by native joint ticks and value rails. */
export function g3LineOutline(
  start: Point,
  end: Point,
  width: number,
  roundStart = true,
  roundEnd = true,
): Point[] {
  const dx = end[0] - start[0],
    dy = end[1] - start[1],
    length = Math.hypot(dx, dy);
  if (length < 1e-9 || width <= 0) return [];
  const u: Point = [dx / length, dy / length],
    v: Point = [-u[1], u[0]],
    cap = lineCap(0, width, 0.618, length * 0.45);
  const a = cap.map(
      (p) => [start[0] + u[0] * p[0] + v[0] * p[1], start[1] + u[1] * p[0] + v[1] * p[1]] as Point,
    ),
    b = cap.map(
      (p) => [end[0] - u[0] * p[0] - v[0] * p[1], end[1] - u[1] * p[0] - v[1] * p[1]] as Point,
    );
  return [
    roundStart
      ? a[a.length - 1]
      : ([start[0] + v[0] * width * 0.5, start[1] + v[1] * width * 0.5] as Point),
    ...(roundEnd
      ? b
      : ([
          [end[0] + v[0] * width * 0.5, end[1] + v[1] * width * 0.5],
          [end[0] - v[0] * width * 0.5, end[1] - v[1] * width * 0.5],
        ] as Point[])),
    roundStart ? a[0] : ([start[0] - v[0] * width * 0.5, start[1] - v[1] * width * 0.5] as Point),
    ...(roundStart ? a.slice(1, -1) : []),
  ].reverse();
}

/** ImGui-compatible solid triangles and explicit AA fringes. No browser path AA is used. */
export class NativeRaster {
  readonly canvas = document.createElement('canvas');
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private buffer: WebGLBuffer;
  private extent: WebGLUniformLocation | null;
  private vertices = new Float32Array(131072);
  private count = 0;
  private width = 90;
  private height = 90;
  constructor() {
    const gl = this.canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('ViewCube requires WebGL2 for native antialiasing parity');
    this.gl = gl;
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, source);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(s) || 'ViewCube shader compilation failed');
      return s;
    };
    const vs = shader(
      gl.VERTEX_SHADER,
      '#version 300 es\nin vec2 position;in vec4 color;uniform vec2 extent;out vec4 ink;void main(){gl_Position=vec4(position.x/extent.x*2.-1.,1.-position.y/extent.y*2.,0.,1.);ink=color;}',
    );
    const fs = shader(
      gl.FRAGMENT_SHADER,
      '#version 300 es\nprecision highp float;in vec4 ink;out vec4 fragColor;void main(){fragColor=vec4(ink.rgb*ink.a,ink.a);}',
    );
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(this.program) || 'ViewCube shader link failed');
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertices.byteLength, gl.DYNAMIC_DRAW);
    gl.useProgram(this.program);
    const p = gl.getAttribLocation(this.program, 'position'),
      c = gl.getAttribLocation(this.program, 'color');
    gl.enableVertexAttribArray(p);
    gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(c);
    gl.vertexAttribPointer(c, 4, gl.FLOAT, false, 24, 8);
    this.extent = gl.getUniformLocation(this.program, 'extent');
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }
  resize(width: number, height: number, dpr: number) {
    this.width = width;
    this.height = height;
    const w = Math.round(width * dpr),
      h = Math.round(height * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    this.gl.uniform2f(this.extent, width, height);
    this.count = 0;
  }
  dispose() {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteProgram(this.program);
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
  private color(value: string): number[] {
    if (value.startsWith('#')) {
      const v = value.slice(1);
      return [
        parseInt(v.slice(0, 2), 16) / 255,
        parseInt(v.slice(2, 4), 16) / 255,
        parseInt(v.slice(4, 6), 16) / 255,
        v.length === 8 ? parseInt(v.slice(6, 8), 16) / 255 : 1,
      ];
    }
    if (value === 'white') return [1, 1, 1, 1];
    const values = value.match(/[\d.]+/g)!.map(Number);
    return [
      Math.round(values[0]) / 255,
      Math.round(values[1]) / 255,
      Math.round(values[2]) / 255,
      Math.round((values[3] ?? 1) * 255) / 255,
    ];
  }
  private vertex(point: Point, color: number[], opacity = 1) {
    if (this.count + 6 > this.vertices.length) throw new Error('ViewCube vertex capacity exceeded');
    this.vertices[this.count++] = point[0];
    this.vertices[this.count++] = point[1];
    this.vertices[this.count++] = color[0];
    this.vertices[this.count++] = color[1];
    this.vertices[this.count++] = color[2];
    this.vertices[this.count++] = color[3] * opacity;
  }
  private triangle(a: Point, b: Point, c: Point, color: number[], aa = 1, ab = 1, ac = 1) {
    this.vertex(a, color, aa);
    this.vertex(b, color, ab);
    this.vertex(c, color, ac);
  }
  private normalized(points: Point[]): Point[] {
    let area = 0;
    for (let i = 0; i < points.length; i++) {
      const a = points[i],
        b = points[(i + 1) % points.length];
      area += a[0] * b[1] - b[0] * a[1];
    }
    return area < 0 ? [...points].reverse() : points;
  }
  private miters(points: Point[]): Point[] {
    const normals = points.map((p, i) => {
      const b = points[(i + 1) % points.length],
        dx = b[0] - p[0],
        dy = b[1] - p[1],
        d = Math.hypot(dx, dy);
      return [dy / d, -dx / d] as Point;
    });
    return normals.map((n, i) => {
      const previous = normals[(i + normals.length - 1) % normals.length],
        x = (n[0] + previous[0]) * 0.5,
        y = (n[1] + previous[1]) * 0.5,
        factor = Math.min(100, 1 / Math.max(x * x + y * y, 1e-30));
      return [x * factor, y * factor];
    });
  }
  private triangulate(points: Point[], color: number[], paint = points) {
    const cross = (a: Point, b: Point, c: Point) =>
      (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (
      points.every(
        (p, i) => cross(p, points[(i + 1) % points.length], points[(i + 2) % points.length]) >= 0,
      )
    ) {
      for (let i = 1; i < points.length - 1; i++)
        this.triangle(paint[0], paint[i], paint[i + 1], color);
      return;
    }
    const remaining = points.map((_, i) => i);
    let guard = points.length ** 2;
    while (remaining.length > 3 && guard--) {
      let found = false;
      for (let i = 0; i < remaining.length; i++) {
        const a = points[remaining[(i + remaining.length - 1) % remaining.length]],
          b = points[remaining[i]],
          c = points[remaining[(i + 1) % remaining.length]];
        if (cross(a, b, c) <= 1e-12) continue;
        let inside = false;
        for (let j = 0; j < remaining.length; j++) {
          if (
            j === i ||
            j === (i + 1) % remaining.length ||
            j === (i + remaining.length - 1) % remaining.length
          )
            continue;
          const p = points[remaining[j]];
          if (cross(a, b, p) >= -1e-12 && cross(b, c, p) >= -1e-12 && cross(c, a, p) >= -1e-12) {
            inside = true;
            break;
          }
        }
        if (inside) continue;
        this.triangle(
          paint[remaining[(i + remaining.length - 1) % remaining.length]],
          paint[remaining[i]],
          paint[remaining[(i + 1) % remaining.length]],
          color,
        );
        remaining.splice(i, 1);
        found = true;
        break;
      }
      if (!found) throw new Error('ViewCube polygon triangulation failed');
    }
    if (remaining.length === 3)
      this.triangle(paint[remaining[0]], paint[remaining[1]], paint[remaining[2]], color);
  }
  fill(raw: Point[], value: string, fringe = 1, centered = false, opacity = 1) {
    const points = this.normalized(raw),
      color = this.color(value),
      miters = this.miters(points),
      inner = points.map(
        (p, i) =>
          [
            p[0] - miters[i][0] * (centered ? fringe * 0.5 : 0),
            p[1] - miters[i][1] * (centered ? fringe * 0.5 : 0),
          ] as Point,
      ),
      outer = points.map(
        (p, i) =>
          [
            p[0] + miters[i][0] * fringe * (centered ? 0.5 : 1),
            p[1] + miters[i][1] * fringe * (centered ? 0.5 : 1),
          ] as Point,
      );
    color[3] *= opacity;
    this.triangulate(points, color, inner);
    for (let i = 0; i < points.length; i++) {
      const j = (i + 1) % points.length;
      this.triangle(inner[i], inner[j], outer[j], color, 1, 1, 0);
      this.triangle(inner[i], outer[j], outer[i], color, 1, 0, 0);
    }
  }
  /** Native triangle_fan_fill keeps fan indices and fringes only the complete outer boundary. */
  fan(points: Point[], value: string, opacity = 1) {
    if (points.length < 3) return;
    const color = this.color(value);
    color[3] *= opacity;
    for (let i = 1; i < points.length - 1; i++)
      this.triangle(points[0], points[i], points[i + 1], color);
    const outline = this.normalized(points),
      miters = this.miters(outline),
      outer = outline.map((p, i) => [p[0] + miters[i][0], p[1] + miters[i][1]] as Point);
    for (let i = 0; i < outline.length; i++) {
      const j = (i + 1) % outline.length;
      this.triangle(outline[i], outline[j], outer[j], color, 1, 1, 0);
      this.triangle(outline[i], outer[j], outer[i], color, 1, 0, 0);
    }
  }
  strokeOpen(
    points: Point[],
    value: string,
    width: number,
    opacity = 1,
    roundStart = true,
    roundEnd = true,
  ) {
    this.ribbon(arcRibbonMesh(points, width, roundStart, roundEnd), value, opacity);
  }
  /** Native indexed_fill uses authored triangles and fringes only the boundary. */
  ribbon(mesh: ArcRibbonMesh, value: string, opacity = 1) {
    if (!mesh.vertices.length) return;
    if (mesh.indices === null) {
      this.fill(mesh.vertices, value, 1, false, opacity);
      return;
    }
    const color = this.color(value);
    color[3] *= opacity;
    for (let i = 0; i < mesh.indices.length; i += 3)
      this.triangle(
        mesh.vertices[mesh.indices[i]],
        mesh.vertices[mesh.indices[i + 1]],
        mesh.vertices[mesh.indices[i + 2]],
        color,
      );
    const outline = this.normalized(mesh.vertices),
      miters = this.miters(outline),
      outer = outline.map((p, i) => [p[0] + miters[i][0], p[1] + miters[i][1]] as Point);
    for (let i = 0; i < outline.length; i++) {
      const j = (i + 1) % outline.length;
      this.triangle(outline[i], outline[j], outer[j], color, 1, 1, 0);
      this.triangle(outline[i], outer[j], outer[i], color, 1, 0, 0);
    }
  }
  strokeLineG3(
    start: Point,
    end: Point,
    value: string,
    width: number,
    opacity = 1,
    roundStart = true,
    roundEnd = true,
  ) {
    const outline = g3LineOutline(start, end, width, roundStart, roundEnd);
    if (outline.length) this.fill(outline, value, 1, false, opacity);
  }
  stroke(raw: Point[], value: string, width: number, opacity = 1, closed = true) {
    const points = closed ? this.normalized(raw) : raw,
      m = this.miters(points),
      color = this.color(value),
      half = Math.max(0, (width - 1) * 0.5);
    if (!closed) {
      const normal = (a: Point, b: Point): Point => {
        const dx = b[0] - a[0],
          dy = b[1] - a[1],
          l = Math.hypot(dx, dy);
        return [dy / l, -dx / l];
      };
      m[0] = normal(points[0], points[1]);
      m[m.length - 1] = normal(points[points.length - 2], points[points.length - 1]);
    }
    color[3] *= opacity;
    const strips = points.map((p, i) =>
      [half + 1, half, -half, -half - 1].map(
        (d) => [p[0] + m[i][0] * d, p[1] + m[i][1] * d] as Point,
      ),
    );
    for (let i = 0; i < (closed ? points.length : points.length - 1); i++) {
      const j = (i + 1) % points.length;
      for (let band = 0; band < 3; band++) {
        const a = band === 0 ? 0 : 1,
          b = band === 2 ? 0 : 1;
        this.triangle(strips[i][band], strips[j][band], strips[j][band + 1], color, a, a, b);
        this.triangle(strips[i][band], strips[j][band + 1], strips[i][band + 1], color, a, b, b);
      }
    }
  }
  flush(context: CanvasRenderingContext2D) {
    if (!this.count) return;
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertices.subarray(0, this.count));
    gl.drawArrays(gl.TRIANGLES, 0, this.count / 6);
    context.drawImage(this.canvas, 0, 0, this.width, this.height);
    this.count = 0;
  }
}

const rgba = (rgb: number[], alpha = 1) => `rgba(${rgb.join(',')},${alpha})`;
export class ViewCubeOverlay {
  private context: CanvasRenderingContext2D;
  private raster = new NativeRaster();
  private balls: Ball[] = [];
  private center: Point = [45, 45];
  private scale = 90 / 91;
  private hover: Ball | null = null;
  private originHover = false;
  private pointer: Point = [-999, -999];
  private pressed: {
    id: number;
    start: Point;
    last: Point;
    travel: number;
    origin: boolean;
  } | null = null;
  private camera: Camera | null = null;
  private observer: ResizeObserver;
  private geometryCache = new Map<string, Point[]>();
  private renderKey = '';
  constructor(
    private canvas: HTMLCanvasElement,
    private options: ViewCubeOptions,
  ) {
    const context = canvas.getContext('2d');
    if (!context) throw new Error('2D canvas unavailable');
    this.context = context;
    canvas.style.touchAction = 'none';
    canvas.setAttribute(
      'aria-label',
      'View orientation. Drag to orbit; click an axis to align; click center to switch projection.',
    );
    canvas.addEventListener('pointermove', this.move);
    canvas.addEventListener('pointerdown', this.down);
    canvas.addEventListener('pointerup', this.up);
    canvas.addEventListener('pointercancel', this.cancel);
    canvas.addEventListener('lostpointercapture', this.cancel);
    canvas.addEventListener('pointerleave', this.leave);
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(canvas);
  }
  update(camera: Camera) {
    this.camera = camera;
    this.render();
  }
  dispose() {
    this.raster.dispose();
    this.observer.disconnect();
    for (const [name, handler] of [
      ['pointermove', this.move],
      ['pointerdown', this.down],
      ['pointerup', this.up],
      ['pointercancel', this.cancel],
      ['lostpointercapture', this.cancel],
      ['pointerleave', this.leave],
    ] as const)
      this.canvas.removeEventListener(name, handler);
    this.geometryCache.clear();
  }
  private position(e: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }
  private updateHover() {
    this.hover = hitTest(this.balls, this.pointer);
    this.originHover =
      !this.hover &&
      Math.hypot(this.pointer[0] - this.center[0], this.pointer[1] - this.center[1]) <=
        6 * this.scale &&
      !hitTest(this.balls, this.center);
    this.canvas.style.cursor = this.pressed
      ? 'grabbing'
      : this.hover || this.originHover
        ? 'pointer'
        : 'default';
  }
  private move = (e: PointerEvent) => {
    this.pointer = this.position(e);
    if (this.pressed && e.pointerId === this.pressed.id) {
      const p = this.pressed;
      const travel =
        Math.abs(this.pointer[0] - p.start[0]) + Math.abs(this.pointer[1] - p.start[1]);
      p.travel = Math.max(p.travel, travel);
      if (p.travel >= 4)
        this.options.onOrbit(this.pointer[0] - p.last[0], this.pointer[1] - p.last[1]);
      p.last = this.pointer;
      e.stopPropagation();
    }
    this.updateHover();
    this.render();
  };
  private down = (e: PointerEvent) => {
    if (e.button !== 0 || this.pressed) return;
    this.pointer = this.position(e);
    this.updateHover();
    if (!this.hover && !this.originHover) return;
    this.pressed = {
      id: e.pointerId,
      start: this.pointer,
      last: this.pointer,
      travel: 0,
      origin: this.originHover,
    };
    this.canvas.setPointerCapture(e.pointerId);
    e.stopPropagation();
    e.preventDefault();
    this.render();
  };
  private up = (e: PointerEvent) => {
    const p = this.pressed;
    if (!p || e.pointerId !== p.id) return;
    this.move(e);
    if (p.travel < 4) {
      if (p.origin && this.originHover) this.options.onProjectionToggle?.();
      else if (!p.origin && this.hover) this.options.onAxis(this.hover.axis, this.hover.sign);
    }
    this.pressed = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    e.stopPropagation();
    this.render();
  };
  private cancel = () => {
    this.pressed = null;
    this.updateHover();
    this.render();
  };
  private leave = () => {
    if (!this.pressed) {
      this.pointer = [-999, -999];
      this.updateHover();
      this.render();
    }
  };
  private polygon(points: Point[], color: string) {
    this.raster.fill(points, color);
  }
  private circle(x: number, y: number, r: number, color: string, n = 24, indexed = false) {
    const points: Point[] = [];
    for (let i = 0; i < n; i++) {
      const t = (i * Math.PI * 2) / n;
      points.push([x + r * Math.cos(t), y + r * Math.sin(t)]);
    }
    this.raster.fill(points, color, indexed ? Math.min(1, 0.25 * this.scale) : 1, !indexed);
  }
  private render() {
    if (!this.camera) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    this.camera.updateMatrixWorld();
    const key = [
      rect.width,
      rect.height,
      window.devicePixelRatio,
      ...this.camera.matrixWorld.elements,
      ...this.pointer,
      Boolean(this.pressed),
      document.fonts.status,
    ].join(':');
    if (key === this.renderKey) return;
    this.renderKey = key;
    const dpr = window.devicePixelRatio || 1,
      width = Math.round(rect.width * dpr),
      height = Math.round(rect.height * dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.raster.resize(rect.width, rect.height, dpr);
    this.center = [rect.width * 0.5, rect.height * 0.5];
    this.scale = Math.min(rect.width, rect.height) / 91;
    this.camera.updateMatrixWorld();
    const e = this.camera.matrixWorld.elements;
    this.balls = viewCubeLayout(
      [e[0], e[1], e[2]],
      [e[4], e[5], e[6]],
      [-e[8], -e[9], -e[10]],
      this.center,
      this.scale,
    );
    this.updateHover();
    const ctx = this.context;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const s = this.scale;
    this.circle(...this.center, BACKDROP * s, 'rgba(22,25,28,.55)', 64);
    const core = (this.originHover ? 2.5 : 2) * s;
    this.circle(...this.center, core + 0.5 * s, 'rgb(165.75,165.75,165.75)', 16, true);
    this.circle(
      ...this.center,
      core,
      this.originHover ? 'white' : 'rgb(247.35,247.35,249.9)',
      16,
      true,
    );
    for (const b of this.balls) {
      if (b.alpha <= 0) continue;
      const rgb = COLORS[b.axis],
        hovered = this.hover === b;
      const face =
        b.sign > 0
          ? hovered
            ? rgb.map((c) => Math.min(255, c * 1.35 + 25.5))
            : rgb
          : hovered
            ? rgb
            : rgb.map((c, i) => c * 0.28 + DARK_BASE[i] * 0.72);
      if (b.sign > 0) {
        const dx = b.x - this.center[0],
          dy = b.y - this.center[1],
          distance = Math.hypot(dx, dy),
          ux = distance ? dx / distance : 1,
          uy = distance ? dy / distance : 0,
          key = `${distance}:${s}`;
        let local = this.geometryCache.get(key);
        if (!local) {
          local = lollipopPoints(Math.max(0, distance - INSET * s), b.radius, LINE * s);
          if (this.geometryCache.size >= 256)
            this.geometryCache.delete(this.geometryCache.keys().next().value!);
          this.geometryCache.set(key, local);
        }
        this.polygon(
          local.map((p) => [b.x + p[0] * ux - p[1] * uy, b.y + p[0] * uy + p[1] * ux]),
          rgba(face, b.alpha),
        );
      } else {
        this.circle(b.x, b.y, b.radius, rgba(face, b.alpha));
        const circle: Point[] = [];
        for (let i = 0; i < 24; i++) {
          const a = (i * Math.PI * 2) / 24;
          circle.push([b.x + (b.radius - 0.5) * Math.cos(a), b.y + (b.radius - 0.5) * Math.sin(a)]);
        }
        this.raster.stroke(circle, rgba(rgb, b.alpha), 1.6 * s);
      }
      this.raster.flush(ctx);
      const alpha = b.sign > 0 || hovered ? b.alpha : 1 - backAlpha(-b.depth);
      if (alpha > 0) {
        let fontSize = (14 * rect.width) / 90;
        ctx.font = `${fontSize}px MiSans`;
        ctx.fillStyle = rgba([247.35, 247.35, 249.9], alpha);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        const label = (b.sign < 0 ? '-' : '') + 'XYZ'[b.axis];
        let metrics = ctx.measureText(label);
        const inkWidth = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight;
        if (inkWidth > b.radius * 1.5) {
          fontSize *= (b.radius * 1.5) / inkWidth;
          ctx.font = `${fontSize}px MiSans`;
          metrics = ctx.measureText(label);
        }
        const baseline = (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) * 0.5,
          penX = b.x + (metrics.actualBoundingBoxLeft - metrics.actualBoundingBoxRight) * 0.5;
        ctx.fillText(label, penX, b.y + baseline);
      }
    }
  }
}
