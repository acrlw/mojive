/** Scalar-joint projection and pointer ownership ported from python/ui/gizmo/joint_ranges.py. */
import * as THREE from 'three';
import type { Joint } from '../contracts';
import {
  type P,
  axisAlpha,
  ringAlpha,
  project,
  pathDistance,
  segmentDistance,
  basisForAxis,
  arrow,
} from './geometry';
export interface JointTick {
  kind: 'lower' | 'upper' | 'current';
  value: number;
  start: P;
  end: P;
}
export interface PrecisionRail {
  start: P;
  end: P;
  current: P;
  source: P;
  rect: [number, number, number, number];
  lower: number;
  upper: number;
}
export interface JointProjection {
  id: string;
  slide: boolean;
  alpha: number;
  allowed: P[];
  complement: P[];
  full: boolean;
  ticks: JointTick[];
  current: P;
  lower: P | null;
  upper: P | null;
  arrows: P[][];
  hitArrows: P[][];
  rail: PrecisionRail | null;
}
const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
export function precisionRail(
  lower: P,
  upper: P,
  current: number,
  minimum: number,
  maximum: number,
  width: number,
  height: number,
  slide: boolean,
): PrecisionRail | null {
  if (
    (!slide && maximum - minimum > Math.PI / 4) ||
    Math.hypot(upper[0] - lower[0], upper[1] - lower[1]) >= 30
  )
    return null;
  const track = Math.min(128, width - 40);
  if (track < 60 || height < 50) return null;
  const source: P = [(lower[0] + upper[0]) * 0.5, (lower[1] + upper[1]) * 0.5],
    half = (track + 20) * 0.5,
    x = clamp(source[0], 10 + half, width - 10 - half),
    above = source[1] - 44,
    below = source[1] + 44,
    y =
      above - 15 >= 10 ? above : below + 15 <= height - 10 ? below : clamp(above, 25, height - 25),
    start: P = [x - track * 0.5, y],
    end: P = [x + track * 0.5, y],
    t = clamp((current - minimum) / Math.max(maximum - minimum, 1e-12), 0, 1);
  return {
    source,
    start,
    end,
    current: [start[0] + track * t, y],
    rect: [x - half, y - 15, x + half, y + 15],
    lower: minimum,
    upper: maximum,
  };
}
export function projectJoint(
  joint: Joint,
  origin: THREE.Vector3,
  axis: THREE.Vector3,
  scale: number,
  camera: THREE.Camera,
  width: number,
  height: number,
): JointProjection | null {
  const slide = joint.type === 'prismatic',
    center = slide ? origin.clone().addScaledVector(axis, joint.value) : origin,
    blend = Number(
      camera.userData.projectionBlend ?? (camera instanceof THREE.OrthographicCamera ? 1 : 0),
    ),
    view = center
      .clone()
      .sub(camera.position)
      .normalize()
      .multiplyScalar(1 - blend)
      .addScaledVector(camera.getWorldDirection(new THREE.Vector3()), blend)
      .normalize(),
    alpha = slide ? axisAlpha(axis.dot(view)) : ringAlpha(axis.dot(view));
  if (alpha <= 0) return null;
  const finite = Number.isFinite(joint.lower) && Number.isFinite(joint.upper),
    lower = finite ? joint.lower : 0,
    upper = finite ? joint.upper : Math.PI * 2;
  const output: JointProjection = {
    id: joint.id,
    slide,
    alpha,
    allowed: [],
    complement: [],
    full: false,
    ticks: [],
    current: [0, 0],
    lower: null,
    upper: null,
    arrows: [],
    hitArrows: [],
    rail: null,
  };
  const screen = (v: THREE.Vector3) => project(v, camera, width, height);
  if (slide) {
    if (!finite) return null;
    const a = screen(origin.clone().addScaledVector(axis, lower)),
      b = screen(origin.clone().addScaledVector(axis, upper)),
      current = screen(origin.clone().addScaledVector(axis, joint.value)),
      dx = b[0] - a[0],
      dy = b[1] - a[1],
      length = Math.hypot(dx, dy);
    if (length < 1e-6) return null;
    const tangent: P = [dx / length, dy / length],
      normal: P = [-tangent[1], tangent[0]],
      tick = (p: P, size: number, kind: JointTick['kind'], value: number): JointTick => ({
        kind,
        value,
        start: [p[0] - normal[0] * size * 0.5, p[1] - normal[1] * size * 0.5],
        end: [p[0] + normal[0] * size * 0.5, p[1] + normal[1] * size * 0.5],
      });
    output.allowed = [a, b];
    output.current = current;
    output.lower = a;
    output.upper = b;
    output.ticks = [
      tick(current, 20, 'current', joint.value),
      tick(a, 12, 'lower', lower),
      tick(b, 12, 'upper', upper),
    ];
    output.arrows = [1, -1].map((sign) => {
      const start: P = [
          current[0] + normal[0] * 17 + tangent[0] * 9 * sign,
          current[1] + normal[1] * 17 + tangent[1] * 9 * sign,
        ],
        end: P = [
          current[0] + normal[0] * 17 + tangent[0] * 48 * sign,
          current[1] + normal[1] * 17 + tangent[1] * 48 * sign,
        ],
        center: P = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
      output.hitArrows.push(arrow(start, end));
      return arrow(start, end, true).map(
        (p) => [center[0] + (p[0] - center[0]) * 0.8, center[1] + (p[1] - center[1]) * 0.8] as P,
      );
    });
  } else {
    const radial = basisForAxis(axis)[0],
      tangent = new THREE.Vector3().crossVectors(axis, radial),
      span = clamp(upper - lower, 0, Math.PI * 2),
      segments = Math.ceil(
        64 * Math.min(4, 1 / Math.max(Math.sqrt(Math.abs(axis.dot(view))), 0.25)),
      ),
      full = span >= Math.PI * 2 - 1e-9;
    const point = (angle: number, radius = 0.78) =>
      screen(
        origin
          .clone()
          .addScaledVector(radial, Math.cos(angle) * radius * scale)
          .addScaledVector(tangent, Math.sin(angle) * radius * scale),
      );
    const arc = (lo: number, hi: number, count: number, endpoint = true) =>
      Array.from({ length: count }, (_, i) =>
        point(lo + ((hi - lo) * i) / (endpoint ? count - 1 : count)),
      );
    output.allowed =
      span > 1e-6
        ? arc(
            lower,
            lower + span,
            full ? segments : Math.max(2, Math.ceil((segments * span) / (Math.PI * 2)) + 1),
            !full,
          )
        : [];
    output.complement = full
      ? []
      : arc(
          lower + span,
          lower + Math.PI * 2,
          Math.max(2, Math.ceil((segments * (Math.PI * 2 - span)) / (Math.PI * 2)) + 1),
        );
    output.full = full;
    output.current = point(joint.value);
    const tick = (value: number, size: number, kind: JointTick['kind']) => {
      const start = point(value),
        out = point(value, 0.78 + 1 / 88),
        dx = out[0] - start[0],
        dy = out[1] - start[1],
        length = Math.hypot(dx, dy);
      return {
        kind,
        value,
        start,
        end: [start[0] + (dx / length) * size, start[1] + (dy / length) * size] as P,
      };
    };
    output.ticks = [tick(joint.value, 20, 'current')];
    if (!full && finite) {
      output.lower = point(lower);
      output.upper = point(upper);
      output.ticks.push(tick(lower, 14, 'lower'), tick(upper, 14, 'upper'));
    }
  }
  if (output.lower && output.upper)
    output.rail = precisionRail(
      output.lower,
      output.upper,
      joint.value,
      lower,
      upper,
      width,
      height,
      slide,
    );
  return output;
}
export function closestLimit(p: P, projection: JointProjection): JointTick | null {
  const candidates: { distance: number; endDistance: number; tick: JointTick }[] = [];
  for (const tick of projection.ticks) {
    if (tick.kind === 'current') continue;
    const dx = tick.end[0] - tick.start[0],
      dy = tick.end[1] - tick.start[1],
      den = dx * dx + dy * dy;
    if (den <= 1e-12) continue;
    let t = ((p[0] - tick.start[0]) * dx + (p[1] - tick.start[1]) * dy) / den;
    if ((!projection.slide && t < 0.3) || (projection.slide && Math.abs(t - 0.5) < 0.2)) continue;
    t = clamp(t, 0, 1);
    const distance = Math.hypot(p[0] - tick.start[0] - dx * t, p[1] - tick.start[1] - dy * t);
    if (distance <= 3.5 * 0.5 + 3)
      candidates.push({
        distance,
        endDistance: Math.hypot(p[0] - tick.end[0], p[1] - tick.end[1]),
        tick,
      });
  }
  candidates.sort(
    (a, b) =>
      a.distance - b.distance ||
      a.endDistance - b.endDistance ||
      a.tick.kind.localeCompare(b.tick.kind),
  );
  return candidates[0]?.tick ?? null;
}
export function jointHit(
  p: P,
  projection: JointProjection,
  axisHit = false,
): { hit: boolean; limit: JointTick | null } {
  const range =
    pathDistance(p, projection.allowed, projection.full) <= (projection.slide ? 7 : 5.5) ||
    pathDistance(p, projection.complement) <= 5.5;
  const current = projection.ticks.find((t) => t.kind === 'current'),
    currentHit = current
      ? segmentDistance(p, current.start, current.end) <= (projection.slide ? 7 : 5.5)
      : false;
  const limit = closestLimit(p, projection),
    limitOwns = !!limit && (projection.slide || !(range || axisHit)),
    arrowHit = projection.hitArrows.some(
      (path) => pathDistance(p, path, true) <= 4 || pointInPolygon(p, path),
    );
  return {
    hit: range || currentHit || !!limit || arrowHit || axisHit,
    limit: limitOwns ? limit : null,
  };
}
function pointInPolygon(p: P, path: P[]) {
  let inside = false;
  for (let i = 0, j = path.length - 1; i < path.length; j = i++) {
    const a = path[i],
      b = path[j];
    if (
      a[1] > p[1] !== b[1] > p[1] &&
      p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]
    )
      inside = !inside;
  }
  return inside;
}
export function inRail(p: P, rail: PrecisionRail | null) {
  return (
    !!rail &&
    p[0] >= rail.rect[0] &&
    p[0] <= rail.rect[2] &&
    p[1] >= rail.rect[1] &&
    p[1] <= rail.rect[3]
  );
}
/** Dwell timing is independent of animation/frame rate and remains deterministic in tests. */
export class JointDwell {
  private key: string | null = null;
  private started = 0;
  private lastSeen = 0;
  private visibleUntil = 0;
  reset() {
    this.key = null;
    this.visibleUntil = 0;
  }
  visible(now: number) {
    return now <= this.visibleUntil;
  }
  update(
    id: string | null,
    eligible: boolean,
    hovered: boolean,
    railHovered: boolean,
    active: boolean,
    now: number,
  ) {
    if (!eligible || !id) {
      this.reset();
      return false;
    }
    if (railHovered || active) this.visibleUntil = now + 0.3;
    if (hovered || railHovered) {
      if (this.key !== id) {
        this.key = id;
        this.started = now;
      }
      this.lastSeen = now;
    } else if (this.key !== id || now - this.lastSeen > 0.12) {
      this.key = null;
      return this.visible(now);
    }
    if (now - this.started >= 0.5) this.visibleUntil = now + 0.3;
    return this.visible(now) || active;
  }
}
