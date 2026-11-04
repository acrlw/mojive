import * as THREE from 'three';
import type { SceneDocument } from '../contracts';
import type { TransitionCameraView } from './camera-transition';
import { sceneFocusBounds, sceneNodeBounds } from './focus-bounds.ts';
import { basisForAxis } from './geometry.ts';

const Z = new THREE.Vector3(0, 0, 1);
const radians = THREE.MathUtils.degToRad;
const clamp = THREE.MathUtils.clamp;
const valid = (v: THREE.Vector3) => v.toArray().every(Number.isFinite) && v.length() > 1e-9;
const normalized = (v: THREE.Vector3, fallback: THREE.Vector3) =>
  (valid(v) ? v : fallback).clone().normalize();

/** Native camera.py focus direction helpers; evaluated only on a focus request. */
export function elevatedFocusDirection(
  offset: THREE.Vector3,
  fallback: THREE.Vector3,
  minimumElevation = 30,
): THREE.Vector3 {
  const direction = normalized(offset, fallback),
    elevation = radians(clamp(minimumElevation, 0, 89.9)),
    minimumZ = Math.sin(elevation);
  if (direction.z >= minimumZ) return direction;
  let horizontal = direction.clone().setZ(0);
  if (!valid(horizontal)) horizontal = fallback.clone().setZ(0);
  if (!valid(horizontal)) horizontal.set(1, 0, 0);
  return horizontal.normalize().multiplyScalar(Math.cos(elevation)).setZ(minimumZ);
}

export function perpendicularFocusDirection(
  axis: THREE.Vector3,
  offset: THREE.Vector3,
  fallback: THREE.Vector3,
): THREE.Vector3 {
  const normal = normalized(axis, Z);
  for (const candidate of [offset, fallback]) {
    const projected = candidate.clone().addScaledVector(normal, -candidate.dot(normal));
    if (valid(projected)) return projected.normalize();
  }
  const coordinates = normal.toArray().map(Math.abs),
    index = coordinates.indexOf(Math.min(...coordinates)),
    reference = new THREE.Vector3().setComponent(index, 1);
  return reference.addScaledVector(normal, -reference.dot(normal)).normalize();
}

export function obliqueFocusDirections(
  axis: THREE.Vector3,
  offset: THREE.Vector3,
  fallback: THREE.Vector3,
  angle = 35,
  markers: THREE.Vector3[] = [],
): THREE.Vector3[] {
  const minimum = clamp(angle, 0, 45),
    maximum = 90 - minimum,
    current = normalized(offset, fallback),
    elevated = elevatedFocusDirection(current, fallback, minimum),
    normal = normalized(axis, Z),
    pitch = THREE.MathUtils.radToDeg(Math.asin(clamp(elevated.z, -1, 1))),
    yaw = Math.atan2(elevated.y, elevated.x),
    readable = (direction: THREE.Vector3) => {
      const angle = THREE.MathUtils.radToDeg(
        Math.acos(clamp(Math.abs(direction.dot(normal)), -1, 1)),
      );
      return angle >= minimum - 1e-6 && angle <= maximum + 1e-6;
    },
    score = (direction: THREE.Vector3) => ({ turn: 1 - direction.dot(current), direction });
  let scored: { turn: number; direction: THREE.Vector3 }[] = [];
  if (readable(elevated)) scored.push(score(elevated));
  const pitches = [pitch];
  for (let value = minimum; value < 89.9 + 2.5; value += 5)
    if (Math.abs(value - pitch) > 1e-6) pitches.push(value);
  for (const p of pitches) {
    const pitch = radians(clamp(p, minimum, 89.9));
    for (let offset = 0; offset < 360; offset += 5) {
      const a = yaw + radians(offset),
        direction = new THREE.Vector3(
          Math.cos(pitch) * Math.cos(a),
          Math.cos(pitch) * Math.sin(a),
          Math.sin(pitch),
        );
      if (readable(direction)) scored.push(score(direction));
    }
  }
  if (!scored.length) return [elevated];
  scored.sort((a, b) => a.turn - b.turn);
  const planarMarkers = markers
    .filter((marker) => marker.toArray().every(Number.isFinite))
    .map((marker) => marker.clone().addScaledVector(normal, -marker.dot(normal)))
    .filter(valid)
    .map((marker) => marker.normalize());
  if (planarMarkers.length) {
    const clearance = scored.map(({ direction }) => {
        const perpendicular = new THREE.Vector3().crossVectors(normal, direction).normalize();
        return Math.min(...planarMarkers.map((marker) => Math.abs(perpendicular.dot(marker))));
      }),
      nearestAngle = Math.acos(clamp(1 - scored[0].turn, -1, 1)),
      nearby = scored.map(
        ({ turn }) => Math.acos(clamp(1 - turn, -1, 1)) <= nearestAngle + radians(25) + 1e-9,
      );
    let filtered = scored.filter((_, index) => nearby[index] && clearance[index] >= 0.2 - 1e-6);
    if (!filtered.length) {
      const maximum = Math.max(...clearance);
      filtered = scored.filter((_, index) => nearby[index] && clearance[index] >= maximum - 1e-6);
      if (!filtered.length) filtered = scored.slice(0, 1);
    }
    scored = filtered;
  }
  const candidates: THREE.Vector3[] = [],
    separation = Math.cos(radians(10));
  for (const { direction } of scored) {
    if (candidates.some((candidate) => direction.dot(candidate) > separation)) continue;
    candidates.push(direction);
    if (candidates.length >= 9) break;
  }
  return candidates;
}

export interface FocusRayHit {
  distance: number;
  target: boolean;
}
/** Native elevation/nearby-turn filtering, then lexicographic obstruction scoring. */
export function leastOccludedFocusDirection(
  candidates: THREE.Vector3[],
  offset: THREE.Vector3,
  fallback: THREE.Vector3,
  radius: number,
  sceneSpan: number,
  probe?: (direction: THREE.Vector3, distance: number) => FocusRayHit | null,
): THREE.Vector3 {
  let directions: THREE.Vector3[] = [];
  for (const value of candidates) {
    if (!valid(value)) continue;
    const direction = value.clone().normalize();
    if (!directions.some((known) => direction.dot(known) > 1 - 1e-7)) directions.push(direction);
  }
  if (!directions.length) return fallback.clone();
  const current = valid(offset) ? offset.clone().normalize() : directions[0],
    preferred = directions.filter((direction) => direction.z >= Math.sin(radians(30)) - 1e-6),
    above = directions.filter((direction) => direction.z >= -1e-6);
  if (preferred.length) directions = preferred;
  else if (above.length) directions = above;
  const turn = (direction: THREE.Vector3) => 1 - clamp(direction.dot(current), -1, 1),
    angles = directions.map((direction) => Math.acos(clamp(direction.dot(current), -1, 1))),
    limit = Math.min(...angles) + radians(25);
  directions = directions.filter((_, index) => angles[index] <= limit + 1e-9);
  if (!probe) return directions.reduce((best, value) => (turn(value) < turn(best) ? value : best));
  const distance = Math.max(offset.length(), sceneSpan * 0.75, radius * 8, 1);
  let best = directions[0],
    bestScore = [2, 0, 0];
  for (const direction of directions) {
    const hit = probe(direction, distance),
      blocked = Boolean(
        hit && !hit.target && Number.isFinite(hit.distance) && hit.distance < distance,
      ),
      score = [blocked ? 1 : 0, blocked ? -clamp(hit!.distance, 0, distance) : 0, turn(direction)];
    if (
      score[0] < bestScore[0] ||
      (score[0] === bestScore[0] &&
        (score[1] < bestScore[1] || (score[1] === bestScore[1] && score[2] < bestScore[2])))
    ) {
      best = direction;
      bestScore = score;
    }
  }
  return best;
}

/** Resolve the native joint anchor, extent, dial clearance and one-shot obstruction test. */
export function jointFocusPlan(doc: SceneDocument, id: string, view: TransitionCameraView) {
  const joint = doc.joints.find((joint) => joint.id === id);
  if (!joint) return null;
  doc.root.updateWorldMatrix(true, true);
  const body = doc.nodes.find((node) =>
      joint.bodyId ? node.id === joint.bodyId : node.kind === 'link' && node.name === joint.child,
    ),
    world = new THREE.Matrix4().multiplyMatrices(
      joint.object.parent?.matrixWorld ?? new THREE.Matrix4(),
      joint.origin,
    ),
    rotation = new THREE.Quaternion().setFromRotationMatrix(world),
    axis = normalized(joint.axis.clone().applyQuaternion(rotation), Z),
    slide = joint.type === 'slide' || joint.type === 'prismatic',
    free = joint.type === 'free' || joint.type === 'floating',
    hinge = ['hinge', 'revolute', 'continuous'].includes(joint.type),
    center =
      slide || free
        ? (body?.object ?? joint.object).getWorldPosition(new THREE.Vector3())
        : new THREE.Vector3().setFromMatrixPosition(world),
    limited =
      Number.isFinite(joint.lower) && Number.isFinite(joint.upper) && joint.upper > joint.lower;
  let halfSpan = 0;
  if (slide && limited && Number.isFinite(joint.value)) {
    center.addScaledVector(axis, (joint.lower + joint.upper) * 0.5 - joint.value);
    halfSpan = (joint.upper - joint.lower) * 0.5;
  }
  const sceneBounds = sceneFocusBounds(doc),
    sceneSpan = Math.hypot(...sceneBounds.half) * 2,
    offset = new THREE.Vector3().fromArray(view.position).sub(center),
    backward = new THREE.Vector3()
      .fromArray(view.position)
      .sub(new THREE.Vector3().fromArray(view.target))
      .normalize(),
    right = new THREE.Vector3()
      .crossVectors(backward.clone().negate(), new THREE.Vector3().fromArray(view.up))
      .normalize();
  let radius = 0;
  for (const bounds of [sceneNodeBounds(doc, id, 'local'), sceneNodeBounds(doc, id)]) {
    if (!bounds) continue;
    const largest = Math.max(...bounds.half.map(Math.abs));
    if (Number.isFinite(largest) && largest > 1e-6) {
      radius = Math.max(largest, 1e-4);
      break;
    }
  }
  if (!radius)
    radius = Math.max(sceneSpan > 1e-6 ? sceneSpan * 0.025 : offset.length() * 0.04, 1e-4);
  radius += halfSpan;
  let candidates: THREE.Vector3[];
  if (hinge) {
    const markers: THREE.Vector3[] = [];
    if (limited && Number.isFinite(joint.value)) {
      const angles = [joint.value],
        [u, v] = basisForAxis(axis);
      if (joint.upper - joint.lower < 2 * Math.PI - 1e-9) angles.push(joint.lower, joint.upper);
      for (const angle of angles)
        markers.push(u.clone().multiplyScalar(Math.cos(angle)).addScaledVector(v, Math.sin(angle)));
    }
    candidates = obliqueFocusDirections(axis, offset, right, 35, markers);
  } else if (slide) {
    const nearest = perpendicularFocusDirection(axis, offset, right),
      tangent = new THREE.Vector3().crossVectors(axis, nearest);
    candidates = [
      nearest,
      nearest.clone().negate(),
      tangent,
      tangent.clone().negate(),
      elevatedFocusDirection(offset, right),
    ];
  } else candidates = [elevatedFocusDirection(offset, right)];
  const ray = new THREE.Raycaster(),
    nodes = new Map(doc.nodes.map((node) => [node.id, node])),
    bodyFor = (object: THREE.Object3D) => {
      let node = nodes.get(object.userData.nodeId);
      while (node && node.kind !== 'link' && node.parentId) node = nodes.get(node.parentId);
      return node?.id;
    },
    direction = leastOccludedFocusDirection(
      candidates,
      offset,
      backward,
      radius,
      sceneSpan,
      (direction, distance) => {
        ray.set(center.clone().addScaledVector(direction, distance), direction.clone().negate());
        for (const hit of ray.intersectObject(doc.root, true)) {
          if (!(hit.object instanceof THREE.Mesh)) continue;
          let visible = true;
          for (let object: THREE.Object3D | null = hit.object; object; object = object.parent)
            if (!object.visible) {
              visible = false;
              break;
            }
          if (!visible) continue;
          return { distance: hit.distance, target: bodyFor(hit.object) === body?.id };
        }
        return null;
      },
    );
  return {
    center: center.toArray(),
    half: [radius, radius, radius],
    radius,
    direction: direction.toArray(),
  };
}
