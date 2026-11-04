import * as THREE from 'three';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import type { SceneNode } from '../contracts';
import { renderSceneCamera } from '../scene-camera';
import masters from './entity-masters.json';

const CIRCLE_SEGMENTS = 48;
export type HelperSegments = { positions: number[]; width: number };
function append(result: number[], a: THREE.Vector3, b: THREE.Vector3) {
  result.push(a.x, a.y, a.z, b.x, b.y, b.z);
}
function circle(center: THREE.Vector3, normal: THREE.Vector3, radius: number) {
  const u = new THREE.Vector3()
    .crossVectors(
      normal,
      Math.abs(normal.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0),
    )
    .normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();
  return Array.from({ length: CIRCLE_SEGMENTS }, (_, i) =>
    center
      .clone()
      .addScaledVector(u, Math.cos((i * Math.PI * 2) / CIRCLE_SEGMENTS) * radius)
      .addScaledVector(v, Math.sin((i * Math.PI * 2) / CIRCLE_SEGMENTS) * radius),
  );
}
function ring(result: number[], points: THREE.Vector3[]) {
  for (let i = 0; i < points.length; i++)
    append(result, points[i], points[(i + 1) % points.length]);
}
function worldPixels(
  camera: THREE.Camera,
  position: THREE.Vector3,
  height: number,
  pixels: number,
) {
  const view = position.clone().applyMatrix4(camera.matrixWorldInverse);
  if (view.z >= 0) return 0;
  const clip = new THREE.Vector4(view.x, view.y, view.z, 1).applyMatrix4(camera.projectionMatrix);
  return (2 * Math.abs(clip.w) * pixels) / (height * Math.abs(camera.projectionMatrix.elements[5]));
}
function cone(position: THREE.Vector3, direction: THREE.Vector3, length: number, angle: number) {
  const result: number[] = [],
    points = circle(
      position.clone().addScaledVector(direction, length),
      direction,
      length * Math.tan(angle),
    );
  ring(result, points);
  for (let i = 0; i < CIRCLE_SEGMENTS; i += 6) append(result, position, points[i]);
  return result;
}
/** Native scene_entities.py: true frustum bounds, selected-only influence and bounded spot cones. */
export function entityInfluence(
  node: SceneNode,
  editor: THREE.Camera,
  width: number,
  height: number,
): HelperSegments[] {
  const object = node.object,
    result: HelperSegments[] = [];
  for (let current: THREE.Object3D | null = object; current; current = current.parent)
    if (!current.visible) return result;
  if (!node.visible) return result;
  object.updateWorldMatrix(true, false);
  if (object instanceof THREE.PerspectiveCamera || object instanceof THREE.OrthographicCamera) {
    const camera = renderSceneCamera(node),
      points: THREE.Vector3[] = [],
      positions: number[] = [];
    for (const z of [-1, 1])
      for (const [x, y] of [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ])
        points.push(new THREE.Vector3(x, y, z).unproject(camera));
    for (const [a, b] of [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 4],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ])
      append(positions, points[a], points[b]);
    return [{ positions, width: 1.6 }];
  }
  if (!(object instanceof THREE.Light) || object.intensity <= 0) return result;
  const position = object.getWorldPosition(new THREE.Vector3());
  const direction =
    object instanceof THREE.DirectionalLight || object instanceof THREE.SpotLight
      ? object.target.getWorldPosition(new THREE.Vector3()).sub(position).normalize()
      : new THREE.Vector3(0, 0, -1).applyQuaternion(
          object.getWorldQuaternion(new THREE.Quaternion()),
        );
  if (!(object instanceof THREE.PointLight) && !(object instanceof THREE.HemisphereLight)) {
    const length = worldPixels(editor, position, height, 30),
      mask = worldPixels(editor, position, height, 6);
    if (length > 0) {
      const positions: number[] = [],
        end = position.clone().addScaledVector(direction, length),
        start = position.clone().addScaledVector(direction, mask);
      const side = direction
        .clone()
        .cross(new THREE.Vector3().setFromMatrixColumn(editor.matrixWorld, 2))
        .normalize()
        .multiplyScalar(mask * 0.6);
      append(positions, start, end);
      append(positions, end, end.clone().addScaledVector(direction, -mask).add(side));
      append(positions, end, end.clone().addScaledVector(direction, -mask).sub(side));
      result.push({ positions, width: 1.8 });
    }
  }
  if (object instanceof THREE.PointLight && object.distance > 0) {
    const positions: number[] = [];
    for (const normal of [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])
      ring(positions, circle(position, new THREE.Vector3().fromArray(normal), object.distance));
    result.push({ positions, width: 1.2 });
  } else if (object instanceof THREE.SpotLight && object.distance > 0) {
    const angle = THREE.MathUtils.clamp(
      object.angle,
      THREE.MathUtils.degToRad(0.1),
      THREE.MathUtils.degToRad(89),
    );
    let length = Math.min(
      object.distance,
      worldPixels(editor, position, height, 180),
      worldPixels(editor, position, height, 120) / Math.tan(angle),
    );
    const anchor = position.clone().project(editor),
      budget = Math.hypot(180, 120),
      near = (editor as THREE.PerspectiveCamera).near ?? 1e-6;
    for (let iteration = 0; iteration < 6 && length > 0; iteration++) {
      const positions = cone(position, direction, length, angle);
      let extent = 0;
      for (let i = 0; i < positions.length; i += 3) {
        const p = new THREE.Vector3().fromArray(positions, i),
          z = -p.clone().applyMatrix4(editor.matrixWorldInverse).z;
        if (z <= Math.max(near, 1e-6)) {
          extent = Infinity;
          break;
        }
        p.project(editor);
        extent = Math.max(
          extent,
          Math.hypot(((p.x - anchor.x) * width) / 2, ((p.y - anchor.y) * height) / 2),
        );
      }
      if (Number.isFinite(extent) && extent <= budget) {
        result.push({ positions, width: 1.4 });
        break;
      }
      length *= Number.isFinite(extent)
        ? Math.max(0.05, Math.min(0.75, (0.95 * budget) / extent))
        : 0.5;
    }
  } else if (object instanceof THREE.RectAreaLight) {
    const positions: number[] = [],
      q = object.getWorldQuaternion(new THREE.Quaternion());
    // Three's physically rendered area emitter is rectangular, so the helper uses its actual bounds.
    const points = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, y]) =>
      new THREE.Vector3((x * object.width) / 2, (y * object.height) / 2, 0)
        .applyQuaternion(q)
        .add(position),
    );
    ring(positions, points);
    result.push({ positions, width: 1.4 });
  }
  return result;
}
export class SceneEntityInfluence {
  private scene = new THREE.Scene();
  private lines: LineSegments2[] = [];
  update(
    node: SceneNode | undefined,
    camera: THREE.Camera,
    width: number,
    height: number,
    enabled: boolean,
  ) {
    const sets = enabled && node ? entityInfluence(node, camera, width, height) : [];
    for (let i = 0; i < Math.max(sets.length, this.lines.length); i++) {
      let line = this.lines[i];
      if (!line) {
        const material = new LineMaterial({
          color: new THREE.Color().setRGB(
            masters.selected_color[0],
            masters.selected_color[1],
            masters.selected_color[2],
            THREE.SRGBColorSpace,
          ),
          transparent: true,
          depthWrite: false,
          toneMapped: false,
        });
        line = new LineSegments2(new LineSegmentsGeometry(), material);
        line.frustumCulled = false;
        this.scene.add(line);
        this.lines.push(line);
      }
      line.visible = Boolean(sets[i]);
      if (!sets[i]) continue;
      const attribute = line.geometry.getAttribute('instanceStart') as
        | THREE.InterleavedBufferAttribute
        | undefined;
      if (attribute && attribute.count === sets[i].positions.length / 6) {
        (attribute.data.array as Float32Array).set(sets[i].positions);
        attribute.data.needsUpdate = true;
      } else {
        line.geometry.dispose();
        line.geometry = new LineSegmentsGeometry().setPositions(sets[i].positions);
      }
      line.material.linewidth = sets[i].width;
      line.material.resolution.set(width, height);
    }
  }
  render(renderer: THREE.WebGLRenderer, camera: THREE.Camera) {
    if (!this.lines.some((line) => line.visible)) return;
    const auto = renderer.autoClear;
    renderer.autoClear = false;
    for (const line of this.lines) {
      line.material.depthFunc = THREE.GreaterDepth;
      line.material.opacity = 0.28;
    }
    renderer.render(this.scene, camera);
    for (const line of this.lines) {
      line.material.depthFunc = THREE.LessEqualDepth;
      line.material.opacity = 1;
    }
    renderer.render(this.scene, camera);
    renderer.autoClear = auto;
  }
  dispose() {
    for (const line of this.lines) {
      line.geometry.dispose();
      line.material.dispose();
    }
  }
}
