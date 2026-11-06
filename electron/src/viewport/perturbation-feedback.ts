import * as THREE from 'three';
import { DragLinkRaster } from './drag-link';
import { NativeRaster } from './viewcube';
import { AXIS_COLORS, SIZE, arrow, axisAlpha, cornerTemplate, project, worldScale, type P } from './geometry';
import type { PerturbationGesture } from './perturbation';

/** Native perturb.py dimensions and silhouettes, using the shared shader rasterizers. */
export class PerturbationFeedback {
  private link: DragLinkRaster | null = null;
  private raster: NativeRaster | null = null;
  draw(context: CanvasRenderingContext2D, gesture: PerturbationGesture, camera: THREE.Camera,
    width: number, height: number, dpr: number): void {
    const feedback = gesture.feedback;
    if (!feedback?.body) return;
    const { body, target, bounds } = feedback;
    const center = new THREE.Vector3().fromArray(body.position), rotation = new THREE.Quaternion().fromArray(target.rotation);
    const screen = (p: THREE.Vector3) => project(p, camera, width, height);
    const front = (p: THREE.Vector3) => p.clone().applyMatrix4(camera.matrixWorldInverse).z < 0;
    if (target.mode === 'translation') {
      const start = new THREE.Vector3().fromArray(target.localPosition)
        .applyQuaternion(new THREE.Quaternion().fromArray(body.rotation)).add(center);
      const end = new THREE.Vector3().fromArray(target.localPosition).applyQuaternion(rotation)
        .add(new THREE.Vector3().fromArray(target.position));
      if (front(start) && front(end)) {
        this.link ??= new DragLinkRaster();
        this.link.draw(context, screen(start), screen(end), width, height, dpr, 6);
      }
      return;
    }
    if (!front(center)) return;
    const length = worldScale(center, camera, height), pixel = length / SIZE;
    const half = bounds.isEmpty() ? new THREE.Vector3().setScalar(length / 2)
      : bounds.getSize(new THREE.Vector3()).multiplyScalar(0.5).addScalar(6 * pixel).max(new THREE.Vector3().setScalar(2 * pixel));
    const origin = bounds.isEmpty() ? center : bounds.getCenter(new THREE.Vector3()).applyQuaternion(rotation).add(center);
    const corners: { world: THREE.Vector3; point: P }[] = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      const world = new THREE.Vector3(x, y, z).multiply(half).applyQuaternion(rotation).add(origin);
      if (!front(world)) return;
      corners.push({ world, point: screen(world) });
    }
    // The convex hull of the projected box is its silhouette, including edge-on views.
    const sorted = corners.sort((a, b) => a.point[0] - b.point[0] || a.point[1] - b.point[1]);
    const cross = (a: P, b: P, c: P) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const chain = (items: typeof corners) => {
      const hull: typeof corners = [];
      for (const item of items) {
        while (hull.length > 1 && cross(hull.at(-2)!.point, hull.at(-1)!.point, item.point) <= 0) hull.pop();
        hull.push(item);
      }
      return hull.slice(0, -1);
    };
    const hull = [...chain(sorted), ...chain([...sorted].reverse())];
    // Apply the same canonical G3 turn in each world-space corner plane before projection.
    const path = hull.flatMap((item, i) => {
      const previous = hull[(i + hull.length - 1) % hull.length], next = hull[(i + 1) % hull.length];
      const a = Math.hypot(item.point[0] - previous.point[0], item.point[1] - previous.point[1]);
      const b = Math.hypot(item.point[0] - next.point[0], item.point[1] - next.point[1]);
      if (Math.min(a, b) < 1e-5) return [item.point];
      const trim = Math.min(4, 0.4 * a, 0.4 * b);
      const before = previous.world.clone().sub(item.world).multiplyScalar(trim / a), after = next.world.clone().sub(item.world).multiplyScalar(trim / b);
      return cornerTemplate.map(([x, y]) => screen(item.world.clone().addScaledVector(before, 1 - x).addScaledVector(after, y)));
    });
    this.raster ??= new NativeRaster();
    this.raster.resize(width, height, dpr);
    this.raster.stroke(path, '#adb5c2', 3.5, 1, true);
    this.raster.stroke(path, '#fafafa', 2, 1, true);
    const forward = camera.getWorldDirection(new THREE.Vector3()), c = screen(center);
    const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]
      .map((axis, index) => ({ axis: axis.applyQuaternion(rotation), index }));
    axes.sort((a, b) => b.axis.dot(forward) - a.axis.dot(forward));
    for (const { axis, index } of axes) {
      const end = screen(center.clone().addScaledVector(axis, length));
      const dx = end[0] - c[0], dy = end[1] - c[1], distance = Math.hypot(dx, dy);
      const start: P = distance <= SIZE * 0.11 ? end : [c[0] + dx / distance * SIZE * 0.11, c[1] + dy / distance * SIZE * 0.11];
      this.raster.fill(arrow(start, end), AXIS_COLORS[index], 1, true, axisAlpha(axis.dot(forward)));
    }
    const circle = (radius: number): P[] => Array.from({ length: 40 }, (_, i) => [c[0] + radius * Math.cos(i * Math.PI / 20), c[1] + radius * Math.sin(i * Math.PI / 20)]);
    this.raster.fill(circle(SIZE * 0.075 + 0.75), '#adb5c2');
    this.raster.fill(circle(SIZE * 0.075), '#ebebeb');
    this.raster.flush(context);
  }
  dispose(): void { this.link?.dispose(); this.raster?.dispose(); }
}
