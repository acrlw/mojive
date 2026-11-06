import * as THREE from 'three';
import type { SceneDocument } from '../contracts';
import { sceneCommands, type PerturbationCommands, type PerturbationTarget } from '../scene-commands';
import type { P } from './geometry';

/** Pointer ownership and target geometry; no engine objects or simulation scheduling. */
export class PerturbationGesture {
  private port: PerturbationCommands | null = null;
  private target: PerturbationTarget | null = null;
  private pointer = -1;
  private button = -1;
  private camera = new THREE.PerspectiveCamera();
  private plane = new THREE.Plane();
  private ray = new THREE.Raycaster();
  private cursor = new THREE.Vector2();
  private grab = new THREE.Vector3();
  private start = new THREE.Vector3();
  private hit = new THREE.Vector3();
  private delta = new THREE.Vector3();
  private right = new THREE.Vector3();
  private up = new THREE.Vector3();
  private rotation = new THREE.Quaternion();
  private increment = new THREE.Quaternion();
  private last: P = [0, 0];
  private width = 1;
  private height = 1;
  private bounds = new THREE.Box3();

  get active(): boolean { return this.target !== null; }
  get feedback() { return this.target ? { target: this.target, body: this.port?.bodyAt(this.target.bodyId), bounds: this.bounds } : null; }
  owns(pointer: number): boolean { return this.active && this.pointer === pointer; }
  releases(pointer: number, button: number): boolean { return this.owns(pointer) && this.button === button; }

  begin(document: SceneDocument, id: string | null, camera: THREE.Camera, point: P,
    width: number, height: number, pointer: number, button: number, onBegin: () => void,
    surfaceHit?: { id: string; point: THREE.Vector3 }): boolean {
    if (this.active || !id || (button !== 0 && button !== 2)) return false;
    const port = sceneCommands(document)?.perturbation, body = port?.bodyAt(id);
    if (!port || !body) return false;
    // Copy the actual render projection, including orthographic transition matrices.
    this.camera.matrixWorld.copy(camera.matrixWorld);
    this.camera.matrixWorldInverse.copy(camera.matrixWorldInverse);
    this.camera.projectionMatrix.copy(camera.projectionMatrix);
    this.camera.projectionMatrixInverse.copy(camera.projectionMatrixInverse);
    this.width = width; this.height = height;
    this.start.fromArray(body.position);
    this.rotation.fromArray(body.rotation);
    this.plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(this.delta), this.start);
    if (button === 2) this.grab.copy(this.start);
    else if (surfaceHit && port.bodyAt(surfaceHit.id)?.id === body.id) {
      // Native translation grabs the selected body's surface when it is under the cursor.
      this.grab.copy(surfaceHit.point);
      this.plane.setFromNormalAndCoplanarPoint(this.delta, this.grab);
    } else if (!this.pointOnPlane(point, this.grab)) return false;
    this.right.setFromMatrixColumn(camera.matrixWorld, 0);
    this.up.setFromMatrixColumn(camera.matrixWorld, 1);
    const local = this.grab.clone().sub(this.start).applyQuaternion(this.rotation.clone().invert());
    this.bounds.makeEmpty();
    const inverse = new THREE.Matrix4().compose(this.start, this.rotation, new THREE.Vector3(1, 1, 1)).invert();
    document.root.updateMatrixWorld(true);
    for (const node of document.nodes) {
      if (!(node.object instanceof THREE.Mesh) || port.bodyAt(node.id)?.id !== body.id) continue;
      node.object.geometry.computeBoundingBox();
      if (node.object.geometry.boundingBox) this.bounds.union(node.object.geometry.boundingBox.clone()
        .applyMatrix4(new THREE.Matrix4().multiplyMatrices(inverse, node.object.matrixWorld)));
    }
    onBegin();
    this.port = port; this.pointer = pointer; this.button = button; this.last = [...point];
    this.target = { bodyId: body.id, position: [...body.position], rotation: [...body.rotation],
      localPosition: local.toArray(), mode: button === 0 ? 'translation' : 'rotation', strength: 1 };
    port.update(structuredClone(this.target));
    return true;
  }
  private pointOnPlane(point: P, out: THREE.Vector3): boolean {
    // Unproject both clip planes rather than relying on a Camera subclass tag.
    this.cursor.set(point[0] / this.width * 2 - 1, 1 - point[1] / this.height * 2);
    this.ray.ray.origin.set(this.cursor.x, this.cursor.y, -1).unproject(this.camera);
    this.ray.ray.direction.set(this.cursor.x, this.cursor.y, 1).unproject(this.camera)
      .sub(this.ray.ray.origin).normalize();
    return this.ray.ray.intersectPlane(this.plane, out) !== null;
  }
  move(point: P): void {
    if (!this.target || !this.port) return;
    if (this.target.mode === 'translation') {
      if (!this.pointOnPlane(point, this.hit)) return;
      this.target.position = this.hit.sub(this.grab).add(this.start).toArray();
    } else {
      this.delta.copy(this.up).multiplyScalar(point[0] - this.last[0])
        .addScaledVector(this.right, point[1] - this.last[1]).multiplyScalar(Math.PI / 450);
      const angle = this.delta.length();
      if (angle) this.rotation.premultiply(this.increment.setFromAxisAngle(this.delta.divideScalar(angle), angle)).normalize();
      this.target.rotation = this.rotation.toArray();
    }
    this.last = [...point];
    this.port.update(structuredClone(this.target));
  }
  end(): boolean {
    if (!this.active) return false;
    this.port?.clear();
    this.port = null; this.target = null; this.pointer = -1; this.button = -1;
    return true;
  }
}
