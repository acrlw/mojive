import type { MainModule, MjModel, MjData, MjvPerturb } from '@mujoco/mujoco';
import type { PerturbationTarget, PhysicsSource } from '../../contracts';

interface HeldTarget { target: PerturbationTarget; mass: number }
/** Reuse MuJoCo's critically damped perturbation spring, including point mobility. */
export class MujocoPerturbation {
  private perturb: MjvPerturb;
  private held: HeldTarget | null = null;
  private body = -1;
  private mj: MainModule;
  private model: MjModel;
  private data: MjData;
  private source: PhysicsSource;
  constructor(mj: MainModule, model: MjModel, data: MjData, source: PhysicsSource) {
    this.mj = mj; this.model = model; this.data = data; this.source = source;
    this.perturb = new mj.MjvPerturb();
  }
  capture(): HeldTarget | undefined { return this.held ? structuredClone(this.held) : undefined; }
  private validate(target: PerturbationTarget): number {
    const body = this.source.bodies.findIndex((item) => item.id === target?.bodyId);
    if (body <= 0 || !this.source.bodies[body].dynamic) throw new Error('Perturbation requires a dynamic body.');
    if (!['translation', 'rotation'].includes(target.mode) ||
      ![target.position, target.localPosition].every((v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)) ||
      !Array.isArray(target.rotation) || target.rotation.length !== 4 || !target.rotation.every(Number.isFinite) ||
      Math.abs(Math.hypot(...target.rotation) - 1) > 1e-6 ||
      !Number.isFinite(target.strength) || target.strength < 0 || target.strength > 20)
      throw new Error('Invalid perturbation target.');
    return body;
  }
  set(target: PerturbationTarget): void {
    const body = this.validate(target);
    const changed = body !== this.body || target.localPosition.some((v, i) => v !== this.held?.target.localPosition[i]);
    if (changed) {
      this.clear();
      this.perturb.select = body;
      this.perturb.localpos.set(target.localPosition);
      // initPerturb also computes a screen scale. A valid temporary camera is
      // required by that API; pointer projection itself belongs to the viewer.
      const scene = new this.mj.MjvScene(this.model, 0), camera = new this.mj.MjvCamera();
      try {
        this.mj.mjv_defaultFreeCamera(this.model, camera);
        this.mj.mjv_updateCamera(this.model, this.data, camera, scene);
        this.mj.mjv_initPerturb(this.model, this.data, scene, this.perturb);
      } finally { camera.delete(); scene.delete(); }
    }
    this.body = body;
    this.held = { target: structuredClone(target), mass: this.perturb.localmass };
    this.apply();
  }
  restore(value: unknown): void {
    if (value === undefined) { this.clear(); return; }
    const held = value as HeldTarget;
    const body = this.validate(held?.target);
    if (!Number.isFinite(held.mass) || held.mass <= 0) throw new Error('Invalid perturbation mass.');
    this.clear();
    this.body = body;
    this.held = structuredClone(held);
    this.perturb.select = body;
    this.perturb.localmass = held.mass;
    this.perturb.localpos.set(held.target.localPosition);
  }
  apply(): void {
    if (!this.held) return;
    const { target } = this.held, p = this.perturb, q = target.rotation;
    p.active = target.mode === 'translation' ? this.mj.mjtPertBit.mjPERT_TRANSLATE.value : this.mj.mjtPertBit.mjPERT_ROTATE.value;
    p.active2 = 0;
    if (target.mode === 'translation') {
      const [x, y, z] = target.localPosition, [qx, qy, qz, qw] = q;
      const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
      p.refselpos.set([target.position[0] + x + qw * tx + qy * tz - qz * ty,
        target.position[1] + y + qw * ty + qz * tx - qx * tz,
        target.position[2] + z + qw * tz + qx * ty - qy * tx]);
    } else {
      const i = this.body * 4, b = this.model.body_iquat;
      const [x, y, z, w] = q, bw = b[i], bx = b[i + 1], by = b[i + 2], bz = b[i + 3];
      p.refquat.set([w * bw - x * bx - y * by - z * bz, w * bx + x * bw + y * bz - z * by,
        w * by - x * bz + y * bw + z * bx, w * bz + x * by - y * bx + z * bw]);
    }
    const start = this.body * 6;
    this.data.xfrc_applied.fill(0, start, start + 6);
    this.mj.mjv_applyPerturbForce(this.model, this.data, p);
    for (let i = start; i < start + 6; i++) this.data.xfrc_applied[i] *= target.strength;
  }
  clear(): void {
    if (this.body > 0) this.data.xfrc_applied.fill(0, this.body * 6, this.body * 6 + 6);
    this.held = null; this.body = -1; this.perturb.active = 0; this.perturb.active2 = 0;
  }
  dispose(): void { this.clear(); this.perturb.delete(); }
}
