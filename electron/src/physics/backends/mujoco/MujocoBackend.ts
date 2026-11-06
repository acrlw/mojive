import type { MainModule, MjData, MjModel, MjOption, DoubleBuffer } from '@mujoco/mujoco';
import type { PhysicsBackend, PhysicsCommand, PhysicsFrame, PhysicsSnapshot, PhysicsSource, EnvironmentValues } from '../../contracts';
import type { ModelBundle } from '../../../resources/modelBundle';
import { createModelBundle } from '../../../resources/modelBundle.ts';
import { validateEnvironment } from '../../environment.ts';
import { MujocoOptions } from './options.ts';
import { readSource } from './source.ts';
import { MujocoPerturbation } from './perturbation.ts';
import { writeQuaternion } from '../../rotation.ts';
import { MujocoDeformables } from './deformables.ts';

/** Owns all WASM handles. Neither the renderer nor presenters receive engine objects. */
export class MujocoBackend implements PhysicsBackend {
  readonly source: PhysicsSource;
  readonly defaults: EnvironmentValues;
  private mj: MainModule;
  private model: MjModel;
  private data: MjData;
  private options: MujocoOptions;
  private stateBuffer: DoubleBuffer;
  private spec: number;
  private format: string;
  private disposed = false;
  private perturbation: MujocoPerturbation;
  private deformables?: MujocoDeformables;

  static async load(mj: MainModule, input: ModelBundle): Promise<MujocoBackend> {
    const bundle = await createModelBundle(input.name, input.entry, input.files);
    if (bundle.id !== input.id) throw new Error('Model package identity does not match its contents.');
    const vfs = new mj.MjVFS();
    let model: MjModel | undefined;
    let backend: MujocoBackend | undefined;
    let data: MjData | undefined, option: MjOption | undefined, buffer: DoubleBuffer | undefined;
    try {
      for (const file of bundle.files) vfs.addBuffer(file.path, file.bytes);
      model = mj.MjModel.from_xml_path(bundle.entry, vfs);
      data = new mj.MjData(model);
      option = model.opt;
      const options = new MujocoOptions(mj, option);
      const source = readSource(mj, model, bundle, options.fields);
      buffer = new mj.DoubleBuffer(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION.value));
      mj.mj_forward(model, data);
      backend = new MujocoBackend(mj, model, data, options, source, buffer);
      const initial = await backend.capture();
      source.keyframes = [];
      for (let i = 0; i < model.nkey; i++) {
        mj.mj_resetDataKeyframe(model, data, i);
        mj.mj_forward(model, data);
        source.keyframes.push({ id: `physics:key:${i}`,
          name: mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY.value, i) || `Key ${i + 1}`,
          time: data.time, frame: await backend.frame(), snapshot: await backend.capture() });
      }
      await backend.execute({ type: 'restore', snapshot: initial });
      return backend;
    } catch (error) {
      if (backend) backend.dispose();
      else { buffer?.delete(); option?.delete(); data?.delete(); model?.delete(); }
      throw error;
    } finally { vfs.delete(); }
  }

  private constructor(mj: MainModule, model: MjModel, data: MjData, options: MujocoOptions, source: PhysicsSource, buffer: DoubleBuffer) {
    this.mj = mj;
    this.model = model;
    this.data = data;
    this.options = options;
    this.spec = mj.mjtState.mjSTATE_INTEGRATION.value;
    this.stateBuffer = buffer;
    this.format = `mujoco/${mj.mj_versionString()}/integration/${this.spec}`;
    this.source = source;
    this.defaults = structuredClone(options.defaults);
    this.perturbation = new MujocoPerturbation(mj, model, data, source);
    if (model.nflex || model.nskin) this.deformables = new MujocoDeformables(mj, model, data, source);
  }

  private requireLive(): void {
    if (this.disposed) throw new Error('The simulation has been closed.');
  }

  async frame(): Promise<PhysicsFrame> {
    this.requireLive();
    const { model, data, source } = this;
    const transforms = new Float64Array(model.nbody * 7);
    for (let i = 0; i < model.nbody; i++) {
      transforms.set(data.xpos.subarray(i * 3, i * 3 + 3), i * 7);
      transforms.set(data.xquat.subarray(i * 4 + 1, i * 4 + 4), i * 7 + 3);
      transforms[i * 7 + 6] = data.xquat[i * 4];
    }
    const positions = new Float64Array(model.nq);
    const jointFrames = new Float64Array(model.njnt * 6);
    for (let i = 0; i < source.joints.length; i++) {
      const joint = source.joints[i], address = model.jnt_qposadr[i];
      positions.set(data.qpos.subarray(address, address + joint.width), joint.offset);
      if (joint.width > 1) {
        const q = joint.width === 7 ? 3 : 0;
        positions.set(data.qpos.subarray(address + q + 1, address + q + 4), joint.offset + q);
        positions[joint.offset + q + 3] = data.qpos[address + q];
      }
      jointFrames.set(data.xanchor.subarray(i * 3, i * 3 + 3), i * 6);
      jointFrames.set(data.xaxis.subarray(i * 3, i * 3 + 3), i * 6 + 3);
    }
    const cameraTransforms = new Float64Array(model.ncam * 7);
    for (let i = 0; i < model.ncam; i++) {
      cameraTransforms.set(data.cam_xpos.subarray(i * 3, i * 3 + 3), i * 7);
      writeQuaternion(data.cam_xmat, i * 9, cameraTransforms, i * 7 + 3);
    }
    const lightFrames = new Float64Array(model.nlight * 6);
    for (let i = 0; i < model.nlight; i++) {
      lightFrames.set(data.light_xpos.subarray(i * 3, i * 3 + 3), i * 6);
      lightFrames.set(data.light_xdir.subarray(i * 3, i * 3 + 3), i * 6 + 3);
    }
    return { sourceId: source.id, time: data.time, bodyTransforms: transforms,
      jointPositions: positions, controls: new Float64Array(data.ctrl), jointFrames, cameraTransforms, lightFrames,
      ...(this.deformables ? { geometryUpdates: this.deformables.frame() } : {}) };
  }

  async environment(): Promise<EnvironmentValues> {
    this.requireLive();
    return this.options.read();
  }

  private state(): number[] {
    this.mj.mj_getState(this.model, this.data, this.stateBuffer.GetView(), this.spec);
    return Array.from(this.stateBuffer.GetView());
  }

  async capture(): Promise<PhysicsSnapshot> {
    this.requireLive();
    return { sourceId: this.source.id, backend: 'mujoco', format: this.format,
      state: this.state(), environment: this.options.read(),
      ...(this.perturbation.capture() ? { interaction: this.perturbation.capture() } : {}) };
  }

  private restore(snapshot: PhysicsSnapshot): void {
    if (snapshot.sourceId !== this.source.id || snapshot.backend !== 'mujoco' || snapshot.format !== this.format)
      throw new Error('The simulation snapshot belongs to a different model or backend version.');
    if (snapshot.state.length !== this.stateBuffer.GetElementCount() || !snapshot.state.every(Number.isFinite))
      throw new Error('Invalid simulation state.');
    if (Object.keys(snapshot.environment).length !== this.options.fields.length)
      throw new Error('The simulation snapshot has incomplete environment settings.');
    validateEnvironment(this.options.fields, snapshot.environment);
    this.perturbation.restore(snapshot.interaction);
    this.options.write(snapshot.environment);
    this.mj.mj_setState(this.model, this.data, snapshot.state, this.spec);
    this.mj.mj_forward(this.model, this.data);
    // Forward computes display transforms but also updates warm-start state. Restore
    // the integration state again so continuation reproduces the captured trajectory.
    this.mj.mj_setState(this.model, this.data, snapshot.state, this.spec);
  }

  async execute(command: PhysicsCommand): Promise<PhysicsFrame> {
    this.requireLive();
    const { mj, model, data, source } = this;
    const checkpoint = await this.capture();
    try {
      switch (command.type) {
        case 'step': {
          if (!Number.isInteger(command.count) || command.count < 1 || command.count > 10000)
            throw new Error('Step count must be an integer between 1 and 10000.');
          const start = data.time;
          for (let i = 0; i < command.count; i++) { this.perturbation.apply(); mj.mj_step(model, data); }
          if (data.time <= start || !Array.from(data.qpos as Float64Array).every(Number.isFinite) ||
            !Array.from(data.qvel as Float64Array).every(Number.isFinite))
            throw new Error('Simulation became unstable. The last valid state has been restored.');
          mj.mj_forward(model, data);
          break;
        }
        case 'perturbTarget': this.perturbation.set(command.target); break;
        case 'clearPerturbation': this.perturbation.clear(); break;
        case 'reset': this.perturbation.clear(); mj.mj_resetData(model, data); mj.mj_forward(model, data); break;
        case 'environment': this.options.write(command.values); break;
        case 'resetEnvironment': this.options.write(this.defaults); break;
        case 'restore': this.restore(command.snapshot); break;
        case 'controls': {
          const updates = command.values.map(({ id, value }) => {
            const index = source.actuators.findIndex((actuator) => actuator.id === id);
            if (index < 0) throw new Error(`Unknown actuator: ${id}`);
            const range = source.actuators[index].range;
            if (!Number.isFinite(value) || (range && (value < range[0] || value > range[1])))
              throw new Error(`Invalid control for ${source.actuators[index].name}.`);
            return { index, value };
          });
          for (const { index, value } of updates) data.ctrl[index] = value;
          break;
        }
        case 'jointPose': {
          const index = source.joints.findIndex((joint) => joint.id === command.id);
          if (index < 0) throw new Error(`Unknown joint: ${command.id}`);
          const joint = source.joints[index], values = [...command.position];
          if (values.length !== joint.width || !values.every(Number.isFinite)) throw new Error('Invalid joint position.');
          if (joint.width === 1 && joint.limits && (values[0] < joint.limits[0] || values[0] > joint.limits[1]))
            throw new Error('Joint position is outside its limits.');
          if (joint.width > 1) {
            const offset = joint.width === 7 ? 3 : 0, quaternion = values.slice(offset);
            const length = Math.hypot(...quaternion);
            if (length < 1e-12) throw new Error('Joint orientation must be a nonzero quaternion.');
            values.splice(offset, 4, quaternion[3] / length, quaternion[0] / length, quaternion[1] / length, quaternion[2] / length);
          }
          data.qpos.set(values, model.jnt_qposadr[index]);
          const velocity = model.jnt_dofadr[index], width = joint.width === 7 ? 6 : joint.width === 4 ? 3 : 1;
          data.qvel.fill(0, velocity, velocity + width);
          mj.mj_forward(model, data);
          break;
        }
        case 'perturb': {
          const body = source.bodies.findIndex((item) => item.id === command.bodyId);
          if (body <= 0 || !source.bodies[body].dynamic) throw new Error('Perturbation requires a dynamic body.');
          if (command.force.length !== 3 || command.torque.length !== 3 ||
            ![...command.force, ...command.torque].every(Number.isFinite)) throw new Error('Invalid perturbation.');
          data.xfrc_applied.set([...command.force, ...command.torque], body * 6);
          break;
        }
      }
      return await this.frame();
    } catch (error) {
      this.restore(checkpoint);
      throw error;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.perturbation.dispose();
    this.deformables?.dispose();
    this.stateBuffer.delete(); this.options.dispose(); this.data.delete(); this.model.delete();
  }
}
