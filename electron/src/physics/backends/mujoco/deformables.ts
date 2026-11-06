import type { MainModule, MjModel, MjData, MjvScene, MjvOption, MjvPerturb, MjvCamera } from '@mujoco/mujoco';
import type { PhysicsSource, PhysicsFrame, PhysicsGeometry } from '../../contracts';

/** MuJoCo owns skinning and flex surface generation; only neutral mesh buffers leave the backend. */
export class MujocoDeformables {
  private mj: MainModule;
  private model: MjModel;
  private data: MjData;
  private scene: MjvScene;
  private option: MjvOption;
  private perturb: MjvPerturb;
  private camera: MjvCamera;
  private meshes: { id: string; kind: 'flex' | 'skin'; index: number; size: number }[] = [];

  constructor(mj: MainModule, model: MjModel, data: MjData, source: PhysicsSource) {
    this.mj = mj; this.model = model; this.data = data;
    this.scene = new mj.MjvScene(model, Math.max(1000, model.ngeom + model.nflexedge + model.nflexvert));
    this.option = new mj.MjvOption(); this.perturb = new mj.MjvPerturb(); this.camera = new mj.MjvCamera();
    mj.mjv_defaultOption(this.option); mj.mjv_defaultPerturb(this.perturb); mj.mjv_defaultFreeCamera(model, this.camera);
    this.option.flexgroup.fill(1); this.option.skingroup.fill(1);
    this.update();
    for (const kind of ['flex', 'skin'] as const) {
      const count = kind === 'flex' ? model.nflex : model.nskin;
      for (let i = 0; i < count; i++) {
        const size = kind === 'flex' ? this.scene.flexfacenum[i] * 9 : this.scene.skinvertnum[i] * 3;
        if (!size) { source.warnings.push(`No surface mesh for ${kind} ${i}.`); continue; }
        const id = `physics:${kind}:${i}`, matid = kind === 'flex' ? model.flex_matid[i] : model.skin_matid[i];
        const colors = kind === 'flex' ? model.flex_rgba : model.skin_rgba;
        const group = kind === 'flex' ? model.flex_group[i] : model.skin_group[i];
        const material = matid >= 0 ? this.material(matid) : undefined;
        const color = Array.from(colors.subarray(i * 4, i * 4 + 4)) as PhysicsGeometry['color'];
        if (matid >= 0 && color.every((v, k) => Math.abs(v - (k === 3 ? 1 : .5)) < 1e-6))
          color.splice(0, 4, ...model.mat_rgba.subarray(matid * 4, matid * 4 + 4));
        const uv = kind === 'flex' ? new Float32Array(this.scene.flextexcoord.subarray(this.scene.flexfaceadr[i] * 6, this.scene.flexfaceadr[i] * 6 + size / 3 * 2))
          : model.skin_texcoordadr[i] >= 0 ? new Float32Array(model.skin_texcoord.subarray(model.skin_texcoordadr[i] * 2, model.skin_texcoordadr[i] * 2 + size / 3 * 2)) : undefined;
        const indices = kind === 'skin' ? new Uint32Array(model.skin_face.subarray(model.skin_faceadr[i] * 3, (model.skin_faceadr[i] + model.skin_facenum[i]) * 3)) : undefined;
        source.geometries.push({ id, name: mj.mj_id2name(model, kind === 'flex' ? mj.mjtObj.mjOBJ_FLEX.value : mj.mjtObj.mjOBJ_SKIN.value, i) || `${kind} ${i + 1}`,
          bodyId: source.bodies[0].id, position: [0, 0, 0], rotation: [0, 0, 0, 1],
          shape: { type: 'mesh', positions: new Float32Array(size), normals: new Float32Array(size), uv, indices },
          color, material, group, defaultVisible: group < 3, role: 'both', deformable: true, collidable: kind === 'flex',
          metallic: matid < 0 ? 0 : model.mat_metallic[matid], roughness: matid < 0 ? .5 : model.mat_roughness[matid] });
        this.meshes.push({ id, kind, index: i, size });
      }
    }
  }

  private material(i: number): NonNullable<PhysicsGeometry['material']> {
    const { mj, model: m } = this, roles = mj.mjtTextureRole;
    let texture = m.mat_texid[i * roles.mjNTEXROLE.value + roles.mjTEXROLE_RGB.value];
    if (texture < 0) texture = m.mat_texid[i * roles.mjNTEXROLE.value + roles.mjTEXROLE_RGBA.value];
    return { name: mj.mj_id2name(m, mj.mjtObj.mjOBJ_MATERIAL.value, i) || `Material ${i}`, texture,
      repeat: [m.mat_texrepeat[i * 2], m.mat_texrepeat[i * 2 + 1]], uniform: !!m.mat_texuniform[i],
      emission: m.mat_emission[i], specular: m.mat_specular[i], shininess: m.mat_shininess[i], reflectance: m.mat_reflectance[i] };
  }

  private update(): void {
    this.mj.mjv_updateScene(this.model, this.data, this.option, this.perturb, this.camera, 7, this.scene);
  }

  frame(): NonNullable<PhysicsFrame['geometryUpdates']> {
    this.update();
    return this.meshes.map(({ id, kind, index: i, size }) => {
      const s = this.scene, offset = kind === 'flex' ? s.flexfaceadr[i] * 9 : s.skinvertadr[i] * 3;
      const positions = kind === 'flex' ? s.flexface : s.skinvert;
      const normals = kind === 'flex' ? s.flexnormal : s.skinnormal;
      return { id, positions: new Float32Array(positions.subarray(offset, offset + size)),
        normals: new Float32Array(normals.subarray(offset, offset + size)),
        count: kind === 'flex' ? s.flexfaceused[i] * 3 : this.model.skin_facenum[i] * 3 };
    });
  }

  dispose(): void { this.scene.delete(); this.option.delete(); this.perturb.delete(); this.camera.delete(); }
}
