import type { MainModule, MjModel } from '@mujoco/mujoco';
import type { GeometryShape, PhysicsSource, Quaternion, Vector3, EnvironmentField } from '../../contracts';
import type { ModelBundle } from '../../../resources/modelBundle';
import { readSurface } from './material.ts';

const v3 = (values: ArrayLike<number>, index: number): Vector3 => [values[3 * index], values[3 * index + 1], values[3 * index + 2]];
export const xyzw = (values: ArrayLike<number>, offset: number): Quaternion => [values[offset + 1], values[offset + 2], values[offset + 3], values[offset]];
const rgba = (values: ArrayLike<number>, index: number): [number, number, number, number] =>
  [values[4 * index], values[4 * index + 1], values[4 * index + 2], values[4 * index + 3]];
export const bodyId = (index: number) => `physics:body:${index}`;
export const jointId = (index: number) => `physics:joint:${index}`;

function mesh(model: MjModel, id: number): GeometryShape {
  const faces = model.mesh_facenum[id];
  const faceStart = model.mesh_faceadr[id];
  const vertices = model.mesh_vertadr[id];
  const normals = model.mesh_normaladr[id];
  const coords = model.mesh_texcoordadr[id];
  const positions = new Float32Array(faces * 9);
  const normal = new Float32Array(faces * 9);
  const uv = coords >= 0 ? new Float32Array(faces * 6) : undefined;
  for (let face = 0; face < faces; face++) {
    for (let corner = 0; corner < 3; corner++) {
      const index = (faceStart + face) * 3 + corner;
      const vertex = model.mesh_face[index] + vertices;
      const n = model.mesh_facenormal[index] + normals;
      for (let axis = 0; axis < 3; axis++) {
        positions[face * 9 + corner * 3 + axis] = model.mesh_vert[vertex * 3 + axis];
        normal[face * 9 + corner * 3 + axis] = model.mesh_normal[n * 3 + axis];
      }
      if (uv) {
        const texcoord = model.mesh_facetexcoord[index] + coords;
        uv[face * 6 + corner * 2] = model.mesh_texcoord[texcoord * 2];
        uv[face * 6 + corner * 2 + 1] = model.mesh_texcoord[texcoord * 2 + 1];
      }
    }
  }
  return { type: 'mesh', positions, normals: normal, uv };
}

function shape(mj: MainModule, model: MjModel, index: number): GeometryShape {
  const size = v3(model.geom_size, index), type = model.geom_type[index], G = mj.mjtGeom;
  if (type === G.mjGEOM_PLANE.value) return { type: 'plane', size };
  if (type === G.mjGEOM_BOX.value) return { type: 'box', size: size.map((v) => v * 2) as Vector3 };
  if (type === G.mjGEOM_SPHERE.value) return { type: 'sphere', radius: size[0] };
  if (type === G.mjGEOM_ELLIPSOID.value) return { type: 'ellipsoid', size };
  if (type === G.mjGEOM_CYLINDER.value || type === G.mjGEOM_CAPSULE.value)
    return { type: type === G.mjGEOM_CYLINDER.value ? 'cylinder' : 'capsule', radius: size[0], length: size[1] * 2 };
  const dataId = model.geom_dataid[index];
  if (type === G.mjGEOM_MESH.value || type === G.mjGEOM_SDF.value) return mesh(model, dataId);
  if (type === G.mjGEOM_HFIELD.value) {
    const rows = model.hfield_nrow[dataId], columns = model.hfield_ncol[dataId];
    const offset = model.hfield_adr[dataId];
    return { type: 'heightfield', rows, columns, size: rgba(model.hfield_size, dataId),
      heights: new Float32Array(model.hfield_data.subarray(offset, offset + rows * columns)) };
  }
  throw new Error(`Unsupported MuJoCo geometry type ${type} at geometry ${index}.`);
}

export function readSource(mj: MainModule, model: MjModel, bundle: ModelBundle, fields: EnvironmentField[]): PhysicsSource {
  const name = (type: number, index: number, fallback: string) => mj.mj_id2name(model, type, index) || fallback;
  const O = mj.mjtObj, J = mj.mjtJoint;
  const visual = model.vis, map = visual.map, headlight = visual.headlight, colors = visual.rgba, stats = model.stat;
  const extent = stats.extent;
  const ambient = headlight.active ? v3(headlight.ambient, 0) : [0, 0, 0] as Vector3;
  for (let i = 0; i < model.nlight; i++) if (model.light_active[i])
    for (let axis = 0; axis < 3; axis++) ambient[axis] += model.light_ambient[i * 3 + axis];
  const visualEnvironment = { ambient, headlight: headlight.active ? v3(headlight.diffuse, 0) : [0, 0, 0] as Vector3,
    fog: v3(colors.fog, 0), fogStart: map.fogstart * extent, fogEnd: map.fogend * extent,
    near: map.znear * extent, far: map.zfar * extent, extent };
  stats.delete(); colors.delete(); headlight.delete(); map.delete(); visual.delete();
  // Match the native adapter: collision proxies are separate only when their
  // welded body also owns a non-colliding appearance. World floors stay shared.
  const collision = Array.from({ length: model.ngeom }, (_, i) => !!(model.geom_contype[i] || model.geom_conaffinity[i]));
  for (let i = 0; i < model.npair; i++) { collision[model.pair_geom1[i]] = true; collision[model.pair_geom2[i]] = true; }
  const owner = (i: number) => model.body_weldid[model.geom_bodyid[i]] || model.geom_bodyid[i];
  const visualBodies = new Set(collision.flatMap((value, i) => value ? [] : [owner(i)]));
  let offset = 0;
  const source: PhysicsSource = {
    id: `mujoco:${mj.mj_versionString()}:${bundle.id}`, name: bundle.name, backend: 'mujoco', version: mj.mj_versionString(),
    capabilities: ['environment', 'controls', 'jointPose', 'perturbation', 'state'],
    bodies: Array.from({ length: model.nbody }, (_, i) => ({
      id: bodyId(i), name: name(O.mjOBJ_BODY.value, i, i ? `Body ${i}` : 'World'),
      parentId: i ? bodyId(model.body_parentid[i]) : null,
      dynamic: model.body_weldid[i] !== 0,
      position: v3(model.body_pos, i), rotation: xyzw(model.body_quat, i * 4),
    })),
    geometries: Array.from({ length: model.ngeom }, (_, i) => {
      const material = model.geom_matid[i];
      const geomColor = rgba(model.geom_rgba, i);
      const defaultColor = geomColor.every((v, k) => Math.abs(v - (k === 3 ? 1 : 0.5)) < 1e-6);
      return { id: `physics:geom:${i}`, name: name(O.mjOBJ_GEOM.value, i, `Geometry ${i + 1}`),
        bodyId: bodyId(model.geom_bodyid[i]), position: v3(model.geom_pos, i), rotation: xyzw(model.geom_quat, i * 4),
        shape: shape(mj, model, i), color: material < 0 || !defaultColor ? geomColor : rgba(model.mat_rgba, material),
        ...readSurface(mj, model, material),
        group: model.geom_group[i], collidable: collision[i], defaultVisible: model.geom_group[i] < 3,
        role: !collision[i] ? 'visual' : model.geom_bodyid[i] === 0 || !visualBodies.has(owner(i)) ? 'both' : 'collision',
      };
    }),
    joints: Array.from({ length: model.njnt }, (_, i) => {
      const type = model.jnt_type[i];
      const width = type === J.mjJNT_FREE.value ? 7 : type === J.mjJNT_BALL.value ? 4 : 1;
      const joint: PhysicsSource['joints'][number] = {
        id: jointId(i), name: name(O.mjOBJ_JOINT.value, i, `Joint ${i + 1}`), bodyId: bodyId(model.jnt_bodyid[i]),
        type: type === J.mjJNT_FREE.value ? 'free' : type === J.mjJNT_BALL.value ? 'ball'
          : type === J.mjJNT_SLIDE.value ? 'prismatic' : model.jnt_limited[i] ? 'revolute' : 'continuous',
        axis: v3(model.jnt_axis, i), anchor: v3(model.jnt_pos, i),
        ...(model.jnt_limited[i] ? { limits: [model.jnt_range[2 * i], model.jnt_range[2 * i + 1]] as [number, number] } : {}),
        offset, width,
      };
      offset += width;
      return joint;
    }),
    actuators: Array.from({ length: model.nu }, (_, i) => {
      const transmission = model.actuator_trntype[i];
      const hasJoint = transmission === mj.mjtTrn.mjTRN_JOINT.value || transmission === mj.mjtTrn.mjTRN_JOINTINPARENT.value;
      return { id: `physics:actuator:${i}`, name: name(O.mjOBJ_ACTUATOR.value, i, `Actuator ${i + 1}`),
        ...(hasJoint ? { jointId: jointId(model.actuator_trnid[i * 2]) } : {}),
        ...(model.actuator_ctrllimited[i] ? { range: [model.actuator_ctrlrange[2 * i], model.actuator_ctrlrange[2 * i + 1]] as [number, number] } : {}),
      };
    }),
    cameras: Array.from({ length: model.ncam }, (_, i) => ({
      id: `physics:camera:${i}`, name: name(O.mjOBJ_CAMERA.value, i, `Camera ${i + 1}`), bodyId: bodyId(model.cam_bodyid[i]),
      position: v3(model.cam_pos, i), rotation: xyzw(model.cam_quat, i * 4), fov: model.cam_fovy[i],
      orthographic: model.cam_projection[i] === 1,
      resolution: [model.cam_resolution[i * 2], model.cam_resolution[i * 2 + 1]],
    })),
    lights: Array.from({ length: model.nlight }, (_, i) => ({
      id: `physics:light:${i}`, name: name(O.mjOBJ_LIGHT.value, i, `Light ${i + 1}`), bodyId: bodyId(model.light_bodyid[i]),
      position: v3(model.light_pos, i), rotation: [0, 0, 0, 1], direction: v3(model.light_dir, i),
      type: model.light_type[i] === mj.mjtLightType.mjLIGHT_DIRECTIONAL.value ? 'directional'
        : model.light_type[i] === mj.mjtLightType.mjLIGHT_POINT.value ? 'point' : 'spot',
      active: !!model.light_active[i], castShadow: !!model.light_castshadow[i], ambient: v3(model.light_ambient, i),
      color: v3(model.light_diffuse, i), cutoff: model.light_cutoff[i],
    })),
    textures: Array.from({ length: model.ntex }, (_, i) => {
      const width = model.tex_width[i], height = model.tex_height[i], channels = model.tex_nchannel[i];
      const start = Number(model.tex_adr[i]);
      return { name: name(O.mjOBJ_TEXTURE.value, i, `Texture ${i}`), width, height, channels,
        colorSpace: model.tex_colorspace[i] === mj.mjtColorSpace.mjCOLORSPACE_LINEAR.value ? 'linear' as const
          : model.tex_colorspace[i] === mj.mjtColorSpace.mjCOLORSPACE_SRGB.value ? 'srgb' as const : undefined,
        type: model.tex_type[i] === mj.mjtTexture.mjTEXTURE_2D.value ? '2d' as const
          : model.tex_type[i] === mj.mjtTexture.mjTEXTURE_SKYBOX.value ? 'skybox' as const : 'cube' as const,
        pixels: new Uint8Array(model.tex_data.subarray(start, start + width * height * channels)) };
    }),
    visual: visualEnvironment, environment: fields, warnings: [],
  };
  return source;
}
