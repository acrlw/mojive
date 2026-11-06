import type { MainModule, MjModel } from '@mujoco/mujoco';
import type { PhysicsGeometry } from '../../contracts';

export function readSurface(mj: MainModule, model: MjModel, index: number): Pick<PhysicsGeometry, 'material' | 'roughness' | 'metallic'> {
  if (index < 0) return { roughness: .5, metallic: 0 };
  const material = readMaterial(mj, model, index), maps = material.maps;
  // MuJoCo uses -1 for unspecified PBR factors. A texture needs a neutral
  // multiplier; otherwise retain Mojive's legacy shininess-to-roughness mapping.
  const roughness = model.mat_roughness[index], metallic = model.mat_metallic[index];
  const bounded = (value: number, field: string) => {
    if (!Number.isFinite(value)) throw new Error(`Material "${material.name}" has non-finite ${field}.`);
    return Math.max(0, Math.min(1, value));
  };
  const roughnessFactor = bounded(roughness, 'roughness'), metallicFactor = bounded(metallic, 'metallic');
  return {
    material,
    roughness: roughness >= 0 ? roughnessFactor : maps?.roughness !== undefined || maps?.orm !== undefined
      ? 1 : 1 - bounded(material.shininess ?? .5, 'shininess'),
    metallic: metallic >= 0 ? metallicFactor : maps?.metallic !== undefined || maps?.orm !== undefined ? 1 : 0,
  };
}

/** Rigid geoms, flexes and skins share the compiled material contract. */
export function readMaterial(mj: MainModule, model: MjModel, index: number): NonNullable<PhysicsGeometry['material']> {
  const roles = mj.mjtTextureRole;
  const textureAt = (role: number) => model.mat_texid[index * roles.mjNTEXROLE.value + role];
  const rgb = textureAt(roles.mjTEXROLE_RGB.value), rgba = textureAt(roles.mjTEXROLE_RGBA.value);
  const mapRoles = {
    normal: roles.mjTEXROLE_NORMAL.value, occlusion: roles.mjTEXROLE_OCCLUSION.value,
    roughness: roles.mjTEXROLE_ROUGHNESS.value, metallic: roles.mjTEXROLE_METALLIC.value,
    opacity: roles.mjTEXROLE_OPACITY.value, orm: roles.mjTEXROLE_ORM.value,
  };
  return {
    name: mj.mj_id2name(model, mj.mjtObj.mjOBJ_MATERIAL.value, index) || `Material ${index}`,
    texture: rgb >= 0 ? rgb : rgba, textureAlpha: rgb < 0 && rgba >= 0,
    maps: Object.fromEntries(Object.entries(mapRoles).map(([role, value]) => [role, textureAt(value)]).filter(([, value]) => Number(value) >= 0)),
    repeat: [model.mat_texrepeat[index * 2], model.mat_texrepeat[index * 2 + 1]], uniform: !!model.mat_texuniform[index],
    emission: model.mat_emission[index], specular: model.mat_specular[index],
    shininess: model.mat_shininess[index], reflectance: model.mat_reflectance[index],
  };
}
