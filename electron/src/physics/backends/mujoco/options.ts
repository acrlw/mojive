import type { MainModule, MjOption } from '@mujoco/mujoco';
import type { EnvironmentField, EnvironmentValues } from '../../contracts';
import { validateEnvironment } from '../../environment.ts';

const numeric = [
  ['timestep', 'Timestep', 'Simulation', 1e-9, false],
  ['density', 'Density', 'Physical parameters', 0, false],
  ['viscosity', 'Viscosity', 'Physical parameters', 0, false],
  ['impratio', 'Impedance ratio', 'Physical parameters', 1e-9, false],
  ['iterations', 'Iterations', 'Solver parameters', 1, true],
  ['tolerance', 'Tolerance', 'Solver parameters', 0, false],
  ['ls_iterations', 'Line search iterations', 'Solver parameters', 1, true],
  ['ls_tolerance', 'Line search tolerance', 'Solver parameters', 0, false],
  ['noslip_iterations', 'Noslip iterations', 'Solver parameters', 0, true],
  ['noslip_tolerance', 'Noslip tolerance', 'Solver parameters', 0, false],
  ['ccd_iterations', 'CCD iterations', 'Solver parameters', 1, true],
  ['ccd_tolerance', 'CCD tolerance', 'Solver parameters', 0, false],
  ['sleep_tolerance', 'Sleep tolerance', 'Solver parameters', 0, false],
  ['sdf_iterations', 'SDF iterations', 'Solver parameters', 1, true],
  ['sdf_initpoints', 'SDF initial points', 'Solver parameters', 1, true],
  ['o_margin', 'Margin', 'Contact override', 0, false],
] as const;
const vectors = [
  ['gravity', 'Gravity', 'Simulation', 3],
  ['wind', 'Wind', 'Physical parameters', 3],
  ['magnetic', 'Magnetic field', 'Physical parameters', 3],
  ['o_solref', 'Solver reference', 'Contact override', 2],
  ['o_solimp', 'Solver impedance', 'Contact override', 5],
  ['o_friction', 'Friction', 'Contact override', 5],
] as const;
const labels: Record<string, string> = {
  EULER: 'Euler', RK4: 'RK4', IMPLICIT: 'Implicit', IMPLICITFAST: 'Implicit fast',
  DISCRETE: 'Discrete', PGS: 'PGS', CG: 'CG', NEWTON: 'Newton', AUTO: 'Auto',
  PYRAMIDAL: 'Pyramidal', ELLIPTIC: 'Elliptic', DENSE: 'Dense', SPARSE: 'Sparse',
  FRICTIONLOSS: 'Friction loss', CLAMPCTRL: 'Control clamping', WARMSTART: 'Warm start',
  FILTERPARENT: 'Parent collision filtering', REFSAFE: 'Reference safety',
  EULERDAMP: 'Euler damping', AUTORESET: 'Automatic reset', NATIVECCD: 'Native CCD',
  MULTICCD: 'Multiple CCD', FWDINV: 'Forward/inverse comparison',
  INVDISCRETE: 'Discrete inverse dynamics', DIAGEXACT: 'Exact solver diagonal',
};
const enumLabel = (key: string) => {
  const tail = key.slice(key.indexOf('_') + 1);
  return labels[tail] ?? tail[0] + tail.slice(1).toLowerCase();
};

/** Options and enum availability belong to the adapter, never to UI controls. */
export class MujocoOptions {
  readonly fields: EnvironmentField[];
  readonly defaults: EnvironmentValues;
  private option: MjOption;
  private flags = new Map<string, { field: 'disableflags' | 'enableflags'; bit: number }>();

  constructor(mj: MainModule, option: MjOption) {
    this.option = option;
    const fields: EnvironmentField[] = [];
    for (const [id, label, group, min, integer] of numeric) {
      if (!(id in option)) continue;
      fields.push({ id, label, group, type: 'number', min, integer,
        advanced: group !== 'Simulation', step: integer ? 1 : id === 'timestep' ? 0.0001 : 0.001,
        ...(id === 'timestep' ? { unit: 's' } : {}) });
    }
    for (const [id, label, group, size] of vectors) {
      if (!(id in option)) continue;
      fields.push({ id, label, group, type: 'vector', size, step: 0.01,
        advanced: group !== 'Simulation', ...(id === 'gravity' ? { unit: 'm/s²' } : {}) });
    }
    for (const [id, label, values] of [
      ['integrator', 'Integrator', mj.mjtIntegrator], ['solver', 'Solver', mj.mjtSolver],
      ['cone', 'Friction cone', mj.mjtCone], ['jacobian', 'Jacobian', mj.mjtJacobian],
    ] as const) {
      fields.push({ id, label, group: 'Algorithms', type: 'enum', advanced: true,
        choices: Object.entries(values).filter(([key, value]) => key.startsWith('mj') && Number.isFinite(value?.value))
          .map(([key, value]) => ({ value: value.value, label: enumLabel(key) })) });
    }
    for (const [field, group, prefix, values] of [
      ['disableflags', 'Disable flags', 'mjDSBL_', mj.mjtDisableBit],
      ['enableflags', 'Enable flags', 'mjENBL_', mj.mjtEnableBit],
    ] as const) {
      for (const [key, value] of Object.entries(values)) {
        if (!key.startsWith(prefix)) continue;
        const id = `${field}.${key}`;
        this.flags.set(id, { field, bit: value.value });
        fields.push({ id, label: enumLabel(key), group, type: 'boolean', advanced: true });
      }
    }
    this.fields = fields;
    this.defaults = this.read();
  }

  read(): EnvironmentValues {
    const values: EnvironmentValues = {};
    const option = this.option as unknown as Record<string, number | Float64Array>;
    for (const field of this.fields) {
      const flag = this.flags.get(field.id);
      values[field.id] = flag ? !!(this.option[flag.field] & flag.bit)
        : field.type === 'vector' ? Array.from(option[field.id] as Float64Array)
        : option[field.id] as number;
    }
    return values;
  }

  write(patch: EnvironmentValues): void {
    validateEnvironment(this.fields, patch);
    const option = this.option as unknown as Record<string, number | Float64Array>;
    for (const [id, value] of Object.entries(patch)) {
      const flag = this.flags.get(id);
      if (flag) this.option[flag.field] = value
        ? this.option[flag.field] | flag.bit : this.option[flag.field] & ~flag.bit;
      else if (Array.isArray(value)) (option[id] as Float64Array).set(value);
      else option[id] = value as number;
    }
  }

  dispose(): void { this.option.delete(); }
}
