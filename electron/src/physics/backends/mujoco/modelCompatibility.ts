export type MujocoCompatibilityCode = 'removed-flex-internal' | 'flex-integrator-migration';

/** A model/compiler version mismatch, preserving the engine diagnostic and source bytes. */
export class MujocoModelCompatibilityError extends Error {
  readonly code: MujocoCompatibilityCode;
  readonly entry: string;
  readonly runtimeVersion: string;

  constructor(code: MujocoCompatibilityCode, entry: string, runtimeVersion: string, detail: string, cause: unknown) {
    const diagnostic = cause instanceof Error ? cause.message : String(cause);
    super(`Cannot load "${entry}" with MuJoCo ${runtimeVersion}. ${detail}\n\n${diagnostic}`, { cause });
    this.name = 'MujocoModelCompatibilityError';
    this.code = code;
    this.entry = entry;
    this.runtimeVersion = runtimeVersion;
  }
}

export function explainModelCompatibility(error: unknown, entry: string, runtimeVersion: string): unknown {
  const diagnostic = error instanceof Error ? error.message : String(error);
  if (/unrecognized attribute: ['"]internal['"]/.test(diagnostic) && /Element ['"]contact['"]/.test(diagnostic)) {
    return new MujocoModelCompatibilityError('removed-flex-internal', entry, runtimeVersion,
      'The flex contact "internal" option was removed in MuJoCo 3.15. Use a model compatible with this engine version, '
      + 'or explicitly migrate and validate its flex collision settings. The source model was not changed.', error);
  }
  if (diagnostic.includes('flex elasticity is no longer integrated implicitly')) {
    return new MujocoModelCompatibilityError('flex-integrator-migration', entry, runtimeVersion,
      'MuJoCo 3.13 moved implicit flex elasticity to integrator="discrete" with a primal solver (CG or Newton). '
      + 'Migrate and validate the source model, or use its original engine version. The integrator was not changed automatically.', error);
  }
  return error;
}
