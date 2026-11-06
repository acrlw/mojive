import type { EnvironmentField, EnvironmentValues } from './contracts';

/** Validate the whole patch before any backend writes, including vector lengths. */
export function validateEnvironment(fields: EnvironmentField[], patch: EnvironmentValues): void {
  for (const [id, value] of Object.entries(patch)) {
    const field = fields.find((item) => item.id === id);
    if (!field) throw new Error(`Unknown environment setting: ${id}`);
    if (field.type === 'boolean') {
      if (typeof value !== 'boolean') throw new Error(`${field.label} must be a boolean.`);
    } else if (field.type === 'enum') {
      if (!field.choices.some((choice) => choice.value === value))
        throw new Error(`Invalid ${field.label} choice.`);
    } else {
      const values = field.type === 'vector' ? value : [value];
      if (!Array.isArray(values) || (field.type === 'vector' && values.length !== field.size))
        throw new Error(`Invalid ${field.label} dimensions.`);
      for (const number of values) {
        if (typeof number !== 'number' || !Number.isFinite(number) ||
          (field.min !== undefined && number < field.min) ||
          (field.max !== undefined && number > field.max) ||
          (field.type === 'number' && field.integer && !Number.isInteger(number)))
          throw new Error(`Invalid ${field.label} value.`);
      }
    }
  }
}
