/** NumPy's nearest-even scalar snapping, including the signed zero used by native readouts. */
export function roundEven(value: number): number {
  const lower = Math.floor(value),
    fraction = value - lower;
  const result = fraction === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(value);
  return result === 0 && value < 0 ? -0 : result;
}
export function snapValue(value: number, step: number): number {
  return roundEven(value / step) * step;
}
