/** Convert a row-major orthonormal matrix to the neutral xyzw convention. */
export function writeQuaternion(matrix: ArrayLike<number>, offset: number, out: Float64Array, target: number): void {
  const m00 = matrix[offset], m11 = matrix[offset + 4], m22 = matrix[offset + 8];
  const trace = m00 + m11 + m22;
  let x: number, y: number, z: number, w: number;
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    w = s / 4; x = (matrix[offset + 7] - matrix[offset + 5]) / s;
    y = (matrix[offset + 2] - matrix[offset + 6]) / s; z = (matrix[offset + 3] - matrix[offset + 1]) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    x = s / 4; w = (matrix[offset + 7] - matrix[offset + 5]) / s;
    y = (matrix[offset + 1] + matrix[offset + 3]) / s; z = (matrix[offset + 2] + matrix[offset + 6]) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    y = s / 4; w = (matrix[offset + 2] - matrix[offset + 6]) / s;
    x = (matrix[offset + 1] + matrix[offset + 3]) / s; z = (matrix[offset + 5] + matrix[offset + 7]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    z = s / 4; w = (matrix[offset + 3] - matrix[offset + 1]) / s;
    x = (matrix[offset + 2] + matrix[offset + 6]) / s; y = (matrix[offset + 5] + matrix[offset + 7]) / s;
  }
  const inverseLength = 1 / Math.hypot(x, y, z, w);
  out[target] = x * inverseLength; out[target + 1] = y * inverseLength;
  out[target + 2] = z * inverseLength; out[target + 3] = w * inverseLength;
}
