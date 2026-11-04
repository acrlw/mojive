import test from 'node:test';
import assert from 'node:assert/strict';
import { snapValue } from '../src/viewport/interaction_math.ts';
test('native NumPy snapping uses nearest-even ties and preserves negative zero', () => {
  assert.deepEqual(
    [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5].map((value) => snapValue(value, 1)),
    [-2, -2, -0, 0, 2, 2],
  );
  assert.ok(Object.is(snapValue(-0.025, 0.1), -0));
  assert.equal(snapValue(0.375, 0.25), 0.5);
});
