import test from 'node:test';
import assert from 'node:assert/strict';
import { sameHistoryValue } from '../src/historyValue.ts';

test('history comparison preserves authored changes independently of object key order', () => {
  const state = { pose: { joint: [0, 1, 2] }, keys: [{ time: 0, name: 'Start' }] };
  assert.ok(sameHistoryValue(state, { keys: structuredClone(state.keys), pose: structuredClone(state.pose) }));
  for (const mutate of [
    (s) => { s.pose.joint[1] = 3; },
    (s) => { s.keys[0].name = 'Updated'; },
    (s) => { s.keys.push({ time: 1, name: 'End' }); },
    (s) => { delete s.pose.joint; },
  ]) {
    const changed = structuredClone(state);
    mutate(changed);
    assert.equal(sameHistoryValue(state, changed), false);
  }
  assert.equal(sameHistoryValue({ a: undefined }, { b: undefined }), false);
  assert.equal(sameHistoryValue([], {}), false);
  assert.equal(sameHistoryValue(null, {}), false);
});

test('large numeric history data compares without JSON serialization', () => {
  const values = Array.from({ length: 250_000 }, (_, i) => i / 3);
  const other = values.slice();
  Object.defineProperty(values, 'toJSON', { value() { throw new Error('Geometry must not be serialized for comparison.'); } });
  assert.ok(sameHistoryValue(values, other));
  other[other.length - 1]++;
  assert.equal(sameHistoryValue(values, other), false);
});
