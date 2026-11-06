import assert from 'node:assert/strict';
import test from 'node:test';
import {
  clampView,
  fittedView,
  navigationLimits,
  normalizeWheel,
  panView,
  zoomView,
} from '../src/timelineNavigation.ts';

const span = (view) => view.end - view.start;
const at = (view, ratio) => view.start + ratio * span(view);
const close = (actual, expected, message, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(actual), `${message}: nonfinite ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `${message}: ${actual} != ${expected}`,
  );
};
const assertBounded = (view, limits) => {
  assert.ok(Number.isFinite(view.start) && Number.isFinite(view.end));
  assert.ok(span(view) > 0, 'range must remain increasing');
  assert.ok(span(view) >= limits.minSpan * (1 - 1e-8));
  assert.ok(span(view) <= limits.maxSpan * (1 + 1e-10));
  assert.ok(view.start >= limits.lower - 1e-9);
  assert.ok(view.end <= limits.upper + 1e-9);
};

test('fit retains useful subsecond scale and includes the current playhead', () => {
  const short = fittedView([0, 0.25]);
  close(short.start, 0, 'short clip start');
  close(short.end, 0.27, 'short clip end with native eight-percent padding');
  const offset = fittedView([10, 10.25]);
  close(offset.start, 9.98, 'offset clip start');
  close(offset.end, 10.27, 'offset clip end');
  const fine = clampView(fittedView([0, 0.0001]), navigationLimits([0, 0.0001]));
  close(span(fine), 0.001, 'submillisecond clips fit at the interactive minimum');
  const playhead = fittedView([12, 0, 0.25]);
  assert.ok(playhead.start <= 0 && playhead.end > 12, 'fit includes the playhead and content');
  assert.deepEqual(fittedView([5]), { start: 4.5, end: 5.5 });
  assert.deepEqual(fittedView([NaN, Infinity]), { start: 0, end: 1 });
});

test('fitting the displayed take is independent of inactive-take navigation context', () => {
  const activeTimes = [0, 0.25];
  const limits = navigationLimits([...activeTimes, 3600]);
  const fit = clampView(fittedView(activeTimes), limits);
  close(fit.start, 0, 'active take start');
  close(fit.end, 0.27, 'inactive take does not dilute the displayed take');
  assert.equal(limits.maxSpan, 14400, 'inactive take remains navigable separately');
});

test('wheel units have equivalent modest steps and retain precise fractional motion', () => {
  close(normalizeWheel(100, 0), 1, 'pixel notch');
  close(normalizeWheel(3, 1), 1, 'line notch');
  close(normalizeWheel(1, 2), 1, 'page notch');
  close(normalizeWheel(-100, 0), -1, 'negative pixel notch');
  close(normalizeWheel(-3, 1), -1, 'negative line notch');
  close(normalizeWheel(-1, 2), -1, 'negative page notch');
  close(normalizeWheel(0.25, 0), 0.0025, 'fractional trackpad motion');
  for (const mode of [0, 1, 2]) {
    assert.equal(normalizeWheel(1e300, mode), 4);
    assert.equal(normalizeWheel(-1e300, mode), -4);
    for (const delta of [NaN, Infinity, -Infinity]) assert.equal(normalizeWheel(delta, mode), 0);
  }
});

test('zoom uses the native exponential factor and preserves the cursor time away from boundaries', () => {
  const limits = navigationLimits([0, 30]);
  const original = { start: 20, end: 30 };
  for (const ratio of [0.05, 0.35, 0.95]) {
    const anchor = at(original, ratio);
    const outward = zoomView(original, ratio, -1, limits);
    close(span(outward), 10 * Math.exp(0.18), 'one outward notch');
    close(at(outward, ratio), anchor, 'outward pointer anchor');
    const inward = zoomView(outward, ratio, 1, limits);
    close(inward.start, original.start, 'reverse wheel start');
    close(inward.end, original.end, 'reverse wheel end');
  }
});

test('zooming out at zero retains the scale without revealing negative time', () => {
  const limits = navigationLimits([0, 10]);
  assert.equal(limits.lower, 0);
  for (const ratio of [0, 0.05, 0.35, 0.95, 1]) {
    const outward = zoomView({ start: 0, end: 10 }, ratio, -1, limits);
    assert.equal(outward.start, 0);
    close(span(outward), 10 * Math.exp(0.18), 'zero-boundary wheel response');
    const inward = zoomView(outward, ratio, 1, limits);
    close(at(inward, ratio), at(outward, ratio), 'inward zoom resumes cursor anchoring');
    assert.ok(span(inward) < span(outward), 'reverse wheel responds immediately');
  }
  assert.deepEqual(clampView({ start: -4000, end: -3990 }, limits), { start: 0, end: 10 });
  assert.deepEqual(clampView({ start: -1, end: 9 }, limits), { start: 0, end: 10 });
  assert.equal(navigationLimits([0, 10], { start: -4000, end: 10 }).maxSpan, 60);
});

test('precise wheel packets compose without rounding away motion or changing the anchor', () => {
  const limits = navigationLimits([0, 20]);
  const original = { start: 1, end: 11 };
  const ratio = 0.63;
  let packets = original;
  for (let index = 0; index < 100; index++)
    packets = zoomView(packets, ratio, normalizeWheel(1, 0), limits);
  const single = zoomView(original, ratio, normalizeWheel(100, 0), limits);
  close(packets.start, single.start, 'packet-independent start');
  close(packets.end, single.end, 'packet-independent end');
  close(at(packets, ratio), at(original, ratio), 'packet-independent pointer time');
});

test('repeated extreme wheel input saturates safely and reversing remains immediate', () => {
  const limits = navigationLimits([0, 0.15]);
  assert.equal(limits.maxSpan, 60);
  for (const ratio of [0.05, 0.35, 0.95]) {
    let view = { start: 0, end: 1 };
    for (let index = 0; index < 500; index++) {
      view = zoomView(view, ratio, -1e300, limits);
      assertBounded(view, limits);
      assert.equal(view.start, 0, 'outward zoom stops at the first frame');
    }
    close(span(view), limits.maxSpan, 'maximum span');
    const anchor = at(view, ratio);
    view = zoomView(view, ratio, 1, limits);
    assert.ok(span(view) < limits.maxSpan, 'reverse wheel must leave the limit immediately');
    for (let index = 0; index < 500; index++) {
      view = zoomView(view, ratio, 1e300, limits);
      assertBounded(view, limits);
      close(at(view, ratio), anchor, 'inward saturation anchor');
    }
    close(span(view), limits.minSpan, 'minimum span');
    view = zoomView(view, ratio, -1, limits);
    assert.ok(span(view) > limits.minSpan, 'reverse wheel must leave the minimum immediately');
  }
});

test('panning keeps the scale and stops at both content-context boundaries', () => {
  const limits = navigationLimits([0, 4]);
  const original = { start: 0, end: 10 };
  const left = panView(original, -1e300, limits);
  assertBounded(left, limits);
  close(left.start, limits.lower, 'lower pan boundary');
  close(span(left), span(original), 'lower pan span');
  const right = panView(left, 1e300, limits);
  assertBounded(right, limits);
  close(right.end, limits.upper, 'upper pan boundary');
  close(span(right), span(original), 'upper pan span');
  const returned = panView(panView(original, 2, limits), -2, limits);
  close(returned.start, original.start, 'reversible pan start');
  close(returned.end, original.end, 'reversible pan end');
  for (const invalid of [NaN, Infinity, -Infinity])
    assert.deepEqual(panView(original, invalid, limits), original);
});

test('invalid or legacy runaway views recover near content without widening navigation limits', () => {
  const times = [0, 0.15];
  const limits = navigationLimits(times);
  for (const invalid of [
    { start: -4.99e7, end: 9.63e7 },
    { start: NaN, end: Infinity },
    { start: 12, end: 2 },
  ]) {
    const view = clampView(invalid, limits);
    assertBounded(view, limits);
    assert.ok(view.start <= 0 && view.end >= 0.15, 'recovery must bring content back into view');
  }
  const original = { start: 0, end: 1 };
  for (const invalid of [NaN, Infinity, -Infinity])
    assert.deepEqual(zoomView(original, 0.35, invalid, limits), original);
  assert.deepEqual(navigationLimits(times), limits, 'navigation never becomes authored content');
  assert.deepEqual(times, [0, 0.15], 'timestamps remain unchanged');
});

test('long recordings expand navigation bounds while viewport extent does not become content', () => {
  const empty = navigationLimits([]);
  assert.equal(empty.maxSpan, 60);
  const short = navigationLimits([0, 12]);
  assert.equal(short.maxSpan, 60);
  let view = { start: 0, end: 10 };
  for (let index = 0; index < 100; index++) view = zoomView(view, 0.35, -4, short);
  assert.equal(navigationLimits([0, 12]).maxSpan, 60);
  const long = navigationLimits([0, 7200]);
  assert.equal(long.maxSpan, 28800);
  const longView = clampView({ start: 0, end: 7200 }, long);
  close(longView.end, 7200, 'recording beyond the initial ten seconds stays available');
});

test('explicit long and fine manual ranges are retained until the caller clears them', () => {
  const manual = { start: 3600, end: 7200 };
  const long = navigationLimits([0, 0.15], manual);
  assert.equal(long.maxSpan, 28800);
  assert.deepEqual(clampView(manual, long), manual);
  assert.equal(navigationLimits([0, 0.15], null).maxSpan, 60);
  const fine = { start: 0, end: 0.00001 };
  const precise = navigationLimits([0], fine);
  close(precise.minSpan, span(fine), 'explicit fine span');
  assert.deepEqual(clampView(fine, precise), fine);
});

test('far-offset content retains representable endpoints at the minimum zoom span', () => {
  const times = [1e16, 1e16 + 120];
  const limits = navigationLimits(times);
  let view = { start: times[0], end: times[1] };
  for (let index = 0; index < 100; index++) view = zoomView(view, 0.5, 4, limits);
  assertBounded(view, limits);
  assert.ok(span(view) >= 16, 'eight ULPs at this offset prevent collapsed endpoints');
});
