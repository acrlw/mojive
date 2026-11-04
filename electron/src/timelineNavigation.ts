/** View navigation mirrors python/interaction/timeline.py; it never edits content time. */
export interface TimeRange {
  start: number;
  end: number;
}
export interface ViewLimits {
  lower: number;
  upper: number;
  minSpan: number;
  maxSpan: number;
}
const MAX_SPAN = Number.MAX_VALUE / 4;
const minimumSpan = (start: number, end: number) =>
  Math.max(1e-6, 8 * Number.EPSILON * Math.max(Math.abs(start), Math.abs(end)));
const valid = ({ start, end }: TimeRange) =>
  Number.isFinite(start) &&
  Number.isFinite(end) &&
  Number.isFinite(end - start) &&
  end - start >= 1e-6;

export function navigationLimits(times: readonly number[], manual?: TimeRange | null): ViewLimits {
  let lo = Infinity,
    hi = -Infinity;
  for (const value of times)
    if (Number.isFinite(value)) {
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
  if (manual && valid(manual)) {
    lo = Math.min(lo, manual.start);
    hi = Math.max(hi, manual.end);
  }
  if (lo === Infinity) lo = hi = 0;
  const maxSpan = Math.max(60, Math.min(MAX_SPAN / 4, hi - lo) * 4, minimumSpan(lo, hi));
  return {
    lower: Math.max(-Number.MAX_VALUE, lo - maxSpan),
    upper: Math.min(Number.MAX_VALUE, hi + maxSpan),
    minSpan: manual && valid(manual) ? Math.min(0.001, manual.end - manual.start) : 0.001,
    maxSpan,
  };
}

export function clampView(range: TimeRange, limits: ViewLimits): TimeRange {
  let { start, end } = valid(range) ? range : { start: limits.lower, end: limits.upper };
  const originalSpan = end - start;
  const span = Math.min(
    limits.maxSpan,
    Math.max(limits.minSpan, minimumSpan(start, end), originalSpan),
  );
  // Reconstructing endpoints may exceed the bound by a few ULPs. This is
  // saturation, not a runaway view that needs recentering.
  if (originalSpan > limits.maxSpan + minimumSpan(start, end) * 1e-3)
    start = limits.lower / 2 + limits.upper / 2 - span / 2;
  else if (span !== originalSpan) start = start / 2 + end / 2 - span / 2;
  start = Math.max(limits.lower, Math.min(start, limits.upper - span));
  return { start, end: start + span };
}

export function panView(range: TimeRange, shift: number, limits: ViewLimits): TimeRange {
  const current = clampView(range, limits);
  if (!Number.isFinite(shift)) return current;
  const span = current.end - current.start;
  const start = Math.max(limits.lower, Math.min(current.start + shift, limits.upper - span));
  return { start, end: start + span };
}

export function zoomView(
  range: TimeRange,
  ratio: number,
  wheel: number,
  limits: ViewLimits,
): TimeRange {
  const current = clampView(range, limits);
  if (!Number.isFinite(ratio) || !Number.isFinite(wheel)) return current;
  ratio = Math.max(0, Math.min(1, ratio));
  const span = current.end - current.start,
    anchor = current.start + ratio * span;
  const next = Math.min(
    limits.maxSpan,
    Math.max(
      limits.minSpan,
      minimumSpan(current.start, current.end),
      span * Math.exp(-Math.max(-4, Math.min(4, wheel)) * 0.18),
    ),
  );
  const start = anchor - ratio * next;
  return clampView({ start, end: start + next }, limits);
}

/** Browser pixels, lines and pages are different units, unlike GLFW wheel offsets. */
export function normalizeWheel(delta: number, deltaMode: number): number {
  if (!Number.isFinite(delta)) return 0;
  return Math.max(-4, Math.min(4, delta / (deltaMode === 1 ? 3 : deltaMode === 2 ? 1 : 100)));
}

/** Fit rendered timeline content, independently from the broader navigation bounds. */
export function fittedView(times: readonly number[], fallback = 0): TimeRange {
  let lo = Infinity,
    hi = -Infinity;
  for (const value of times)
    if (Number.isFinite(value)) {
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
  if (lo === Infinity) {
    const center = Number.isFinite(fallback) ? fallback : 0;
    return { start: Math.min(0, center), end: Math.max(1, center + 0.5) };
  }
  if (hi - lo < 1e-9) {
    const context = Math.max(1, Math.abs(lo) * 0.1);
    return {
      start: lo >= 0 ? Math.max(0, lo - context * 0.5) : lo - context * 0.5,
      end: hi + context * 0.5,
    };
  }
  const padding = (hi - lo) * 0.08;
  return { start: lo >= 0 ? Math.max(0, lo - padding) : lo - padding, end: hi + padding };
}
