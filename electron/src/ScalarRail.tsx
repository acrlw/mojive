import { useEffect, useRef, useState } from 'react';
import './joint-panel.css';

interface ScalarRailProps {
  value: number;
  min: number;
  max: number;
  label: string;
  valueText?: string;
  referenceValue?: number;
  disabled?: boolean;
  unboundedSpan?: number;
  onBegin: () => void;
  onCommit: () => void;
  onCancel?: () => void;
  onChange: (value: number) => void;
}
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/** One pointer/key owner, including outside release and Escape at a limit. */
export function ScalarRail({
  value, min: lo, max: hi, label, valueText, referenceValue, disabled = false, unboundedSpan = 1,
  onBegin,
  onCommit,
  onCancel,
  onChange,
}: ScalarRailProps) {
  const drag = useRef<{
    initial: number;
    x: number;
    offset: number;
    pointer: number | null;
  } | null>(null);
  const element = useRef<HTMLDivElement>(null);
  const [initial, setInitial] = useState<number | null>(null);
  const callbacks = useRef({ onBegin, onCommit, onCancel, onChange });
  callbacks.current = { onBegin, onCommit, onCancel, onChange };
  const bounded = Number.isFinite(lo) && Number.isFinite(hi) && hi > lo;
  const fraction = (value: number) =>
    bounded ? clamp((value - lo) / (hi - lo || 1), 0, 1) : 0.5;
  const current = fraction(value);
  const reference = bounded && referenceValue !== undefined && Number.isFinite(referenceValue) &&
    referenceValue >= lo && referenceValue <= hi ? fraction(referenceValue) : null;
  const finish = (cancel = false) => {
    const state = drag.current;
    if (!state) return;
    drag.current = null;
    setInitial(null);
    if (state.pointer !== null && element.current?.hasPointerCapture(state.pointer))
      element.current.releasePointerCapture(state.pointer);
    if (cancel) {
      callbacks.current.onChange(state.initial);
      (callbacks.current.onCancel ?? callbacks.current.onCommit)();
    } else callbacks.current.onCommit();
  };
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !drag.current) return;
      event.preventDefault();
      event.stopPropagation();
      finish(true);
    };
    const blur = () => finish(true);
    window.addEventListener('keydown', escape, true);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', escape, true);
      window.removeEventListener('blur', blur);
    };
  }, []);
  const pointerValue = (x: number, precision = false) => {
    const state = drag.current,
      rect = element.current!.getBoundingClientRect();
    if (!state) return;
    const span = bounded ? hi - lo : unboundedSpan;
    const scale = rect.width / element.current!.offsetWidth,
      inset = 8 * scale,
      travel = Math.max(1, rect.width - 16 * scale);
    const next =
      bounded && !precision
        ? lo + ((x - state.offset - rect.left - inset) / travel) * span
        : state.initial + ((x - state.x) / travel) * span * (precision ? 0.1 : 1);
    onChange(bounded ? clamp(next, lo, hi) : next);
  };
  return (
    <div
      ref={element}
      className={`joint-rail ${initial !== null ? 'held' : ''} ${bounded ? 'bounded' : 'unbounded'}`}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={label}
      aria-disabled={!!disabled}
      aria-valuemin={bounded ? lo : undefined}
      aria-valuemax={bounded ? hi : undefined}
      aria-valuenow={value}
      aria-valuetext={valueText}
      onPointerDown={(event) => {
        if (event.button !== 0 || disabled) return;
        event.preventDefault();
        event.currentTarget.focus();
        const rect = event.currentTarget.getBoundingClientRect(),
          scale = rect.width / event.currentTarget.offsetWidth;
        const center = rect.left + 8 * scale + fraction(value) * (rect.width - 16 * scale);
        const offset = Math.abs(event.clientX - center) <= 7 * scale ? event.clientX - center : 0;
        drag.current = { initial: value, x: event.clientX, offset, pointer: event.pointerId };
        setInitial(value);
        onBegin();
        event.currentTarget.setPointerCapture(event.pointerId);
        pointerValue(event.clientX, event.altKey);
      }}
      onPointerMove={(event) => {
        if (drag.current?.pointer === event.pointerId) pointerValue(event.clientX, event.altKey);
      }}
      onPointerUp={(event) => {
        if (drag.current?.pointer !== event.pointerId) return;
        finish();
      }}
      onPointerCancel={() => finish(true)}
      onLostPointerCapture={() => finish(true)}
      onBlur={() => {
        if (drag.current?.pointer === null) finish();
      }}
      onKeyDown={(event) => {
        if (
          disabled ||
          !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        if (!drag.current) {
          drag.current = { initial: value, x: 0, offset: 0, pointer: null };
          setInitial(value);
          onBegin();
        }
        const step =
          ((bounded ? hi - lo : unboundedSpan) / 200) *
          (event.shiftKey ? 10 : event.altKey ? 0.1 : 1);
        const next =
          event.key === 'Home' && bounded
            ? lo
            : event.key === 'End' && bounded
              ? hi
              : value + step * (['ArrowLeft', 'ArrowDown'].includes(event.key) ? -1 : 1);
        onChange(bounded ? clamp(next, lo, hi) : next);
      }}
      onKeyUp={(event) => {
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))
          finish();
      }}
    >
      <div className="joint-rail-track">
        <span className="joint-rail-background" aria-hidden="true" />
        <span className="joint-rail-fill" style={{ width: `calc(${current * 100}% + 4px)` }} />
        {reference !== null && <span className="joint-rail-reference" aria-hidden="true"
          data-value={referenceValue} style={{ left: `${reference * 100}%` }} />}
        <span className="joint-rail-thumb" style={{ left: `${current * 100}%` }} />
      </div>
    </div>
  );
}
