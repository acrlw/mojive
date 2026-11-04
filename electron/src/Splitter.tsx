import { useEffect, useRef } from 'react';
import { t } from './i18n';
/** One pointer owns a resize until release; Escape/blur restore the original size. */
export function Splitter({
  axis,
  value,
  onChange,
  min,
  max,
  reverse = false,
  label,
  style,
  onReset,
}: {
  axis: 'x' | 'y';
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  reverse?: boolean;
  label: string;
  style: React.CSSProperties;
  onReset: () => void;
}) {
  const drag = useRef<{
    pointer: number;
    start: number;
    value: number;
    target: HTMLElement;
  } | null>(null);
  const change = useRef(onChange);
  change.current = onChange;
  function finish(cancel = false) {
    const d = drag.current;
    if (!d) return;
    drag.current = null;
    if (cancel) change.current(d.value);
    if (d.target.hasPointerCapture(d.pointer)) d.target.releasePointerCapture(d.pointer);
    document.body.classList.remove('resizing-x', 'resizing-y');
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && drag.current) {
        e.preventDefault();
        e.stopImmediatePropagation();
        finish(true);
      }
    };
    const blur = () => finish(true);
    window.addEventListener('keydown', key, true);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', key, true);
      window.removeEventListener('blur', blur);
      finish();
    };
  }, []);
  return (
    <div
      role="separator"
      aria-label={t(label)}
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={Math.round(max)}
      tabIndex={0}
      className={`panel-splitter splitter-${axis}`}
      data-resize-edge={reverse ? 'start' : 'end'}
      style={style}
      onDoubleClick={onReset}
      onKeyDown={(e) => {
        const delta = ['ArrowRight', 'ArrowDown'].includes(e.key)
          ? 10
          : ['ArrowLeft', 'ArrowUp'].includes(e.key)
            ? -10
            : 0;
        if (delta) {
          e.preventDefault();
          onChange(Math.max(min, Math.min(max, value + delta * (reverse ? -1 : 1))));
        }
      }}
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = {
          pointer: e.pointerId,
          start: axis === 'x' ? e.clientX : e.clientY,
          value,
          target: e.currentTarget,
        };
        document.body.classList.add(`resizing-${axis}`);
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d || d.pointer !== e.pointerId) return;
        const zoom =
          Number(
            getComputedStyle(e.currentTarget.closest('.app-shell') ?? document.documentElement)
              .zoom,
          ) || 1;
        onChange(
          Math.max(
            min,
            Math.min(
              max,
              d.value +
                (((axis === 'x' ? e.clientX : e.clientY) - d.start) * (reverse ? -1 : 1)) / zoom,
            ),
          ),
        );
      }}
      onPointerUp={() => finish()}
      onPointerCancel={() => finish(true)}
      onLostPointerCapture={() => finish(true)}
    />
  );
}
