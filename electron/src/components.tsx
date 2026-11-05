import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ComponentProps,
  type CSSProperties,
  type HTMLAttributes,
  type InputHTMLAttributes,
  type RefObject,
} from 'react';
import { flushSync } from 'react-dom';
import { icons } from './icons';
import { t, message } from './i18n';
import './controls.css';
import './smooth-corners';
export { Select, Switch, SwitchRow } from './CustomControls';
export { ColorPicker } from './ColorPicker';
export function Icon({
  name,
  size = 20,
  className = '',
}: {
  name: string;
  size?: number;
  className?: string;
}) {
  const svg = icons[name];
  if (!svg) throw new Error(`Unknown Mojive icon: ${name}`);
  return (
    <span
      data-icon={name}
      className={`icon ${className}`}
      style={{ width: size, height: size }}
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
/** Declare the authored radius; the shared SVG painter owns the continuous contour. */
export function Squircle({
  radius = 14,
  children,
  className = '',
  style,
  ...props
}: HTMLAttributes<HTMLDivElement> & { radius?: number }) {
  return (
    <div
      {...props}
      className={className}
      style={
        {
          '--surface-radius': `${radius}px`,
          borderRadius: radius,
          ...style,
        } as CSSProperties
      }
    >
      {children}
    </div>
  );
}
/** Connected choices share one silhouette and one state palette across panels. */
export function Segmented({ className = '', ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div role="group" {...props} className={`segmented ${className}`} />;
}
export function Button({
  icon,
  children,
  active = false,
  activityIndicator = false,
  variant,
  title,
  hintText,
  hintKeys,
  style,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: string;
  active?: boolean;
  activityIndicator?: boolean;
  variant?: 'primary' | 'secondary';
  hintText?: string;
  hintKeys?: string[];
}) {
  const iconOnly = children == null || children === false || children === '';
  const hint = title ?? (iconOnly ? props['aria-label'] : undefined);
  return (
    <button
      {...props}
      data-button-variant={
        variant ??
        (props.className?.split(/\s+/).includes('primary-button') ? 'primary' : undefined)
      }
      aria-label={props['aria-label'] ?? (hint ? t(hint) : undefined)}
      data-tooltip={hint ? t(hint) : undefined}
      data-tooltip-caption={hintText ? t(hintText) : undefined}
      data-tooltip-keys={hintKeys ? JSON.stringify(hintKeys) : undefined}
      className={`button ${active ? 'active' : ''} ${!children ? 'icon-button' : ''} ${props.className ?? ''}`}
      style={style}
    >
      {icon && <Icon name={icon} />} {typeof children === 'string' ? t(children) : children}
      {activityIndicator && (
        <span className="activity-indicator" aria-hidden="true">
          <span />
        </span>
      )}
    </button>
  );
}
export function SearchField({
  label,
  value,
  onChange,
  onClear,
  clearLabel,
  className = '',
  inputRef,
  placeholder = `${t('Search')}…`,
  ...inputProps
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'className' | 'size'> & {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onClear?: () => void;
  clearLabel?: string;
  className?: string;
  inputRef?: RefObject<HTMLInputElement | null>;
}) {
  const localInput = useRef<HTMLInputElement>(null);
  const input = inputRef ?? localInput;
  return (
    <div className={`search-field ${className}`} data-capsule-layout="">
      <Icon name="search" size={16} />
      <input
        {...inputProps}
        ref={input}
        aria-label={label}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {value && (
        <Button
          type="button"
          icon="close"
          aria-label={clearLabel ?? t('Clear search')}
          onClick={() => {
            if (onClear) onClear();
            else onChange('');
            input.current?.focus();
          }}
        />
      )}
    </div>
  );
}
export function Section({
  title,
  children,
  action,
  initial = true,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  initial?: boolean;
}) {
  const [open, setOpen] = useState(initial);
  const content = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const element = content.current;
    if (!element?.closest('.inspector')) return;
    const rows = [...element.querySelectorAll<HTMLElement>('.vector-group')].filter(
      (row) => row.closest('.section-content') === element,
    );
    if (!rows.length) return;
    const grid = element.closest('.inspector-content')!;
    const context = document.createElement('canvas').getContext('2d')!;
    const measure = () => {
      // Reflow from a stable signed three-decimal budget, never the current values.
      const scale = element.getBoundingClientRect().width / element.offsetWidth;
      let labelWidth = parseFloat(getComputedStyle(grid).gridTemplateColumns) || 0,
        fieldsWidth = 0;
      for (const row of rows) {
        const label = row.querySelector('label')!,
          range = document.createRange();
        range.selectNodeContents(label);
        labelWidth = Math.max(labelWidth, range.getBoundingClientRect().width / scale);
        const input = row.querySelector('input')!,
          style = getComputedStyle(input);
        context.font = `${style.fontSize} ${style.fontFamily}`;
        const axis = row.querySelector('.axis')!;
        const fieldWidth =
          context.measureText('-999.999').width +
          parseFloat(style.paddingLeft) +
          parseFloat(style.paddingRight) +
          axis.getBoundingClientRect().width / scale;
        fieldsWidth = Math.max(fieldsWidth, fieldWidth * 3 + 8);
      }
      const width = element.clientWidth;
      element.dataset.vectorLabels = width >= labelWidth + 12 + fieldsWidth ? 'inline' : 'above';
      element.dataset.vectorFields = width >= fieldsWidth ? 'row' : 'stack';
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    for (const row of rows) observer.observe(row.querySelector('label')!);
    // Another section can change the shared column without resizing this section's bounds.
    const structure = new MutationObserver(measure);
    structure.observe(grid, { childList: true, subtree: true });
    document.fonts.addEventListener('loadingdone', measure);
    measure();
    return () => {
      // Disconnect in the DOM commit, before observers can measure removed rows.
      observer.disconnect();
      structure.disconnect();
      document.fonts.removeEventListener('loadingdone', measure);
    };
  }, [open, children]);
  return (
    <section className="section">
      <div className="section-heading">
        <button className="section-title" onClick={() => setOpen(!open)} aria-expanded={open}>
          {t(title)}
        </button>
        {action && <div className="section-actions">{action}</div>}
        <Button
          className="section-disclosure"
          icon={open ? 'down' : 'right'}
          title={`${t(open ? 'Collapse' : 'Expand')} ${t(title)}`}
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        />
      </div>
      {open && (
        <div className="section-content" ref={content}>
          {children}
        </div>
      )}
    </section>
  );
}
/** Native %.3f caption semantics, including normalization of negative zero. */
export function numericCaption(value: number, precision = 3) {
  if (!Number.isFinite(value)) return '0';
  const text = value.toFixed(precision);
  return Number(text) === 0 ? (0).toFixed(precision) : text;
}
export function Numeric({
  value,
  onChange,
  onBegin,
  onCommit,
  onCancel,
  label,
  step = 0.01,
  disabled = false,
  unit,
  precision = 3,
  formatValue,
  min = -Infinity,
  max = Infinity,
  integer = false,
}: {
  value: number;
  onChange: (n: number) => void;
  onBegin?: () => void;
  onCommit?: () => void;
  onCancel?: () => void;
  label: string;
  step?: number;
  disabled?: boolean;
  unit?: string;
  precision?: number;
  formatValue?: (value: number) => string;
  min?: number;
  max?: number;
  integer?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null),
    frame = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<string | null>(null),
    [dragging, setDragging] = useState(false),
    [pressed, setPressed] = useState(false);
  const edit = useRef<{
    initial: number;
    current: number;
    dragValue: number;
    pointer: number | null;
    x: number;
    startX: number;
    dragged: boolean;
    begun: boolean;
  } | null>(null);
  const latest = useRef({ value, onChange, onBegin, onCommit, onCancel, min, max, integer });
  latest.current = { value, onChange, onBegin, onCommit, onCancel, min, max, integer };
  const invalid = draft !== null && (!draft.trim() || !Number.isFinite(Number(draft)));
  const begin = (notify = true) => {
    edit.current ??= {
      initial: latest.current.value,
      current: latest.current.value,
      dragValue: latest.current.value,
      pointer: null,
      x: 0,
      startX: 0,
      dragged: false,
      begun: false,
    };
    if (notify && !edit.current.begun) {
      edit.current.begun = true;
      latest.current.onBegin?.();
    }
  };
  const change = (number: number) => {
    const options = latest.current;
    number = Math.max(
      options.min,
      Math.min(options.max, options.integer ? Math.round(number) : number),
    );
    if (edit.current) edit.current.current = number;
    options.onChange(number);
    return number;
  };
  const finish = (cancel = false) => {
    const state = edit.current;
    if (!state) return;
    edit.current = null;
    setDraft(null);
    setDragging(false);
    setPressed(false);
    if (state.pointer !== null && frame.current?.hasPointerCapture(state.pointer))
      frame.current.releasePointerCapture(state.pointer);
    if (!state.begun) return;
    if (cancel) {
      latest.current.onChange(state.initial);
      if (latest.current.onCancel) latest.current.onCancel();
      else latest.current.onCommit?.();
    } else latest.current.onCommit?.();
  };
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && edit.current?.pointer !== null && edit.current) {
        event.preventDefault();
        event.stopPropagation();
        finish(true);
      }
    };
    const blur = () => {
      if (edit.current?.pointer != null) finish(true);
    };
    window.addEventListener('keydown', escape, true);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', escape, true);
      window.removeEventListener('blur', blur);
    };
  }, []);
  const textEntry = () => {
    begin();
    // Commit the displayed draft before native focus/selection; deferred selection
    // can overwrite the user's first character or an accessibility fill action.
    flushSync(() => setDraft(String(edit.current!.current)));
    input.current?.focus();
    input.current?.select();
  };
  return (
    <div
      ref={frame}
      tabIndex={-1}
      className={`number-field ${pressed ? 'pressed' : ''} ${dragging ? 'scrubbing' : ''} ${draft !== null ? 'editing' : ''}`}
      onPointerDown={(event) => {
        if (event.button !== 0 || disabled || draft !== null) return;
        event.preventDefault();
        frame.current?.focus({ preventScroll: true });
        begin(false);
        setPressed(true);
        Object.assign(edit.current!, {
          pointer: event.pointerId,
          x: event.clientX,
          startX: event.clientX,
        });
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const state = edit.current;
        if (!state || state.pointer !== event.pointerId) return;
        if (!state.dragged && Math.abs(event.clientX - state.startX) < 3) return;
        if (!state.dragged) {
          state.dragged = true;
          begin();
          setDragging(true);
        }
        const scale = (event.shiftKey ? 10 : 1) * (event.altKey ? 0.01 : 1);
        // Preserve fractional motion for integer fields; only the committed value is rounded.
        state.dragValue = Math.max(
          latest.current.min,
          Math.min(latest.current.max, state.dragValue + (event.clientX - state.x) * step * scale),
        );
        change(state.dragValue);
        state.x = event.clientX;
      }}
      onPointerUp={(event) => {
        const state = edit.current;
        if (!state || state.pointer !== event.pointerId) return;
        finish();
      }}
      onDoubleClick={(event) => {
        if (disabled || draft !== null || event.button !== 0) return;
        event.preventDefault();
        textEntry();
      }}
      onPointerCancel={() => finish(true)}
      onLostPointerCapture={() => {
        if (edit.current?.pointer != null) finish(true);
      }}
    >
      <input
        ref={input}
        aria-label={t(label)}
        aria-invalid={invalid || undefined}
        type="text"
        inputMode={integer ? 'numeric' : 'decimal'}
        disabled={disabled}
        value={
          draft ??
          (formatValue ? formatValue(value) : numericCaption(value, integer ? 0 : precision))
        }
        onFocus={() => {
          if (edit.current?.pointer == null && draft === null) textEntry();
        }}
        onChange={(event) => {
          begin();
          setDraft(event.target.value);
          if (event.target.value.trim() && Number.isFinite(Number(event.target.value)))
            change(Number(event.target.value));
        }}
        onBlur={() => {
          if (edit.current?.pointer === null) finish(invalid);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            finish(true);
            event.currentTarget.blur();
          }
          if (event.key === 'Enter') {
            event.preventDefault();
            event.stopPropagation();
            if (!invalid) {
              finish();
              event.currentTarget.blur();
            }
          }
          if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
            event.preventDefault();
            event.stopPropagation();
            begin();
            const number = change(
              edit.current!.current +
                step *
                  (event.key === 'ArrowUp' ? 1 : -1) *
                  (event.shiftKey ? 10 : 1) *
                  (event.altKey ? 0.01 : 1),
            );
            setDraft(String(Number(number.toPrecision(12))));
          }
        }}
      />
      {unit && <span className="unit">{unit}</span>}
    </div>
  );
}
export function NumericChannel({
  badge,
  tone,
  onReset,
  ...numeric
}: ComponentProps<typeof Numeric> & {
  badge: string;
  tone: 'x' | 'y' | 'z' | 'alpha';
  onReset: () => void;
}) {
  return (
    <div className="axis-field">
      <Button
        className={`axis axis-${tone}`}
        title="Reset"
        aria-label={`${t('Reset')} ${t(numeric.label)}`}
        onClick={onReset}
      >
        {badge}
      </Button>
      <Numeric {...numeric} />
    </div>
  );
}
export function VectorFields({
  label,
  values,
  onChange,
  onBegin,
  onCommit,
  unit,
  resetValues,
}: {
  label: string;
  values: number[];
  onChange: (axis: number, n: number) => void;
  onBegin: () => void;
  onCommit: () => void;
  unit?: string;
  resetValues?: readonly number[];
}) {
  return (
    <div className="vector-group">
      <label>
        {t(label)}
        {unit && <span className="vector-unit">({unit})</span>}
      </label>
      <div className="vector-fields">
        {['X', 'Y', 'Z'].map((axis, i) => (
          <NumericChannel
            key={axis}
            badge={axis}
            tone={['x', 'y', 'z'][i] as 'x' | 'y' | 'z'}
            label={t('{{label}} {{axis}}', { label: message(label), axis })}
            value={values[i]}
            onChange={(v) => onChange(i, v)}
            onBegin={onBegin}
            onCommit={onCommit}
            onReset={() => {
              onBegin();
              onChange(i, resetValues?.[i] ?? 0);
              onCommit();
            }}
          />
        ))}
      </div>
    </div>
  );
}
export function Empty({
  icon,
  title,
  children,
  action,
}: {
  icon?: string;
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty">
      {icon && <Icon name={icon} size={32} />}
      <strong>{t(title)}</strong>
      {children && <div className="empty-detail">{children}</div>}
      {action && <div className="empty-actions">{action}</div>}
    </div>
  );
}
export const cssVars = (values: Record<string, string | number>) => values as CSSProperties;
