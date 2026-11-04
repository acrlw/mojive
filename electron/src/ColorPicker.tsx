import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Button, Icon, Numeric, Squircle } from './components';
import { useControlPopover } from './CustomControls';
import { t } from './i18n';
import './color-picker.css';
type RGBA = [number, number, number, number];
export function parseColor(text: string, normalized?: boolean, previousAlpha = 1): RGBA {
  text = text.trim();
  if (text.startsWith('#') || /^[\da-f]{6}([\da-f]{2})?$/i.test(text)) {
    let digits = text.startsWith('#') ? text.slice(1) : text;
    if (digits.length === 3 || digits.length === 4) digits = [...digits].map((x) => x + x).join('');
    if (!/^[\da-f]{6}([\da-f]{2})?$/i.test(digits))
      throw Error('Use #RGB, #RGBA, #RRGGBB or #RRGGBBAA.');
    return [
      parseInt(digits.slice(0, 2), 16) / 255,
      parseInt(digits.slice(2, 4), 16) / 255,
      parseInt(digits.slice(4, 6), 16) / 255,
      digits.length === 8 ? parseInt(digits.slice(6, 8), 16) / 255 : previousAlpha,
    ];
  }
  const fn = text.match(/^(rgba?(?:255|1)?)\s*\(\s*([\d.eE+,\s-]+)\s*\)$/),
    channels = fn ? fn[2] : text;
  if (!/^[\d.eE+,\s-]+$/.test(channels)) throw Error('Paste a hex color or RGB/RGBA values.');
  const values = channels.split(/[,\s]+/).map(Number);
  if (
    ![3, 4].includes(values.length) ||
    values.some((x) => !Number.isFinite(x)) ||
    (fn && values.length !== (fn[1].startsWith('rgba') ? 4 : 3))
  )
    throw Error('Use three or four finite color channels.');
  if (fn) normalized = fn[1].endsWith('1') ? true : fn[1].endsWith('255') ? false : undefined;
  const bytes = normalized === false || (normalized === undefined && values.some((x) => x > 1)),
    limit = bytes ? 255 : 1;
  if (values.some((x) => x < 0 || x > limit))
    throw Error(`Color channels must be between 0 and ${limit}.`);
  const out = values.map((x) => x / limit);
  if (fn?.[1] === 'rgba' && bytes && values[3] <= 1) out[3] = values[3];
  return [out[0], out[1], out[2], out[3] ?? previousAlpha];
}
export const colorHex = (rgba: RGBA, alpha = false) =>
  '#' +
  rgba
    .slice(0, alpha ? 4 : 3)
    .map((x) =>
      Math.round(Math.max(0, Math.min(1, x)) * 255)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')
    .toUpperCase();
export function rgbToHsv([r, g, b]: RGBA): [number, number, number] {
  const max = Math.max(r, g, b),
    min = Math.min(r, g, b),
    d = max - min;
  let h =
    d === 0 ? 0 : max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(((h / 6) % 1) + 1) % 1, max === 0 ? 0 : d / max, max];
}
export function hsvToRgb(h: number, s: number, v: number, a = 1): RGBA {
  const k = (n: number) => (n + h * 6) % 6,
    f = (n: number) => v - v * s * Math.max(0, Math.min(k(n), 4 - k(n), 1));
  return [f(5), f(3), f(1), a];
}
export function ColorPicker({
  value,
  onChange,
  onBegin,
  onCommit,
  onCancel,
  label = 'Color',
  alpha = false,
  disabled = false,
  className = '',
}: {
  value: string;
  onChange: (value: string) => void;
  onBegin?: () => void;
  onCommit?: () => void;
  onCancel?: () => void;
  label?: string;
  alpha?: boolean;
  disabled?: boolean;
  className?: string;
}) {
  const trigger = useRef<HTMLDivElement>(null),
    popup = useRef<HTMLDivElement>(null),
    transaction = useRef<string | null>(null);
  const [open, setOpen] = useState(false),
    [mode, setMode] = useState(0),
    [draft, setDraft] = useState(value),
    [error, setError] = useState('');
  const [color, setColor] = useState<RGBA>(() => parseColor(value)),
    [hsv, setHsv] = useState(() => rgbToHsv(parseColor(value)));
  const generation = useRef(0);
  const current = useRef(color),
    latest = useRef({ onChange, onCommit, onCancel, alpha });
  latest.current = { onChange, onCommit, onCancel, alpha };
  const finish = (cancel = false) => {
    const original = transaction.current;
    if (original === null) return;
    transaction.current = null;
    generation.current++;
    if (cancel) {
      latest.current.onChange(original);
      if (latest.current.onCancel) latest.current.onCancel();
      else latest.current.onCommit?.();
    } else latest.current.onCommit?.();
    setOpen(false);
  };
  useControlPopover(trigger, popup, open, () => finish());
  useEffect(
    () => () => {
      generation.current++;
      if (transaction.current !== null) {
        transaction.current = null;
        latest.current.onCommit?.();
      }
    },
    [],
  );
  const apply = (next: RGBA, keepHue = false) => {
    current.current = next;
    setColor(next);
    if (!keepHue) {
      const nextHsv = rgbToHsv(next);
      setHsv((old) => [nextHsv[1] === 0 ? old[0] : nextHsv[0], nextHsv[1], nextHsv[2]]);
    }
    const hex = colorHex(next, alpha);
    setDraft(hex);
    setError('');
    onChange(hex);
  };
  const show = () => {
    const rgba = parseColor(value);
    transaction.current = value;
    generation.current++;
    current.current = rgba;
    setColor(rgba);
    setHsv(rgbToHsv(rgba));
    setDraft(colorHex(rgba, alpha));
    setError('');
    onBegin?.();
    setOpen(true);
  };
  const fromText = () => {
    try {
      apply(parseColor(draft, mode === 0 ? undefined : mode === 2, current.current[3]));
      return true;
    } catch (error) {
      setError((error as Error).message);
      return false;
    }
  };
  const pick = (kind: 'sv' | 'hue' | 'alpha', event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect(),
      x = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
      y = Math.max(0, Math.min(1, (event.clientY - box.top) / box.height));
    const next: [number, number, number] =
      kind === 'sv'
        ? [hsv[0], x, 1 - y]
        : kind === 'hue'
          ? [Math.min(1 - 1e-7, y), hsv[1], hsv[2]]
          : hsv;
    setHsv(next);
    apply(hsvToRgb(...next, kind === 'alpha' ? 1 - y : current.current[3]), true);
  };
  const pickerEvents = (kind: 'sv' | 'hue' | 'alpha') => ({
    onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.focus();
      event.currentTarget.setPointerCapture(event.pointerId);
      pick(kind, event);
    },
    onPointerMove: (event: PointerEvent<HTMLDivElement>) => {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) pick(kind, event);
    },
  });
  const copy = async () => {
    try {
      const text =
        mode === 0
          ? colorHex(color, color[3] < 1)
          : `rgba${mode === 1 ? '255' : '1'}(${color
              .map((x) => (mode === 1 ? Math.round(x * 255) : Number(x.toPrecision(6))))
              .join(', ')})`;
      if (window.mojive) await window.mojive.writeClipboardText(text);
      else await navigator.clipboard.writeText(text);
      setError('');
    } catch {
      setError('Clipboard is unavailable.');
    }
  };
  const paste = async () => {
    const owner = generation.current;
    try {
      const text = window.mojive
        ? await window.mojive.readClipboardText()
        : await navigator.clipboard.readText();
      if (owner !== generation.current || transaction.current === null) return;
      const next = parseColor(text, mode === 0 ? undefined : mode === 2, current.current[3]);
      apply(next);
    } catch (error) {
      if (owner === generation.current && transaction.current !== null)
        setError((error as Error).message);
    }
  };
  return (
    <div className={`color-control ${className}`} ref={trigger}>
      <Button
        className="color-trigger"
        disabled={disabled}
        aria-label={t(label)}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => (open ? finish() : show())}
      >
        <span className="color-swatch">
          <span style={{ background: value }} />
        </span>
        <span>{value.toUpperCase()}</span>
        <Icon name="down" />
      </Button>
      {open && (
        <div
          ref={popup}
          popover="auto"
          className="control-popover color-popover"
          role="dialog"
          aria-label={t(label)}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.preventDefault();
              event.stopPropagation();
              finish(true);
            }
          }}
        >
          <Squircle radius={12} className="color-picker-surface">
            <div className="color-comparison">
              <Button
                className="color-comparison-item"
                title="Restore original color"
                onClick={() => apply(parseColor(transaction.current!))}
              >
                <span className="color-swatch">
                  <span style={{ background: transaction.current! }} />
                </span>
                {t('Original')}
              </Button>
              <span className="color-comparison-item">
                <span className="color-swatch">
                  <span style={{ background: colorHex(color, true) }} />
                </span>
                {t('Current')}
              </span>
              <span className="color-actions">
                <Button icon="copy" title="Copy" aria-label={t('Copy')} onClick={copy} />
                <Button icon="paste" title="Paste" aria-label={t('Paste')} onClick={paste} />
              </span>
            </div>
            <div className="color-picker-plane">
              <div
                className="color-sv"
                role="slider"
                tabIndex={0}
                aria-label={t('Saturation and value')}
                aria-valuetext={`${Math.round(hsv[1] * 100)}%, ${Math.round(hsv[2] * 100)}%`}
                style={{ backgroundColor: `hsl(${hsv[0] * 360} 100% 50%)` }}
                {...pickerEvents('sv')}
                onKeyDown={(event) => {
                  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key))
                    return;
                  event.preventDefault();
                  const next: [number, number, number] = [
                    hsv[0],
                    Math.max(
                      0,
                      Math.min(
                        1,
                        hsv[1] +
                          (event.key === 'ArrowRight'
                            ? 0.01
                            : event.key === 'ArrowLeft'
                              ? -0.01
                              : 0),
                      ),
                    ),
                    Math.max(
                      0,
                      Math.min(
                        1,
                        hsv[2] +
                          (event.key === 'ArrowUp' ? 0.01 : event.key === 'ArrowDown' ? -0.01 : 0),
                      ),
                    ),
                  ];
                  setHsv(next);
                  apply(hsvToRgb(...next, color[3]), true);
                }}
              >
                <span
                  style={{
                    left: `${hsv[1] * 100}%`,
                    top: `${(1 - hsv[2]) * 100}%`,
                    background: colorHex(color),
                  }}
                />
              </div>
              <div
                className="color-hue color-bar"
                role="slider"
                tabIndex={0}
                aria-label={t('Hue')}
                aria-valuemin={0}
                aria-valuemax={360}
                aria-valuenow={Math.round(hsv[0] * 360)}
                {...pickerEvents('hue')}
                onKeyDown={(event) => {
                  if (!['ArrowDown', 'ArrowUp'].includes(event.key)) return;
                  event.preventDefault();
                  const next: [number, number, number] = [
                    Math.max(
                      0,
                      Math.min(1 - 1e-7, hsv[0] + (event.key === 'ArrowDown' ? 1 : -1) / 360),
                    ),
                    hsv[1],
                    hsv[2],
                  ];
                  setHsv(next);
                  apply(hsvToRgb(...next, color[3]), true);
                }}
              >
                <span style={{ top: `${hsv[0] * 100}%` }} />
              </div>
              {alpha && (
                <div
                  className="color-alpha color-bar"
                  role="slider"
                  tabIndex={0}
                  aria-label={t('Alpha')}
                  aria-valuemin={0}
                  aria-valuemax={1}
                  aria-valuenow={color[3]}
                  {...pickerEvents('alpha')}
                  onKeyDown={(event) => {
                    if (!['ArrowUp', 'ArrowDown'].includes(event.key)) return;
                    event.preventDefault();
                    apply([
                      ...color.slice(0, 3),
                      Math.max(0, Math.min(1, color[3] + (event.key === 'ArrowUp' ? 0.01 : -0.01))),
                    ] as RGBA);
                  }}
                >
                  <div
                    style={{
                      background: `linear-gradient(to bottom,${colorHex(color)},transparent)`,
                    }}
                  />
                  <span style={{ top: `${(1 - color[3]) * 100}%` }} />
                </div>
              )}
            </div>
            <Squircle
              radius={4}
              className="color-format"
              role="group"
              aria-label={t('Color format')}
            >
              {['Hex', `${alpha ? 'RGBA' : 'RGB'} 255`, `${alpha ? 'RGBA' : 'RGB'} 1`].map(
                (name, index) => (
                  <Button
                    active={mode === index}
                    key={name}
                    onClick={() => {
                      setMode(index);
                      setDraft(colorHex(color, alpha));
                      setError('');
                    }}
                  >
                    {name}
                  </Button>
                ),
              )}
            </Squircle>
            {mode === 0 ? (
              <input
                className="color-hex"
                aria-label={t('Hex color')}
                value={draft}
                aria-invalid={!!error}
                onChange={(event) => setDraft(event.target.value)}
                onFocus={(event) => event.currentTarget.select()}
                onBlur={fromText}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    event.stopPropagation();
                    fromText();
                  }
                }}
              />
            ) : (
              <div className="color-channel-fields">
                {color.slice(0, alpha ? 4 : 3).map((channel, index) => (
                  <div className="color-channel" key={index}>
                    <label>{'RGBA'[index]}</label>
                    <Numeric
                      label={'RGBA'[index]}
                      value={mode === 1 ? channel * 255 : channel}
                      precision={mode === 1 ? 0 : 3}
                      min={0}
                      max={mode === 1 ? 255 : 1}
                      integer={mode === 1}
                      step={mode === 1 ? 1 : 0.001}
                      onChange={(number) => {
                        const next = [...current.current] as RGBA;
                        next[index] = number / (mode === 1 ? 255 : 1);
                        apply(next);
                      }}
                    />
                  </div>
                ))}
              </div>
            )}
            {error && (
              <div className="color-error" role="alert">
                {t(error)}
              </div>
            )}
            <div className="color-footer">
              <Button variant="secondary" onClick={() => finish(true)}>
                {t('Cancel')}
              </Button>
              <Button variant="primary" disabled={!!error} onClick={() => finish()}>
                {t('Done')}
              </Button>
            </div>
          </Squircle>
        </div>
      )}
    </div>
  );
}
