import { Icon } from './components';
import { t } from './i18n';
import './input-hints.css';
import { keyLabel, pointerParts, type InputBindings } from './input-bindings';

/** Guide and tooltips share the native status mouse masters and separate keycaps. */
export function KeyHint({ keys, label }: { keys: string[]; label?: string }) {
  return (
    <span className="input-chord" role="img" aria-label={label ?? keys.join('+')}>
      {keys.map((key, index) => (
        <span className="input-key-part" key={`${index}:${key}`}>
          {index > 0 && (
            <span className="input-plus" aria-hidden="true">
              +
            </span>
          )}
          <kbd>{key}</kbd>
        </span>
      ))}
    </span>
  );
}
export function MouseHint({
  button,
  twice = false,
  size = 24,
}: {
  button: 'left' | 'right' | 'wheel';
  twice?: boolean;
  size?: number;
}) {
  const label = t(
    button === 'left'
      ? 'Left mouse button'
      : button === 'right'
        ? 'Right mouse button'
        : 'Mouse wheel',
  );
  return (
    <span
      className="input-mouse"
      role="img"
      aria-label={`${label}${twice ? ` · ${t('Double-click')}` : ''}`}
    >
      <Icon name={`status-mouse-${button}`} size={size} />
      {twice && (
        <span className="input-repeat" aria-hidden="true">
          ×2
        </span>
      )}
    </span>
  );
}

export function PointerHint({
  chord,
  bindings,
  size,
}: {
  chord?: string;
  bindings: InputBindings;
  size?: number;
}) {
  if (!chord) return <span className="guide-note">{t('Unbound')}</span>;
  const { button, keys } = pointerParts(chord, bindings);
  if (keys.includes(null)) return <span className="guide-note">{t('Unbound')}</span>;
  return (
    <span className="input-gesture">
      {!!keys.length && (
        <>
          <KeyHint keys={keys.map(keyLabel)} />
          <span className="input-plus">+</span>
        </>
      )}
      <MouseHint
        button={button === 'middle' ? 'wheel' : (button as 'left' | 'right' | 'wheel')}
        size={size}
      />
    </span>
  );
}
