import { useId, useLayoutEffect, useRef, useState } from 'react';
import { Button } from './components';
import { t, useLanguage } from './i18n';
import './confirm-scene.css';
export interface ConfirmSceneDialogProps {
  filename: string;
  reason: 'open' | 'close';
  onCancel: () => void;
  onDiscard: () => void;
  /** Save and resolve the pending parent action; false retains the confirmation. */
  onSave: () => Promise<boolean>;
}
export function ConfirmSceneDialog({
  filename,
  reason,
  onCancel,
  onDiscard,
  onSave,
}: ConfirmSceneDialogProps) {
  useLanguage();
  const dialog = useRef<HTMLDialogElement>(null),
    mounted = useRef(false),
    id = useId();
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const focusCancel = () =>
    dialog.current
      ?.querySelector<HTMLButtonElement>('[data-confirm-cancel]')
      ?.focus({ preventScroll: true });
  useLayoutEffect(() => {
    const element = dialog.current;
    if (!element) return;
    mounted.current = true;
    let zoom = 1;
    for (let ancestor: HTMLElement | null = element; ancestor; ancestor = ancestor.parentElement)
      zoom *= parseFloat(getComputedStyle(ancestor).zoom) || 1;
    element.style.setProperty('--confirm-zoom', String(zoom));
    element.showModal();
    focusCancel();
    return () => {
      mounted.current = false;
      if (element.open) element.close();
    };
  }, []);
  const save = async () => {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      if (await onSave()) {
        if (mounted.current) dialog.current?.close();
      }
    } catch (error) {
      if (mounted.current) setError(error instanceof Error ? error.message : String(error));
    } finally {
      if (mounted.current) {
        setBusy(false);
        requestAnimationFrame(focusCancel);
      }
    }
  };
  const message = t(
    reason === 'close'
      ? 'Save changes to “{filename}” before closing?'
      : 'Save changes to “{filename}” before opening another scene?',
  ).replace('{filename}', filename);
  return (
    <dialog
      ref={dialog}
      role="alertdialog"
      className="confirm-scene-dialog"
      aria-labelledby={id + '-title'}
      aria-describedby={id + '-message'}
      aria-busy={busy || undefined}
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onCancel();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Escape' || (event.metaKey && event.key === '.')) {
          event.preventDefault();
          if (!busy) onCancel();
        }
        if (event.key === 'Tab') {
          const buttons = [
            ...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)'),
          ];
          if (!buttons.length) {
            event.preventDefault();
            return;
          }
          const first = buttons[0],
            last = buttons.at(-1)!;
          if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
          }
        }
      }}
    >
      <div className="confirm-scene-surface">
        <h2 id={id + '-title'}>{t('Unsaved changes')}</h2>
        <p id={id + '-message'}>{message}</p>
        {error && (
          <p role="alert" className="confirm-scene-error">
            {t(error)}
          </p>
        )}
        <div className="confirm-scene-actions">
          <Button variant="secondary" data-confirm-cancel disabled={busy} onClick={onCancel}>
            {t('Cancel')}
          </Button>
          <Button variant="secondary" disabled={busy} onClick={onDiscard}>
            {t("Don't Save")}
          </Button>
          <Button
            variant="primary"
            className="confirm-scene-save"
            disabled={busy}
            onClick={() => void save()}
          >
            {t(busy ? 'Saving…' : 'Save')}
          </Button>
        </div>
      </div>
    </dialog>
  );
}
