import { useEffect, useRef } from 'react';
import { Button, Icon } from './components';
import { t, useLanguage } from './i18n';
import './operation-notice.css';

export interface OperationNoticeData {
  /** A publication identity, even when the new message repeats the previous text. */
  id: string | number;
  message: string;
  level?: 'info' | 'warning' | 'error';
  /** The document owner must recheck the committed history identity on invocation. */
  action?: { label: string; onClick: () => void };
}

export function OperationNotice({
  notice,
  onDismiss,
  durationMs = 4500,
}: {
  notice: OperationNoticeData;
  onDismiss: (id: OperationNoticeData['id']) => void;
  durationMs?: number | null;
}) {
  useLanguage();
  const root = useRef<HTMLDivElement>(null),
    timeout = useRef<ReturnType<typeof setTimeout> | null>(null),
    remaining = useRef(durationMs ?? 0),
    deadline = useRef(0),
    hovered = useRef(false),
    focused = useRef(false),
    dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  const stop = () => {
    if (timeout.current === null) return;
    clearTimeout(timeout.current);
    timeout.current = null;
    remaining.current = Math.max(0, deadline.current - performance.now());
  };
  const resume = () => {
    if (durationMs === null || hovered.current || focused.current || timeout.current !== null)
      return;
    const id = notice.id;
    deadline.current = performance.now() + remaining.current;
    timeout.current = setTimeout(() => {
      timeout.current = null;
      dismiss.current(id);
    }, remaining.current);
  };
  useEffect(() => {
    stop();
    remaining.current = durationMs ?? 0;
    // Replacing text under a stationary pointer does not emit another pointerenter event.
    hovered.current = root.current?.matches(':hover') ?? false;
    focused.current = !!root.current?.contains(document.activeElement);
    resume();
    return stop;
  }, [notice.id, durationMs]);
  const message = t(notice.message).replace(/\s*\n\s*/g, ' '),
    level = notice.level ?? 'info';
  return (
    <div
      className="operation-notice"
      data-notice-id={notice.id}
      data-level={level}
      data-has-action={!!notice.action}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      ref={root}
      onPointerEnter={() => {
        hovered.current = true;
        stop();
      }}
      onPointerLeave={() => {
        hovered.current = false;
        resume();
      }}
      onFocusCapture={() => {
        focused.current = true;
        stop();
      }}
      onBlurCapture={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        focused.current = false;
        resume();
      }}
      onPointerDown={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Escape') {
          event.preventDefault();
          dismiss.current(notice.id);
        }
      }}
    >
      <Icon name={{ info: 'info', warning: 'warn', error: 'err' }[level]} size={18} />
      <span className="operation-notice-message" title={message}>
        {message}
      </span>
      {notice.action && (
        <Button
          variant="secondary"
          className="operation-notice-action"
          onClick={notice.action.onClick}
        >
          {t(notice.action.label)}
        </Button>
      )}
      <Button
        icon="close"
        className="operation-notice-close"
        title="Dismiss notification"
        aria-label={t('Dismiss notification')}
        onClick={() => dismiss.current(notice.id)}
      />
    </div>
  );
}
