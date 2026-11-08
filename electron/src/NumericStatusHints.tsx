import type { ReactNode } from 'react';
import { MouseHint } from './InputHints';
import { useNumericStatus } from './numericHints';
import { t } from './i18n';

export function NumericStatusHints({ children }: { children: ReactNode }) {
  const { mode, feedback } = useNumericStatus();
  if (!mode) return children;
  if (mode === 'edit')
    return (
      <>
        <span className="status-hint">
          <kbd>Enter</kbd>
          {t('Apply')}
        </span>
        <span className="status-hint">
          <kbd>Esc</kbd>
          {t('Cancel')}
        </span>
      </>
    );
  return (
    <>
      {mode !== 'copy' && (
        <>
          <span className="status-hint">
            <MouseHint button="left" size={20} />
            {t('Adjust value')}
          </span>
          <span className="status-hint">
            <MouseHint button="left" twice size={20} />
            {t('Enter value')}
          </span>
        </>
      )}
      {mode !== 'adjust' && (
        <span className="status-hint">
          <MouseHint button="right" size={20} />
          {t(feedback || 'Copy value')}
        </span>
      )}
    </>
  );
}
