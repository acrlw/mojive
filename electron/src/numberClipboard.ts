import { useEffect, useRef, useState } from 'react';
import { t } from './i18n';

/** Copy the underlying value, never the rounded or ellipsized caption. */
export function useNumberClipboard() {
  const [status, setStatus] = useState('');
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = async (value: number) => {
    try {
      const text = String(value);
      if (window.mojive) await window.mojive.writeClipboardText(text);
      else await navigator.clipboard.writeText(text);
      setStatus(t('Copied'));
    } catch {
      setStatus(t('Copy failed'));
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus(''), 2000);
  };
  return { copy, status };
}
