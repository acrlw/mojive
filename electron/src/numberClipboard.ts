import { numericCopyFeedback } from './numericHints';

/** Copy the underlying value, never the rounded or ellipsized caption. */
export async function copyNumber(value: number) {
  const report = numericCopyFeedback();
  try {
    const text = String(value);
    if (window.mojive) await window.mojive.writeClipboardText(text);
    else await navigator.clipboard.writeText(text);
    report('Copied');
  } catch {
    report('Copy failed');
  }
}
