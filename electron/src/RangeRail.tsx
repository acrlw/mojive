import type { ComponentProps } from 'react';
import { ScalarRail } from './ScalarRail';
import { useNumberClipboard } from './numberClipboard';
import { t } from './i18n';
import './range-rail.css';

function RangeEndpoint({ value, side }: { value: number; side: 'min' | 'max' }) {
  const { copy, status } = useNumberClipboard();
  return (
    <span
      className={`range-endpoint range-${side}`}
      data-value={String(value)}
      data-tooltip={`${String(value)}\n${status || t('Right-click to copy exact value')}`}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        void copy(value);
      }}
    >
      {Number(value.toPrecision(12)).toString().replace('-', '−')}
    </span>
  );
}

/** Limits describe the model; only the slider and current-value readout edit its state. */
export function RangeRail({
  display = (value) => value,
  ...props
}: ComponentProps<typeof ScalarRail> & {
  display?: (value: number) => number;
}) {
  return (
    <div className="range-rail">
      <RangeEndpoint value={display(props.min)} side="min" />
      <ScalarRail {...props} />
      <RangeEndpoint value={display(props.max)} side="max" />
    </div>
  );
}
