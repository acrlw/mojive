import type { ComponentProps } from 'react';
import { ScalarRail } from './ScalarRail';
import { copyNumber } from './numberClipboard';
import { useNumericHint } from './numericHints';
import './range-rail.css';

function RangeEndpoint({ value, side }: { value: number; side: 'min' | 'max' }) {
  const hint = useNumericHint('copy');
  return (
    <span
      className={`range-endpoint range-${side}`}
      tabIndex={0}
      {...hint}
      data-value={String(value)}
      data-tooltip={String(value)}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        void copyNumber(value);
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
