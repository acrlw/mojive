import { Button, Numeric } from '../components';
import { EntityCard } from '../EntityCard';
import { RangeRail } from '../RangeRail';
import { t } from '../i18n';
import type { PhysicsActuator } from './contracts';

export interface ActuatorControlsProps {
  actuators: PhysicsActuator[];
  values: Float64Array;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onFocus: (id: string) => void;
  onBegin: () => void;
  onCommit: () => void;
  onChange: (id: string, value: number) => void;
}

/** Control signals retain backend units; actuator transmission determines their meaning. */
export function ActuatorControls(props: ActuatorControlsProps) {
  const { actuators, values, selectedId, onSelect, onFocus, onBegin, onCommit, onChange } = props;
  return actuators.map((actuator, index) => {
    const [min, max] = actuator.range ?? [-Infinity, Infinity];
    const bounded = Number.isFinite(min) && Number.isFinite(max) && max > min;
    const value = values[index];
    return (
      <EntityCard
        key={actuator.id}
        name={actuator.name}
        icon="control"
        data-actuator-name={actuator.name}
        selected={!!actuator.jointId && actuator.jointId === selectedId}
        controls={
          <>
            <Numeric
              label={t('{{name}} input', { name: actuator.name })}
              value={value}
              appearance="readout"
              copyable
              min={bounded ? min : undefined}
              max={bounded ? max : undefined}
              step={bounded ? (max - min) / 200 : 0.01}
              onBegin={onBegin}
              onCommit={onCommit}
              onChange={(next) => onChange(actuator.id, next)}
            />
            <Button
              icon="reset"
              title="Clear control input"
              onClick={() => {
                onBegin();
                onChange(actuator.id, Math.max(min, Math.min(max, 0)));
                onCommit();
              }}
            />
          </>
        }
        heading={{
          onClick: (event) => {
            if (event.detail > 1) return;
            if (actuator.jointId) onSelect(actuator.jointId);
          },
          onDoubleClick: () => {
            if (actuator.jointId) onFocus(actuator.jointId);
          },
          onKeyDown: (event) => {
            if (event.key === ' ' || event.key === 'Enter') {
              event.stopPropagation();
              if (event.repeat) event.preventDefault();
            }
          },
          onKeyUp: (event) => {
            if (event.key === ' ' || event.key === 'Enter') event.stopPropagation();
          },
        }}
      >
        {bounded && (
          <RangeRail
            value={value}
            min={min}
            max={max}
            referenceValue={0}
            label={t('{{name}} control', { name: actuator.name })}
            onBegin={onBegin}
            onCommit={onCommit}
            onChange={(next) => onChange(actuator.id, next)}
          />
        )}
      </EntityCard>
    );
  });
}
