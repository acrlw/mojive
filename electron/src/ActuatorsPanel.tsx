import { useEffect, useState } from 'react';
import type { SceneDocument } from './contracts';
import { Icon, Empty } from './components';
import { EntityCard } from './EntityCard';
import { jointIcon } from './JointEditor';
import { t } from './i18n';
import { ActuatorControls, type ActuatorControlsProps } from './physics/ActuatorControls';
import './joint-panel.css';
import './actuators-panel.css';

type Actuator = {
  name: string;
  joint: string;
  mechanicalReduction?: number;
  type?: string;
  transmission?: string;
};

export function ActuatorsPanel({
  doc,
  selectedId,
  onSelect,
  onFocus,
  controls,
}: {
  doc: SceneDocument | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onFocus: (id: string) => void;
  controls?: Pick<
    ActuatorControlsProps,
    'actuators' | 'values' | 'onBegin' | 'onCommit' | 'onChange'
  >;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  useEffect(() => setExpanded({}), [doc?.root.uuid]);
  const actuators = (doc?.root.userData.actuators ?? []) as Actuator[];
  return (
    <div className="joint-panel actuators-panel">
      <div className="joint-panel-header">
        <h2>{t('Actuators')}</h2>
      </div>
      <div className="joint-panel-list">
        {controls ? (
          <ActuatorControls
            key={doc?.root.uuid}
            {...controls}
            selectedId={selectedId}
            onSelect={onSelect}
            onFocus={onFocus}
          />
        ) : (
          actuators.map((actuator, index) => {
            const joint = doc?.joints.find((item) => item.name === actuator.joint);
            const key = `${actuator.transmission ?? ''}:${actuator.name}:${index}`;
            const open = expanded[key] ?? false;
            return (
              <EntityCard
                key={key}
                className="actuator-entry"
                data-actuator-name={actuator.name}
                name={actuator.name}
                icon="control"
                selected={!!joint && selectedId === joint.id}
                expanded={open}
                heading={{
                  title: actuator.name,
                  'aria-label': `${t(open ? 'Collapse actuator details' : 'Expand actuator details')}: ${actuator.name}`,
                  onClick: (event) => {
                    if (event.detail > 1) return;
                    if (joint) onSelect(joint.id);
                    setExpanded((previous) => ({ ...previous, [key]: !open }));
                  },
                  onDoubleClick: () => {
                    if (joint) onFocus(joint.id);
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
                <div className="actuator-summary">
                  <span className="actuator-joint" data-tooltip={actuator.joint}>
                    <Icon name={joint ? jointIcon(joint.type) : 'joint'} size={20} />
                    <span>{actuator.joint || t('No joint')}</span>
                  </span>
                  <span className="actuator-reduction" data-tooltip={t('Reduction')}>
                    <span>{t('Reduction')}</span>
                    <span>{actuator.mechanicalReduction ?? 1}</span>
                  </span>
                </div>
                {open && (
                  <dl className="actuator-details">
                    {actuator.type && (
                      <>
                        <dt>{t('Type')}</dt>
                        <dd data-tooltip={actuator.type}>{actuator.type}</dd>
                      </>
                    )}
                    {actuator.transmission && (
                      <>
                        <dt>{t('Transmission')}</dt>
                        <dd data-tooltip={actuator.transmission}>{actuator.transmission}</dd>
                      </>
                    )}
                  </dl>
                )}
              </EntityCard>
            );
          })
        )}
        {!(controls?.actuators.length ?? actuators.length) && (
          <Empty title={t('No actuators defined')}>
            {t('Actuators are provided by the scene source.')}
          </Empty>
        )}
      </div>
    </div>
  );
}
