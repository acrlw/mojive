import { useEffect, useState } from 'react';
import type { SceneDocument } from './contracts';
import { Button, Empty } from './components';
import { EntityCard } from './EntityCard';
import { jointIcon } from './JointEditor';
import { t } from './i18n';
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
}: {
  doc: SceneDocument | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onFocus: (id: string) => void;
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
        {actuators.map((actuator, index) => {
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
                <Button
                  icon={joint ? jointIcon(joint.type) : 'joint'}
                  disabled={!joint}
                  title={`${t('Inspect joint')}: ${actuator.joint}`}
                  onClick={() => joint && onSelect(joint.id)}
                >
                  <span>{actuator.joint || t('No joint')}</span>
                </Button>
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
        })}
        {!actuators.length && (
          <Empty title={t('No actuators defined')}>
            {t('Actuators are provided by the scene source.')}
          </Empty>
        )}
      </div>
    </div>
  );
}
