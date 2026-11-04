import { useState } from 'react';
import type { SceneDocument, SceneNode } from './contracts';
import { ColorPicker, Numeric, Section, Select, SwitchRow, VectorFields } from './components';
import { readLight, updateLight, type LightState, type LightKind } from './scene-light';
import { t, useLanguage } from './i18n';
import './camera-panel.css';
export interface LightPropertiesProps {
  doc: SceneDocument;
  node: SceneNode;
  onBegin: () => void;
  onCommit: () => void;
  onChange: () => void;
}
export function LightProperties({ doc, node, onBegin, onCommit, onChange }: LightPropertiesProps) {
  useLanguage();
  const [error, setError] = useState(''),
    state = readLight(node);
  const change = (patch: Partial<LightState>) => {
    try {
      updateLight(doc, node.id, patch);
      setError('');
      onChange();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const edit = (patch: Partial<LightState>) => {
    onBegin();
    change(patch);
    onCommit();
  };
  const row = (
    label: string,
    key: 'intensity' | 'range' | 'decay' | 'angle' | 'penumbra' | 'width' | 'height',
    min: number,
    max: number,
    unit?: string,
    step = 0.01,
  ) => (
    <div
      className="camera-property-row"
      key={key}
      title={key === 'range' ? t('0 means unlimited') : undefined}
    >
      <label>
        {t(label)}
        {unit && <span> ({unit})</span>}
      </label>
      <Numeric
        label={label}
        value={state[key]}
        min={min}
        max={max}
        step={step}
        onChange={(value) => change({ [key]: value })}
        onBegin={onBegin}
        onCommit={onCommit}
      />
    </div>
  );
  return (
    <div className="light-properties">
      <Section title="Light source">
        <SwitchRow
          label="Enabled"
          aria-label={t('Light enabled')}
          checked={state.enabled}
          onChange={(enabled) => edit({ enabled })}
        />
        <div className="camera-property-row">
          <label>{t('Type')}</label>
          <Select
            label="Light type"
            value={state.type}
            options={(['point', 'directional', 'spot', 'area', 'hemisphere'] as LightKind[]).map(
              (value) => ({
                value,
                label: t(value === 'area' ? 'Area' : value[0].toUpperCase() + value.slice(1)),
              }),
            )}
            onChange={(type) => edit({ type })}
          />
        </div>
        <div className="camera-property-row">
          <label>{t('Color')}</label>
          <ColorPicker
            label={t('Light color')}
            value={state.color}
            onChange={(color) => change({ color })}
            onBegin={onBegin}
            onCommit={onCommit}
          />
        </div>
        {row(
          'Light intensity',
          'intensity',
          0,
          1e6,
          state.type === 'directional'
            ? 'lx'
            : state.type === 'area'
              ? 'cd/m²'
              : state.type === 'hemisphere'
                ? undefined
                : 'cd',
          0.1,
        )}
        {state.type === 'hemisphere' ? (
          <div className="camera-property-row">
            <label>{t('Ground color')}</label>
            <ColorPicker
              label={t('Ground color')}
              value={state.groundColor}
              onChange={(groundColor) => change({ groundColor })}
              onBegin={onBegin}
              onCommit={onCommit}
            />
          </div>
        ) : (
          state.type !== 'area' && (
            <SwitchRow
              label="Cast shadow"
              aria-label={t('Light cast shadow')}
              checked={state.castShadow}
              onChange={(castShadow) => edit({ castShadow })}
            />
          )
        )}
      </Section>
      {(state.type === 'point' || state.type === 'spot') && (
        <Section title="Light attenuation">
          {row('Light range', 'range', 0, 1e6, 'm', 0.05)}
          {row('Decay', 'decay', 0, 10)}
          {state.type === 'spot' && (
            <>
              {row('Spot cutoff', 'angle', 0.1, 89.9, '°', 0.25)}
              {row('Penumbra', 'penumbra', 0, 1)}
            </>
          )}
        </Section>
      )}
      {state.type === 'area' && (
        <Section title="Area emitter">
          {row('Emitter width', 'width', 0.001, 1e6, 'm')}
          {row('Emitter height', 'height', 0.001, 1e6, 'm')}
        </Section>
      )}
      {state.type !== 'point' && state.type !== 'hemisphere' && (
        <Section title="Light direction">
          <VectorFields
            label="Direction (world)"
            values={state.direction}
            resetValues={[0, 0, -1]}
            onChange={(axis, value) => {
              const direction = [...state.direction];
              direction[axis] = value;
              change({ direction });
            }}
            onBegin={onBegin}
            onCommit={onCommit}
          />
        </Section>
      )}
      {error && (
        <p className="camera-error" role="status">
          {t(error)}
        </p>
      )}
    </div>
  );
}
