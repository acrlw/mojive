import { useState } from 'react';
import type { SceneDocument, SceneNode } from './contracts';
import { Segmented, Button, Numeric, Section, Select, VectorFields } from './components';
import { readCameraView, updateCameraView, type CameraPatch } from './scene-camera';
import { t, useLanguage } from './i18n';
import './camera-panel.css';

export interface CameraPropertiesProps {
  doc: SceneDocument;
  node: SceneNode;
  active?: boolean;
  piloting?: boolean;
  onBegin: () => void;
  onCommit: () => void;
  onChange: () => void;
  onView: (pilot?: boolean) => void;
  onExit: () => void;
  onAlign: () => void;
  onCapture: () => void;
}
export function CameraProperties({
  doc,
  node,
  active,
  piloting,
  onBegin,
  onCommit,
  onChange,
  onView,
  onExit,
  onAlign,
  onCapture,
}: CameraPropertiesProps) {
  useLanguage();
  const [error, setError] = useState('');
  const view = readCameraView(node);
  const change = (patch: CameraPatch) => {
    try {
      updateCameraView(doc, node.id, patch);
      setError('');
      onChange();
    } catch (reason) {
      setError(String(reason instanceof Error ? reason.message : reason));
    }
  };
  const edit = (patch: CameraPatch) => {
    onBegin();
    change(patch);
    onCommit();
  };
  const field = (
    label: string,
    key: 'fov' | 'near' | 'far' | 'orthoHeight' | 'sensorWidth' | 'focalLength',
    unit: string,
    min: number,
    max: number,
    step = 0.01,
  ) => (
    <div className="camera-property-row" key={key}>
      <label>
        {t(label)} <span>({unit})</span>
      </label>
      <Numeric
        label={label}
        value={view[key]}
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
    <div className="camera-properties" data-camera-properties={node.id}>
      <Section title="Camera view">
        <div className="camera-action-grid">
          <Button
            title={
              active && !piloting
                ? 'Return to Editor Camera'
                : 'View Camera follows the scene. Navigation returns to the editor camera.'
            }
            aria-label={t(active && !piloting ? 'Return to editor' : 'View Camera')}
            icon={active && !piloting ? 'previous' : 'camera'}
            active={active && !piloting}
            onClick={() => (active && !piloting ? onExit() : onView(false))}
          >
            <span className="camera-action-label">
              {t(active && !piloting ? 'Return to editor' : 'View Camera')}
            </span>
          </Button>
          <Button
            disabled={!!node.object.userData.physicsOwned}
            title={
              active && piloting
                ? 'Return to Editor Camera'
                : 'Orbit, pan, and zoom now edit this camera.'
            }
            aria-label={t(active && piloting ? 'Return to editor' : 'Pilot camera')}
            icon={active && piloting ? 'previous' : 'move'}
            active={active && piloting}
            onClick={() => (active && piloting ? onExit() : onView(true))}
          >
            <span className="camera-action-label">
              {t(active && piloting ? 'Return to editor' : 'Pilot camera')}
            </span>
          </Button>
          <Button icon="camera-align" title="Align to view" disabled={!!node.object.userData.physicsOwned} onClick={onAlign}>
            <span className="camera-action-label">{t('Align to view')}</span>
          </Button>
          <Button icon="capture-image" onClick={onCapture}>
            <span className="camera-action-label">{t('Capture image')}</span>
          </Button>
        </div>
      </Section>
      <Section title="Projection">
        <div className="camera-property-row">
          <label>{t('Projection type')}</label>
          <Segmented
            className="camera-projection"
            role="radiogroup"
            aria-label={t('Camera projection')}
          >
            {(['perspective', 'orthographic'] as const).map((projection) => (
              <Button
                key={projection}
                disabled={!!node.object.userData.physicsOwned}
                icon={projection}
                title={projection === 'perspective' ? 'Perspective' : 'Orthographic'}
                aria-label={t(projection === 'perspective' ? 'Perspective' : 'Orthographic')}
                role="radio"
                aria-checked={view.projection === projection}
                tabIndex={view.projection === projection ? 0 : -1}
                active={view.projection === projection}
                onClick={() => {
                  if (view.projection !== projection) edit({ projection });
                }}
                onKeyDown={(event) => {
                  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
                    event.preventDefault();
                    const next =
                      event.key === 'Home'
                        ? 'perspective'
                        : event.key === 'End'
                          ? 'orthographic'
                          : view.projection === 'perspective'
                            ? 'orthographic'
                            : 'perspective';
                    if (view.projection !== next) edit({ projection: next });
                    const buttons =
                      event.currentTarget.parentElement!.querySelectorAll<HTMLButtonElement>(
                        '[role=radio]',
                      );
                    buttons[next === 'perspective' ? 0 : 1]?.focus();
                  }
                }}
              >
                <span className="projection-full">
                  {t(projection === 'perspective' ? 'Perspective' : 'Orthographic')}
                </span>
                <span className="projection-short">
                  {t(projection === 'perspective' ? 'Persp.' : 'Ortho.')}
                </span>
              </Button>
            ))}
          </Segmented>
        </div>
        {view.projection === 'perspective' ? (
          <>
            {field('Vertical field of view', 'fov', '°', 1, 179, 0.1)}
            {field('Focal length', 'focalLength', 'mm', 0.1, 10000, 0.1)}
            {field('Sensor width', 'sensorWidth', 'mm', 0.1, 1000, 0.1)}
          </>
        ) : (
          field('Orthographic height', 'orthoHeight', 'm', 0.0001, 1e6)
        )}
        {field('Near clip', 'near', 'm', 0.00000001, view.far - 0.00000001, 0.001)}
        {field('Far clip', 'far', 'm', view.near + 0.00001, 1e7, 1)}
      </Section>
      <Section title="Output resolution">
        <div className="camera-property-row">
          <label>{t('Preset')}</label>
          <Select
            label="Resolution preset"
            value={`${view.resolution[0]}x${view.resolution[1]}`}
            options={[
              {
                value: `${view.resolution[0]}x${view.resolution[1]}`,
                label: `${view.resolution[0]} × ${view.resolution[1]}`,
              },
              ...[
                [1280, 720],
                [1920, 1080],
                [2560, 1440],
                [3840, 2160],
                [1080, 1080],
                [1080, 1920],
              ]
                .filter(([w, h]) => w !== view.resolution[0] || h !== view.resolution[1])
                .map(([w, h]) => ({ value: `${w}x${h}`, label: `${w} × ${h}` })),
            ]}
            onChange={(value) =>
              edit({ resolution: value.split('x').map(Number) as [number, number] })
            }
          />
        </div>
        {(['Width', 'Height'] as const).map((label, index) => (
          <div className="camera-property-row" key={label}>
            <label>
              {t(label)} <span>(px)</span>
            </label>
            <Numeric
              label={`Camera ${label.toLowerCase()}`}
              value={view.resolution[index]}
              precision={0}
              integer
              min={1}
              max={Math.min(8192, Math.floor(16777216 / view.resolution[1 - index]))}
              step={1}
              onChange={(value) => {
                const resolution: [number, number] = [...view.resolution];
                resolution[index] = value;
                change({ resolution });
              }}
              onBegin={onBegin}
              onCommit={onCommit}
            />
          </div>
        ))}
      </Section>
      {!node.object.userData.physicsOwned && <Section title="World view">
        {(['position', 'target', 'up'] as const).map((key) => (
          <VectorFields
            key={key}
            label={key === 'position' ? 'Eye' : key === 'target' ? 'Target' : 'Up direction'}
            values={view[key]}
            unit={key === 'up' ? undefined : 'm'}
            resetValues={key === 'up' ? [0, 0, 1] : undefined}
            onChange={(axis, value) => {
              const vector = [...view[key]];
              vector[axis] = value;
              change({ [key]: vector });
            }}
            onBegin={onBegin}
            onCommit={onCommit}
          />
        ))}
      </Section>}
      {error && (
        <p className="camera-error" role="status">
          {t(error)}
        </p>
      )}
    </div>
  );
}
