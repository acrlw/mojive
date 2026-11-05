import { useEffect, useRef } from 'react';
import { Button, SwitchRow, Squircle, Section } from './components';
import { DEFAULT_VIEWPORT_LAYERS, type ViewportLayers } from './viewport/layers';
import './ViewportLayers.css';
export { DEFAULT_VIEWPORT_LAYERS, type ViewportLayers };
const sections: { title: string; rows: [keyof ViewportLayers, string][] }[] = [
  {
    title: 'Scene display',
    rows: [
      ['geometry', 'Scene geometry'],
      ['grid', 'Grid'],
    ],
  },
  {
    title: 'Editing aids',
    rows: [
      ['gizmos', 'Gizmos'],
      ['helpers', 'Camera/light helpers'],
      ['influence', 'Selected frustum / light range'],
      ['viewCube', 'View cube'],
    ],
  },
  { title: 'Selection feedback', rows: [['selection', 'Selection highlight']] },
];
/** Native semantic groups, using shared frontend switches and one controlled visibility source. */
export function LayersPopup({
  value,
  onChange,
  onClose,
  translate = (text) => text,
}: {
  value: ViewportLayers;
  onChange: (next: ViewportLayers) => void;
  onClose?: () => void;
  translate?: (text: string) => string;
}) {
  const host = useRef<HTMLDivElement>(null),
    close = useRef(onClose),
    restoreFocus = useRef(false);
  close.current = onClose;
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const toggle =
      opener instanceof HTMLButtonElement && opener.querySelector('[data-icon=layers]')
        ? opener
        : null;
    host.current?.querySelector<HTMLButtonElement>('[role=switch]')?.focus({ preventScroll: true });
    const dismiss = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Node && !host.current?.contains(target) && !toggle?.contains(target)) {
        restoreFocus.current = false;
        close.current?.();
      }
    };
    document.addEventListener('pointerdown', dismiss, true);
    return () => {
      document.removeEventListener('pointerdown', dismiss, true);
      if (restoreFocus.current && opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);
  return (
    <div
      ref={host}
      className="viewport-layers-popup"
      role="dialog"
      aria-label={translate('Viewport layers')}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Escape') {
          event.preventDefault();
          restoreFocus.current = true;
          onClose?.();
        }
      }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <Squircle radius={16} className="viewport-layers-surface">
        <header>
          <strong>{translate('Viewport layers')}</strong>
          <Button
            icon="reset"
            title={translate('Reset')}
            aria-label={translate('Reset viewport layers')}
            onClick={() => onChange({ ...DEFAULT_VIEWPORT_LAYERS })}
          />
        </header>
        <LayerGroups value={value} onChange={onChange} translate={translate} />
      </Squircle>
    </div>
  );
}

export function LayersPanel({
  value,
  onChange,
  translate = (text) => text,
}: {
  value: ViewportLayers;
  onChange: (next: ViewportLayers) => void;
  translate?: (text: string) => string;
}) {
  return (
    <div className="viewport-layers-panel">
      <LayerGroups value={value} onChange={onChange} translate={translate} />
    </div>
  );
}
function LayerGroups({
  value,
  onChange,
  translate,
}: {
  value: ViewportLayers;
  onChange: (next: ViewportLayers) => void;
  translate: (text: string) => string;
}) {
  return (
    <>
      {sections.map((section) => (
        <Section key={section.title} title={translate(section.title)}>
          {section.rows.map(([key, label]) => (
            <SwitchRow
              key={key}
              className="viewport-layer-row"
              checked={value[key]}
              disabled={key === 'influence' && !value.helpers}
              onChange={(checked: boolean) => onChange({ ...value, [key]: checked })}
              label={translate(label)}
            />
          ))}
        </Section>
      ))}
    </>
  );
}
