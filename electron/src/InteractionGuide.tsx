import type { ReactNode } from 'react';
import { KeyHint, MouseHint, PointerHint } from './InputHints';
import { shortcut, shortcutParts } from './platform';
import {
  actionParts,
  type InputAction,
  type InputBindings,
  type NavigationAction,
} from './input-bindings';
import { t } from './i18n';
import './interaction-guide.css';

/** Follow the native guide's primary-input rows, resolved from current bindings. */
export function InteractionGuide({ bindings }: { bindings: InputBindings }) {
  const key = (value: string, modifiers: ('command' | 'shift')[] = []) => (
    <KeyHint keys={shortcutParts(value, modifiers)} label={shortcut(value, modifiers)} />
  );
  const or = (
    <span className="guide-separator" aria-label={t('or')}>
      /
    </span>
  );
  const action = (name: InputAction) =>
    bindings.keys[name] ? (
      <KeyHint keys={actionParts(bindings, name)} />
    ) : (
      <span className="guide-note">{t('Unbound')}</span>
    );
  const actions = (names: InputAction[]) =>
    names
      .filter((name) => bindings.keys[name])
      .map((name) => (
        <span className="guide-gesture" key={name}>
          {action(name)}
        </span>
      ));
  const pointer = (name: NavigationAction) => {
    const primary = bindings.pointer[name][0];
    return primary ? (
      <PointerHint chord={primary} bindings={bindings} />
    ) : (
      <span className="guide-note">{t('Unbound')}</span>
    );
  };
  const rows: { title: string; inputs: ReactNode; note?: string }[] = [
    { title: 'Orbit', inputs: pointer('orbit'), note: 'Drag an empty area' },
    { title: 'Pan', inputs: pointer('pan'), note: 'Drag' },
    { title: 'Zoom', inputs: pointer('zoom') },
    ...(bindings.pointer.dolly.length
      ? [{ title: 'Dolly drag', inputs: pointer('dolly'), note: 'Drag' }]
      : []),
    {
      title: 'Fly camera',
      inputs: actions(['flyForward', 'flyLeft', 'flyBack', 'flyRight', 'flyUp', 'flyDown']),
    },
    { title: 'Transform tools', inputs: actions(['select', 'position', 'rotation']) },
    { title: 'World / body frame', inputs: action('frame') },
    { title: 'Snapping', inputs: action('snap') },
    { title: 'Translation perturbation', inputs: <>{action('perturb')}<MouseHint button="left" /></>, note: 'Dynamic body' },
    { title: 'Rotation perturbation', inputs: <>{action('perturb')}<MouseHint button="right" /></>, note: 'Dynamic body' },
    {
      title: 'Focus',
      inputs: (
        <>
          {action('focus')}
          {or}
          <MouseHint button="left" twice />
        </>
      ),
    },
    { title: 'Settings', inputs: key(',', ['command']) },
    { title: 'Precise gizmo input', inputs: <MouseHint button="left" twice />, note: 'Handle' },
    {
      title: 'Undo / Redo',
      inputs: (
        <>
          {key('Z', ['command'])}
          {or}
          {key('Z', ['shift', 'command'])}
        </>
      ),
    },
    { title: 'Playback', inputs: action('playback') },
    { title: 'Previous simulation frame', inputs: action('previousFrame') },
    { title: 'Step simulation', inputs: action('stepFrame') },
    { title: 'Capture keyframe', inputs: action('capture') },
  ];
  return (
    <div className="guide-grid interaction-guide">
      {rows.map(({ title, inputs, note }) => (
        <div className="guide-row" key={title}>
          <span className="guide-action">{t(title)}</span>
          <div className="guide-inputs">
            {inputs}
            {note && <span className="guide-note">{t(note)}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
