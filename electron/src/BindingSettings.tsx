import { Fragment, useEffect, useRef, useState, type ReactNode } from 'react';
import { Button, Select } from './components';
import { KeyHint, PointerHint } from './InputHints';
import { t } from './i18n';
import {
  INPUT_ACTIONS,
  FLY_ACTIONS,
  NAVIGATION_ACTIONS,
  DEFAULT_INPUT_BINDINGS,
  NAVIGATION_PRESETS,
  actionParts,
  keyboardChord,
  mouseChord,
  remapKey,
  remapPointer,
  type InputAction,
  type InputBindings,
  type NavigationAction,
} from './input-bindings';

/** Capture is local to the focused field; editor commands cannot consume the same input. */
function BindingRecorder({
  label,
  active,
  pointer,
  children,
  onStart,
  onCancel,
  onRecord,
  onClear,
}: {
  label: string;
  active: boolean;
  pointer?: boolean;
  children: ReactNode;
  onStart: () => void;
  onCancel: () => void;
  onRecord: (value: string) => void;
  onClear?: () => void;
}) {
  const button = useRef<HTMLButtonElement>(null),
    modifier = useRef<string | null>(null),
    skipClick = useRef(false);
  useEffect(() => {
    if (active) button.current?.focus({ preventScroll: true });
  }, [active]);
  useEffect(() => {
    if (!active || !pointer || !button.current) return;
    const element = button.current;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      onRecord(mouseChord(event, true));
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => element.removeEventListener('wheel', wheel);
  }, [active, pointer, onRecord]);
  return (
    <div className={`binding-control ${active ? 'recording' : ''}`}>
      <button
        ref={button}
        className="binding-recorder"
        aria-label={t(label)}
        aria-pressed={active}
        data-tooltip={t(
          pointer
            ? 'Click, then use a mouse button or wheel'
            : 'Click, then press a key or key combination',
        )}
        onClick={() => {
          if (skipClick.current) {
            skipClick.current = false;
            return;
          }
          modifier.current = null;
          onStart();
        }}
        onBlur={() => {
          modifier.current = null;
          if (active) onCancel();
        }}
        onContextMenu={(e) => e.preventDefault()}
        onPointerDown={(e) => {
          if (active && pointer) {
            e.preventDefault();
            e.stopPropagation();
            skipClick.current = e.button === 0;
            onRecord(mouseChord(e));
          }
        }}
        onKeyDown={(e) => {
          if (!active) return;
          if (e.key === 'Tab') {
            onCancel();
            return;
          }
          e.preventDefault();
          e.stopPropagation();
          if (e.key === 'Escape') {
            onCancel();
            return;
          }
          if (pointer || e.repeat) return;
          if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key))
            modifier.current = keyboardChord(e);
          else {
            modifier.current = null;
            onRecord(keyboardChord(e));
          }
        }}
        onKeyUp={(e) => {
          if (active && !pointer && modifier.current) {
            e.preventDefault();
            e.stopPropagation();
            const chord = modifier.current;
            modifier.current = null;
            onRecord(chord);
          }
        }}
      >
        {active ? t(pointer ? 'Use mouse…' : 'Press a key…') : children}
      </button>
      {onClear && (
        <Button
          className="binding-clear"
          icon="close"
          title={`${t('Clear binding')}: ${t(label)}`}
          onClick={onClear}
        />
      )}
    </div>
  );
}

export function BindingSettings({
  bindings,
  onChange,
  matches,
}: {
  bindings: InputBindings;
  onChange: (value: InputBindings) => void;
  matches: (value: string) => boolean;
}) {
  const [recording, setRecording] = useState<string | null>(null),
    [error, setError] = useState('');
  const apply = (change: () => InputBindings) => {
    try {
      onChange(change());
      setError('');
      setRecording(null);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };
  const actions = (Object.keys(INPUT_ACTIONS) as InputAction[]).filter((action) =>
    matches(
      `Interaction Shortcuts Key bindings ${INPUT_ACTIONS[action]} ${t(INPUT_ACTIONS[action], 'zh')}`,
    ),
  );
  const pointers = (Object.keys(NAVIGATION_ACTIONS) as NavigationAction[]).filter((action) =>
    matches(
      `Interaction Mouse gestures Navigation preset ${NAVIGATION_ACTIONS[action]} ${t(NAVIGATION_ACTIONS[action], 'zh')}`,
    ),
  );
  const start = (id: string) => {
    setRecording(id);
    setError('');
  };
  return (
    <>
      {!!actions.length && (
        <section className="settings-group binding-settings" data-settings-category="Interaction">
          <div className="binding-heading">
            <h3>{t('Shortcuts')}</h3>
            <Button
              icon="reset"
              title="Reset shortcuts"
              onClick={() => {
                onChange(DEFAULT_INPUT_BINDINGS);
                setRecording(null);
                setError('');
              }}
            />
          </div>
          {actions.map((action) => (
            <Fragment key={action}>
              {action === actions.find((candidate) => candidate in FLY_ACTIONS) && (
                <h3>{t('Fly camera')}</h3>
              )}
              <div className="settings-row">
                <span>{t(INPUT_ACTIONS[action])}</span>
                <BindingRecorder
                  label={INPUT_ACTIONS[action]}
                  active={recording === action}
                  onStart={() => start(action)}
                  onCancel={() => setRecording(null)}
                  onRecord={(chord) => apply(() => remapKey(bindings, action, chord))}
                  onClear={
                    bindings.keys[action]
                      ? () => apply(() => remapKey(bindings, action, null))
                      : undefined
                  }
                >
                  {bindings.keys[action] ? (
                    <KeyHint keys={actionParts(bindings, action)} />
                  ) : (
                    t('Unbound')
                  )}
                </BindingRecorder>
              </div>
            </Fragment>
          ))}
        </section>
      )}
      {!!pointers.length && (
        <section className="settings-group binding-settings" data-settings-category="Interaction">
          <h3>{t('Mouse gestures')}</h3>
          <div className="settings-row">
            <span>{t('Navigation preset')}</span>
            <Select
              label="Navigation preset"
              value={
                Object.keys(NAVIGATION_PRESETS).find(
                  (name) =>
                    JSON.stringify(NAVIGATION_PRESETS[name as keyof typeof NAVIGATION_PRESETS]) ===
                    JSON.stringify(bindings.pointer),
                ) ?? 'custom'
              }
              options={[
                { value: 'custom', label: 'Custom', disabled: true },
                ...Object.keys(NAVIGATION_PRESETS).map((name) => ({ value: name, label: name })),
              ]}
              onChange={(name) => {
                onChange({
                  ...bindings,
                  pointer: NAVIGATION_PRESETS[name as keyof typeof NAVIGATION_PRESETS],
                });
                setError('');
                setRecording(null);
              }}
            />
          </div>
          {pointers.map((action) => (
            <div className="settings-row" key={action}>
              <span>{t(NAVIGATION_ACTIONS[action])}</span>
              <div className="binding-pointer-values">
                {(bindings.pointer[action].length ? bindings.pointer[action] : ['']).map(
                  (chord, index) => {
                    const id = `${action}:${index}`,
                      label = `${t(NAVIGATION_ACTIONS[action])} ${index + 1}`;
                    return (
                      <BindingRecorder
                        key={id}
                        label={label}
                        pointer
                        active={recording === id}
                        onStart={() => start(id)}
                        onCancel={() => setRecording(null)}
                        onRecord={(value) =>
                          apply(() =>
                            remapPointer(
                              bindings,
                              action,
                              bindings.pointer[action].length
                                ? bindings.pointer[action]
                                    .map((old, i) => (i === index ? value : old))
                                    .join(';')
                                : value,
                            ),
                          )
                        }
                        onClear={
                          chord
                            ? () =>
                                apply(() =>
                                  remapPointer(
                                    bindings,
                                    action,
                                    bindings.pointer[action]
                                      .filter((_, i) => i !== index)
                                      .join(';'),
                                  ),
                                )
                            : undefined
                        }
                      >
                        {chord ? (
                          <PointerHint chord={chord} bindings={bindings} size={20} />
                        ) : (
                          t('Unbound')
                        )}
                      </BindingRecorder>
                    );
                  },
                )}
                {recording === `${action}:add` ? (
                  <BindingRecorder
                    label={`${t('Add gesture')}: ${t(NAVIGATION_ACTIONS[action])}`}
                    active
                    pointer
                    onStart={() => {}}
                    onCancel={() => setRecording(null)}
                    onRecord={(value) =>
                      apply(() =>
                        remapPointer(
                          bindings,
                          action,
                          [...bindings.pointer[action], value].join(';'),
                        ),
                      )
                    }
                  >
                    {t('Use mouse…')}
                  </BindingRecorder>
                ) : (
                  !!bindings.pointer[action].length && (
                    <Button
                      className="binding-add"
                      icon="plus"
                      title={`${t('Add gesture')}: ${t(NAVIGATION_ACTIONS[action])}`}
                      onClick={() => start(`${action}:add`)}
                    />
                  )
                )}
              </div>
            </div>
          ))}
        </section>
      )}
      {error && (
        <p className="binding-error" role="alert">
          {t(error)}
        </p>
      )}
    </>
  );
}
