import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Empty, Icon, Numeric, Squircle, Select, SwitchRow } from './components';
import { t, setLanguage, useLanguage, type Language } from './i18n';
import themes from './locales/themes.json';
import { buttonPalette } from './button-palette';
import './settings.css';
export interface EditorSettings {
  language: Language;
  appearance: 'dark' | 'light' | 'system';
  uiScale: number;
  uiFontSize: number;
  numericFont: 'mono' | 'body';
  outputFont: 'mono' | 'body';
  numericFontSize: number | null;
  outputFontSize: number | null;
  showGrid: boolean;
  showViewCube: boolean;
  showEntityIcons: boolean;
  showEntityInfluence: boolean;
  selectionOutline: boolean;
  outlineWidth: number;
  selectionHighlight: boolean;
  cameraFov: number;
  orbitSpeed: number;
  panSpeed: number;
  zoomSpeed: number;
  focusMargin: number;
  transitionDuration: number;
  translationSnap: number;
  rotationSnap: number;
  rotationTickScale: number;
}
export const DEFAULT_SETTINGS: Readonly<EditorSettings> = Object.freeze({
  language: 'zh',
  appearance: 'dark',
  uiScale: 1,
  uiFontSize: 14,
  numericFont: 'mono',
  outputFont: 'mono',
  numericFontSize: null,
  outputFontSize: null,
  showGrid: true,
  showViewCube: true,
  showEntityIcons: true,
  showEntityInfluence: true,
  selectionOutline: true,
  outlineWidth: 3,
  selectionHighlight: true,
  cameraFov: 42,
  orbitSpeed: 0.35,
  panSpeed: 1,
  zoomSpeed: 1,
  focusMargin: 1.15,
  transitionDuration: 0.36,
  translationSnap: 0.1,
  rotationSnap: 5,
  rotationTickScale: 1.25,
});
export const SETTINGS_KEY = 'mojive.electron.settings.v1';
const limits: Partial<Record<keyof EditorSettings, [number, number]>> = {
  uiScale: [0.75, 2],
  uiFontSize: [12, 20],
  cameraFov: [10, 120],
  orbitSpeed: [0.05, 2],
  panSpeed: [0.1, 5],
  zoomSpeed: [0.1, 5],
  focusMargin: [1, 4],
  transitionDuration: [0, 3],
  translationSnap: [0.001, 10],
  rotationSnap: [1, 90],
  rotationTickScale: [0.5, 3],
  outlineWidth: [1, 6],
};
export function normalizeSettings(value: unknown): EditorSettings {
  const source = value && typeof value === 'object' ? (value as Record<string, unknown>) : {},
    result = { ...DEFAULT_SETTINGS };
  for (const key of Object.keys(result) as (keyof EditorSettings)[]) {
    const v = source[key];
    if (typeof result[key] === 'boolean' && typeof v === 'boolean')
      (result as unknown as Record<string, unknown>)[key] = v;
    const range = limits[key];
    if (range && typeof v === 'number' && Number.isFinite(v))
      (result as unknown as Record<string, unknown>)[key] = Math.min(
        range[1],
        Math.max(range[0], v),
      );
  }
  if (source.language === 'zh' || source.language === 'en') result.language = source.language;
  if (['dark', 'light', 'system'].includes(String(source.appearance)))
    result.appearance = source.appearance as EditorSettings['appearance'];
  for (const key of ['numericFont', 'outputFont'] as const)
    if (source[key] === 'body' || source[key] === 'mono') result[key] = source[key];
  for (const key of ['numericFontSize', 'outputFontSize'] as const) {
    const v = source[key];
    if (
      v === null ||
      (typeof v === 'number' && Number.isInteger(v) && v >= 10 && v <= 22 && v % 2 === 0)
    )
      result[key] = v;
  }
  return result;
}
export function readSettings(storage: Pick<Storage, 'getItem'> = localStorage) {
  try {
    return normalizeSettings(JSON.parse(storage.getItem(SETTINGS_KEY) ?? 'null'));
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}
export function applySettings(
  settings: EditorSettings,
  root: HTMLElement = document.documentElement,
  systemDark = matchMedia('(prefers-color-scheme: dark)').matches,
) {
  const theme =
      settings.appearance === 'system' ? (systemDark ? 'dark' : 'light') : settings.appearance,
    palette = themes[theme];
  root.dataset.theme = theme;
  root.lang = settings.language === 'zh' ? 'zh-CN' : 'en';
  root.style.colorScheme = theme;
  for (const [role, value] of Object.entries(palette))
    root.style.setProperty('--ui-' + role.replaceAll('_', '-'), value);
  for (const [name, value] of Object.entries(buttonPalette(palette)))
    root.style.setProperty('--' + name, value);
  const aliases: Record<string, keyof typeof palette> = {
    'text-color': 'text',
    'muted-color': 'secondary',
    'dim-color': 'disabled',
    'sage-color': 'primary',
    'amber-color': 'warning',
    'panel-color': 'background',
    'input-color': 'field',
    'border-color': 'divider',
    'floating-color': 'floating',
    'popup-color': 'popup',
  };
  for (const [name, role] of Object.entries(aliases))
    root.style.setProperty('--' + name, palette[role]);
  const font = (family: 'mono' | 'body') =>
    family === 'mono' ? "'JetBrains Mono','MiSans',monospace" : "'MiSans',sans-serif";
  root.style.setProperty('--font-ui', "'MiSans',sans-serif");
  root.style.setProperty('--font-mono', font(settings.numericFont));
  root.style.setProperty('--font-output', font(settings.outputFont));
  root.style.setProperty('--ui-scale', String(settings.uiScale));
  root.style.setProperty('--body-font-size', settings.uiFontSize + 'px');
  root.style.setProperty('--numeric-font-size', (settings.numericFontSize ?? 14) + 'px');
  root.style.setProperty('--mono-font-size', (settings.numericFontSize ?? 14) + 'px');
  root.style.setProperty('--output-font-size', (settings.outputFontSize ?? 14) + 'px');
  setLanguage(settings.language);
}
export function useSettings(): [EditorSettings, (patch: Partial<EditorSettings>) => void] {
  const [settings, setSettings] = useState(readSettings);
  useLanguage();
  const update = useCallback(
    (patch: Partial<EditorSettings>) =>
      setSettings((current) => normalizeSettings({ ...current, ...patch })),
    [],
  );
  useEffect(() => {
    applySettings(settings);
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    const media = matchMedia('(prefers-color-scheme: dark)'),
      change = () => applySettings(settings);
    media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, [settings]);
  return [settings, update];
}
type Category = 'General' | 'Camera' | 'Interaction' | 'Rendering';
const categories: { name: Category; icon: string }[] = [
  { name: 'General', icon: 'general' },
  { name: 'Camera', icon: 'camera' },
  { name: 'Interaction', icon: 'mouse' },
  { name: 'Rendering', icon: 'shading' },
];
type SettingField = {
  key: keyof EditorSettings;
  label: string;
  group: string;
  category: Category;
  kind: 'number' | 'toggle';
  step?: number;
  unit?: string;
};
export const SETTING_FIELDS: SettingField[] = [
  {
    key: 'uiScale',
    label: 'UI scale',
    group: 'Appearance',
    category: 'General',
    kind: 'number',
    step: 0.1,
    unit: '×',
  },
  {
    key: 'uiFontSize',
    label: 'UI font size',
    group: 'Typography',
    category: 'General',
    kind: 'number',
    step: 1,
    unit: 'px',
  },
  {
    key: 'cameraFov',
    label: 'Camera FOV',
    group: 'Camera controls',
    category: 'Camera',
    kind: 'number',
    step: 1,
    unit: '°',
  },
  {
    key: 'orbitSpeed',
    label: 'Orbit speed',
    group: 'Camera controls',
    category: 'Camera',
    kind: 'number',
    step: 0.05,
  },
  {
    key: 'panSpeed',
    label: 'Pan speed',
    group: 'Camera controls',
    category: 'Camera',
    kind: 'number',
    step: 0.1,
    unit: '×',
  },
  {
    key: 'zoomSpeed',
    label: 'Zoom speed',
    group: 'Camera controls',
    category: 'Camera',
    kind: 'number',
    step: 0.1,
    unit: '×',
  },
  {
    key: 'focusMargin',
    label: 'Focus distance',
    group: 'Focus',
    category: 'Camera',
    kind: 'number',
    step: 0.05,
    unit: '×',
  },
  {
    key: 'transitionDuration',
    label: 'Transition duration',
    group: 'Focus',
    category: 'Camera',
    kind: 'number',
    step: 0.05,
    unit: 's',
  },
  {
    key: 'translationSnap',
    label: 'Position snap',
    group: 'Snap · Shift',
    category: 'Interaction',
    kind: 'number',
    step: 0.01,
    unit: 'm',
  },
  {
    key: 'rotationSnap',
    label: 'Rotation snap',
    group: 'Snap · Shift',
    category: 'Interaction',
    kind: 'number',
    step: 1,
    unit: '°',
  },
  {
    key: 'rotationTickScale',
    label: 'Rotation tick scale',
    group: 'Snap · Shift',
    category: 'Interaction',
    kind: 'number',
    step: 0.05,
    unit: '×',
  },
  {
    key: 'selectionOutline',
    label: 'Selection outline',
    group: 'Selection',
    category: 'Interaction',
    kind: 'toggle',
  },
  {
    key: 'outlineWidth',
    label: 'Outline width',
    group: 'Selection',
    category: 'Interaction',
    kind: 'number',
    step: 1,
    unit: 'px',
  },
  {
    key: 'selectionHighlight',
    label: 'Selection highlight',
    group: 'Selection',
    category: 'Interaction',
    kind: 'toggle',
  },
  { key: 'showGrid', label: 'Show grid', group: 'Viewport', category: 'Rendering', kind: 'toggle' },
  {
    key: 'showViewCube',
    label: 'Show view cube',
    group: 'Viewport',
    category: 'Rendering',
    kind: 'toggle',
  },
  {
    key: 'showEntityInfluence',
    label: 'Selected frustum / light range',
    group: 'Viewport',
    category: 'Rendering',
    kind: 'toggle',
  },
  {
    key: 'showEntityIcons',
    label: 'Camera and light icons',
    group: 'Viewport',
    category: 'Rendering',
    kind: 'toggle',
  },
];
export function SettingsDialog({
  open,
  settings,
  onChange,
  onClose,
}: {
  open: boolean;
  settings: EditorSettings;
  onChange: (patch: Partial<EditorSettings>) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null),
    searchInput = useRef<HTMLInputElement>(null),
    [category, setCategory] = useState<Category>('General'),
    [search, setSearch] = useState('');
  useLanguage();
  useEffect(() => {
    const element = dialog.current;
    if (open && !element?.open) element?.showModal();
    else if (!open && element?.open) element.close();
    if (open && element) {
      let zoom = 1;
      for (let ancestor: HTMLElement | null = element; ancestor; ancestor = ancestor.parentElement)
        zoom *= parseFloat(getComputedStyle(ancestor).zoom) || 1;
      element.style.setProperty('--settings-zoom', String(zoom));
    }
  }, [open, settings.uiScale]);
  const matched = (text: string) =>
    search
      .trim()
      .toLocaleLowerCase()
      .split(/\s+/)
      .every((token) => (text + ' ' + t(text, 'zh')).toLocaleLowerCase().includes(token));
  const fields = useMemo(
    () =>
      SETTING_FIELDS.filter(
        (field) =>
          (search.trim() || field.category === category) &&
          matched(
            [field.label, field.group, field.category]
              .map((value) => value + ' ' + t(value, 'zh'))
              .join(' '),
          ),
      ),
    [category, search, settings.language],
  );
  const searching = !!search.trim();
  const generalFields: { key: keyof EditorSettings; labels: string[] }[] = [
    { key: 'language', labels: ['Language'] },
    { key: 'appearance', labels: ['Appearance', 'Dark', 'Light', 'System'] },
    { key: 'uiFontSize', labels: ['Typography', 'UI font', 'UI font size'] },
    { key: 'numericFont', labels: ['Typography', 'Numeric', 'Font family'] },
    { key: 'numericFontSize', labels: ['Typography', 'Numeric', 'Font size'] },
    { key: 'outputFont', labels: ['Typography', 'Output Messages', 'Font family'] },
    { key: 'outputFontSize', labels: ['Typography', 'Output Messages', 'Font size'] },
  ];
  const generalKeys = new Set(
    generalFields
      .filter(
        (field) =>
          (searching || category === 'General') &&
          matched(
            ['General', ...field.labels].map((label) => label + ' ' + t(label, 'zh')).join(' '),
          ),
      )
      .map((field) => field.key),
  );
  const general =
    generalKeys.has('language') ||
    generalKeys.has('appearance') ||
    fields.some((field) => field.key === 'uiScale');
  const typography = [...generalKeys].some((key) => key !== 'language' && key !== 'appearance');
  const groups = categories.flatMap((item) =>
    [
      ...new Set(
        fields
          .filter((field) => field.category === item.name && item.name !== 'General')
          .map((field) => field.group),
      ),
    ].map((group) => ({ category: item.name, group })),
  );
  const clearSearch = () => {
    setSearch('');
    searchInput.current?.focus();
  };
  const fontRole = (role: 'numeric' | 'output') => {
    const family = role === 'numeric' ? 'numericFont' : 'outputFont',
      size = role === 'numeric' ? 'numericFontSize' : 'outputFontSize';
    return (
      <div className="settings-font-group" key={role}>
        <h3>{t(role === 'numeric' ? 'Numeric' : 'Output Messages')}</h3>
        {generalKeys.has(family) && (
          <div className="settings-row">
            <span>{t('Font family')}</span>
            <Select
              label={t(role === 'numeric' ? 'Numeric' : 'Output Messages') + ' ' + t('Font family')}
              value={settings[family]}
              onChange={(value) => onChange({ [family]: value })}
              options={[
                { value: 'mono', label: 'JetBrains Mono' },
                { value: 'body', label: 'MiSans' },
              ]}
            />
          </div>
        )}
        {generalKeys.has(size) && (
          <div className="settings-row">
            <span>{t('Font size')}</span>
            <Select<string | number>
              label={t(role === 'numeric' ? 'Numeric' : 'Output Messages') + ' ' + t('Font size')}
              value={settings[size] ?? 'default'}
              onChange={(value) => onChange({ [size]: value === 'default' ? null : Number(value) })}
              options={[
                { value: 'default', label: 'Design default' },
                ...[10, 12, 14, 16, 18, 20, 22].map((n) => ({ value: n, label: String(n) })),
              ]}
            />
          </div>
        )}
      </div>
    );
  };
  return (
    <dialog
      className="settings-dialog"
      ref={dialog}
      aria-label={t('Settings')}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <div className="settings-surface">
        <header className="settings-header">
          <Icon name="settings" />
          <h2>{t('Settings')}</h2>
          <Button icon="close" aria-label={t('Close')} title={t('Close')} onClick={onClose} />
        </header>
        <div className="settings-global-search" role="search" aria-label={t('Search all settings')}>
          <Squircle className="settings-search search-field" radius={9999}>
            <Icon name="search" size={16} />
            <input
              ref={searchInput}
              autoFocus
              aria-label={t('Search all settings')}
              placeholder={t('Search all settings')}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search && <Button icon="close" aria-label={t('Clear search')} onClick={clearSearch} />}
          </Squircle>
        </div>
        <div className="settings-layout">
          <nav className="settings-navigation" aria-label={t('Settings')}>
            {categories.map((item) => (
              <Button
                key={item.name}
                icon={item.icon}
                active={category === item.name && !searching}
                aria-current={!searching && category === item.name ? 'page' : undefined}
                onClick={() => {
                  setCategory(item.name);
                  setSearch('');
                }}
              >
                {t(item.name)}
              </Button>
            ))}
          </nav>
          <div className="settings-page">
            <div className="settings-page-heading">
              <h2>{t(searching ? 'Search results' : category)}</h2>
              <Button
                icon="reset"
                title={t('Restore defaults')}
                disabled={searching && !fields.length && !generalKeys.size}
                onClick={() => {
                  if (searching) {
                    const patch: Partial<EditorSettings> = {};
                    for (const f of fields)
                      (patch as Record<string, unknown>)[f.key] = DEFAULT_SETTINGS[f.key];
                    for (const key of generalKeys)
                      (patch as Record<string, unknown>)[key] = DEFAULT_SETTINGS[key];
                    onChange(patch);
                  } else if (category === 'General')
                    onChange({
                      language: DEFAULT_SETTINGS.language,
                      appearance: DEFAULT_SETTINGS.appearance,
                      uiScale: 1,
                      uiFontSize: 14,
                      numericFont: 'mono',
                      outputFont: 'mono',
                      numericFontSize: null,
                      outputFontSize: null,
                    });
                  else {
                    const patch: Partial<EditorSettings> = {};
                    for (const f of SETTING_FIELDS.filter((f) => f.category === category))
                      (patch as Record<string, unknown>)[f.key] = DEFAULT_SETTINGS[f.key];
                    onChange(patch);
                  }
                }}
              >
                {t('Reset')}
              </Button>
            </div>
            <div className="settings-fields">
              {general && (
                <section className="settings-group" data-settings-category="General">
                  {searching && <h3>{t('General')}</h3>}
                  {generalKeys.has('language') && (
                    <div className="settings-row">
                      <span>{t('Language')}</span>
                      <Select<Language>
                        label="Language"
                        value={settings.language}
                        onChange={(language) => onChange({ language })}
                        options={[
                          { value: 'zh', label: '简体中文' },
                          { value: 'en', label: 'English' },
                        ]}
                      />
                    </div>
                  )}
                  {generalKeys.has('appearance') && (
                    <div className="settings-row">
                      <span>{t('Appearance')}</span>
                      <Squircle
                        radius={4}
                        className="settings-segment"
                        role="group"
                        aria-label={t('Appearance')}
                      >
                        {(['dark', 'light', 'system'] as const).map((mode, index) => (
                          <Button
                            key={mode}
                            icon={['moon', 'sun', 'general'][index]}
                            active={settings.appearance === mode}
                            onClick={() => onChange({ appearance: mode })}
                          >
                            {t(['Dark', 'Light', 'System'][index])}
                          </Button>
                        ))}
                      </Squircle>
                    </div>
                  )}
                  {fields.some((f) => f.key === 'uiScale') && (
                    <div className="settings-row">
                      <span>{t('UI scale')} (×)</span>
                      <Numeric
                        value={settings.uiScale}
                        label={t('UI scale')}
                        onChange={(n) => onChange({ uiScale: n })}
                        step={0.1}
                      />
                    </div>
                  )}
                </section>
              )}
              {groups.map(({ category: source, group }) => (
                <section
                  className="settings-group"
                  key={source + group}
                  data-settings-category={source}
                >
                  <h3>{searching ? `${t(source)} › ${t(group)}` : t(group)}</h3>
                  {fields
                    .filter((f) => f.category === source && f.group === group)
                    .map((field) =>
                      field.kind === 'toggle' ? (
                        <SwitchRow
                          key={field.key}
                          className="settings-switch-row"
                          id={'setting-' + field.key}
                          label={field.label}
                          checked={Boolean(settings[field.key])}
                          onChange={(checked) => onChange({ [field.key]: checked })}
                        />
                      ) : (
                        <div className="settings-row" key={field.key}>
                          <label htmlFor={'setting-' + field.key}>
                            {t(field.label)}
                            {field.unit ? ` (${field.unit})` : ''}
                          </label>
                          <Numeric
                            value={Number(settings[field.key])}
                            label={t(field.label)}
                            onChange={(n) => onChange({ [field.key]: n })}
                            step={field.step}
                          />
                        </div>
                      ),
                    )}
                </section>
              ))}
              {typography && (
                <section className="settings-group" data-settings-category="General">
                  <h3>{searching ? `${t('General')} › ${t('Typography')}` : t('Typography')}</h3>
                  {generalKeys.has('uiFontSize') && (
                    <>
                      <div className="settings-row">
                        <span>{t('UI font')}</span>
                        <span className="settings-readonly">MiSans</span>
                      </div>
                      <div className="settings-row">
                        <span>{t('UI font size')} (px)</span>
                        <Numeric
                          value={settings.uiFontSize}
                          label={t('UI font size')}
                          onChange={(n) => onChange({ uiFontSize: n })}
                          step={1}
                          integer
                        />
                      </div>
                    </>
                  )}
                  {(generalKeys.has('numericFont') || generalKeys.has('numericFontSize')) &&
                    fontRole('numeric')}
                  {(generalKeys.has('outputFont') || generalKeys.has('outputFontSize')) &&
                    fontRole('output')}
                </section>
              )}
              {!fields.length && !generalKeys.size && (
                <div className="settings-empty">
                  <Empty
                    title="No matching settings"
                    action={<Button onClick={clearSearch}>Clear search</Button>}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </dialog>
  );
}
