import { commandKey } from './platform';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { SearchField, Button, Icon } from './components';
import { t, useLanguage, type Language, type LocalizedText } from './i18n';
import './output.css';
export interface OutputEntry {
  id: number;
  level: 'info' | 'warn' | 'err' | 'warning' | 'error' | 'debug' | 'success' | 'critical' | 'fatal';
  time: string;
  text: LocalizedText;
}
export type OutputLevel = 'info' | 'warning' | 'error';
export type OutputOrder = 'oldest' | 'newest';
export function outputLevel(level: OutputEntry['level']): OutputLevel {
  return ['warn', 'warning'].includes(level)
    ? 'warning'
    : ['err', 'error', 'critical', 'fatal'].includes(level)
      ? 'error'
      : 'info';
}
export function filterOutput(
  entries: readonly OutputEntry[],
  query: string,
  levels: ReadonlySet<OutputLevel>,
  order: OutputOrder = 'oldest',
  language: Language = 'zh',
) {
  const tokens = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean),
    labels = { info: 'Info', warning: 'Warning', error: 'Error' };
  return entries
    .filter((entry) => {
      const level = outputLevel(entry.level),
        searchable =
          `${entry.time} ${entry.level} ${level} ${t(labels[level], language)} ${t(entry.text, 'en')} ${t(entry.text, language)}`.toLocaleLowerCase();
      return levels.has(level) && tokens.every((token) => searchable.includes(token));
    })
    .sort((a, b) => (order === 'oldest' ? a.id - b.id : b.id - a.id));
}
export function copyOutput(entries: readonly OutputEntry[], selection: ReadonlySet<number>) {
  const selected = entries.filter((entry) => selection.has(entry.id));
  return selected.length
    ? selected.map((entry) => t(entry.text, 'en')).join('\n')
    : entries
        .map((entry) => `${entry.time} [${outputLevel(entry.level)}] ${t(entry.text, 'en')}`)
        .join('\n');
}
const PREFERENCES_KEY = 'mojive.electron.output.v1';
function loadPreferences(): { autoScroll: boolean; order: OutputOrder } {
  try {
    const stored = JSON.parse(localStorage.getItem(PREFERENCES_KEY) ?? 'null');
    return {
      autoScroll: typeof stored?.autoScroll === 'boolean' ? stored.autoScroll : true,
      order: stored?.order === 'newest' ? ('newest' as const) : ('oldest' as const),
    };
  } catch {
    return { autoScroll: true, order: 'oldest' as const };
  }
}
export function OutputPanel({
  entries,
  onClear,
  onCollapse,
  language,
}: {
  entries: readonly OutputEntry[];
  onClear: () => void;
  onCollapse?: () => void;
  language?: Language;
}) {
  const currentLanguage = useLanguage(),
    locale = language ?? currentLanguage,
    tr = (value: LocalizedText) => t(value, locale),
    initial = useRef(loadPreferences()),
    [autoScroll, setAutoScroll] = useState(initial.current.autoScroll),
    [order, setOrder] = useState<OutputOrder>(initial.current.order),
    [levels, setLevels] = useState<Set<OutputLevel>>(new Set(['info', 'warning', 'error'])),
    [search, setSearch] = useState(''),
    [showSearch, setShowSearch] = useState(false),
    [selection, setSelection] = useState<Set<number>>(new Set()),
    [copyError, setCopyError] = useState(false),
    anchor = useRef<number | null>(null),
    scroll = useRef<HTMLDivElement>(null),
    searchInput = useRef<HTMLInputElement>(null),
    programmaticScroll = useRef(false);
  const shown = useMemo(
    () => filterOutput(entries, search, levels, order, locale),
    [entries, search, levels, order, locale],
  );
  const counts = useMemo(() => {
    const result = { info: 0, warning: 0, error: 0 };
    for (const entry of entries) result[outputLevel(entry.level)]++;
    return result;
  }, [entries]);
  useLayoutEffect(() => {
    const element = scroll.current;
    if (!element) return;
    const context = document.createElement('canvas').getContext('2d')!;
    let mounted = true;
    const alignIcons = () => {
      if (!mounted) return;
      const style = getComputedStyle(element);
      const size = parseFloat(style.fontSize) * (16 / 14);
      context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      const sample = locale === 'zh' ? 'H国' : 'H';
      const metrics = context.measureText(sample);
      const probe = document.createElement('span');
      probe.style.cssText =
        'position:absolute;visibility:hidden;pointer-events:none;display:block;width:100px;white-space:nowrap';
      const baselineMarker = document.createElement('span');
      baselineMarker.style.cssText =
        'display:inline-block;width:0;height:0;vertical-align:baseline';
      probe.append(sample, baselineMarker);
      element.append(probe);
      const bounds = probe.getBoundingClientRect();
      const baseline =
        (baselineMarker.getBoundingClientRect().top - bounds.top) / (bounds.width / 100);
      probe.remove();
      // Center the complete SVG canvas on first-line ink, including CJK fallback metrics.
      // A fixed top margin drifts when the output font or its size changes.
      const center =
        baseline - (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2;
      element.style.setProperty('--output-icon-size', `${size}px`);
      element.style.setProperty('--output-icon-top', `${center - size / 2}px`);
    };
    alignIcons();
    const observer = new MutationObserver(alignIcons);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] });
    document.fonts.addEventListener('loadingdone', alignIcons);
    void document.fonts.ready.then(alignIcons);
    return () => {
      mounted = false;
      observer.disconnect();
      document.fonts.removeEventListener('loadingdone', alignIcons);
    };
  }, [locale]);
  useEffect(() => {
    localStorage.setItem(PREFERENCES_KEY, JSON.stringify({ autoScroll, order }));
  }, [autoScroll, order]);
  useEffect(() => {
    if (showSearch) searchInput.current?.focus();
  }, [showSearch]);
  useEffect(() => {
    const ids = new Set(entries.map((entry) => entry.id));
    setSelection((previous) => new Set([...previous].filter((id) => ids.has(id))));
  }, [entries]);
  useLayoutEffect(() => {
    const element = scroll.current;
    if (!element || !autoScroll) return;
    programmaticScroll.current = true;
    element.scrollTop = order === 'newest' ? 0 : element.scrollHeight;
    const frame = requestAnimationFrame(() => {
      programmaticScroll.current = false;
    });
    return () => cancelAnimationFrame(frame);
  }, [shown, autoScroll, order]);
  const copy = async () => {
    try {
      const text = copyOutput(shown, selection);
      if (window.mojive) await window.mojive.writeClipboardText(text);
      else await navigator.clipboard.writeText(text);
      setCopyError(false);
    } catch {
      setCopyError(true);
    }
  };
  const select = (id: number, event: React.MouseEvent | React.KeyboardEvent) => {
    if (event.shiftKey && anchor.current !== null) {
      const a = shown.findIndex((entry) => entry.id === anchor.current),
        b = shown.findIndex((entry) => entry.id === id);
      if (a >= 0 && b >= 0) {
        const range = shown.slice(Math.min(a, b), Math.max(a, b) + 1).map((entry) => entry.id);
        setSelection((previous) => new Set(commandKey(event) ? [...previous, ...range] : range));
        return;
      }
    }
    anchor.current = id;
    if (commandKey(event))
      setSelection((previous) => {
        const next = new Set(previous);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    else setSelection(new Set([id]));
  };
  const toggleSearch = () => {
    setShowSearch((value) => !value);
    if (showSearch) setSearch('');
  };
  return (
    <section
      className="mojive-output"
      aria-label={tr('Output')}
      onKeyDown={(event) => {
        if (event.target instanceof HTMLInputElement) return;
        if (commandKey(event) && event.key.toLowerCase() === 'a') {
          event.preventDefault();
          setSelection(new Set(shown.map((entry) => entry.id)));
        }
        if (commandKey(event) && event.key.toLowerCase() === 'c') {
          event.preventDefault();
          void copy();
        }
        if (commandKey(event) && event.key.toLowerCase() === 'f') {
          event.preventDefault();
          setShowSearch(true);
          searchInput.current?.focus();
        }
      }}
    >
      <div className="output-controls">
        <div
          className={`output-toolbar ${showSearch ? 'with-search' : ''}`}
          role="toolbar"
          aria-label={tr('Output')}
        >
          <div className="output-severities" role="group" aria-label={tr('Severity')}>
            {(
              [
                ['info', 'Info', 'info'],
                ['warning', 'Warning', 'warn'],
                ['error', 'Error', 'err'],
              ] as const
            ).map(([level, label, icon]) => (
              <Button
                key={level}
                icon={icon}
                active={levels.has(level)}
                className={'output-level output-level-' + level}
                aria-pressed={levels.has(level)}
                aria-label={tr(label)}
                title={`${tr(label)} ${counts[level]}`}
                onClick={() =>
                  setLevels((previous) => {
                    const next = new Set(previous);
                    if (next.has(level)) next.delete(level);
                    else next.add(level);
                    return next;
                  })
                }
              >
                <span>{counts[level]}</span>
              </Button>
            ))}
          </div>
          {showSearch && (
            <SearchField
              className="output-search"
              inputRef={searchInput}
              label={tr('Search output')}
              placeholder={tr('Search output')}
              value={search}
              onChange={setSearch}
              clearLabel={tr('Clear search')}
              onKeyDown={(event) => {
                if (event.key === 'Escape') {
                  event.stopPropagation();
                  setSearch('');
                  setShowSearch(false);
                }
              }}
            />
          )}
          <div className="output-actions">
            <Button
              icon="search"
              active={showSearch}
              aria-label={tr('Search output')}
              title={tr('Search output')}
              aria-pressed={showSearch}
              onClick={toggleSearch}
            />
            <Button
              icon="sort"
              active={order === 'newest'}
              aria-label={tr('Sort output')}
              title={tr(order === 'oldest' ? 'Oldest first' : 'Newest first')}
              onClick={() => setOrder((value) => (value === 'oldest' ? 'newest' : 'oldest'))}
            />
            <Button
              icon="copy"
              disabled={!shown.length}
              aria-label={tr(selection.size ? 'Copy selection' : 'Copy shown')}
              title={tr(selection.size ? 'Copy selection' : 'Copy shown')}
              onClick={() => void copy()}
            />
            <Button
              icon="follow"
              active={autoScroll}
              aria-pressed={autoScroll}
              aria-label={tr('Auto-scroll')}
              title={tr('Auto-scroll')}
              onClick={() => setAutoScroll((value) => !value)}
            />
            <Button
              icon="delete"
              disabled={!entries.length}
              aria-label={tr('Clear output')}
              title={tr('Clear output')}
              onClick={() => {
                setSelection(new Set());
                anchor.current = null;
                onClear();
              }}
            />
            {onCollapse && (
              <Button
                icon="down"
                aria-label={tr('Collapse Output')}
                title={tr('Collapse Output')}
                onClick={onCollapse}
              />
            )}
          </div>
        </div>
      </div>
      {copyError && (
        <div className="output-copy-error" role="alert">
          {tr('Copy failed')}
        </div>
      )}
      <div
        ref={scroll}
        className="output-messages"
        role="listbox"
        aria-label={tr('Output Messages')}
        aria-multiselectable="true"
        tabIndex={0}
        onWheel={(event) => {
          if (autoScroll && (order === 'oldest' ? event.deltaY < 0 : event.deltaY > 0))
            setAutoScroll(false);
        }}
        onScroll={(event) => {
          if (programmaticScroll.current || !autoScroll) return;
          const node = event.currentTarget,
            following =
              order === 'newest'
                ? node.scrollTop <= 4
                : node.scrollHeight - node.scrollTop - node.clientHeight <= 4;
          if (!following) setAutoScroll(false);
        }}
      >
        {shown.map((entry, index) => {
          const level = outputLevel(entry.level);
          return (
            <div
              key={entry.id}
              role="option"
              aria-selected={selection.has(entry.id)}
              tabIndex={selection.has(entry.id) || (!selection.size && index === 0) ? 0 : -1}
              className={`output-message ${selection.has(entry.id) ? 'selected' : ''}`}
              onClick={(event) => select(entry.id, event)}
              onKeyDown={(event) => {
                if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                  event.preventDefault();
                  const next =
                    shown[
                      Math.max(
                        0,
                        Math.min(shown.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)),
                      )
                    ];
                  select(next.id, event);
                  scroll.current
                    ?.querySelector<HTMLElement>(`[data-output-id="${next.id}"]`)
                    ?.focus();
                }
                if (event.key === ' ') {
                  event.preventDefault();
                  select(entry.id, event);
                }
              }}
              data-output-id={entry.id}
            >
              <Icon
                name={level === 'warning' ? 'warn' : level === 'error' ? 'err' : 'info'}
                size={16}
                className={'output-severity-' + level}
              />
              <time>{entry.time}</time>
              <span className="output-message-text">{tr(entry.text)}</span>
            </div>
          );
        })}
        {!shown.length && (
          <div className="output-empty">
            <span>{tr(entries.length ? 'No matching messages' : 'No messages')}</span>
            {entries.length > 0 && (
              <Button
                onClick={() => {
                  setSearch('');
                  setLevels(new Set(['info', 'warning', 'error']));
                }}
              >
                {tr('Clear filters')}
              </Button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
