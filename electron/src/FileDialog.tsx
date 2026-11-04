import { useEffect, useRef, useState } from 'react';
import type { FileEntry, FileListing } from './contracts';
import { Button, Icon, Squircle, SwitchRow } from './components';
import { t, useLanguage } from './i18n';
import './file-dialog.css';

export type FileDialogKind = 'urdf' | 'scene' | 'project' | 'save';
export type FileDialogProps = {
  kind: FileDialogKind;
  initialPath?: string;
  suggestedName?: string;
  extension?: '.mojive.json' | '.png' | '.webm';
  caption?: string;
  onCancel: () => void;
  onChoose: (path: string, overwrite: boolean) => Promise<void>;
};
const copy = {
  en: {
    urdf: 'Open URDF',
    scene: 'Open saved scene',
    project: 'Open project folder',
    save: 'Save scene as',
    path: 'Location',
    parent: 'Parent folder',
    refresh: 'Refresh',
    filter: 'Filter files',
    name: 'File name',
    open: 'Open',
    select: 'Select folder',
    saveAction: 'Save',
    hidden: 'Show hidden files',
    empty: 'No matching files',
    loading: 'Loading folder…',
    unavailable:
      'File browsing is available in the desktop app. You can also drag a URDF resource into the workspace.',
    replace: 'Replace existing file?',
    replaceDetail: 'A file with this name already exists. Replace it?',
    replaceAction: 'Replace',
    truncated: 'Showing the first 5,000 entries. Use a more specific folder.',
    home: 'Home',
    desktop: 'Desktop',
    documents: 'Documents',
    downloads: 'Downloads',
    folder: 'Folder',
    file: 'File',
    cancel: 'Cancel',
    allFiles: 'All supported files',
    invalidName: 'Enter a file name without a path separator.',
    selectFile: 'Select a matching file.',
    error: 'Could not complete this action.',
    recent: 'Back',
    forward: 'Forward',
  },
  zh: {
    urdf: '打开 URDF',
    scene: '打开已保存场景',
    project: '打开项目文件夹',
    save: '场景另存为',
    path: '位置',
    parent: '上级目录',
    refresh: '刷新',
    filter: '筛选文件',
    name: '文件名',
    open: '打开',
    select: '选择文件夹',
    saveAction: '保存',
    hidden: '显示隐藏文件',
    empty: '没有匹配的文件',
    loading: '正在读取文件夹…',
    unavailable: '文件浏览仅在桌面应用中可用。也可以将 URDF 资源拖入工作区。',
    replace: '替换已有文件？',
    replaceDetail: '此位置已有同名文件。是否替换？',
    replaceAction: '替换',
    truncated: '当前显示前 5,000 个条目，请打开更具体的文件夹。',
    home: '个人文件夹',
    desktop: '桌面',
    documents: '文稿',
    downloads: '下载',
    folder: '文件夹',
    file: '文件',
    cancel: '取消',
    allFiles: '所有支持的文件',
    invalidName: '请输入不含路径分隔符的文件名。',
    selectFile: '请选择匹配的文件。',
    error: '无法完成此操作。',
    recent: '后退',
    forward: '前进',
  },
};
export function FileDialog({
  kind,
  initialPath,
  suggestedName = 'scene.mojive.json',
  extension = '.mojive.json',
  caption,
  onCancel,
  onChoose,
}: FileDialogProps) {
  const language = useLanguage(),
    tr = copy[language];
  const confirm = useRef<HTMLDialogElement>(null);
  const dialog = useRef<HTMLDialogElement>(null),
    request = useRef(0),
    mounted = useRef(true);
  const [listing, setListing] = useState<FileListing | null>(null),
    [location, setLocation] = useState(initialPath ?? ''),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const [filter, setFilter] = useState(''),
    [hidden, setHidden] = useState(false),
    [selected, setSelected] = useState<string | null>(null),
    [filename, setFilename] = useState(suggestedName);
  const [history, setHistory] = useState<string[]>([]),
    [historyIndex, setHistoryIndex] = useState(-1),
    [replacement, setReplacement] = useState<string | null>(null);
  const accepts = (entry: FileEntry) =>
    entry.directory ||
    (kind === 'save' &&
      entry.name.toLowerCase().endsWith(extension === '.mojive.json' ? '.json' : extension)) ||
    (kind === 'urdf'
      ? entry.name.toLowerCase().endsWith('.urdf')
      : kind === 'scene'
        ? entry.name.toLowerCase().endsWith('.json')
        : false);
  const entries = (listing?.entries ?? []).filter(
    (entry) =>
      accepts(entry) && entry.name.toLocaleLowerCase().includes(filter.toLocaleLowerCase()),
  );
  const navigate = async (path?: string, includeHidden = hidden, fromHistory = false) => {
    const sequence = ++request.current;
    setLoading(true);
    setError('');
    setSelected(null);
    try {
      if (!window.mojive) throw new Error(tr.unavailable);
      const result = await window.mojive.browseFiles(path, includeHidden);
      if (!mounted.current || sequence !== request.current) return;
      setListing(result);
      setLocation(result.path);
      setSelected(null);
      setFilter('');
      if (!fromHistory) {
        setHistory((previous) => [...previous.slice(0, historyIndex + 1), result.path]);
        setHistoryIndex(historyIndex + 1);
      }
      return result;
    } catch (error) {
      if (mounted.current && sequence === request.current)
        setError(String(error instanceof Error ? error.message : error));
    } finally {
      if (mounted.current && sequence === request.current) setLoading(false);
    }
  };
  useEffect(() => {
    mounted.current = true;
    dialog.current?.showModal();
    const element = dialog.current!;
    let zoom = 1;
    for (let ancestor: HTMLElement | null = element; ancestor; ancestor = ancestor.parentElement)
      zoom *= parseFloat(getComputedStyle(ancestor).zoom) || 1;
    element.style.setProperty('--file-dialog-zoom', String(zoom));
    void navigate(initialPath);
    return () => {
      mounted.current = false;
      request.current++;
    };
  }, []);
  useEffect(() => {
    if (replacement && confirm.current && !confirm.current.open) confirm.current.showModal();
  }, [replacement]);
  const choose = async (path: string, overwrite = false) => {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      await onChoose(path, overwrite);
    } catch (error) {
      const message = String(error instanceof Error ? error.message : error);
      if (message.includes('[FILE_EXISTS]') && kind === 'save') setReplacement(path);
      else setError(message);
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  const submit = async () => {
    if (!listing || loading || busy) return;
    let current = listing,
      selection = selected;
    // The visible location is authoritative even when the user confirms without Enter.
    // Resolve it before using any selection from the previous directory.
    if (location !== listing.path) {
      const resolved = await navigate(location);
      if (!resolved) return;
      current = resolved;
      selection = null;
    }
    if (kind === 'project') {
      const entry = current.entries.find((entry) => entry.path === selection);
      void choose(entry?.directory ? entry.path : current.path);
      return;
    }
    if (kind === 'save') {
      const name = filename.trim();
      if (!name || /[\\/\0]/.test(name) || name === '.' || name === '..') {
        setError(tr.invalidName);
        return;
      }
      const leaf = name.toLowerCase().endsWith(extension === '.mojive.json' ? '.json' : extension)
        ? name
        : name + extension;
      const target =
        current.path.replace(/[\\/]$/, '') + (current.path.includes('\\') ? '\\' : '/') + leaf;
      if (current.entries.some((entry) => entry.path === target && !entry.directory))
        setReplacement(target);
      else void choose(target);
      return;
    }
    const entry = current.entries.find((entry) => entry.path === selection);
    if (entry?.directory) {
      void navigate(entry.path);
      return;
    }
    if (!entry || !accepts(entry)) {
      setError(tr.selectFile);
      return;
    }
    void choose(entry.path);
  };
  const selectEntry = (entry: FileEntry) => {
    if (busy || loading) return;
    setSelected(entry.path);
    if (kind === 'save' && !entry.directory) setFilename(entry.name);
  };
  const activate = (entry: FileEntry) => {
    if (busy || loading) return;
    if (entry.directory) void navigate(entry.path);
    else if (kind === 'save') {
      selectEntry(entry);
      setReplacement(entry.path);
    } else void choose(entry.path);
  };
  const goHistory = (index: number) => {
    setHistoryIndex(index);
    void navigate(history[index], hidden, true);
  };
  return (
    <dialog
      ref={dialog}
      className="file-dialog"
      aria-label={caption ?? tr[kind]}
      onCancel={(event) => {
        event.preventDefault();
        if (replacement) setReplacement(null);
        else if (!busy) onCancel();
      }}
    >
      <Squircle radius={16} className="file-dialog-surface">
        <header className="file-dialog-header">
          <Icon name={kind === 'save' ? 'save' : 'folder'} />
          <h2>{caption ?? tr[kind]}</h2>
          <Button icon="close" title="Close" disabled={busy} onClick={onCancel} />
        </header>
        <div className="file-dialog-location">
          <Button
            icon="right"
            className="file-dialog-back"
            title={tr.recent}
            disabled={loading || busy || historyIndex <= 0}
            onClick={() => goHistory(historyIndex - 1)}
          />
          <Button
            icon="right"
            title={tr.forward}
            disabled={loading || busy || historyIndex >= history.length - 1}
            onClick={() => goHistory(historyIndex + 1)}
          />
          <Button
            icon="down"
            className="file-dialog-parent"
            title={tr.parent}
            disabled={loading || busy || !listing?.parent}
            onClick={() => void navigate(listing!.parent!)}
          />
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void navigate(location);
            }}
          >
            <input
              aria-label={tr.path}
              value={location}
              onChange={(event) => {
                setLocation(event.target.value);
                setSelected(null);
              }}
              disabled={busy || loading}
              spellCheck={false}
            />
          </form>
          <Button
            icon="reset"
            title={tr.refresh}
            disabled={loading || busy}
            onClick={() => void navigate(listing?.path, hidden, true)}
          />
        </div>
        <div className="file-dialog-browser">
          <nav className="file-dialog-shortcuts" aria-label={tr.path}>
            {listing?.shortcuts.map((shortcut) => (
              <Button
                key={shortcut.path}
                icon="folder"
                active={listing.path === shortcut.path}
                onClick={() => void navigate(shortcut.path)}
                disabled={loading || busy}
              >
                {tr[shortcut.label.toLowerCase() as 'home'] ?? shortcut.label}
              </Button>
            ))}
          </nav>
          <div className="file-dialog-files">
            <div className="file-dialog-filter">
              <Icon name="search" size={16} />
              <input
                aria-label={tr.filter}
                placeholder={tr.filter + '…'}
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              />
              {filter && <Button icon="close" title="Clear search" onClick={() => setFilter('')} />}
            </div>
            <div
              className="file-dialog-list"
              role="listbox"
              aria-label={caption ?? tr[kind]}
              aria-busy={loading}
              tabIndex={0}
              onKeyDown={(event) => {
                if (busy || loading) return;
                const index = entries.findIndex((entry) => entry.path === selected);
                if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
                  event.preventDefault();
                  const next =
                    event.key === 'Home'
                      ? 0
                      : event.key === 'End'
                        ? entries.length - 1
                        : Math.max(
                            0,
                            Math.min(
                              entries.length - 1,
                              index + (event.key === 'ArrowDown' ? 1 : -1),
                            ),
                          );
                  if (entries[next]) {
                    selectEntry(entries[next]);
                    requestAnimationFrame(() =>
                      dialog.current
                        ?.querySelector(`[data-file-index="${next}"]`)
                        ?.scrollIntoView({ block: 'nearest' }),
                    );
                  }
                } else if (event.key === 'Enter') {
                  event.preventDefault();
                  const entry = entries[index];
                  if (entry) activate(entry);
                } else if (event.key === 'Backspace' && listing?.parent) {
                  event.preventDefault();
                  void navigate(listing.parent);
                }
              }}
            >
              {loading ? (
                <div className="file-dialog-empty">{tr.loading}</div>
              ) : (
                entries.map((entry, index) => (
                  <div
                    key={entry.path}
                    data-file-index={index}
                    role="option"
                    aria-label={entry.name}
                    aria-selected={selected === entry.path}
                    className={`file-dialog-entry ${selected === entry.path ? 'selected' : ''}`}
                    onClick={() => selectEntry(entry)}
                    onDoubleClick={() => activate(entry)}
                  >
                    <Icon
                      name={
                        entry.directory
                          ? 'folder'
                          : entry.name.toLowerCase().endsWith('.urdf')
                            ? 'robot'
                            : 'file'
                      }
                      size={20}
                    />
                    <span title={entry.name}>{entry.name}</span>
                    <small>
                      {entry.directory ? tr.folder : entry.name.split('.').pop()?.toUpperCase()}
                    </small>
                  </div>
                ))
              )}
              {!loading && !entries.length && !error && (
                <div className="file-dialog-empty">{tr.empty}</div>
              )}
            </div>
            {listing?.truncated && <p className="file-dialog-note">{tr.truncated}</p>}
          </div>
        </div>
        {error && (
          <div className="file-dialog-error" role="alert">
            <Icon name="warn" size={16} />
            <span>{error}</span>
          </div>
        )}
        <footer className="file-dialog-footer">
          {kind === 'save' && (
            <label className="file-dialog-filename">
              <span>{tr.name}</span>
              <input
                aria-label={tr.name}
                value={filename}
                disabled={busy}
                onChange={(event) => setFilename(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    submit();
                  }
                }}
              />
            </label>
          )}
          <div className="file-dialog-actions">
            <SwitchRow
              className="file-dialog-hidden"
              label={tr.hidden}
              checked={hidden}
              disabled={busy}
              onChange={(value) => {
                setHidden(value);
                void navigate(listing?.path, value, true);
              }}
            />
            <span className="file-dialog-type">
              {kind === 'urdf'
                ? '*.urdf'
                : kind === 'save'
                  ? `*${extension}`
                  : kind === 'scene'
                    ? '*.mojive.json'
                    : tr.folder}
            </span>
            <div />
            <Button variant="secondary" onClick={onCancel} disabled={busy}>
              {tr.cancel}
            </Button>
            <Button
              className="primary-button"
              disabled={
                !listing || loading || busy || (kind !== 'save' && kind !== 'project' && !selected)
              }
              onClick={submit}
            >
              {kind === 'save' ? tr.saveAction : kind === 'project' ? tr.select : tr.open}
            </Button>
          </div>
        </footer>
        {replacement && (
          <dialog
            ref={confirm}
            className="file-dialog-confirm-dialog"
            aria-label={tr.replace}
            role="alertdialog"
            onCancel={(event) => {
              event.preventDefault();
              event.stopPropagation();
              if (!busy) setReplacement(null);
            }}
          >
            <Squircle radius={12} className="file-dialog-confirm">
              <h3>{tr.replace}</h3>
              <p>{tr.replaceDetail}</p>
              <code>{replacement}</code>
              <div>
                <Button
                  variant="secondary"
                  autoFocus
                  disabled={busy}
                  onClick={() => setReplacement(null)}
                >
                  {tr.cancel}
                </Button>
                <Button
                  className="primary-button"
                  disabled={busy}
                  onClick={() => {
                    const target = replacement;
                    setReplacement(null);
                    void choose(target, true);
                  }}
                >
                  {tr.replaceAction}
                </Button>
              </div>
            </Squircle>
          </dialog>
        )}
      </Squircle>
    </dialog>
  );
}
