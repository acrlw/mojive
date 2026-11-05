import { useEffect, useRef, useState } from 'react';
import type { ProjectFile } from './contracts';
import { Button, Empty, Icon } from './components';
import { t } from './i18n';

type Project = { path: string; name: string; files: ProjectFile[] };
export function ProjectExplorer({
  project,
  query,
  onOpen,
  onClearSearch,
}: {
  project: Project;
  query: string;
  onOpen: (file: ProjectFile) => void;
  onClearSearch: () => void;
}) {
  const [directories, setDirectories] = useState(() => new Map([[project.path, project.files]]));
  const [expanded, setExpanded] = useState(new Set<string>());
  const [pending, setPending] = useState(new Set<string>());
  const [error, setError] = useState('');
  const [result, setResult] = useState<{
    query: string;
    files: ProjectFile[];
    truncated: boolean;
  } | null>(null);
  const alive = useRef(true),
    generation = useRef(0);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      generation.current++;
    };
  }, []);
  useEffect(() => {
    generation.current++;
    setDirectories(new Map([[project.path, project.files]]));
    setExpanded(new Set());
    setPending(new Set());
    setError('');
  }, [project]);
  useEffect(() => {
    let cancelled = false;
    setResult(null);
    setError('');
    if (!query.trim()) return;
    const timer = setTimeout(async () => {
      try {
        const found = await window.mojive!.searchProject(project.path, query);
        if (!cancelled) setResult({ query, ...found });
      } catch (reason) {
        if (!cancelled) setError(String(reason instanceof Error ? reason.message : reason));
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [project, query]);
  async function load(directory: string) {
    if (directories.has(directory)) return;
    const current = generation.current;
    setPending((previous) => new Set([...previous, directory]));
    try {
      const files = await window.mojive!.readProjectDirectory(directory);
      if (alive.current && generation.current === current)
        setDirectories((previous) => new Map(previous).set(directory, files));
    } finally {
      if (alive.current && generation.current === current)
        setPending((previous) => {
          const next = new Set(previous);
          next.delete(directory);
          return next;
        });
    }
  }
  async function expand(directory: string, open = !expanded.has(directory)) {
    setError('');
    setExpanded((previous) => {
      const next = new Set(previous);
      if (open) next.add(directory);
      else next.delete(directory);
      return next;
    });
    if (open)
      try {
        await load(directory);
      } catch (reason) {
        if (alive.current) setError(String(reason instanceof Error ? reason.message : reason));
      }
  }
  async function reveal(directory: string) {
    const separator = project.path.includes('\\') ? '\\' : '/';
    const relative = directory.slice(project.path.length).replace(/^[\\/]/, '');
    const ancestors: string[] = [];
    let current = project.path;
    for (const part of relative.split(/[\\/]/).filter(Boolean)) {
      current += separator + part;
      ancestors.push(current);
    }
    try {
      for (const ancestor of ancestors) await load(ancestor);
      if (!alive.current) return;
      setExpanded((previous) => new Set([...previous, ...ancestors]));
      onClearSearch();
    } catch (reason) {
      if (alive.current) setError(String(reason instanceof Error ? reason.message : reason));
    }
  }
  function rows(files: ProjectFile[], depth = 0, flat = false): React.ReactNode {
    return files.map((file) => {
      const directory = file.kind === 'directory',
        open = expanded.has(file.path);
      return (
        <div key={file.path} role="none">
          <button
            className="file-row"
            role="treeitem"
            data-tooltip={file.path}
            data-tooltip-overflow="span:last-child"
            aria-expanded={directory && !flat ? open : undefined}
            aria-busy={pending.has(file.path) || undefined}
            style={{ paddingLeft: 12 + (flat ? 0 : depth * 12) }}
            onClick={(event) => {
              if (directory && event.detail <= 1)
                void (flat ? reveal(file.path) : expand(file.path));
            }}
            onDoubleClick={() => {
              if (!directory) onOpen(file);
            }}
            onKeyDown={(event) => {
              if (directory && !flat && ['ArrowRight', 'ArrowLeft'].includes(event.key)) {
                event.preventDefault();
                void expand(file.path, event.key === 'ArrowRight');
              } else if (!directory && event.key === 'Enter') {
                event.preventDefault();
                onOpen(file);
              }
            }}
          >
            {!flat && (
              <span className="file-chevron">
                {directory && <Icon name={open ? 'down' : 'right'} size={12} />}
              </span>
            )}
            <Icon
              name={
                directory ? 'folder' : file.name.toLowerCase().endsWith('.urdf') ? 'robot' : 'geom'
              }
              size={16}
            />
            <span>{file.name}</span>
          </button>
          {!flat && open && rows(directories.get(file.path) ?? [], depth + 1)}
        </div>
      );
    });
  }
  const found = result?.query === query ? result : null;
  return (
    <>
      {error && <div role="alert">{t(error)}</div>}
      {query.trim() ? (
        found ? (
          <>
            {found.files.length ? (
              <div role="tree" aria-label={t('Project files')}>
                {rows(found.files, 0, true)}
              </div>
            ) : (
              <Empty
                title={t('No matching files')}
                action={<Button onClick={onClearSearch}>{t('Clear search')}</Button>}
              />
            )}
            {found.truncated && (
              <div role="status">{t('Search limit reached. Refine your search.')}</div>
            )}
          </>
        ) : (
          !error && <Empty title={t('Searching…')} />
        )
      ) : (
        <div role="tree" aria-label={t('Project files')}>
          {rows(directories.get(project.path) ?? project.files)}
        </div>
      )}
    </>
  );
}
