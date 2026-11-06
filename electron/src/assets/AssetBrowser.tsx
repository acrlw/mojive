import { useEffect, useMemo, useState } from 'react';
import { Button, Empty, Icon, SearchField, Segmented, Squircle } from '../components';
import { t } from '../i18n';
import { SAMPLE_ASSETS, fileAsset, type Asset, type AssetLibrary } from './catalog';
import './assets.css';

export function AssetBrowser({ library, uploads, project, onPreview, onOpen, onFavorite, onOpenProject }: {
  library: AssetLibrary; uploads: Asset[]; project: { path: string; name: string } | null;
  onPreview: (asset: Asset) => void; onOpen: (asset: Asset) => void;
  onFavorite: (asset: Asset) => void; onOpenProject: () => void;
}) {
  const [tab, setTab] = useState(uploads.length ? 'Project' : 'Samples'), [query, setQuery] = useState('');
  const [projectAssets, setProjectAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false), [error, setError] = useState('');
  const [truncated, setTruncated] = useState(false);
  useEffect(() => { if (uploads.length) setTab('Project'); }, [uploads]);
  useEffect(() => {
    let cancelled = false;
    setProjectAssets([]); setError(''); setTruncated(false);
    if (tab !== 'Project' || !project || !window.mojive) { setLoading(false); return; }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const result = await window.mojive!.searchModelAssets(project.path, query);
        if (cancelled) return;
        setProjectAssets(result.files.filter((file) => file.kind !== 'directory' && /\.(xml|urdf)$/i.test(file.name))
          .map((file) => fileAsset(file.path)));
        setTruncated(result.truncated);
      } catch (reason) { if (!cancelled) setError(String(reason instanceof Error ? reason.message : reason)); }
      finally { if (!cancelled) setLoading(false); }
    }, 150);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [tab, project, query]);
  const assets = useMemo(() => {
    const source = tab === 'Samples' ? SAMPLE_ASSETS : tab === 'Recent' ? library.recent :
      tab === 'Favorites' ? library.favorites : [...uploads, ...projectAssets];
    return source.filter((asset) => `${t(asset.name)} ${asset.name} ${asset.format}`.toLowerCase().includes(query.toLowerCase()));
  }, [tab, query, projectAssets, uploads, library]);
  return <section className="asset-browser" aria-label={t('Assets')}>
    <div className="panel-header"><strong>{t('Assets')}</strong><span className="flex-spacer" />
      <Button icon="folder" title="Open project folder" onClick={onOpenProject} />
    </div>
    <div className="asset-browser-controls">
      <SearchField label="Search assets" value={query} onChange={setQuery} />
      <Segmented aria-label={t('Asset collection')}>
        {['Samples', 'Project', 'Recent', 'Favorites'].map((name) =>
          <Button key={name} active={tab === name} onClick={() => setTab(name)}>{name}</Button>)}
      </Segmented>
    </div>
    {error ? <Empty title="Could not read assets">{error}</Empty> : loading && tab === 'Project' ?
      <Empty title="Loading assets…" /> : assets.length ? <div className="asset-list" role="list">
        {assets.map((asset) => <Squircle radius={12} className="asset-row" key={asset.id} role="listitem">
          <button className="asset-row-preview" onClick={() => onPreview(asset)} data-tooltip={t('Preview model')}>
            <Icon name={asset.format === 'MJCF' ? 'env' : 'robot'} size={24} />
            <span><strong>{t(asset.name)}</strong><span className="asset-format">{asset.format}</span></span>
          </button>
          <Button icon="favorite" title={library.favorites.some((item) => item.id === asset.id) ? 'Remove favorite' : 'Add favorite'}
            active={library.favorites.some((item) => item.id === asset.id)} onClick={() => onFavorite(asset)} />
          <Button icon="folder" title="Open model" onClick={() => onOpen(asset)} />
        </Squircle>)}
      </div> : <Empty title={query ? 'No matching assets' : tab === 'Project' ? 'No models in this folder' :
        tab === 'Favorites' ? 'No favorite models' : 'No recent models'}
        action={query ? <Button onClick={() => setQuery('')}>Clear search</Button> :
          tab === 'Project' ? <Button icon="folder" onClick={onOpenProject}>Open folder</Button> :
          <Button onClick={() => setTab('Samples')}>Browse samples</Button>} />}
    {truncated && tab === 'Project' && <p className="asset-search-status">{t('Search limit reached. Refine your search.')}</p>}
  </section>;
}
