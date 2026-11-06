import { useEffect, useRef, useState } from 'react';
import { Button, Empty } from '../components';
import { InfoDialog } from '../InfoDialog';
import { Viewport, type ViewportHandle, type ViewportSettings } from '../Viewport';
import { t } from '../i18n';
import { modelLoadStageLabels } from '../SceneLoadProgress';
import type { ModelLoadProgress } from '../resources/loading';
import type { Asset } from './catalog';
import { PreviewSession, type AssetPreview, type AssetPreviewLoader } from './PreviewSession';
import './assets.css';

export function AssetPreviewDialog({ asset, load, onClose, onOpen, settings }: {
  asset: Asset; load: AssetPreviewLoader; onClose: () => void;
  onOpen: (asset: Asset) => void; settings: ViewportSettings;
}) {
  const [preview, setPreview] = useState<AssetPreview | null>(null);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState<ModelLoadProgress>({ stage: 'resources' });
  const viewport = useRef<ViewportHandle>(null);
  useEffect(() => {
    const session = new PreviewSession();
    let cancelled = false;
    setPreview(null); setError(''); setProgress({ stage: 'resources' });
    void session.load(asset, load, { onProgress: setProgress }).then((next) => { if (!cancelled) setPreview(next); })
      .catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { cancelled = true; session.dispose(); };
  }, [asset, load]);
  return <InfoDialog label={t('Asset preview')} className="asset-preview-dialog" onClose={onClose}>
    <div className="dialog-heading">
      <strong>{t(asset.name)}</strong><span className="asset-format">{asset.format}</span>
      <span className="flex-spacer" />
      <Button icon="fit" title="Frame all" disabled={!preview} onClick={() => viewport.current?.focus()} />
      <Button icon="close" title="Close preview" onClick={onClose} />
    </div>
    <div className="asset-preview-stage">
      {preview ? <Viewport ref={viewport} document={preview.document} selected={null} tool="select"
        frame="world" geometry="default" revision={0} settings={settings} translate={t}
        onSelect={() => {}} onChange={() => {}} /> :
        <Empty title={error ? 'Could not preview model' : 'Loading preview…'}>
          {error || <span role="status">{t(modelLoadStageLabels[progress.stage])}
            {progress.total !== undefined && ` · ${progress.completed ?? 0} / ${progress.total}`}</span>}
        </Empty>}
    </div>
    <div className="asset-preview-footer">
      {preview && <span>{preview.document.nodes.length} {t('Objects')} · {preview.document.joints.length} {t('Joints')}</span>}
      <span className="flex-spacer" />
      <Button onClick={onClose}>Close</Button>
      <Button variant="primary" icon="folder" disabled={!preview} onClick={() => onOpen(asset)}>Open model</Button>
    </div>
  </InfoDialog>;
}
