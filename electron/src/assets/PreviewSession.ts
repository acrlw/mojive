import type { SceneDocument } from '../contracts';
import type { Asset } from './catalog';

export interface AssetPreview { document: SceneDocument; dispose: () => void }
export type AssetPreviewLoader = (asset: Asset) => Promise<AssetPreview>;

/** Owns only preview resources. Late loads are disposed instead of replacing a newer preview. */
export class PreviewSession {
  private generation = 0;
  private current: AssetPreview | null = null;
  async load(asset: Asset, loader: AssetPreviewLoader): Promise<AssetPreview | null> {
    const generation = ++this.generation;
    this.current?.dispose(); this.current = null;
    const preview = await loader(asset);
    if (generation !== this.generation) { preview.dispose(); return null; }
    this.current = preview;
    return preview;
  }
  dispose(): void { this.generation++; this.current?.dispose(); this.current = null; }
}
