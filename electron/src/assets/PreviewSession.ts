import type { SceneDocument } from '../contracts';
import type { Asset } from './catalog';
import type { ModelLoadOptions } from '../resources/loading';

export interface AssetPreview { document: SceneDocument; dispose: () => void }
export type AssetPreviewLoader = (asset: Asset, options?: ModelLoadOptions) => Promise<AssetPreview>;

/** Owns only preview resources. Late loads are disposed instead of replacing a newer preview. */
export class PreviewSession {
  private generation = 0;
  private controller: AbortController | null = null;
  private current: AssetPreview | null = null;
  async load(asset: Asset, loader: AssetPreviewLoader, options?: ModelLoadOptions): Promise<AssetPreview | null> {
    const generation = ++this.generation;
    this.controller?.abort();
    const controller = new AbortController(); this.controller = controller;
    const abort = () => controller.abort();
    options?.signal?.addEventListener('abort', abort, { once: true });
    if (options?.signal?.aborted) controller.abort();
    this.current?.dispose(); this.current = null;
    let preview: AssetPreview;
    try {
      controller.signal.throwIfAborted();
      preview = await loader(asset, { signal: controller.signal, onProgress: (progress) => {
        if (!controller.signal.aborted && generation === this.generation) options?.onProgress?.(progress);
      } });
    } catch (error) {
      if (controller.signal.aborted) return null;
      throw error;
    } finally {
      options?.signal?.removeEventListener('abort', abort);
      if (this.controller === controller) this.controller = null;
    }
    if (controller.signal.aborted) { preview.dispose(); return null; }
    if (generation !== this.generation) { preview.dispose(); return null; }
    this.current = preview;
    return preview;
  }
  dispose(): void { this.generation++; this.controller?.abort(); this.controller = null; this.current?.dispose(); this.current = null; }
}
