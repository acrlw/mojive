import type { SceneDocument } from '../contracts';
import { createCatalogScene } from '../SceneCatalog';
import { disposeDocument, loadUrdf } from '../scene';
import { PhysicsDocument } from '../physics/PhysicsDocument';
import { loadMujocoBackend } from '../physics/WorkerBackend';
import { loadModelBundle, loadModelResource } from '../resources/loadModelBundle';
import { bundleUrls } from '../resources/browserBundle';
import { sampleAsset, type Asset } from './catalog';
import type { AssetPreview } from './PreviewSession';
import { checkModelLoad, yieldModelLoad, type ModelLoadOptions } from '../resources/loading';

export interface LoadedAsset { document: SceneDocument; runtime: PhysicsDocument | null }

/** The composition root selects a loader; the browser and preview know no physics implementation. */
export async function loadAsset(asset: Asset, options?: ModelLoadOptions): Promise<LoadedAsset> {
  checkModelLoad(options);
  const localDocument = async (load: () => Promise<SceneDocument>): Promise<LoadedAsset> => {
    options?.onProgress?.({ stage: 'scene' });
    await yieldModelLoad(options);
    const document = await load();
    try { checkModelLoad(options); return { document, runtime: null }; }
    catch (error) { disposeDocument(document); throw error; }
  };
  // Persisted library entries can predate a sample's backend migration.
  if (asset.source.kind === 'sample') asset = sampleAsset(asset.source.scene);
  const source = asset.source;
  if (source.kind === 'sample' && asset.format !== 'MJCF')
    return localDocument(() => createCatalogScene(source.scene));
  let bundle;
  if (source.kind === 'file') {
    if (!window.mojive) throw new Error('Reopen this local model in the desktop app.');
    options?.onProgress?.({ stage: 'resources', resource: source.path });
    const file = await window.mojive.openModelPath(source.path);
    checkModelLoad(options);
    if (asset.format === 'URDF')
      return localDocument(() => loadUrdf(file.content, file.resourceBase, file.files));
    bundle = await loadModelResource(file.name, file.resourceBase, options);
  } else if (source.kind === 'sample') {
    const entry = `${source.scene}.xml`;
    const base = new URL('./demo/', window.location.href);
    bundle = await loadModelBundle(asset.name, entry, async (path) => {
      const response = await fetch(new URL(path, base), { signal: options?.signal });
      if (!response.ok) throw new Error(`Cannot load sample resource: ${path}`);
      return new Uint8Array(await response.arrayBuffer());
    }, options);
  } else bundle = source.bundle;
  if (asset.format === 'URDF') {
    const resources = bundleUrls(bundle);
    try {
      return await localDocument(() => loadUrdf(new TextDecoder().decode(bundle.files.find((file) => file.path === bundle.entry)!.bytes),
        resources.base, [], resources.resolve));
    } finally { resources.dispose(); }
  }
  const runtime = await PhysicsDocument.load(bundle, loadMujocoBackend, options);
  if (source.kind === 'sample') {
    runtime.document.name = asset.name;
    runtime.document.root.userData.catalogId = source.scene;
  }
  return { document: runtime.document, runtime };
}

export async function loadAssetPreview(asset: Asset, options?: ModelLoadOptions): Promise<AssetPreview> {
  const loaded = await loadAsset(asset, options);
  return { document: loaded.document, dispose: () => { loaded.runtime?.dispose(); disposeDocument(loaded.document); } };
}
