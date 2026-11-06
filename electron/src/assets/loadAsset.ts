import type { SceneDocument } from '../contracts';
import { createCatalogScene } from '../SceneCatalog';
import { disposeDocument, loadUrdf } from '../scene';
import { PhysicsDocument } from '../physics/PhysicsDocument';
import { loadMujocoBackend } from '../physics/WorkerBackend';
import { loadModelBundle, loadModelResource } from '../resources/loadModelBundle';
import { bundleUrls } from '../resources/browserBundle';
import { sampleAsset, type Asset } from './catalog';
import type { AssetPreview } from './PreviewSession';

export interface LoadedAsset { document: SceneDocument; runtime: PhysicsDocument | null }

/** The composition root selects a loader; the browser and preview know no physics implementation. */
export async function loadAsset(asset: Asset): Promise<LoadedAsset> {
  // Persisted library entries can predate a sample's backend migration.
  if (asset.source.kind === 'sample') asset = sampleAsset(asset.source.scene);
  const source = asset.source;
  if (source.kind === 'sample' && asset.format !== 'MJCF')
    return { document: await createCatalogScene(source.scene), runtime: null };
  let bundle;
  if (source.kind === 'file') {
    if (!window.mojive) throw new Error('Reopen this local model in the desktop app.');
    const file = await window.mojive.openModelPath(source.path);
    if (asset.format === 'URDF')
      return { document: await loadUrdf(file.content, file.resourceBase, file.files), runtime: null };
    bundle = await loadModelResource(file.name, file.resourceBase);
  } else if (source.kind === 'sample') {
    const entry = `${source.scene}.xml`;
    const base = new URL('./demo/', window.location.href);
    bundle = await loadModelBundle(asset.name, entry, async (path) => {
      const response = await fetch(new URL(path, base));
      if (!response.ok) throw new Error(`Cannot load sample resource: ${path}`);
      return new Uint8Array(await response.arrayBuffer());
    });
  } else bundle = source.bundle;
  if (asset.format === 'URDF') {
    const resources = bundleUrls(bundle);
    try {
      return { document: await loadUrdf(new TextDecoder().decode(bundle.files.find((file) => file.path === bundle.entry)!.bytes),
        resources.base, [], resources.resolve), runtime: null };
    } finally { resources.dispose(); }
  }
  const runtime = await PhysicsDocument.load(bundle, loadMujocoBackend);
  if (source.kind === 'sample') {
    runtime.document.name = asset.name;
    runtime.document.root.userData.catalogId = source.scene;
  }
  return { document: runtime.document, runtime };
}

export async function loadAssetPreview(asset: Asset): Promise<AssetPreview> {
  const loaded = await loadAsset(asset);
  return { document: loaded.document, dispose: () => { loaded.runtime?.dispose(); disposeDocument(loaded.document); } };
}
