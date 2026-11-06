import { SCENE_CATALOG } from '../SceneCatalog';
import type { ModelBundle } from '../resources/modelBundle';

export type Asset = {
  id: string;
  name: string;
  format: 'Scene' | 'URDF' | 'MJCF';
  source: { kind: 'sample'; scene: string } | { kind: 'file'; path: string } |
    { kind: 'bundle'; bundle: ModelBundle };
};

export const SAMPLE_ASSETS: Asset[] = [
  ...SCENE_CATALOG.map((scene): Asset => ({ id: `sample:${scene.id}`, name: scene.name,
    format: scene.id === 'joint_gizmo' ? 'MJCF' : scene.id === 'atelier_arm' ? 'URDF' : 'Scene',
    source: { kind: 'sample', scene: scene.id } })),
  { id: 'sample:physics_pendulum', name: 'Driven pendulum', format: 'MJCF',
    source: { kind: 'sample', scene: 'physics_pendulum' } },
  { id: 'sample:physics_drop', name: 'Drop test', format: 'MJCF',
    source: { kind: 'sample', scene: 'physics_drop' } },
];

export function sampleAsset(scene: string): Asset {
  const asset = SAMPLE_ASSETS.find((item) => item.source.kind === 'sample' && item.source.scene === scene);
  if (!asset) throw new Error(`Unknown example scene: ${scene}`);
  return asset;
}

export function fileAsset(path: string): Asset {
  const name = path.split(/[\\/]/).pop()!;
  return { id: `file:${path}`, name, format: name.toLowerCase().endsWith('.urdf') ? 'URDF' : 'MJCF',
    source: { kind: 'file', path } };
}

export interface AssetLibrary { recent: Asset[]; favorites: Asset[] }
const key = 'mojive.asset-library.v1';
function retainedAssets(value: unknown, limit: number): Asset[] {
  if (!Array.isArray(value)) return [];
  return value.filter((asset: Asset) => asset && typeof asset.id === 'string' &&
    typeof asset.name === 'string' && ['Scene', 'URDF', 'MJCF'].includes(asset.format) &&
    ((asset.source?.kind === 'sample' && SAMPLE_ASSETS.some((sample) =>
      sample.id === asset.id && sample.source.kind === 'sample' && sample.source.scene === (asset.source as { scene: string }).scene)) ||
     (asset.source?.kind === 'file' && typeof asset.source.path === 'string' && asset.id === `file:${asset.source.path}`)))
    .slice(0, limit)
    .map((asset) => asset.source.kind === 'sample' ? sampleAsset(asset.source.scene) : asset);
}
export function readAssetLibrary(storage: Pick<Storage, 'getItem'>): AssetLibrary {
  try {
    const saved = JSON.parse(storage.getItem(key) ?? '{}');
    return { recent: retainedAssets(saved.recent, 40), favorites: retainedAssets(saved.favorites, 200) };
  } catch { return { recent: [], favorites: [] }; }
}
export function writeAssetLibrary(storage: Pick<Storage, 'setItem'>, library: AssetLibrary): void {
  // Browser uploads stay in memory; persisted entries never contain engine handles or model bytes.
  storage.setItem(key, JSON.stringify({ recent: library.recent.filter((asset) => asset.source.kind !== 'bundle'),
    favorites: library.favorites.filter((asset) => asset.source.kind !== 'bundle') }));
}

export function rememberAsset(library: AssetLibrary, asset: Asset): AssetLibrary {
  return { ...library, recent: [asset, ...library.recent.filter((item) => item.id !== asset.id)].slice(0, 40) };
}
