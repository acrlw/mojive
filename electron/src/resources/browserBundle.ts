import type { ResourceFile } from './dropFiles';
import { createModelBundle, modelPath, type ModelBundle } from './modelBundle';

/** Preserve directory-relative names; ambiguous entry points are chosen in the asset browser. */
export async function browserBundles(files: ResourceFile[]): Promise<ModelBundle[]> {
  if (files.length > 10000 || files.reduce((total, file) => total + file.size, 0) > 512 * 1024 * 1024)
    throw new Error('Model resources exceed the package limit (10,000 files or 512 MiB).');
  const resources = await Promise.all(files.map(async (file) => ({
    path: modelPath(file.webkitRelativePath || file.name), bytes: new Uint8Array(await file.arrayBuffer()),
  })));
  const entries = resources.filter((file) => /\.(urdf|xml)$/i.test(file.path) &&
    /<(?:robot|mujoco)(?:\s|>)/.test(new TextDecoder().decode(file.bytes)));
  if (!entries.length) throw new Error('Choose a URDF or MJCF model together with its resource files.');
  return Promise.all(entries.map((entry) => createModelBundle(entry.path.split('/').pop()!, entry.path, resources)));
}

export function bundleUrls(bundle: ModelBundle): { base: string; resolve: (url: string) => string; dispose: () => void } {
  const root = 'https://mojive.invalid/', urls = new Map<string, string>();
  for (const file of bundle.files)
    urls.set(file.path, URL.createObjectURL(new Blob([new Uint8Array(file.bytes)])));
  return {
    base: new URL(bundle.entry.slice(0, bundle.entry.lastIndexOf('/') + 1), root).href,
    resolve: (url) => {
      const path = decodeURIComponent(new URL(url, root).pathname.slice(1));
      const result = urls.get(path);
      if (!result) throw new Error(`Missing model resource: ${path}`);
      return result;
    },
    dispose: () => { for (const url of urls.values()) URL.revokeObjectURL(url); urls.clear(); },
  };
}
