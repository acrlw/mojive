import { createModelBundle, modelPath, type ModelBundle } from './modelBundle';
import { checkModelLoad, type ModelLoadOptions } from './loading';

class ResourceReadError extends Error {}

/** Resolve declared resources only; never scan an entire project to open one model. */
export async function loadModelBundle(
  name: string,
  entry: string,
  read: (path: string) => Promise<Uint8Array>,
  options?: ModelLoadOptions,
): Promise<ModelBundle> {
  const files = new Map<string, Uint8Array>();
  const pending = new Map<string, Promise<Uint8Array>>();
  let total = 0;
  const load = async (path: string) => {
    checkModelLoad(options);
    path = modelPath(path);
    if (files.has(path)) return files.get(path)!;
    if (pending.has(path)) return pending.get(path)!;
    const request = (async () => {
      let bytes: Uint8Array;
      try { bytes = await read(path); }
      catch (error) {
        checkModelLoad(options);
        throw new ResourceReadError(`${path}: ${error instanceof Error ? error.message : String(error)}`);
      }
      checkModelLoad(options);
      // Development servers can return the app shell with HTTP 200 for a missing asset.
      const prefix = new TextDecoder().decode(bytes.subarray(0, 512)).trimStart();
      if (/^(?:<!doctype\s+html\b|<html\b)/i.test(prefix))
        throw new ResourceReadError(`${path}: received HTML instead of a model resource`);
      total += bytes.byteLength;
      if (files.size >= 10000 || total > 512 * 1024 * 1024)
        throw new Error('Model resources exceed the package limit (10,000 files or 512 MiB).');
      files.set(path, bytes);
      options?.onProgress?.({ stage: 'resources', completed: files.size, resource: path });
      return bytes;
    })();
    pending.set(path, request);
    try { return await request; } finally { pending.delete(path); }
  };
  const dirname = (path: string) => path.slice(0, path.lastIndexOf('/') + 1);
  const resolve = async (candidates: string[], declaration: string): Promise<string> => {
    const failures: string[] = [];
    const tried = new Set<string>();
    for (const candidate of candidates) {
      let path: string;
      try { path = modelPath(candidate); }
      catch (error) { failures.push(String(error)); continue; }
      if (tried.has(path)) continue;
      tried.add(path);
      try { await load(path); return path; }
      catch (error) {
        if (!(error instanceof ResourceReadError)) throw error;
        failures.push(error.message);
      }
    }
    throw new Error(`Cannot resolve ${declaration}. Tried ${failures.join('; ')}`);
  };
  const compiled = new Set<string>();
  const dependencies = async (entry: string, visiting = new Set<string>()): Promise<void> => {
    entry = modelPath(entry);
    if (visiting.has(entry)) throw new Error(`Cyclic model attachment: ${entry}`);
    if (compiled.has(entry)) return;
    const entryDir = dirname(entry);
    const elements: { element: Element; path: string }[] = [];
    const parse = async (path: string, parents = new Set<string>()): Promise<void> => {
      if (parents.has(path)) throw new Error(`Cyclic model include: ${path}`);
      const document = new DOMParser().parseFromString(new TextDecoder().decode(await load(path)), 'application/xml');
      if (document.querySelector('parsererror')) throw new Error(`Invalid model XML: ${path}`);
      const walk = async (element: Element): Promise<void> => {
        elements.push({ element, path });
        if (element.tagName === 'include' && element.hasAttribute('file')) {
          const file = element.getAttribute('file')!;
          const included = await resolve([entryDir + file, dirname(path) + file], `<include file="${file}"> in ${path}`);
          await parse(included, new Set(parents).add(path));
        } else for (const child of element.children) await walk(child);
      };
      await walk(document.documentElement);
    };
    await parse(entry);
    let assetDir = '', meshDir = '', textureDir = '';
    // Includes are expanded in document order, and share the entry's compiler settings.
    for (const { element } of elements) if (element.tagName === 'compiler') {
      assetDir = element.getAttribute('assetdir') ?? assetDir;
      meshDir = element.getAttribute('meshdir') ?? meshDir;
      textureDir = element.getAttribute('texturedir') ?? textureDir;
    }
    const assets: (() => Promise<unknown>)[] = [];
    const attachments: (() => Promise<void>)[] = [];
    for (const { element: resource, path } of elements) {
      const tag = resource.tagName;
      const resolveAsset = (file: string, directory: string, attribute = 'file') =>
        // MuJoCo tries compiler/main-model paths before the declaring include's directory.
        resolve([entryDir + (directory ? directory + '/' : '') + file, dirname(path) + file],
          `<${tag} ${attribute}="${file}"> in ${path}`);
      if (['mesh', 'texture', 'hfield', 'skin', 'flexcomp'].includes(tag)) {
        // Flexcomp's mesh/GMSH loader also applies compiler.meshdir after XML parsing.
        const directory = tag === 'texture' ? textureDir || assetDir
          : tag === 'mesh' || tag === 'flexcomp' ? meshDir || assetDir : assetDir;
        const attributes = resource.tagName === 'texture'
          ? ['file', 'fileleft', 'fileright', 'fileup', 'filedown', 'filefront', 'fileback'] : ['file'];
        for (const attribute of attributes) {
          const file = resource.getAttribute(attribute);
          if (file) assets.push(() => resolveAsset(file, directory, attribute));
        }
      }
      // Attached models have their own compiler directories, unlike includes.
      if (tag === 'model' && resource.parentElement?.tagName === 'asset' && resource.hasAttribute('file')) {
        attachments.push(async () => {
          const attached = await resolveAsset(resource.getAttribute('file')!, '');
          await dependencies(attached, new Set(visiting).add(entry));
        });
      }
      if (tag === 'mesh' && resource.hasAttribute('filename')) {
        const file = resource.getAttribute('filename')!.replace(/^package:\/\//, '');
        assets.push(() => resolveAsset(file, meshDir || assetDir, 'filename'));
      }
    }
    // Preserve include/compiler resolution order, then read independent assets concurrently.
    let next = 0, failed = false;
    const results = await Promise.allSettled(Array.from({ length: Math.min(8, assets.length) }, async () => {
      try {
        while (!failed && next < assets.length) { checkModelLoad(options); await assets[next++](); }
      } catch (error) { failed = true; throw error; }
    }));
    const failure = results.find((result) => result.status === 'rejected');
    if (failure?.status === 'rejected') throw failure.reason;
    for (const attachment of attachments) await attachment();
    compiled.add(entry);
  };
  options?.onProgress?.({ stage: 'resources', completed: 0, resource: entry });
  await dependencies(entry);
  options?.onProgress?.({ stage: 'resources', completed: files.size, total: files.size });
  options?.onProgress?.({ stage: 'verify' });
  const bundle = await createModelBundle(name, entry, [...files].map(([path, bytes]) => ({ path, bytes })));
  checkModelLoad(options);
  return bundle;
}

export async function loadModelResource(name: string, resourceBase: string, options?: ModelLoadOptions): Promise<ModelBundle> {
  const location = new URL(name, resourceBase);
  // Electron resources are confined to a registered project token.
  const parts = location.pathname.split('/').filter(Boolean);
  const base = new URL(`/${parts[0]}/`, location);
  const entry = parts.slice(1).map(decodeURIComponent).join('/');
  return loadModelBundle(name, entry, async (path) => {
    const url = new URL(path.split('/').map(encodeURIComponent).join('/'), base);
    const response = await fetch(url, { signal: options?.signal });
    if (!response.ok) throw new Error(`Cannot read model resource ${path}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  }, options);
}
