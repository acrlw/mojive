import { createModelBundle, modelPath, type ModelBundle } from './modelBundle';

/** Resolve declared resources only; never scan an entire project to open one model. */
export async function loadModelBundle(
  name: string,
  entry: string,
  read: (path: string) => Promise<Uint8Array>,
): Promise<ModelBundle> {
  const files = new Map<string, Uint8Array>();
  let total = 0;
  const load = async (path: string) => {
    path = modelPath(path);
    if (files.has(path)) return files.get(path)!;
    const bytes = await read(path);
    total += bytes.byteLength;
    if (files.size >= 10000 || total > 512 * 1024 * 1024)
      throw new Error('Model resources exceed the package limit (10,000 files or 512 MiB).');
    files.set(path, bytes);
    return bytes;
  };
  const dirname = (path: string) => path.slice(0, path.lastIndexOf('/') + 1);
  const parse = async (path: string, xml: Map<string, Document>, visiting = new Set<string>()): Promise<void> => {
    path = modelPath(path);
    if (visiting.has(path)) throw new Error(`Cyclic model include: ${path}`);
    if (xml.has(path)) return;
    const document = new DOMParser().parseFromString(new TextDecoder().decode(await load(path)), 'application/xml');
    if (document.querySelector('parsererror')) throw new Error(`Invalid model XML: ${path}`);
    xml.set(path, document);
    for (const include of document.querySelectorAll('include[file]'))
      await parse(modelPath(dirname(path) + include.getAttribute('file')!), xml, new Set(visiting).add(path));
  };
  const compiled = new Set<string>();
  const dependencies = async (entry: string, visiting = new Set<string>()): Promise<void> => {
    entry = modelPath(entry);
    if (visiting.has(entry)) throw new Error(`Cyclic model attachment: ${entry}`);
    if (compiled.has(entry)) return;
    const xml = new Map<string, Document>();
    await parse(entry, xml);
    const entryDir = dirname(entry);
    let meshDir = '', textureDir = '';
    for (const document of xml.values()) for (const compiler of document.querySelectorAll('compiler')) {
      meshDir = compiler.getAttribute('meshdir') ?? compiler.getAttribute('assetdir') ?? meshDir;
      textureDir = compiler.getAttribute('texturedir') ?? compiler.getAttribute('assetdir') ?? textureDir;
    }
    for (const document of xml.values()) {
      for (const resource of document.querySelectorAll('mesh, texture, hfield, skin, flexcomp')) {
        const directory = resource.tagName === 'texture' ? textureDir : meshDir;
        const attributes = resource.tagName === 'texture'
          ? ['file', 'fileleft', 'fileright', 'fileup', 'filedown', 'filefront', 'fileback'] : ['file'];
        for (const attribute of attributes) {
          const file = resource.getAttribute(attribute);
          if (file) await load(modelPath(entryDir + (directory ? directory + '/' : '') + file));
        }
      }
      // Attached models have their own compiler directories, unlike includes.
      for (const resource of document.querySelectorAll('asset > model[file]'))
        await dependencies(modelPath(entryDir + resource.getAttribute('file')!), new Set(visiting).add(entry));
      for (const resource of document.querySelectorAll('mesh[filename]')) {
        const file = resource.getAttribute('filename')!.replace(/^package:\/\//, '');
        await load(modelPath(entryDir + file));
      }
    }
    compiled.add(entry);
  };
  await dependencies(entry);
  return createModelBundle(name, entry, [...files].map(([path, bytes]) => ({ path, bytes })));
}

export async function loadModelResource(name: string, resourceBase: string): Promise<ModelBundle> {
  const location = new URL(name, resourceBase);
  // Electron resources are confined to a registered project token.
  const parts = location.pathname.split('/').filter(Boolean);
  const base = new URL(`/${parts[0]}/`, location);
  const entry = parts.slice(1).map(decodeURIComponent).join('/');
  return loadModelBundle(name, entry, async (path) => {
    const url = new URL(path.split('/').map(encodeURIComponent).join('/'), base);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Cannot read model resource ${path}: ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  });
}
