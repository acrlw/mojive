/** A complete model package with engine-independent, project-relative resource names. */
export interface ModelBundle {
  id: string;
  name: string;
  entry: string;
  files: { path: string; bytes: Uint8Array }[];
}

export function modelPath(path: string): string {
  const value = path.replaceAll('\\', '/');
  if (!value || value.startsWith('/') || /^[a-zA-Z][\w+.-]*:/.test(value) || value.includes('\0'))
    throw new Error(`Model resources must use relative paths: ${path}`);
  const parts: string[] = [];
  for (const part of value.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      if (!parts.length) throw new Error(`Resource escapes the model package: ${path}`);
      parts.pop();
    } else parts.push(part);
  }
  if (!parts.length) throw new Error(`Empty model resource path: ${path}`);
  return parts.join('/');
}

export async function createModelBundle(
  name: string,
  entry: string,
  resources: { path: string; bytes: Uint8Array }[],
): Promise<ModelBundle> {
  const files = resources.map(({ path, bytes }) => ({ path: modelPath(path), bytes: bytes.slice() }))
    .sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  const paths = new Set(files.map((file) => file.path));
  if (paths.size !== files.length) throw new Error('Duplicate model resource paths.');
  const canonicalEntry = modelPath(entry);
  if (!paths.has(canonicalEntry)) throw new Error(`Missing model entry: ${canonicalEntry}`);
  // Hash resource contents independently to avoid concatenating the entire asset package.
  const hex = (buffer: ArrayBuffer) => Array.from(new Uint8Array(buffer), (v) => v.toString(16).padStart(2, '0')).join('');
  const hashes = await Promise.all(files.map(async (file) => [file.path, hex(await crypto.subtle.digest('SHA-256', file.bytes))]));
  const identity = new TextEncoder().encode(JSON.stringify([canonicalEntry, hashes]));
  const id = hex(await crypto.subtle.digest('SHA-256', identity));
  return { id, name, entry: canonicalEntry, files };
}
