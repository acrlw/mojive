const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

function within(root, filename) {
  const relative = path.relative(root, filename);
  return (
    relative === '' ||
    (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative))
  );
}
class Projects {
  constructor() {
    this.roots = new Map();
  }
  async register(root) {
    root = await fs.realpath(root);
    for (const [token, value] of this.roots) if (value === root) return token;
    const token = randomUUID();
    this.roots.set(token, root);
    return token;
  }
  async authorized(filename) {
    const resolved = await fs.realpath(filename);
    for (const [token, root] of this.roots)
      if (within(root, resolved)) return { token, root, filename: resolved };
    throw new Error('This file is outside the opened project.');
  }
  async resolve(urlText) {
    const url = new URL(urlText);
    if (url.protocol !== 'mojive-resource:' || url.hostname !== 'project')
      throw new Error('Invalid resource URL.');
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
    const root = this.roots.get(parts.shift());
    if (!root) throw new Error('This project is not open.');
    if (
      parts.some(
        (part) => part === '..' || part.includes('/') || part.includes('\\') || part.includes('\0'),
      )
    )
      throw new Error('Invalid resource path.');
    let filename;
    if (parts[0] === '@package') {
      parts.shift();
      const packageName = parts.shift();
      if (!packageName) throw new Error('Missing package name.');
      const candidates = [path.join(root, packageName, ...parts)];
      if (path.basename(root) === packageName) candidates.unshift(path.join(root, ...parts));
      // A selected ROS package can declare a name different from its directory.
      try {
        const packageXml = await fs.readFile(path.join(root, 'package.xml'), 'utf8');
        if (
          new RegExp(
            `<name>\\s*${packageName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*</name>`,
          ).test(packageXml)
        )
          candidates.unshift(path.join(root, ...parts));
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
      for (const candidate of candidates) {
        try {
          const resolved = await fs.realpath(candidate);
          if (within(root, resolved)) {
            filename = resolved;
            break;
          }
        } catch (error) {
          if (error.code !== 'ENOENT') throw error;
        }
      }
      if (!filename)
        throw new Error(
          `Package resource is missing inside the opened project: ${packageName}/${parts.join('/')}`,
        );
    } else filename = await fs.realpath(path.join(root, ...parts));
    if (!within(root, filename))
      throw new Error('The resource resolves outside the opened project.');
    return filename;
  }
  async tree(root, depth = 0, budget = { count: 0 }) {
    if (depth > 8 || budget.count >= 3000) return [];
    const entries = await fs.readdir(root, { withFileTypes: true });
    const nodes = [];
    for (const entry of entries.sort(
      (a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name),
    )) {
      if (
        entry.name.startsWith('.') ||
        ['node_modules', 'build', 'dist'].includes(entry.name) ||
        entry.isSymbolicLink()
      )
        continue;
      if (++budget.count > 3000) break;
      const filename = path.join(root, entry.name);
      if (entry.isDirectory())
        nodes.push({
          name: entry.name,
          path: filename,
          kind: 'directory',
          children: await this.tree(filename, depth + 1, budget),
        });
      else if (entry.isFile()) nodes.push({ name: entry.name, path: filename, kind: 'file' });
    }
    return nodes;
  }
  async list(directory) {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    return entries
      .filter(
        (entry) =>
          !entry.name.startsWith('.') &&
          !['node_modules', 'build', 'dist'].includes(entry.name) &&
          (entry.isDirectory() || entry.isFile()),
      )
      .sort(
        (a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name),
      )
      .map((entry) => ({
        name: entry.name,
        path: path.join(directory, entry.name),
        kind: entry.isDirectory() ? 'directory' : 'file',
      }));
  }
  async search(root, query, extensions = null) {
    if (typeof query !== 'string' || (!query.trim() && !extensions) || query.length > 256)
      throw new Error('Enter a project search query of at most 256 characters.');
    const needle = query.trim().toLocaleLowerCase(),
      files = [],
      pending = [root];
    let visited = 0;
    for (let index = 0; index < pending.length; index++) {
      for (const file of await this.list(pending[index])) {
        if (++visited > 100000) return { files, truncated: true };
        if ((!extensions || (file.kind === 'file' && extensions.includes(path.extname(file.name).toLowerCase()))) &&
          file.name.toLocaleLowerCase().includes(needle)) {
          if (files.length === 1000) return { files, truncated: true };
          files.push(file);
        }
        if (file.kind === 'directory') pending.push(file.path);
      }
    }
    return { files, truncated: false };
  }
}
module.exports = { Projects, within };
