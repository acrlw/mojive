const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
function filePath(value) {
  if (
    typeof value !== 'string' ||
    !value.trim() ||
    value.includes('\0') ||
    value.length > 4096 ||
    !path.isAbsolute(value)
  )
    throw new Error('Choose an absolute local path.');
  return path.normalize(value);
}
class FileAccess {
  constructor(shortcuts) {
    this.shortcuts = shortcuts;
    this.lastDirectory = shortcuts[0].path;
  }
  async browse(directory, includeHidden = false) {
    if (typeof includeHidden !== 'boolean') throw new Error('Invalid hidden-file option.');
    const resolved = await fs.realpath(filePath(directory ?? this.lastDirectory));
    if (!(await fs.stat(resolved)).isDirectory()) throw new Error('Choose a folder to browse.');
    const all = await fs.readdir(resolved, { withFileTypes: true });
    const candidates = all
      .filter((entry) => includeHidden || !entry.name.startsWith('.'))
      .sort(
        (a, b) =>
          Number(b.isDirectory()) - Number(a.isDirectory()) ||
          a.name.localeCompare(b.name, undefined, { numeric: true }),
      );
    const entries = [];
    for (const entry of candidates.slice(0, 5000)) {
      let directory = entry.isDirectory();
      if (entry.isSymbolicLink()) {
        try {
          directory = (await fs.stat(path.join(resolved, entry.name))).isDirectory();
        } catch {
          continue;
        }
      } else if (!entry.isDirectory() && !entry.isFile()) continue;
      entries.push({ name: entry.name, path: path.join(resolved, entry.name), directory });
    }
    entries.sort(
      (a, b) =>
        Number(b.directory) - Number(a.directory) ||
        a.name.localeCompare(b.name, undefined, { numeric: true }),
    );
    const shortcuts = [];
    for (const shortcut of this.shortcuts) {
      try {
        if ((await fs.stat(shortcut.path)).isDirectory()) shortcuts.push(shortcut);
      } catch {}
    }
    this.lastDirectory = resolved;
    return {
      path: resolved,
      parent: path.dirname(resolved) === resolved ? null : path.dirname(resolved),
      entries,
      shortcuts,
      truncated: candidates.length > 5000,
    };
  }
  async readScene(filename) {
    filename = await fs.realpath(filePath(filename));
    if (!filename.toLowerCase().endsWith('.json')) throw new Error('Choose a .mojive.json scene.');
    const stat = await fs.stat(filename);
    if (!stat.isFile() || stat.size > 100 * 1024 * 1024)
      throw new Error('Scene must be a file no larger than 100 MiB.');
    return { content: await fs.readFile(filename, 'utf8'), path: filename };
  }
  async saveScene(filename, content, overwrite = false) {
    filename = filePath(filename);
    if (!filename.toLowerCase().endsWith('.json'))
      throw new Error('Save the scene with a .mojive.json extension.');
    if (typeof content !== 'string' || Buffer.byteLength(content, 'utf8') > 100 * 1024 * 1024)
      throw new Error('Invalid project document.');
    if (typeof overwrite !== 'boolean') throw new Error('Invalid overwrite confirmation.');
    JSON.parse(content);
    return this.writeFile(filename, content, overwrite);
  }
  async saveBinary(filename, content, overwrite = false) {
    filename = filePath(filename);
    if (!(content instanceof Uint8Array) || content.byteLength > 512 * 1024 * 1024)
      throw new Error('Invalid media file.');
    const bytes = Buffer.from(content);
    const ext = path.extname(filename).toLowerCase();
    const png = bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
    const webm = bytes.subarray(0, 4).equals(Buffer.from([26, 69, 223, 163]));
    if (!((ext === '.png' && png) || (ext === '.webm' && webm)))
      throw new Error('Choose a matching PNG or WebM file.');
    return this.writeFile(filename, bytes, overwrite);
  }
  async writeFile(filename, content, overwrite) {
    if (typeof overwrite !== 'boolean') throw new Error('Invalid overwrite confirmation.');
    const directory = await fs.realpath(path.dirname(filename));
    filename = path.join(directory, path.basename(filename));
    let existing;
    try {
      existing = await fs.lstat(filename);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (existing && !existing.isFile())
      throw new Error('Choose a regular file, not a folder or symbolic link.');
    if (existing && !overwrite)
      throw new Error('[FILE_EXISTS] A file already exists at this path.');
    const temporary = path.join(directory, `.mojive-${randomUUID()}.tmp`);
    let handle;
    try {
      handle = await fs.open(temporary, 'wx', existing?.mode ?? 0o600);
      await handle.writeFile(content, 'utf8');
      await handle.sync();
      await handle.close();
      handle = null;
      if (overwrite) await fs.rename(temporary, filename);
      else {
        // A hard-link publication fails atomically if another writer creates the destination.
        try {
          await fs.link(temporary, filename);
        } catch (error) {
          if (error.code === 'EEXIST')
            throw new Error('[FILE_EXISTS] A file already exists at this path.');
          throw error;
        }
        await fs.unlink(temporary);
      }
      return filename;
    } finally {
      await handle?.close();
      await fs.rm(temporary, { force: true });
    }
  }
}
module.exports = { FileAccess, filePath };
