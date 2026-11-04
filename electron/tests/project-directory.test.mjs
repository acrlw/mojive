import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';
import { mkdtemp, mkdir, writeFile, realpath, rm, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
const { Projects } = createRequire(import.meta.url)('../main/resource-paths.cjs');

test('a large first subtree never consumes sibling directory entries and recursive search reaches unloaded branches', async () => {
  const temporary = await realpath(await mkdtemp(path.join(os.tmpdir(), 'mojive-large-project-')));
  const projects = new Projects();
  try {
    await mkdir(path.join(temporary, '3rdparty'));
    await mkdir(path.join(temporary, 'assets', 'robots'), { recursive: true });
    for (let first = 0; first < 3050; first += 50)
      await Promise.all(
        Array.from({ length: 50 }, (_, index) =>
          writeFile(path.join(temporary, '3rdparty', `file-${first + index}.txt`), ''),
        ),
      );
    const target = path.join(temporary, 'assets', 'robots', 'target.urdf');
    await writeFile(target, '<robot name="target"/>');
    await writeFile(path.join(temporary, 'README.md'), '# Project');
    await symlink(os.tmpdir(), path.join(temporary, 'outside-link'));
    const root = await projects.list(temporary);
    assert.deepEqual(
      root.map((file) => file.name),
      ['3rdparty', 'assets', 'README.md'],
    );
    assert.ok(
      root.every((file) => file.children === undefined),
      'opening a root never eagerly walks a subtree',
    );
    assert.equal((await projects.list(path.join(temporary, '3rdparty'))).length, 3050);
    const found = await projects.search(temporary, 'target');
    assert.equal(found.truncated, false);
    assert.deepEqual(
      found.files.map((file) => file.path),
      [target],
    );
    const broad = await projects.search(temporary, 'file-');
    assert.equal(broad.files.length, 1000);
    assert.equal(broad.truncated, true, 'bounded search explicitly reports incomplete results');
    await projects.register(temporary);
    await assert.rejects(projects.authorized(os.tmpdir()), /outside the opened project/);
    await assert.rejects(projects.search(temporary, ''), /search query/);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});
