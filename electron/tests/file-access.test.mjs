import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';
import {
  mkdtemp,
  mkdir,
  writeFile,
  readFile,
  readdir,
  symlink,
  rm,
  realpath,
} from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
const { FileAccess } = createRequire(import.meta.url)('../main/file-access.cjs');
test('custom file browser lists folders, filters hidden files and validates paths without exposing arbitrary reads', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'mojive-files-'));
  try {
    await mkdir(path.join(root, 'folder'));
    await writeFile(path.join(root, 'robot.urdf'), '<robot/>');
    await writeFile(path.join(root, '.hidden'), 'hidden');
    const files = new FileAccess([{ label: 'Home', path: root }]);
    const listing = await files.browse();
    assert.equal(listing.path, await realpath(root));
    assert.equal(listing.entries[0].name, 'folder');
    assert.equal(listing.entries.length, 2);
    assert.equal((await files.browse(root, true)).entries.length, 3);
    await assert.rejects(() => files.browse('relative'), /absolute/);
    await assert.rejects(() => files.browse(path.join(root, 'robot.urdf')), /folder/);
    await assert.rejects(() => files.readScene(path.join(root, 'robot.urdf')), /mojive.json/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
test('scene saves require explicit overwrite, preserve original on invalid content and publish atomically', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'mojive-save-'));
  try {
    const files = new FileAccess([{ label: 'Home', path: root }]),
      filename = path.join(root, 'scene.mojive.json');
    assert.equal(await files.saveScene(filename, '{"value":1}'), await realpath(filename));
    await assert.rejects(() => files.saveScene(filename, '{"value":2}'), /FILE_EXISTS/);
    assert.equal(await readFile(filename, 'utf8'), '{"value":1}');
    await assert.rejects(() => files.saveScene(filename, 'bad', true));
    assert.equal(await readFile(filename, 'utf8'), '{"value":1}');
    await files.saveScene(filename, '{"value":2}', true);
    assert.equal((await files.readScene(filename)).content, '{"value":2}');
    const link = path.join(root, 'linked.json');
    await symlink(filename, link);
    await assert.rejects(() => files.saveScene(link, '{}', true), /symbolic link/);
    assert.deepEqual((await readdir(root)).sort(), ['linked.json', 'scene.mojive.json']);
    const race = path.join(root, 'race.json');
    const attempts = await Promise.allSettled([
      files.saveScene(race, '{"value":3}'),
      files.saveScene(race, '{"value":4}'),
    ]);
    assert.equal(attempts.filter((x) => x.status === 'fulfilled').length, 1);
    assert.equal(attempts.filter((x) => x.status === 'rejected').length, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('PNG/WebM export validates type and atomically protects existing files', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'mojive-media-'));
  try {
    const files = new FileAccess([{ label: 'Home', path: root }]);
    const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0]);
    const filename = path.join(root, 'capture.png');
    await files.saveBinary(filename, png);
    assert.deepEqual(new Uint8Array(await readFile(filename)), png);
    await assert.rejects(() => files.saveBinary(filename, png), /FILE_EXISTS/);
    await assert.rejects(() => files.saveBinary(filename, new Uint8Array([0]), true), /matching/);
    await assert.rejects(() => files.saveBinary(path.join(root, 'capture.json'), png), /matching/);
    await files.saveBinary(path.join(root, 'take.webm'), new Uint8Array([26, 69, 223, 163, 0]));
    assert.deepEqual((await readdir(root)).sort(), ['capture.png', 'take.webm']);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
